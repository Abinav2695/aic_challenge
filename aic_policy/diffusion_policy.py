import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from base_policy import BasePolicy


class SinusoidalPosEmb(nn.Module):
    """Convert a single integer timestep into a rich vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class SpatialSoftmax(nn.Module):
    """Convert 2D feature maps into (x,y) keypoint coordinates."""

    def __init__(
        self, num_keypoints: int, feature_height: int, feature_width: int, in_channels: int
    ):
        super().__init__()
        self.compress = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)

        # Create fixed coordinate grids: where is each pixel?
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, feature_width),
            torch.linspace(-1, 1, feature_height),
            indexing="xy",
        )

        self.register_buffer("pos_x", pos_x.reshape(1, 1, -1))
        self.register_buffer("pos_y", pos_y.reshape(1, 1, -1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, num_keypoints * 2)"""
        features = self.compress(features)  # (B, K, H, W)
        B, K, H, W = features.shape
        features = features.reshape(B, K, -1)  # (B, K, H*W)
        attention = F.softmax(features, dim=-1)  # (B, K, H*W)
        x = (attention * self.pos_x).sum(dim=-1)  # (B, K)
        y = (attention * self.pos_y).sum(dim=-1)  # (B, K)
        return torch.cat([x, y], dim=-1)  # (B, K*2)


class RGBEncoder(nn.Module):
    """Single camera encoder: ResNet18 backbone + SpatialSoftmax."""

    def __init__(
        self, num_keypoints: int = 32, pretrained: bool = True, use_group_norm: bool = True
    ):
        super().__init__()
        import torchvision.models as models

        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Why GroupNorm instead of BatchNorm?
        # BatchNorm computes statistics across the batch.
        # With small batches (which we might have),
        # those statistics are noisy and training becomes unstable.
        # GroupNorm computes statistics within each sample across channel groups

        # Replace BatchNorm with GroupNorm for small batch stability
        if use_group_norm:
            self._replace_bn(backbone)

        # Remove the classfication head (avgpool + fc), keep only feature extractor
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])

        # For 96x96 input -> ResNet outputs (512, 3, 3) feature maps
        self.spatial_softmax = SpatialSoftmax(
            num_keypoints=num_keypoints, feature_height=3, feature_width=3, in_channels=512
        )
        self.output_dim = num_keypoints * 2

    def _replace_bn(self, module: nn.Module):
        """Swap all BatchNorm2d layers with GroupNorm."""
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(module, name, nn.GroupNorm(min(8, child.num_features), child.num_features))
            else:
                self._replace_bn(child)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,96,96) -> (B, num_keypoints * 2)"""
        features = self.backbone(x)  # (B, 512, 3, 3)
        return self.spatial_softmax(features)  # (B, 64)


class MultiCameraEncoder(nn.Module):
    """Encode all camera views and concatenate features."""

    def __init__(
        self,
        camera_names: list[str],
        num_keypoints: int = 32,
        pretrained: bool = True,
        use_group_norm: bool = True,
        use_separate_encoders: bool = True,
    ):
        super().__init__()
        self.camera_names = camera_names

        # Sanitize names for ModuleDict (dots not allowed as keys)
        self._key_map = {name: name.replace(".", "_") for name in camera_names}

        if use_separate_encoders:
            # Each camera gets its own ResNet - learns camera-specific features
            self.encoders = nn.ModuleDict(
                {
                    self._key_map[name]: RGBEncoder(num_keypoints, pretrained, use_group_norm)
                    for name in camera_names
                }
            )
            # nn.ModuleDict not a regular dict.
            # If you store nn.Modules in a regular Python dict, PyTorch can't find them
            # they won't move to GPU, won't appear in .parameters(),
            # won't be saved. nn.ModuleDict registers them properly.
            # Same idea as register_buffer but for trainable sub-modules.
        else:
            # All cameras share one ResNet — fewer parameters, less overfitting
            shared = RGBEncoder(num_keypoints, pretrained, use_group_norm)
            self.encoders = nn.ModuleDict({self._key_map[name]: shared for name in camera_names})

        self.output_dim = num_keypoints * 2 * len(camera_names)

    def forward(self, images: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            images: {camera_name: (B, n_obs, 3, H, W)}
        Returns:
            (B, n_obs, total_feature_dim)
        """
        first_key = self.camera_names[0]
        B, n_obs = images[first_key].shape[:2]

        features_per_step = []
        for t in range(n_obs):
            cam_features = []
            for name in self.camera_names:
                img = images[name][:, t]  # (B, 3, H, W)
                # images[name][:, t, :, :, :] this happened above
                feat = self.encoders[self._key_map[name]](img)  # (B, 64)
                cam_features.append(feat)
            features_per_step.append(torch.cat(cam_features, dim=-1))  # (B, 192)

        return torch.stack(features_per_step, dim=1)  # (B, n_obs, 192)


class StateEncoder(nn.Module):
    """Encode the proprioceptive state (joints, force, etc)."""

    def __init__(self, state_dim: int, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Mish(),
            # nn.Mish() — an activation function.
            # Without it, stacking two Linear layers is pointless (two matrix multiplications = one matrix multiplication).
            # Mish adds non-linearity so the network can learn curved relationships.
            # It's smoother than ReLU, works well for diffusion policies.
            # Mish: f(x) = x * tanh(ln(1 + e^x))
            nn.Linear(128, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """(B, n_obs, state_dim) -> (B, n_obs, output_dim)"""
        return self.net(state)


class DiTBlock(nn.Module):
    """Single Diffusion Transformer block with adaptive LayerNorm."""

    def __init__(
        self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        # elementwise_affine=false let it calculate mean to 0 and cov to 1,
        # restrict the gamma(scale) and beta(shift) to calculated as learn params
        # we will do that on the fly
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        # dropout is 0.1 here which turns off (gives 0)10 percent of neurons so it work on
        # unseen data in realworld since all learn in regularization manner
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.Mish(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

        # adaLN: conditioning process scale & shift for both norms
        # 6 values: scales, shifts, gate1, scale2, shift2, gate2
        self.adaLN_modulation = nn.Linear(embed_dim, embed_dim * 6)
        # The actual reason: We need separate scales (scale1, scale2)
        # and shifts (shift1, shift2)
        # because the Attention layer and the MLP layer do completely different jobs.
        # Attention relates action token #1 with similar action token #5
        # MLP here does computation: to process what token just learn from attention layer

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, embed_dim) — noisy action tokens
            cond: (B, embed_dim) — timestep + observation conditioning
        """
        # cond is flatten vector with timestep features (t) + observation features (0)
        # Get 6 modulation parameters from conditioning
        mod = self.adaLN_modulation(cond).unsqueeze(1)  # (B, 1, 6*embed_dim)
        scale1, shift1, gate1, scale2, shift2, gate2 = mod.chunk(6, dim=-1)
        # The gate is simply a dynamic multiplier—a "volume knob" or a valve—that controls exactly
        # how much of that new information (h) is allowed to mix back into the main stream (x).

        # Modulated self-attention
        h = self.norm1(x) * (1 + scale1) + shift1
        h, _ = self.attn(h, h, h)
        # (h,h,h) is Query, Key, and Value. _ part is the attn_output_weights
        x = x + gate1 * h

        # Modulated feedforward or MLP
        h = self.norm2(x) * (1 + scale2) + shift2
        h = self.mlp(h)
        x = x + gate2 * h

        return x


class DiTBackbone(nn.Module):
    """Full Diffusion Transformer: projects actions to tokens, stacks DiT blocks."""

    def __init__(
        self,
        action_dim: int,
        global_cond_dim: int,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project each action timestep into a token
        self.action_proj = nn.Linear(action_dim, embed_dim)

        # Learnable position embedding for each slot in the horizon
        self.pos_embed = nn.Parameter(torch.zeros(1, 1024, embed_dim))
        # The pos_embed tells transformer the sequence of action we fill in
        # nn.Parameter are trainable quantities

        # Project timestep + obs conditioning to embed_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(global_cond_dim, embed_dim),
            nn.Mish(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Stack of DiT blocks
        self.blocks = nn.ModuleList(
            [DiTBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )

        # Final norm + project back to action dim
        self.final_norm = nn.LayerNorm(embed_dim)
        # Layer norm will calculate gamma(scale) and beta(shift) now
        # Only on embed_dim ideally it can be (B, seq_len, embed_dim) so on embed_dim only single token
        self.final_proj = nn.Linear(embed_dim, action_dim)

        # Initialize pos_embed
        nn.init.normal_(self.pos_embed, std=0.02)
        # Cannot init to zero we need it to add actual feature.

    def forward(
        self, x: torch.Tensor, timestep_emb: torch.Tensor, global_cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, horizon, action_dim) — noisy actions
            timestep_emb: (B, timestep_embed_dim) — from SinusoidalPosEmb
            global_cond: (B, obs_feature_dim) — encoded observations
        Returns:
            (B, horizon, action_dim) — predicted noise
        """

        B, seq_len, _ = x.shape

        # Project actions to tokens and add position info
        x = self.action_proj(x)  # (B, horizon, embed_dim)
        x = x + self.pos_embed[:, :seq_len, :]  # add positional info
        # we have (1, 1024, embed_dim) only get for the size of seq len

        # Build conditioning vector
        cond = self.cond_proj(torch.cat([timestep_emb, global_cond], dim=-1))  # (B, embed_dim)

        # Pass through all DiT blocks
        for block in self.blocks:
            x = block(x, cond)

        # Project back to action space
        x = self.final_norm(x)  # (B, horizon, embed_dim)
        x = self.final_proj(x)  # (B, horizon, action_dim)

        return x


class DiffusionPolicy(BasePolicy):
    """Complete diffusion policy: encoders + backbone + noise scheduler."""

    def __init__(self, config):
        super().__init__(config)
        data = config.data
        diff = config.diffusion

        # Timestep embedding
        self.timestep_embed_dim = 128
        self.timestep_emb = nn.Sequential(
            SinusoidalPosEmb(self.timestep_embed_dim),
            nn.Linear(self.timestep_embed_dim, self.timestep_embed_dim),
            nn.Mish(),
            nn.Linear(self.timestep_embed_dim, self.timestep_embed_dim),
        )

        # Image encoder
        camera_names = [
            "observation.images.left_camera",
            "observation.images.center_camera",
            "observation.images.right_camera",
        ][: data.n_cameras]

        self.image_encoder = MultiCameraEncoder(
            camera_names=camera_names,
            num_keypoints=diff.spatial_softmax_num_keypoints,
            pretrained=diff.pretrained_backbone,
            use_group_norm=diff.use_group_norm,
            use_separate_encoders=diff.use_separate_rgb_encoder_per_camera,
        )

        # State encoder
        self.state_encoder = StateEncoder(data.obs_state_dim, output_dim=64)

        # Compute global conditioning size
        per_step_dim = self.image_encoder.output_dim + self.state_encoder.output_dim
        global_cond_dim = self.timestep_embed_dim + per_step_dim * data.n_obs_steps

        # ── Denoising backbone (the switchable part) ──
        if diff.backbone == "dit":
            self.backbone = DiTBackbone(
                action_dim=data.action_dim,
                global_cond_dim=global_cond_dim,
                embed_dim=diff.dit_embed_dim,
                depth=diff.dit_depth,
                num_heads=diff.dit_num_heads,
                mlp_ratio=diff.dit_mlp_ratio,
                dropout=diff.dit_dropout,
            )
        else:
            raise NotImplementedError(f"Backbone '{diff.backbone}' not yet built")

        # Noise scheduler from diffusers library
        from diffusers import DDIMScheduler, DDPMScheduler
        # Two schedulers — DDPMScheduler for training (adds noise),
        # DDIMScheduler for inference (removes noise in fewer steps).
        # Same noise schedule, different sampling strategy.

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=diff.num_train_timesteps,
            beta_schedule=diff.noise_schedule,
            prediction_type=diff.prediction_type,
        )

        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=diff.num_train_timesteps,
            beta_schedule=diff.noise_schedule,
            prediction_type=diff.prediction_type,
        )

        self._data = data
        self._diff = diff

    def _encode_observations(self, batch):
        """Encode images + state into flat conditioning vector."""
        # Encode images
        images = {k: batch[k] for k in self.image_encoder.camera_names if k in batch}
        img_feat = self.image_encoder(images)  # (B, n_obs, 192)

        # Encode state
        state_feat = self.state_encoder(batch["observation.state"])  # (B, n_obs, 64)

        # Concatenate and flatten across obs steps
        combined = torch.cat([img_feat, state_feat], dim=-1)  # (B, n_obs, 256)

        return combined.flatten(start_dim=1)  # (B, 512)

    def forward(self, batch):
        """Training: add noise to actions, predict it, compute loss"""
        actions = batch["action"]  # (B, horizon, action_dim)
        B = actions.shape[0]

        # Step 1: Encode Observations
        obs_feat = self._encode_observations(batch)  # (B, 512)

        # Step 2: Sample random timesteps
        timesteps = torch.randint(0, self._diff.num_train_timesteps, (B,), device=actions.device)
        # randomly generate timesteps to populate error for that timestep stage
        # across B batches so we remain consistant

        # Step 3: Embed timesteps
        t_emb = self.timestep_emb(timesteps)  # (B, 128)

        # Step 4: Sample noise and add to actions
        noise = torch.randn_like(actions)
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

        # Step 5: Predict noise
        noise_pred = self.backbone(noisy_actions, t_emb, obs_feat)

        # Step 7: MSE loss
        loss = F.mse_loss(noise_pred, noise)

        return {"loss": loss}

    @torch.no_grad()
    def predict_action(self, observation):
        """Inference: generate actions by denoising from pure noise."""
        self.eval()
        # self.eval() — switches dropout off and BatchNorm/LayerNorm to eval mode.
        # Important: during training, dropout randomly kills neurons.
        # During inference, we want the full network.
        obs_feat = self._encode_observations(observation)  # (B, 512)
        B = obs_feat.shape[0]

        # Start from pure random noise
        noisy_actions = torch.randn(
            B, self._data.horizon, self._data.action_dim, device=obs_feat.device
        )

        # Set up DDIM scheduler for fast inference
        self.inference_scheduler.set_timesteps(self._diff.num_inference_steps)

        # Iteratively denoise
        for t in self.inference_scheduler.timesteps:
            t_batch = t.expand(B).to(obs_feat.device)
            t_emb = self.timestep_emb(t_batch)

            # Predict noise at this timestep
            noise_pred = self.backbone(noisy_actions, t_emb, obs_feat)

            # Remove predicted noise (one DDIM step)
            noisy_actions = self.inference_scheduler.step(noise_pred, t, noisy_actions).prev_sample

        # Extract only the actions we'll execute
        start = self._data.n_obs_steps - 1
        end = start + self._data.n_action_steps
        return noisy_actions[:, start:end]
