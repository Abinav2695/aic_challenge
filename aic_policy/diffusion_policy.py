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

    def forward(self, x:torch.Tensor)-> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)
    

    
class SpatialSoftmax(nn.Module):
    """Convert 2D feature maps into (x,y) keypoint coordinates."""

    def __init__(self, num_keypoints: int, feature_height: int, feature_width: int,
                 in_channels: int):
        super().__init__()
        self.compress = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)

        # Create fixed coordinate grids: where is each pixel?
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, feature_width),
            torch.linspace(-1, 1, feature_height),
            indexing = "xy"
        )

        self.register_buffer("pos_x", pos_x.reshape(1, 1, -1))
        self.register_buffer("pos_y", pos_y.reshape(1, 1, -1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, num_keypoints * 2)"""
        features = self.compress(features)             # (B, K, H, W)
        B, K, H, W = features.shape
        features = features.reshape(B, K, -1)          # (B, K, H*W)
        attention = F.softmax(features, dim=-1)        # (B, K, H*W)
        x = (attention * self.pos_x).sum(dim=-1)       # (B, K)
        y = (attention * self.pos_y).sum(dim=-1)       # (B, K)
        return torch.cat([x, y], dim=-1)               # (B, K*2)
    

class RGBEncoder(nn.Module):
    """Single camera encoder: ResNet18 backbone + SpatialSoftmax."""

    def __init__(self, num_keypoints: int = 32, pretrained: bool = True,
                use_group_norm: bool = True):
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
            num_keypoints= num_keypoints,
            feature_height=3,
            feature_width=3,
            in_channels=512
        )
        self.output_dim = num_keypoints * 2

    def _replace_bn(self, module: nn.Module):
        """Swap all BatchNorm2d layers with GroupNorm."""
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                setattr(module, name, nn.GroupNorm(
                    min(8, child.num_features), child.num_features
                ))
            else:
                self._replace_bn(child)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """(B,3,96,96) -> (B, num_keypoints * 2)"""
        features = self.backbone(x)           # (B, 512, 3, 3)
        return self.spatial_softmax(features)    # (B, 64)
    

class MultiCameraEncoder(nn.Module):
    """Encode all camera views and concatenate features."""

    def __init__(self, camera_names: list[str], num_keypoints: int = 32,
                 pretrained: bool = True, use_group_norm: bool = True,
                  use_separate_encoders: bool = True):
        super().__init__()
        self.camera_names = camera_names

        if use_separate_encoders:
            # Each camera gets its own ResNet - learns camera-specific features
            self.encoders = nn.ModuleDict({
                name: RGBEncoder(num_keypoints, pretrained, use_group_norm)
                for name in camera_names
            })
            # nn.ModuleDict not a regular dict. 
            # If you store nn.Modules in a regular Python dict, PyTorch can't find them
            # they won't move to GPU, won't appear in .parameters(), 
            # won't be saved. nn.ModuleDict registers them properly. 
            # Same idea as register_buffer but for trainable sub-modules.
        else:
            # All cameras share one ResNet — fewer parameters, less overfitting
            shared = RGBEncoder(num_keypoints, pretrained, use_group_norm)
            self.encoders = nn.ModuleDict({
                name: shared for name in camera_names
            })

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

        features_per_step =[]
        for t in range(n_obs):
            cam_features =[]
            for name in self.camera_names:
                img = images[name][:, t]               # (B, 3, H, W)
                # images[name][:, t, :, :, :] this happened above
                feat = self.encoders[name](img)        # (B, 64)
                cam_features.append(feat)
            features_per_step.append(torch.cat(cam_features, dim=-1))    # (B, 192)
        
        return torch.stack(features_per_step, dim=1)   # (B, n_obs, 192) 


        

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
    

    

    








        

    




