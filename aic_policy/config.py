from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    """Dataset and preprocessing settings."""
    dataset_repo_id: str = ""            # HuggingFace repo or local path
    local_dir: Optional[str] = None       # local dataset path override

    # Dimensions (filled from your dataset)
    obs_state_dim: int = 32               # 7 pose + 6 vel + 6 error + 7 joints + 3 force + 3 torque
    action_dim: int = 13                  # 7 pose + 6 stiffness diagonal
    n_cameras: int = 3                    # left, center, right

    # Action chunking
    n_obs_steps: int = 2                  # how many past frames to observe
    horizon: int = 16                     # total prediction window
    n_action_steps: int = 8               # how many predicted actions to actually execute


@dataclass
class TrainConfig:
    """Training loop settings."""
    total_steps: int = 200_000
    seed: int = 42
    device: str = "cuda"
    use_amp: bool = True                 # mixed precision (faster, less memory)
    #use_amp — Automatic Mixed Precision. 
    #Runs some operations in float16 instead of float32. 
    #Roughly 2x faster training and uses less GPU memory. 
    #Almost always worth enabling.


    # Optimizer
    lr: float = 1e-4
    weight_decay: float =1e-6
    max_grad_norm: float = 10.0          # clip gradients to prevent explosions
    #max_grad_norm — gradient clipping. 
    #If the loss spikes, gradients can become huge and destroy your weights. 
    #This caps them at 10.0. Safety net.



    # Learning rate schedule
    warmup_steps: int = 500              # LR ramps up from 0 to lr over this many steps
    lr_backbone_multiplier: float = 0.1  # pretrained ResNet gets 10x lower LR


    # EMA
    use_ema: bool = True
    ema_decay: float = 0.995 
    #use_ema / ema_decay — Exponential Moving Average. 
    #We keep a smoothed copy of the model weights: ema_weights = 0.995 * ema_weights + 0.005 * current_weights. 
    #At inference we use the EMA weights, which are more stable than the raw trained weights. 
    #This is standard practice for diffusion models.

    # Logging and checkpointing
    log_freq: int = 50
    save_freq: int = 10_000
    output_dir: str = "outputs/train"
    wandb_project: str = "aic-cable-insertion"
    wandb_enabled: bool = True


    # Data loading
    batch_size: int = 64
    num_workers: int = 4

@dataclass
class DiffusionConfig:
    """Setting specific to diffusion-based policies."""
    # ----- Common diffusion settings -----
    num_train_timesteps: int = 100
    num_inference_steps: int = 10
    noise_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "epsilon"

    # ------ Architecture selector -------
    backbone: str = "dit"             # "unet" or "dit"

    # ------- UNet settings (used when backbone="unet")----
    unet_down_dims: tuple = (256, 512, 1024)
    unet_kernel_size: int = 5
    unet_n_groups: int = 8 
    
    # -------- DiT settings (used when backbone="dit") -----
    dit_embed_dim: int = 384            # transformer hidden dimension
    # dit_embed_dim=384 — the width of the transformer. 
    # Every token (each timestep in the action sequence) becomes a 384-dim vector. 
    # Bigger = more capacity but slower.

    dit_num_heads: int = 6              # attention heads
    # dit_num_heads=6 — multi-head attention splits the 384 dims into 6 heads of 64 dims each. 
    # Each head learns different attention patterns 
    # (one might focus on nearby timesteps, another on the overall trajectory shape).

    dit_depth: int = 8                  # number of transformer blocks
    # dit_depth=8 — how many transformer blocks stacked. 
    # Each block does self-attention + feedforward. 
    # More depth = the model can reason about more complex action dependencies.

    dit_mlp_ratio: float = 4.0          # FFN hidden dim = embed_dim * mlp_ratio
    # dit_mlp_ratio=4.0 — inside each block, 
    # the feedforward network expands to 384 * 4 = 1536 dims, 
    # then projects back to 384. This is where most of the "thinking" happens.
    dit_dropout: float = 0.1

    # ------- Image encoder (shared by all backbones)--------
    spatial_softmax_num_keypoints: int = 32
    pretrained_backbone: bool = True
    use_group_norm: bool = True
    use_separate_rgb_encoder_per_camera: bool = True



@dataclass
class ExperimentConfig:
    """Top-level config combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    # default_factory - tells python "call DataConfig() to create a fresh instance each time"

    def validate(self):
        """
        Check that settings are consistent.
        """
        assert self.data.n_action_steps <= self.data.horizon - self.data.n_obs_steps + 1, (
            f"n_action_steps ({self.data.n_action_steps}) must be <= "
            f"horizon - n_obs_steps + 1 ({self.data.horizon - self.data.n_obs_steps + 1})"
        )
        assert self.diffusion.backbone in ("unet", "dit"), (
            f"Unknown backbone : {self.diffusion.backbone}"
        )
        

