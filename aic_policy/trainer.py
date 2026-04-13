from pathlib import Path

import torch
from config import ExperimentConfig
from diffusion_policy import DiffusionPolicy

# LeRobot — we use these for data loading and normalization
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """Handles the full training lifecycle: data, model, optimization, logging."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.train.device)

        # Seed for reproducibility
        torch.manual_seed(config.train.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.train.seed)

        # Load dataset metadata (stats, features, fps)
        self.metadata = LeRobotDatasetMetadata(config.data.dataset_repo_id)
        print(f"Dataset: {config.data.dataset_repo_id}")
        print(f"  FPS: {self.metadata.fps}")
        print(f"  Episodes: {self.metadata.total_episodes}")
        print(f"  Features: {list(self.metadata.features.keys())}")

        # Build delta_timestamps for action chunking
        fps = self.metadata.fps
        delta_timestamps = {
            "observation.state": [i / fps for i in range(-(config.data.n_obs_steps - 1), 1)],
            "action": [
                i / fps
                for i in range(
                    -(config.data.n_obs_steps - 1),
                    config.data.horizon - config.data.n_obs_steps + 1,
                )
            ],
        }

        # Add image timestamps (same as state)
        for key in self.metadata.features:
            if "image" in key:
                delta_timestamps[key] = delta_timestamps["observation.state"]

        print(f"  Obs timestamps: {delta_timestamps['observation.state']}")
        print(
            f"  Action timestamps: {delta_timestamps['action'][:3]}...{delta_timestamps['action'][-1:]}"
        )

        # Create dataset and dataloader
        self.dataset = LeRobotDataset(
            config.data.dataset_repo_id, delta_timestamps=delta_timestamps
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=config.train.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        # drop_last=True — if the last batch has fewer samples than batch_size, discard it.
        # Prevents shape mismatches.

        # pin_memory=True — pre-loads batch data into GPU-ready memory.
        # Speeds up CPU→GPU transfer.

        # Create Policy
        self.policy = DiffusionPolicy(config).to(self.device)
        total_params = sum(p.numel() for p in self.policy.parameters()) / 1e6
        print(f"  Policy params: {total_params:.1f}M")

        # ── Optimizer with separate backbone LR ──
        param_groups = self.policy.get_optimizer_groups(
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
            lr_backbone_multiplier=config.train.lr_backbone_multiplier,
        )

        self.optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.95, 0.999),
            eps=1e-8,
        )

        # AdamW — the optimizer.
        # Adam with proper weight decay. The (0.95, 0.999) betas control momentum — 0.95 is slightly lower than default 0.9,
        # which means less smoothing of gradients, faster response to changes.
        # Standard for diffusion training.

        # ── Learning rate scheduler with warmup ──
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / max(1, config.train.warmup_steps)),
        )
        # LambdaLR with warmup — the lambda function returns a multiplier for the learning rate.
        # At step 0 → multiplier = 0/500 = 0 (LR is 0).
        # At step 250 → 250/500 = 0.5 (half LR).
        # At step 500+ → 1.0 (full LR).
        # This ramp prevents early training instability.

        # ── EMA (Exponential Moving Average) ──
        self.ema_policy = None
        if config.train.use_ema:
            import copy

            self.ema_policy = copy.deepcopy(self.policy)
            self.ema_policy.eval()
            self.ema_policy.requires_grad_(False)

        # ── Mixed precision scaler ──
        self.scaler = torch.amp.GradScaler("cuda") if config.train.use_amp else None

        # GradScaler — part of AMP (mixed precision).
        # Some operations run in float16,
        # but loss scaling prevents gradients from underflowing to zero in float16.
        # The scaler automatically handles this.

        # ── Output directory ──
        self.output_dir = Path(config.train.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0

        # ── Wandb logging ──
        self.wandb_enabled = config.train.wandb_enabled
        if self.wandb_enabled:
            import wandb

            wandb.init(
                project=config.train.wandb_project,
                config={
                    "policy": "diffusion_dit",
                    "backbone": config.diffusion.backbone,
                    "obs_state_dim": config.data.obs_state_dim,
                    "action_dim": config.data.action_dim,
                    "horizon": config.data.horizon,
                    "n_obs_steps": config.data.n_obs_steps,
                    "n_action_steps": config.data.n_action_steps,
                    "lr": config.train.lr,
                    "batch_size": config.train.batch_size,
                    "total_steps": config.train.total_steps,
                    "ema_decay": config.train.ema_decay,
                    "dit_depth": config.diffusion.dit_depth,
                    "dit_embed_dim": config.diffusion.dit_embed_dim,
                    "total_params_M": total_params,
                },
            )

    def _update_ema(self):
        """Blend training weights into EMA weights."""
        decay = self.config.train.ema_decay
        for ema_param, param in zip(self.ema_policy.parameters(), self.policy.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

    def train_step(self, batch: dict) -> dict:
        """Single training step: forward, backward, optimize."""
        # Move batch to GPU
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

        self.policy.train()

        if self.scaler is not None:
            # Mixed precision forward
            with torch.amp.autocast("cuda"):
                output = self.policy.forward(batch)
                loss = output["loss"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.config.train.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            """
            1. scaler.scale(loss)    → multiply loss by a big number (prevents float16 underflow)
            2. .backward()           → compute gradients (still scaled)
            3. scaler.unscale_()     → undo the scaling on gradients (back to real values)
            4. clip_grad_norm_()     → clip gradients (safety net)
            5. scaler.step()         → optimizer.step() only if gradients are valid
            6. scaler.update()       → adjust the scale factor for next iteration
            """

        else:
            # Regular precision forward
            output = self.policy.forward(batch)
            loss = output["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.config.train.max_grad_norm
            )
            self.optimizer.step()

        self.optimizer.zero_grad()
        self.lr_scheduler.step()

        # Update EMA
        if self.ema_policy is not None:
            self._update_ema()

        self.global_step += 1
        return {"loss": loss.item(), "lr": self.optimizer.param_groups[0]["lr"]}

    def save_checkpoint(self, tag: str = "latest"):
        """Save model, optimizer, and EMA weights."""
        checkpoint_dir = self.output_dir / f"checkpoint_{tag}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "step": self.global_step,
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "ema_state_dict": self.ema_policy.state_dict() if self.ema_policy else None,
                "config": self.config,
            },
            checkpoint_dir / "checkpoint.pt",
        )

        print(f"  Saved checkpoint at step {self.global_step} → {checkpoint_dir}")

    def train(self):
        """Main training loop."""
        cfg = self.config.train
        print(f"\nStarting training for {cfg.total_steps} steps...")
        print(f"  Batch size: {cfg.batch_size}")
        print(f"  Device: {self.device}")
        print(f"  AMP: {cfg.use_amp}")
        print(f"  EMA: {cfg.use_ema}")
        print()

        pbar = tqdm(total=cfg.total_steps, desc="Training", unit="step")
        pbar.update(self.global_step)
        epoch = 0

        while self.global_step < cfg.total_steps:
            epoch += 1
            for batch in self.dataloader:
                if self.global_step >= cfg.total_steps:
                    break

                metrics = self.train_step(batch)

                # ── Update progress bar ──
                pbar.set_postfix(
                    loss=f"{metrics['loss']:.4f}", lr=f"{metrics['lr']:.2e}", epoch=epoch
                )
                pbar.update(1)

                # ── Wandb logging ──
                if self.wandb_enabled and self.global_step % cfg.log_freq == 0:
                    import wandb

                    wandb.log(
                        {
                            "train/loss": metrics["loss"],
                            "train/lr": metrics["lr"],
                            "train/epoch": epoch,
                            "train/step": self.global_step,
                        },
                        step=self.global_step,
                    )

                # ── Checkpointing ──
                if self.global_step % cfg.save_freq == 0 and self.global_step > 0:
                    self.save_checkpoint(tag=f"step_{self.global_step}")
                    self.save_checkpoint(tag="latest")

        pbar.close()

        # Final save
        self.save_checkpoint(tag="final")

        if self.wandb_enabled:
            import wandb

            wandb.finish()

        print(f"\nTraining complete! Total steps: {self.global_step}")
