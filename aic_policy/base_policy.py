from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BasePolicy(ABC, nn.Module):
    """
    Abstract base for all trainable policies.
    Every policy (diffusion, flow matching, etc.) must inherit this
    and implement forwar() and predict_action().
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            batch: Dict with keys like 'observation.state',
            'observation.images.left_camera', 'action', etc.
            All tensors already normalized by LeRobot preprocessor.

        Returns:
            Dict with at least 'loss' key (scalar tensor for backprop).
            Can include extras for logging: {'loss': tensor, 'mse': 0.05}
        """
        ...

    @abstractmethod
    def predict_action(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Inference: generate action chunk from current observations.

        Args:
            observation: Dict with current state and images (no 'action' key).

        Returns:
            Action tensor of shape (n_actions_steps, action_dim)
        """
        ...

    def get_optimizer_groups(self, lr: float, weight_decay: float,
                             lr_backbone_mutliplier: float = 0.1) -> list[dict]:
        """
        Split parameters into two groups:
        - Backbone (pretrained vision encoder) -> low learning rate 
        - Everything else -> normal learning rate
        """
        backbone_params = []
        other_params = []

        # self.named_parameters() — this is a PyTorch nn.Module method that yields (name, parameter) pairs. 
        # The name is like "rgb_encoder.backbone.layer1.conv1.weight". 
        # We check if "rgb_encoder" or "backbone" appears in the name to decide which group it belongs to.

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "rgb_encoder" in name or "backbone" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
        
        return [
            {"params": other_params, "lr": lr, "weight_decay": weight_decay},
            {"params": backbone_params, "lr": lr * lr_backbone_multiplier,
             "weight_decay": weight_decay},
        ]




    
