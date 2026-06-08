"""Optimizer and learning rate scheduler factories.

Provides factory functions for creating optimizers and schedulers
with sensible defaults for speech separation training.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def create_optimizer(
    parameters,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    betas: Tuple[float, float] = (0.9, 0.999),
    **kwargs: Any,
) -> optim.Optimizer:
    """Create an optimizer for training.

    Args:
        parameters: Model parameters or parameter groups.
        optimizer_type: Type of optimizer.
            Options: ``"adam"``, ``"adamw"``, ``"sgd"``, ``"radam"``.
        learning_rate: Initial learning rate.
        weight_decay: Weight decay (L2 regularization) coefficient.
        betas: Adam beta parameters.
        **kwargs: Additional optimizer-specific arguments.

    Returns:
        PyTorch optimizer instance.

    Raises:
        ValueError: If optimizer_type is unsupported.
    """
    optimizer_type = optimizer_type.lower()

    if optimizer_type == "adam":
        return optim.Adam(
            parameters,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            **kwargs,
        )
    elif optimizer_type == "adamw":
        return optim.AdamW(
            parameters,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            **kwargs,
        )
    elif optimizer_type == "sgd":
        return optim.SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get("momentum", 0.9),
            nesterov=kwargs.get("nesterov", True),
        )
    elif optimizer_type == "radam":
        return optim.RAdam(
            parameters,
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            f"Choose from: adam, adamw, sgd, radam."
        )


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = "cosine_warmup",
    num_epochs: int = 100,
    warmup_epochs: int = 10,
    steps_per_epoch: Optional[int] = None,
    min_lr: float = 1e-6,
    **kwargs: Any,
) -> lr_scheduler.LRScheduler:
    """Create a learning rate scheduler.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_type: Type of scheduler.
            Options:
                - ``"cosine_warmup"``: Cosine annealing with linear warmup
                - ``"cosine"``: Cosine annealing without warmup
                - ``"step"``: Step decay
                - ``"plateau"``: Reduce on plateau
                - ``"onecycle"``: OneCycleLR
        num_epochs: Total number of training epochs.
        warmup_epochs: Number of warmup epochs (used for cosine_warmup).
        steps_per_epoch: Number of optimizer steps per epoch.
            Required if the scheduler is step-based rather than epoch-based.
        min_lr: Minimum learning rate.
        **kwargs: Additional scheduler-specific arguments.

    Returns:
        PyTorch LR scheduler.

    Raises:
        ValueError: If scheduler_type is unsupported.
    """
    scheduler_type = scheduler_type.lower()

    if scheduler_type == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=min_lr,
        )
    elif scheduler_type == "cosine_warmup":
        return CosineAnnealingWithWarmup(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=num_epochs,
            eta_min=min_lr,
        )
    elif scheduler_type == "step":
        step_size = kwargs.get("step_size", 30)
        gamma = kwargs.get("gamma", 0.5)
        return lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
    elif scheduler_type == "plateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 5),
            min_lr=min_lr,
        )
    elif scheduler_type == "onecycle":
        if steps_per_epoch is None:
            raise ValueError("steps_per_epoch is required for OneCycleLR")
        total_steps = steps_per_epoch * num_epochs
        return lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs.get("max_lr", optimizer.param_groups[0]["lr"]),
            total_steps=total_steps,
            pct_start=warmup_epochs / num_epochs,
            anneal_strategy="cos",
            final_div_factor=1e4,
        )
    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            f"Choose from: cosine, cosine_warmup, step, plateau, onecycle."
        )


class CosineAnnealingWithWarmup(lr_scheduler._LRScheduler):
    """Cosine annealing LR scheduler with linear warmup.

    During warmup: LR increases linearly from 0 to base_lr.
    After warmup: LR decays via cosine annealing to eta_min.

    Args:
        optimizer: Wrapped optimizer.
        warmup_epochs: Number of warmup epochs.
        total_epochs: Total number of epochs (including warmup).
        eta_min: Minimum LR at the end of annealing.
        last_epoch: The index of the last epoch.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        eta_min: float = 1e-6,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.cosine_epochs = total_epochs - warmup_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            progress = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * progress for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(1, self.cosine_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_decay
                for base_lr in self.base_lrs
            ]
