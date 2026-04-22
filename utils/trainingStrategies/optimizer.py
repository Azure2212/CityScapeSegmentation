"""Optimizer and scheduler factories for training runs."""

import torch.optim as optim
from torch.nn import Module


def build_optimizer(model: Module, lr: float) -> optim.Optimizer:
    """Build the optimizer used to update model parameters each step."""
    return optim.RAdam(model.parameters(), lr=lr)


def build_lr_scheduler(optimizer: optim.Optimizer, patience: int, min_lr: float = 1e-7):
    """Reduce the learning rate when validation IoU stops improving."""
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=patience, min_lr=min_lr
    )
