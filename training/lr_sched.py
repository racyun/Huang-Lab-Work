"""Cosine schedule with linear warmup; scales every param group from its initial LR."""

from __future__ import annotations

import math

import torch


def set_epoch_learning_rates(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    *,
    epochs: int,
    warmup_epochs: int,
    min_lr: float,
    peak_lr: float,
) -> float:
    """Set each group's ``lr`` to ``initial_lr * factor`` (``initial_lr`` captured on first call)."""
    for g in optimizer.param_groups:
        if "initial_lr" not in g:
            g["initial_lr"] = g["lr"]

    min_lr_ratio = (min_lr / peak_lr) if peak_lr > 0 else 0.0

    if warmup_epochs > 0 and epoch < warmup_epochs:
        factor = (epoch + 1) / warmup_epochs
    else:
        if epochs <= warmup_epochs:
            t = 1.0
        else:
            t = (epoch - warmup_epochs) / max(1, (epochs - warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        factor = min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    for g in optimizer.param_groups:
        g["lr"] = g["initial_lr"] * factor
    return optimizer.param_groups[0]["lr"]
