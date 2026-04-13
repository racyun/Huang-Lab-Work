"""Stiffness (kPa) conditioning: scalar -> patch-space bias."""

from __future__ import annotations

import torch
import torch.nn as nn


class StiffnessMLP(nn.Module):
    """Maps normalized kPa [B,1] to embedding [B, D] added to every patch token."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, kpa: torch.Tensor) -> torch.Tensor:
        return self.net(kpa)


def normalize_kpa(kpa: torch.Tensor, mode: str, divisor: float = 900.0) -> torch.Tensor:
    if mode == "identity":
        return kpa
    if mode == "log1p":
        return torch.log1p(kpa.clamp_min(0.0))
    if mode == "divide":
        return kpa / divisor
    raise ValueError(f"Unknown kpa mode {mode!r}")
