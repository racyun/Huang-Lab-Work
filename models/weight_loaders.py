"""Load partial checkpoints (e.g. MAE encoder weights) with flexible key prefixes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def load_checkpoint_dict(path: Path | str) -> dict[str, Any]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "model" in ck:
        return ck["model"]
    return ck


def strip_prefix_load(
    module: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load keys ``prefix + name`` into ``module`` as ``name``."""
    sub = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
    return module.load_state_dict(sub, strict=strict)


def load_mae_encoder_from_multimae_ckpt(
    encoder: nn.Module,
    ckpt_path: Path | str,
    prefix: str = "mae_focused.encoder.",
) -> None:
    """Load encoder weights saved from ``MultiEncoderMAE`` (2D MAE head)."""
    sd = load_checkpoint_dict(ckpt_path)
    missing, unexpected = strip_prefix_load(encoder, sd, prefix, strict=False)
    if missing or unexpected:
        import warnings

        warnings.warn(f"Partial MAE encoder load: missing={len(missing)}, unexpected={len(unexpected)}")
