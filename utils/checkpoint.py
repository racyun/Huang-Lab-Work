from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"model": model.state_dict()}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=strict)
    else:
        model.load_state_dict(ckpt, strict=strict)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
