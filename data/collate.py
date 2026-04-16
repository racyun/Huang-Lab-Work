from __future__ import annotations

from typing import Any

import torch


def tissue_chip_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack image tensors; keep ``boxes`` as a list of variable-length tensors."""
    out: dict[str, Any] = {
        "zstack": torch.stack([b["zstack"] for b in batch], dim=0),
        "focused": torch.stack([b["focused"] for b in batch], dim=0),
        "hybrid": torch.stack([b["hybrid"] for b in batch], dim=0),
        "stiffness": torch.stack([b["stiffness"] for b in batch], dim=0),
        "boxes": [b["boxes"] for b in batch],
    }
    if "well_id" in batch[0]:
        out["well_id"] = [b["well_id"] for b in batch]
    if "split" in batch[0]:
        out["split"] = [b["split"] for b in batch]
    return out


def pretrain_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Batch tensors for ``MultiEncoderMAE`` (drops string metadata)."""
    z = torch.stack([b["zstack"] for b in batch], dim=0)
    z = z.permute(0, 2, 1, 3, 4).contiguous()
    return {
        "zstack": z,
        "focused": torch.stack([b["focused"] for b in batch], dim=0),
        "hybrid": torch.stack([b["hybrid"] for b in batch], dim=0),
        "stiffness": torch.stack([b["stiffness"] for b in batch], dim=0),
    }
