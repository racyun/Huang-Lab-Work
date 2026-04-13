"""Resize / crop samples so they match ``MultiMAEConfig`` spatial sizes."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from config.settings import FullConfig, MultiMAEConfig
from data.combined import TissueChipDataset


def _align_z(x: torch.Tensor, target_z: int) -> torch.Tensor:
    """x: [Z, C, H, W]"""
    z = x.shape[0]
    if z == target_z:
        return x
    if z > target_z:
        start = max(0, (z - target_z) // 2)
        return x[start : start + target_z].contiguous()
    pad = torch.zeros(
        target_z - z, *x.shape[1:], dtype=x.dtype, device=x.device
    )
    return torch.cat([x, pad], dim=0)


def _resize_chw(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if x.shape[-2] == h and x.shape[-1] == w:
        return x
    return F.interpolate(x.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)


class TissueChipPretrainDataset(Dataset):
    def __init__(self, base: TissueChipDataset, mm: MultiMAEConfig):
        self.base = base
        self.mm = mm

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.base[idx]
        z = _align_z(s["zstack"], self.mm.volume_z)
        z_ncdhw = z.permute(1, 0, 2, 3).unsqueeze(0)
        z_ncdhw = F.interpolate(
            z_ncdhw,
            size=(self.mm.volume_z, self.mm.volume_h, self.mm.volume_w),
            mode="trilinear",
            align_corners=False,
        )
        z = z_ncdhw.squeeze(0).permute(1, 0, 2, 3).contiguous()
        foc = _resize_chw(s["focused"], self.mm.image_size, self.mm.image_size)
        hyb = _resize_chw(s["hybrid"], self.mm.image_size, self.mm.image_size)
        return {
            "zstack": z,
            "focused": foc,
            "hybrid": hyb,
            "stiffness": s["stiffness"],
            "well_id": s.get("well_id", ""),
            "split": s.get("split", ""),
        }


def build_pretrain_dataset(cfg: FullConfig) -> Dataset:
    from pathlib import Path

    from data.cache import CachedTissueChipDataset, default_cache_key
    from data.combined import build_tissue_chip_dataset

    chip = build_tissue_chip_dataset(cfg.dataset)
    ds: Dataset = TissueChipPretrainDataset(chip, cfg.multi_mae)
    if cfg.dataset.cache_dir:
        ds = CachedTissueChipDataset(ds, Path(cfg.dataset.cache_dir), default_cache_key)
    return ds
