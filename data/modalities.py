from __future__ import annotations

import re
from pathlib import Path
from fnmatch import fnmatch
from typing import Optional

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from config.settings import SplitConfig

IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _pil_to_chw_float(img: Image.Image) -> torch.Tensor:
    return TF.to_tensor(img.convert("RGB"))


def _maybe_resize_chw(t: torch.Tensor, size_hw: Optional[tuple[int, int]]) -> torch.Tensor:
    if size_hw is None:
        return t
    h, w = size_hw
    return torch.nn.functional.interpolate(t.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)


def discover_focus_paths(root: Path, glob_pat: str) -> dict[str, Path]:
    """Map well id (e.g. W001) -> path using regex on full path string."""
    root = Path(root)
    out: dict[str, Path] = {}
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        if not fnmatch(p.name, glob_pat):
            continue
        m = re.search(r"(W\d{3})", str(p))
        if m:
            wid = m.group(1)
            if wid not in out:
                out[wid] = p
    return out


class ZStackModalDataset(Dataset):
    """One z-stack per well: tensor [Z, C, H, W] in float32."""

    def __init__(
        self,
        split: SplitConfig,
        well_ids: list[str],
        zstack_subdir: str,
        expected_z_slices: Optional[int],
        resize_hw: Optional[tuple[int, int]] = None,
    ):
        self.root = Path(split.zstack_root)
        self.stiffness = float(split.stiffness_kpa)
        self.well_ids = list(well_ids)
        self.zstack_subdir = zstack_subdir
        self.expected_z_slices = expected_z_slices
        self.resize_hw = resize_hw

    def __len__(self) -> int:
        return len(self.well_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        well_id = self.well_ids[idx]
        pdir = self.root / well_id / self.zstack_subdir
        paths = sorted([p for p in pdir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
        if not paths:
            raise FileNotFoundError(f"No images under {pdir}")
        if self.expected_z_slices is not None and len(paths) != self.expected_z_slices:
            import warnings

            warnings.warn(
                f"{well_id}: expected {self.expected_z_slices} z-slices, found {len(paths)}",
                stacklevel=2,
            )
        slices = []
        for p in paths:
            with Image.open(p) as im:
                t = _pil_to_chw_float(im)
            t = _maybe_resize_chw(t, self.resize_hw)
            slices.append(t)
        stack = torch.stack(slices, dim=0)
        return stack, well_id


class FocusedModalDataset(Dataset):
    def __init__(
        self,
        split: SplitConfig,
        well_ids: list[str],
        well_to_path: dict[str, Path],
        resize_hw: Optional[tuple[int, int]] = None,
    ):
        self.stiffness = float(split.stiffness_kpa)
        self.well_ids = list(well_ids)
        self.well_to_path = dict(well_to_path)
        self.resize_hw = resize_hw

    def __len__(self) -> int:
        return len(self.well_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        well_id = self.well_ids[idx]
        p = self.well_to_path[well_id]
        with Image.open(p) as im:
            t = _pil_to_chw_float(im)
        t = _maybe_resize_chw(t, self.resize_hw)
        return t, well_id


class HybridModalDataset(Dataset):
    def __init__(
        self,
        split: SplitConfig,
        well_ids: list[str],
        hybrid_folder_template: str,
        resize_hw: Optional[tuple[int, int]] = None,
        hybrid_filename_template: Optional[str] = None,
    ):
        self.root = Path(split.hybrid_root)
        self.stiffness = float(split.stiffness_kpa)
        self.well_ids = list(well_ids)
        self.hybrid_folder_template = hybrid_folder_template
        self.hybrid_filename_template = hybrid_filename_template
        self.resize_hw = resize_hw

    def __len__(self) -> int:
        return len(self.well_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        well_id = self.well_ids[idx]
        if self.hybrid_filename_template:
            # Flat layout: one file per well directly in hybrid_root
            p = self.root / self.hybrid_filename_template.format(well_id=well_id)
            if not p.is_file():
                raise FileNotFoundError(f"Hybrid image not found: {p}")
        else:
            # Subfolder layout: hybrid_root/hybrid_results_W001/<image>
            folder = self.root / self.hybrid_folder_template.format(well_id=well_id)
            paths = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
            if not paths:
                raise FileNotFoundError(f"No hybrid image in {folder}")
            p = paths[0]
        with Image.open(p) as im:
            t = _pil_to_chw_float(im)
        t = _maybe_resize_chw(t, self.resize_hw)
        return t, well_id
