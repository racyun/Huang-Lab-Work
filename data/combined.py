from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import torch
from torch.utils.data import ConcatDataset, Dataset

from config.settings import DatasetConfig, SplitConfig
from data.boxes import load_boxes_txt
from data.modalities import (
    FocusedModalDataset,
    HybridModalDataset,
    ZStackModalDataset,
    discover_focus_paths,
)


def _well_ids_for_split(
    split: SplitConfig,
    cfg: DatasetConfig,
) -> list[str]:
    """Wells that have z-stack folder, hybrid folder, label file, and a discovered focus image."""
    if not split.zstack_root or not split.focused_root or not split.hybrid_root or not split.labels_root:
        return []

    zroot = Path(split.zstack_root)
    hroot = Path(split.hybrid_root)
    lroot = Path(split.labels_root)
    focus_map = discover_focus_paths(Path(split.focused_root), cfg.focus_filename_glob)

    wells: list[str] = []
    for i in range(1, cfg.well_count + 1):
        well_id = f"{cfg.well_prefix}{i:03d}"
        zdir = zroot / well_id / cfg.zstack_subdir
        if not zdir.is_dir():
            continue
        tifs = [p for p in zdir.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}]
        if not tifs:
            continue
        hy_folder = hroot / cfg.hybrid_folder_template.format(well_id=well_id)
        if not hy_folder.is_dir():
            continue
        hy_tifs = [p for p in hy_folder.iterdir() if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}]
        if not hy_tifs:
            continue
        if well_id not in focus_map:
            continue
        lbl = lroot / f"{well_id}.txt"
        if not lbl.is_file():
            continue
        wells.append(well_id)

    return sorted(wells)


def _resize_tuple(x: Optional[list[int]]) -> Optional[tuple[int, int]]:
    if x is None:
        return None
    if len(x) != 2:
        raise ValueError(f"resize must be [H, W], got {x}")
    return int(x[0]), int(x[1])


@dataclass
class _SplitBundle:
    split_name: str
    well_ids: list[str]
    z_ds: ZStackModalDataset
    f_ds: FocusedModalDataset
    h_ds: HybridModalDataset
    labels_root: Path
    stiffness: float


class _JoinedSplitDataset(Dataset):
    """Single stiffness condition: aligned z-stack, focus, hybrid + boxes."""

    def __init__(self, bundle: _SplitBundle):
        self.bundle = bundle
        n = len(bundle.well_ids)
        if not (len(bundle.z_ds) == n == len(bundle.f_ds) == len(bundle.h_ds)):
            raise AssertionError("Internal dataset length mismatch")

    def __len__(self) -> int:
        return len(self.bundle.well_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        zstack, wz = self.bundle.z_ds[idx]
        focused, wf = self.bundle.f_ds[idx]
        hybrid, wh = self.bundle.h_ds[idx]
        if wz != wf or wz != wh:
            raise RuntimeError(f"Well id mismatch {wz} {wf} {wh}")
        well_id = wz
        boxes = load_boxes_txt(self.bundle.labels_root / f"{well_id}.txt")
        stiffness = torch.tensor([self.bundle.stiffness], dtype=torch.float32)
        return {
            "zstack": zstack,
            "focused": focused,
            "hybrid": hybrid,
            "stiffness": stiffness,
            "boxes": boxes,
            "well_id": well_id,
            "split": self.bundle.split_name,
        }


class TissueChipDataset(Dataset):
    """Concatenation over stiffness splits; each item is one well."""

    def __init__(self, subsets: list[_JoinedSplitDataset]):
        if not subsets:
            raise ValueError("TissueChipDataset requires at least one non-empty split.")
        self._concat = ConcatDataset(subsets)

    def __len__(self) -> int:
        return len(self._concat)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._concat[idx]


def build_tissue_chip_dataset(cfg: DatasetConfig) -> TissueChipDataset:
    bundles: list[_SplitBundle] = []
    rz = _resize_tuple(cfg.resize.get("zstack"))
    rf = _resize_tuple(cfg.resize.get("focused"))
    rh = _resize_tuple(cfg.resize.get("hybrid"))

    for split in cfg.splits:
        wells = _well_ids_for_split(split, cfg)
        if not wells:
            continue
        focus_map = discover_focus_paths(Path(split.focused_root), cfg.focus_filename_glob)
        z_ds = ZStackModalDataset(split, wells, cfg.zstack_subdir, cfg.expected_z_slices, rz)
        f_ds = FocusedModalDataset(split, wells, focus_map, rf)
        h_ds = HybridModalDataset(split, wells, cfg.hybrid_folder_template, rh, cfg.hybrid_filename_template)
        bundles.append(
            _SplitBundle(
                split_name=split.name,
                well_ids=wells,
                z_ds=z_ds,
                f_ds=f_ds,
                h_ds=h_ds,
                labels_root=Path(split.labels_root),
                stiffness=float(split.stiffness_kpa),
            )
        )

    if not bundles:
        raise ValueError(
            "No valid samples found. Check that config paths are set and directory layout matches "
            "(z-stack W###/P00001/*.tif, focus *focus_stacked*.tif with W### in path, "
            "hybrid hybrid_results_W###/*.tif, labels W###.txt)."
        )

    return TissueChipDataset([_JoinedSplitDataset(b) for b in bundles])


def iter_splits_metadata(cfg: DatasetConfig) -> Iterator[tuple[str, list[str]]]:
    """Yield (split_name, well_ids) for logging."""
    for split in cfg.splits:
        yield split.name, _well_ids_for_split(split, cfg)
