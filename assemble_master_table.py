#!/usr/bin/env python3
"""Stage 2 prerequisite — assemble the master cell table.

Concatenates every per-image Stage 1 CSV into one long table, injects
``image_id`` (from filename) and ``condition`` (from folder), and z-scores the
morphology/intensity feature columns **globally** across the whole dataset.
``endmt_score`` is kept raw ([0, 1]) — only NaNs are imputed with the global
median — so its "0 = endothelial, 1 = mesenchymal" meaning survives.

See docs/stage2_graph_construction.md §1 for the rationale.

Input layout (Stage 1 output):
    <metadata_dir>/<condition>/<name>_metadata.csv

Outputs:
    <out_dir>/master_table.csv          one row per cell, normalized features
    <out_dir>/normalization_stats.json  per-column mean/std + endmt median

Usage:
    python assemble_master_table.py                       # use default paths
    python assemble_master_table.py --metadata-dir PATH --out-dir PATH
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

# Drive location (relative to the gdrive: rclone remote), matching the other stages.
DRIVE_ROOT = "Fusion AI/Prof Huang Project/Cellpose feature extractions"

# Feature columns that get GLOBAL z-scoring (large-magnitude / open-ended).
ZSCORE_COLS = [
    "area_px",
    "elongation",
    "ch1_cellwise_mean_intensity",
    "ch2_cellwise_mean_membrane_intensity",
    "ch3_cellwise_mean_intensity",
]
# Kept raw (already calibrated [0, 1]); NaNs imputed with the global median.
RAW_COLS = ["endmt_score"]
# Carried through untouched (geometry + identity); NOT node features.
PASSTHROUGH_COLS = ["cell_id", "centroid_x", "centroid_y"]


def _default_local_root() -> Path:
    """Mirror the Stage 1 LOCAL_ROOT logic so paths line up across stages."""
    default = Path("/teamspace/studios/this_studio/cellpose_work")
    if "CELLPOSE_LOCAL_ROOT" in os.environ:
        return Path(os.environ["CELLPOSE_LOCAL_ROOT"]).expanduser()
    if default.parent.parent.exists():  # /teamspace/studios exists -> Lightning
        return default
    return Path.home() / "cellpose_work"


def discover_csvs(metadata_dir: Path) -> list[tuple[str, Path]]:
    """Return (condition, csv_path) for every per-image CSV under metadata_dir."""
    if not metadata_dir.exists():
        raise FileNotFoundError(f"metadata dir not found: {metadata_dir}")
    out: list[tuple[str, Path]] = []
    for cond_dir in sorted(p for p in metadata_dir.iterdir() if p.is_dir()):
        for csv in sorted(cond_dir.glob("*.csv")):
            out.append((cond_dir.name, csv))
    return out


def image_id_from_path(csv_path: Path) -> str:
    """`<name>_metadata.csv` -> `<name>`."""
    return re.sub(r"_metadata$", "", csv_path.stem)


def push_file_to_drive(local: Path, drive_subpath: str = "") -> None:
    """rclone-copy a single file up to gdrive:<DRIVE_ROOT>/<drive_subpath>."""
    remote = f"gdrive:{DRIVE_ROOT}"
    if drive_subpath:
        remote = f"{remote}/{drive_subpath}"
    print(f"  $ rclone copyto {local.name} -> {remote}/{local.name}")
    r = subprocess.run(["rclone", "copyto", str(local), f"{remote}/{local.name}"])
    if r.returncode != 0:
        raise RuntimeError(f"rclone copyto failed (exit {r.returncode})")


def assemble(metadata_dir: Path, out_dir: Path, push_to_drive: bool = False) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = discover_csvs(metadata_dir)
    if not entries:
        raise RuntimeError(f"No per-image CSVs found under {metadata_dir}")
    print(f"Found {len(entries)} per-image CSV(s) under {metadata_dir}")

    frames = []
    for condition, csv in entries:
        df = pd.read_csv(csv)
        if df.empty:
            continue
        df.insert(0, "image_id", image_id_from_path(csv))
        df.insert(1, "condition", condition)
        frames.append(df)

    master = pd.concat(frames, ignore_index=True)
    print(f"Concatenated {len(master)} cells from {master['image_id'].nunique()} "
          f"image(s) across {master['condition'].nunique()} condition(s)")

    stats: dict = {"zscore": {}, "endmt_median": None, "n_cells": int(len(master))}

    # ----- global z-score (NaN-aware): (x - mean) / std, then NaN -> 0 (the mean) -----
    for col in ZSCORE_COLS:
        if col not in master.columns:
            print(f"  [warn] expected feature column missing: {col}")
            continue
        vals = master[col].to_numpy(dtype=float)
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals))  # population std (ddof=0)
        std_safe = std if std > 1e-12 else 1.0
        z = (vals - mean) / std_safe
        z[np.isnan(z)] = 0.0
        master[col] = z
        stats["zscore"][col] = {"mean": mean, "std": std}
        print(f"  z-scored {col:<40} mean={mean:.4g} std={std:.4g}")

    # ----- endmt_score: keep raw [0,1], impute NaN with global median -----
    if "endmt_score" in master.columns:
        em = master["endmt_score"].to_numpy(dtype=float)
        n_nan = int(np.isnan(em).sum())
        median = float(np.nanmedian(em)) if np.isfinite(np.nanmedian(em)) else 0.5
        em[np.isnan(em)] = median
        master["endmt_score"] = em
        stats["endmt_median"] = median
        print(f"  endmt_score kept raw; imputed {n_nan} NaN(s) with median={median:.4g}")

    master_path = out_dir / "master_table.csv"
    stats_path = out_dir / "normalization_stats.json"
    master.to_csv(master_path, index=False)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nWrote {master_path}")
    print(f"Wrote {stats_path}")

    if push_to_drive:
        print("\nPushing to Drive...")
        push_file_to_drive(master_path)
        push_file_to_drive(stats_path)

    return master


def main() -> None:
    root = _default_local_root()
    ap = argparse.ArgumentParser(description="Assemble the Stage 2 master cell table.")
    ap.add_argument("--metadata-dir", type=Path, default=root / "cellwise_metadata",
                    help="dir holding <condition>/<name>_metadata.csv (default: %(default)s)")
    ap.add_argument("--out-dir", type=Path, default=root,
                    help="where to write master_table.csv (default: %(default)s)")
    ap.add_argument("--push-to-drive", action="store_true",
                    help="rclone-copy master_table.csv + stats up to gdrive: when done")
    args = ap.parse_args()
    assemble(args.metadata_dir, args.out_dir, push_to_drive=args.push_to_drive)


if __name__ == "__main__":
    main()
