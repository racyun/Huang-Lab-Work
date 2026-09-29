#!/usr/bin/env python3
"""Decide empirically whether the extracted features used the FIXED channel map.

Commit 1db5c10 (2026-06-29) corrected the channel constants: the tiles are in
RGB-plane order, so plane 0 is TAGLN (CH3) and plane 2 is DAPI (CH1). Before
that commit the extractor read DAPI from plane 0 and TAGLN from plane 2, which
means every CSV produced earlier has its ch1 and ch3 columns swapped AND an
EndMT score computed with DAPI standing in for the mesenchymal marker.

The extractor sets SKIP_EXISTING = True, so simply re-running it does NOT
regenerate CSVs that already exist. File dates are therefore weak evidence and
a re-run may silently have been a no-op.

This settles it directly, without reference to dates: recompute the per-cell
mean of each tile plane inside the segmentation mask, then check which plane
each recorded column actually matches. Correlation is used rather than exact
equality so that dtype and rounding differences do not matter.

    ch1 matches plane 2 and ch3 matches plane 0  -> FIXED map, features are fine
    ch1 matches plane 0 and ch3 matches plane 2  -> STALE, re-extraction needed

Usage:
    python verify_channel_extraction.py --metadata-dir <root>/cellwise_metadata \
        --imgs-dir <root>/imgs --masks-dir <root>/masks
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

IMG_EXTS = (".tif", ".tiff")


def _default_local_root() -> Path:
    default = Path("/teamspace/studios/this_studio/cellpose_work")
    if "CELLPOSE_LOCAL_ROOT" in os.environ:
        return Path(os.environ["CELLPOSE_LOCAL_ROOT"]).expanduser()
    if default.parent.parent.exists():
        return default
    return Path.home() / "cellpose_work"


def find_triple(metadata_dir: Path, imgs_dir: Path, masks_dir: Path):
    """First (metadata csv, tile, mask) trio that exists, across conditions."""
    for csv in sorted(metadata_dir.rglob("*_metadata.csv")):
        cond = csv.parent.name
        stem = csv.name[: -len("_metadata.csv")]
        tiles = [p for p in (imgs_dir / cond).rglob(f"{stem}.*")
                 if p.suffix.lower() in IMG_EXTS]
        masks = [p for p in (masks_dir / cond).rglob(f"{stem}_masks.*")
                 if p.suffix.lower() in IMG_EXTS]
        if not masks:
            masks = [p for p in masks_dir.rglob(f"{stem}_masks.*")
                     if p.suffix.lower() in IMG_EXTS]
        if tiles and masks:
            return csv, tiles[0], masks[0], cond, stem
    return None


def main() -> None:
    import tifffile
    from scipy import ndimage
    root = _default_local_root()
    ap = argparse.ArgumentParser(description="Which planes did Stage 1 read?")
    ap.add_argument("--metadata-dir", type=Path,
                    default=root / "cellwise_metadata")
    ap.add_argument("--imgs-dir", type=Path, default=root / "imgs")
    ap.add_argument("--masks-dir", type=Path, default=root / "masks")
    ap.add_argument("--n-images", type=int, default=3,
                    help="how many (csv, tile, mask) trios to check")
    args = ap.parse_args()

    checked = 0
    verdicts = []
    for csv in sorted(args.metadata_dir.rglob("*_metadata.csv")):
        if checked >= args.n_images:
            break
        cond = csv.parent.name
        stem = csv.name[: -len("_metadata.csv")]
        tiles = [p for p in (args.imgs_dir / cond).rglob(f"{stem}.*")
                 if p.suffix.lower() in IMG_EXTS]
        masks = [p for p in (args.masks_dir / cond).rglob(f"{stem}_masks.*")
                 if p.suffix.lower() in IMG_EXTS]
        if not masks:
            masks = [p for p in args.masks_dir.rglob(f"{stem}_masks.*")
                     if p.suffix.lower() in IMG_EXTS]
        if not (tiles and masks):
            continue

        df = pd.read_csv(csv)
        if "ch1_cellwise_mean_intensity" not in df.columns:
            continue
        im = tifffile.imread(str(tiles[0]))
        mask = tifffile.imread(str(masks[0]))
        if im.ndim != 3 or im.shape[-1] != 3:
            print(f"[skip] {tiles[0].name}: shape {im.shape}")
            continue

        ids = df["cell_id"].to_numpy()
        plane_means = {p: np.asarray(ndimage.mean(im[..., p], labels=mask,
                                                 index=ids), dtype=float)
                       for p in range(3)}
        print(f"\n=== {cond}/{stem}  ({len(ids)} cells) ===")
        print(f"{'recorded column':<34}{'plane 0':>12}{'plane 1':>12}{'plane 2':>12}"
              f"   best")
        matches = {}
        for col, label in (("ch1_cellwise_mean_intensity", "ch1 (labelled DAPI)"),
                           ("ch3_cellwise_mean_intensity", "ch3 (labelled TAGLN)")):
            rec = df[col].to_numpy(dtype=float)
            cors = []
            for p in range(3):
                ok = np.isfinite(rec) & np.isfinite(plane_means[p])
                cors.append(np.corrcoef(rec[ok], plane_means[p][ok])[0, 1]
                            if ok.sum() > 2 else np.nan)
            best = int(np.nanargmax(cors))
            matches[col] = best
            print(f"{label:<34}" + "".join(f"{c:>12.4f}" for c in cors) +
                  f"   plane {best}")

        ch1_p = matches["ch1_cellwise_mean_intensity"]
        ch3_p = matches["ch3_cellwise_mean_intensity"]
        if ch1_p == 2 and ch3_p == 0:
            v = "FIXED"
            print("  -> ch1 = plane 2 (DAPI), ch3 = plane 0 (TAGLN): "
                  "post-fix map. Features are correct.")
        elif ch1_p == 0 and ch3_p == 2:
            v = "STALE"
            print("  -> ch1 = plane 0, ch3 = plane 2: PRE-FIX map. ch1/ch3 are "
                  "swapped and the EndMT score used DAPI as the mesenchymal "
                  "marker. Re-extraction required.")
        else:
            v = "UNCLEAR"
            print(f"  -> unexpected pairing (ch1->{ch1_p}, ch3->{ch3_p}); "
                  f"inspect this image manually.")
        verdicts.append(v)
        checked += 1

    print("\n" + "=" * 68)
    if not verdicts:
        print("No (metadata, tile, mask) trio found. Check --metadata-dir, "
              "--imgs-dir and --masks-dir.")
    elif all(v == "FIXED" for v in verdicts):
        print("VERDICT: FIXED — the extracted features used the corrected")
        print("channel map. No re-extraction needed; motif 2's TAGLN (ch3)")
        print("value of -0.24 stands, and DAPI (ch1) is the one at +0.30.")
    elif all(v == "STALE" for v in verdicts):
        print("VERDICT: STALE — features predate commit 1db5c10.")
        print("ch1/ch3 are swapped and every EndMT score is wrong.")
        print("Delete the old CSVs (SKIP_EXISTING=True will otherwise skip")
        print("them), re-run Stage 1, then rebuild Stages 2-4.")
    else:
        print(f"VERDICT: MIXED {verdicts} — some images predate the fix.")
        print("Safest course is to delete all CSVs and re-extract.")
    print("=" * 68)


if __name__ == "__main__":
    main()
