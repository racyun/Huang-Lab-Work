#!/usr/bin/env python3
"""Diagnose which tile plane holds which marker — settle it by looking.

The whole pipeline assumes the focus-stacked tiles are in RGB-plane order with
plane 0 = TAGLN (CH3), 1 = VE-cadherin (CH2), 2 = DAPI (CH1). Stage 1 extracts
features through that assumption, so if it is wrong the EndMT score is computed
from the wrong channels and everything downstream is affected — not just the
figure colours.

This writes one figure with:
  row 1  each plane on its own, in greyscale, labelled by index
  row 2  the two candidate colour composites, so the correct one can simply be
         picked by eye

DAPI is unmistakable: discrete, compact, round nuclei, roughly one per cell.
VE-cadherin outlines cell borders. TAGLN is diffuse and cytoplasmic. Whichever
greyscale panel shows clean round nuclei IS the DAPI plane.

Per-plane statistics are printed as a cross-check. "signal compactness" is the
fraction of pixels holding half the plane's total intensity — nuclei are small
and bright, so the DAPI plane concentrates its signal into the smallest area
and should have the LOWEST value.

Usage:
    python check_channel_order.py --tile <path to one .tif>
    python check_channel_order.py --imgs-dir <root>/imgs        # picks one
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_EXTS = (".tif", ".tiff")


def _default_local_root() -> Path:
    default = Path("/teamspace/studios/this_studio/cellpose_work")
    if "CELLPOSE_LOCAL_ROOT" in os.environ:
        return Path(os.environ["CELLPOSE_LOCAL_ROOT"]).expanduser()
    if default.parent.parent.exists():
        return default
    return Path.home() / "cellpose_work"


def _stretch(ch):
    lo, hi = np.percentile(ch, [1, 99])
    return np.clip((ch.astype(float) - lo) / (hi - lo + 1e-6), 0, 1)


def compactness(ch) -> float:
    """Fraction of pixels that hold 50% of the plane's total intensity."""
    v = np.sort(ch.astype(float).ravel())[::-1]
    tot = v.sum()
    if tot <= 0:
        return float("nan")
    return float(np.searchsorted(np.cumsum(v), 0.5 * tot) + 1) / v.size


def blob_stats(ch):
    """Count and median size of bright connected regions (nuclei-like objects)."""
    try:
        from scipy import ndimage
    except ImportError:
        return float("nan"), float("nan")
    thr = np.percentile(ch, 97)
    lab, n = ndimage.label(ch > thr)
    if n == 0:
        return 0, float("nan")
    sizes = np.bincount(lab.ravel())[1:]
    return int(n), float(np.median(sizes))


def main() -> None:
    import tifffile
    root = _default_local_root()
    ap = argparse.ArgumentParser(description="Which plane is which marker?")
    ap.add_argument("--tile", type=Path, default=None)
    ap.add_argument("--imgs-dir", type=Path, default=root / "imgs")
    ap.add_argument("--out", type=Path, default=Path("channel_order_check.png"))
    args = ap.parse_args()

    tile = args.tile
    if tile is None:
        cands = [p for p in args.imgs_dir.rglob("*")
                 if p.suffix.lower() in IMG_EXTS]
        if not cands:
            raise SystemExit(f"no tiles found under {args.imgs_dir}")
        tile = sorted(cands)[len(cands) // 2]
    print(f"Tile: {tile}")

    im = tifffile.imread(str(tile))
    print(f"Shape: {im.shape}  dtype: {im.dtype}")
    if im.ndim != 3 or im.shape[-1] != 3:
        raise SystemExit(f"expected an (H, W, 3) tile, got {im.shape}")

    print("\nPER-PLANE STATISTICS")
    print(f"{'plane':<7}{'mean':>10}{'std':>10}{'compactness':>14}"
          f"{'bright blobs':>14}{'median blob px':>16}")
    stats = []
    for i in range(3):
        ch = im[..., i]
        c = compactness(ch)
        n, med = blob_stats(ch)
        stats.append((i, float(ch.mean()), float(ch.std()), c, n, med))
        print(f"{i:<7}{ch.mean():>10.1f}{ch.std():>10.1f}{c:>14.4f}"
              f"{n:>14}{med:>16.1f}")

    nuclear = min(stats, key=lambda s: s[3] if np.isfinite(s[3]) else 9e9)[0]
    print(f"\nMost compact (most nucleus-like) plane: {nuclear}")
    print("  The pipeline assumes plane 2 = DAPI. If the most compact plane is")
    print("  NOT 2, the plane order assumption is probably wrong — confirm by")
    print("  eye in the figure, then say which plane shows round nuclei.")

    # ---- figure: planes alone, then the two candidate composites ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.4))
    names = ["plane 0", "plane 1", "plane 2"]
    for i in range(3):
        axes[0, i].imshow(_stretch(im[..., i]), cmap="gray")
        axes[0, i].set_title(
            f"{names[i]}  (greyscale)\ncompactness {stats[i][3]:.4f}   "
            f"{stats[i][4]} blobs",
            fontsize=11, fontweight="bold" if i == nuclear else "normal")
        axes[0, i].axis("off")

    ident = np.dstack([_stretch(im[..., 0]), _stretch(im[..., 1]),
                       _stretch(im[..., 2])])
    rev = np.dstack([_stretch(im[..., 2]), _stretch(im[..., 1]),
                     _stretch(im[..., 0])])
    axes[1, 0].imshow(ident)
    axes[1, 0].set_title("A: planes straight through\n"
                         "R=p0  G=p1  B=p2   (current code)", fontsize=11)
    axes[1, 1].imshow(rev)
    axes[1, 1].set_title("B: red/blue reversed\n"
                         "R=p2  G=p1  B=p0   (previous code)", fontsize=11)
    for a in (axes[1, 0], axes[1, 1]):
        a.axis("off")
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0, 0.5,
        "WHICH COMPOSITE IS RIGHT?\n\n"
        "Correct one should show:\n"
        "  - blue round nuclei, ~1 per cell\n"
        "  - green outlining cell borders\n"
        "    (VE-cadherin at junctions)\n"
        "  - red diffuse in cytoplasm\n"
        "    (TAGLN)\n\n"
        "If A is right, no further change.\n"
        "If B is right, the plane order is\n"
        "reversed and STAGE 1 read the\n"
        "wrong channels -> features and\n"
        "EndMT scores need re-extracting.",
        fontsize=11, va="center", family="monospace")
    fig.suptitle(f"Channel-order check — {tile.name}", fontsize=13,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {args.out.resolve()}")
    print("Open it and say which of A or B looks biologically correct.")


if __name__ == "__main__":
    main()
