#!/usr/bin/env python3
"""Stage 4 visualisation — paint motif labels back onto the tissue.

Produces "cluster maps": the Stage 2 spatial graph for an image, with each cell
coloured by the motif it was assigned. This is the final step of the project
doc (Step 6): "colour the tissue image by motif label to see where different
motifs appear spatially."

Motif colours are FIXED across every panel and every image, so a motif is the
same colour everywhere and panels can be compared directly.

Inputs:
    <stage4>/motifs.parquet                (from cluster_motifs.py)
    <root>/graphs/<condition>/<image_id>.pt  (Stage 2 graphs: pos + edges)
    optionally <root>/imgs/<condition>/tiles/<image_id>.tif  (--overlay)

Outputs:
    <out>/motif_map_grid.png       one panel per selected image, shared legend
    <out>/motif_map_<cond>__<img>.png   individual full-size maps

Usage:
    python plot_motif_maps.py --motifs <stage4>/motifs.parquet
    python plot_motif_maps.py --images-per-condition 2 --overlay
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _default_local_root() -> Path:
    default = Path("/teamspace/studios/this_studio/cellpose_work")
    if "CELLPOSE_LOCAL_ROOT" in os.environ:
        return Path(os.environ["CELLPOSE_LOCAL_ROOT"]).expanduser()
    if default.parent.parent.exists():
        return default
    return Path.home() / "cellpose_work"


def motif_colors(n_motifs: int):
    """Fixed colour per motif id, consistent across every image drawn."""
    base = plt.cm.tab10 if n_motifs <= 10 else plt.cm.tab20
    return [base(i % base.N) for i in range(n_motifs)]


def _stretch(ch):
    lo, hi = np.percentile(ch, [1, 99])
    return np.clip((ch.astype(float) - lo) / (hi - lo + 1e-6), 0, 1)


def _composite(im):
    """R=TAGLN(ch3), G=VE-cad(ch2), B=DAPI(ch1) — matches the other viewers."""
    if im.ndim == 2:
        s = _stretch(im)
        return np.dstack([s, s, s])
    rgb = np.zeros((*im.shape[:2], 3), dtype=float)
    rgb[..., 0] = _stretch(im[..., 2])
    rgb[..., 1] = _stretch(im[..., 1])
    rgb[..., 2] = _stretch(im[..., 0])
    return rgb


def draw_map(ax, g, motifs_for_image, colors, overlay_img=None, node_size=28,
             show_edges=True):
    """Draw one image's graph with nodes coloured by motif."""
    pos = g.pos.numpy()
    if overlay_img is not None:
        ax.imshow(_composite(overlay_img))
    if show_edges:
        ei = g.edge_index.numpy()
        edge_col = "white" if overlay_img is not None else "gray"
        alpha = 0.35 if overlay_img is not None else 0.4
        for s, d in ei.T:
            ax.plot([pos[s, 0], pos[d, 0]], [pos[s, 1], pos[d, 1]],
                    lw=0.35, color=edge_col, alpha=alpha, zorder=1)
    node_cols = [colors[m] for m in motifs_for_image]
    ax.scatter(pos[:, 0], pos[:, 1], c=node_cols, s=node_size, zorder=2,
               edgecolors="black", linewidths=0.25)
    if overlay_img is None:
        ax.invert_yaxis()          # imshow already puts row 0 at the top
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    root = _default_local_root()
    ap = argparse.ArgumentParser(description="Plot motif cluster maps (Stage 4).")
    ap.add_argument("--motifs", type=Path, default=root / "stage4" / "motifs.parquet")
    ap.add_argument("--graphs-dir", type=Path, default=root / "graphs")
    ap.add_argument("--imgs-dir", type=Path, default=root / "imgs")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: alongside the motifs file")
    ap.add_argument("--images-per-condition", type=int, default=1,
                    help="how many images to draw from each condition")
    ap.add_argument("--images", nargs="*", default=None,
                    help="explicit 'condition/image_id' entries instead of auto-select")
    ap.add_argument("--overlay", action="store_true",
                    help="draw on the raw microscopy tile instead of a blank panel")
    ap.add_argument("--no-edges", action="store_true", help="hide graph edges")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = args.out_dir or args.motifs.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    mdf = pd.read_parquet(args.motifs)
    n_motifs = int(mdf["motif"].max()) + 1
    colors = motif_colors(n_motifs)
    print(f"Loaded {len(mdf)} motif labels across {n_motifs} motifs")

    # ---- choose which images to draw ----
    if args.images:
        picks = [tuple(s.split("/", 1)) for s in args.images]
    else:
        rng = np.random.default_rng(args.seed)
        picks = []
        for cond, sub in mdf.groupby("condition"):
            imgs = sub["image_id"].unique()
            take = rng.choice(imgs, size=min(args.images_per_condition, len(imgs)),
                              replace=False)
            picks += [(cond, i) for i in take]
    print(f"Drawing {len(picks)} image(s)")

    panels = []
    for cond, img_id in picks:
        gpath = args.graphs_dir / cond / f"{img_id}.pt"
        if not gpath.exists():
            print(f"  [skip] no graph for {cond}/{img_id}")
            continue
        g = torch.load(str(gpath), weights_only=False)
        sub = mdf[(mdf.condition == cond) & (mdf.image_id == img_id)]
        if len(sub) != g.num_nodes:
            print(f"  [skip] {cond}/{img_id}: {len(sub)} labels vs {g.num_nodes} nodes")
            continue
        # rows are in node order (embed_cells walked cells 0..n-1 per graph)
        motifs_for_image = sub["motif"].to_numpy()

        raw = None
        if args.overlay:
            try:
                import tifffile
                tiles = args.imgs_dir / cond / "tiles"
                p = tiles / f"{img_id}.tif"
                if not p.exists():
                    cands = sorted(tiles.glob(f"{img_id}.*")) or sorted(tiles.glob(f"{img_id}*"))
                    p = cands[0] if cands else None
                if p is not None:
                    raw = tifffile.imread(str(p))
            except Exception as e:
                print(f"  [warn] overlay unavailable for {img_id}: {e}")

        # individual full-size map
        fig, ax = plt.subplots(figsize=(9, 9))
        draw_map(ax, g, motifs_for_image, colors, raw, node_size=36,
                 show_edges=not args.no_edges)
        ax.set_title(f"{cond} / {img_id}   ({g.num_nodes} cells)", fontsize=11)
        handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[m],
                          markeredgecolor="black", markersize=9, label=f"motif {m}")
                   for m in range(n_motifs)]
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  fontsize=9, frameon=False)
        fig.tight_layout()
        fname = out_dir / f"motif_map_{cond}__{img_id}.png"
        fig.savefig(fname, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {fname.name}")
        panels.append((cond, img_id, g, motifs_for_image, raw))

    # ---- comparison grid: all panels together, shared legend ----
    if panels:
        ncol = min(3, len(panels))
        nrow = int(np.ceil(len(panels) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 6 * nrow),
                                 squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for k, (cond, img_id, g, mfi, raw) in enumerate(panels):
            ax = axes[k // ncol][k % ncol]
            draw_map(ax, g, mfi, colors, raw, node_size=18,
                     show_edges=not args.no_edges)
            ax.set_title(f"{cond}\n{img_id}", fontsize=10)
        handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[m],
                          markeredgecolor="black", markersize=10, label=f"motif {m}")
                   for m in range(n_motifs)]
        fig.legend(handles=handles, loc="lower center", ncol=min(n_motifs, 8),
                   fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle("Motif cluster maps — cells coloured by assigned motif",
                     fontsize=13)
        fig.tight_layout()
        grid = out_dir / "motif_map_grid.png"
        fig.savefig(grid, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"\nWrote comparison grid -> {grid}")


if __name__ == "__main__":
    main()
