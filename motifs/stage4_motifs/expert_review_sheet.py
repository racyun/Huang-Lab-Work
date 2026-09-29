#!/usr/bin/env python3
"""Stage 4 — materials for expert (wet-lab) review of the motifs.

Every other visualisation in this pipeline draws node-link graphs, which a
bench biologist cannot assess. This script produces the two things an
experienced cell biologist CAN assess:

  1. A TRANSLATION TABLE. Motif feature profiles converted out of z-scores
     back into physical units (raw intensity AU, area in px^2, optionally um^2)
     by inverting the z-score with normalization_stats.json. "VE-cadherin 412
     vs 890 AU, area 780 vs 1850 px^2" is assessable by someone with calibrated
     bench intuition; "-0.77 SD" is not.

  2. IMAGE CROPS of real cells from each motif, cut from the original
     micrographs -- not diagrams. Three modes:
       --mode catalog   one page per motif, grid of crops, physical profile in
                        the title. "What does this motif actually look like?"
       --mode contrast  two motifs side by side, one per column. Human vision
                        is far better at relative than absolute judgements, so
                        a paired contrast is legible where a single motif page
                        is not.
       --mode blind     crops from all motifs shuffled and numbered, plus a
                        separate answer key. The expert labels them without
                        seeing the model's answer; agreement is then a real
                        accuracy number rather than another robustness metric.

Crops are centred on a cell's centroid. --pick chooses which cells:
  typical  = closest to the motif's centre in feature space (most representative)
  extreme  = furthest along the motif's own discriminating direction (easiest
             to see; use when motifs look alike at typical values)

Outputs (in --out-dir/expert_review/):
    motif_profiles_physical.csv
    motif_crops_<NN>.png           (catalog)
    contrast_<A>_vs_<B>.png        (contrast)
    blind_sheet_<N>.png + blind_answer_key.csv   (blind)

Usage:
    python expert_review_sheet.py --motifs <s4>/motifs.parquet \
        --graphs-dir <root>/graphs --imgs-dir <root>/imgs \
        --master-table <root>/master_table.csv \
        --norm-stats <root>/normalization_stats.json \
        --mode catalog --pick extreme
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_motif_maps import (_composite, _default_local_root, load_tile,
                             motif_colors)

# Physical meaning of each feature, for labels a biologist reads.
PRETTY = {
    "area_px": "cell area",
    "elongation": "elongation (major/minor)",
    "ch1_cellwise_mean_intensity": "DAPI (nuclear)",
    "ch2_cellwise_mean_membrane_intensity": "VE-cadherin (junction band)",
    "ch3_cellwise_mean_intensity": "TAGLN (mesenchymal)",
    "endmt_score": "EndMT score",
}
ZCOLS = ["area_px", "elongation", "ch1_cellwise_mean_intensity",
         "ch2_cellwise_mean_membrane_intensity", "ch3_cellwise_mean_intensity"]


def physical_profiles(mdf: pd.DataFrame, master: pd.DataFrame,
                      stats: dict, um_per_px: float | None) -> pd.DataFrame:
    """Per-motif means in physical units, by inverting the global z-score.

    raw = z * std + mean, with std/mean taken from normalization_stats.json.
    endmt_score was never z-scored, so it passes through untouched.
    """
    df = master.merge(mdf, on=["condition", "image_id", "cell_id"], how="inner")
    rows = []
    zs = stats.get("zscore", {})
    for m, grp in df.groupby("motif"):
        r = {"motif": int(m), "n_cells": len(grp),
             "pct_of_cells": round(100 * len(grp) / len(df), 2)}
        for c in ZCOLS:
            if c not in grp.columns or c not in zs:
                continue
            raw = grp[c].mean() * zs[c]["std"] + zs[c]["mean"]
            r[PRETTY[c]] = round(float(raw), 1)
            if c == "area_px" and um_per_px:
                r["cell area (um^2)"] = round(float(raw) * um_per_px ** 2, 1)
        if "endmt_score" in grp.columns:
            r[PRETTY["endmt_score"]] = round(float(grp["endmt_score"].mean()), 3)
        top = grp["condition"].value_counts()
        r["top condition"] = top.index[0]
        r["top condition %"] = round(100 * top.iloc[0] / len(grp), 1)
        rows.append(r)
    return pd.DataFrame(rows).set_index("motif")


def collect_cells(mdf, master, graphs_dir: Path, n_motifs: int, pick: str,
                  per_motif: int, scan_images: int, seed: int) -> dict:
    """Choose which real cells to crop, per motif.

    Returns {motif: [(cond, img, cell_id, cx, cy), ...]}.
    'typical' picks cells nearest the motif's feature-space centre; 'extreme'
    picks cells furthest from the GLOBAL centre along the motif's own mean
    direction, i.e. the clearest examples of what makes that motif distinct.
    """
    feats = [c for c in ZCOLS if c in master.columns] + (
        ["endmt_score"] if "endmt_score" in master.columns else [])
    df = master.merge(mdf, on=["condition", "image_id", "cell_id"], how="inner")

    pairs = df[["condition", "image_id"]].drop_duplicates()
    if len(pairs) > scan_images:
        pairs = pairs.sample(scan_images, random_state=seed)
    keep = pairs.set_index(["condition", "image_id"]).index
    df = df.set_index(["condition", "image_id"]).loc[
        df.set_index(["condition", "image_id"]).index.isin(keep)].reset_index()
    if df.empty:
        return {m: [] for m in range(n_motifs)}

    X = df[feats].to_numpy(dtype=float)
    out = {}
    for m in range(n_motifs):
        sel = (df["motif"] == m).to_numpy()
        if not sel.any():
            out[m] = []
            continue
        Xm = X[sel]
        centre = Xm.mean(0)
        if pick == "extreme":
            direction = centre - X.mean(0)
            nrm = np.linalg.norm(direction)
            score = -(Xm @ direction) / nrm if nrm > 1e-9 else np.zeros(len(Xm))
        else:
            score = np.linalg.norm(Xm - centre, axis=1)
        order = np.argsort(score)
        sub = df[sel].iloc[order]
        # spread picks over distinct images so one field doesn't dominate
        chosen, seen = [], {}
        for row in sub.itertuples(index=False):
            k = (row.condition, row.image_id)
            if seen.get(k, 0) >= max(1, per_motif // 3):
                continue
            seen[k] = seen.get(k, 0) + 1
            chosen.append((row.condition, row.image_id, int(row.cell_id)))
            if len(chosen) >= per_motif:
                break
        out[m] = chosen
    return out


def centroid_lookup(graphs_dir: Path, cond: str, img: str) -> dict:
    """cell_id -> (x, y) from the stage-2 graph, which stores true centroids."""
    gp = graphs_dir / cond / f"{img}.pt"
    if not gp.exists():
        return {}
    g = torch.load(str(gp), weights_only=False)
    pos = g.pos.numpy()
    ids = (g.cell_id.numpy() if hasattr(g, "cell_id")
           else np.arange(1, g.num_nodes + 1))
    return {int(c): (float(pos[i, 0]), float(pos[i, 1]))
            for i, c in enumerate(ids)}


def crop(tile, cx: float, cy: float, half: int):
    """Square window of the RGB composite centred on (cx, cy), edge-clamped."""
    rgb = _composite(tile)
    h, w = rgb.shape[:2]
    x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
    x1, y1 = int(min(cx + half, w)), int(min(cy + half, h))
    return rgb[y0:y1, x0:x1]


def _grid(items, ncol, cell_in=2.6):
    nrow = max(1, int(np.ceil(len(items) / ncol)))
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(cell_in * ncol, cell_in * nrow + 1.1))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    return fig, axes, nrow


def main() -> None:
    root = _default_local_root()
    ap = argparse.ArgumentParser(description="Expert-review materials for motifs.")
    ap.add_argument("--motifs", type=Path, default=root / "stage4" / "motifs.parquet")
    ap.add_argument("--graphs-dir", type=Path, default=root / "graphs")
    ap.add_argument("--imgs-dir", type=Path, default=root / "imgs")
    ap.add_argument("--master-table", type=Path, default=root / "master_table.csv")
    ap.add_argument("--norm-stats", type=Path,
                    default=root / "normalization_stats.json")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--mode", choices=["catalog", "contrast", "blind", "table"],
                    default="catalog")
    ap.add_argument("--pick", choices=["typical", "extreme"], default="extreme")
    ap.add_argument("--per-motif", type=int, default=12)
    ap.add_argument("--crop-size", type=int, default=160,
                    help="crop width in pixels (centred on the cell)")
    ap.add_argument("--contrast-pair", default=None,
                    help="two motif ids for --mode contrast, e.g. '2,3'. "
                         "Default: the most and least extreme motifs.")
    ap.add_argument("--um-per-px", type=float, default=None,
                    help="microns per pixel, to report area in um^2. Ask the "
                         "imaging metadata; omitted rather than guessed.")
    ap.add_argument("--scan-images", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = (args.out_dir or args.motifs.parent) / "expert_review"
    out_dir.mkdir(parents=True, exist_ok=True)

    mdf = pd.read_parquet(args.motifs)
    master = pd.read_csv(args.master_table)
    stats = (json.loads(args.norm_stats.read_text())
             if args.norm_stats.exists() else {})
    if not stats:
        print(f"[warn] {args.norm_stats} missing -> cannot convert to physical "
              f"units; the table will be in z-scores.")
    n_motifs = int(mdf["motif"].max()) + 1
    colors = motif_colors(n_motifs)

    # ---- 1. translation table (always) ----
    prof = physical_profiles(mdf, master, stats, args.um_per_px)
    prof.to_csv(out_dir / "motif_profiles_physical.csv")
    print("\nMOTIF PROFILES IN PHYSICAL UNITS")
    print("(raw intensities in arbitrary fluorescence units; area in pixels)")
    print(prof.to_string())
    print(f"\nWrote {out_dir / 'motif_profiles_physical.csv'}")
    if not args.um_per_px:
        print("[note] pass --um-per-px to add a um^2 column (not guessed)")
    if args.mode == "table":
        return

    # ---- 2. pick real cells and cut crops ----
    picks = collect_cells(mdf, master, args.graphs_dir, n_motifs, args.pick,
                          args.per_motif, args.scan_images, args.seed)
    half = args.crop_size // 2
    print(f"\nCutting {args.crop_size}px crops ({args.pick} cells), "
          f"up to {args.per_motif} per motif...")

    cache: dict = {}
    crops: dict = {m: [] for m in range(n_motifs)}
    for m, items in picks.items():
        for cond, img, cid in items:
            if (cond, img) not in cache:
                tile = load_tile(args.imgs_dir, cond, img)
                cache[(cond, img)] = (tile, centroid_lookup(args.graphs_dir,
                                                            cond, img))
            tile, cents = cache[(cond, img)]
            if tile is None or cid not in cents:
                continue
            cx, cy = cents[cid]
            c = crop(tile, cx, cy, half)
            if c.size and min(c.shape[:2]) > 8:
                crops[m].append((c, cond, img, cid))
        print(f"  motif {m:>2}: {len(crops[m])} crop(s)")

    def label_for(m):
        p = prof.loc[m] if m in prof.index else {}
        ve = p.get(PRETTY["ch2_cellwise_mean_membrane_intensity"], "?")
        ta = p.get(PRETTY["ch3_cellwise_mean_intensity"], "?")
        ar = p.get("cell area (um^2)", p.get(PRETTY["area_px"], "?"))
        em = p.get(PRETTY["endmt_score"], "?")
        return (f"Motif {m} — {p.get('pct_of_cells','?')}% of cells | "
                f"VE-cad {ve} · TAGLN {ta} · area {ar} · EndMT {em}\n"
                f"most common condition: {p.get('top condition','?')} "
                f"({p.get('top condition %','?')}%)")

    # ---- 3a. catalog: one page per motif ----
    if args.mode == "catalog":
        for m in range(n_motifs):
            if not crops[m]:
                continue
            fig, axes, _ = _grid(crops[m], ncol=4)
            for ax, (c, cond, img, cid) in zip(axes, crops[m]):
                ax.imshow(c)
                ax.set_title(f"{cond}\ncell {cid}", fontsize=7)
            fig.suptitle(label_for(m), fontsize=11, color=colors[m],
                         fontweight="bold")
            fig.text(0.5, 0.005, "R = TAGLN   G = VE-cadherin   B = DAPI",
                     ha="center", fontsize=8, color="dimgrey")
            fp = out_dir / f"motif_crops_{m:02d}.png"
            fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)
            print(f"  wrote {fp.name}")

    # ---- 3b. contrast: two motifs, column per motif ----
    if args.mode == "contrast":
        if args.contrast_pair:
            a, b = (int(x) for x in args.contrast_pair.split(","))
        else:
            key = PRETTY["ch2_cellwise_mean_membrane_intensity"]
            srt = prof[key].sort_values() if key in prof.columns else prof.index
            a, b = int(srt.index[0]), int(srt.index[-1])
        n = min(len(crops[a]), len(crops[b]), 8)
        if n:
            fig, axes = plt.subplots(n, 2, figsize=(6.2, 3.0 * n))
            axes = np.atleast_2d(axes)
            for i in range(n):
                for j, m in enumerate((a, b)):
                    c, cond, img, cid = crops[m][i]
                    axes[i, j].imshow(c); axes[i, j].axis("off")
                    if i == 0:
                        axes[i, j].set_title(f"Motif {m}", fontsize=13,
                                             color=colors[m], fontweight="bold")
                    axes[i, j].text(0.02, 0.02, cond, transform=axes[i, j].transAxes,
                                    fontsize=6, color="white")
            fig.suptitle(f"Motif {a} vs Motif {b} — same scale, {args.pick} cells\n"
                         f"{label_for(a)}\n{label_for(b)}", fontsize=9)
            fp = out_dir / f"contrast_{a}_vs_{b}.png"
            fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)
            print(f"  wrote {fp.name}")

    # ---- 3c. blind: shuffled, numbered, with a separate answer key ----
    if args.mode == "blind":
        rng = np.random.default_rng(args.seed)
        pool = [(m, *c) for m in range(n_motifs) for c in crops[m]]
        rng.shuffle(pool)
        per_page = 20
        key_rows = []
        for page in range(int(np.ceil(len(pool) / per_page))):
            chunk = pool[page * per_page:(page + 1) * per_page]
            fig, axes, _ = _grid(chunk, ncol=5, cell_in=2.4)
            for k, (ax, (m, c, cond, img, cid)) in enumerate(zip(axes, chunk)):
                num = page * per_page + k + 1
                ax.imshow(c)
                ax.set_title(f"#{num}", fontsize=11, fontweight="bold")
                key_rows.append({"crop": num, "motif": m, "condition": cond,
                                 "image_id": img, "cell_id": cid})
            fig.suptitle("Blind review — label each crop without the model's answer\n"
                         "R = TAGLN   G = VE-cadherin   B = DAPI", fontsize=11)
            fp = out_dir / f"blind_sheet_{page + 1}.png"
            fig.savefig(fp, dpi=150, bbox_inches="tight"); plt.close(fig)
            print(f"  wrote {fp.name}")
        pd.DataFrame(key_rows).to_csv(out_dir / "blind_answer_key.csv", index=False)
        print(f"  wrote blind_answer_key.csv  ({len(key_rows)} crops)")
        print("  -> keep the key hidden until the expert has labelled the sheets")


if __name__ == "__main__":
    main()
