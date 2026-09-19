#!/usr/bin/env python3
"""Stage 4 — motif frequency across conditions.

Stacked bars showing what proportion of each condition's cells fall into each
motif: the headline scientific figure, since a shared motif vocabulary is only
useful if you can then compare how often each motif occurs per condition.

Two figures are produced:

  motif_frequency.png          stacked bars, one per condition (columns sum to
                               100%), ordered by substrate stiffness
  motif_frequency_grouped.png  grouped bars with +/- SEM computed ACROSS
                               IMAGES. The stacked view pools every cell, which
                               hides how variable a condition is between
                               images; this one shows whether a difference is
                               bigger than the image-to-image spread.

Motif colours match plot_motif_maps.py, so a motif is the same colour here as
in the cluster maps and catalog.

Outputs:
    <out>/motif_frequency.png
    <out>/motif_frequency_grouped.png
    <out>/motif_frequency.csv        the underlying proportions

Usage:
    python plot_motif_frequency.py --motifs <stage4>/motifs.parquet
    python plot_motif_frequency.py ... --merge-by-stiffness \
        --labels "endothelial sheet,transition focus,mesenchymal cluster,..."
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DRIVE_ROOT = "Fusion AI/Prof Huang Project/Cellpose feature extractions"


def push_to_drive(out_dir: Path, root: Path) -> None:
    """rclone the output directory up, mirroring its path under the local root.

    ~/cellpose_work/v4/stage4_quick  ->  gdrive:<DRIVE_ROOT>/v4/stage4_quick
    so the Drive layout matches the Studio layout without hardcoding either.
    """
    try:
        rel = out_dir.resolve().relative_to(root.resolve())
    except ValueError:
        rel = Path(out_dir.name)
    dest = f"gdrive:{DRIVE_ROOT}/{rel.as_posix()}"
    print(f"\nPushing to Drive -> {dest}")
    r = subprocess.run(["rclone", "copy", str(out_dir), dest,
                        "--transfers=8", "--checkers=16", "--progress"])
    if r.returncode != 0:
        raise RuntimeError(
            f"rclone failed (exit {r.returncode}). Is rclone installed and is "
            f"RCLONE_CONFIG exported in this shell?")
    print("Push complete.")


def _default_local_root() -> Path:
    default = Path("/teamspace/studios/this_studio/cellpose_work")
    if "CELLPOSE_LOCAL_ROOT" in os.environ:
        return Path(os.environ["CELLPOSE_LOCAL_ROOT"]).expanduser()
    if default.parent.parent.exists():
        return default
    return Path.home() / "cellpose_work"


def motif_colors(n: int):
    """Same palette/indexing as plot_motif_maps.py so colours agree."""
    base = plt.cm.tab10 if n <= 10 else plt.cm.tab20
    return [base(i % base.N) for i in range(n)]


def parse_stiffness(condition: str):
    """'260516_500kPa' -> (500.0, '500 kPa').  TC plastic sorts last (~GPa)."""
    s = str(condition)
    m = re.search(r"(\d+(?:\.\d+)?)\s*kPa", s, flags=re.I)
    if m:
        v = float(m.group(1))
        return v, f"{m.group(1)} kPa"
    if "tc" in s.lower():
        return 1e9, "TC plastic"
    return float("inf"), s


def condition_order(conds):
    """Sort conditions soft -> stiff; keep date to disambiguate repeats."""
    info = []
    for c in conds:
        v, lab = parse_stiffness(c)
        date = str(c).partition("_")[0]
        info.append((v, date, c, lab))
    info.sort(key=lambda t: (t[0], t[1]))
    # if a stiffness appears more than once, append the date so labels are unique
    counts = {}
    for _, _, _, lab in info:
        counts[lab] = counts.get(lab, 0) + 1
    ordered, labels = [], []
    for v, date, c, lab in info:
        ordered.append(c)
        labels.append(f"{lab}\n({date})" if counts[lab] > 1 else lab)
    return ordered, labels


def main() -> None:
    root = _default_local_root()
    ap = argparse.ArgumentParser(description="Motif frequency across conditions.")
    ap.add_argument("--motifs", type=Path, default=root / "stage4" / "motifs.parquet")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--labels", default=None,
                    help="comma-separated motif names, in motif-id order, e.g. "
                         "'endothelial sheet,transition focus,...'. Default: 'Motif N'")
    ap.add_argument("--merge-by-stiffness", action="store_true",
                    help="pool conditions with the same stiffness imaged on "
                         "different dates (cleaner scientifically; hides batch)")
    ap.add_argument("--title", default="Motif frequency across substrate stiffness")
    ap.add_argument("--push-to-drive", action="store_true",
                    help="rclone the output directory up to gdrive: when done")
    args = ap.parse_args()

    out_dir = args.out_dir or args.motifs.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    mdf = pd.read_parquet(args.motifs)
    n_motifs = int(mdf["motif"].max()) + 1
    colors = motif_colors(n_motifs)
    names = ([s.strip() for s in args.labels.split(",")] if args.labels
             else [f"Motif {m}" for m in range(n_motifs)])
    if len(names) != n_motifs:
        raise SystemExit(f"--labels has {len(names)} names but there are {n_motifs} motifs")

    group = "stiffness_group" if args.merge_by_stiffness else "condition"
    if args.merge_by_stiffness:
        mdf[group] = mdf["condition"].map(lambda c: parse_stiffness(c)[1])
        conds = sorted(mdf[group].unique(), key=lambda l: parse_stiffness(l)[0])
        labels = conds
    else:
        conds, labels = condition_order(mdf["condition"].unique())

    # ---- pooled proportions (stacked figure) ----
    ct = pd.crosstab(mdf[group], mdf["motif"]).reindex(conds).fillna(0)
    for m in range(n_motifs):
        if m not in ct.columns:
            ct[m] = 0
    ct = ct[sorted(ct.columns)]
    prop = ct.div(ct.sum(axis=1), axis=0)
    prop.to_csv(out_dir / "motif_frequency.csv")
    print("Motif proportions per condition:")
    print((100 * prop).round(1).to_string())

    fig, ax = plt.subplots(figsize=(1.9 * len(conds) + 4, 7))
    bottom = np.zeros(len(conds))
    for m in range(n_motifs):
        vals = 100 * prop[m].to_numpy()
        ax.bar(range(len(conds)), vals, bottom=bottom, color=colors[m],
               edgecolor="white", linewidth=0.8, label=f"Motif {m}: {names[m]}"
               if args.labels else f"Motif {m}")
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 4:                                   # label only legible slices
                ax.text(i, b + v / 2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        bottom += vals
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Cells assigned (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    ax.set_title(args.title, fontsize=14, fontweight="bold", loc="left")
    ax.text(0, 1.02, "Proportion of cells assigned to each motif, per condition",
            transform=ax.transAxes, fontsize=10, color="dimgrey")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=min(3, n_motifs), frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / "motif_frequency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {out_dir / 'motif_frequency.png'}")

    # ---- per-image spread (grouped figure with SEM) ----
    per_img = (mdf.groupby([group, "image_id", "motif"]).size()
               .rename("n").reset_index())
    tot = per_img.groupby([group, "image_id"])["n"].transform("sum")
    per_img["frac"] = per_img["n"] / tot
    piv = (per_img.pivot_table(index=[group, "image_id"], columns="motif",
                               values="frac", fill_value=0.0)
           .reindex(columns=range(n_motifs), fill_value=0.0))
    mean = piv.groupby(level=0).mean().reindex(conds)
    sem = piv.groupby(level=0).sem().reindex(conds)

    fig, ax = plt.subplots(figsize=(1.9 * len(conds) + 5, 6.5))
    width = 0.8 / n_motifs
    x = np.arange(len(conds))
    for m in range(n_motifs):
        ax.bar(x + m * width - 0.4 + width / 2, 100 * mean[m], width,
               yerr=100 * sem[m], capsize=3, color=colors[m],
               edgecolor="white", linewidth=0.5,
               label=f"Motif {m}: {names[m]}" if args.labels else f"Motif {m}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Cells assigned (%)  mean $\\pm$ SEM across images", fontsize=11)
    ax.set_title(args.title + " — per-image variability", fontsize=13,
                 fontweight="bold", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
              ncol=min(3, n_motifs), frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_dir / "motif_frequency_grouped.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_dir / 'motif_frequency_grouped.png'}")
    print(f"Wrote {out_dir / 'motif_frequency.csv'}")

    if args.push_to_drive:
        push_to_drive(out_dir, root)


if __name__ == "__main__":
    main()
