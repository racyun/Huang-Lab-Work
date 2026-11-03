#!/usr/bin/env python3
"""Quantify and draw the EndMT distribution on the one date-matched session.

Restricted to 260516, the only session holding two stiffnesses, so no imaging
session difference can enter. The readout is TAGLN divided by the same cell's
DAPI, which is the only readout that survived the same-biology control in
marker_ratio_stats.py.

Answers two questions that "are there motifs?" does not:

  HOW MUCH does the population spread, and does the spread differ by substrate
  WHERE does the extra signal come from — every cell shifting up together, or a
  growing tail of high-expressing cells

The second is read off the quantile ratios. A constant ratio across percentiles
means the whole population moved; a ratio that climbs with percentile means a
sub-population is driving the mean.

Spatial maps reassemble each imaging position from its 3x3 grid of tiles.
Centroids are stored per tile in a 0..~681 frame, so tile r#c# is offset by
(c-1, r-1) tile widths. Positions shown are the ones closest to their own
condition's median, not the most striking ones.

Usage
-----
    python plot_endmt_landscape.py
    python plot_endmt_landscape.py --out-dir ~/cellpose_work/landscape
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ch1 is DAPI and ch3 is TAGLN: the chN names follow CH numbering, which is
# the reverse of the RGB plane order used in Stage 1.
DAPI = "ch1_cellwise_mean_intensity"
TAGLN = "ch3_cellwise_mean_intensity"
DATE = "260516"
TILE = 681.0                      # tile frame width, from the centroid range
QUANTILES = [5, 10, 25, 50, 75, 90, 95, 99]


class _Tee:
    def __init__(self, path: Path):
        self.stream = sys.__stdout__
        self.fh = open(path, "w")

    def write(self, s):  # noqa: D102
        self.stream.write(s)
        self.fh.write(s)

    def flush(self):  # noqa: D102
        self.stream.flush()
        self.fh.flush()

    def close(self):  # noqa: D102
        self.fh.close()


def _default_local_root() -> Path:
    default = Path("/teamspace/studios/this_studio/cellpose_work")
    if "CELLPOSE_LOCAL_ROOT" in os.environ:
        return Path(os.environ["CELLPOSE_LOCAL_ROOT"]).expanduser()
    if default.parent.parent.exists():
        return default
    return Path.home() / "cellpose_work"


def load(master: Path, stats_path: Path) -> pd.DataFrame:
    stats = json.load(open(stats_path))
    scope = stats.get("zscore_scope") or "global"
    if scope != "global":
        raise SystemExit(f"[error] z-scoring scope is '{scope}'; raw intensities "
                         "are only recoverable from a global table")
    z = stats["zscore"]
    df = pd.read_csv(master, usecols=["condition", "image_id", "cell_id",
                                      "centroid_x", "centroid_y",
                                      "endmt_score", TAGLN, DAPI])
    for c in (TAGLN, DAPI):
        df[c] = df[c] * z[c]["std"] + z[c]["mean"]
    df = df[df.condition.str.startswith(DATE) & (df[DAPI] >= 1.0)].copy()
    df["value"] = df[TAGLN] / df[DAPI]
    n_bad = int((df["value"] <= 0).sum())
    if n_bad:
        print(f"  [note] dropped {n_bad} cell(s) with non-positive TAGLN "
              f"({100 * n_bad / len(df):.2f}%)")
        df = df[df["value"] > 0].copy()
    df["stiff"] = df["condition"].str.split("_").str[1]
    parts = df["image_id"].str.extract(r"^(?P<pos>.+?)_r(?P<r>\d+)c(?P<c>\d+)$")
    df["position"] = parts["pos"].fillna(df["image_id"])
    df["grid_r"] = pd.to_numeric(parts["r"], errors="coerce").fillna(1).astype(int)
    df["grid_c"] = pd.to_numeric(parts["c"], errors="coerce").fillna(1).astype(int)
    df["field_x"] = df["centroid_x"] + (df["grid_c"] - 1) * TILE
    df["field_y"] = df["centroid_y"] + (df["grid_r"] - 1) * TILE
    return df


def report_numbers(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    from scipy import stats as sps

    soft = df[df.stiff == "5kPa"]["value"].to_numpy()
    stiff = df[df.stiff == "500kPa"]["value"].to_numpy()

    print(f"\n{'=' * 72}\nHOW MUCH — spread within each substrate\n{'=' * 72}")
    for name, v in (("500kPa (stiff)", stiff), ("5kPa (soft)", soft)):
        print(f"  {name:<16} n={len(v):>7,}  mean={v.mean():6.3f}  "
              f"median={np.median(v):6.3f}  CV={v.std() / v.mean():5.3f}  "
              f"skew={sps.skew(v):+5.2f}")

    print(f"\n{'=' * 72}\nWHERE — is it a uniform shift, or a growing tail?\n{'=' * 72}")
    print(f"{'percentile':>11}{'500kPa':>10}{'5kPa':>10}{'ratio':>8}")
    rows = []
    for q in QUANTILES:
        a, b = float(np.percentile(stiff, q)), float(np.percentile(soft, q))
        rows.append({"percentile": q, "stiff_500kPa": a, "soft_5kPa": b,
                     "ratio": b / a})
        print(f"{q:>10}%{a:>10.3f}{b:>10.3f}{b / a:>8.2f}")
    tbl = pd.DataFrame(rows)
    tbl.to_csv(out_dir / "quantile_table.csv", index=False)

    lo, hi = tbl.ratio.iloc[0], tbl.ratio.iloc[-1]
    print(f"\n  ratio climbs {lo:.2f} -> {hi:.2f} from the 5th to the 99th "
          "percentile.")
    print("  The mean moves "
          f"{soft.mean() / stiff.mean():.2f}x but the 99th percentile moves "
          f"{hi:.2f}x, so the\n  increase is concentrated in a high-expressing "
          "tail rather than applied\n  evenly to every cell. The bottom 5% is "
          "actually LOWER on soft "
          f"({lo:.2f}x):\n  the population spreads apart, it does not "
          "translate.")

    thr = float(np.percentile(stiff, 90))
    fs, fh = float(np.mean(soft > thr)), float(np.mean(stiff > thr))
    print(f"\n  cells above the stiff 90th percentile ({thr:.2f}):")
    print(f"    500kPa {fh:.1%}   5kPa {fs:.1%}   -> {fs / fh:.1f}x as many")
    return tbl


def plot_distribution(df: pd.DataFrame, tbl: pd.DataFrame, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    soft = df[df.stiff == "5kPa"]["value"].to_numpy()
    stiff = df[df.stiff == "500kPa"]["value"].to_numpy()
    c_soft, c_stiff = "#c0392b", "#2471a3"

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    # Histogram the LOG of the value on even bins. Plotting the raw value on a
    # log axis with density=True divides each count by its bin width in raw
    # units, and log-spaced bins are vanishingly narrow at the low end, which
    # inflates a few percent of dim cells into a spurious second peak.
    bins = np.linspace(-2, 1.8, 80)
    ax.hist(np.log10(stiff), bins=bins, density=True, color=c_stiff,
            alpha=0.55, label="500 kPa (stiff)")
    ax.hist(np.log10(soft), bins=bins, density=True, color=c_soft,
            alpha=0.55, label="5 kPa (soft)")
    ticks = np.array([0.01, 0.1, 1, 10])
    ax.set_xticks(np.log10(ticks))
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_xlabel("TAGLN / DAPI  (log scale)")
    ax.set_ylabel("density")
    ax.set_title("One smooth peak each — a gradient,\nnot two cell types",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    ax.plot(tbl.percentile, tbl.ratio, "o-", color="#6b3fa0", lw=2)
    ax.axhline(1.0, color="#888888", ls=":", lw=1)
    ax.axhline(float(tbl.ratio.mean()), color="#bbbbbb", ls="--", lw=1,
               label="a uniform shift would be flat")
    ax.set_xlabel("percentile of the population")
    ax.set_ylabel("soft / stiff")
    ax.set_title("The top of the distribution moves\nmuch more than the bottom",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    grid = np.logspace(np.log10(0.05), np.log10(30), 160)
    ax.plot(grid, [(stiff > t).mean() for t in grid], color=c_stiff, lw=2,
            label="500 kPa")
    ax.plot(grid, [(soft > t).mean() for t in grid], color=c_soft, lw=2,
            label="5 kPa")
    thr = float(np.percentile(stiff, 90))
    ax.axvline(thr, color="#444444", ls=":", lw=1)
    ax.annotate(f"stiff 90th pct\n{(soft > thr).mean():.0%} vs "
                f"{(stiff > thr).mean():.0%}",
                xy=(thr, 0.55), xytext=(thr * 1.4, 0.72), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xscale("log")
    ax.set_xlabel("TAGLN / DAPI threshold")
    ax.set_ylabel("fraction of cells above")
    ax.set_title("How many cells clear a given level", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=8)
    fig.tight_layout()
    path = out_dir / "endmt_distribution.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def pick_typical(df: pd.DataFrame, stiff: str) -> str:
    """The position whose mean sits closest to that condition's median."""
    sub = df[df.stiff == stiff]
    per_pos = sub.groupby("position")["value"].mean()
    return str((per_pos - per_pos.median()).abs().idxmin())


def plot_maps(df: pd.DataFrame, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from scipy.ndimage import gaussian_filter

    picks = [("5kPa", pick_typical(df, "5kPa"), "5 kPa (soft)"),
             ("500kPa", pick_typical(df, "500kPa"), "500 kPa (stiff)")]
    vmin, vmax = 0.1, 20.0

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10))
    for row, (stiff, pos, label) in enumerate(picks):
        g = df[(df.stiff == stiff) & (df.position == pos)]
        x, y, v = (g.field_x.to_numpy(), g.field_y.to_numpy(),
                   g.value.to_numpy())

        ax = axes[row, 0]
        sc = ax.scatter(x, y, c=np.clip(v, vmin, vmax), s=7, cmap="magma",
                        norm=LogNorm(vmin=vmin, vmax=vmax), linewidths=0)
        ax.set_title(f"{label} — {len(g)} cells at {pos}\n"
                     f"each dot is one cell", fontsize=9)
        fig.colorbar(sc, ax=ax, fraction=0.046, label="TAGLN / DAPI")

        # binned + smoothed, so regional structure would be visible if present
        ax = axes[row, 1]
        nb = 34
        edges = np.linspace(0, 3 * TILE, nb + 1)
        tot, _, _ = np.histogram2d(y, x, bins=[edges, edges], weights=v)
        cnt, _, _ = np.histogram2d(y, x, bins=[edges, edges])
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = tot / cnt
        filled = np.where(cnt > 0, mean, np.nan)
        sm = gaussian_filter(np.nan_to_num(filled, nan=np.nanmean(filled)),
                             sigma=1.1)
        im = ax.imshow(sm, origin="lower", cmap="magma",
                       norm=LogNorm(vmin=vmin, vmax=vmax),
                       extent=(0, 3 * TILE, 0, 3 * TILE))
        ax.contour(sm, levels=[1, 2, 4, 8], colors="white", linewidths=0.6,
                   alpha=0.55, extent=(0, 3 * TILE, 0, 3 * TILE))
        ax.set_title(f"{label} — smoothed local average", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, label="TAGLN / DAPI")

        for a in axes[row]:
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle("High-TAGLN cells are scattered throughout, not gathered into "
                 "regions\n(same colour scale on all four panels)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out_dir / "endmt_spatial_maps.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    root = _default_local_root()
    ap = argparse.ArgumentParser(
        description="Quantify and draw the EndMT distribution on 260516.")
    ap.add_argument("--master", type=Path, default=root / "master_table.csv")
    ap.add_argument("--norm-stats", type=Path,
                    default=root / "normalization_stats.json")
    ap.add_argument("--out-dir", type=Path, default=root / "landscape")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tee = _Tee(args.out_dir / "endmt_landscape_log.txt")
    sys.stdout = tee
    try:
        print(f"plot_endmt_landscape.py — {datetime.now():%Y-%m-%d %H:%M:%S}")
        df = load(args.master, args.norm_stats)
        print(f"  {len(df):,} cells on {DATE}; "
              f"{df.groupby(['stiff', 'position']).ngroups} positions")
        tbl = report_numbers(df, args.out_dir)
        p1 = plot_distribution(df, tbl, args.out_dir)
        p2 = plot_maps(df, args.out_dir)
        print(f"\nWrote {args.out_dir / 'quantile_table.csv'}")
        print(f"Wrote {p1}")
        print(f"Wrote {p2}")
    finally:
        sys.stdout = sys.__stdout__
        print(f"Log saved -> {args.out_dir / 'endmt_landscape_log.txt'}")
        tee.close()


if __name__ == "__main__":
    main()
