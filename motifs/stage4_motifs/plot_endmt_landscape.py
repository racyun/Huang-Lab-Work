#!/usr/bin/env python3
"""Gradient or clusters, how wide, and where — EndMT on the date-matched pair.

Restricted to 260516, the only session holding two stiffnesses. Readout is the
Stage 1 ``endmt_score`` = TAGLN / (TAGLN + VE-cadherin), background-subtracted,
in [0, 1]. Three corrections are applied before anything is measured:

  imputed   Stage 2 fills cells with no valid score with the global median.
            Those are a spike at one exact value, not measurements; dropped.
  clamped   Stage 1 clamps a marker below background to zero, which pins the
            score to exactly 0 or exactly 1. These pile-ups are what make a
            raw dip test reject "one peak"; shape tests exclude them, and
            their share is reported because it differs a lot by condition.
  size      The score falls with cell area (r ~ -0.42 within condition) and
            soft cells are ~2.4x smaller. Soft-vs-stiff contrasts are
            therefore also reported between cells of matched size.

Questions answered, each with numbers:

  A  Is the score one continuous range, or separate groups?   (dip test)
  B  How wide is the range, and how does it differ by substrate?
  C  Within a tile, do similar cells sit together, how far does that
     similarity reach, and do high-EndMT cells favour crowded or sparse
     spots?  (Moran's I against a shuffle null; pair correlation by distance)
  D  Across a whole 3x3 field, are there broader patches, and how much of
     that is a fixed imaging pattern rather than the cells?

Earlier versions of this script plotted TAGLN/DAPI. That readout was built on
a swapped channel assignment and, once fixed, measured cell size; it is gone.

Usage
-----
    python plot_endmt_landscape.py
    python plot_endmt_landscape.py --out-dir ~/cellpose_work/landscape
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DATE = "260516"
TILE = 681.0                      # tile frame width, from the centroid range
K = 8                             # neighbours for Moran's I
N_PERM = 49                       # shuffles per tile for the Moran null
CROWD_RADIUS = 60.0               # px, for the local-density count
DIST_BINS = [0, 25, 50, 75, 100, 150, 200, 300, 450]
QUANTILES = [5, 10, 25, 50, 75, 90, 95]
SOFT, STIFF = "5kPa", "500kPa"
C_SOFT, C_STIFF = "#c0392b", "#2471a3"


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


# --------------------------------------------------------------------- data
def load(master: Path, stats_path: Path) -> pd.DataFrame:
    stats = json.load(open(stats_path))
    if (stats.get("zscore_scope") or "global") != "global":
        raise SystemExit("[error] cell area can only be recovered from a "
                         "globally z-scored table")
    za, med = stats["zscore"]["area_px"], stats.get("endmt_median")

    df = pd.read_csv(master, usecols=["condition", "image_id", "cell_id",
                                      "area_px", "centroid_x", "centroid_y",
                                      "endmt_score"])
    df = df[df.condition.str.startswith(DATE)].copy()
    df["area_px"] = df["area_px"] * za["std"] + za["mean"]
    df["stiff"] = df["condition"].str.split("_").str[1]

    n0 = len(df)
    if med is not None:
        df = df[df["endmt_score"] != med]
    print(f"  dropped {n0 - len(df):,} imputed cells (score == stored median "
          f"{med:.4f})")
    df = df[df["area_px"] > 0].copy()

    df["clamped"] = df["endmt_score"].isin([0.0, 1.0])
    df["log_area"] = np.log(df["area_px"])

    parts = df["image_id"].str.extract(r"^(?P<pos>.+?)_r(?P<r>\d+)c(?P<c>\d+)$")
    df["position"] = parts["pos"].fillna(df["image_id"])
    df["field_x"] = df["centroid_x"] + (pd.to_numeric(parts["c"]).fillna(1) - 1) * TILE
    df["field_y"] = df["centroid_y"] + (pd.to_numeric(parts["r"]).fillna(1) - 1) * TILE

    # Within-condition size adjustment, re-centred on the condition mean. It
    # removes size variation INSIDE a condition (for shape and spatial tests)
    # and deliberately leaves the between-condition gap alone; that gap gets
    # its own size-matched comparison.
    df["endmt_sizeadj"] = np.nan
    for _, g in df.groupby("stiff"):
        b = np.polyfit(g["log_area"], g["endmt_score"], 1)
        df.loc[g.index, "endmt_sizeadj"] = (g["endmt_score"]
                                            - np.polyval(b, g["log_area"])
                                            + g["endmt_score"].mean())
    return df


# ------------------------------------------------------------------ A and B
def shape_and_spread(df: pd.DataFrame, out_dir: Path, rng) -> dict:
    import diptest

    res: dict = {}
    print(f"\n{'=' * 74}\nA  ONE CONTINUOUS RANGE, OR SEPARATE GROUPS?\n{'=' * 74}")
    print("  share of cells pinned at exactly 0 or 1 by the background clamp:")
    for s in (STIFF, SOFT):
        v = df[df.stiff == s]["endmt_score"]
        print(f"    {s:>7}: at 0 {np.mean(v == 0):5.1%}   at 1 {np.mean(v == 1):5.1%}")

    print("\n  Hartigan dip test, null = one peak; 40 subsamples of 5,000,")
    print("  clamped cells excluded:")
    for s in (STIFF, SOFT):
        v = df[(df.stiff == s) & ~df.clamped]["endmt_score"].to_numpy()
        ps = np.array([diptest.diptest(rng.choice(v, 5000, replace=False))[1]
                       for _ in range(40)])
        verdict = "separate groups" if np.mean(ps < .05) > .5 else "one continuous range"
        res[f"dip_p_{s}"] = float(np.median(ps))
        print(f"    {s:>7}: median p={np.median(ps):.3f}, rejects one peak in "
              f"{np.mean(ps < .05):.0%}  -> {verdict}")

    print(f"\n{'=' * 74}\nB  HOW WIDE, AND HOW DOES IT DIFFER BY SUBSTRATE?\n{'=' * 74}")
    rows = []
    hdr = "".join(f"{q:>7}%" for q in QUANTILES)
    print(f"  {'':>8}{hdr}{'SD':>8}{'IQR':>8}{'>0.5':>8}")
    for s in (STIFF, SOFT):
        v = df[df.stiff == s]["endmt_score"].to_numpy()
        p = np.percentile(v, QUANTILES)
        iqr = float(np.percentile(v, 75) - np.percentile(v, 25))
        rows.append({"stiffness": s, **{f"p{q}": float(x) for q, x in zip(QUANTILES, p)},
                     "sd": float(v.std()), "iqr": iqr, "frac_above_0.5": float(np.mean(v > .5)),
                     "n": len(v)})
        print(f"  {s:>8}" + "".join(f"{x:>8.3f}" for x in p)
              + f"{v.std():>8.3f}{iqr:>8.3f}{np.mean(v > .5):>8.1%}")
    pd.DataFrame(rows).to_csv(out_dir / "spread_table.csv", index=False)

    # soft minus stiff between cells of the same size
    d = df.copy()
    d["abin"] = pd.qcut(d["area_px"], 20, labels=False, duplicates="drop")
    band = []
    for b, g in d.groupby("abin"):
        a = g[g.stiff == SOFT]["endmt_score"]
        c = g[g.stiff == STIFF]["endmt_score"]
        if len(a) > 100 and len(c) > 100:
            band.append({"area_band": int(b), "median_area": float(g.area_px.median()),
                         "soft": float(a.mean()), "stiff": float(c.mean()),
                         "diff": float(a.mean() - c.mean()), "n": len(g)})
    band = pd.DataFrame(band)
    band.to_csv(out_dir / "size_matched_difference.csv", index=False)
    raw = (df[df.stiff == SOFT]["endmt_score"].mean()
           - df[df.stiff == STIFF]["endmt_score"].mean())
    adj = float(np.average(band["diff"], weights=band["n"]))
    res.update(raw_diff=float(raw), size_matched_diff=adj, band=band)
    print(f"\n  soft minus stiff, mean score:")
    print(f"    all cells             {raw:+.3f}")
    print(f"    same-size cells only  {adj:+.3f}   ({adj / raw:.0%} of the raw gap; "
          f"soft is higher in {np.mean(band['diff'] > 0):.0%} of {len(band)} size bands)")
    print("  The rest of the raw gap is soft cells being smaller, and smaller cells "
          "scoring\n  higher in both conditions.")

    above = {s: float(np.mean(df[df.stiff == s]["endmt_score"] > .5)) for s in (STIFF, SOFT)}
    res["above_half"] = above
    print(f"\n  cells scoring above 0.5 (TAGLN signal exceeds VE-cadherin signal):")
    print(f"    {STIFF} {above[STIFF]:.1%}   {SOFT} {above[SOFT]:.1%}   "
          f"-> {above[SOFT] / above[STIFF]:.1f}x")
    return res


# ------------------------------------------------------------------------ C
def _morans(v, idx):
    z = v - v.mean()
    den = (z * z).sum()
    return float((z * z[idx].mean(1)).sum() / den) if den > 0 else np.nan


def spatial(df: pd.DataFrame, out_dir: Path, rng) -> tuple[pd.DataFrame, dict]:
    from scipy import stats
    from sklearn.neighbors import NearestNeighbors

    print(f"\n{'=' * 74}\nC  WHERE — do similar cells sit together, and how far does it reach?\n{'=' * 74}")
    rows = []
    for (cond, img), g in df.groupby(["condition", "image_id"], sort=False):
        if len(g) < 40:
            continue
        xy = g[["centroid_x", "centroid_y"]].to_numpy(float)
        _, idx = NearestNeighbors(n_neighbors=K + 1).fit(xy).kneighbors(xy)
        idx = idx[:, 1:]
        r = {"stiff": g.stiff.iloc[0], "image_id": img, "n_cells": len(g)}
        for col in ("endmt_score", "endmt_sizeadj"):
            v = g[col].to_numpy()
            obs = _morans(v, idx)
            null = np.array([_morans(rng.permutation(v), idx) for _ in range(N_PERM)])
            r[f"moran_{col}"] = obs
            r[f"z_{col}"] = (obs - null.mean()) / null.std() if null.std() > 0 else np.nan
        nb = NearestNeighbors(radius=CROWD_RADIUS).fit(xy).radius_neighbors(
            xy, return_distance=False)
        crowd = np.array([len(x) - 1 for x in nb], float)
        r["rho_crowding"] = (stats.spearmanr(crowd, g["endmt_score"])[0]
                             if crowd.std() > 0 else np.nan)
        rows.append(r)
    tiles = pd.DataFrame(rows)
    tiles.to_csv(out_dir / "tile_spatial_stats.csv", index=False)

    print(f"  Moran's I over {K} nearest neighbours, per tile, against {N_PERM} "
          "shuffles of\n  the same scores over the same cell positions. "
          "0 = arrangement looks random;\n  z > 2 = similar cells sit together "
          "more than chance allows.")
    for col, lab in (("endmt_score", "raw score"), ("endmt_sizeadj", "size removed")):
        for s in (STIFF, SOFT):
            g = tiles[tiles.stiff == s]
            print(f"    {lab:<13} {s:>7}: median I={g[f'moran_{col}'].median():+.3f}  "
                  f"median z={g[f'z_{col}'].median():4.1f}  "
                  f"tiles with z>2: {np.mean(g[f'z_{col}'] > 2):.0%}  (of {len(g)})")

    print(f"\n  Correlation between two cells' scores (size removed), by distance:")
    curve = {}
    for s in (STIFF, SOFT):
        acc = {i: [] for i in range(len(DIST_BINS) - 1)}
        groups = df[df.stiff == s].groupby("image_id")
        for img in rng.choice(list(groups.groups), min(150, groups.ngroups), replace=False):
            g = groups.get_group(img)
            xy = g[["centroid_x", "centroid_y"]].to_numpy(float)
            v = g["endmt_sizeadj"].to_numpy()
            if v.std() == 0:
                continue
            z = (v - v.mean()) / v.std()
            iu = np.triu_indices(len(g), 1)
            dist = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))[iu]
            prod = (z[:, None] * z[None])[iu]
            b = np.digitize(dist, DIST_BINS) - 1
            for i in acc:
                sel = b == i
                if sel.any():
                    acc[i].append(prod[sel].mean())
        curve[s] = [float(np.mean(acc[i])) for i in acc]
        print(f"    {s:>7}: " + "  ".join(
            f"{DIST_BINS[i]}-{DIST_BINS[i + 1]}:{curve[s][i]:+.2f}" for i in acc))
    diam = {s: float(2 * np.sqrt(df[df.stiff == s]["area_px"].median() / np.pi))
            for s in (STIFF, SOFT)}
    print(f"    (typical cell diameter: {STIFF} {diam[STIFF]:.0f} px, "
          f"{SOFT} {diam[SOFT]:.0f} px)")
    pd.DataFrame({"bin_lo": DIST_BINS[:-1], "bin_hi": DIST_BINS[1:],
                  **{f"corr_{s}": curve[s] for s in curve}}).to_csv(
        out_dir / "correlation_by_distance.csv", index=False)

    print(f"\n  Do high-EndMT cells favour crowded or sparse spots? Spearman of a "
          f"cell's\n  score against the number of cells within {CROWD_RADIUS:.0f} px, per tile:")
    for s in (STIFF, SOFT):
        g = tiles[tiles.stiff == s]["rho_crowding"].dropna()
        lean = "crowded" if g.median() > 0 else "sparse"
        print(f"    {s:>7}: median rho={g.median():+.3f}, positive in "
              f"{np.mean(g > 0):.0%} of tiles  -> high scores lean {lean}")
    return tiles, {"curve": curve, "diam": diam}


def field_scale(df: pd.DataFrame, out_dir: Path) -> None:
    """Structure at the scale of a whole imaging field, which the per-tile
    statistics above cannot see.

    Two different things show up here and need keeping apart. A pattern that
    sits in the same grid cell in every field is the microscope, not the cells:
    biology has no way to know where the centre of the camera is. A pattern
    that moves from field to field could be biology or uneven staining, and
    this dataset cannot tell those apart.
    """
    from scipy import stats

    print(f"\n{'=' * 74}\nD  LARGER PATCHES — across a whole 3x3 field\n{'=' * 74}")
    d = df.copy()
    rc = d["image_id"].str.extract(r"_r(?P<r>\d+)c(?P<c>\d+)$").astype(float)
    d["r"], d["c"] = rc["r"], rc["c"]
    t = (d.dropna(subset=["r", "c"])
         .groupby(["stiff", "position", "r", "c"])["endmt_score"].mean()
         .reset_index())
    rows = []
    for s in (STIFF, SOFT):
        g = t[t.stiff == s].copy()
        g["dev"] = g["endmt_score"] - g.groupby("position")["endmt_score"].transform("mean")
        total = float(((g.endmt_score - g.endmt_score.mean()) ** 2).mean())
        within = float((g.dev ** 2).mean())
        grid = g.groupby(["r", "c"])["dev"].mean()
        fixed_share = float(grid.var(ddof=0) / g.dev.var(ddof=0))
        p_r = stats.f_oneway(*[x.dev for _, x in g.groupby("r")]).pvalue
        p_c = stats.f_oneway(*[x.dev for _, x in g.groupby("c")]).pvalue
        print(f"  {s}:")
        print(f"    {within / total:.0%} of tile-to-tile variation lies between tiles "
              "of the same field")
        print(f"    of that, {fixed_share:.0%} is a fixed grid pattern shared by every "
              f"field (row p={p_r:.2g}, column p={p_c:.2g})")
        print(f"    centre tile vs field mean: {grid.loc[(2.0, 2.0)]:+.3f}   "
              f"grid pattern range {grid.max() - grid.min():.3f}   "
              f"typical tile spread (SD) {g.dev.std():.3f}")
        rows.append({"stiffness": s, "share_within_field": within / total,
                     "fixed_grid_share": fixed_share,
                     "centre_tile_dev": float(grid.loc[(2.0, 2.0)]),
                     "grid_range": float(grid.max() - grid.min()),
                     "tile_sd": float(g.dev.std())})
    pd.DataFrame(rows).to_csv(out_dir / "field_scale.csv", index=False)
    print("  A fixed centre-vs-edge pattern is an imaging signature (uneven "
          "illumination\n  or focus across the field). It is real but small; "
          "most field-scale\n  variation moves from field to field.")


# -------------------------------------------------------------------- plots
def plot_distribution(df: pd.DataFrame, res: dict, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    ax = axes[0]
    bins = np.linspace(0, 1, 51)
    for s, c, lab in ((STIFF, C_STIFF, "500 kPa (stiff)"), (SOFT, C_SOFT, "5 kPa (soft)")):
        v = df[(df.stiff == s) & ~df.clamped]["endmt_score"]
        ax.hist(v, bins=bins, density=True, color=c, alpha=0.5, label=lab)
    ax.axvline(0.5, color="#555555", ls=":", lw=1)
    ax.set_xlabel("EndMT score  (0 = endothelial, 1 = mesenchymal)")
    ax.set_ylabel("density")
    ax.set_title("One continuous range per substrate\n(cells pinned at 0 or 1 by the "
                 "clamp left out)", fontsize=9.5)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    grid = np.linspace(0, 1, 201)
    for s, c in ((STIFF, C_STIFF), (SOFT, C_SOFT)):
        v = df[df.stiff == s]["endmt_score"].to_numpy()
        ax.plot(grid, [(v > t).mean() for t in grid], color=c, lw=2,
                label=f"{s}: {res['above_half'][s]:.0%} above 0.5")
    ax.axvline(0.5, color="#555555", ls=":", lw=1)
    ax.set_xlabel("EndMT score threshold")
    ax.set_ylabel("fraction of cells above")
    ax.set_title("How many cells clear a given score", fontsize=9.5)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    b = res["band"]
    ax.plot(b["median_area"], b["soft"], "o-", color=C_SOFT, ms=4, label="5 kPa")
    ax.plot(b["median_area"], b["stiff"], "o-", color=C_STIFF, ms=4, label="500 kPa")
    ax.set_xscale("log")
    ax.set_xlabel("cell area (px, log scale)")
    ax.set_ylabel("mean EndMT score")
    ax.set_title(f"Same-size cells: soft still higher, by {res['size_matched_diff']:+.3f}\n"
                 f"(vs {res['raw_diff']:+.3f} unmatched)", fontsize=9.5)
    ax.legend(fontsize=8, frameon=False)

    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
        a.tick_params(labelsize=8)
    fig.tight_layout()
    path = out_dir / "endmt_distribution.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _typical(df: pd.DataFrame, s: str) -> str:
    per = df[df.stiff == s].groupby("position")["endmt_score"].mean()
    return str((per - per.median()).abs().idxmin())


def plot_maps(df: pd.DataFrame, sp: dict, out_dir: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    fig = plt.figure(figsize=(13, 9.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.9])
    nb = 40
    edges = np.linspace(0, 3 * TILE, nb + 1)
    ext = (0, 3 * TILE, 0, 3 * TILE)

    for row, (s, lab) in enumerate(((SOFT, "5 kPa (soft)"), (STIFF, "500 kPa (stiff)"))):
        pos = _typical(df, s)
        g = df[(df.stiff == s) & (df.position == pos)]

        ax = fig.add_subplot(gs[row, 0])
        sc = ax.scatter(g.field_x, g.field_y, c=g.endmt_score, cmap="coolwarm",
                        vmin=0, vmax=1, s=6, linewidths=0)
        ax.set_title(f"{lab}, {pos}: {len(g):,} cells\none dot per cell", fontsize=9)
        ax.set_xlim(0, 3 * TILE)
        ax.set_ylim(0, 3 * TILE)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.046, label="EndMT score")

        # normalised convolution: smooth the sum and the count separately, so
        # empty bins do not drag the average toward an invented fill value
        tot, _, _ = np.histogram2d(g.field_y, g.field_x, bins=[edges, edges],
                                   weights=g.endmt_score)
        cnt, _, _ = np.histogram2d(g.field_y, g.field_x, bins=[edges, edges])
        sm_t, sm_c = gaussian_filter(tot, 1.2), gaussian_filter(cnt, 1.2)
        with np.errstate(invalid="ignore", divide="ignore"):
            sm = np.where(sm_c > 0.15, sm_t / sm_c, np.nan)
        ax = fig.add_subplot(gs[row, 1])
        im = ax.imshow(sm, origin="lower", cmap="coolwarm", vmin=0, vmax=1, extent=ext)
        cs = ax.contour(sm, levels=[0.3, 0.5, 0.7], colors="black",
                        linewidths=0.6, alpha=0.6, extent=ext)
        ax.clabel(cs, fontsize=6, fmt="%.1f")
        ax.set_title(f"{lab}: local average, contours at 0.3 / 0.5 / 0.7", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, label="EndMT score")

    ax = fig.add_subplot(gs[:, 2])
    mids = [(DIST_BINS[i] + DIST_BINS[i + 1]) / 2 for i in range(len(DIST_BINS) - 1)]
    for s, c, lab in ((SOFT, C_SOFT, "5 kPa"), (STIFF, C_STIFF, "500 kPa")):
        ax.plot(mids, sp["curve"][s], "o-", color=c, lw=2, ms=4, label=lab)
        ax.axvline(sp["diam"][s], color=c, ls=":", lw=1)
    ax.axhline(0, color="#888888", lw=0.8)
    ax.set_xlabel("distance between two cells (px)")
    ax.set_ylabel("correlation of their EndMT scores\n(cell size removed)")
    ax.set_title("How far does similarity reach?\ndotted lines = one typical cell "
                 "diameter", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Neighbouring cells resemble each other over a few cell diameters; "
                 "some fields also carry broader patches (left, middle)",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = out_dir / "endmt_spatial_maps.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def main() -> None:
    root = _default_local_root()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--master", type=Path, default=root / "master_table.csv")
    ap.add_argument("--norm-stats", type=Path, default=root / "normalization_stats.json")
    ap.add_argument("--out-dir", type=Path, default=root / "landscape")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tee = _Tee(args.out_dir / "endmt_landscape_log.txt")
    sys.stdout = tee
    try:
        print(f"plot_endmt_landscape.py — {datetime.now():%Y-%m-%d %H:%M:%S}")
        df = load(args.master, args.norm_stats)
        print(f"  {len(df):,} cells on {DATE}: "
              + ", ".join(f"{s} {n:,}" for s, n in df.groupby('stiff').size().items()))
        res = shape_and_spread(df, args.out_dir, rng)
        _, sp = spatial(df, args.out_dir, rng)
        field_scale(df, args.out_dir)
        p1 = plot_distribution(df, res, args.out_dir)
        p2 = plot_maps(df, sp, args.out_dir)
        print(f"\n  [caveat] one dish per substrate: these numbers describe this dish "
              "pair.\n  [caveat] touching cells can share some signal at their "
              "boundary, which\n           inflates similarity at the shortest "
              "distance; the reach beyond one\n           cell diameter is the part "
              "that bleed-through cannot explain.")
        for f in ("spread_table.csv", "size_matched_difference.csv",
                  "tile_spatial_stats.csv", "correlation_by_distance.csv",
                  "field_scale.csv"):
            print(f"Wrote {args.out_dir / f}")
        print(f"Wrote {p1}\nWrote {p2}")
    finally:
        sys.stdout = sys.__stdout__
        print(f"Log saved -> {args.out_dir / 'endmt_landscape_log.txt'}")
        tee.close()


if __name__ == "__main__":
    main()
