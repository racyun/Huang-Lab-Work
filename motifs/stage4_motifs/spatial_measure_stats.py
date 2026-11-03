#!/usr/bin/env python3
"""Do spatial measures of EndMT survive the same-biology session control?

Companion to marker_ratio_stats.py. Short answer from the 2026-11-03 run: the
arrangement measures do not, and the variability measures do.

Same two tests as marker_ratio_stats.py, applied to spatial statistics rather
than to per-cell levels:

  SIGNAL  5kPa vs 500kPa on 260516 (date-matched)
  BATCH   500kPa on 260516 vs 260522 (same biology, answer must be zero)

Each measure is computed per tile from the cell centroids, then averaged to the
imaging position, which is the honest replicate unit.

Measures, all on DAPI-normalised TAGLN unless noted:
  moran_i        spatial autocorrelation over 8 nearest neighbours. High = the
                 marker comes in patches; ~0 = scattered at random.
  neighbor_corr  correlation of a cell's value with its neighbours' mean. The
                 measure used earlier in the project.
  local_sd       mean within-neighbourhood spread. High = neighbouring cells
                 disagree.
  nn_dist        median distance to nearest neighbour; a crowding control that
                 carries no marker information at all.
  moran_i_endmt  Moran's I on the raw endmt_score, to show whether the session
                 problem is in the readout or in the spatial statistic.
"""
import json
import pathlib

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors

H = pathlib.Path.home()
K = 8
MIN_CELLS = 30          # Moran's I is unstable on tiny tiles

# ch1 is DAPI and ch3 is TAGLN: the chN names follow CH numbering, which is
# the reverse of the RGB plane order used in Stage 1.
DAPI = "ch1_cellwise_mean_intensity"
TAGLN = "ch3_cellwise_mean_intensity"


def hedges_g(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp < 1e-12:
        return float("nan")
    d = (a.mean() - b.mean()) / sp
    return float(d * (1 - 3 / (4 * (na + nb) - 9)))


def tile_measures(g):
    """Spatial statistics for one tile."""
    n = len(g)
    if n < MIN_CELLS:
        return None
    xy = g[["centroid_x", "centroid_y"]].to_numpy(float)
    nn = NearestNeighbors(n_neighbors=min(K, n - 1) + 1).fit(xy)
    dist, idx = nn.kneighbors(xy)
    dist, idx = dist[:, 1:], idx[:, 1:]          # drop self

    out = {"n_cells": n, "nn_dist": float(np.median(dist[:, 0]))}

    for name, col in (("", "tagln_per_dapi"), ("_endmt", "endmt_score")):
        v = g[col].to_numpy(float)
        z = v - v.mean()
        sd = v.std()
        if sd < 1e-12:
            out[f"moran_i{name}"] = np.nan
            if not name:
                out["neighbor_corr"] = np.nan
                out["local_sd"] = np.nan
            continue
        nb_mean = v[idx].mean(1)
        # Moran's I with row-standardised weights reduces to this
        out[f"moran_i{name}"] = float((z * (z[idx].mean(1))).sum() / (z * z).sum())
        if not name:
            out["neighbor_corr"] = float(np.corrcoef(v, nb_mean)[0, 1])
            out["local_sd"] = float(np.mean(v[idx].std(1)))
            # local_sd rides on the mean: triple every value and it triples
            # too. Dividing by the local mean makes it a genuine measure of
            # how much neighbours disagree, independent of overall level.
            loc_mu = v[idx].mean(1)
            ok = loc_mu > 1e-9
            out["local_cv"] = (float(np.mean(v[idx].std(1)[ok] / loc_mu[ok]))
                               if ok.any() else np.nan)
            # same idea for the whole tile, as a cross-check
            out["tile_cv"] = float(sd / v.mean()) if v.mean() > 1e-9 else np.nan
    return out


S = json.load(open(H / "cellpose_work/normalization_stats.json"))["zscore"]
cols = ["condition", "image_id", "cell_id", "centroid_x", "centroid_y",
        "endmt_score", TAGLN, DAPI]
m = pd.read_csv(H / "cellpose_work/master_table.csv", usecols=cols)
for c in (TAGLN, DAPI):
    m[c] = m[c] * S[c]["std"] + S[c]["mean"]
m = m[m[DAPI] >= 1.0].copy()
m["tagln_per_dapi"] = m[TAGLN] / m[DAPI]
m["date"] = m["condition"].str[:6]
m["stiffness"] = m["condition"].str.split("_").str[1:].str.join("_")
m["position"] = m["image_id"].str.extract(r"^(xy\d+)")[0].fillna(m["image_id"])

print(f"{len(m):,} cells; computing spatial stats per tile (k={K})...")
rows = []
for (cond, img), g in m.groupby(["condition", "image_id"], sort=False):
    r = tile_measures(g)
    if r is None:
        continue
    r.update(condition=cond, image_id=img, date=g["date"].iloc[0],
             stiffness=g["stiffness"].iloc[0], position=g["position"].iloc[0])
    rows.append(r)
tiles = pd.DataFrame(rows)
print(f"  {len(tiles):,} tiles with >= {MIN_CELLS} cells "
      f"({m.groupby(['condition','image_id']).ngroups - len(tiles):,} skipped)")

MEASURES = ["moran_i", "neighbor_corr", "local_sd", "local_cv", "tile_cv", "nn_dist", "moran_i_endmt"]
pos = tiles.groupby(["date", "stiffness", "position"], as_index=False)[MEASURES].mean()

sig_a = (pos.date == "260516") & (pos.stiffness == "5kPa")
sig_b = (pos.date == "260516") & (pos.stiffness == "500kPa")
bat_a = (pos.date == "260516") & (pos.stiffness == "500kPa")
bat_b = (pos.date == "260522") & (pos.stiffness == "500kPa")

print("\n" + "=" * 84)
print(f"SIGNAL  5kPa vs 500kPa on 260516  ({int(sig_a.sum())} vs {int(sig_b.sum())} positions)")
print(f"BATCH   500kPa on 260516 vs 260522 ({int(bat_a.sum())} vs {int(bat_b.sum())} positions)")
print("=" * 84)
print(f"{'measure':<18}{'5kPa':>9}{'500kPa':>9}{'signal g':>10}{'signal p':>11}"
      f"{'batch g':>9}{'ratio':>8}  verdict")
print("-" * 84)
for c in MEASURES:
    a = pos[sig_a][c].dropna().to_numpy()
    b = pos[sig_b][c].dropna().to_numpy()
    ba = pos[bat_a][c].dropna().to_numpy()
    bb = pos[bat_b][c].dropna().to_numpy()
    sg = hedges_g(a, b)
    _, sp_ = stats.ttest_ind(a, b, equal_var=False)
    bg = hedges_g(ba, bb)
    ratio = abs(sg) / max(abs(bg), 1e-9)
    verdict = ("biology" if ratio >= 3 else "mixed" if ratio >= 1.5
               else "SESSION-DOMINATED")
    print(f"{c:<18}{a.mean():>9.3f}{b.mean():>9.3f}{sg:>+10.2f}{sp_:>11.2g}"
          f"{bg:>+9.2f}{ratio:>8.2f}  {verdict}")
print("-" * 84)

print("\nper-condition means (position level):")
print(pos.groupby(["date", "stiffness"])[MEASURES].mean().round(3).to_string())
