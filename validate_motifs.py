#!/usr/bin/env python3
"""Stage 4 validation — are the motifs real, spatial, and recurring?

A clustering algorithm always returns clusters. These tests ask whether those
clusters mean anything:

  1. SPATIAL COHERENCE  Do neighbouring cells share a motif more than chance?
     A "spatial motif" should form contiguous domains, not salt-and-pepper.
     Measured as edge concordance (fraction of graph edges joining same-motif
     cells) against a within-image label-permutation null. Ratio ~1.0 means the
     motifs carry NO spatial information — the single most important check.

  2. RECURRENCE  Does each motif appear across many images, or is it a one-off?
     A recurring pattern should show up in most images at a stable frequency.

  3. STABILITY  Re-cluster with different seeds; do the same motifs come back?
     Adjusted Rand Index between runs. Low ARI = the partition is arbitrary.

  4. SEPARATION  Silhouette score: are the clusters distinct, or arbitrary cuts
     through one continuous blob?

Outputs:
    <out>/validation_report.json
    <out>/motif_recurrence.csv

Usage:
    python validate_motifs.py --motifs <stage4>/motifs.parquet
    python validate_motifs.py --motifs ... --embeddings ... --max-images 500
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _default_local_root() -> Path:
    default = Path("/teamspace/studios/this_studio/cellpose_work")
    if "CELLPOSE_LOCAL_ROOT" in os.environ:
        return Path(os.environ["CELLPOSE_LOCAL_ROOT"]).expanduser()
    if default.parent.parent.exists():
        return default
    return Path.home() / "cellpose_work"


def spatial_coherence(mdf: pd.DataFrame, graphs_dir: Path, max_images: int,
                      n_perm: int, seed: int, n_motifs: int) -> dict:
    """Edge concordance vs a within-image permutation null.

    observed = P(neighbours share a motif). null = same after shuffling labels
    within each image (preserves motif composition, destroys spatial layout).
    """
    rng = np.random.default_rng(seed)
    pairs = mdf[["condition", "image_id"]].drop_duplicates()
    if len(pairs) > max_images:
        pairs = pairs.sample(max_images, random_state=seed)
    print(f"\n1. SPATIAL COHERENCE  ({len(pairs)} images, {n_perm} permutations each)")

    obs_match = obs_tot = 0
    null_match = null_tot = 0
    per_motif_obs = np.zeros(n_motifs); per_motif_tot = np.zeros(n_motifs)
    per_motif_null = np.zeros(n_motifs)
    used = 0

    for cond, img in pairs.itertuples(index=False):
        gp = graphs_dir / cond / f"{img}.pt"
        if not gp.exists():
            continue
        g = torch.load(str(gp), weights_only=False)
        sub = mdf[(mdf.condition == cond) & (mdf.image_id == img)]
        if len(sub) != g.num_nodes or g.edge_index.shape[1] == 0:
            continue
        lab = sub["motif"].to_numpy()
        src, dst = g.edge_index.numpy()
        used += 1

        match = lab[src] == lab[dst]
        obs_match += match.sum(); obs_tot += len(match)
        # per-motif: of edges leaving a motif-m cell, how many land on motif m?
        for m in range(n_motifs):
            sel = lab[src] == m
            per_motif_tot[m] += sel.sum()
            per_motif_obs[m] += match[sel].sum()

        for _ in range(n_perm):
            perm = rng.permutation(lab)
            pm = perm[src] == perm[dst]
            null_match += pm.sum(); null_tot += len(pm)
            for m in range(n_motifs):
                sel = perm[src] == m
                per_motif_null[m] += pm[sel].sum()

    if obs_tot == 0:
        print("  [skip] no usable images")
        return {}

    obs = obs_match / obs_tot
    null = null_match / max(1, null_tot)
    ratio = obs / null if null > 0 else float("nan")
    print(f"  observed neighbour concordance : {obs:.3f}")
    print(f"  permuted (null) concordance    : {null:.3f}")
    print(f"  ratio                          : {ratio:.2f}x")
    if ratio >= 1.5:
        verdict = "STRONG — motifs form spatial domains"
    elif ratio >= 1.15:
        verdict = "MODERATE — some spatial structure"
    else:
        verdict = ("WEAK — motifs are spatially ~random (salt-and-pepper); they are "
                   "not describing spatial neighbourhoods")
    print(f"  => {verdict}")

    per_motif = {}
    for m in range(n_motifs):
        if per_motif_tot[m] > 0:
            o = per_motif_obs[m] / per_motif_tot[m]
            nl = per_motif_null[m] / max(1, per_motif_tot[m] * n_perm)
            per_motif[int(m)] = {"observed": float(o), "null": float(nl),
                                 "ratio": float(o / nl) if nl > 0 else None}
    print("  per-motif concordance ratio:")
    for m, v in per_motif.items():
        r = v["ratio"]
        print(f"    motif {m:>2}: obs={v['observed']:.3f} null={v['null']:.3f} "
              f"ratio={r:.2f}x" if r else f"    motif {m}: n/a")
    return {"n_images_used": used, "observed": float(obs), "null": float(null),
            "ratio": float(ratio), "verdict": verdict, "per_motif": per_motif}


def recurrence(mdf: pd.DataFrame, out_dir: Path, n_motifs: int) -> dict:
    """How consistently does each motif appear across images?"""
    print("\n2. RECURRENCE across images")
    frac = (mdf.groupby(["condition", "image_id", "motif"]).size()
            .rename("n").reset_index())
    tot = frac.groupby(["condition", "image_id"])["n"].transform("sum")
    frac["fraction"] = frac["n"] / tot
    n_images = mdf[["condition", "image_id"]].drop_duplicates().shape[0]

    rows = []
    piv = frac.pivot_table(index=["condition", "image_id"], columns="motif",
                           values="fraction", fill_value=0.0)
    for m in range(n_motifs):
        col = piv[m] if m in piv.columns else pd.Series(0.0, index=piv.index)
        rows.append({
            "motif": m,
            "mean_fraction": float(col.mean()),
            "std_fraction": float(col.std()),
            "pct_images_present": float(100 * (col > 0).mean()),
            "pct_images_above_5pct": float(100 * (col >= 0.05).mean()),
        })
    rec = pd.DataFrame(rows).set_index("motif")
    rec.to_csv(out_dir / "motif_recurrence.csv")
    print(f"  {n_images} images total")
    print(rec.round(3).to_string())
    widespread = int((rec["pct_images_above_5pct"] >= 50).sum())
    print(f"  => {widespread}/{n_motifs} motifs appear at >=5% of cells in "
          f"the majority of images (recurring, not one-offs)")
    return {"n_images": int(n_images),
            "per_motif": rec.reset_index().to_dict(orient="records"),
            "n_widespread": widespread}


def stability(emb_path: Path, mdf: pd.DataFrame, n_motifs: int,
              n_cells: int, seed: int) -> dict:
    """Re-cluster with different seeds; ARI between partitions."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    print("\n3. STABILITY (re-cluster with different seeds) + 4. SEPARATION")
    df = pd.read_parquet(emb_path) if emb_path.suffix == ".parquet" else pd.read_csv(emb_path)
    cols = [c for c in df.columns if c.startswith("emb_")]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(n_cells, len(df)), replace=False)
    X = df[cols].to_numpy(dtype=np.float32)[idx]

    labs = [KMeans(n_clusters=n_motifs, n_init=10, random_state=s).fit_predict(X)
            for s in (0, 1, 2)]
    aris = [adjusted_rand_score(labs[i], labs[j]) for i, j in ((0, 1), (0, 2), (1, 2))]
    ari = float(np.mean(aris))
    print(f"  mean ARI across 3 seeds: {ari:.3f}")
    if ari >= 0.7:
        s_verdict = "STABLE — the same partition recurs"
    elif ari >= 0.4:
        s_verdict = "MODERATE — partition partly seed-dependent"
    else:
        s_verdict = "UNSTABLE — the partition is largely arbitrary"
    print(f"  => {s_verdict}")

    sil = float(silhouette_score(X, labs[0], sample_size=min(10000, len(X)),
                                 random_state=seed))
    print(f"  silhouette score: {sil:.3f}")
    if sil >= 0.5:
        sep = "well-separated clusters"
    elif sil >= 0.25:
        sep = "weak but real separation"
    else:
        sep = ("essentially no separation — clusters are arbitrary cuts through "
               "one continuous blob")
    print(f"  => {sep}")
    return {"mean_ari": ari, "ari_pairs": [float(a) for a in aris],
            "stability_verdict": s_verdict, "silhouette": sil, "separation": sep}


def main() -> None:
    root = _default_local_root()
    ap = argparse.ArgumentParser(description="Validate discovered motifs (Stage 4).")
    ap.add_argument("--motifs", type=Path, default=root / "stage4" / "motifs.parquet")
    ap.add_argument("--graphs-dir", type=Path, default=root / "graphs")
    ap.add_argument("--embeddings", type=Path, default=None,
                    help="enables the stability + separation tests")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-images", type=int, default=500,
                    help="images sampled for the spatial-coherence test")
    ap.add_argument("--n-perm", type=int, default=3, help="permutations per image")
    ap.add_argument("--stability-cells", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = args.out_dir or args.motifs.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    mdf = pd.read_parquet(args.motifs)
    n_motifs = int(mdf["motif"].max()) + 1
    print(f"Validating {n_motifs} motifs over {len(mdf)} cells")

    report = {"n_motifs": n_motifs, "n_cells": int(len(mdf))}
    report["spatial_coherence"] = spatial_coherence(
        mdf, args.graphs_dir, args.max_images, args.n_perm, args.seed, n_motifs)
    report["recurrence"] = recurrence(mdf, out_dir, n_motifs)
    if args.embeddings:
        report.update(stability(args.embeddings, mdf, n_motifs,
                                args.stability_cells, args.seed))

    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out_dir / 'validation_report.json'}")

    sc = report.get("spatial_coherence", {})
    print("\n" + "=" * 66)
    print("BOTTOM LINE")
    if sc:
        print(f"  spatial coherence : {sc['ratio']:.2f}x  ({sc['verdict']})")
    print(f"  recurring motifs  : {report['recurrence']['n_widespread']}/{n_motifs}")
    if "mean_ari" in report:
        print(f"  stability (ARI)   : {report['mean_ari']:.3f}  "
              f"({report['stability_verdict']})")
        print(f"  separation        : silhouette {report['silhouette']:.3f}  "
              f"({report['separation']})")
    print("=" * 66)


if __name__ == "__main__":
    main()
