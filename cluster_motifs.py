#!/usr/bin/env python3
"""Stage 4 — cluster neighborhood embeddings into MOTIFS.

Implements Step 6 of the project doc: run the trained encoder's embeddings
through dimensionality reduction + graph clustering so that each cluster of
similar neighborhoods becomes one recurring spatial motif, and every cell gets
a motif label.

Pipeline:
  1. load embeddings (one 64-d vector per cell, from embed_cells.py)
  2. optionally UMAP-reduce 64 -> N dims (more stable clustering; doc suggests ~10)
  3. fit clustering on a subsample (Leiden on a kNN graph; k-means fallback)
  4. assign EVERY cell to the nearest cluster centroid
  5. characterise each motif and check it is not a batch artifact

Outputs:
    <out>/motifs.parquet          condition, image_id, cell_id, motif
    <out>/motif_summary.csv       per-motif size + mean features (what the motif IS)
    <out>/motif_by_condition.csv  motif x condition counts + row proportions
    <out>/motif_by_date.csv       motif x imaging date  (BATCH ARTIFACT CHECK)
    <out>/motif_report.json       parameters + batch diagnostics
    <out>/umap_by_motif.png       embedding coloured by motif

Usage:
    python cluster_motifs.py --embeddings <path> --out-dir <path>
    python cluster_motifs.py --method kmeans --n-clusters 8
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Per-cell columns pulled from the master table to describe each motif.
FEATURE_COLS = [
    "endmt_score",
    "area_px",
    "elongation",
    "ch1_cellwise_mean_intensity",
    "ch2_cellwise_mean_membrane_intensity",
    "ch3_cellwise_mean_intensity",
]


def _default_local_root() -> Path:
    default = Path("/teamspace/studios/this_studio/cellpose_work")
    if "CELLPOSE_LOCAL_ROOT" in os.environ:
        return Path(os.environ["CELLPOSE_LOCAL_ROOT"]).expanduser()
    if default.parent.parent.exists():
        return default
    return Path.home() / "cellpose_work"


def load_embeddings(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    csv = path.with_suffix(".csv")
    if csv.exists():
        return pd.read_csv(csv)
    raise SystemExit(f"embeddings not found: {path} (or .csv). Run embed_cells.py first.")


def parse_date(condition: str) -> str:
    """'260516_500kPa' -> '260516' (the imaging batch)."""
    return str(condition).partition("_")[0]


def cluster_leiden(X: np.ndarray, n_neighbors: int, resolution: float, seed: int = 0):
    """Leiden community detection on a kNN similarity graph. None if unavailable."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError:
        return None
    from sklearn.neighbors import kneighbors_graph

    print(f"  building kNN graph (k={n_neighbors}) over {len(X)} points...")
    A = kneighbors_graph(X, n_neighbors=n_neighbors, mode="connectivity",
                         include_self=False, n_jobs=-1)
    sources, targets = A.nonzero()
    g = ig.Graph(n=X.shape[0], edges=list(zip(sources.tolist(), targets.tolist())),
                 directed=False)
    g.simplify()
    print(f"  running Leiden (resolution={resolution})...")
    part = leidenalg.find_partition(
        g, leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution, seed=seed,
    )
    return np.asarray(part.membership)


def cluster_kmeans(X: np.ndarray, n_clusters: int, seed: int = 0):
    from sklearn.cluster import KMeans
    print(f"  running k-means (k={n_clusters})...")
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(X)


def main() -> None:
    root = _default_local_root()
    ap = argparse.ArgumentParser(description="Cluster embeddings into motifs (Stage 4).")
    ap.add_argument("--embeddings", type=Path, default=root / "stage3" / "embeddings.parquet")
    ap.add_argument("--master-table", type=Path, default=root / "master_table.csv")
    ap.add_argument("--out-dir", type=Path, default=root / "stage4")
    ap.add_argument("--method", choices=["leiden", "kmeans"], default="leiden",
                    help="leiden needs python-igraph + leidenalg; falls back to kmeans")
    ap.add_argument("--resolution", type=float, default=1.0,
                    help="Leiden resolution: higher = more motifs")
    ap.add_argument("--n-clusters", type=int, default=8, help="k for k-means")
    ap.add_argument("--n-neighbors", type=int, default=15,
                    help="k for the similarity graph (doc suggests 15)")
    ap.add_argument("--umap-dims", type=int, default=10,
                    help="reduce embeddings to this many dims before clustering; "
                         "0 = cluster the raw 64-d embeddings")
    ap.add_argument("--fit-cells", type=int, default=200000,
                    help="cells used to FIT the clustering (all cells are then labelled)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = load_embeddings(args.embeddings)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    X_all = df[emb_cols].to_numpy(dtype=np.float32)
    print(f"Loaded {len(df)} cell embeddings ({len(emb_cols)} dims)")

    # ---- 1. optional UMAP reduction (stabilises clustering; doc step 6.3) ----
    reducer = None
    if args.umap_dims and args.umap_dims > 0:
        try:
            import umap
            print(f"UMAP-reducing {len(emb_cols)} -> {args.umap_dims} dims "
                  f"(fit on <= {args.fit_cells} cells)...")
            fit_idx = np.random.default_rng(args.seed).choice(
                len(X_all), size=min(args.fit_cells, len(X_all)), replace=False)
            reducer = umap.UMAP(n_components=args.umap_dims, n_neighbors=args.n_neighbors,
                                min_dist=0.0, random_state=args.seed)
            reducer.fit(X_all[fit_idx])
            Z_all = reducer.transform(X_all)
        except ImportError:
            print("  [warn] umap-learn not installed; clustering raw embeddings instead")
            Z_all = X_all
    else:
        Z_all = X_all

    # ---- 2. fit clustering on a subsample ----
    rng = np.random.default_rng(args.seed)
    fit_idx = rng.choice(len(Z_all), size=min(args.fit_cells, len(Z_all)), replace=False)
    Z_fit = Z_all[fit_idx]
    print(f"Clustering ({args.method}) on {len(Z_fit)} cells...")

    labels_fit = None
    if args.method == "leiden":
        labels_fit = cluster_leiden(Z_fit, args.n_neighbors, args.resolution, args.seed)
        if labels_fit is None:
            print("  [warn] leidenalg/igraph not installed "
                  "(pip install leidenalg python-igraph); falling back to k-means")
    if labels_fit is None:
        labels_fit = cluster_kmeans(Z_fit, args.n_clusters, args.seed)
        method_used = "kmeans"
    else:
        method_used = "leiden"

    # drop tiny clusters (<0.1% of the fit sample) — usually Leiden noise
    uniq, counts = np.unique(labels_fit, return_counts=True)
    keep = uniq[counts >= max(10, int(0.001 * len(labels_fit)))]
    print(f"  found {len(uniq)} clusters, keeping {len(keep)} above the size floor")

    # ---- 3. label EVERY cell by nearest centroid ----
    centroids = np.stack([Z_fit[labels_fit == c].mean(axis=0) for c in keep])
    print(f"Assigning all {len(Z_all)} cells to {len(centroids)} motif centroids...")
    motif = np.empty(len(Z_all), dtype=np.int32)
    for i in range(0, len(Z_all), 100_000):                    # chunked to bound memory
        chunk = Z_all[i:i + 100_000]
        d = ((chunk[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        motif[i:i + 100_000] = d.argmin(1)

    out = df[["condition", "image_id", "cell_id"]].copy()
    out["motif"] = motif
    out.to_parquet(args.out_dir / "motifs.parquet", index=False)
    print(f"Wrote {args.out_dir / 'motifs.parquet'}")

    # ---- 4. characterise each motif using the per-cell features ----
    master = pd.read_csv(args.master_table)
    have = [c for c in FEATURE_COLS if c in master.columns]
    merged = out.merge(master[["condition", "image_id", "cell_id"] + have],
                       on=["condition", "image_id", "cell_id"], how="left")
    summary = merged.groupby("motif")[have].mean()
    summary.insert(0, "n_cells", merged.groupby("motif").size())
    summary.insert(1, "pct_of_cells", 100 * summary["n_cells"] / len(merged))
    summary.to_csv(args.out_dir / "motif_summary.csv")
    print("\nMotif summary (mean feature values — what each motif IS):")
    print(summary.round(3).to_string())

    # ---- 5. motif x condition (the scientific readout) ----
    ct = pd.crosstab(merged["motif"], merged["condition"])
    prop = ct.div(ct.sum(axis=1), axis=0)                      # row-normalised
    ct.join(prop, rsuffix="_prop").to_csv(args.out_dir / "motif_by_condition.csv")
    print("\nMotif x condition (row proportions — is a motif enriched anywhere?):")
    print(prop.round(3).to_string())

    # ---- 6. BATCH ARTIFACT CHECK: motif x imaging date ----
    # A motif dominated by one date is a technical artifact, not biology.
    merged["date"] = merged["condition"].map(parse_date)
    ct_d = pd.crosstab(merged["motif"], merged["date"])
    prop_d = ct_d.div(ct_d.sum(axis=1), axis=0)
    ct_d.join(prop_d, rsuffix="_prop").to_csv(args.out_dir / "motif_by_date.csv")
    overall_d = merged["date"].value_counts(normalize=True)
    print("\nMotif x DATE (batch check — a motif concentrated on one date is suspect):")
    print(prop_d.round(3).to_string())
    print(f"  dataset-wide date mix: {overall_d.round(3).to_dict()}")

    # enrichment = motif's share of a date / that date's overall share
    enrich = (prop_d / overall_d).max(axis=1)
    max_date_frac = prop_d.max(axis=1)
    flagged = [int(m) for m in prop_d.index
               if max_date_frac[m] >= 0.60 and enrich[m] >= 1.5]
    print("\nPer-motif batch risk (max single-date share, enrichment vs dataset):")
    for m in prop_d.index:
        flag = "  <-- BATCH-SUSPECT" if int(m) in flagged else ""
        print(f"  motif {int(m):>2}: max_date_share={max_date_frac[m]:.3f} "
              f"enrichment={enrich[m]:.2f}x{flag}")
    if flagged:
        print(f"  => {len(flagged)} of {len(prop_d)} motifs look date-driven: {flagged}")
    else:
        print("  => no motif is date-dominated; motifs span imaging batches")

    report = {
        "method": method_used,
        "n_motifs": int(len(centroids)),
        "n_cells_labelled": int(len(out)),
        "params": {"resolution": args.resolution, "n_clusters": args.n_clusters,
                   "n_neighbors": args.n_neighbors, "umap_dims": args.umap_dims,
                   "fit_cells": int(len(Z_fit)), "seed": args.seed},
        "motif_sizes": {int(k): int(v) for k, v in summary["n_cells"].items()},
        "batch_check": {
            "max_date_share": {int(k): float(v) for k, v in max_date_frac.items()},
            "date_enrichment": {int(k): float(v) for k, v in enrich.items()},
            "batch_suspect_motifs": flagged,
        },
    }
    (args.out_dir / "motif_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.out_dir / 'motif_report.json'}")

    # ---- 7. UMAP coloured by motif ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import umap as umap_lib

        n_plot = min(50000, len(X_all))
        pidx = rng.choice(len(X_all), size=n_plot, replace=False)
        print(f"Rendering UMAP of {n_plot} cells coloured by motif...")
        XY = umap_lib.UMAP(n_neighbors=15, min_dist=0.1,
                           random_state=args.seed).fit_transform(X_all[pidx])
        fig, ax = plt.subplots(figsize=(9, 8))
        for m in sorted(np.unique(motif[pidx])):
            sel = motif[pidx] == m
            ax.scatter(XY[sel, 0], XY[sel, 1], s=2, alpha=0.5, label=f"motif {m}")
        ax.legend(markerscale=6, fontsize=8, loc="best", ncol=2)
        ax.set_title(f"Cell embeddings coloured by MOTIF ({method_used}, "
                     f"{len(centroids)} motifs)")
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(args.out_dir / "umap_by_motif.png", dpi=130)
        plt.close(fig)
        print(f"Wrote {args.out_dir / 'umap_by_motif.png'}")
    except ImportError:
        print("[skip] umap-learn/matplotlib missing; no motif UMAP")


if __name__ == "__main__":
    main()
