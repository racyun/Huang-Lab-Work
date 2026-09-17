#!/usr/bin/env python3
"""Baseline: cluster CELL BY CELL with no neighbourhood information.

Answers the question "does the GNN do anything a trivial method couldn't?"
Takes the same 6 node features the encoder saw (5 z-scored + endmt_score),
runs k-means on them directly, and writes outputs in exactly the format
cluster_motifs.py / embed_cells.py produce — so validate_motifs.py and
robustness_motifs.py run on it unchanged and the numbers are directly
comparable to the GNN run.

If this baseline matches the GNN on coherence / stability, the encoder is not
earning its complexity. If the GNN is clearly better, spatial context matters.

Outputs (in --out-dir):
    motifs.parquet        condition, image_id, cell_id, motif
    embeddings.parquet    condition, image_id, cell_id, emb_0..emb_5  (= raw features)
    motif_summary.csv     per-motif feature profile, same columns as the GNN run

Usage:
    python baseline_no_neighbors.py --master-table <root>/master_table.csv \
        --out-dir <root>/baseline_cells --n-clusters 6
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "area_px", "elongation",
    "ch1_cellwise_mean_intensity",
    "ch2_cellwise_mean_membrane_intensity",
    "ch3_cellwise_mean_intensity",
    "endmt_score",
]


def _default_local_root() -> Path:
    default = Path("/teamspace/studios/this_studio/cellpose_work")
    if "CELLPOSE_LOCAL_ROOT" in os.environ:
        return Path(os.environ["CELLPOSE_LOCAL_ROOT"]).expanduser()
    if default.parent.parent.exists():
        return default
    return Path.home() / "cellpose_work"


def main() -> None:
    from sklearn.cluster import KMeans
    root = _default_local_root()
    ap = argparse.ArgumentParser(description="No-neighbour clustering baseline.")
    ap.add_argument("--master-table", type=Path, default=root / "master_table.csv")
    ap.add_argument("--out-dir", type=Path, default=root / "baseline_cells")
    ap.add_argument("--n-clusters", type=int, default=6)
    ap.add_argument("--fit-cells", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.master_table)
    X = df[FEATURES].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
    print(f"Loaded {len(df):,} cells x {len(FEATURES)} raw features (no neighbours)")

    rng = np.random.default_rng(args.seed)
    fit_idx = rng.choice(len(X), size=min(args.fit_cells, len(X)), replace=False)
    km = KMeans(n_clusters=args.n_clusters, n_init=10, random_state=args.seed)
    km.fit(X[fit_idx])
    motif = np.empty(len(X), dtype=np.int32)
    for i in range(0, len(X), 100_000):
        motif[i:i + 100_000] = km.predict(X[i:i + 100_000])
    print(f"k-means k={args.n_clusters} fit on {len(fit_idx):,} cells, all assigned")

    key = df[["condition", "image_id", "cell_id"]].copy()
    out = key.copy(); out["motif"] = motif
    out.to_parquet(args.out_dir / "motifs.parquet", index=False)

    emb = key.copy()
    for j in range(X.shape[1]):
        emb[f"emb_{j}"] = X[:, j]
    emb.to_parquet(args.out_dir / "embeddings.parquet", index=False)

    summ = pd.DataFrame(X, columns=FEATURES)
    summ["motif"] = motif
    prof = summ.groupby("motif").mean()
    prof.insert(0, "n_cells", summ.groupby("motif").size())
    prof.insert(1, "pct_of_cells", 100 * prof["n_cells"] / len(summ))
    prof.to_csv(args.out_dir / "motif_summary.csv")
    print("\nPer-motif profile (raw-feature clustering):")
    print(prof.round(3).to_string())
    print(f"\nWrote {args.out_dir}/motifs.parquet, embeddings.parquet, motif_summary.csv")
    print("Now run validate_motifs.py and robustness_motifs.py on these paths.")


if __name__ == "__main__":
    main()
