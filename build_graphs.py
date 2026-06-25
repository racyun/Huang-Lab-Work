#!/usr/bin/env python3
"""Stage 2 — spatial graph construction (kNN-union on centroids).

Implements docs/stage2_graph_construction.md §2–§8. One PyG ``Data`` graph per
image: nodes are cells (carrying the master table's normalized feature vector),
edges are physical adjacency from k-NN on centroids only, symmetrized by union
and pruned by an adaptive per-image ``d_max``.

Principle (doc §0): features ride on nodes, geometry defines edges — k-NN runs
on centroids, never on features. Raw (x, y) is NOT a node feature.

Inputs:
    <root>/master_table.csv     (from assemble_master_table.py)

Outputs (doc §8):
    <root>/graphs/<condition>/<image_id>.pt   one PyG Data per image
    <root>/graphs/feature_names.json          node-feature column order
    <root>/graph_qc.csv                        one QC row per image
    <root>/graph_overlays/*.png               a few eyeball-check overlays

Usage:
    python build_graphs.py                 # defaults, k=8
    python build_graphs.py --k 8 --d-max-mult 3.0 --overlays-per-condition 5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from torch_geometric.data import Data
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "build_graphs.py needs torch + torch_geometric:\n"
        "  pip install torch torch_geometric\n"
        f"(import error: {e})"
    )

from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# Node-feature columns, in order. z-scored cols (from the master table) +
# raw endmt_score + the on_border flag appended per image. Centroids and
# identity columns are deliberately excluded (doc §0, §1).
ZSCORE_COLS = [
    "area_px",
    "elongation",
    "ch1_cellwise_mean_intensity",
    "ch2_cellwise_mean_membrane_intensity",
    "ch3_cellwise_mean_intensity",
]
RAW_COLS = ["endmt_score"]
NODE_FEATURE_NAMES = ZSCORE_COLS + RAW_COLS + ["on_border"]


def _default_local_root() -> Path:
    default = Path("/teamspace/studios/this_studio/cellpose_work")
    if "CELLPOSE_LOCAL_ROOT" in os.environ:
        return Path(os.environ["CELLPOSE_LOCAL_ROOT"]).expanduser()
    if default.parent.parent.exists():
        return default
    return Path.home() / "cellpose_work"


def build_graph(df_image: pd.DataFrame, k: int, d_max_mult: float,
                image_w: float | None = None, image_h: float | None = None):
    """Build one PyG Data from a single image's rows. Returns (data, qc_dict).

    df_image must contain centroid_x/centroid_y, the ZSCORE_COLS, endmt_score,
    image_id and condition.

    image_w/image_h: true pixel dimensions of the field of view, used for the
    on_border flag (a cell is on_border if its centroid is within d_max of pixel
    0 or of the image width/height). If None, fall back to the bounding box of
    the cell centroids (less accurate — see docs §7).
    """
    image_id = str(df_image["image_id"].iloc[0])
    condition = str(df_image["condition"].iloc[0])
    n = len(df_image)

    pos_np = df_image[["centroid_x", "centroid_y"]].to_numpy(dtype=float)

    qc = {
        "image_id": image_id, "condition": condition,
        "n_nodes": n, "n_edges": 0, "mean_degree": 0.0, "median_degree": 0.0,
        "n_components": 0, "median_edge_len": float("nan"),
        "d_max": float("nan"), "n_on_border": 0, "status": "ok",
    }

    # ----- tiny-image handling (doc §7): degrade k gracefully; skip if <2 -----
    if n < 2:
        qc["status"] = "skipped_too_few_cells"
        kq = 0
    else:
        kq = min(k, n - 1)
        if kq < k:
            qc["status"] = "reduced_k"

    # ----- k-NN on centroids (cKDTree); also record directed neighbor sets so
    #       is_mutual can be computed BEFORE the union (doc §4 caveat) -----
    edges: dict[tuple[int, int], bool] = {}   # (i<j) -> is_mutual
    d_max = float("nan")
    if kq >= 1:
        tree = cKDTree(pos_np)
        dist_knn, idx_knn = tree.query(pos_np, k=kq + 1)  # col 0 is self
        if dist_knn.ndim == 1:  # k+1 == 1 edge case guard
            dist_knn = dist_knn[:, None]
            idx_knn = idx_knn[:, None]
        nbr_dist = dist_knn[:, 1]                          # nearest non-self
        median_nn = float(np.median(nbr_dist))
        d_max = d_max_mult * median_nn
        qc["d_max"] = d_max

        nbr_sets = [set(idx_knn[i, 1:]) for i in range(n)]
        for i in range(n):
            for j in nbr_sets[i]:
                a, b = (i, int(j)) if i < j else (int(j), i)
                if a == b:
                    continue
                mutual = (j in nbr_sets[i]) and (i in nbr_sets[int(j)])
                # OR over directions: an edge stays mutual if either pass flags it
                edges[(a, b)] = edges.get((a, b), False) or mutual

    # ----- distance prune + assemble undirected edge list -----
    src, dst, e_dist, e_mutual = [], [], [], []
    for (a, b), mutual in edges.items():
        d = float(np.linalg.norm(pos_np[a] - pos_np[b]))
        if np.isfinite(d_max) and d > d_max:
            continue
        # store both directions (undirected graph for message passing)
        src += [a, b]; dst += [b, a]
        e_dist += [d, d]; e_mutual += [1.0 if mutual else 0.0] * 2

    n_undirected = len(e_dist) // 2
    qc["n_edges"] = n_undirected

    # ----- on_border node feature (doc §7): centroid within d_max of a field-
    #       of-view edge. Use true image dims (image_w/image_h) when given;
    #       else fall back to the centroid bounding box (less accurate) -----
    on_border = np.zeros(n, dtype=float)
    if n >= 1 and np.isfinite(d_max):
        if image_w is not None and image_h is not None:
            x0, y0, x1, y1 = 0.0, 0.0, float(image_w), float(image_h)
        else:
            x0, y0 = pos_np.min(axis=0)
            x1, y1 = pos_np.max(axis=0)
        near = (
            (pos_np[:, 0] - x0 < d_max) | (x1 - pos_np[:, 0] < d_max) |
            (pos_np[:, 1] - y0 < d_max) | (y1 - pos_np[:, 1] < d_max)
        )
        on_border = near.astype(float)
    qc["n_on_border"] = int(on_border.sum())

    # ----- node feature matrix x (doc §0 ordering) -----
    feats = [df_image[c].to_numpy(dtype=float) for c in ZSCORE_COLS + RAW_COLS]
    feats.append(on_border)
    x = torch.tensor(np.stack(feats, axis=1), dtype=torch.float)
    pos = torch.tensor(pos_np, dtype=torch.float)

    if n_undirected > 0:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        dist_t = torch.tensor(e_dist, dtype=torch.float)
        dist_norm = dist_t / d_max
        inv_dist = 1.0 / (dist_t + 1e-6)
        is_mutual = torch.tensor(e_mutual, dtype=torch.float)
        edge_attr = torch.stack([dist_t, dist_norm, inv_dist, is_mutual], dim=1)
        qc["median_edge_len"] = float(np.median(e_dist[::2]))  # one per undirected edge
        # degree from undirected edges
        deg = np.bincount(np.array(src), minlength=n)
        qc["mean_degree"] = float(deg.mean())
        qc["median_degree"] = float(np.median(deg))
        # connected components on the undirected adjacency
        adj = coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n))
        qc["n_components"] = int(connected_components(adj, directed=False)[0])
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 4), dtype=torch.float)
        qc["n_components"] = n  # every node isolated

    data = Data(
        x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr,
        image_id=image_id, condition=condition,
    )
    return data, qc


def save_overlay(data, out_png: Path) -> None:
    """Plot the graph (centroids + edges) for the eyeball check (doc §6)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = data.pos.numpy()
    fig, ax = plt.subplots(figsize=(7, 7))
    ei = data.edge_index.numpy()
    for s, d in ei.T:
        ax.plot([pos[s, 0], pos[d, 0]], [pos[s, 1], pos[d, 1]],
                lw=0.3, color="tab:blue", alpha=0.5)
    ax.scatter(pos[:, 0], pos[:, 1], s=6, color="black", zorder=3)
    ax.set_title(f"{data.condition} / {data.image_id}  "
                 f"(N={data.num_nodes}, E={data.edge_index.shape[1] // 2})")
    ax.set_aspect("equal")
    ax.invert_yaxis()  # image coordinates
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main() -> None:
    root = _default_local_root()
    ap = argparse.ArgumentParser(description="Build per-image spatial graphs (Stage 2).")
    ap.add_argument("--master-table", type=Path, default=root / "master_table.csv")
    ap.add_argument("--out-dir", type=Path, default=root,
                    help="root for graphs/, graph_qc.csv, graph_overlays/")
    ap.add_argument("--k", type=int, default=8, help="k for k-NN (default: 8)")
    ap.add_argument("--d-max-mult", type=float, default=3.0,
                    help="d_max = mult x median NN distance per image (default: 3.0)")
    ap.add_argument("--image-size", type=int, default=682,
                    help="square image side in px for on_border; 0 = use centroid "
                         "bounding box instead (default: 682)")
    ap.add_argument("--overlays-per-condition", type=int, default=5)
    args = ap.parse_args()

    if not args.master_table.exists():
        raise SystemExit(f"master table not found: {args.master_table}\n"
                         "Run assemble_master_table.py first.")

    master = pd.read_csv(args.master_table)
    missing = [c for c in ZSCORE_COLS + RAW_COLS + ["centroid_x", "centroid_y",
               "image_id", "condition"] if c not in master.columns]
    if missing:
        raise SystemExit(f"master table missing columns: {missing}")

    graphs_dir = args.out_dir / "graphs"
    overlays_dir = args.out_dir / "graph_overlays"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    with open(graphs_dir / "feature_names.json", "w") as f:
        json.dump(NODE_FEATURE_NAMES, f, indent=2)

    qc_rows = []
    overlay_counts: dict[str, int] = {}
    n_graphs = 0
    img_dim = float(args.image_size) if args.image_size and args.image_size > 0 else None

    # group by image; keep condition order stable
    for (condition, image_id), df_img in master.groupby(["condition", "image_id"], sort=True):
        data, qc = build_graph(df_img, k=args.k, d_max_mult=args.d_max_mult,
                               image_w=img_dim, image_h=img_dim)
        cond_dir = graphs_dir / condition
        cond_dir.mkdir(parents=True, exist_ok=True)
        torch.save(data, cond_dir / f"{image_id}.pt")
        qc_rows.append(qc)
        n_graphs += 1

        c = overlay_counts.get(condition, 0)
        if c < args.overlays_per_condition and data.edge_index.shape[1] > 0:
            try:
                save_overlay(data, overlays_dir / f"{condition}__{image_id}.png")
                overlay_counts[condition] = c + 1
            except Exception as e:  # overlays are optional — never abort the build
                if n_graphs == 1:  # warn once
                    print(f"  [warn] overlay generation skipped ({e}). "
                          f"Install matplotlib to enable overlays.")

    qc_df = pd.DataFrame(qc_rows)
    qc_path = args.out_dir / "graph_qc.csv"
    qc_df.to_csv(qc_path, index=False)

    print(f"Built {n_graphs} graph(s) -> {graphs_dir}")
    print(f"Wrote QC -> {qc_path}")
    print(f"Wrote {sum(overlay_counts.values())} overlay(s) -> {overlays_dir}")
    print("\nPer-condition summary:")
    summ = qc_df.groupby("condition").agg(
        images=("image_id", "count"),
        cells=("n_nodes", "sum"),
        mean_deg=("mean_degree", "mean"),
        med_edge_len=("median_edge_len", "median"),
    )
    print(summ.to_string())


if __name__ == "__main__":
    main()
