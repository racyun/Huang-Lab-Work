#!/usr/bin/env python3
"""Stage 3 — embed every cell with the frozen encoder.

Runs the trained NeighborhoodEncoder over each cell's 2-hop neighborhood and
writes one 64-d embedding per cell — the input to Stage 4 clustering
(docs/stage3_encoder_methodology.md §0, §11; stage3_gnn_encoder.md §8).

Uses the encoder embedding ``h`` (NOT the projection ``z``): clustering operates
on ``h`` (methodology §5). No augmentation here — this is deterministic
inference over the real neighborhoods.

Inputs:
    <root>/graphs/<condition>/<image_id>.pt   (Stage 2 graphs)
    <root>/stage3/encoder.pt                  (from train_encoder.py)

Outputs:
    <root>/stage3/embeddings.parquet          one row per cell:
        condition, image_id, cell_id, emb_0 ... emb_63

Usage:
    python embed_cells.py
    python embed_cells.py --encoder <path> --batch-size 512 --push-to-drive
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import k_hop_subgraph

from models.neighborhood_encoder import NeighborhoodEncoder

DRIVE_ROOT = "Fusion AI/Prof Huang Project/Cellpose feature extractions"


def _default_local_root() -> Path:
    default = Path("/teamspace/studios/this_studio/cellpose_work")
    if "CELLPOSE_LOCAL_ROOT" in os.environ:
        return Path(os.environ["CELLPOSE_LOCAL_ROOT"]).expanduser()
    if default.parent.parent.exists():
        return default
    return Path.home() / "cellpose_work"


def extract_neighborhood(data: Data, center: int, hops: int = 2) -> Data:
    subset, edge_index, _, edge_mask = k_hop_subgraph(
        center, hops, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes,
    )
    return Data(
        x=data.x[subset],
        edge_index=edge_index,
        edge_attr=data.edge_attr[edge_mask] if getattr(data, "edge_attr", None) is not None else None,
    )


def load_encoder(ckpt_path: Path, device) -> NeighborhoodEncoder:
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = NeighborhoodEncoder(
        in_dim=cfg["in_dim"], hidden_dim=cfg["hidden_dim"], emb_dim=cfg["emb_dim"],
        heads=cfg["heads"], edge_dim=cfg["edge_dim"], dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded encoder (epoch {ckpt.get('epoch')}, loss {ckpt.get('loss'):.4f}); "
          f"emb_dim={cfg['emb_dim']}")
    return model, cfg["emb_dim"]


def cellids_for_graph(master: pd.DataFrame, condition: str, image_id: str, n: int):
    """cell_id per node (node order == per-image row order); fallback to 0..n-1."""
    rows = master[(master.image_id == image_id) & (master.condition == condition)]
    if len(rows) == n:
        return rows["cell_id"].to_numpy()
    return np.arange(n)


@torch.no_grad()
def main() -> None:
    root = _default_local_root()
    ap = argparse.ArgumentParser(description="Embed every cell (Stage 3 inference).")
    ap.add_argument("--graphs-dir", type=Path, default=root / "graphs")
    ap.add_argument("--encoder", type=Path, default=root / "stage3" / "encoder.pt")
    ap.add_argument("--master-table", type=Path, default=root / "master_table.csv")
    ap.add_argument("--out", type=Path, default=root / "stage3" / "embeddings.parquet")
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=512,
                    help="neighborhoods encoded per forward pass")
    ap.add_argument("--push-to-drive", action="store_true")
    args = ap.parse_args()

    if not args.encoder.exists():
        raise SystemExit(f"encoder not found: {args.encoder}\nRun train_encoder.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model, emb_dim = load_encoder(args.encoder, device)

    master = pd.read_csv(args.master_table, usecols=["image_id", "condition", "cell_id"])
    graph_paths = sorted(glob.glob(str(args.graphs_dir / "*" / "*.pt")))
    if not graph_paths:
        raise SystemExit(f"no graphs under {args.graphs_dir}")
    print(f"Embedding cells from {len(graph_paths)} image graphs...")

    rows_meta = []          # (condition, image_id, cell_id) per cell
    embs = []               # embedding rows
    pending: list[Data] = []
    pending_meta: list[tuple] = []

    def flush():
        if not pending:
            return
        batch = Batch.from_data_list(pending).to(device)
        h, _ = model(batch)                       # use h (not z)
        embs.append(h.cpu().numpy())
        rows_meta.extend(pending_meta)
        pending.clear(); pending_meta.clear()

    n_cells = 0
    for gi, p in enumerate(graph_paths):
        g = torch.load(p, weights_only=False)
        cond, img = g.condition, g.image_id
        cids = cellids_for_graph(master, cond, img, g.num_nodes)
        for c in range(g.num_nodes):
            pending.append(extract_neighborhood(g, c, args.hops))
            pending_meta.append((cond, img, int(cids[c])))
            n_cells += 1
            if len(pending) >= args.batch_size:
                flush()
        if (gi + 1) % 500 == 0:
            print(f"  {gi + 1}/{len(graph_paths)} graphs, {n_cells} cells")
    flush()

    E = np.concatenate(embs, axis=0)
    meta = pd.DataFrame(rows_meta, columns=["condition", "image_id", "cell_id"])
    emb_cols = pd.DataFrame(E, columns=[f"emb_{i}" for i in range(emb_dim)])
    out_df = pd.concat([meta.reset_index(drop=True), emb_cols], axis=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_df.to_parquet(args.out, index=False)
        out_path = args.out
    except Exception as e:
        out_path = args.out.with_suffix(".csv")
        print(f"  [warn] parquet failed ({e}); writing CSV instead -> {out_path.name}")
        out_df.to_csv(out_path, index=False)

    print(f"\nEmbedded {len(out_df)} cells -> {out_path}  (dim={emb_dim})")
    if args.push_to_drive:
        dest = f"gdrive:{DRIVE_ROOT}/stage3"
        print(f"  $ rclone copy {out_path} {dest}")
        r = subprocess.run(["rclone", "copyto", str(out_path), f"{dest}/{out_path.name}"])
        if r.returncode != 0:
            raise RuntimeError(f"rclone failed (exit {r.returncode})")


if __name__ == "__main__":
    main()
