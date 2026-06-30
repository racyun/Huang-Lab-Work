#!/usr/bin/env python3
"""Stage 3 — train the neighborhood GNN encoder (self-supervised, contrastive).

Follows docs/stage3_encoder_methodology.md:
  - subgraphs are sampled ON-THE-FLY (2-hop ego graphs) from the Stage 2 image
    graphs — no need for the materialized 1.27M (§2, §10);
  - each subgraph gets TWO mild augmented views (edge drop / feature mask /
    outer-node drop / feature noise — §8);
  - both views are encoded by the GATv2 encoder + projection head (§3, §5) and
    trained with the NT-Xent contrastive loss (§7);
  - AdamW + cosine LR, checkpointing, optional Drive push (§10).

The adversarial scrubbing head is intentionally NOT included — the methodology
(§9) says train contrastive-only first, then add it only if the confound probe
shows leakage.

Inputs:
    <root>/graphs/<condition>/<image_id>.pt   (from build_graphs.py)

Outputs:
    <out>/stage3/encoder.pt        trained encoder weights + config
    <out>/stage3/train_log.jsonl   per-epoch loss
    <out>/stage3/config.json       hyperparameters used

Usage:
    python train_encoder.py --epochs 100 --batch-size 256
    python train_encoder.py --epochs 100 --push-to-drive
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph

from models.neighborhood_encoder import NeighborhoodEncoder, nt_xent_loss

DRIVE_ROOT = "Fusion AI/Prof Huang Project/Cellpose feature extractions"


def _default_local_root() -> Path:
    default = Path("/teamspace/studios/this_studio/cellpose_work")
    if "CELLPOSE_LOCAL_ROOT" in os.environ:
        return Path(os.environ["CELLPOSE_LOCAL_ROOT"]).expanduser()
    if default.parent.parent.exists():
        return default
    return Path.home() / "cellpose_work"


# --------------------------------------------------------------------------- #
# On-the-fly 2-hop subgraph extraction (mirrors sample_subgraphs.extract_neighborhood)
# --------------------------------------------------------------------------- #

def extract_neighborhood(data: Data, center: int, hops: int = 2) -> Data:
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        center, hops, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes,
    )
    return Data(
        x=data.x[subset],
        edge_index=edge_index,
        edge_attr=data.edge_attr[edge_mask] if getattr(data, "edge_attr", None) is not None else None,
        center=mapping,
    )


# --------------------------------------------------------------------------- #
# Augmentations (methodology §8) — mild, identity-preserving; never drop center
# --------------------------------------------------------------------------- #

def augment(sub: Data, edge_drop=0.2, feat_mask=0.1, node_drop=0.1, feat_noise=0.05) -> Data:
    x = sub.x.clone()
    edge_index = sub.edge_index
    edge_attr = sub.edge_attr
    center = int(sub.center)
    n = x.shape[0]

    # ----- feature masking: zero a random subset of feature DIMENSIONS -----
    if feat_mask > 0 and x.shape[1] > 0:
        d_mask = torch.rand(x.shape[1], device=x.device) < feat_mask
        x[:, d_mask] = 0.0

    # ----- feature noise on (already z-scored) features -----
    if feat_noise > 0:
        x = x + feat_noise * torch.randn_like(x)

    # ----- outer-node dropping: drop a few non-center nodes -----
    keep = torch.ones(n, dtype=torch.bool)
    if node_drop > 0 and n > 2:
        drop = (torch.rand(n) < node_drop)
        drop[center] = False                          # never drop the center
        keep = ~drop
    if keep.all():
        x_k, edge_index_k, edge_attr_k, center_k = x, edge_index, edge_attr, center
    else:
        # relabel surviving nodes 0..k-1 and filter edges to survivors
        new_id = -torch.ones(n, dtype=torch.long)
        new_id[keep] = torch.arange(int(keep.sum()))
        x_k = x[keep]
        em = keep[edge_index[0]] & keep[edge_index[1]]
        edge_index_k = new_id[edge_index[:, em]]
        edge_attr_k = edge_attr[em] if edge_attr is not None else None
        center_k = int(new_id[center])

    # ----- edge dropping -----
    if edge_drop > 0 and edge_index_k.shape[1] > 0:
        em = torch.rand(edge_index_k.shape[1]) >= edge_drop
        edge_index_k = edge_index_k[:, em]
        edge_attr_k = edge_attr_k[em] if edge_attr_k is not None else None

    return Data(x=x_k, edge_index=edge_index_k, edge_attr=edge_attr_k,
                center=torch.tensor(center_k))


# --------------------------------------------------------------------------- #
# Dataset: yields TWO augmented views of a random cell's 2-hop neighborhood
# --------------------------------------------------------------------------- #

class ContrastiveNeighborhoods(torch.utils.data.Dataset):
    """Each __getitem__ returns (view_a, view_b) for one (graph, center) pair.

    Centers are precomputed (cells with >=1 edge). `samples_per_epoch` caps how
    many neighborhoods are drawn per epoch (a subset of the full ~1.27M).
    """

    def __init__(self, graph_paths, hops=2, samples_per_epoch=None, aug_kwargs=None):
        self.hops = hops
        self.aug_kwargs = aug_kwargs or {}
        self._graph_cache: dict[str, Data] = {}
        # build (path, center) index over all cells that have neighbors
        self.index: list[tuple[str, int]] = []
        for p in graph_paths:
            g = torch.load(p, weights_only=False)
            if g.edge_index.shape[1] == 0:
                continue
            self._graph_cache[p] = g
            deg = torch.bincount(g.edge_index[0], minlength=g.num_nodes)
            for c in torch.nonzero(deg > 0).flatten().tolist():
                self.index.append((p, c))
        self.samples_per_epoch = samples_per_epoch or len(self.index)
        print(f"  {len(self.index)} candidate neighborhoods over {len(self._graph_cache)} graphs")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, i):
        # random draw each call (i is ignored beyond bounding the epoch length)
        path, center = self.index[random.randrange(len(self.index))]
        sub = extract_neighborhood(self._graph_cache[path], center, self.hops)
        return augment(sub, **self.aug_kwargs), augment(sub, **self.aug_kwargs)


def paired_collate(batch):
    """Collate list[(a, b)] -> (Batch_a, Batch_b) with aligned ordering."""
    from torch_geometric.data import Batch
    a = Batch.from_data_list([p[0] for p in batch])
    b = Batch.from_data_list([p[1] for p in batch])
    return a, b


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def push_to_drive(stage3_dir: Path) -> None:
    dest = f"gdrive:{DRIVE_ROOT}/stage3"
    print(f"  $ rclone copy {stage3_dir} {dest}")
    r = subprocess.run(["rclone", "copy", str(stage3_dir), dest,
                        "--transfers=8", "--checkers=16"])
    if r.returncode != 0:
        raise RuntimeError(f"rclone failed (exit {r.returncode})")


def main() -> None:
    root = _default_local_root()
    ap = argparse.ArgumentParser(description="Train the Stage 3 neighborhood encoder.")
    ap.add_argument("--graphs-dir", type=Path, default=root / "graphs")
    ap.add_argument("--out-dir", type=Path, default=root)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--samples-per-epoch", type=int, default=50000,
                    help="neighborhoods drawn per epoch (subset of the full set)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--push-to-drive", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("  [warn] no GPU detected — training will be slow. A GPU is recommended for Stage 3.")

    graph_paths = sorted(glob.glob(str(args.graphs_dir / "*" / "*.pt")))
    if not graph_paths:
        raise SystemExit(f"no graphs found under {args.graphs_dir}; run build_graphs.py first.")
    print(f"Loading {len(graph_paths)} image graphs...")

    in_dim = torch.load(graph_paths[0], weights_only=False).x.shape[1]
    ds = ContrastiveNeighborhoods(graph_paths, hops=args.hops,
                                  samples_per_epoch=args.samples_per_epoch)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=paired_collate,
                        drop_last=True, persistent_workers=args.num_workers > 0)

    model = NeighborhoodEncoder(in_dim=in_dim, hidden_dim=args.hidden_dim,
                                emb_dim=args.emb_dim, heads=args.heads,
                                edge_dim=4, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(ds) // args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_at(step: int) -> float:
        if step < warmup_steps:                       # linear warmup
            return args.lr * step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * args.lr * (1 + math.cos(math.pi * prog))   # cosine decay

    stage3 = args.out_dir / "stage3"
    stage3.mkdir(parents=True, exist_ok=True)
    (stage3 / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    log_path = stage3 / "train_log.jsonl"
    log_path.write_text("")

    print(f"Training {args.epochs} epochs x {steps_per_epoch} steps "
          f"(batch {args.batch_size}, {len(ds)} samples/epoch)")
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0; nb = 0
        for view_a, view_b in loader:
            for grp in opt.param_groups:
                grp["lr"] = lr_at(step)
            view_a = view_a.to(device); view_b = view_b.to(device)
            _, z1 = model(view_a)
            _, z2 = model(view_b)
            loss = nt_xent_loss(z1, z2, temperature=args.temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1; step += 1

        ep_loss /= max(1, nb)
        rec = {"epoch": epoch, "loss": ep_loss, "lr": lr_at(step),
               "elapsed_min": round((time.time() - t0) / 60, 2)}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  epoch {epoch:3d}  loss={ep_loss:.4f}  lr={lr_at(step):.2e}  "
              f"({rec['elapsed_min']:.1f} min)")

        # checkpoint each epoch (last) — cheap, and resumable
        torch.save({"model_state": model.state_dict(),
                    "config": {"in_dim": in_dim, "hidden_dim": args.hidden_dim,
                               "emb_dim": args.emb_dim, "heads": args.heads,
                               "edge_dim": 4, "dropout": args.dropout},
                    "epoch": epoch, "loss": ep_loss},
                   stage3 / "encoder.pt")

    print(f"\nDone. Encoder -> {stage3 / 'encoder.pt'}")
    if args.push_to_drive:
        print("Pushing stage3/ to Drive...")
        push_to_drive(stage3)


if __name__ == "__main__":
    main()
