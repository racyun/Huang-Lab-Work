#!/usr/bin/env python3
"""Stage 3, Step 3 — sample neighborhood subgraphs from the Stage 2 graphs.

For a center cell, extract its k-hop ego-subgraph (the cell + neighbors +
neighbors' neighbors) as a small PyG ``Data`` object — the unit a motif will be
assigned to. See docs/stage3_gnn_encoder.md §2.

Default mode writes a small INSPECTION SAMPLE (a list of subgraphs in one .pt +
a few preview PNGs) so you can verify the 2-hop neighborhoods look right before
wiring the same extractor into training. The full ~1.27M subgraphs are NOT
meant to be persisted as individual files — in training they're extracted
on-the-fly with the same `extract_neighborhood` function.

Inputs:
    <root>/graphs/<condition>/<image_id>.pt   (from build_graphs.py)

Outputs (default sample mode):
    <out>/neighborhood_subgraphs_sample.pt     list[Data], one per sampled cell
    <out>/subgraph_previews/*.png              a few overlays for the eyeball check

Usage:
    python sample_subgraphs.py                          # 200-subgraph sample, 2-hop
    python sample_subgraphs.py --n-sample 500 --hops 2
    python sample_subgraphs.py --push-to-drive          # rclone the sample up
    python sample_subgraphs.py --all                    # shard ALL subgraphs per condition
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import subprocess
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

DRIVE_ROOT = "Fusion AI/Prof Huang Project/Cellpose feature extractions"
DRIVE_SUBDIR = "neighborhood subgraphs"   # target folder on Drive (note the space)


def _default_local_root() -> Path:
    default = Path("/teamspace/studios/this_studio/cellpose_work")
    if "CELLPOSE_LOCAL_ROOT" in os.environ:
        return Path(os.environ["CELLPOSE_LOCAL_ROOT"]).expanduser()
    if default.parent.parent.exists():
        return default
    return Path.home() / "cellpose_work"


def extract_neighborhood(data: Data, center: int, hops: int = 2) -> Data:
    """Return the k-hop ego-subgraph centered on node `center` as a PyG Data.

    The subgraph carries the induced node features, edges + edge attributes, the
    center's index within the subgraph (`center`), the original image-graph node
    index of the center (`center_node`), and the parent image's metadata.
    """
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        center, hops, data.edge_index,
        relabel_nodes=True, num_nodes=data.num_nodes,
    )
    sub = Data(
        x=data.x[subset],
        edge_index=edge_index,
        edge_attr=data.edge_attr[edge_mask] if getattr(data, "edge_attr", None) is not None else None,
        pos=data.pos[subset] if getattr(data, "pos", None) is not None else None,
    )
    sub.center = mapping                 # index of the center within this subgraph
    sub.center_node = int(center)        # original node index in the image graph
    sub.orig_node_idx = subset           # map subgraph nodes -> image-graph nodes
    sub.n_cells = int(subset.numel())
    sub.image_id = data.image_id
    sub.condition = data.condition
    return sub


def save_preview(sub: Data, out_png: Path) -> None:
    """Plot one subgraph: edges + nodes, center highlighted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = sub.pos.numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    ei = sub.edge_index.numpy()
    for s, d in ei.T:
        ax.plot([pos[s, 0], pos[d, 0]], [pos[s, 1], pos[d, 1]],
                lw=0.6, color="gray", alpha=0.6, zorder=1)
    ax.scatter(pos[:, 0], pos[:, 1], s=30, color="tab:blue", zorder=2)
    c = int(sub.center)
    ax.scatter(pos[c, 0], pos[c, 1], s=120, color="red", marker="*", zorder=3,
               label=f"center (cell node {sub.center_node})")
    ax.set_title(f"{sub.condition}/{sub.image_id}  center={sub.center_node}  "
                 f"({sub.n_cells} cells, {ei.shape[1]//2} edges)")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off"); ax.legend(loc="best", fontsize=7)
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def _rclone_copy(local: Path, is_dir: bool) -> None:
    dest = f"gdrive:{DRIVE_ROOT}/{DRIVE_SUBDIR}"
    if is_dir:
        args = ["copy", str(local), f"{dest}/{local.name}", "--transfers=8", "--checkers=16"]
    else:
        args = ["copyto", str(local), f"{dest}/{local.name}"]
    print(f"  $ rclone {' '.join(args)}")
    r = subprocess.run(["rclone", *args])
    if r.returncode != 0:
        raise RuntimeError(f"rclone failed (exit {r.returncode})")


def sample_mode(graphs_dir: Path, out_dir: Path, n_sample: int, hops: int,
                n_previews: int, push: bool) -> None:
    all_pts = glob.glob(str(graphs_dir / "*" / "*.pt"))
    random.shuffle(all_pts)
    subgraphs: list[Data] = []
    previews_dir = out_dir / "subgraph_previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    for p in all_pts:
        if len(subgraphs) >= n_sample:
            break
        g = torch.load(p, weights_only=False)
        if g.edge_index.shape[1] == 0:
            continue
        deg = torch.bincount(g.edge_index[0], minlength=g.num_nodes)
        centers = torch.nonzero(deg > 0).flatten().tolist()
        random.shuffle(centers)
        for c in centers[:5]:                       # a few centers per image
            if len(subgraphs) >= n_sample:
                break
            sub = extract_neighborhood(g, c, hops=hops)
            subgraphs.append(sub)
            if len(subgraphs) <= n_previews:
                save_preview(sub, previews_dir / f"sub_{len(subgraphs):03d}.png")

    out_pt = out_dir / "neighborhood_subgraphs_sample.pt"
    torch.save(subgraphs, out_pt)
    sizes = [s.n_cells for s in subgraphs]
    print(f"Sampled {len(subgraphs)} subgraphs ({hops}-hop)")
    print(f"  cells/subgraph: min={min(sizes)} median={sorted(sizes)[len(sizes)//2]} max={max(sizes)}")
    print(f"Wrote {out_pt}")
    print(f"Wrote {min(n_previews, len(subgraphs))} preview(s) -> {previews_dir}")

    if push:
        print("\nPushing sample to Drive...")
        _rclone_copy(out_pt, is_dir=False)
        _rclone_copy(previews_dir, is_dir=True)


def all_mode(graphs_dir: Path, out_dir: Path, hops: int, push: bool) -> None:
    """Materialize EVERY subgraph, sharded into one .pt list per condition."""
    out_all = out_dir / "neighborhood_subgraphs"
    out_all.mkdir(parents=True, exist_ok=True)
    conditions = sorted(p.name for p in graphs_dir.iterdir() if p.is_dir())
    grand = 0
    for cond in conditions:
        shard: list[Data] = []
        for p in sorted((graphs_dir / cond).glob("*.pt")):
            g = torch.load(p, weights_only=False)
            for c in range(g.num_nodes):
                shard.append(extract_neighborhood(g, c, hops=hops))
        shard_path = out_all / f"{cond}.pt"
        torch.save(shard, shard_path)
        grand += len(shard)
        print(f"  {cond}: {len(shard)} subgraphs -> {shard_path.name}")
    print(f"Total {grand} subgraphs across {len(conditions)} condition(s) -> {out_all}")
    if push:
        print("\nPushing all shards to Drive...")
        _rclone_copy(out_all, is_dir=True)


def main() -> None:
    root = _default_local_root()
    ap = argparse.ArgumentParser(description="Sample neighborhood subgraphs (Stage 3, Step 3).")
    ap.add_argument("--graphs-dir", type=Path, default=root / "graphs")
    ap.add_argument("--out-dir", type=Path, default=root)
    ap.add_argument("--hops", type=int, default=2, help="ego-subgraph radius (default: 2)")
    ap.add_argument("--n-sample", type=int, default=200, help="subgraphs to sample (default: 200)")
    ap.add_argument("--n-previews", type=int, default=8, help="preview PNGs to render (default: 8)")
    ap.add_argument("--all", action="store_true",
                    help="materialize ALL subgraphs, sharded per condition (large)")
    ap.add_argument("--push-to-drive", action="store_true",
                    help="rclone outputs up to gdrive:.../neighborhood subgraphs")
    args = ap.parse_args()

    if not args.graphs_dir.exists():
        raise SystemExit(f"graphs dir not found: {args.graphs_dir}\nRun build_graphs.py first.")

    if args.all:
        all_mode(args.graphs_dir, args.out_dir, args.hops, args.push_to_drive)
    else:
        sample_mode(args.graphs_dir, args.out_dir, args.n_sample, args.hops,
                    args.n_previews, args.push_to_drive)


if __name__ == "__main__":
    main()
