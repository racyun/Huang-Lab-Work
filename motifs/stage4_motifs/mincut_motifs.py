#!/usr/bin/env python3
"""End-to-end motif discovery: train a soft-assignment GNN on the tissue graphs.

The alternative to "encoder, then k-means". One network, one objective, and the
objective is partition quality (see models/mincut_motifs.py). The cluster count
is a CEILING the model may undershoot rather than a target it must hit.

Runs locally on CPU or Apple MPS — the unit of work is one image (hundreds of
cells), not the full 1.27M-cell matrix, so no GPU cluster is needed.

ENSEMBLE. A single run is one stochastic optimisation. Following CytoCommunity,
several runs are trained and combined by majority vote. The paper does not say
how labels are matched across runs before voting, and that step is essential:
cluster ids are arbitrary, so run 1's "cluster 3" and run 2's "cluster 3" are
unrelated and voting on raw ids produces noise. Labels are therefore aligned
with the Hungarian matching already written for robustness_motifs.py before any
vote is taken.

Outputs:
    <out>/motifs.parquet          same schema as cluster_motifs.py
    <out>/mincut_report.json      per-run effective counts, losses, agreement
    <out>/mincut_log.txt
    <out>/run<N>_model.pt

Usage:
    python mincut_motifs.py --graphs-dir ~/cellpose_work/graphs \\
        --out-dir ~/cellpose_work/v4/mincut --max-clusters 20 --runs 5
    python mincut_motifs.py ... --adversarial        # keep the date scrubbing
    python mincut_motifs.py ... --ceiling-sweep 10,20,30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.mincut_motifs import (  # noqa: E402
    ConditionAdversary, SoftAssignGNN, effective_clusters, grad_reverse,
    partition_losses,
)


class _Tee:
    def __init__(self, path: Path):
        self.file = open(path, "w")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


def _default_local_root() -> Path:
    if "CELLPOSE_LOCAL_ROOT" in os.environ:
        return Path(os.environ["CELLPOSE_LOCAL_ROOT"]).expanduser()
    legacy = Path("/teamspace/studios/this_studio/cellpose_work")
    if legacy.parent.parent.exists():
        return legacy
    return Path.home() / "cellpose_work"


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS is available on this Mac but several scatter kernels PyG needs still
    # fall back to CPU, which ends up slower than plain CPU for these graph
    # sizes. CPU is the honest default; --device mps is there to retest.
    return torch.device("cpu")


def load_graphs(graphs_dir: Path, limit: int | None, seed: int):
    """Load per-image PyG graphs, carrying (condition, image_id) with each."""
    files = sorted(graphs_dir.rglob("*.pt"))
    files = [f for f in files if f.name != "feature_names.json"]
    if not files:
        raise SystemExit(
            f"no graphs under {graphs_dir}\n"
            f"Lightning is gone; fetch them:  ./motifs/fetch_from_drive.sh --graphs")
    if limit and len(files) > limit:
        rng = np.random.default_rng(seed)
        files = [files[i] for i in sorted(rng.choice(len(files), limit, replace=False))]
    graphs, meta = [], []
    for f in files:
        try:
            g = torch.load(str(f), weights_only=False)
        except Exception as e:
            print(f"  [skip] {f.name}: {e}")
            continue
        if g.num_nodes < 4 or g.edge_index.numel() == 0:
            continue
        graphs.append(g)
        meta.append((f.parent.name, f.stem))
    print(f"  loaded {len(graphs)} graphs, {sum(g.num_nodes for g in graphs):,} cells")
    return graphs, meta


def date_of(condition: str) -> str:
    return str(condition).partition("_")[0]


def train_one(graphs, meta, c_max, args, seed, device):
    """Train one soft-assignment model; return per-graph hard labels."""
    from torch_geometric.loader import DataLoader
    torch.manual_seed(seed)
    np.random.seed(seed)

    in_dim = graphs[0].x.shape[1]
    edge_dim = graphs[0].edge_attr.shape[1] if graphs[0].edge_attr is not None else 0
    model = SoftAssignGNN(in_dim, args.hidden_dim, c_max, args.heads,
                          edge_dim, args.dropout, args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    adv, adv_opt, dates = None, None, None
    if args.adversarial:
        dates = sorted({date_of(c) for c, _ in meta})
        d_index = {d: i for i, d in enumerate(dates)}
        adv = ConditionAdversary(args.hidden_dim, len(dates), args.adv_hidden, 2).to(device)
        adv_opt = torch.optim.Adam(adv.parameters(), lr=args.lr)
        print(f"    date adversary over {len(dates)} dates: {dates}")

    for i, g in enumerate(graphs):
        g.graph_idx = torch.tensor([i])
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=True)
    total_steps = max(1, args.epochs * len(loader))
    step = 0

    model.train()
    for ep in range(args.epochs):
        agg = {"mincut": 0.0, "ortho": 0.0, "adv": 0.0, "n": 0}
        for batch in loader:
            batch = batch.to(device)
            s_logits, h, mask = model(batch.x, batch.edge_index,
                                      batch.edge_attr, batch.batch)
            s = torch.softmax(s_logits, dim=-1)
            mincut, ortho = partition_losses(h, batch.edge_index, batch.batch, s, mask)
            loss = mincut + args.ortho_weight * ortho

            adv_val = 0.0
            if adv is not None:
                # DANN ramp: let the partition form before scrubbing hard.
                p = step / total_steps
                lam = args.adv_lambda_max * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)
                y = torch.tensor(
                    [d_index[date_of(meta[int(gi)][0])] for gi in batch.graph_idx],
                    device=device)
                y_node = y[batch.batch]
                # ConditionAdversary applies the GRL itself; passing a
                # pre-reversed tensor would reverse the gradient twice and
                # leave the sign unchanged, silently disabling the scrubbing.
                logits = adv(h, lam)
                adv_loss = torch.nn.functional.cross_entropy(logits, y_node)
                loss = loss + adv_loss
                adv_val = float(adv_loss)

            opt.zero_grad(set_to_none=True)
            if adv_opt is not None:
                adv_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if adv_opt is not None:
                adv_opt.step()

            agg["mincut"] += float(mincut); agg["ortho"] += float(ortho)
            agg["adv"] += adv_val; agg["n"] += 1
            step += 1
        if ep % max(1, args.epochs // 10) == 0 or ep == args.epochs - 1:
            n = max(agg["n"], 1)
            msg = (f"    epoch {ep + 1:>3}/{args.epochs}  mincut {agg['mincut']/n:+.4f}"
                   f"  ortho {agg['ortho']/n:.4f}")
            if adv is not None:
                msg += f"  adv {agg['adv']/n:.4f}"
            print(msg)

    # ---- inference: hard labels for every cell, in graph order ----
    model.eval()
    labels, shares = [], None
    eval_loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            s_logits, _, mask = model(batch.x, batch.edge_index,
                                      batch.edge_attr, batch.batch)
            s = torch.softmax(s_logits, dim=-1)
            n_eff, shares = effective_clusters(s, mask)
            hard = s.argmax(-1)
            for b in range(s.size(0)):
                labels.append(hard[b][mask[b]].cpu().numpy())
    flat = np.concatenate(labels)
    used = np.bincount(flat, minlength=c_max) / len(flat)
    n_eff = int((used >= 0.01).sum())
    return flat, n_eff, used, model


def consensus(runs, c_max):
    """Majority vote across runs, after aligning cluster ids.

    Cluster ids are arbitrary per run, so every run is first matched onto run 0
    by maximising overlap (Hungarian). The CytoCommunity paper omits this step;
    without it the vote is meaningless.
    """
    from scipy.optimize import linear_sum_assignment
    ref = runs[0]
    aligned = [ref]
    agreements = []
    for other in runs[1:]:
        C = np.zeros((c_max, c_max), dtype=np.int64)
        np.add.at(C, (ref, other), 1)
        row, col = linear_sum_assignment(-C)
        mapping = np.arange(c_max)
        for r, cc in zip(row, col):
            mapping[cc] = r
        a = mapping[other]
        aligned.append(a)
        agreements.append(float((a == ref).mean()))
    stack = np.stack(aligned)
    votes = np.zeros((len(ref), c_max), dtype=np.int32)
    for r in range(stack.shape[0]):
        votes[np.arange(len(ref)), stack[r]] += 1
    final = votes.argmax(1).astype(np.int32)
    confidence = votes.max(1) / stack.shape[0]
    return final, confidence, agreements


def main() -> None:
    root = _default_local_root()
    ap = argparse.ArgumentParser(description="MinCut soft-assignment motif discovery.")
    ap.add_argument("--graphs-dir", type=Path, default=root / "graphs")
    ap.add_argument("--out-dir", type=Path, default=root / "v4" / "mincut")
    ap.add_argument("--max-clusters", type=int, default=20,
                    help="CEILING on cluster count; the model may use fewer")
    ap.add_argument("--ceiling-sweep", default=None,
                    help="comma-separated ceilings, e.g. 10,20,30. The effective "
                         "count should be stable across them if it is real.")
    ap.add_argument("--runs", type=int, default=5, help="ensemble size")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16, help="images per batch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--ortho-weight", type=float, default=1.0,
                    help="raise if assignments stay mushy; lower if cluster sizes "
                         "are forced too equal")
    ap.add_argument("--adversarial", action="store_true",
                    help="port the v4 date-scrubbing adversary (recommended: "
                         "without it this reintroduces the batch confound)")
    ap.add_argument("--adv-lambda-max", type=float, default=1.0)
    ap.add_argument("--adv-hidden", type=int, default=64)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tee = _Tee(args.out_dir / "mincut_log.txt")
    sys.stdout = tee
    try:
        device = pick_device(args.device)
        print(f"mincut_motifs.py — {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"  graphs-dir : {args.graphs_dir}")
        print(f"  device     : {device}")
        print(f"  ceiling    : {args.max_clusters}   runs: {args.runs}   "
              f"epochs: {args.epochs}")
        print(f"  adversarial: {args.adversarial}")
        if not args.adversarial:
            print("  [warn] no date adversary: this run reintroduces the batch "
                  "confound that v4 removed (+0.289 -> +0.168). Use --adversarial "
                  "for anything you intend to report.")

        print("\nLoading graphs...")
        graphs, meta = load_graphs(args.graphs_dir, args.max_images, args.seed)
        keys = pd.DataFrame(
            [(c, i, int(cid)) for (c, i), g in zip(meta, graphs)
             for cid in (g.cell_id.tolist() if hasattr(g, "cell_id")
                         else range(1, g.num_nodes + 1))],
            columns=["condition", "image_id", "cell_id"])

        ceilings = ([int(x) for x in args.ceiling_sweep.split(",")]
                    if args.ceiling_sweep else [args.max_clusters])
        report = {"ceilings": {}, "device": str(device),
                  "adversarial": args.adversarial, "n_graphs": len(graphs)}

        for c_max in ceilings:
            print(f"\n{'=' * 70}\nCEILING c = {c_max}\n{'=' * 70}")
            runs, effs = [], []
            for r in range(args.runs):
                t0 = time.time()
                print(f"  run {r + 1}/{args.runs} (seed {args.seed + r})")
                lab, n_eff, used, model = train_one(
                    graphs, meta, c_max, args, args.seed + r, device)
                print(f"    -> {n_eff} of {c_max} clusters carry >=1% of cells "
                      f"({time.time() - t0:.0f}s)")
                runs.append(lab); effs.append(n_eff)
                torch.save({"model_state": model.state_dict(), "c_max": c_max,
                            "seed": args.seed + r, "n_effective": n_eff},
                           args.out_dir / f"run{r}_c{c_max}_model.pt")

            final, conf, agree = consensus(runs, c_max)
            n_final = int((np.bincount(final, minlength=c_max) / len(final) >= 0.01).sum())
            print(f"\n  effective counts per run : {effs}")
            print(f"  pairwise agreement vs run 0: "
                  f"{[f'{a:.3f}' for a in agree] if agree else 'n/a'}")
            print(f"  consensus                 : {n_final} clusters, "
                  f"mean vote confidence {conf.mean():.3f}")
            report["ceilings"][str(c_max)] = {
                "effective_per_run": effs, "consensus_clusters": n_final,
                "agreement_vs_run0": agree, "vote_confidence": float(conf.mean())}

            if c_max == ceilings[-1]:
                out = keys.copy()
                out["motif"] = final[:len(out)] if len(final) >= len(out) else np.pad(
                    final, (0, len(out) - len(final)))
                out.to_parquet(args.out_dir / "motifs.parquet", index=False)
                np.save(args.out_dir / "consensus_confidence.npy", conf)
                print(f"\n  Wrote {args.out_dir / 'motifs.parquet'}")

        if len(ceilings) > 1:
            counts = [report["ceilings"][str(c)]["consensus_clusters"] for c in ceilings]
            print(f"\n{'=' * 70}\nCEILING SENSITIVITY")
            for c, n in zip(ceilings, counts):
                print(f"  ceiling {c:>3}  ->  {n} clusters")
            if max(counts) - min(counts) <= 1:
                print("  STABLE — the count barely moves with the ceiling, so the "
                      "model is finding a number rather than filling what it was given.")
            else:
                print("  UNSTABLE — the count tracks the ceiling, so c is still "
                      "setting the answer. Treat the number as a parameter, not a "
                      "discovery, and say so.")
            print("=" * 70)

        (args.out_dir / "mincut_report.json").write_text(json.dumps(report, indent=2))
        print(f"\nWrote {args.out_dir / 'mincut_report.json'}")
    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f"Log saved -> {args.out_dir / 'mincut_log.txt'}")


if __name__ == "__main__":
    main()
