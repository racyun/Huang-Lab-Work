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

from models.neighborhood_encoder import (
    NeighborhoodEncoder, ConditionAdversary, EndMTHead,
    nt_xent_loss, batchnorm_intensities_,
)

INTENSITY_COLS = [
    "ch1_cellwise_mean_intensity",
    "ch2_cellwise_mean_membrane_intensity",
    "ch3_cellwise_mean_intensity",
]


def adv_label_from_condition(condition: str, target: str) -> str:
    """Derive the adversary's target label from a condition folder name.

    '260516_500kPa' -> date '260516' | stiffness '500kPa' | condition (whole).

    Scrubbing `date` targets the BATCH confound only, leaving stiffness (real
    biology, and the cause of EndMT) untouched — unlike scrubbing `condition`,
    which fuses the two and destroys legitimate signal.
    """
    if target == "date":
        return str(condition).partition("_")[0]
    if target == "stiffness":
        return str(condition).partition("_")[2] or "unknown"
    return str(condition)

DRIVE_ROOT = "Fusion AI/Prof Huang Project/Cellpose feature extractions"


def _init_wandb(args):
    """Start a W&B run if --wandb is set. Safe no-op if wandb missing/disabled."""
    if not args.wandb:
        return None
    try:
        import os
        import wandb
    except ImportError:
        print("[wandb] not installed (pip install wandb); continuing without logging.")
        return None
    try:
        key = os.environ.get("WANDB_API_KEY")
        if key:
            wandb.login(key=key)
        return wandb.init(project=args.wandb_project, entity=args.wandb_entity or None,
                          name=args.wandb_run_name, tags=["stage3", "encoder"],
                          config=vars(args))
    except Exception as e:
        print(f"[wandb] WARNING: could not init ({e}); continuing without logging.")
        return None


def _wandb_log(run, metrics: dict, step: int) -> None:
    if run is not None:
        run.log(metrics, step=step)


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

def augment(sub: Data, edge_drop=0.2, feat_mask=0.1, node_drop=0.1, feat_noise=0.05,
            intensity_jitter=0.0, intensity_cols=None) -> Data:
    x = sub.x.clone()
    edge_index = sub.edge_index
    edge_attr = sub.edge_attr
    center = int(sub.center)
    n = x.shape[0]

    # ----- intensity jitter: ONE shared scale+offset across all cells in this
    # view, applied to the marker channels only. Mimics a real staining /
    # illumination difference (coherent per image), unlike feat_noise which is
    # independent per cell. Because the two views get different jitter, the
    # contrastive loss teaches "absolute brightness does not define a
    # neighborhood" — batch-invariance learned WITHOUT fighting biology.
    # endmt_score is deliberately excluded: it is a within-cell ratio and is
    # already batch-robust by construction.
    if intensity_jitter > 0 and intensity_cols:
        scale = 1.0 + intensity_jitter * (torch.rand(1).item() * 2.0 - 1.0)
        offset = intensity_jitter * (torch.rand(1).item() * 2.0 - 1.0)
        x[:, intensity_cols] = x[:, intensity_cols] * scale + offset

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

    def __init__(self, graph_paths, hops=2, samples_per_epoch=None, aug_kwargs=None,
                 endmt_idx=None, batchnorm_cols=None, adv_target="condition"):
        self.hops = hops
        self.aug_kwargs = aug_kwargs or {}
        self.endmt_idx = endmt_idx                 # feature index of endmt_score (or None)
        self.adv_target = adv_target
        self._graph_cache: dict[str, Data] = {}
        # build (path, center) index over all cells that have neighbors
        self.index: list[tuple[str, int]] = []
        labels = set()
        for p in graph_paths:
            g = torch.load(p, weights_only=False)
            if g.edge_index.shape[1] == 0:
                continue
            if batchnorm_cols:                     # per-image intensity batch-norm
                batchnorm_intensities_(g.x, batchnorm_cols)
            self._graph_cache[p] = g
            labels.add(adv_label_from_condition(g.condition, adv_target))
            deg = torch.bincount(g.edge_index[0], minlength=g.num_nodes)
            for c in torch.nonzero(deg > 0).flatten().tolist():
                self.index.append((p, c))
        # stable adversary label -> integer (target set by --adv-target)
        self.conditions = sorted(labels)
        self.cond_to_idx = {c: i for i, c in enumerate(self.conditions)}
        self._path_cond = {
            p: self.cond_to_idx[adv_label_from_condition(g.condition, adv_target)]
            for p, g in self._graph_cache.items()
        }
        self.samples_per_epoch = samples_per_epoch or len(self.index)
        print(f"  {len(self.index)} candidate neighborhoods over "
              f"{len(self._graph_cache)} graphs")
        print(f"  adversary target '{adv_target}': {len(self.conditions)} classes "
              f"{self.conditions}")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, i):
        # random draw each call (i is ignored beyond bounding the epoch length)
        path, center = self.index[random.randrange(len(self.index))]
        g = self._graph_cache[path]
        sub = extract_neighborhood(g, center, self.hops)
        cond = self._path_cond[path]
        # center cell's EndMT score, taken BEFORE augmentation (clean target)
        endmt = float(g.x[center, self.endmt_idx]) if self.endmt_idx is not None else 0.0
        va, vb = augment(sub, **self.aug_kwargs), augment(sub, **self.aug_kwargs)
        for v in (va, vb):
            v.y = torch.tensor([cond])                       # condition label
            v.endmt = torch.tensor([endmt], dtype=torch.float)  # EndMT target
        return va, vb


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
    ap.add_argument("--seed", type=int, default=0,
                    help="random seed for torch/numpy/python. Change it to train an "
                         "independent encoder for the reproducibility test.")
    # ----- adversarial confound scrubbing (methodology §6, §9) -----
    ap.add_argument("--adversarial", action="store_true",
                    help="add a gradient-reversal adversary to scrub the confound")
    ap.add_argument("--adv-target", choices=["condition", "date", "stiffness"],
                    default="condition",
                    help="what the adversary is trained to forget. 'date' scrubs the "
                         "BATCH confound only, leaving stiffness biology intact "
                         "(recommended); 'condition' fuses both (default, legacy)")
    ap.add_argument("--adv-lambda-max", type=float, default=1.0,
                    help="peak GRL strength lambda (ramped 0 -> this)")
    ap.add_argument("--adv-gamma", type=float, default=10.0,
                    help="DANN lambda-ramp steepness")
    ap.add_argument("--adv-hidden", type=int, default=64,
                    help="hidden width of the adversary MLP")
    ap.add_argument("--adv-layers", type=int, default=1,
                    help="depth of the adversary MLP (>1 = stronger adversary)")
    ap.add_argument("--adv-steps", type=int, default=1,
                    help="adversary-only update steps per batch (>1 = stronger adversary)")
    # ----- EndMT-retention head (methodology §A) -----
    ap.add_argument("--endmt-head", action="store_true",
                    help="add an EndMT regression head to anchor biology in the embedding")
    ap.add_argument("--endmt-weight", type=float, default=1.0,
                    help="weight of the EndMT regression loss")
    # ----- input-level batch correction (methodology §C) -----
    ap.add_argument("--batch-norm-features", action="store_true",
                    help="per-image standardize intensity channels. NOT RECOMMENDED: "
                         "measured WORSE batch leakage than without it (+0.365 vs "
                         "+0.289 pure-batch excess). Kept for reproducibility.")
    ap.add_argument("--intensity-jitter", type=float, default=0.0,
                    help="augmentation: coherent per-view scale+offset on the marker "
                         "channels (e.g. 0.3), teaching batch-invariance through the "
                         "contrastive loss instead of fighting it adversarially")
    ap.add_argument("--push-to-drive", action="store_true")
    ap.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases")
    ap.add_argument("--wandb-project", default="huang-lab-stage3")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-run-name", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    if args.seed != 0:
        print(f"  seed={args.seed} (non-default: use this to train an independent "
              f"encoder for the reproducibility test)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("  [warn] no GPU detected — training will be slow. A GPU is recommended for Stage 3.")

    graph_paths = sorted(glob.glob(str(args.graphs_dir / "*" / "*.pt")))
    if not graph_paths:
        raise SystemExit(f"no graphs found under {args.graphs_dir}; run build_graphs.py first.")
    print(f"Loading {len(graph_paths)} image graphs...")

    g0 = torch.load(graph_paths[0], weights_only=False)
    in_dim = g0.x.shape[1]

    # resolve feature-column indices from feature_names.json
    endmt_idx = None; batchnorm_cols = None; intensity_cols = None
    fn_path = args.graphs_dir / "feature_names.json"
    if fn_path.exists():
        feat_names = json.loads(fn_path.read_text())
        if "endmt_score" in feat_names:
            endmt_idx = feat_names.index("endmt_score")
        idx = [feat_names.index(c) for c in INTENSITY_COLS if c in feat_names]
        if args.batch_norm_features:
            batchnorm_cols = idx
            print(f"  batch-norm ON — per-image standardizing columns {batchnorm_cols}")
        if args.intensity_jitter > 0:
            intensity_cols = idx
            print(f"  intensity jitter ON (±{args.intensity_jitter}) on columns {idx}")
    if args.endmt_head and endmt_idx is None:
        raise SystemExit("--endmt-head needs endmt_score in feature_names.json")

    aug_kwargs = {"intensity_jitter": args.intensity_jitter,
                  "intensity_cols": intensity_cols}
    ds = ContrastiveNeighborhoods(graph_paths, hops=args.hops,
                                  samples_per_epoch=args.samples_per_epoch,
                                  aug_kwargs=aug_kwargs,
                                  endmt_idx=endmt_idx, batchnorm_cols=batchnorm_cols,
                                  adv_target=args.adv_target)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=paired_collate,
                        drop_last=True, persistent_workers=args.num_workers > 0)

    model = NeighborhoodEncoder(in_dim=in_dim, hidden_dim=args.hidden_dim,
                                emb_dim=args.emb_dim, heads=args.heads,
                                edge_dim=4, dropout=args.dropout).to(device)

    # ----- EndMT-retention head (optional): anchors biology, trained normally -----
    endmt_head = None
    enc_params = list(model.parameters())
    if args.endmt_head:
        endmt_head = EndMTHead(args.emb_dim, hidden=args.adv_hidden).to(device)
        enc_params += list(endmt_head.parameters())
        print(f"  EndMT-retention head ON — weight {args.endmt_weight}")

    # ----- adversary (optional): predicts condition from h through a GRL -----
    adversary = None
    if args.adversarial:
        adversary = ConditionAdversary(args.emb_dim, len(ds.conditions),
                                       hidden=args.adv_hidden, layers=args.adv_layers).to(device)
        # default (adv_steps==1): adversary trained in the combined GRL step (as before).
        # adv_steps>1 adds extra adversary-only updates via a separate optimizer.
        enc_params += list(adversary.parameters())
        print(f"  adversarial scrubbing ON — target='{args.adv_target}' "
              f"({len(ds.conditions)} classes), lambda 0->{args.adv_lambda_max}, "
              f"adv_layers={args.adv_layers}, adv_steps={args.adv_steps}")
    opt = torch.optim.AdamW(enc_params, lr=args.lr, weight_decay=args.weight_decay)
    adv_opt = (torch.optim.AdamW(adversary.parameters(), lr=args.lr)
               if (adversary is not None and args.adv_steps > 1) else None)

    steps_per_epoch = max(1, len(ds) // args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_at(step: int) -> float:
        if step < warmup_steps:                       # linear warmup
            return args.lr * step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * args.lr * (1 + math.cos(math.pi * prog))   # cosine decay

    def lambda_at(step: int) -> float:
        # DANN schedule: lambda = lambda_max * (2/(1+exp(-gamma*p)) - 1), p in [0,1]
        p = step / max(1, total_steps)
        return args.adv_lambda_max * (2.0 / (1.0 + math.exp(-args.adv_gamma * p)) - 1.0)

    stage3 = args.out_dir / "stage3"
    stage3.mkdir(parents=True, exist_ok=True)
    (stage3 / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    log_path = stage3 / "train_log.jsonl"
    log_path.write_text("")

    wandb_run = _init_wandb(args)
    if wandb_run is not None:
        print(f"[wandb] logging to {args.wandb_project}")

    print(f"Training {args.epochs} epochs x {steps_per_epoch} steps "
          f"(batch {args.batch_size}, {len(ds)} samples/epoch)")
    ce = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()
    best_scrub = float("inf")          # |adv_acc - chance|, lower = better scrubbed
    adv_chance = 1.0 / max(1, len(ds.conditions))
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        if adversary is not None:
            adversary.train()
        if endmt_head is not None:
            endmt_head.train()
        ep_loss = ep_con = ep_adv = ep_adv_acc = ep_endmt = 0.0; nb = 0
        for view_a, view_b in loader:
            for grp in opt.param_groups:
                grp["lr"] = lr_at(step)
            # BUGFIX: the adversary's own optimizer must follow the SAME schedule.
            # Previously its LR stayed pinned at args.lr while the encoder's decayed
            # to ~0, so in the final epochs the encoder was effectively frozen while
            # the adversary kept training at full speed — it won by default, and the
            # last-epoch checkpoint captured the encoder at its worst.
            if adv_opt is not None:
                for grp in adv_opt.param_groups:
                    grp["lr"] = lr_at(step)
            view_a = view_a.to(device); view_b = view_b.to(device)
            h1, z1 = model(view_a)
            h2, z2 = model(view_b)
            loss_con = nt_xent_loss(z1, z2, temperature=args.temperature)
            loss = loss_con
            h = torch.cat([h1, h2], dim=0)                      # both views

            # ----- EndMT-retention head (anchors biology; no reversal) -----
            endmt_val = 0.0
            if endmt_head is not None:
                tgt = torch.cat([view_a.endmt, view_b.endmt], dim=0).view(-1)
                loss_endmt = mse(endmt_head(h), tgt)
                loss = loss + args.endmt_weight * loss_endmt
                endmt_val = loss_endmt.item()

            # ----- adversary -----
            lam = 0.0; loss_adv_val = 0.0; adv_acc = 0.0
            if adversary is not None:
                lam = lambda_at(step)
                y = torch.cat([view_a.y, view_b.y], dim=0).view(-1)
                # extra adversary-only updates (strengthen the spy) if adv_steps>1
                if adv_opt is not None:
                    for _ in range(args.adv_steps - 1):
                        la = ce(adversary(h.detach(), 1.0), y)
                        adv_opt.zero_grad(); la.backward(); adv_opt.step()
                logits = adversary(h, lam)                      # GRL flips enc. gradient
                loss_adv = ce(logits, y)
                loss = loss + loss_adv
                loss_adv_val = loss_adv.item()
                adv_acc = (logits.argmax(1) == y).float().mean().item()

            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); ep_con += loss_con.item()
            ep_adv += loss_adv_val; ep_adv_acc += adv_acc; ep_endmt += endmt_val
            nb += 1; step += 1
            if step % 10 == 0:
                m = {"train/step_loss": loss.item(), "train/lr": lr_at(step)}
                if adversary is not None:
                    m.update({"adv/lambda": lam, "adv/step_loss": loss_adv_val,
                              "adv/step_acc": adv_acc})
                if endmt_head is not None:
                    m["endmt/step_loss"] = endmt_val
                _wandb_log(wandb_run, m, step)

        ep_loss /= max(1, nb); ep_con /= max(1, nb)
        ep_adv /= max(1, nb); ep_adv_acc /= max(1, nb); ep_endmt /= max(1, nb)
        rec = {"epoch": epoch, "loss": ep_loss, "con_loss": ep_con,
               "lr": lr_at(step), "elapsed_min": round((time.time() - t0) / 60, 2)}
        wm = {"train/loss": ep_loss, "train/con_loss": ep_con,
              "train/lr": lr_at(step), "epoch": epoch}
        extra = ""
        if adversary is not None:
            rec.update({"adv_loss": ep_adv, "adv_acc": ep_adv_acc, "lambda": lambda_at(step)})
            wm.update({"adv/loss": ep_adv, "adv/acc": ep_adv_acc, "adv/lambda": lambda_at(step)})
            extra += f"  adv_acc={ep_adv_acc:.3f}  lam={lambda_at(step):.2f}"
        if endmt_head is not None:
            rec["endmt_loss"] = ep_endmt
            wm["endmt/loss"] = ep_endmt
            extra += f"  endmt_mse={ep_endmt:.3f}"
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  epoch {epoch:3d}  loss={ep_loss:.4f}  con={ep_con:.4f}"
              f"{extra}  ({rec['elapsed_min']:.1f} min)")
        _wandb_log(wandb_run, wm, step)

        # checkpoint each epoch (last) — cheap, and resumable
        torch.save({"model_state": model.state_dict(),
                    "config": {"in_dim": in_dim, "hidden_dim": args.hidden_dim,
                               "emb_dim": args.emb_dim, "heads": args.heads,
                               "edge_dim": 4, "dropout": args.dropout,
                               "batch_norm_features": args.batch_norm_features,
                               "adv_target": args.adv_target if args.adversarial else None,
                               "intensity_jitter": args.intensity_jitter},
                    "epoch": epoch, "loss": ep_loss},
                   stage3 / "encoder.pt")

        # Track the best-scrubbed epoch: adversary accuracy closest to chance.
        # Only consider epochs where lambda has substantially ramped, otherwise an
        # early epoch wins simply because the adversary hasn't learned yet.
        if adversary is not None and lambda_at(step) >= 0.5 * args.adv_lambda_max:
            scrub = abs(ep_adv_acc - adv_chance)
            if scrub < best_scrub:
                best_scrub = scrub
                torch.save({"model_state": model.state_dict(),
                            "config": {"in_dim": in_dim, "hidden_dim": args.hidden_dim,
                                       "emb_dim": args.emb_dim, "heads": args.heads,
                                       "edge_dim": 4, "dropout": args.dropout,
                                       "batch_norm_features": args.batch_norm_features,
                                       "adv_target": args.adv_target,
                                       "intensity_jitter": args.intensity_jitter},
                            "epoch": epoch, "loss": ep_loss, "adv_acc": ep_adv_acc},
                           stage3 / "encoder_best.pt")
                print(f"    ^ best scrub so far (adv_acc={ep_adv_acc:.3f} "
                      f"vs chance {adv_chance:.3f}) -> encoder_best.pt")

    print(f"\nDone. Encoder -> {stage3 / 'encoder.pt'}")
    if wandb_run is not None:
        wandb_run.finish()
    if args.push_to_drive:
        print("Pushing stage3/ to Drive...")
        push_to_drive(stage3)


if __name__ == "__main__":
    main()
