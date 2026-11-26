#!/usr/bin/env python3
"""Step 2: train one GNN experiment on the frozen well split.

    --label {tagln_high, red}   target: label_tagln_high or label_red
    --version {A, B}            A = morphology only; B = A + neighbours' markers/labels
    --hide-touching-vecad       (B only) junction check: hide VE-cad of touching neighbours

Each training step samples target cells from the training wells, cuts their
2-hop ego graphs (target's own markers hidden), and minimises
BCEWithLogits(pos_weight = negatives / positives). The checkpoint kept is the
epoch with the best AUPRC on a fixed sample of validation cells; training stops
after --patience epochs without improvement. The test wells are never read.

Outputs in <work>/runs/<label>_<version>[_nojunction]/:
    best.pt                 weights + everything needed to rebuild the inputs
    config.json             arguments, input columns, split hash
    train_log.jsonl         one line per epoch
    val_predictions.csv.gz  every validation target: node, field, well, y, p
    val_metrics.json        AUROC, AUPRC, within-field AUROC, best-F1 threshold

Usage
-----
    python exn_pipeline/train_exn.py --label red --version A
    # smoke test (minutes, CPU):
    python exn_pipeline/train_exn.py --label red --version B --epochs 1 \
        --samples-per-epoch 2000 --val-cells 2000 --max-val-cells 5000 --run-name smoke
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exn_columns import LABELS  # noqa: E402
from exn_data import (RULES, TargetCells, best_f1_threshold, build_inputs,  # noqa: E402
                      default_work_dir, load_cache, metrics, run_name)
from exn_model import ExnCellClassifier  # noqa: E402


def make_loader(cache, inputs, nodes, y, batch_size, num_workers, shuffle=False):
    from torch_geometric.loader import DataLoader

    return DataLoader(TargetCells(cache, inputs, nodes, y), batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers,
                      persistent_workers=False, pin_memory=False)


@torch.no_grad()
def predict(model, cache, inputs, nodes, device, batch_size=1024, num_workers=0) -> np.ndarray:
    """Probability for each node in `nodes`, in order."""
    model.eval()
    out = np.empty(len(nodes), dtype=np.float64)
    loader = make_loader(cache, inputs, nodes, np.zeros(len(nodes)), batch_size, num_workers)
    i = 0
    for batch in loader:
        p = torch.sigmoid(model(batch.to(device))).cpu().numpy()
        out[i:i + len(p)] = p
        i += len(p)
    return out


def load_run(run_dir: Path, cache, device="cpu"):
    """Rebuild a trained run: (model, inputs, checkpoint dict)."""
    ck = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    cfg = ck["config"]
    if cfg["wells_sha256"] != cache.meta["wells_sha256"]:
        raise SystemExit(f"[error] {run_dir.name} was trained on a different split")
    inputs = build_inputs(cache, cfg["label"], cfg["version"], cfg["hide_touching_vecad"],
                          cfg["use_conditions"])
    if inputs.cols != cfg["inputs"]["node_cols"]:
        raise SystemExit(f"[error] {run_dir.name}: input columns changed since training")
    model = ExnCellClassifier(inputs.in_dim, hidden_dim=cfg["hidden_dim"],
                              heads=cfg["heads"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, inputs, ck


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work-dir", type=Path, default=default_work_dir())
    ap.add_argument("--label", required=True, choices=sorted(LABELS))
    ap.add_argument("--version", required=True, choices=["A", "B"])
    ap.add_argument("--hide-touching-vecad", action="store_true")
    ap.add_argument("--use-conditions", action="store_true")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--samples-per-epoch", type=int, default=200_000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-cells", type=int, default=50_000,
                    help="validation targets scored each epoch for early stopping")
    ap.add_argument("--max-val-cells", type=int, default=0,
                    help="cap on the final validation scoring (0 = all; smoke tests only)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="overwrite an existing run")
    args = ap.parse_args()
    if args.hide_touching_vecad and args.version != "B":
        ap.error("--hide-touching-vecad only applies to --version B")

    work = args.work_dir.expanduser()
    name = args.run_name or run_name(args.label, args.version, args.hide_touching_vecad,
                                     args.use_conditions)
    out = work / "runs" / name
    if (out / "best.pt").exists() and not args.force:
        raise SystemExit(f"[error] {out} already has a trained model; pass --force to retrain")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)

    t0 = time.time()
    cache = load_cache(work)
    inputs = build_inputs(cache, args.label, args.version, args.hide_touching_vecad,
                          args.use_conditions)
    y_all = cache.y(args.label).astype(np.float32)
    train_nodes, val_nodes = cache.targets("train"), cache.targets("val")
    y_tr = y_all[train_nodes]
    pos_weight = float((1 - y_tr).sum() / max(y_tr.sum(), 1))
    val_sub = np.sort(rng.choice(val_nodes, size=min(args.val_cells, len(val_nodes)),
                                 replace=False))
    print(f"train_exn.py  {datetime.now():%Y-%m-%d %H:%M:%S}  run={name}  device={device}")
    print(f"  label={args.label} ({LABELS[args.label]})  version={args.version}  "
          f"hide_touching_vecad={args.hide_touching_vecad}")
    print(f"  node inputs ({len(inputs.cols)} + {inputs.n_flags} flags): {', '.join(inputs.cols)}")
    print(f"  hidden at the target: {[inputs.cols[i] for i in inputs.hide_at_center] or 'none'}")
    print(f"  train targets {len(train_nodes):,} ({100 * y_tr.mean():.2f}% positive)  "
          f"val targets {len(val_nodes):,}  pos_weight {pos_weight:.2f}  "
          f"(loaded in {time.time() - t0:.0f} s)")

    config = {**vars(args), "work_dir": str(work), "run_name": name,
              "wells_sha256": cache.meta["wells_sha256"], "rules": RULES,
              "inputs": inputs.describe(), "in_dim": inputs.in_dim, "pos_weight": pos_weight,
              "device": str(device), "started": datetime.now().isoformat(timespec="seconds")}
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str))

    model = ExnCellClassifier(inputs.in_dim, hidden_dim=args.hidden_dim, heads=args.heads,
                              dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    best, best_epoch, stale = -math.inf, -1, 0
    log = open(out / "train_log.jsonl", "w")
    for epoch in range(1, args.epochs + 1):
        te = time.time()
        model.train()
        pick = rng.choice(len(train_nodes), size=min(args.samples_per_epoch, len(train_nodes)),
                          replace=False)
        loader = make_loader(cache, inputs, train_nodes[pick], y_all[train_nodes[pick]],
                             args.batch_size, args.num_workers)
        tot, n, skipped = 0.0, 0, 0
        for batch in loader:
            batch = batch.to(device)
            loss = loss_fn(model(batch), batch.y)
            if not torch.isfinite(loss):
                skipped += 1
                if skipped > 20:
                    raise SystemExit("[error] training diverged (non-finite loss); "
                                     "lower --lr")
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            tot += float(loss) * batch.num_graphs
            n += batch.num_graphs
        p_val = predict(model, cache, inputs, val_sub, device, 2 * args.batch_size,
                        args.num_workers)
        m = metrics(y_all[val_sub], p_val, cache.field[val_sub])
        improved = m["auprc"] > best
        if improved:
            best, best_epoch, stale = m["auprc"], epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": {
                "label": args.label, "version": args.version,
                "hide_touching_vecad": args.hide_touching_vecad,
                "use_conditions": args.use_conditions, "hidden_dim": args.hidden_dim,
                "heads": args.heads, "dropout": args.dropout,
                "wells_sha256": cache.meta["wells_sha256"], "inputs": inputs.describe(),
                "epoch": epoch, "val_subset_auprc": m["auprc"]}}, out / "best.pt")
        else:
            stale += 1
        rec = {"epoch": epoch, "train_loss": tot / max(n, 1), "skipped_batches": skipped,
               "val_auprc": m["auprc"], "val_auroc": m["auroc"],
               "val_within_field_auroc": m["within_field_auroc"],
               "seconds": round(time.time() - te, 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"  epoch {epoch:2d}  loss {rec['train_loss']:.4f}  val AUPRC {m['auprc']:.4f}  "
              f"AUROC {m['auroc']:.4f}  within-field AUROC {m['within_field_auroc']:.4f}  "
              f"{'*' if improved else ' '}  {rec['seconds']:.0f} s", flush=True)
        if stale >= args.patience:
            print(f"  no improvement for {args.patience} epochs; stopping")
            break
    log.close()

    # ---- final validation scoring with the best checkpoint
    model, inputs, ck = load_run(out, cache, device)
    nodes = val_nodes
    if args.max_val_cells and len(nodes) > args.max_val_cells:
        nodes = np.sort(rng.choice(nodes, size=args.max_val_cells, replace=False))
    p = predict(model, cache, inputs, nodes, device, 2 * args.batch_size, args.num_workers)
    y = y_all[nodes]
    pd.DataFrame({"node": nodes, "field": cache.field[nodes], "well": cache.well[nodes],
                  "y": y.astype(np.int8), "p": p}).to_csv(out / "val_predictions.csv.gz",
                                                          index=False)
    res = metrics(y, p, cache.field[nodes])
    res.update(best_epoch=best_epoch, threshold_best_f1=best_f1_threshold(y, p),
               n_val_cells_scored=int(len(nodes)), complete=len(nodes) == len(val_nodes),
               minutes=round((time.time() - t0) / 60, 1))
    (out / "val_metrics.json").write_text(json.dumps(res, indent=2))
    print("\n" + "=" * 72)
    print(f"{name}: best epoch {best_epoch}  |  validation ({len(nodes):,} cells, "
          f"{100 * res['prevalence']:.2f}% positive)")
    print(f"  AUPRC {res['auprc']:.4f}  (chance = {res['prevalence']:.4f})")
    print(f"  AUROC {res['auroc']:.4f}  |  within-field AUROC {res['within_field_auroc']:.4f}")
    print(f"  wrote {out}  ({res['minutes']} min)")
    print("=" * 72)


if __name__ == "__main__":
    main()
