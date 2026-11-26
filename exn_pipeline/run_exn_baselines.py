#!/usr/bin/env python3
"""Step 3: non-graph baselines, so each GNN result has a reference point.

For each label, on the same wells and target cells as the GNN:

    own_lr                logistic regression, target cell's own 12 features
    own_hgb               gradient-boosted trees, same 12 features
    nbrA_hgb              own features + mean of the 1-hop neighbours' features
                          + mean edge distance and fraction touching
                          -> what Version A could get from simple averages
    nbrB_hgb              nbrA + mean of the neighbours' markers and labels
                          -> what Version B could get from simple averages
    nbrB_nojunction_hgb   nbrA + neighbours' markers/labels averaged over
                          NON-touching neighbours only (junction check)

None of them reads the target cell's own markers or label. Models are fitted on
the training wells and scored on validation; score_exn.py scores them on test.

Outputs in <work>/baselines/<label>__<name>/:
    model.joblib, config.json, val_predictions.csv.gz, val_metrics.json

Usage
-----
    python exn_pipeline/run_exn_baselines.py                 # both labels
    python exn_pipeline/run_exn_baselines.py --label red --max-train-cells 20000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exn_columns import LABELS  # noqa: E402
from exn_data import (EDGE_COLS, TOUCHING_EDGE_COL, best_f1_threshold,  # noqa: E402
                      build_inputs, default_work_dir, load_cache, metrics)

BASELINES = ["own_lr", "own_hgb", "nbrA_hgb", "nbrB_hgb", "nbrB_nojunction_hgb"]


class NeighbourTables:
    """Per-node feature tables for the baselines, built once for all nodes."""

    def __init__(self, cache, label: str):
        inp = build_inputs(cache, label, "B")
        self.cols = inp.cols
        hidden = set(inp.hide_at_center.tolist())
        self.own_idx = [j for j in range(len(inp.cols)) if j not in hidden]
        self.mark_idx = sorted(hidden)
        V = inp.V
        n = cache.n_nodes
        deg = np.diff(cache.indptr)
        src = np.repeat(np.arange(n), deg)
        dst = cache.dst.astype(np.int64)
        touch = cache.E[:, TOUCHING_EDGE_COL] > 0.5

        def mean_over(cols, mask=None):
            w = np.ones(len(dst)) if mask is None else mask.astype(float)
            cnt = np.bincount(src, weights=w, minlength=n)
            out = np.full((n, len(cols)), np.nan, dtype=np.float32)
            for k, j in enumerate(cols):
                s = np.bincount(src, weights=V[dst, j] * w, minlength=n)
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[:, k] = s / cnt
            return out, cnt

        self.own = V[:, self.own_idx]
        nbr_feat, _ = mean_over(self.own_idx)
        e_dist = np.bincount(src, weights=cache.E[:, EDGE_COLS.index("dist_norm")], minlength=n)
        e_touch = np.bincount(src, weights=touch.astype(float), minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            edge_summ = np.stack([e_dist / deg, e_touch / deg], axis=1).astype(np.float32)
        self.nbrA = np.concatenate([self.own, nbr_feat, edge_summ], axis=1)
        nbr_mark, _ = mean_over(self.mark_idx)
        self.nbrB = np.concatenate([self.nbrA, nbr_mark], axis=1)
        nbr_mark_nt, cnt_nt = mean_over(self.mark_idx, ~touch)
        self.nbrB_nj = np.concatenate([self.nbrA, nbr_mark_nt,
                                       cnt_nt[:, None].astype(np.float32)], axis=1)
        own_names = [inp.cols[j] for j in self.own_idx]
        mark_names = [inp.cols[j] for j in self.mark_idx]
        self.names = {
            "own": own_names,
            "nbrA": own_names + [f"nbr_mean_{c}" for c in own_names] +
                    ["nbr_mean_dist_norm", "nbr_frac_touching"],
        }
        self.names["nbrB"] = self.names["nbrA"] + [f"nbr_mean_{c}" for c in mark_names]
        self.names["nbrB_nj"] = self.names["nbrA"] + \
            [f"nontouching_nbr_mean_{c}" for c in mark_names] + ["n_nontouching_nbrs"]

    def table(self, baseline: str):
        key = {"own_lr": "own", "own_hgb": "own", "nbrA_hgb": "nbrA", "nbrB_hgb": "nbrB",
               "nbrB_nojunction_hgb": "nbrB_nj"}[baseline]
        return {"own": self.own, "nbrA": self.nbrA, "nbrB": self.nbrB,
                "nbrB_nj": self.nbrB_nj}[key], self.names[key]


def make_model(baseline: str):
    if baseline == "own_lr":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.1,
                                          early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=20, random_state=0)


def predict_baseline(model, X: np.ndarray) -> np.ndarray:
    if getattr(model, "_lr_nan_fill", False):      # logistic regression cannot take NaN
        X = np.nan_to_num(X, nan=0.0)
    return model.predict_proba(X)[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work-dir", type=Path, default=default_work_dir())
    ap.add_argument("--label", choices=sorted(LABELS), nargs="+", default=sorted(LABELS))
    ap.add_argument("--baselines", nargs="+", choices=BASELINES, default=BASELINES)
    ap.add_argument("--max-train-cells", type=int, default=0, help="subsample (smoke tests)")
    ap.add_argument("--max-val-cells", type=int, default=0, help="subsample (smoke tests)")
    args = ap.parse_args()

    import joblib

    work = args.work_dir.expanduser()
    t0 = time.time()
    cache = load_cache(work)
    rng = np.random.default_rng(0)
    tr, va = cache.targets("train"), cache.targets("val")
    if args.max_train_cells and len(tr) > args.max_train_cells:
        tr = np.sort(rng.choice(tr, args.max_train_cells, replace=False))
    full_val = len(va)
    if args.max_val_cells and len(va) > args.max_val_cells:
        va = np.sort(rng.choice(va, args.max_val_cells, replace=False))
    print(f"run_exn_baselines.py  {datetime.now():%Y-%m-%d %H:%M:%S}  "
          f"train {len(tr):,}  val {len(va):,}")
    for label in args.label:
        tables = NeighbourTables(cache, label)
        y = cache.y(label)
        for b in args.baselines:
            tb = time.time()
            X, names = tables.table(b)
            model = make_model(b)
            Xtr = X[tr]
            if b == "own_lr":
                Xtr = np.nan_to_num(Xtr, nan=0.0)
                model._lr_nan_fill = True
            model.fit(Xtr, y[tr])
            p = predict_baseline(model, X[va])
            out = work / "baselines" / f"{label}__{b}"
            out.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, out / "model.joblib")
            (out / "config.json").write_text(json.dumps(
                {"label": label, "baseline": b, "features": names,
                 "wells_sha256": cache.meta["wells_sha256"],
                 "n_train": int(len(tr))}, indent=2))
            pd.DataFrame({"node": va, "field": cache.field[va], "well": cache.well[va],
                          "y": y[va], "p": p}).to_csv(out / "val_predictions.csv.gz",
                                                      index=False)
            m = metrics(y[va], p, cache.field[va])
            m.update(threshold_best_f1=best_f1_threshold(y[va], p),
                     n_val_cells_scored=int(len(va)), complete=len(va) == full_val)
            (out / "val_metrics.json").write_text(json.dumps(m, indent=2))
            print(f"  {label:10s} {b:20s} AUPRC {m['auprc']:.4f}  AUROC {m['auroc']:.4f}  "
                  f"within-field AUROC {m['within_field_auroc']:.4f}  "
                  f"({len(names)} features, {time.time() - tb:.0f} s)", flush=True)
    print(f"Done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
