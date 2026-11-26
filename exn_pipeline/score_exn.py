#!/usr/bin/env python3
"""Step 4: compare the GNNs with the baselines, on validation or (once) on test.

    --split val            reads the saved validation predictions; run as often
                           as you like while developing
    --split test --final   predicts the locked test wells with every trained
                           GNN and baseline, scores them, and records that the
                           test set has been used. Run it once, after all models
                           are final. A second run needs --rescore and is logged.

For each label it reports AUROC, AUPRC (chance = positive rate) and within-field
AUROC, each with a 95% CI from resampling wells, plus paired differences:

    GNN-A  - HGB own            do neighbours add anything beyond the cell itself?
    GNN-A  - HGB nbr avg (A)    does the graph add anything beyond neighbour averages?
    GNN-B  - GNN-A              do neighbours' markers/labels add anything?
    GNN-B  - HGB nbr avg (B)
    GNN-B  - GNN-B no junction  how much of B comes from touching neighbours' VE-cad?

On test, precision / recall / F1 / balanced accuracy use the threshold chosen
on validation (best F1), so the test set never picks its own threshold.

Outputs in <work>/results/: <split>_results.json, <split>_summary.csv, <split>_summary.md
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
from exn_data import (bootstrap, default_work_dir, load_cache, metrics,  # noqa: E402
                      threshold_metrics)

# display name -> (kind, directory suffix)
MODELS = {
    "LR own": ("baseline", "own_lr"),
    "HGB own": ("baseline", "own_hgb"),
    "HGB nbr avg (A)": ("baseline", "nbrA_hgb"),
    "GNN-A": ("gnn", "A"),
    "HGB nbr avg (B)": ("baseline", "nbrB_hgb"),
    "GNN-B": ("gnn", "B"),
    "HGB nbr avg (B, no junction)": ("baseline", "nbrB_nojunction_hgb"),
    "GNN-B no junction": ("gnn", "B_nojunction"),
}
PAIRS = [("GNN-A", "HGB own"), ("GNN-A", "HGB nbr avg (A)"), ("GNN-B", "GNN-A"),
         ("GNN-B", "HGB nbr avg (B)"), ("GNN-B", "GNN-B no junction"),
         ("HGB nbr avg (B)", "HGB nbr avg (B, no junction)")]


def model_dir(work: Path, label: str, kind: str, suffix: str) -> Path:
    return (work / "runs" / f"{label}_{suffix}" if kind == "gnn"
            else work / "baselines" / f"{label}__{suffix}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work-dir", type=Path, default=default_work_dir())
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final", action="store_true", help="required to score the test wells")
    ap.add_argument("--rescore", action="store_true", help="score test again (logged)")
    ap.add_argument("--label", nargs="+", choices=sorted(LABELS), default=sorted(LABELS))
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args()

    work = args.work_dir.expanduser()
    t0 = time.time()
    cache = load_cache(work)
    res_dir = work / "results"
    res_dir.mkdir(exist_ok=True)

    if args.split == "test":
        if not args.final:
            raise SystemExit("[error] the test wells are locked; pass --final when every "
                             "model is final")
        lock = work / "test" / "SCORED.json"
        if lock.exists() and not args.rescore:
            raise SystemExit(f"[error] the test set was already scored ({lock}); pass "
                             f"--rescore to score it again (this is recorded)")
        (work / "test").mkdir(exist_ok=True)
        history = json.loads(lock.read_text())["history"] if lock.exists() else []
        history.append({"when": datetime.now().isoformat(timespec="seconds"),
                        "labels": args.label})
        lock.write_text(json.dumps({"history": history}, indent=2))
        if len(history) > 1:
            print(f"[warn] test set scored {len(history)} times; report that")

    results, rows = {}, []
    for label in args.label:
        print(f"\n=== {label}  ({LABELS[label]})")
        preds, thresholds = {}, {}
        tables = None
        if args.split == "test":
            nodes = cache.targets("test")
            y = cache.y(label)[nodes]
        for disp, (kind, suf) in MODELS.items():
            d = model_dir(work, label, kind, suf)
            vm = d / "val_metrics.json"
            if not vm.exists():
                continue
            thresholds[disp] = json.loads(vm.read_text())["threshold_best_f1"]
            if args.split == "val":
                df = pd.read_csv(d / "val_predictions.csv.gz")
                preds[disp] = df.set_index("node")["p"]
                continue
            tp = work / "test" / f"{d.name}.csv.gz"
            if kind == "gnn":
                import torch
                from train_exn import load_run, pick_device, predict
                dev = pick_device(args.device)
                model, inputs, _ = load_run(d, cache, dev)
                p = predict(model, cache, inputs, nodes, dev, args.batch_size,
                            args.num_workers)
                del model, inputs
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            else:
                import joblib
                from run_exn_baselines import NeighbourTables, predict_baseline
                if tables is None:
                    tables = NeighbourTables(cache, label)
                X, _ = tables.table(suf)
                p = predict_baseline(joblib.load(d / "model.joblib"), X[nodes])
            pd.DataFrame({"node": nodes, "field": cache.field[nodes],
                          "well": cache.well[nodes], "y": y, "p": p}).to_csv(tp, index=False)
            preds[disp] = pd.Series(p, index=nodes)
            print(f"  predicted test with {disp}  ({time.time() - t0:.0f} s)", flush=True)
        if not preds:
            print("  no trained models found")
            continue

        # score every model on the same cells
        common = sorted(set.intersection(*(set(s.index) for s in preds.values())))
        nodes_c = np.asarray(common)
        y_c = cache.y(label)[nodes_c]
        field_c, well_c = cache.field[nodes_c], cache.well[nodes_c]
        P = {k: s.reindex(nodes_c).to_numpy() for k, s in preds.items()}
        if any(len(s) != len(nodes_c) for s in preds.values()):
            print(f"  [note] models were scored on different cells; using the "
                  f"{len(nodes_c):,} they share")
        m = {k: metrics(y_c, p, field_c) for k, p in P.items()}
        if args.split == "test":
            for k in m:
                m[k]["at_val_threshold"] = threshold_metrics(y_c, P[k], thresholds[k])
        boot = bootstrap(y_c, P, field_c, well_c, n_boot=args.n_boot,
                         pairs=[p for p in PAIRS if p[0] in P and p[1] in P])
        results[label] = {"metrics": m, "bootstrap": boot, "n_cells": int(len(nodes_c)),
                          "n_wells": int(len(np.unique(well_c))),
                          "prevalence": float(y_c.mean())}

        print(f"  {len(nodes_c):,} cells in {len(np.unique(well_c))} wells, "
              f"{100 * y_c.mean():.2f}% positive (= AUPRC of a random guess)")
        print(f"  {'model':30s} {'AUPRC':>22s} {'AUROC':>22s} {'within-field AUROC':>22s}")
        for k in P:
            ci = boot["models"][k]
            cells = [f"{m[k][q]:.3f} [{ci[q]['low']:.3f}, {ci[q]['high']:.3f}]"
                     for q in ("auprc", "auroc", "within_field_auroc")]
            print(f"  {k:30s} {cells[0]:>22s} {cells[1]:>22s} {cells[2]:>22s}")
            row = {"split": args.split, "label": label, "model": k,
                   **{q: m[k][q] for q in ("auprc", "auroc", "within_field_auroc")},
                   **{f"{q}_ci_low": ci[q]["low"] for q in ci},
                   **{f"{q}_ci_high": ci[q]["high"] for q in ci},
                   "prevalence": m[k]["prevalence"], "n_cells": m[k]["n_cells"]}
            if args.split == "test":
                row.update({f"test_{q}": v for q, v in m[k]["at_val_threshold"].items()
                            if q in ("threshold", "precision", "recall", "f1",
                                     "balanced_accuracy")})
            rows.append(row)
        if boot["differences"]:
            print("  paired differences (95% CI over wells):")
            for pair, d in boot["differences"].items():
                print(f"    {pair:45s} AUPRC {_fmt(d['auprc'])}   "
                      f"within-field AUROC {_fmt(d['within_field_auroc'])}")
                diff_a, diff_b = pair.split(" - ")
                rows.append({"split": args.split, "label": label, "model": f"Δ {pair}",
                             "auprc": m[diff_a]["auprc"] - m[diff_b]["auprc"],
                             "auroc": m[diff_a]["auroc"] - m[diff_b]["auroc"],
                             "within_field_auroc": m[diff_a]["within_field_auroc"]
                             - m[diff_b]["within_field_auroc"],
                             **{f"{q}_ci_low": d[q]["low"] for q in d},
                             **{f"{q}_ci_high": d[q]["high"] for q in d}})

    (res_dir / f"{args.split}_results.json").write_text(json.dumps(results, indent=2))
    table = pd.DataFrame(rows)
    table.to_csv(res_dir / f"{args.split}_summary.csv", index=False)
    with open(res_dir / f"{args.split}_summary.md", "w") as f:
        f.write(f"# ExN cell classification, {args.split} split  "
                f"({datetime.now():%Y-%m-%d %H:%M})\n\n")
        for label, r in results.items():
            f.write(f"## {label}: {r['n_cells']:,} cells, {r['n_wells']} wells, "
                    f"{100 * r['prevalence']:.2f}% positive\n\n")
            f.write("| model | AUPRC | AUROC | within-field AUROC |\n|---|---|---|---|\n")
            for k, mm in r["metrics"].items():
                ci = r["bootstrap"]["models"][k]
                f.write(f"| {k} | " + " | ".join(
                    f"{mm[q]:.3f} [{ci[q]['low']:.3f}, {ci[q]['high']:.3f}]"
                    for q in ("auprc", "auroc", "within_field_auroc")) + " |\n")
            if r["bootstrap"]["differences"]:
                f.write("\n| difference | AUPRC | within-field AUROC |\n|---|---|---|\n")
                for pair, d in r["bootstrap"]["differences"].items():
                    a, b = pair.split(" - ")
                    f.write(f"| {pair} | "
                            f"{r['metrics'][a]['auprc'] - r['metrics'][b]['auprc']:+.3f} "
                            f"{_fmt(d['auprc'])} | "
                            f"{r['metrics'][a]['within_field_auroc'] - r['metrics'][b]['within_field_auroc']:+.3f} "
                            f"{_fmt(d['within_field_auroc'])} |\n")
            f.write("\n")
    print(f"\nWrote {res_dir}/{args.split}_summary.md, .csv, _results.json  "
          f"({(time.time() - t0) / 60:.1f} min)")


def _fmt(ci: dict) -> str:
    return f"[{ci['low']:+.3f}, {ci['high']:+.3f}]"


if __name__ == "__main__":
    main()
