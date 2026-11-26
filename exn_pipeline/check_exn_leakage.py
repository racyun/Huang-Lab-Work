#!/usr/bin/env python3
"""Step 2b: prove the target cell's own markers and label cannot reach its prediction.

Checks are made on model outputs, not on code: change a value, rerun the model,
and see whether the prediction moves. Run it before training (random weights)
and again after (pass --trained to use the saved runs).

    1. split        every well and field sits in exactly one split
    2. columns      Version A has no marker/label column at all; Version B lists
                    every marker/label column as hidden at the target
    3. target       changing the target's own markers/labels by +25 SD leaves
                    the output EXACTLY unchanged (Version B)
    4. control      changing a neighbour's markers DOES change it (so check 3
                    can fail)
    5. junction     with --hide-touching-vecad, changing a touching neighbour's
                    VE-cad leaves the output unchanged; a non-touching one's
                    changes it
    6. exactness    for Version A, the ego-graph output equals the output of the
                    same model run on the whole field (ego extraction is exact)
    7. baselines    no baseline table contains the target's own markers/labels

Usage
-----
    python exn_pipeline/check_exn_leakage.py
    python exn_pipeline/check_exn_leakage.py --trained
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exn_columns import LABEL_SOURCES, LABELS  # noqa: E402
from exn_data import (SPLIT_CODE, TOUCHING_EDGE_COL, build_inputs,  # noqa: E402
                      default_work_dir, ego_graph, load_cache, run_name)
from exn_model import ExnCellClassifier  # noqa: E402

FAILS = []
CONFIGS = [(lab, ver, hide) for lab in sorted(LABELS) for ver, hide in
           (("A", False), ("B", False), ("B", True))]


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


@torch.no_grad()
def out(model, cache, inputs, c, V=None) -> float:
    return float(model(ego_graph(cache, inputs, int(c), V=V)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work-dir", type=Path, default=default_work_dir())
    ap.add_argument("--trained", action="store_true", help="test the saved runs")
    ap.add_argument("--n-targets", type=int, default=60)
    args = ap.parse_args()
    work = args.work_dir.expanduser()
    cache = load_cache(work)
    rng = np.random.default_rng(1)
    torch.manual_seed(1)

    print("1. split")
    for key, arr in (("well", cache.well), ("field", cache.field)):
        n_splits = np.bincount(arr, weights=None, minlength=arr.max() + 1)
        combos = np.unique(np.stack([arr, cache.split]), axis=1)
        per = np.bincount(combos[0], minlength=arr.max() + 1)
        check(f"every {key} is in exactly one split", (per[n_splits > 0] == 1).all())
    for s in ("train", "val", "test"):
        print(f"     {s:5s} targets {len(cache.targets(s)):,}")
    val = cache.targets("val")
    tgt = rng.choice(val, size=min(args.n_targets, len(val)), replace=False)

    for label, version, hide in CONFIGS:
        name = run_name(label, version, hide)
        print(f"\n{name}")
        inputs = build_inputs(cache, label, version, hide)
        hidden = [inputs.cols[i] for i in inputs.hide_at_center]
        if args.trained:
            from train_exn import load_run
            rd = work / "runs" / name
            if not (rd / "best.pt").exists():
                print(f"  (no trained run at {rd}; skipped)")
                continue
            model, inputs, _ = load_run(rd, cache)
        else:
            model = ExnCellClassifier(inputs.in_dim).eval()

        print("2. columns")
        sources = [c for c in inputs.cols if c in LABEL_SOURCES or c.startswith("label_")]
        if version == "A":
            check("Version A has no marker or label column", not sources, str(sources))
        else:
            check("every marker/label column is hidden at the target",
                  set(sources) <= set(hidden), f"hidden={hidden}")
            check(f"the target label {LABELS[label]} is hidden", LABELS[label] in hidden)

        if version == "B":
            print("3/4. target vs neighbour perturbation")
            same, moved, tried = 0, 0, 0
            for c in tgt:
                base = out(model, cache, inputs, c)
                V2 = inputs.V.copy()
                V2[c, inputs.hide_at_center] += 25.0
                same += out(model, cache, inputs, c, V2) == base
                nb = cache.dst[cache.indptr[c]:cache.indptr[c + 1]]
                if len(nb):
                    V3 = inputs.V.copy()
                    V3[nb[0], inputs.hide_at_center] += 25.0
                    moved += abs(out(model, cache, inputs, c, V3) - base) > 1e-6
                    tried += 1
            check("target's own markers/labels change nothing (exact)", same == len(tgt),
                  f"{same}/{len(tgt)} identical")
            check("a neighbour's markers do change the output", moved >= 0.8 * tried,
                  f"{moved}/{tried} changed")

        if hide:
            print("5. junction")
            ok_t, n_t, mv, n_nt = 0, 0, 0, 0
            for c in tgt:
                e0, e1 = cache.indptr[c], cache.indptr[c + 1]
                nb = cache.dst[e0:e1]
                touch = cache.E[e0:e1, TOUCHING_EDGE_COL] > 0.5
                base = out(model, cache, inputs, c)
                if touch.any():
                    V2 = inputs.V.copy()
                    V2[np.ix_(nb[touch], inputs.vecad_cols)] += 25.0
                    ok_t += out(model, cache, inputs, c, V2) == base
                    n_t += 1
                if (~touch).any():
                    V3 = inputs.V.copy()
                    V3[np.ix_(nb[~touch][:1], inputs.vecad_cols)] += 25.0
                    mv += abs(out(model, cache, inputs, c, V3) - base) > 1e-6
                    n_nt += 1
            check("touching neighbours' VE-cad changes nothing", ok_t == n_t,
                  f"{ok_t}/{n_t} identical")
            check("a non-touching neighbour's VE-cad does change it", mv >= 0.8 * n_nt,
                  f"{mv}/{n_nt} changed")

        if version == "A":
            print("6. exactness")
            worst = 0.0
            for f in np.unique(cache.field[tgt])[:3]:
                nodes = np.flatnonzero(cache.field == f)
                local = np.full(cache.n_nodes, -1, dtype=np.int64)
                local[nodes] = np.arange(len(nodes))
                e0, e1 = cache.indptr[nodes[0]], cache.indptr[nodes[-1] + 1]
                src = np.repeat(nodes, np.diff(cache.indptr[nodes[0]:nodes[-1] + 2]))
                ei = torch.from_numpy(np.stack([local[src], local[cache.dst[e0:e1]]]))
                with torch.no_grad():
                    full = model.forward_all(torch.from_numpy(inputs.V[nodes]), ei,
                                             torch.from_numpy(cache.E[e0:e1])).numpy()
                for c in tgt[cache.field[tgt] == f]:
                    worst = max(worst, abs(out(model, cache, inputs, c) - full[local[c]]))
            check("ego-graph output == whole-field output", worst < 1e-4,
                  f"max |diff| {worst:.2e}")

    print("\n7. baselines")
    from run_exn_baselines import NeighbourTables
    for label in sorted(LABELS):
        t = NeighbourTables(cache, label)
        own = [c for c in t.names["own"] if c in LABEL_SOURCES or c.startswith("label_")]
        check(f"{label}: own-feature table has no marker/label column", not own, str(own))
        raw = [c for k in ("nbrA", "nbrB", "nbrB_nj") for c in t.names[k]
               if (c in LABEL_SOURCES or c.startswith("label_"))]
        check(f"{label}: marker/label columns appear only as neighbour averages", not raw,
              str(raw))

    print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} CHECK(S) FAILED: {FAILS}"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
