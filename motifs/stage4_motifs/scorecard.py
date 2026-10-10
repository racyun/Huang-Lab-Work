#!/usr/bin/env python3
"""Go / no-go scorecard: are the motifs good enough to write up?

Scores each run against five criteria fixed BEFORE looking at the results, so
the decision is not argued after the fact:

  1. separation   >= 3 motifs whose mean EndMT scores differ pairwise by
                  Cohen's d >= 0.5 (v4: 5 of 6 motifs sit near 0.39)
  2. reproducible cross-seed encoder ARI >= 0.6
                  (robustness_motifs.py --embeddings-b <seed-1 embeddings>)
  3. not batch    the softest-vs-stiffest contrast is significant and every
                  same-stiffness date contrast is smaller than it
                  (check_date_vs_stiffness.py)
  4. spatial      spatial coherence beats the no-neighbour baseline validated
                  on the SAME graphs (validate_motifs.py on both)
  5. visible      a biologist tells >= 2 motifs apart by eye (entered by hand
                  with --expert)

A criterion whose inputs are missing shows NOT RUN, and the overall verdict
stays INCOMPLETE rather than passing on partial evidence.

Each run directory should hold motifs.parquet plus whatever the other Stage 4
scripts wrote next to it with their default --out-dir:
    validation_report.json   (validate_motifs.py)
    robustness_report.json   (robustness_motifs.py, with --embeddings-b)
date_check.json is computed here if absent.

Spatial coherence depends on the graph, so validate the baseline once per
edge rule, into separate folders:
    python validate_motifs.py --motifs <root>/baseline_cells/motifs.parquet \
        --graphs-dir <root>/delaunay/graphs --out-dir <root>/baseline_cells/delaunay

Usage:
    python scorecard.py --master-table <root>/master_table.csv \
        --run v4=<root>/v4/stage4 --baseline v4=<root>/baseline_cells \
        --run delaunay=<root>/delaunay/stage4 \
        --baseline delaunay=<root>/baseline_cells/delaunay \
        --expert v4=1 --expert delaunay=3 --out-dir <root>/scorecard
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from check_date_vs_stiffness import run_date_check

MIN_D = 0.5
MIN_SEPARATED = 3
MIN_ARI = 0.6
MIN_EXPERT = 2


def _pairs(values: list[str], flag: str) -> dict[str, str]:
    out = {}
    for v in values or []:
        name, sep, rest = v.partition("=")
        if not sep:
            raise SystemExit(f"{flag} expects NAME=VALUE, got {v!r}")
        out[name] = rest
    return out


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def endmt_separation(mdf: pd.DataFrame, endmt: pd.DataFrame) -> dict:
    """Largest set of motifs whose EndMT means are pairwise >= MIN_D apart.

    Cohen's d uses the pooled within-motif SD. This is an effect size, not a
    test: with a million cells every difference is 'significant', so only the
    size of the gap is informative.
    """
    merged = mdf.merge(endmt, on=["condition", "image_id", "cell_id"], how="inner")
    if len(merged) < len(mdf):
        print(f"  [warn] {len(mdf) - len(merged)} cells had no EndMT score in the "
              "master table")
    stats = merged.groupby("motif")["endmt_score"].agg(["mean", "std", "count"])

    def d(a: int, b: int) -> float:
        sa, sb = stats.loc[a], stats.loc[b]
        pooled = np.sqrt(((sa["count"] - 1) * sa["std"] ** 2 +
                          (sb["count"] - 1) * sb["std"] ** 2) /
                         (sa["count"] + sb["count"] - 2))
        return abs(sa["mean"] - sb["mean"]) / pooled if pooled > 0 else 0.0

    # walk motifs in order of mean EndMT, keeping each one that is far enough
    # from every motif already kept (greedy is exact for 1-D separations with
    # equal SDs, and a close lower bound otherwise)
    kept: list[int] = []
    for m in stats.sort_values("mean").index:
        if all(d(m, k) >= MIN_D for k in kept):
            kept.append(int(m))

    return {"n_separated": len(kept), "separated_motifs": kept,
            "motif_endmt_mean": {int(k): float(v) for k, v in stats["mean"].items()},
            "motif_endmt_sd": {int(k): float(v) for k, v in stats["std"].items()}}


def score_run(name: str, run_dir: Path, endmt: pd.DataFrame,
              baseline_dir: Path | None, expert: int | None) -> dict:
    print(f"\n=== {name}  ({run_dir})")
    mdf = pd.read_parquet(run_dir / "motifs.parquet")
    row: dict = {"run": name, "n_motifs": int(mdf["motif"].max()) + 1}

    # 1. separation
    sep = endmt_separation(mdf, endmt)
    row["1_separated_motifs"] = sep["n_separated"]
    row["1_pass"] = sep["n_separated"] >= MIN_SEPARATED
    means = ", ".join(f"{k}:{v:.3f}" for k, v in sep["motif_endmt_mean"].items())
    print(f"  1. EndMT means by motif: {means}")
    print(f"     pairwise-separated (d >= {MIN_D}): {sep['separated_motifs']}")

    # 2. reproducibility
    rob = _read_json(run_dir / "robustness_report.json")
    ari = (rob or {}).get("encoder_reproducibility", {}).get("ari")
    row["2_cross_seed_ari"] = ari
    row["2_pass"] = None if ari is None else ari >= MIN_ARI
    print(f"  2. cross-seed ARI: {'NOT RUN' if ari is None else f'{ari:.3f}'}")

    # 3. date vs stiffness
    dc = _read_json(run_dir / "date_check.json")
    if dc is None:
        print("  3. date_check.json not found; computing it now")
        dc = run_date_check(mdf)
        dc.pop("per_motif", None)
        (run_dir / "date_check.json").write_text(json.dumps(dc, indent=2))
    row["3_pass"] = dc["passed"]
    print(f"  3. {dc['verdict']}")

    # 4. spatial coherence vs baseline
    val = _read_json(run_dir / "validation_report.json")
    base = _read_json(baseline_dir / "validation_report.json") if baseline_dir else None
    gnn_sc = (val or {}).get("spatial_coherence", {}).get("ratio")
    base_sc = (base or {}).get("spatial_coherence", {}).get("ratio")
    row["4_coherence"] = gnn_sc
    row["4_baseline_coherence"] = base_sc
    row["4_pass"] = (None if gnn_sc is None or base_sc is None
                     else gnn_sc > base_sc)
    print(f"  4. spatial coherence: run {gnn_sc if gnn_sc is not None else 'NOT RUN'}"
          f" vs baseline {base_sc if base_sc is not None else 'NOT RUN'}")

    # 5. expert
    row["5_expert_distinct"] = expert
    row["5_pass"] = None if expert is None else expert >= MIN_EXPERT
    print(f"  5. motifs a biologist can tell apart: "
          f"{'NOT ENTERED' if expert is None else expert}")

    checks = [row[f"{i}_pass"] for i in range(1, 6)]
    if any(c is False for c in checks):
        row["verdict"] = "FAIL"
    elif any(c is None for c in checks):
        row["verdict"] = "INCOMPLETE"
    else:
        row["verdict"] = "PASS"
    row["separation_detail"] = sep
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description="Go / no-go motif scorecard.")
    ap.add_argument("--master-table", type=Path, required=True)
    ap.add_argument("--run", action="append", required=True,
                    help="NAME=DIR holding motifs.parquet (repeatable)")
    ap.add_argument("--baseline", action="append",
                    help="NAME=DIR holding the baseline's validation_report.json "
                         "for the run of the same NAME")
    ap.add_argument("--expert", action="append",
                    help="NAME=N motifs a biologist could tell apart")
    ap.add_argument("--out-dir", type=Path, default=Path("."))
    args = ap.parse_args()

    runs = _pairs(args.run, "--run")
    baselines = _pairs(args.baseline, "--baseline")
    experts = {k: int(v) for k, v in _pairs(args.expert, "--expert").items()}
    for unknown in (set(baselines) | set(experts)) - set(runs):
        raise SystemExit(f"{unknown!r} given to --baseline/--expert but not to --run")

    endmt = pd.read_csv(args.master_table,
                        usecols=["condition", "image_id", "cell_id", "endmt_score"])

    rows = [score_run(name, Path(d).expanduser(), endmt,
                      Path(baselines[name]).expanduser() if name in baselines else None,
                      experts.get(name))
            for name, d in runs.items()]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "separation_detail"}
                          for r in rows])
    table.to_csv(args.out_dir / "scorecard.csv", index=False)
    (args.out_dir / "scorecard.json").write_text(json.dumps(rows, indent=2, default=str))

    def mark(v) -> str:
        return "NOT RUN" if v is None else ("pass" if v else "FAIL")

    print("\n" + "=" * 78)
    print(f"{'run':<12} {'1 separate':<11} {'2 seed ARI':<11} {'3 not batch':<12} "
          f"{'4 spatial':<10} {'5 expert':<9} verdict")
    for r in rows:
        print(f"{r['run']:<12} {mark(r['1_pass']):<11} {mark(r['2_pass']):<11} "
              f"{mark(r['3_pass']):<12} {mark(r['4_pass']):<10} "
              f"{mark(r['5_pass']):<9} {r['verdict']}")
    print("=" * 78)
    print(f"\nWrote {args.out_dir / 'scorecard.csv'} and scorecard.json")


if __name__ == "__main__":
    main()
