#!/usr/bin/env python3
"""Stage 4 robustness — how much can each motif be trusted?

Accuracy is not measurable here: there is no ground truth for motifs, so
nothing can be scored as right or wrong. What IS measurable is robustness —
whether the same answer survives perturbation. Three tests:

  1. ASSIGNMENT CONFIDENCE   Per cell, how far is it from its own centroid
     versus the next-nearest? Cells with a small margin sit on a cluster
     boundary and could easily flip. Reports the fraction of confidently
     assigned cells overall and per motif — this is what explains visual
     heterogeneity inside a motif folder.

  2. BOOTSTRAP STABILITY     Repeatedly subsample cells, re-cluster, match the
     new clusters back to the originals, and measure what fraction of each
     motif's cells keep their label. Gives a PER-MOTIF score, so a trustworthy
     motif is distinguishable from a fragile one even when the overall
     partition is soft.

  3. ENCODER REPRODUCIBILITY (optional, needs --embeddings-b) Cluster a second
     embedding — e.g. from an encoder trained with a different seed — and
     measure agreement. Bootstrap stability only shows the CLUSTERING is
     reproducible given a fixed embedding; this shows the EMBEDDING is too,
     which is the stronger claim.

Outputs:
    <out>/robustness_report.json
    <out>/motif_confidence.csv
    <out>/robustness_log.txt

Usage:
    python robustness_motifs.py --motifs <stage4>/motifs.parquet \
        --embeddings <stage3>/embeddings.parquet
    python robustness_motifs.py ... --embeddings-b <other>/embeddings.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


class _Tee:
    """Mirror console output into a log file."""

    def __init__(self, path: Path):
        self.file = open(path, "w")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s); self.file.write(s)

    def flush(self):
        self.stdout.flush(); self.file.flush()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


def _default_local_root() -> Path:
    default = Path("/teamspace/studios/this_studio/cellpose_work")
    if "CELLPOSE_LOCAL_ROOT" in os.environ:
        return Path(os.environ["CELLPOSE_LOCAL_ROOT"]).expanduser()
    if default.parent.parent.exists():
        return default
    return Path.home() / "cellpose_work"


def load_embeddings(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    csv = path.with_suffix(".csv")
    if csv.exists():
        return pd.read_csv(csv)
    raise SystemExit(f"embeddings not found: {path}")


def match_labels(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    """Map labels in `b` onto `a` by maximising overlap (Hungarian).

    Cluster ids are arbitrary, so before comparing two partitions the clusters
    must be paired up optimally rather than assumed to share numbering.
    """
    from scipy.optimize import linear_sum_assignment
    C = np.zeros((k, k), dtype=np.int64)
    for i, j in zip(a, b):
        if 0 <= i < k and 0 <= j < k:
            C[i, j] += 1
    row, col = linear_sum_assignment(-C)
    mapping = np.full(k, -1, dtype=int)
    for r, c in zip(row, col):
        mapping[c] = r                      # new label c behaves like original r
    return mapping


def assignment_confidence(X: np.ndarray, labels: np.ndarray, k: int,
                          threshold: float) -> dict:
    """Margin between a cell's own centroid and the next-nearest."""
    print("\n1. ASSIGNMENT CONFIDENCE")
    cent = np.stack([X[labels == m].mean(0) if (labels == m).any()
                     else np.zeros(X.shape[1]) for m in range(k)])
    # distances to every centroid, chunked to bound memory
    d1 = np.empty(len(X)); d2 = np.empty(len(X))
    for i in range(0, len(X), 50_000):
        ch = X[i:i + 50_000]
        d = np.linalg.norm(ch[:, None, :] - cent[None, :, :], axis=2)
        srt = np.sort(d, axis=1)
        d1[i:i + 50_000] = srt[:, 0]
        d2[i:i + 50_000] = srt[:, 1]
    margin = (d2 - d1) / np.maximum(d2, 1e-9)      # 0 = on a boundary, 1 = certain

    conf = float((margin >= threshold).mean())
    print(f"  margin = (dist to 2nd-nearest - dist to own) / dist to 2nd-nearest")
    print(f"  median margin           : {np.median(margin):.3f}")
    print(f"  confidently assigned    : {100 * conf:.1f}%  (margin >= {threshold})")
    per = {}
    for m in range(k):
        sel = labels == m
        if sel.any():
            per[int(m)] = {"median_margin": float(np.median(margin[sel])),
                           "pct_confident": float(100 * (margin[sel] >= threshold).mean()),
                           "n_cells": int(sel.sum())}
    print("  per motif:")
    for m, v in per.items():
        print(f"    motif {m:>2}: median margin {v['median_margin']:.3f}   "
              f"{v['pct_confident']:5.1f}% confident   ({v['n_cells']:,} cells)")
    if conf < 0.6:
        print("  => many cells sit near boundaries; motifs grade into each other")
    else:
        print("  => most cells are well inside their cluster")
    return {"threshold": threshold, "median_margin": float(np.median(margin)),
            "pct_confident_overall": 100 * conf, "per_motif": per}


def bootstrap_stability(X: np.ndarray, labels: np.ndarray, k: int,
                        n_boot: int, frac: float, seed: int) -> dict:
    """Subsample, re-cluster, and measure per-motif label retention."""
    from sklearn.cluster import KMeans
    print(f"\n2. BOOTSTRAP STABILITY  ({n_boot} rounds, {int(frac * 100)}% subsamples)")
    rng = np.random.default_rng(seed)
    keep = {m: [] for m in range(k)}
    overall = []
    for b in range(n_boot):
        idx = rng.choice(len(X), size=int(frac * len(X)), replace=False)
        new = KMeans(n_clusters=k, n_init=5, random_state=b).fit_predict(X[idx])
        mapping = match_labels(labels[idx], new, k)
        agreed = mapping[new] == labels[idx]
        overall.append(float(agreed.mean()))
        for m in range(k):
            sel = labels[idx] == m
            if sel.any():
                keep[m].append(float(agreed[sel].mean()))
        print(f"  round {b + 1}: {100 * overall[-1]:.1f}% of cells keep their label")

    per = {}
    for m in range(k):
        if keep[m]:
            per[int(m)] = {"mean_retention": float(np.mean(keep[m])),
                           "std": float(np.std(keep[m]))}
    print("  per-motif retention (higher = more trustworthy):")
    for m, v in sorted(per.items(), key=lambda kv: -kv[1]["mean_retention"]):
        flag = "  <-- fragile" if v["mean_retention"] < 0.7 else ""
        print(f"    motif {m:>2}: {100 * v['mean_retention']:5.1f}% "
              f"(+/- {100 * v['std']:.1f}){flag}")
    return {"n_boot": n_boot, "frac": frac,
            "overall_mean_retention": float(np.mean(overall)), "per_motif": per}


def encoder_reproducibility(Xa: np.ndarray, labels_a: np.ndarray, Xb: np.ndarray,
                            k: int, seed: int) -> dict:
    """Cluster a second embedding and compare partitions cell-for-cell."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    print("\n3. ENCODER REPRODUCIBILITY (second embedding)")
    lb = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xb)
    mapping = match_labels(labels_a, lb, k)
    agreed = mapping[lb] == labels_a
    ari = float(adjusted_rand_score(labels_a, lb))
    print(f"  ARI between encoders     : {ari:.3f}")
    print(f"  cells with matching motif: {100 * agreed.mean():.1f}%")
    per = {}
    for m in range(k):
        sel = labels_a == m
        if sel.any():
            per[int(m)] = float(agreed[sel].mean())
    print("  per-motif agreement:")
    for m, v in sorted(per.items(), key=lambda kv: -kv[1]):
        print(f"    motif {m:>2}: {100 * v:5.1f}%")
    verdict = ("REPRODUCIBLE — the same motifs emerge from an independent encoder"
               if ari >= 0.5 else
               "NOT REPRODUCIBLE — motifs depend on the particular encoder run")
    print(f"  => {verdict}")
    return {"ari": ari, "pct_agree": float(100 * agreed.mean()),
            "per_motif_agreement": per, "verdict": verdict}


def main() -> None:
    root = _default_local_root()
    ap = argparse.ArgumentParser(description="Motif robustness tests (Stage 4).")
    ap.add_argument("--motifs", type=Path, default=root / "stage4" / "motifs.parquet")
    ap.add_argument("--embeddings", type=Path,
                    default=root / "stage3" / "embeddings.parquet")
    ap.add_argument("--embeddings-b", type=Path, default=None,
                    help="second embedding (e.g. a different encoder seed) to test "
                         "encoder-level reproducibility")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--max-cells", type=int, default=100000)
    ap.add_argument("--n-boot", type=int, default=5)
    ap.add_argument("--boot-frac", type=float, default=0.8)
    ap.add_argument("--confidence-threshold", type=float, default=0.10,
                    help="margin above which a cell counts as confidently assigned")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = args.out_dir or args.motifs.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "robustness_log.txt"
    tee = _Tee(log_path)
    sys.stdout = tee
    try:
        print(f"robustness_motifs.py — {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"  motifs      : {args.motifs}")
        print(f"  embeddings  : {args.embeddings}")
        print(f"  embeddings-b: {args.embeddings_b}")
        print(f"  max-cells={args.max_cells}  n-boot={args.n_boot}  "
              f"boot-frac={args.boot_frac}  seed={args.seed}")

        mdf = pd.read_parquet(args.motifs)
        edf = load_embeddings(args.embeddings)
        cols = [c for c in edf.columns if c.startswith("emb_")]
        if len(mdf) != len(edf):
            raise SystemExit(f"row mismatch: motifs {len(mdf)} vs embeddings {len(edf)}")
        k = int(mdf["motif"].max()) + 1

        rng = np.random.default_rng(args.seed)
        idx = (rng.choice(len(edf), size=min(args.max_cells, len(edf)), replace=False)
               if len(edf) > args.max_cells else np.arange(len(edf)))
        X = edf[cols].to_numpy(dtype=np.float32)[idx]
        labels = mdf["motif"].to_numpy()[idx]
        print(f"\nEvaluating {k} motifs over {len(X):,} cells "
              f"(of {len(edf):,} total)")

        report = {"n_motifs": k, "n_cells_evaluated": int(len(X))}
        report["assignment_confidence"] = assignment_confidence(
            X, labels, k, args.confidence_threshold)
        report["bootstrap_stability"] = bootstrap_stability(
            X, labels, k, args.n_boot, args.boot_frac, args.seed)
        if args.embeddings_b:
            edf_b = load_embeddings(args.embeddings_b)
            Xb = edf_b[[c for c in edf_b.columns
                        if c.startswith("emb_")]].to_numpy(dtype=np.float32)[idx]
            report["encoder_reproducibility"] = encoder_reproducibility(
                X, labels, Xb, k, args.seed)

        # per-motif summary table
        rows = []
        for m in range(k):
            c = report["assignment_confidence"]["per_motif"].get(m, {})
            b = report["bootstrap_stability"]["per_motif"].get(m, {})
            rows.append({
                "motif": m,
                "n_cells": c.get("n_cells"),
                "median_margin": round(c.get("median_margin", float("nan")), 3),
                "pct_confident": round(c.get("pct_confident", float("nan")), 1),
                "bootstrap_retention_pct": round(100 * b.get("mean_retention", float("nan")), 1),
            })
        tbl = pd.DataFrame(rows).set_index("motif")
        tbl.to_csv(out_dir / "motif_confidence.csv")

        print("\n" + "=" * 70)
        print("PER-MOTIF ROBUSTNESS")
        print(tbl.to_string())
        print("=" * 70)
        print(f"  overall confidently assigned : "
              f"{report['assignment_confidence']['pct_confident_overall']:.1f}%")
        print(f"  overall bootstrap retention  : "
              f"{100 * report['bootstrap_stability']['overall_mean_retention']:.1f}%")
        if "encoder_reproducibility" in report:
            print(f"  encoder reproducibility (ARI): "
                  f"{report['encoder_reproducibility']['ari']:.3f}")
        print("=" * 70)

        (out_dir / "robustness_report.json").write_text(json.dumps(report, indent=2))
        print(f"\nWrote {out_dir / 'robustness_report.json'}")
        print(f"Wrote {out_dir / 'motif_confidence.csv'}")
        print(f"Wrote {log_path}")
    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f"Robustness log saved -> {log_path}")


if __name__ == "__main__":
    main()
