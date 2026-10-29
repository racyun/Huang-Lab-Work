#!/usr/bin/env python3
"""Is the embedding usable at all? Run this before any clustering.

A contrastive encoder can fail in a way that leaves every downstream number
looking plausible. The v4 run is the worked example: 64-dimensional embeddings
whose first principal component carried 94% of the variance and whose first two
carried 99.86%. A nominally 64-d representation was really about 2-d. v1 was
worse: 99.5% in PC1 alone, effectively a single line.

Nothing downstream announces this. Clustering still returns clusters,
silhouette still returns a number, stability still looks high. What you get is
a tidy, reproducible partition of an artefact — and several months of
interpretation resting on it.

This is DIMENSIONAL COLLAPSE, a known failure mode of contrastive objectives:
the encoder satisfies the loss by mapping everything onto a low-dimensional
subspace instead of learning a rich representation. The checks here are the
cheap ones that catch it:

  effective rank      participation ratio of the eigenvalue spectrum, i.e. how
                      many dimensions the variance is really spread over. Near
                      the nominal width is healthy; near 1 is a collapsed line.
  variance profile    how many components are needed for 99 / 99.9% of variance
  magnitude           embeddings should be O(1). Values of 1e6 mean the weights
                      diverged, not that the features are large.
  condition number    ratio of largest to smallest singular value; huge values
                      mean a near-singular covariance, which is also why a
                      Gaussian mixture refuses to fit.

Usage:
    python check_embedding_health.py --embeddings <path>
    python check_embedding_health.py --embeddings <a> --compare <b>
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _default_local_root() -> Path:
    if "CELLPOSE_LOCAL_ROOT" in os.environ:
        return Path(os.environ["CELLPOSE_LOCAL_ROOT"]).expanduser()
    return Path.home() / "cellpose_work"


def effective_rank(ev: np.ndarray) -> float:
    """Participation ratio: (sum l)^2 / sum(l^2).

    Equals d when variance is spread evenly over d dimensions and 1 when a
    single dimension carries everything, so it reads directly as "how many
    dimensions is this embedding really using".
    """
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def report(path: Path, n_sample: int, seed: int) -> dict:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    cols = [c for c in df.columns if c.startswith("emb_")]
    X = df[cols].to_numpy(dtype=np.float64)
    d = len(cols)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n_sample, len(X)), replace=False)
    S = X[idx] - X[idx].mean(0)
    sv = np.linalg.svd(S, compute_uv=False)
    ev = sv ** 2
    frac = ev / ev.sum()
    cum = np.cumsum(frac)
    eff = effective_rank(ev)

    print(f"\n{'=' * 70}\n{path}\n{'=' * 70}")
    print(f"  cells {len(X):,}   nominal dimensions {d}")
    print(f"  |x| max {np.abs(X).max():.3g}   std {X.std():.3g}   "
          f"NaN {int(np.isnan(X).sum())}")
    print(f"\n  PC1 {frac[0]:.4f}   PC1-2 {cum[1]:.4f}   PC1-5 {cum[4]:.4f}")
    print(f"  dimensions for 99% of variance   : {int(np.searchsorted(cum, 0.99)) + 1}")
    print(f"  dimensions for 99.9% of variance : {int(np.searchsorted(cum, 0.999)) + 1}")
    print(f"  EFFECTIVE RANK                   : {eff:.2f} of {d}  "
          f"({100 * eff / d:.1f}% of the space)")
    print(f"  condition number                 : {sv[0] / max(sv[-1], 1e-300):.3g}")

    flags = []
    if eff < 0.05 * d:
        flags.append("COLLAPSED — the encoder is using almost none of its width. "
                     "Clustering this measures the collapse, not the biology.")
    elif eff < 0.25 * d:
        flags.append("DEGRADED — most dimensions carry no variance; treat "
                     "downstream structure with suspicion.")
    if np.abs(X).max() > 1e3:
        flags.append(f"DIVERGED — values reach {np.abs(X).max():.3g}. Healthy "
                     "embeddings are O(1); this indicates the weights blew up "
                     "during training.")
    if sv[0] / max(sv[-1], 1e-300) > 1e5:
        flags.append("NEAR-SINGULAR — a Gaussian mixture cannot fit this without "
                     "a large reg_covar, and distances are dominated by one axis.")
    print()
    if flags:
        for f in flags:
            print(f"  [!] {f}")
    else:
        print("  [ok] no collapse detected; the embedding looks usable")
    return {"path": str(path), "dims": d, "effective_rank": eff,
            "pc1": float(frac[0]), "max_abs": float(np.abs(X).max()),
            "dims_99": int(np.searchsorted(cum, 0.99)) + 1, "flags": flags}


def main() -> None:
    root = _default_local_root()
    ap = argparse.ArgumentParser(description="Diagnose embedding collapse.")
    ap.add_argument("--embeddings", type=Path,
                    default=root / "v4" / "stage3" / "embeddings.parquet")
    ap.add_argument("--compare", type=Path, default=None,
                    help="a second embedding to report alongside")
    ap.add_argument("--n-sample", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = [report(args.embeddings, args.n_sample, args.seed)]
    if args.compare:
        out.append(report(args.compare, args.n_sample, args.seed))
    if len(out) == 2:
        print(f"\n{'=' * 70}\nCOMPARISON")
        for r in out:
            print(f"  effective rank {r['effective_rank']:6.2f} / {r['dims']}   "
                  f"PC1 {r['pc1']:.3f}   {Path(r['path']).parent.name}")
    print(f"\n{'=' * 70}")
    if any(r["flags"] for r in out):
        print("Fix the encoder before interpreting any clustering result.")


if __name__ == "__main__":
    main()
