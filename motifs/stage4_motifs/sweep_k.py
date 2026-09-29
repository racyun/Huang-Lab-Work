#!/usr/bin/env python3
"""Is there a natural number of motifs, or is the embedding a continuum?

k = 6 was taken from the project plan, not derived from the data, which is the
obvious thing to be asked in a review. This sweeps k over a range and reports,
for each value, the three quantities that would reveal a natural number of
groups if one existed:

  silhouette   how separated the clusters are. A real cluster count shows up as
               a peak; a continuum decays smoothly as k rises.
  stability    ARI between partitions fit from different random seeds. A real
               structure is recovered identically every time; an arbitrary cut
               through a continuum wobbles.
  inertia      within-cluster spread, for the classic elbow. Reported as the
               drop from the previous k, since the raw curve always decreases.

A GAP STATISTIC is also computed: the same clustering is run on uniform random
data spanning the same bounding box, and the gap between the two inertia curves
is compared. If real data is no better separated than uniform noise at any k,
there is no discrete cluster structure to find -- which is a legitimate,
reportable result rather than a failure.

Runs on existing embeddings; no retraining.

Outputs:
    <out>/k_sweep.csv
    <out>/k_sweep.png
    <out>/k_sweep_log.txt

Usage:
    python sweep_k.py --embeddings <stage3>/embeddings.parquet \
        --out-dir <stage4> --k-min 3 --k-max 12
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class _Tee:
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


def main() -> None:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    root = _default_local_root()
    ap = argparse.ArgumentParser(description="Sweep k to justify the motif count.")
    ap.add_argument("--embeddings", type=Path,
                    default=root / "stage3" / "embeddings.parquet")
    ap.add_argument("--out-dir", type=Path, default=root / "stage4")
    ap.add_argument("--k-min", type=int, default=3)
    ap.add_argument("--k-max", type=int, default=12)
    ap.add_argument("--fit-cells", type=int, default=50000)
    ap.add_argument("--sil-cells", type=int, default=10000,
                    help="subsample for silhouette (it is O(n^2))")
    ap.add_argument("--n-seeds", type=int, default=3,
                    help="seeds per k for the stability estimate")
    ap.add_argument("--gap-refs", type=int, default=3,
                    help="uniform reference datasets for the gap statistic")
    ap.add_argument("--highlight", type=int, default=6,
                    help="k to mark on the figure as the current choice")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tee = _Tee(args.out_dir / "k_sweep_log.txt")
    sys.stdout = tee
    try:
        print(f"sweep_k.py — {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"  embeddings : {args.embeddings}")
        print(f"  k range    : {args.k_min}..{args.k_max}   fit-cells={args.fit_cells}"
              f"   seeds={args.n_seeds}   gap-refs={args.gap_refs}")

        edf = pd.read_parquet(args.embeddings)
        cols = [c for c in edf.columns if c.startswith("emb_")]
        X_all = edf[cols].to_numpy(dtype=np.float32)
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(X_all), size=min(args.fit_cells, len(X_all)),
                         replace=False)
        X = X_all[idx]
        sidx = rng.choice(len(X), size=min(args.sil_cells, len(X)), replace=False)
        print(f"\nLoaded {len(X_all):,} cells x {len(cols)} dims; "
              f"fitting on {len(X):,}")

        # uniform reference data spanning the same bounding box, for the gap
        lo, hi = X.min(0), X.max(0)
        refs = [rng.uniform(lo, hi, size=X.shape).astype(np.float32)
                for _ in range(args.gap_refs)]

        rows = []
        prev_inertia = None
        print(f"\n{'k':>3}{'silhouette':>13}{'stability(ARI)':>17}"
              f"{'inertia drop':>15}{'gap':>9}")
        for k in range(args.k_min, args.k_max + 1):
            labs = []
            inertias = []
            for s in range(args.n_seeds):
                km = KMeans(n_clusters=k, n_init=5, random_state=s).fit(X)
                labs.append(km.labels_)
                inertias.append(km.inertia_)
            inertia = float(np.mean(inertias))
            sil = float(silhouette_score(X[sidx], labs[0][sidx]))
            aris = [adjusted_rand_score(labs[i], labs[j])
                    for i in range(len(labs)) for j in range(i + 1, len(labs))]
            ari = float(np.mean(aris)) if aris else float("nan")

            ref_in = [KMeans(n_clusters=k, n_init=3, random_state=0)
                      .fit(r).inertia_ for r in refs]
            gap = float(np.mean(np.log(ref_in)) - np.log(inertia))

            drop = (float("nan") if prev_inertia is None
                    else 100 * (prev_inertia - inertia) / prev_inertia)
            prev_inertia = inertia
            rows.append({"k": k, "silhouette": round(sil, 4),
                         "stability_ari": round(ari, 4),
                         "inertia": round(inertia, 1),
                         "inertia_drop_pct": (None if np.isnan(drop)
                                              else round(drop, 2)),
                         "gap": round(gap, 4)})
            print(f"{k:>3}{sil:>13.4f}{ari:>17.4f}"
                  f"{('' if np.isnan(drop) else f'{drop:>14.1f}%')}"
                  f"{gap:>9.4f}")

        df = pd.DataFrame(rows)
        df.to_csv(args.out_dir / "k_sweep.csv", index=False)

        best_sil = int(df.loc[df["silhouette"].idxmax(), "k"])
        best_gap = int(df.loc[df["gap"].idxmax(), "k"])
        sil_range = float(df["silhouette"].max() - df["silhouette"].min())

        # A real cluster count shows up as an INTERIOR peak. A continuum peaks
        # at the smallest k tested and decays from there -- splitting a smooth
        # spread in two always looks reasonably separated, so a peak at k_min
        # is evidence against discrete structure, not for k_min.
        sil = df["silhouette"].to_numpy()
        interior_peak = best_sil not in (args.k_min, args.k_max)
        monotone_decay = bool(np.all(np.diff(sil) < 1e-3))
        flat = sil_range < 0.10

        print("\n" + "=" * 68)
        print("READING")
        print(f"  highest silhouette at k = {best_sil} "
              f"({df['silhouette'].max():.3f})")
        print(f"  highest gap        at k = {best_gap}")
        print(f"  silhouette spread across the sweep: {sil_range:.3f}")
        if flat or monotone_decay or best_sil == args.k_min:
            print("  => no interior optimum: silhouette is highest at the "
                  "smallest k and")
            print("     decays from there (or is flat). That is the CONTINUUM "
                  "signature --")
            print("     splitting a smooth spread in two always looks separated, so "
                  "this is")
            print("     evidence against discrete cluster structure rather than for "
                  "k = "
                  f"{args.k_min}.")
            print("     The motif count is a reporting choice, not a discovery; say "
                  "so.")
        elif interior_peak and best_sil == args.highlight:
            print(f"  => k = {args.highlight} is an interior silhouette optimum: "
                  "the chosen")
            print("     count is supported by the data.")
        elif interior_peak:
            print(f"  => interior optimum at k = {best_sil}, not the chosen "
                  f"k = {args.highlight}.")
            print("     Worth reporting both, or justifying the choice on")
            print("     interpretability rather than on separation.")
        else:
            print(f"  => optimum sits at the edge of the swept range; widen "
                  f"--k-min/--k-max.")
        if args.highlight in set(df["k"]):
            r = df[df["k"] == args.highlight].iloc[0]
            print(f"\n  at the current k = {args.highlight}: silhouette "
                  f"{r['silhouette']:.3f}, stability {r['stability_ari']:.3f}")
        print("=" * 68)

        # ---- figure ----
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        for ax, col, title, ylab in (
            (axes[0], "silhouette", "Separation", "silhouette"),
            (axes[1], "stability_ari", "Repeatability across seeds", "ARI"),
            (axes[2], "gap", "Gap vs uniform random data", "gap"),
        ):
            ax.plot(df["k"], df[col], "o-", lw=2, color="#2b6cb0")
            if args.highlight in set(df["k"]):
                v = df.loc[df["k"] == args.highlight, col].iloc[0]
                ax.axvline(args.highlight, color="crimson", ls="--", lw=1.2)
                ax.plot([args.highlight], [v], "o", color="crimson", ms=9,
                        label=f"current choice k={args.highlight}")
                ax.legend(fontsize=9, frameon=False)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("number of motifs (k)")
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylim(0, max(0.55, float(df["silhouette"].max()) * 1.2))
        fig.suptitle("How many motifs does the data actually support?",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(args.out_dir / "k_sweep.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nWrote {args.out_dir / 'k_sweep.csv'}")
        print(f"Wrote {args.out_dir / 'k_sweep.png'}")
    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f"k-sweep log saved -> {args.out_dir / 'k_sweep_log.txt'}")


if __name__ == "__main__":
    main()
