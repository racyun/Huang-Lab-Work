#!/usr/bin/env python3
"""Choose the number of motifs from the data instead of declaring it.

k = 6 came from the project plan, not from the embedding. This replaces both
halves of that decision:

  CLUSTERING   A Gaussian mixture instead of k-means. k-means represents a
               cluster as a point, so every cluster is implicitly a sphere of
               the same size; on gradient-like data it slices the gradient into
               equal Voronoi cells. (The v4 run's three middle motifs came out
               at 28.6 / 24.0 / 22.9 percent in monotonic feature order, which
               is what that artefact looks like.) A mixture component carries a
               full covariance, so it can be a long thin ridge or a small tight
               ball, and each cell gets a posterior probability rather than a
               hard label.

  CHOOSING k   Stability selection (Ben-Hur et al. 2002; the scheme CellCharter
               uses). Real structure is reproducible: perturb the problem and
               the same partition returns. Imposed structure is not. So fit
               every k many times under perturbation and keep the k whose
               partition recurs.

THE OUTPUT IS THE FINDING. The shape of the stability curve answers the prior
question of whether discrete motifs exist at all:

    one clear peak      genuine structure at that k
    several peaks       hierarchy; coarse and fine are both valid
    flat / declining    NO natural number of motifs — the embedding is a
                        continuum, and that is a result, not a failure

Two stability definitions are reported because they answer different questions:

  within-k    agreement among runs at the same k. "Is this k reproducible?"
  adjacent-k  agreement between runs at k and runs at k-1 / k+1, which is what
              CellCharter scores. "Is the answer robust to being asked a
              slightly different question?" If k is right, k+1 tends to just
              split one cluster and agreement stays high; on a continuum the
              whole partition reshuffles.

Two similarity metrics are reported because the usual one is biased. CellCharter
defaults to the Fowlkes-Mallows index, which is NOT adjusted for chance: with
few clusters two random partitions agree more often simply because there are
fewer ways to disagree, so FMI drifts up at low k and can pick k too small. ARI
is chance-corrected. If the two disagree, trust ARI and say so.

BIC is also reported as a third, independent criterion: it is a likelihood-based
penalty that knows nothing about stability, so agreement between them is real
evidence and disagreement is worth reporting.

Runs on an existing embeddings.parquet. No retraining, no GPU.

Outputs:
    <out>/k_stability.csv        per-k metrics
    <out>/k_stability.png        the curve (the figure for the paper)
    <out>/gmm_autok_log.txt      full transcript
    <out>/motifs.parquet         labels at the chosen k   (with --assign)
    <out>/motif_summary.csv      per-motif feature profile (with --master-table)
    <out>/motif_posteriors.npy   per-cell posteriors      (with --assign)

Usage:
    python gmm_autok.py --embeddings ~/cellpose_work/v4/stage3/embeddings.parquet \\
        --out-dir ~/cellpose_work/v4/gmm --k-min 2 --k-max 20
    python gmm_autok.py ... --assign --master-table ~/cellpose_work/master_table.csv
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

FEATURE_COLS = [
    "area_px", "elongation",
    "ch1_cellwise_mean_intensity",
    "ch2_cellwise_mean_membrane_intensity",
    "ch3_cellwise_mean_intensity",
    "endmt_score",
]


class _Tee:
    """Mirror console output into a log file."""

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


def load_embeddings(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    csv = path.with_suffix(".csv")
    if csv.exists():
        return pd.read_csv(csv)
    raise SystemExit(
        f"embeddings not found: {path}\n"
        f"Lightning is gone; pull them from Drive first:  ./motifs/fetch_from_drive.sh")


def _fit_one(X_fit, X_eval, k, seed, cov_type, reg):
    """Fit one GMM on a perturbed subsample, label the common evaluation set.

    Every run predicts on the SAME eval cells so that partitions from different
    runs - and from different k - are directly comparable. Perturbation comes
    from both the seed and the subsample, which is a stronger test than seed
    alone.

    A mixture can fail outright when a component collapses onto a handful of
    points and its covariance becomes singular. That happens most readily when
    k exceeds what the data supports, so the failure is itself a signal rather
    than a nuisance: the ridge is escalated a few times, and a fit that still
    will not converge is recorded as degenerate instead of killing the sweep.
    """
    from sklearn.mixture import GaussianMixture
    for attempt, r in enumerate((reg, reg * 100, reg * 10000)):
        try:
            gm = GaussianMixture(
                n_components=k, covariance_type=cov_type, random_state=seed,
                reg_covar=r, max_iter=200, n_init=1, init_params="k-means++",
            )
            gm.fit(X_fit)
            labels = gm.predict(X_eval)
            # A component holding almost nothing is not a cluster. Counting the
            # ones that carry cells shows whether the mixture needed all k.
            n_eff = int((np.bincount(labels, minlength=k) / len(labels) >= 0.01).sum())
            return labels, float(gm.bic(X_eval)), n_eff, bool(gm.converged_), attempt
        except ValueError:
            continue
    return None, np.nan, 0, False, 3


def sweep(X, k_values, n_runs, fit_cells, eval_cells, cov_type, reg, seed, jobs):
    """Fit every k many times; return labels on a common evaluation set."""
    from joblib import Parallel, delayed
    rng = np.random.default_rng(seed)
    eval_idx = rng.choice(len(X), size=min(eval_cells, len(X)), replace=False)
    X_eval = X[eval_idx]
    print(f"  evaluation set: {len(X_eval):,} cells (shared by every run)")
    print(f"  each run fits on {min(fit_cells, len(X)):,} independently drawn cells")

    jobs_list, meta = [], []
    for k in k_values:
        for r in range(n_runs):
            s = int(rng.integers(0, 2**31 - 1))
            fit_idx = rng.choice(len(X), size=min(fit_cells, len(X)), replace=False)
            jobs_list.append(delayed(_fit_one)(X[fit_idx], X_eval, k, s, cov_type, reg))
            meta.append((k, r))

    t0 = time.time()
    print(f"  fitting {len(jobs_list)} mixtures on {jobs} core(s)...")
    results = Parallel(n_jobs=jobs, verbose=5)(jobs_list)
    print(f"  done in {time.time() - t0:.0f}s")

    by_k: dict = {k: {"labels": [], "bic": [], "n_eff": [], "conv": [], "degen": 0}
                  for k in k_values}
    for (k, _), (lab, bic, n_eff, conv, attempt) in zip(meta, results):
        if lab is None:
            by_k[k]["degen"] += 1
            continue
        by_k[k]["labels"].append(lab)
        by_k[k]["bic"].append(bic)
        by_k[k]["n_eff"].append(n_eff)
        by_k[k]["conv"].append(conv)
        if attempt:
            by_k[k]["degen"] += 0      # recovered by a larger ridge; not degenerate
    degen = {k: v["degen"] for k, v in by_k.items() if v["degen"]}
    if degen:
        print(f"  [note] components collapsed at k = {sorted(degen)}; those fits were "
              f"dropped. Collapse at high k means the data does not support that many.")
    empty = [k for k, v in by_k.items() if len(v["labels"]) < 2]
    if empty:
        print(f"  [warn] too few usable fits at k = {empty}; excluded from the curve")
        for k in empty:
            by_k.pop(k)
    if not by_k:
        raise SystemExit(
            "Every mixture failed to fit. Try --covariance-type spherical or a "
            "larger --reg-covar (e.g. 1e-3).")
    return by_k, eval_idx


def _pairwise(a_list, b_list, fn, same):
    """Mean similarity over run pairs. `same` avoids comparing a run to itself."""
    vals = []
    for i, a in enumerate(a_list):
        for j, b in enumerate(b_list):
            if same and j <= i:
                continue
            vals.append(fn(a, b))
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (np.nan, np.nan)


def stability_table(by_k, k_values):
    k_values = [k for k in k_values if k in by_k]
    from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score
    rows = []
    for k in k_values:
        lab = by_k[k]["labels"]
        row = {"k": k}
        # within-k: is this k reproducible on its own terms?
        row["ari_within"], row["ari_within_std"] = _pairwise(
            lab, lab, adjusted_rand_score, True)
        row["fmi_within"], _ = _pairwise(lab, lab, fowlkes_mallows_score, True)
        # adjacent-k: CellCharter's criterion
        ari_adj, fmi_adj = [], []
        for nb in (k - 1, k + 1):
            if nb in by_k:
                ari_adj.append(_pairwise(lab, by_k[nb]["labels"],
                                         adjusted_rand_score, False)[0])
                fmi_adj.append(_pairwise(lab, by_k[nb]["labels"],
                                         fowlkes_mallows_score, False)[0])
        row["ari_adjacent"] = float(np.mean(ari_adj)) if ari_adj else np.nan
        row["fmi_adjacent"] = float(np.mean(fmi_adj)) if fmi_adj else np.nan
        row["bic"] = float(np.mean(by_k[k]["bic"]))
        row["n_effective"] = float(np.mean(by_k[k]["n_eff"]))
        row["converged"] = float(np.mean(by_k[k]["conv"]))
        row["n_fits"] = len(lab)
        row["n_degenerate"] = by_k[k]["degen"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("k")


def separation_test(X, labels, k, rng, max_cells=20000):
    """Is there a density valley between clusters, or solid density?

    Stability cannot answer this, and the synthetic check proves it: points
    strung along a line - a pure continuum with no clusters whatsoever - give
    within-k ARI of 0.936 at k=2, because every run cuts the line at the same
    place. A reproducible cut is not the same as a real boundary.

    So test the boundary directly. For each pair of clusters, project both onto
    the line joining their centres and look at the 1-D density profile between
    the two modes. Genuinely separate clusters show a valley: density falls away
    between them. A continuum that has merely been binned shows none - density
    stays high right through the "boundary", because the boundary is painted on
    solid tissue.

        valley_ratio = density at the lowest point between the modes
                       -------------------------------------------
                            the smaller of the two mode densities

    ~0.0  a deep valley: the clusters really are separate
    ~1.0  no valley at all: one continuous population, arbitrarily divided

    Only ADJACENT pairs may be tested. Two clusters at opposite ends of a
    gradient always look separated, because everything lying between them is
    excluded from the projection - a continuum split into three bins scores a
    valley of 2e-06 between bins 1 and 3 while scoring 0.98 between the
    neighbouring pairs. So each cluster is tested only against its nearest
    neighbour by centroid distance, which is where a boundary could plausibly
    be real.

    The summary is the MINIMUM ratio over those pairs: the single best-separated
    boundary in the partition. If even that one has no valley, nothing in the
    partition is a real group.
    """
    from scipy.ndimage import gaussian_filter1d
    if len(X) > max_cells:
        sel = rng.choice(len(X), max_cells, replace=False)
        X, labels = X[sel], labels[sel]
    present = [c for c in range(k) if (labels == c).sum() >= 50]
    if len(present) < 2:
        return {"valley_ratio_min": float("nan"), "n_pairs": 0}
    cent = {c: X[labels == c].mean(0) for c in present}
    # nearest-neighbour pairs only
    pairs = set()
    for i in present:
        j = min((c for c in present if c != i),
                key=lambda c: np.linalg.norm(cent[c] - cent[i]))
        pairs.add((min(i, j), max(i, j)))
    ratios = []
    for i, j in sorted(pairs):
        if True:
            mi, mj = labels == i, labels == j
            if mi.sum() < 50 or mj.sum() < 50:
                continue
            d = cent[j] - cent[i]
            n = np.linalg.norm(d)
            if n < 1e-9:
                continue
            d = d / n
            p = np.concatenate([X[mi] @ d, X[mj] @ d])
            lo, hi = np.percentile(p, [0.5, 99.5])
            if hi - lo < 1e-9:
                continue
            hist, edges = np.histogram(p, bins=64, range=(lo, hi), density=True)
            # Smooth before reading off modes, so sampling noise does not fake a
            # valley (or fill a real one in).
            hist = gaussian_filter1d(hist, 1.5)
            ci, cj = cent[i] @ d, cent[j] @ d
            bi = int(np.clip(np.searchsorted(edges, ci) - 1, 0, len(hist) - 1))
            bj = int(np.clip(np.searchsorted(edges, cj) - 1, 0, len(hist) - 1))
            a, b = min(bi, bj), max(bi, bj)
            if b - a < 3:
                continue
            peak = min(hist[a], hist[b])
            if peak < 1e-12:
                continue
            ratios.append(min(1.0, float(hist[a + 1:b].min() / peak)))
    if not ratios:
        return {"valley_ratio_min": float("nan"), "n_pairs": 0}
    return {"valley_ratio_min": float(np.min(ratios)),
            "valley_ratio_median": float(np.median(ratios)),
            "n_pairs": len(ratios)}


def separation_at_k(X, k, cov_type, reg, seed, fit_cells, eval_cells):
    """Fit once at k and measure boundary separation on the cells it labelled.

    The fit and evaluation indices are drawn from ONE generator here so the
    labels and the cells they describe cannot drift apart.
    """
    rng = np.random.default_rng(seed + 9973)
    fit_idx = rng.choice(len(X), min(fit_cells, len(X)), replace=False)
    ev_idx = rng.choice(len(X), min(eval_cells, len(X)), replace=False)
    lab = _fit_one(X[fit_idx], X[ev_idx], k, seed, cov_type, reg)[0]
    if lab is None:
        return {"valley_ratio_min": float("nan"), "n_pairs": 0}
    return separation_test(X[ev_idx], lab, k, rng)


def verdict(tbl: pd.DataFrame, tol: float = 0.02) -> dict:
    """Decide between 'real structure at k*' and 'continuum'.

    Taking the single most stable k is wrong, and the synthetic check shows why:
    on five cleanly separated blobs, within-k ARI is 1.000 at BOTH k=4 and k=5,
    then falls away above 5. Merging two blobs is just as reproducible as
    separating them, so high stability only certifies k <= the true number. What
    identifies the true number is the LAST k that still achieves (near-)maximal
    stability, i.e. the point where the curve breaks.

    The call itself rests on the ABSOLUTE stability at that k, not on how
    peaked the curve is. Clean structure is reproducible outright (ARI near 1);
    a continuum sliced into bins reshuffles under perturbation and scores low
    everywhere, with no k rescuing it.
    """
    out: dict = {}
    for col in ("ari_within", "ari_adjacent"):
        s = tbl[col].dropna()
        if s.empty:
            continue
        peak = float(s.max())
        plateau = s[s >= peak - tol]
        k_star = int(plateau.index.max())        # last k still at the plateau
        after = s[s.index > k_star]
        drop = peak - float(after.max()) if len(after) else np.nan
        out[col] = {"best_k": k_star, "stability_at_best": float(s.loc[k_star]),
                    "peak": peak, "argmax_k": int(s.idxmax()),
                    "drop_after": None if np.isnan(drop) else float(drop),
                    "median": float(s.median())}

    lvl = out.get("ari_within", {}).get("stability_at_best", 0.0)
    k_star = out.get("ari_within", {}).get("best_k")
    drop = out.get("ari_within", {}).get("drop_after")
    out["stability_level"] = float(lvl)
    out["best_k"] = k_star
    if drop is not None and drop < 0.02 and lvl >= 0.65:
        out["range_warning"] = ("stability does not fall after this k, so --k-max "
                                "may be too low; extend the sweep")
    return out


def final_call(stability: float, valley: float, k_star) -> str:
    """Combine the two axes into one statement.

    They answer different questions and a partition needs BOTH to count as real
    groups:

        stability   does the same partition come back under perturbation?
        valley      is there a density gap where the boundary was drawn?

    The interesting cell of this table is stable-but-no-valley. That is a
    reproducible partition OF A CONTINUUM: the algorithm reliably cuts in the
    same place, but the place is not a natural edge. It is exactly the pattern
    the v4 run already shows (ARI 0.989 alongside silhouette 0.364), and it is
    a real finding rather than a failed clustering.
    """
    stable = stability >= 0.65
    sep = (not np.isnan(valley)) and valley <= 0.60
    if stable and sep:
        return (f"DISCRETE MOTIFS — k={k_star} is reproducible (ARI {stability:.3f}) "
                f"AND the clusters are separated by real density valleys "
                f"(best valley ratio {valley:.2f}). Report k={k_star} as a finding.")
    if stable and not sep:
        return (f"CONTINUUM, REPRODUCIBLY BINNED — the k={k_star} partition comes back "
                f"reliably (ARI {stability:.3f}), but density does not fall at the "
                f"boundaries (best valley ratio {valley:.2f}: ~1.0 means no gap). The "
                f"cuts are stable bin edges through one continuous population, not "
                f"discovered groups. Report the gradient and treat k as a reporting "
                f"choice — and say so explicitly rather than claiming k motifs.")
    if (not stable) and sep:
        return (f"UNSTABLE BUT SEPARATED — density valleys exist (ratio {valley:.2f}) "
                f"yet the partition does not reproduce (ARI {stability:.3f}). Usually "
                f"means the right k is outside the swept range, or the mixture is "
                f"mis-specified; try --covariance-type full and a wider sweep.")
    return (f"NO STRUCTURE — neither reproducible (ARI {stability:.3f}) nor separated "
            f"(valley ratio {valley:.2f}). The embedding does not contain discrete "
            f"neighbourhood types at any k in this range.")


def plot(tbl: pd.DataFrame, out_png: Path, chosen: int | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.plot(tbl.index, tbl["ari_within"], "o-", label="within-k (ARI)", color="#1f77b4")
    ax.plot(tbl.index, tbl["ari_adjacent"], "s--", label="adjacent-k (ARI)",
            color="#4c9fd4")
    ax.plot(tbl.index, tbl["fmi_within"], "^:", label="within-k (FMI, biased)",
            color="#aaaaaa")
    ax.set_xlabel("number of components (k)")
    ax.set_ylabel("stability")
    ax.set_title("Stability — a peak means real structure", fontsize=11)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    ax.plot(tbl.index, tbl["bic"], "o-", color="#d62728")
    ax.set_xlabel("number of components (k)")
    ax.set_ylabel("BIC (lower is better)")
    ax.set_title("BIC — independent criterion", fontsize=11)

    ax = axes[2]
    ax.plot(tbl.index, tbl["n_effective"], "o-", color="#2ca02c")
    ax.plot(tbl.index, tbl.index, "--", color="grey", lw=1, label="y = k")
    ax.set_xlabel("number of components (k)")
    ax.set_ylabel("components holding >= 1% of cells")
    ax.set_title("Did the mixture need all k?", fontsize=11)
    ax.legend(fontsize=8, frameon=False)

    for ax in axes:
        if chosen is not None:
            ax.axvline(chosen, color="black", lw=1, alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def assign_all(X, keys, k, cov_type, reg, seed, out_dir, master_table):
    """Fit a final mixture at the chosen k and label every cell.

    Writes motifs.parquet with the same schema cluster_motifs.py produces, so
    validate_motifs.py, robustness_motifs.py, plot_motif_maps.py and the triplet
    scorer all run on it unchanged.
    """
    from sklearn.mixture import GaussianMixture
    print(f"\nFitting the final {k}-component mixture on all {len(X):,} cells...")
    gm = GaussianMixture(n_components=k, covariance_type=cov_type,
                         random_state=seed, reg_covar=reg, max_iter=300,
                         n_init=3, init_params="k-means++").fit(X)
    post = np.empty((len(X), k), dtype=np.float32)
    for i in range(0, len(X), 100_000):
        post[i:i + 100_000] = gm.predict_proba(X[i:i + 100_000])
    motif = post.argmax(1).astype(np.int32)

    out = keys.copy()
    out["motif"] = motif
    out.to_parquet(out_dir / "motifs.parquet", index=False)
    np.save(out_dir / "motif_posteriors.npy", post)

    conf = post.max(1)
    print(f"  converged={gm.converged_}  mean posterior of assigned motif="
          f"{conf.mean():.3f}")
    print(f"  cells with posterior >= 0.9: {100 * (conf >= 0.9).mean():.1f}%"
          f"   >= 0.5: {100 * (conf >= 0.5).mean():.1f}%")
    sizes = np.bincount(motif, minlength=k)
    for m in range(k):
        print(f"    motif {m:>2}: {sizes[m]:>9,} cells ({100 * sizes[m] / len(motif):5.2f}%)")

    if master_table and Path(master_table).exists():
        mt = pd.read_csv(master_table)
        if len(mt) == len(out):
            cols = [c for c in FEATURE_COLS if c in mt.columns]
            prof = mt[cols].copy()
            prof["motif"] = motif
            summ = prof.groupby("motif").mean()
            summ.insert(0, "n_cells", prof.groupby("motif").size())
            summ.insert(1, "pct_of_cells", 100 * summ["n_cells"] / len(prof))
            summ.to_csv(out_dir / "motif_summary.csv")
            print("\nPer-motif feature profile:")
            print(summ.round(3).to_string())
        else:
            print(f"  [skip] master table has {len(mt)} rows, embeddings {len(out)}")
    return motif


def _cond(X: np.ndarray) -> float:
    """Condition number of the covariance spectrum."""
    ev = np.linalg.svd(X - X.mean(0), compute_uv=False) ** 2
    ev = ev[ev > 0]
    return float(ev[0] / ev[-1]) if len(ev) else float("nan")


def pca_truncate(X: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Project onto the top n principal components.

    Not whitened on purpose: whitening would rescale the near-zero trailing
    components up to parity with the leading ones, which amplifies noise into
    signal. Truncation alone is what fixes the conditioning.
    """
    from sklearn.decomposition import PCA

    n = min(n, X.shape[1])
    rng = np.random.default_rng(seed)
    sub = X if len(X) <= 200_000 else X[rng.choice(len(X), 200_000, replace=False)]
    p = PCA(n_components=n, random_state=seed).fit(sub)
    Xp = np.ascontiguousarray(p.transform(X), dtype=np.float32)
    kept = float(p.explained_variance_ratio_.sum())
    print(f"  PCA {X.shape[1]} -> {n} dims   variance kept {kept:.4f}")
    print(f"  condition number {_cond(X):.2e} -> {_cond(Xp):.2e}")
    if kept < 0.95:
        print(f"  [warn] only {kept:.1%} of variance kept — raise --pca")
    return Xp


def main() -> None:
    root = _default_local_root()
    ap = argparse.ArgumentParser(
        description="GMM clustering with stability-based selection of k.")
    ap.add_argument("--embeddings", type=Path,
                    default=root / "v4" / "stage3" / "embeddings.parquet")
    ap.add_argument("--out-dir", type=Path, default=root / "v4" / "gmm")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=20)
    ap.add_argument("--n-runs", type=int, default=10,
                    help="fits per k; more is a better stability estimate")
    ap.add_argument("--fit-cells", type=int, default=50000)
    ap.add_argument("--eval-cells", type=int, default=20000,
                    help="common cells every run labels, for comparing partitions")
    ap.add_argument("--covariance-type", default="diag",
                    choices=["full", "diag", "tied", "spherical"],
                    help="'full' is 2080 params/component in 64-d; 'diag' is 64 "
                         "and is the safer default at this sample size")
    ap.add_argument("--reg-covar", type=float, default=1e-5,
                    help="ridge added to covariance diagonals; raise if fits fail")
    ap.add_argument("--pca", type=int, default=0, metavar="N",
                    help="truncate to the top N principal components before "
                         "fitting. The encoder's 64 dims come from 7 input "
                         "features, so most carry no variance and leave the "
                         "covariance near-singular; N around the '99%% of "
                         "variance' figure from check_embedding_health.py "
                         "removes that without discarding signal. 0 = off.")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--assign", action="store_true",
                    help="after the sweep, fit at the chosen k and label every cell")
    ap.add_argument("--force-k", type=int, default=None,
                    help="skip selection and assign at this k (for comparisons)")
    ap.add_argument("--master-table", type=Path, default=root / "master_table.csv")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tee = _Tee(args.out_dir / "gmm_autok_log.txt")
    sys.stdout = tee
    try:
        print(f"gmm_autok.py — {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"  embeddings : {args.embeddings}")
        print(f"  out-dir    : {args.out_dir}")
        print(f"  k range    : {args.k_min}..{args.k_max}   runs/k: {args.n_runs}")
        print(f"  covariance : {args.covariance_type}   reg_covar={args.reg_covar}")

        edf = load_embeddings(args.embeddings)
        cols = [c for c in edf.columns if c.startswith("emb_")]
        X = np.ascontiguousarray(edf[cols].to_numpy(dtype=np.float32))
        keys = edf[["condition", "image_id", "cell_id"]].copy()
        print(f"\nLoaded {len(X):,} cells x {len(cols)} dims")

        if args.pca > 0:
            X = pca_truncate(X, args.pca, args.seed)

        k_values = list(range(args.k_min, args.k_max + 1))
        print(f"\nSweeping k = {k_values[0]}..{k_values[-1]}")
        by_k, _ = sweep(X, k_values, args.n_runs, args.fit_cells, args.eval_cells,
                        args.covariance_type, args.reg_covar, args.seed, args.jobs)

        tbl = stability_table(by_k, k_values)
        tbl.to_csv(args.out_dir / "k_stability.csv")
        print("\nPer-k metrics:")
        print(tbl.round(4).to_string())

        v = verdict(tbl)
        best_ari = v.get("ari_within", {}).get("best_k")
        best_adj = v.get("ari_adjacent", {}).get("best_k")
        fmi_best = int(tbl["fmi_within"].idxmax())

        print("\n" + "=" * 74)
        print("CHOOSING k")
        print(f"  most stable k (within-k, ARI)   : {best_ari}")
        print(f"  most stable k (adjacent-k, ARI) : {best_adj}")
        print(f"  most stable k (FMI, chance-biased): {fmi_best}")
        print(f"  lowest BIC at k                 : {int(tbl['bic'].idxmin())}")
        if fmi_best != best_ari:
            print("  [note] FMI and ARI disagree. FMI is not corrected for chance and")
            print("         drifts upward at low k — trust the ARI answer.")
        print(f"\n  stability at the chosen k: {v['stability_level']:.3f}")
        if "range_warning" in v:
            print(f"  [caution] {v['range_warning']}")

        # Second axis: stability says the cut reproduces; this says whether the
        # cut lands on a real gap. A continuum binned the same way every time
        # scores high on the first and ~1.0 on the second.
        k_star = int(args.force_k or v["best_k"])
        print(f"\n  testing boundaries at k={k_star} for density valleys...")
        sep = separation_at_k(X, k_star, args.covariance_type, args.reg_covar,
                              args.seed, args.fit_cells, args.eval_cells)
        print(f"  valley ratio (lowest over {sep['n_pairs']} adjacent pairs): "
              f"{sep['valley_ratio_min']:.3f}   [~0 = deep gap, ~1 = no gap at all]")
        v["separation"] = sep
        call = final_call(v["stability_level"], sep["valley_ratio_min"], k_star)
        v["verdict"] = call
        print(f"\n  {call}")
        print("=" * 74)

        chosen = args.force_k or best_ari
        plot(tbl, args.out_dir / "k_stability.png", chosen)
        print(f"\nWrote {args.out_dir / 'k_stability.csv'}")
        print(f"Wrote {args.out_dir / 'k_stability.png'}")

        report = {"k_range": [args.k_min, args.k_max], "n_runs": args.n_runs,
                  "covariance_type": args.covariance_type,
                  "best_k_ari_within": best_ari, "best_k_ari_adjacent": best_adj,
                  "best_k_fmi": fmi_best, "best_k_bic": int(tbl["bic"].idxmin()),
                  "selection": v}
        (args.out_dir / "gmm_autok_report.json").write_text(json.dumps(report, indent=2))

        if args.assign or args.force_k:
            if v["stability_level"] < 0.65 and not args.force_k:
                print("\n[warning] The curve is flat, so this k is a bin boundary "
                      "rather than a discovered group. Labels are still written, "
                      "but describe them as a chosen partition of a continuum.")
            assign_all(X, keys, int(chosen), args.covariance_type, args.reg_covar,
                       args.seed, args.out_dir, args.master_table)
            print(f"\nWrote {args.out_dir / 'motifs.parquet'}  (k={chosen})")
            print("Same schema as cluster_motifs.py — validate_motifs.py, "
                  "robustness_motifs.py and plot_motif_maps.py run on it unchanged.")
    finally:
        sys.stdout = tee.stdout
        tee.close()
        print(f"Log saved -> {args.out_dir / 'gmm_autok_log.txt'}")


if __name__ == "__main__":
    main()
