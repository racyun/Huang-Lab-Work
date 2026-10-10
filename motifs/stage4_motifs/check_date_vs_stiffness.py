#!/usr/bin/env python3
"""Do motif frequencies track imaging DATE or substrate STIFFNESS?

500 kPa was imaged on two dates. Same substrate, different day, so any
difference in motif composition between those two sets of images is batch
(exposure, staining, plating), not biology. That gives a yardstick: a motif
readout is only interpretable if the stiffness effect is clearly bigger than
the date effect.

Unit of analysis is the IMAGE, not the cell. Each image becomes one vector of
motif proportions; with ~1.27 M cells a per-cell test calls everything
significant.

Contrasts:
  date       same stiffness, different dates (every stiffness imaged on >= 2
             dates, every pair of its dates)
  stiffness  softest vs stiffest kPa substrate, pooled over dates. TC plastic
             is excluded: it differs from the gels in chemistry as well as
             stiffness. If the two ends were imaged on different dates this
             contrast also carries batch, so it is flagged as confounded.
  within-date stiffness pairs (same date, different stiffness) are reported
             too: they are the cleanest stiffness contrasts, if any exist.

Effect size per contrast = total variation distance (TVD) between the two
groups' mean composition vectors (0 = identical mix of motifs, 1 = no motif in
common). Significance = permutation test shuffling group labels across images.

Verdict: PASS if the stiffness contrast is significant (permutation p < 0.05)
and every date TVD is smaller than the stiffness TVD.

Outputs (in --out-dir, default: next to motifs.parquet):
    date_check.json           all contrasts + verdict (read by scorecard.py)
    date_check_per_motif.csv  per-motif proportion difference for each contrast

Usage:
    python check_date_vs_stiffness.py --motifs <stage4>/motifs.parquet
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def parse_condition(condition: str) -> tuple[str, float, str]:
    """'260516_500kPa' -> ('260516', 500.0, '500 kPa'). TC plastic -> inf.

    Same rules as plot_motif_frequency.parse_stiffness, kept local so this file
    has no matplotlib dependency.
    """
    s = str(condition)
    date = s.partition("_")[0]
    m = re.search(r"(\d+(?:\.\d+)?)\s*kPa", s, flags=re.I)
    if m:
        return date, float(m.group(1)), f"{m.group(1)} kPa"
    if "tc" in s.lower():
        return date, float("inf"), "TC plastic"
    return date, float("nan"), s


def per_image_proportions(mdf: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per image: motif proportions, plus a meta frame (date, stiffness)."""
    n_motifs = int(mdf["motif"].max()) + 1
    counts = (mdf.groupby(["condition", "image_id", "motif"]).size()
              .unstack(fill_value=0)
              .reindex(columns=range(n_motifs), fill_value=0))
    props = counts.div(counts.sum(axis=1), axis=0)
    parsed = [parse_condition(c) for c in props.index.get_level_values("condition")]
    meta = pd.DataFrame(parsed, columns=["date", "kpa", "stiffness"], index=props.index)
    return props, meta


def tvd(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.abs(a.mean(axis=0) - b.mean(axis=0)).sum())


def contrast(props: pd.DataFrame, mask_a, mask_b, label_a: str, label_b: str,
             kind: str, n_perm: int, rng: np.random.Generator,
             confounded: bool = False) -> tuple[dict, pd.DataFrame]:
    A = props[np.asarray(mask_a)].to_numpy()
    B = props[np.asarray(mask_b)].to_numpy()
    observed = tvd(A, B)

    pooled = np.vstack([A, B])
    n_a = len(A)
    hits = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(pooled))
        if tvd(pooled[idx[:n_a]], pooled[idx[n_a:]]) >= observed:
            hits += 1
    p = (hits + 1) / (n_perm + 1)

    rows = []
    for m in range(props.shape[1]):
        if np.ptp(np.concatenate([A[:, m], B[:, m]])) == 0:
            pm = 1.0  # motif absent (or constant) everywhere: nothing to test
        else:
            pm = float(mannwhitneyu(A[:, m], B[:, m]).pvalue)
        rows.append({"kind": kind, "a": label_a, "b": label_b, "motif": m,
                     "mean_a": float(A[:, m].mean()), "mean_b": float(B[:, m].mean()),
                     "diff": float(A[:, m].mean() - B[:, m].mean()),
                     "mannwhitney_p": pm})

    res = {"kind": kind, "a": label_a, "b": label_b,
           "n_images_a": int(len(A)), "n_images_b": int(len(B)),
           "tvd": observed, "perm_p": p, "confounded_with_date": confounded}
    return res, pd.DataFrame(rows)


def run_date_check(mdf: pd.DataFrame, n_perm: int = 2000, seed: int = 0,
                   alpha: float = 0.05) -> dict:
    """All date and stiffness contrasts plus the verdict. Used by scorecard.py."""
    rng = np.random.default_rng(seed)
    props, meta = per_image_proportions(mdf)
    results, per_motif = [], []

    # ---- date contrasts: same stiffness, different dates ----
    for stiff, sub in meta.groupby("stiffness"):
        dates = sorted(sub["date"].unique())
        for d1, d2 in itertools.combinations(dates, 2):
            r, pm = contrast(props,
                             (meta["stiffness"] == stiff) & (meta["date"] == d1),
                             (meta["stiffness"] == stiff) & (meta["date"] == d2),
                             f"{stiff} ({d1})", f"{stiff} ({d2})",
                             "date", n_perm, rng)
            results.append(r); per_motif.append(pm)

    # ---- reference stiffness contrast: softest vs stiffest gel ----
    gels = meta[np.isfinite(meta["kpa"])]
    ref = None
    if gels["kpa"].nunique() >= 2:
        lo, hi = gels["kpa"].min(), gels["kpa"].max()
        lo_lab = gels.loc[gels["kpa"] == lo, "stiffness"].iloc[0]
        hi_lab = gels.loc[gels["kpa"] == hi, "stiffness"].iloc[0]
        shared = (set(gels.loc[gels["kpa"] == lo, "date"])
                  & set(gels.loc[gels["kpa"] == hi, "date"]))
        ref, pm = contrast(props, meta["kpa"] == lo, meta["kpa"] == hi,
                           lo_lab, hi_lab, "stiffness", n_perm, rng,
                           confounded=not shared)
        results.append(ref); per_motif.append(pm)

    # ---- within-date stiffness pairs (cleanest, if the design has any) ----
    for date, sub in meta.groupby("date"):
        stiffs = sorted(sub["stiffness"].unique())
        for s1, s2 in itertools.combinations(stiffs, 2):
            r, pm = contrast(props,
                             (meta["date"] == date) & (meta["stiffness"] == s1),
                             (meta["date"] == date) & (meta["stiffness"] == s2),
                             f"{s1} ({date})", f"{s2} ({date})",
                             "stiffness_within_date", n_perm, rng)
            results.append(r); per_motif.append(pm)

    date_res = [r for r in results if r["kind"] == "date"]
    if not date_res:
        verdict, passed = "NO REPEATS — no stiffness was imaged on two dates", None
    elif ref is None:
        verdict, passed = "NO STIFFNESS CONTRAST — fewer than two gel stiffnesses", None
    else:
        worst = max(r["tvd"] for r in date_res)
        ratio = worst / ref["tvd"] if ref["tvd"] > 0 else float("inf")
        real = ref["perm_p"] < alpha
        # a stiffness effect that is itself noise cannot be "bigger than batch"
        passed = bool(real and worst < ref["tvd"])
        verdict = (f"{'PASS' if passed else 'FAIL'} — largest date effect "
                   f"(TVD {worst:.3f}) is {ratio:.2f}x the stiffness effect "
                   f"(TVD {ref['tvd']:.3f}, p={ref['perm_p']:.4f})")
        if not real:
            verdict += f"; the stiffness effect is not significant (p >= {alpha})"
        if ref["confounded_with_date"]:
            verdict += ("; caution: the stiffness contrast spans different dates, "
                        "so it also contains batch")

    return {"contrasts": results, "passed": passed, "verdict": verdict,
            "n_images": int(len(props)), "n_perm": n_perm,
            "per_motif": pd.concat(per_motif, ignore_index=True)
                         if per_motif else pd.DataFrame()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Motif frequencies: date vs stiffness.")
    ap.add_argument("--motifs", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = args.out_dir or args.motifs.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    mdf = pd.read_parquet(args.motifs)
    res = run_date_check(mdf, args.n_perm, args.seed)

    print(f"{res['n_images']} images, {res['n_perm']} permutations per contrast\n")
    print(f"{'kind':<22} {'a':<24} {'b':<24} {'n_a':>4} {'n_b':>4} {'TVD':>6} {'p':>7}")
    for r in res["contrasts"]:
        flag = "  *batch-confounded" if r["confounded_with_date"] else ""
        print(f"{r['kind']:<22} {r['a']:<24} {r['b']:<24} {r['n_images_a']:>4} "
              f"{r['n_images_b']:>4} {r['tvd']:>6.3f} {r['perm_p']:>7.4f}{flag}")

    pm = res.pop("per_motif")
    if not pm.empty:
        print("\nPer-motif differences on the date contrasts (a - b):")
        print(pm[pm["kind"] == "date"].round(4).to_string(index=False))
        pm.to_csv(out_dir / "date_check_per_motif.csv", index=False)

    print(f"\n=> {res['verdict']}")
    (out_dir / "date_check.json").write_text(json.dumps(res, indent=2))
    print(f"\nWrote {out_dir / 'date_check.json'}")


if __name__ == "__main__":
    main()
