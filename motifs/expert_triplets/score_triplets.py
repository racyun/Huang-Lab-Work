#!/usr/bin/env python3
"""Expert triplets, step 3 — export the annotations and score a model run.

Always writes annotations_export.csv: every live (not undone) answer, one row
each, joined to its triplet and to the three crops' image / cell / condition.
That flat file is the portable record to keep with the run outputs.

Then scores whatever it is given, on --split test by default:

  --motifs motifs.parquet        motif agreement: credit 1 / 0 / 1/3 per answer
                                 (see triplet_metrics.motif_credit), averaged per
                                 triplet, then over triplets.
                                   chance   = 1/3 exactly, and a permutation null
                                              that shuffles motif labels
                                   ceiling  = how often two experts pick the same
                                              odd crop on shared triplets
                                   normalised = (score - chance) / (ceiling - chance)
  --embeddings embeddings.parquet  embedding agreement: accuracy of "the odd one
                                 is the crop left out of the closest pair", and
                                 the odd-one-out NLL loss (uniform = 1.0986), with
                                 the temperature fitted on TRAIN answers.
  --weights w.npy                optional per-dimension weights applied to the
                                 embedding first (expert-informed reweighting).

Intervals: the confidence sequence is valid however often you look while
annotation continues; the bootstrap interval is only valid for a single
analysis at a sample size fixed in advance. Unusable answers are excluded and
counted. Every run writes triplet_scores.json next to the database.

Usage:
    python score_triplets.py --data-dir <root>/expert_triplets \
        --motifs <root>/stage4/motifs.parquet \
        --embeddings <root>/stage3/embeddings.parquet
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from triplet_metrics import (CHANCE, UNIFORM_NLL, bootstrap_ci,  # noqa: E402
                             confidence_sequence, embedding_odd, fit_tau,
                             motif_credit, odd_one_out_nll, pair_sq_dists)

KEY = ["condition", "image_id", "cell_id"]


def load(data_dir: Path) -> pd.DataFrame:
    """Live answers joined to their triplet and crops. Reads either the Google
    Sheet export (triplets.csv, annotations.csv, retractions.csv, crops.csv
    from "Triplets > Export for scoring") or a local annotations.sqlite."""
    if (data_dir / "annotations.csv").exists():
        read = lambda n: pd.read_csv(data_dir / f"{n}.csv", dtype=str,
                                     keep_default_na=False)
        a, t, r = read("annotations"), read("triplets"), read("retractions")
        a = a[~a.annotation_id.isin(r.annotation_id)]
        ann = a.merge(t[["triplet_id", "split", "shared", "created_for",
                         "crop_1", "crop_2", "crop_3"]], on="triplet_id")
        ann = ann.sort_values("answered_at").reset_index(drop=True)
        ann["shared"] = ann["shared"].astype(int)
        ann["odd_crop"] = ann["odd_crop"].replace("", None)
    else:
        db = sqlite3.connect(f"file:{data_dir / 'annotations.sqlite'}?mode=ro", uri=True)
        ann = pd.read_sql_query(
            "SELECT a.*, t.split, t.shared, t.created_for, t.crop_1, t.crop_2, t.crop_3 "
            "FROM annotations a JOIN triplets t ON t.triplet_id = a.triplet_id "
            "LEFT JOIN retractions r ON r.annotation_id = a.annotation_id "
            "WHERE r.annotation_id IS NULL ORDER BY a.answered_at", db)
        db.close()
    crops = pd.read_csv(data_dir / "crops.csv", dtype={"image_id": str})
    meta = crops.set_index("crop_id")[["condition", "image_id", "cell_id"]]
    for i in (1, 2, 3):
        m = meta.add_suffix(f"_{i}")
        ann = ann.join(m, on=f"crop_{i}")
    ann["odd_index"] = [
        (None if pd.isna(o) else [r.crop_1, r.crop_2, r.crop_3].index(o))
        for r, o in zip(ann.itertuples(), ann["odd_crop"])]
    return ann


def lookup(ann: pd.DataFrame, table: pd.DataFrame, cols: list) -> np.ndarray:
    """(n, 3, len(cols)) values for each answer's three crops (crop_1..3 order);
    NaN where a crop's cell is missing from `table`."""
    t = table.set_index(KEY)[cols]
    out = np.full((len(ann), 3, len(cols)), np.nan)
    for i in (1, 2, 3):
        idx = pd.MultiIndex.from_arrays(
            [ann[f"condition_{i}"], ann[f"image_id_{i}"].astype(str),
             ann[f"cell_id_{i}"].astype(int)])
        out[:, i - 1, :] = t.reindex(idx).to_numpy(dtype=float)
    return out


def per_triplet(ann: pd.DataFrame, values: np.ndarray) -> pd.Series:
    """Average answer-level values within a triplet (shared triplets have one
    answer per expert), so each test triplet is one independent observation,
    kept in the order its first answer arrived."""
    s = pd.Series(values, index=ann["triplet_id"].to_numpy())
    first = ann.groupby("triplet_id")["answered_at"].min().sort_values()
    return s.groupby(level=0).mean().reindex(first.index)


def interval_text(x) -> dict:
    lo, hi = confidence_sequence(np.asarray(x))
    blo, bhi = bootstrap_ci(np.asarray(x))
    return {"mean": float(np.mean(x)), "n_triplets": int(len(x)),
            "anytime_95": [round(lo, 4), round(hi, 4)],
            "bootstrap_95_fixed_n_only": [round(blo, 4), round(bhi, 4)]}


def human_ceiling(test: pd.DataFrame) -> dict:
    """Pairwise agreement on the odd crop, over shared triplets two or more
    experts answered."""
    agree = []
    for _, g in test[test.shared == 1].groupby("triplet_id"):
        odds = g.drop_duplicates("annotator")["odd_crop"].tolist()
        if len(odds) < 2:
            continue
        pairs = list(combinations(odds, 2))
        agree.append(np.mean([a == b for a, b in pairs]))
    if not agree:
        return {"n_shared_triplets": 0}
    return {"n_shared_triplets": len(agree), **interval_text(agree)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Export and score expert triplets.")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--motifs", type=Path, default=None)
    ap.add_argument("--embeddings", type=Path, default=None)
    ap.add_argument("--weights", type=Path, default=None,
                    help=".npy of per-dimension embedding weights")
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args()

    ann = load(args.data_dir)
    export = args.data_dir / "annotations_export.csv"
    ann.drop(columns=["odd_index"]).to_csv(export, index=False)
    print(f"Exported {len(ann)} live answer(s) -> {export}")
    if ann.empty:
        return

    report: dict = {"generated_at": datetime.now(timezone.utc).isoformat(
        timespec="seconds"), "split": args.split}
    sub = ann[ann.split == args.split]
    usable = sub[sub.choice != "unusable"].reset_index(drop=True)
    report["answers"] = {
        "total": int(len(sub)), "usable": int(len(usable)),
        "unusable": int((sub.choice == "unusable").sum()),
        "by_annotator": sub.groupby("annotator").size().to_dict(),
        "triplets": int(usable.triplet_id.nunique())}
    # Individual crops flagged unusable (all splits): a QC list to trace back
    # to segmentation or imaging problems. The site stops serving them.
    if "unusable_crops" in ann.columns:
        flags = (ann["unusable_crops"].fillna("").astype(str).str.split(";")
                 .explode().loc[lambda s: s != ""])
        if len(flags):
            crops_meta = pd.read_csv(args.data_dir / "crops.csv", dtype={"image_id": str})
            fl = (flags.value_counts().rename("times_flagged").rename_axis("crop_id")
                  .reset_index().merge(crops_meta, on="crop_id", how="left"))
            fl.to_csv(args.data_dir / "flagged_crops.csv", index=False)
            report["flagged_crops"] = int(len(fl))
            print(f"{len(fl)} crop(s) flagged unusable -> flagged_crops.csv "
                  f"(from {fl.image_id.nunique()} image(s))")
    print(f"\n{args.split.upper()} answers: {len(usable)} usable, "
          f"{report['answers']['unusable']} unusable, over "
          f"{report['answers']['triplets']} triplet(s); "
          f"by annotator {report['answers']['by_annotator']}")
    if usable.empty:
        print("nothing to score yet")
        return

    ceiling = human_ceiling(usable) if args.split == "test" else {}
    if ceiling:
        report["human_ceiling"] = ceiling
        if ceiling.get("n_shared_triplets"):
            print(f"\nHuman ceiling (experts agree on the odd one): "
                  f"{ceiling['mean']:.3f}  anytime 95% {ceiling['anytime_95']}  "
                  f"from {ceiling['n_shared_triplets']} shared triplet(s)")
            if ceiling["n_shared_triplets"] < 30:
                print("  [caution] fewer than 30 shared triplets: the ceiling is "
                      "too uncertain to normalise against yet")
        else:
            print("\nHuman ceiling: no shared triplet answered by two experts yet")

    odd = usable["odd_index"].astype(int).to_numpy()

    # ---------------- motifs ----------------
    if args.motifs:
        mdf = pd.read_parquet(args.motifs)
        mdf["image_id"] = mdf["image_id"].astype(str)
        mot = lookup(usable, mdf, ["motif"])[:, :, 0]
        ok = ~np.isnan(mot).any(axis=1)
        mot_i = mot[ok].astype(int)
        credit = np.array([motif_credit(tuple(m), o) for m, o in zip(mot_i, odd[ok])])
        x = per_triplet(usable[ok], credit)
        res = interval_text(x)
        # permutation null: shuffle motif labels across the crops in this set
        flat = mot_i.ravel().copy()
        rng = np.random.default_rng(0)
        null = np.empty(args.n_perm)
        for p in range(args.n_perm):
            sh = rng.permutation(flat).reshape(-1, 3)
            null[p] = per_triplet(usable[ok], np.array(
                [motif_credit(tuple(m), o) for m, o in zip(sh, odd[ok])])).mean()
        res.update({"chance": CHANCE, "permutation_mean": float(null.mean()),
                    "permutation_p": float((null >= res["mean"]).mean()),
                    "missing_cells": int((~ok).sum()),
                    "decisive_share": float(np.mean(credit != CHANCE))})
        if ceiling.get("n_shared_triplets", 0) >= 30:
            span = ceiling["mean"] - CHANCE
            res["normalised"] = (float((res["mean"] - CHANCE) / span)
                                 if span > 0 else None)
        report["motifs"] = {"file": str(args.motifs), **res}
        print(f"\nMOTIF AGREEMENT ({args.motifs.name})")
        print(f"  score            : {res['mean']:.3f}   (chance 0.333, "
              f"permutation {res['permutation_mean']:.3f}, p = {res['permutation_p']:.4g})")
        print(f"  anytime 95%      : {res['anytime_95']}")
        print(f"  decisive answers : {100 * res['decisive_share']:.0f}% "
              f"(the rest had all-same / all-different motifs: 1/3 credit)")
        if res.get("normalised") is not None:
            print(f"  normalised       : {res['normalised']:.2f} of the way from "
                  f"chance to expert agreement")
        if res["missing_cells"]:
            print(f"  [warn] {res['missing_cells']} answer(s) skipped: a crop's cell "
                  f"is not in this motif file (re-segmented data?)")

    # ---------------- embeddings ----------------
    if args.embeddings:
        emb = pd.read_parquet(args.embeddings)
        emb["image_id"] = emb["image_id"].astype(str)
        cols = [c for c in emb.columns if c.startswith("emb_")]
        w = np.load(args.weights) if args.weights else None

        def dists(frame):
            e = lookup(frame, emb, cols)
            good = ~np.isnan(e).any(axis=(1, 2))
            return pair_sq_dists(e[good], w), good

        d2, good = dists(usable)
        model_odd = embedding_odd(d2)
        acc = per_triplet(usable[good], (model_odd == odd[good]).astype(float))
        res = {"accuracy": interval_text(acc), "chance": CHANCE}
        train = ann[(ann.split == "train") & (ann.choice != "unusable")].reset_index(drop=True)
        if args.split == "test" and len(train) >= 30:
            d2_tr, g_tr = dists(train)
            tau = fit_tau(d2_tr, train["odd_index"].astype(int).to_numpy()[g_tr])
            tau_src = f"fitted on {int(g_tr.sum())} train answers"
        else:
            tau = float(np.median(d2)) or 1.0
            tau_src = "median heuristic (fewer than 30 train answers)"
        res.update({"nll": odd_one_out_nll(d2, odd[good], tau),
                    "uniform_nll": UNIFORM_NLL, "tau": tau, "tau_source": tau_src,
                    "weights": str(args.weights) if args.weights else None})
        report["embeddings"] = {"file": str(args.embeddings), **res}
        print(f"\nEMBEDDING AGREEMENT ({args.embeddings.name}"
              f"{', weighted' if w is not None else ''})")
        print(f"  odd-one-out accuracy : {res['accuracy']['mean']:.3f}   "
              f"(chance 0.333)  anytime 95% {res['accuracy']['anytime_95']}")
        print(f"  odd-one-out loss     : {res['nll']:.4f}   (uniform guessing "
              f"{UNIFORM_NLL:.4f}; lower is better)  tau {tau:.3g}, {tau_src}")

    out = args.data_dir / "triplet_scores.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
