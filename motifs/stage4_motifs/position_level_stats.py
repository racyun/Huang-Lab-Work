#!/usr/bin/env python3
"""Motif-frequency statistics at the right replicate unit.

Tiles are not independent samples. Each ``image_id`` looks like
``xy001_focusstack_r1c1``: ``xy###`` is an imaging position and ``r#c#`` indexes
a 3x3 grid of crops cut out of that one field of view. Nine tiles from the same
position therefore share optics, focus, illumination and in large part the same
cells' neighbourhoods. Treating them as ~1000 independent replicates inflates
every t-statistic; the honest unit is the position, which drops n to ~115 per
condition and still leaves ample power.

This script aggregates motif fractions to (condition, position) and reports:

  * per-condition means with position-level SEM
  * the within-date contrasts, which are the only ones the design supports
  * the same-biology date contrast, as a direct measure of the batch effect

It also prints, for every stiffness, which dates it was imaged on, so a reader
can see immediately which comparisons are confounded.

Usage
-----
    python position_level_stats.py \
        --motifs ~/cellpose_work/v5/gmm/motifs.parquet \
        --out-dir ~/cellpose_work/v5/gmm
"""
from __future__ import annotations

import argparse
import itertools
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

POSITION_RE = re.compile(r"^(?P<pos>xy\d+)_.*?(?:r(?P<r>\d+)c(?P<c>\d+))?$")


class _Tee:
    """Mirror stdout into a log file next to the other outputs."""

    def __init__(self, path: Path):
        self.stream = sys.__stdout__
        self.fh = open(path, "w")

    def write(self, s):  # noqa: D102
        self.stream.write(s)
        self.fh.write(s)

    def flush(self):  # noqa: D102
        self.stream.flush()
        self.fh.flush()

    def close(self):  # noqa: D102
        self.fh.close()


def _default_local_root() -> Path:
    default = Path("/teamspace/studios/this_studio/cellpose_work")
    if "CELLPOSE_LOCAL_ROOT" in os.environ:
        return Path(os.environ["CELLPOSE_LOCAL_ROOT"]).expanduser()
    if default.parent.parent.exists():
        return default
    return Path.home() / "cellpose_work"


def parse_position(image_id: str) -> str:
    """'xy001_focusstack_r1c1' -> 'xy001'. Falls back to the whole id."""
    m = POSITION_RE.match(image_id)
    return m.group("pos") if m and m.group("pos") else image_id


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = out["condition"].str[:6]
    out["stiffness"] = out["condition"].str.split("_").str[1:].str.join("_")
    out["position"] = out["image_id"].map(parse_position)
    return out


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Welch t-test plus Hedges' g. Returns (diff, t, p); g printed by caller."""
    from scipy import stats

    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(a.mean() - b.mean()), float(t), float(p)


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp < 1e-12:
        return float("nan")
    d = (a.mean() - b.mean()) / sp
    return float(d * (1 - 3 / (4 * (na + nb) - 9)))


def fractions_per_position(df: pd.DataFrame, motif: int) -> pd.DataFrame:
    """Fraction of cells in `motif`, one row per (condition, position)."""
    g = df.groupby(["condition", "date", "stiffness", "position"])
    out = g["motif"].agg(n_cells="size",
                         frac=lambda s: float((s == motif).mean()))
    return out.reset_index()


def design_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s, g in df.groupby("stiffness"):
        dates = sorted(g["date"].unique())
        rows.append({"stiffness": s, "n_dates": len(dates),
                     "dates": ",".join(dates),
                     "n_positions": g["position"].nunique(),
                     "n_tiles": g.groupby(["condition", "image_id"]).ngroups,
                     "n_cells": len(g)})
    return pd.DataFrame(rows).sort_values("stiffness").set_index("stiffness")


def main() -> None:
    root = _default_local_root()
    ap = argparse.ArgumentParser(
        description="Position-level motif statistics with honest replicate counts.")
    ap.add_argument("--motifs", type=Path,
                    default=root / "v5" / "gmm" / "motifs.parquet")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: alongside --motifs")
    ap.add_argument("--motif", type=int, default=None,
                    help="which motif's fraction to analyse (default: the "
                         "last one, i.e. the highest-numbered)")
    args = ap.parse_args()

    out_dir = args.out_dir or args.motifs.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tee = _Tee(out_dir / "position_level_stats_log.txt")
    sys.stdout = tee
    try:
        print(f"position_level_stats.py — {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"  motifs : {args.motifs}")

        df = annotate(pd.read_parquet(args.motifs))
        ks = sorted(df["motif"].unique())
        target = args.motif if args.motif is not None else int(ks[-1])
        if target not in ks:
            raise SystemExit(f"[error] motif {target} not present; have {ks}")

        print(f"\n{len(df):,} cells   {df.groupby(['condition','image_id']).ngroups:,} tiles"
              f"   {df.groupby(['condition','position']).ngroups:,} positions")
        print(f"  motifs present: {ks}   analysing fraction in motif {target}")
        tiles_per_pos = (df.groupby(["condition", "position"])["image_id"]
                         .nunique())
        print(f"  tiles per position: median {tiles_per_pos.median():.0f} "
              f"(min {tiles_per_pos.min()}, max {tiles_per_pos.max()})")

        print(f"\n{'=' * 74}\nDESIGN — which dates each stiffness was imaged on")
        dt = design_table(df)
        print(dt.to_string())
        n_dates_total = df["date"].nunique()
        single = dt[dt["n_dates"] == 1].index.tolist()
        if n_dates_total == 1:
            # Every cell comes from one session, so date cannot vary with
            # anything. A per-stiffness "only one date" count is trivially true
            # here and must not be read as a confound.
            print(f"\n  [ok] all cells come from a single session "
                  f"({df['date'].iloc[0]}). Date is constant,\n       so no "
                  "date/stiffness confound is possible within this subset — "
                  "every\n       contrast below is date-matched by construction.")
        elif single:
            print(f"\n  [!] {len(single)} of {len(dt)} stiffness levels were imaged on a "
                  "single date:")
            print(f"      {', '.join(single)}")
            print("      For these, stiffness and imaging session cannot be "
                  "separated. Any\n      comparison involving them is "
                  "descriptive, not causal.")
        dt.to_csv(out_dir / "design_table.csv")

        pos = fractions_per_position(df, target)
        pos.to_csv(out_dir / "position_fractions.csv", index=False)

        print(f"\n{'=' * 74}\nPER-CONDITION (position-level mean +/- SEM)")
        summ = pos.groupby(["date", "stiffness"])["frac"].agg(
            mean="mean", sd=lambda s: s.std(ddof=1), n="size")
        summ["sem"] = summ["sd"] / np.sqrt(summ["n"])
        print(summ[["mean", "sem", "n"]].round(4).to_string())
        summ.to_csv(out_dir / "position_summary.csv")

        # ---- within-date contrasts: the only ones the design licenses ----
        print(f"\n{'=' * 74}\nWITHIN-DATE STIFFNESS CONTRASTS (confound-free)")
        found = False
        for d, g in pos.groupby("date"):
            levels = sorted(g["stiffness"].unique())
            if len(levels) < 2:
                continue
            found = True
            print(f"\n  date {d} — stiffness levels {levels}")
            for a_s, b_s in itertools.combinations(levels, 2):
                a = g[g.stiffness == a_s]["frac"].to_numpy()
                b = g[g.stiffness == b_s]["frac"].to_numpy()
                diff, t, p = welch(a, b)
                print(f"    {a_s} ({len(a)} pos) vs {b_s} ({len(b)} pos): "
                      f"{diff:+.4f}   t={t:.2f}  p={p:.3g}  g={hedges_g(a, b):+.2f}")
        if not found:
            print("  none — no imaging session contains two stiffness levels, so "
                  "this dataset\n  supports no confound-free stiffness comparison "
                  "at all.")

        # ---- same-biology date contrasts: a direct batch-effect estimate ----
        print(f"\n{'=' * 74}\nSAME-STIFFNESS DATE CONTRASTS (pure batch effect)")
        found = False
        for s, g in pos.groupby("stiffness"):
            dates = sorted(g["date"].unique())
            if len(dates) < 2:
                continue
            found = True
            print(f"\n  stiffness {s} — imaged on {dates}")
            for d1, d2 in itertools.combinations(dates, 2):
                a = g[g.date == d1]["frac"].to_numpy()
                b = g[g.date == d2]["frac"].to_numpy()
                diff, t, p = welch(a, b)
                print(f"    {d1} ({len(a)} pos) vs {d2} ({len(b)} pos): "
                      f"{diff:+.4f}   t={t:.2f}  p={p:.3g}  g={hedges_g(a, b):+.2f}")
        if not found:
            print("  none — no stiffness was imaged twice, so the batch effect "
                  "cannot be\n  measured from this data at all.")

        clean = n_dates_total == 1
        label = ("FOR SCALE — stiffness range (single session: date-matched)"
                 if clean else
                 "FOR SCALE — fully confounded across-date range")
        print(f"\n{'=' * 74}\n{label}")
        by_s = pos.groupby("stiffness")["frac"].mean().sort_values()
        print(by_s.round(4).to_string())
        note = ("" if clean else "  (NOT interpretable as a stiffness effect)")
        print(f"  max - min = {by_s.max() - by_s.min():+.4f}{note}")

        print(f"\nWrote {out_dir / 'design_table.csv'}")
        print(f"Wrote {out_dir / 'position_fractions.csv'}")
        print(f"Wrote {out_dir / 'position_summary.csv'}")
    finally:
        sys.stdout = sys.__stdout__
        print(f"Log saved -> {out_dir / 'position_level_stats_log.txt'}")
        tee.close()


if __name__ == "__main__":
    main()
