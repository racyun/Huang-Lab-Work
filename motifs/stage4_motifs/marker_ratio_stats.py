#!/usr/bin/env python3
"""Score candidate EndMT readouts on signal *and* on session nuisance.

Every readout tried so far in this project separated the substrate stiffnesses,
and every one of them also separated two imaging sessions that held identical
biology. A number that does both is not a measurement of stiffness. This script
makes that failure mode visible by running two contrasts side by side:

  SIGNAL  5kPa vs 500kPa on 260516. That session is the only one holding two
          stiffness levels, so the contrast is date-matched by construction and
          no session effect can enter it.

  BATCH   500kPa on 260516 vs 500kPa on 260522. Same substrate, same biology,
          different session. Whatever difference appears here is nuisance, and
          it bounds how much of any stiffness contrast we are entitled to
          believe.

The headline is |signal g| / |batch g|. At or below 1 the readout tracks the
imaging session at least as strongly as the biology; well above 1 it tracks the
biology.

The DAPI channel is the internal control. It stains DNA, so per nucleus it is
near-constant regardless of phenotype, and it therefore absorbs session
differences in exposure, staining efficiency and focus. Dividing a marker by
the same cell's DAPI cancels that common gain. It is not a cure-all: a
membrane-band measure such as VE-cadherin also varies with segmentation
geometry, which DAPI does not track, so it stays contaminated.

Raw intensities are reconstructed by inverting the Stage 2 z-scoring. That is
exact only when one mean/std pair was applied to the whole table, so the script
refuses to run against a date- or image-scoped normalization.

Usage
-----
    python marker_ratio_stats.py                    # defaults below
    python marker_ratio_stats.py --motifs ~/cellpose_work/v6/gmm_260516/motifs.parquet
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

TAGLN = "ch1_cellwise_mean_intensity"                   # mesenchymal marker
VECAD = "ch2_cellwise_mean_membrane_intensity"          # endothelial marker
DAPI = "ch3_cellwise_mean_intensity"                    # nuclear control
INTENSITY_COLS = [TAGLN, VECAD, DAPI]

# Channel names are the REVERSE of the tile's RGB plane order; see
# stage1_segmentation/check_channel_order.py. ch1 is TAGLN, ch2 VE-cadherin,
# ch3 DAPI.

SIGNAL_DATE = "260516"
BATCH_STIFFNESS = "500kPa"

MIN_DAPI = 1.0      # guard the ratio; the observed minimum is ~9 ADU


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


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                 / (na + nb - 2))
    if sp < 1e-12:
        return float("nan")
    d = (a.mean() - b.mean()) / sp
    return float(d * (1 - 3 / (4 * (na + nb) - 9)))


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    from scipy import stats

    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return float(t), float(p)


def load_raw_intensities(master_path: Path, stats_path: Path) -> pd.DataFrame:
    """Read the master table and undo the global z-scoring on the intensities."""
    stats = json.load(open(stats_path))
    scope = stats.get("zscore_scope") or "global"
    if scope != "global":
        raise SystemExit(
            f"[error] {stats_path.name} was written with --zscore-scope "
            f"'{scope}'. Per-group scaling is not invertible from the summary "
            "stats, so raw intensities cannot be recovered. Re-assemble with "
            "--zscore-scope global, or point --master at a globally scaled "
            "table.")

    cols = ["condition", "image_id", "cell_id", "endmt_score"] + INTENSITY_COLS
    df = pd.read_csv(master_path, usecols=cols)

    z = stats["zscore"]
    for col in INTENSITY_COLS:
        entry = z.get(col)
        if entry is None or "std" not in entry:
            raise SystemExit(f"[error] no global mean/std recorded for {col}")
        df[col] = df[col] * entry["std"] + entry["mean"]

    n_bad = int((df[DAPI] < MIN_DAPI).sum())
    if n_bad:
        print(f"  [note] dropped {n_bad} cell(s) with DAPI < {MIN_DAPI} to keep "
              "the ratio finite")
        df = df[df[DAPI] >= MIN_DAPI].copy()
    return df


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = out["condition"].str[:6]
    out["stiffness"] = out["condition"].str.split("_").str[1:].str.join("_")
    out["position"] = (out["image_id"].str.extract(r"^(xy\d+)")[0]
                       .fillna(out["image_id"]))
    out["tagln_per_dapi"] = out[TAGLN] / out[DAPI]
    out["vecad_per_dapi"] = out[VECAD] / out[DAPI]
    out["tagln_vs_vecad"] = out[TAGLN] / (out[TAGLN] + out[VECAD])
    return out


def attach_motif_fraction(df: pd.DataFrame, motifs: Path) -> list[str]:
    """Add one 0/1 column per motif, so motif frequency joins the comparison."""
    if not motifs.exists():
        print(f"  [note] {motifs} not found — skipping motif readouts")
        return []
    mot = pd.read_parquet(motifs)
    key = ["condition", "image_id", "cell_id"]
    missing = [k for k in key + ["motif"] if k not in mot.columns]
    if missing:
        print(f"  [note] {motifs.name} lacks {missing} — skipping motif readouts")
        return []
    before = len(df)
    merged = df.merge(mot[key + ["motif"]], on=key, how="left",
                      validate="one_to_one")
    assert len(merged) == before, "motif merge changed the row count"
    df["motif"] = merged["motif"].to_numpy()

    labelled = df["motif"].notna()
    dates = sorted(df.loc[labelled, "date"].unique())
    print(f"  motif labels cover {int(labelled.sum()):,} cells on date(s) "
          f"{dates}")
    if len(dates) < 2:
        print("       -> only one session is labelled, so motif readouts can be "
              "scored for\n          SIGNAL but NOT for BATCH. An untestable "
              "batch effect is not a clean one.")

    cols = []
    for k in sorted(df.loc[labelled, "motif"].unique()):
        col = f"motif{int(k)}_frac"
        df[col] = np.where(labelled, (df["motif"] == k).astype(float), np.nan)
        cols.append(col)
    return cols


def positions(df: pd.DataFrame, measures: list[str]) -> pd.DataFrame:
    return df.groupby(["date", "stiffness", "position"], as_index=False)[
        measures].mean()


def contrast(pos: pd.DataFrame, col: str, mask_a, mask_b):
    a = pos[mask_a][col].dropna().to_numpy()
    b = pos[mask_b][col].dropna().to_numpy()
    t, p = welch(a, b)
    return hedges_g(a, b), p, len(a), len(b)


def main() -> None:
    root = _default_local_root()
    ap = argparse.ArgumentParser(
        description="Signal-vs-batch scorecard for candidate EndMT readouts.")
    ap.add_argument("--master", type=Path, default=root / "master_table.csv")
    ap.add_argument("--norm-stats", type=Path,
                    default=root / "normalization_stats.json")
    ap.add_argument("--motifs", type=Path,
                    default=root / "v6" / "gmm_260516" / "motifs.parquet",
                    help="optional; adds per-motif fractions to the comparison")
    ap.add_argument("--out-dir", type=Path, default=root / "marker_ratios")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tee = _Tee(args.out_dir / "marker_ratio_stats_log.txt")
    sys.stdout = tee
    try:
        print(f"marker_ratio_stats.py — {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"  master : {args.master}")

        df = annotate(load_raw_intensities(args.master, args.norm_stats))
        motif_cols = attach_motif_fraction(df, args.motifs)

        measures = {
            "endmt_score (current)": "endmt_score",
            "TAGLN raw": TAGLN,
            "VE-cad raw": VECAD,
            "DAPI raw": DAPI,
            "TAGLN / DAPI": "tagln_per_dapi",
            "VE-cad / DAPI": "vecad_per_dapi",
            "TAGLN/(TAGLN+VE-cad)": "tagln_vs_vecad",
        }
        measures.update({f"v6 {c}": c for c in motif_cols})

        pos = positions(df, list(measures.values()))
        pos.to_csv(args.out_dir / "position_measures.csv", index=False)
        print(f"\n{len(df):,} cells -> {len(pos):,} positions")

        # ---- the two contrasts ----
        sig_a = ((pos.date == SIGNAL_DATE) & (pos.stiffness == "5kPa"))
        sig_b = ((pos.date == SIGNAL_DATE) & (pos.stiffness == BATCH_STIFFNESS))
        bat_dates = sorted(pos[pos.stiffness == BATCH_STIFFNESS]["date"].unique())
        if len(bat_dates) < 2:
            raise SystemExit(f"[error] {BATCH_STIFFNESS} appears on "
                             f"{bat_dates}; need two sessions for the batch test")
        d1, d2 = bat_dates[0], bat_dates[1]
        bat_a = ((pos.date == d1) & (pos.stiffness == BATCH_STIFFNESS))
        bat_b = ((pos.date == d2) & (pos.stiffness == BATCH_STIFFNESS))

        print("\n" + "=" * 88)
        print(f"SIGNAL  5kPa vs {BATCH_STIFFNESS}, both on {SIGNAL_DATE} "
              f"(date-matched, {int(sig_a.sum())} vs {int(sig_b.sum())} positions)")
        print(f"BATCH   {BATCH_STIFFNESS} on {d1} vs {d2} (same biology, "
              f"{int(bat_a.sum())} vs {int(bat_b.sum())} positions)")
        print("=" * 88)
        print(f"{'readout':<26}{'signal g':>10}{'signal p':>11}"
              f"{'batch g':>10}{'batch p':>11}{'ratio':>9}  verdict")
        print("-" * 88)

        rows = []
        for label, col in measures.items():
            sg, sp_, _, _ = contrast(pos, col, sig_a, sig_b)
            bg, bp, nba, nbb = contrast(pos, col, bat_a, bat_b)
            if not np.isfinite(bg):
                ratio = float("nan")
                verdict = "batch UNTESTABLE"
            else:
                ratio = abs(sg) / max(abs(bg), 1e-9)
                verdict = ("biology" if ratio >= 3
                           else "mixed" if ratio >= 1.5
                           else "SESSION-DOMINATED")
            rows.append({"readout": label, "signal_g": sg, "signal_p": sp_,
                         "batch_g": bg, "batch_p": bp, "ratio": ratio,
                         "verdict": verdict})
            r = "   n/a" if not np.isfinite(ratio) else f"{ratio:9.2f}"
            bgs = "     n/a" if not np.isfinite(bg) else f"{bg:+10.2f}"
            bps = "       n/a" if not np.isfinite(bp) else f"{bp:>11.2g}"
            print(f"{label:<26}{sg:>+10.2f}{sp_:>11.2g}{bgs}{bps}{r}  {verdict}")

        print("-" * 88)
        tbl = pd.DataFrame(rows)
        tbl.to_csv(args.out_dir / "signal_vs_batch.csv", index=False)

        testable = tbl[np.isfinite(tbl["ratio"])]
        if not testable.empty:
            best = testable.loc[testable["ratio"].idxmax()]
            print(f"\nBest readout: {best['readout']}   "
                  f"signal g={best['signal_g']:+.2f}, "
                  f"batch g={best['batch_g']:+.2f}, ratio {best['ratio']:.1f}")
            worst = testable.loc[testable["ratio"].idxmin()]
            print(f"Worst       : {worst['readout']}   "
                  f"signal g={worst['signal_g']:+.2f}, "
                  f"batch g={worst['batch_g']:+.2f}, ratio {worst['ratio']:.1f}")

        # ---- every within-date stiffness contrast, for the winning readout ----
        if not testable.empty:
            col = measures[best["readout"]]
            print(f"\n{'=' * 88}\nALL WITHIN-DATE STIFFNESS CONTRASTS "
                  f"— {best['readout']}")
            any_found = False
            for d, g in pos.groupby("date"):
                levels = sorted(g["stiffness"].unique())
                if len(levels) < 2:
                    continue
                any_found = True
                for x, y in itertools.combinations(levels, 2):
                    a = g[g.stiffness == x][col].dropna().to_numpy()
                    b = g[g.stiffness == y][col].dropna().to_numpy()
                    _, p = welch(a, b)
                    print(f"  {d}  {x} vs {y}: {a.mean():.3f} vs {b.mean():.3f}"
                          f"   g={hedges_g(a, b):+.2f}  p={p:.3g}")
            if not any_found:
                print("  none — no session holds two stiffness levels")

        print(f"\n{'=' * 88}\nPER-CONDITION MEANS (position-level)")
        summ = pos.groupby(["date", "stiffness"]).agg(
            n_positions=("position", "size"),
            tagln_per_dapi=("tagln_per_dapi", "mean"),
            vecad_per_dapi=("vecad_per_dapi", "mean"),
            endmt_score=("endmt_score", "mean"))
        print(summ.round(3).to_string())
        summ.to_csv(args.out_dir / "per_condition_means.csv")

        print("\n  [caveat] each stiffness is one dish imaged once (apart from "
              f"{BATCH_STIFFNESS}).\n           The ~100 positions per arm are "
              "technical, not biological, replicates,\n           so the p-values "
              "above describe this dish pair and not the population.\n"
              "           A second independent session is what would change that.")

        if not args.no_plot:
            _plot(pos, args.out_dir, d1, d2)

        for f in ("signal_vs_batch.csv", "position_measures.csv",
                  "per_condition_means.csv"):
            print(f"\nWrote {args.out_dir / f}")
    finally:
        sys.stdout = sys.__stdout__
        print(f"Log saved -> {args.out_dir / 'marker_ratio_stats_log.txt'}")
        tee.close()


def _plot(pos: pd.DataFrame, out_dir: Path, d1: str, d2: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("tagln_per_dapi", "TAGLN / DAPI\n(mesenchymal)"),
              ("vecad_per_dapi", "VE-cadherin / DAPI\n(endothelial)"),
              (DAPI, "DAPI, raw\n(internal control)")]
    groups = [(SIGNAL_DATE, "5kPa", "5 kPa\n(soft)", "#c0392b"),
              (SIGNAL_DATE, "500kPa", f"500 kPa\n({SIGNAL_DATE})", "#2471a3"),
              (d2, "500kPa", f"500 kPa\n({d2})", "#7fb3d5")]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    for ax, (col, title) in zip(axes, panels):
        data, labels, colors = [], [], []
        for d, s, lab, c in groups:
            v = pos[(pos.date == d) & (pos.stiffness == s)][col].dropna()
            data.append(v.to_numpy())
            labels.append(lab)
            colors.append(c)
        bp = ax.boxplot(data, labels=labels, widths=0.6, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black"))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.45)
        for i, (v, c) in enumerate(zip(data, colors), start=1):
            x = np.random.default_rng(0).normal(i, 0.055, len(v))
            ax.scatter(x, v, s=5, color=c, alpha=0.55, linewidths=0)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("position-level mean", fontsize=9)
    fig.suptitle("The two rightmost boxes are the same biology on two days — "
                 "a flat pair there is what makes the left contrast credible",
                 fontsize=9, y=0.035)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    path = out_dir / "marker_ratios.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
