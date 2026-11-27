#!/usr/bin/env python3
"""Build a PDF report from train_all.log and baselines.log.

Pages: summary, score tables for both labels, each GNN against its matched
baseline, per-epoch training curves, then the raw logs. With --test-csv
(results/test_summary.csv from score_exn.py --split test) the report opens with
the test results: scores with 95% CIs, precision/recall/F1 at the validation
threshold, and the paired differences. Needs matplotlib and pandas.

Usage
-----
    python exn_pipeline/make_training_report.py <train_all.log> <baselines.log> <out.pdf> \
        [--test-csv test_summary.csv] [--leakage-log leakage_trained.log]
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

_ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_ap.add_argument("train_log", type=Path)
_ap.add_argument("base_log", type=Path)
_ap.add_argument("out", type=Path)
_ap.add_argument("--test-csv", type=Path)
_ap.add_argument("--leakage-log", type=Path)
ARGS = _ap.parse_args()
train_log, base_log, out = ARGS.train_log, ARGS.base_log, ARGS.out

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})

# ---------- parse
runs, cur = {}, None
for line in train_log.read_text().splitlines():
    m = re.search(r"run=(\S+)", line)
    if m:
        cur = m.group(1)
        runs[cur] = {"epochs": [], "auprc": [], "wf": []}
    m = re.match(r"\s+epoch\s+(\d+)\s+loss \S+\s+val AUPRC (\S+)\s+AUROC \S+\s+within-field AUROC (\S+)", line)
    if m and cur:
        runs[cur]["epochs"].append(int(m.group(1)))
        runs[cur]["auprc"].append(float(m.group(2)))
        runs[cur]["wf"].append(float(m.group(3)))
    m = re.match(r"(\S+): best epoch (\d+) .*?\(([\d,]+) cells, ([\d.]+)% positive", line)
    if m:
        cur_final = m.group(1)
        runs[cur_final]["best"] = int(m.group(2))
        runs[cur_final]["prev"] = float(m.group(4)) / 100
    m = re.match(r"\s+AUPRC ([\d.]+)\s+\(chance", line)
    if m:
        runs[cur_final]["final_auprc"] = float(m.group(1))
    m = re.match(r"\s+AUROC ([\d.]+)\s+\|\s+within-field AUROC ([\d.]+)", line)
    if m:
        runs[cur_final]["final_auroc"] = float(m.group(1))
        runs[cur_final]["final_wf"] = float(m.group(2))

base = {}
for line in base_log.read_text().splitlines():
    m = re.match(r"\s+(red|tagln_high)\s+(\S+)\s+AUPRC (\S+)\s+AUROC (\S+)\s+within-field AUROC (\S+)\s+\((\d+) features", line)
    if m:
        base[(m.group(1), m.group(2))] = dict(auprc=float(m.group(3)), auroc=float(m.group(4)),
                                              wf=float(m.group(5)), nfeat=int(m.group(6)))

LABEL_NAME = {"red": "Red display class (red/mesenchymal vs gray/endothelial)",
              "tagln_high": "TAGLN-high"}
ROWS = [  # display, kind, key, inputs description
    ("LR own", "b", "own_lr", "Target cell's own 12 shape/position features"),
    ("HGB own", "b", "own_hgb", "Same 12 features, gradient-boosted trees"),
    ("HGB nbr avg (A)", "b", "nbrA_hgb", "+ mean of 1-hop neighbours' features"),
    ("GNN-A", "g", "A", "Graph of neighbours' features (2 hops)"),
    ("HGB nbr avg (B)", "b", "nbrB_hgb", "+ mean of neighbours' markers and labels"),
    ("GNN-B", "g", "B", "Graph + neighbours' markers and labels"),
    ("HGB nbr avg (B, no junction)", "b", "nbrB_nojunction_hgb", "As above, touching nbrs' VE-cad hidden"),
    ("GNN-B no junction", "g", "B_nojunction", "GNN-B, touching nbrs' VE-cad hidden"),
]


def tables_page(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.07, 0.95, "Validation scores, all models", fontsize=15, weight="bold")
    fig.text(0.07, 0.925, "No model ever sees the target cell's own markers or labels. "
             "GNN rows in bold. AUPRC and AUROC: higher is better.", fontsize=8.5, color=INK2)
    cols = ["Model", "Inputs", "AUPRC", "× chance", "AUROC", "Within-field\nAUROC"]
    top = 0.88
    for label, title in (("red", "Red display class"), ("tagln_high", "TAGLN-high")):
        prev = runs[f"{label}_A"]["prev"]
        fig.text(0.07, top, title, fontsize=11, weight="bold")
        cells = []
        for disp, k, key, desc in ROWS:
            a, r, w = row_vals(label, k, key)
            cells.append([disp, desc, f"{a:.3f}", f"{a / prev:.1f}×", f"{r:.3f}", f"{w:.3f}"])
        ax = fig.add_axes([0.07, top - 0.33, 0.86, 0.31])
        ax.axis("off")
        t = ax.table(cellText=cells, colLabels=cols, cellLoc="left", colLoc="left",
                     colWidths=[0.25, 0.38, 0.08, 0.09, 0.08, 0.12], bbox=[0, 0, 1, 1])
        t.auto_set_font_size(False)
        t.set_fontsize(7.8)
        for (i, j), c in t.get_celld().items():
            c.set_edgecolor(GRID)
            c.set_linewidth(0.6)
            if i == 0:
                c.set_text_props(weight="bold", color=INK)
                c.set_facecolor("#f3f2ef")
            elif cells[i - 1][0].startswith("GNN"):
                c.set_text_props(weight="bold")
                c.set_facecolor("#eef4fc")
        fig.text(0.07, top - 0.355, f"246,577 validation cells in held-out wells; "
                 f"{100 * prev:.2f}% positive, so a random guess scores AUPRC {prev:.3f}. "
                 f"× chance = AUPRC / {prev:.3f}.", fontsize=8, color=MUTED)
        top -= 0.43
    pdf.savefig(fig)
    plt.close(fig)


def row_vals(label, kind, key):
    if kind == "b":
        b = base[(label, key)]
        return b["auprc"], b["auroc"], b["wf"]
    r = runs[f"{label}_{key}"]
    return r["final_auprc"], r["final_auroc"], r["final_wf"]


def text_page(pdf, title, blocks):
    fig = plt.figure(figsize=(8.5, 11))
    y = 0.95
    fig.text(0.07, y, title, fontsize=15, weight="bold")
    y -= 0.04
    for kind, txt in blocks:
        if kind == "h":
            y -= 0.012
            fig.text(0.07, y, txt, fontsize=11, weight="bold")
            y -= 0.028
        elif kind == "p":
            fig.text(0.07, y, txt, fontsize=9, color=INK2, va="top", wrap=True,
                     linespacing=1.45)
            y -= 0.0165 * (txt.count("\n") + 1) + 0.012
        elif kind == "table":
            label = txt
            prev = runs[f"{label}_A"]["prev"]
            cols = ["Model", "Inputs (never the target's own markers/labels)",
                    "AUPRC", "× chance", "AUROC", "Within-field\nAUROC"]
            cells = []
            for disp, k, key, desc in ROWS:
                a, r, w = row_vals(label, k, key)
                cells.append([disp, desc, f"{a:.3f}", f"{a / prev:.1f}×", f"{r:.3f}", f"{w:.3f}"])
            ax = fig.add_axes([0.07, y - 0.27, 0.86, 0.27])
            ax.axis("off")
            t = ax.table(cellText=cells, colLabels=cols, loc="upper left", cellLoc="left",
                         colLoc="left", colWidths=[0.22, 0.40, 0.09, 0.09, 0.09, 0.11])
            t.auto_set_font_size(False)
            t.set_fontsize(8)
            t.scale(1, 1.55)
            for (i, j), c in t.get_celld().items():
                c.set_edgecolor(GRID)
                c.set_linewidth(0.6)
                if i == 0:
                    c.set_text_props(weight="bold", color=INK)
                    c.set_facecolor("#f3f2ef")
                    c.set_height(c.get_height() * 1.5)
                elif cells[i - 1][0].startswith("GNN"):
                    c.set_text_props(weight="bold")
                    c.set_facecolor("#eef4fc")
            y -= 0.30
            fig.text(0.07, y, f"Validation: 246,577 cells in held-out wells; {100 * prev:.2f}% "
                     f"positive, so a random guess scores AUPRC {prev:.3f}.", fontsize=8,
                     color=MUTED)
            y -= 0.03
    pdf.savefig(fig)
    plt.close(fig)


def gap_page(pdf):
    """Dot plot: each GNN against its matched baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.6), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.78, bottom=0.16, wspace=0.12)
    fig.text(0.07, 0.93, "Each GNN against the baseline given the same information",
             fontsize=13, weight="bold")
    fig.text(0.07, 0.875, "Validation AUPRC. Gray = gradient-boosted trees on neighbour "
             "averages; blue = GNN. Dashed line = random guess.", fontsize=9, color=INK2)
    pairs = [("A", "nbrA_hgb", "Version A\n(neighbours' shape)"),
             ("B", "nbrB_hgb", "Version B\n(+ neighbours' markers)"),
             ("B_nojunction", "nbrB_nojunction_hgb", "Version B,\nno junction")]
    for ax, label in zip(axes, ["red", "tagln_high"]):
        prev = runs[f"{label}_A"]["prev"]
        for i, (g, b, name) in enumerate(pairs):
            yb = 2 - i
            bv, gv = base[(label, b)]["auprc"], runs[f"{label}_{g}"]["final_auprc"]
            ax.plot([bv, gv], [yb, yb], color=GRID, lw=2, zorder=1)
            ax.scatter([bv], [yb], s=60, color=MUTED, zorder=2, edgecolor="white", lw=1.5)
            ax.scatter([gv], [yb], s=60, color=BLUE, zorder=3, edgecolor="white", lw=1.5)
            ax.text(gv + 0.006, yb + 0.16, f"{gv:.3f}", color=INK, fontsize=8)
            ax.text(bv - 0.006, yb + 0.16, f"{bv:.3f}", color=INK2, fontsize=8, ha="right")
            ax.text((bv + gv) / 2, yb - 0.32, f"+{gv - bv:.3f}", color=INK2, fontsize=8,
                    ha="center")
        ax.axvline(prev, color=MUTED, ls="--", lw=1)
        ax.text(prev + 0.004, -0.62, f"chance {prev:.3f}", fontsize=7.5, color=MUTED)
        ax.set_yticks([2, 1, 0], [p[2] for p in pairs])
        ax.set_ylim(-0.8, 2.6)
        ax.set_xlim(0, 0.52)
        ax.set_title(LABEL_NAME[label] if label == "tagln_high" else "Red display class",
                     loc="left", fontsize=10, weight="bold")
        ax.set_xlabel("Validation AUPRC")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    pdf.savefig(fig)
    plt.close(fig)


def curve_page(pdf):
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 11), sharex=True)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.86, bottom=0.08, hspace=0.18, wspace=0.22)
    fig.text(0.07, 0.95, "Training curves", fontsize=15, weight="bold")
    fig.text(0.07, 0.905,
             "Each epoch = 200,000 training cells drawn at random. Scores per epoch are on a fixed "
             "50,000-cell sample of the\nvalidation wells, so they differ slightly from the final "
             "scores on all 246,577 cells. Ringed point = epoch kept.", fontsize=8.5,
             color=INK2, linespacing=1.4)
    series = [("A", "Version A", BLUE), ("B", "Version B", ORANGE),
              ("B_nojunction", "Version B, no junction", AQUA)]
    for col, label in enumerate(["red", "tagln_high"]):
        for row, (metric, ylab) in enumerate([("auprc", "Validation AUPRC (50k sample)"),
                                              ("wf", "Within-field AUROC (50k sample)")]):
            ax = axes[row, col]
            for key, name, c in series:
                r = runs[f"{label}_{key}"]
                ax.plot(r["epochs"], r[metric], color=c, lw=2, label=name)
                bi = r["epochs"].index(r["best"])
                ax.scatter([r["best"]], [r[metric][bi]], s=64, facecolor="white",
                           edgecolor=c, lw=2, zorder=3)
                ax.text(r["epochs"][-1] + 0.4, r[metric][-1], name.replace("Version ", ""),
                        color=INK2, fontsize=7.5, va="center")
            if row == 0:
                ax.set_title("Red display class" if label == "red" else "TAGLN-high",
                             loc="left", fontsize=11, weight="bold")
            ax.set_ylabel(ylab)
            ax.set_xlim(0.5, 34)
            ax.grid(color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if row == 1:
                ax.set_xlabel("Epoch")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    pdf.savefig(fig)
    plt.close(fig)


def log_pages(pdf, path, title):
    import textwrap
    lines = [w for ln in path.read_text().splitlines()
             for w in (textwrap.wrap(ln, 140, subsequent_indent="      ") or [""])]
    per = 92
    for p in range(0, len(lines), per):
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.05, 0.965, f"{title}  (page {p // per + 1} of {(len(lines) - 1) // per + 1})",
                 fontsize=10, weight="bold")
        fig.text(0.05, 0.945, "\n".join(lines[p:p + per]), family="monospace", fontsize=5.6,
                 va="top", linespacing=1.32, color=INK)
        pdf.savefig(fig)
        plt.close(fig)


def gap(label, g, b):
    return runs[f"{label}_{g}"]["final_auprc"] - base[(label, b)]["auprc"]


# ---------- test results
MODEL_ORDER = [r_[0] for r_ in ROWS]
PAIR_ORDER = ["GNN-A - HGB own", "GNN-A - HGB nbr avg (A)", "GNN-B - GNN-A",
              "GNN-B - HGB nbr avg (B)", "GNN-B - GNN-B no junction",
              "HGB nbr avg (B) - HGB nbr avg (B, no junction)"]
PAIR_NAME = {
    "GNN-A - HGB own": "GNN-A vs own features only",
    "GNN-A - HGB nbr avg (A)": "GNN-A vs neighbour averages (A)",
    "GNN-B - GNN-A": "Neighbours' markers/labels (B − A)",
    "GNN-B - HGB nbr avg (B)": "GNN-B vs neighbour averages (B)",
    "GNN-B - GNN-B no junction": "Touching nbrs' VE-cad, GNN",
    "HGB nbr avg (B) - HGB nbr avg (B, no junction)": "Touching nbrs' VE-cad, trees",
}
test = pd.read_csv(ARGS.test_csv) if ARGS.test_csv else None


def T(label, model):
    row = test[(test.label == label) & (test.model == model)]
    return row.iloc[0]


def test_tables_page(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.07, 0.95, "Final results: held-out test wells", fontsize=15, weight="bold")
    fig.text(0.07, 0.915,
             "Scored once, after every model was final. 95% intervals from resampling wells. "
             "Precision, recall and F1\nuse the threshold chosen on validation (best F1), so "
             "the test set never picks its own threshold.", fontsize=8.5, color=INK2,
             linespacing=1.4)
    cols = ["Model", "AUPRC [95% CI]", "× chance", "Within-field\nAUROC [95% CI]",
            "Precision", "Recall", "F1"]
    top = 0.86
    for label, title in (("red", "Red display class"), ("tagln_high", "TAGLN-high")):
        prev = T(label, "GNN-A").prevalence
        n = int(T(label, "GNN-A").n_cells)
        fig.text(0.07, top, title, fontsize=11, weight="bold")
        cells = []
        for m in MODEL_ORDER:
            t = T(label, m)
            cells.append([m, f"{t.auprc:.3f} [{t.auprc_ci_low:.3f}, {t.auprc_ci_high:.3f}]",
                          f"{t.auprc / prev:.1f}×",
                          f"{t.within_field_auroc:.3f} [{t.within_field_auroc_ci_low:.3f}, "
                          f"{t.within_field_auroc_ci_high:.3f}]",
                          f"{t.test_precision:.3f}", f"{t.test_recall:.3f}", f"{t.test_f1:.3f}"])
        ax = fig.add_axes([0.07, top - 0.33, 0.86, 0.31])
        ax.axis("off")
        tb = ax.table(cellText=cells, colLabels=cols, cellLoc="left", colLoc="left",
                      colWidths=[0.25, 0.20, 0.08, 0.20, 0.09, 0.09, 0.09], bbox=[0, 0, 1, 1])
        tb.auto_set_font_size(False)
        tb.set_fontsize(7.6)
        for (i, j), c in tb.get_celld().items():
            c.set_edgecolor(GRID)
            c.set_linewidth(0.6)
            if i == 0:
                c.set_text_props(weight="bold", color=INK)
                c.set_facecolor("#f3f2ef")
            elif cells[i - 1][0].startswith("GNN"):
                c.set_text_props(weight="bold")
                c.set_facecolor("#eef4fc")
        fig.text(0.07, top - 0.355, f"{n:,} test cells in 52 held-out wells; {100 * prev:.2f}% "
                 f"positive, so a random guess scores AUPRC {prev:.3f}.", fontsize=8, color=MUTED)
        top -= 0.43
    pdf.savefig(fig)
    plt.close(fig)


def test_diff_page(pdf):
    """Forest plot of the paired AUPRC differences on test."""
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 5.2), sharey=True)
    fig.subplots_adjust(left=0.28, right=0.97, top=0.8, bottom=0.13, wspace=0.08)
    fig.text(0.07, 0.94, "Paired differences on the test wells", fontsize=13, weight="bold")
    fig.text(0.07, 0.88, "AUPRC of the first model minus the second, same cells. Bars = 95% "
             "interval from resampling wells;\nan interval that clears 0 means the difference "
             "holds across wells.", fontsize=8.5, color=INK2, linespacing=1.4)
    ys = list(range(len(PAIR_ORDER)))[::-1]
    for ax, label in zip(axes, ["red", "tagln_high"]):
        for y, p in zip(ys, PAIR_ORDER):
            t = T(label, f"Δ {p}")
            c = MUTED if "VE-cad" in PAIR_NAME[p] else BLUE
            ax.plot([t.auprc_ci_low, t.auprc_ci_high], [y, y], color=c, lw=2,
                    solid_capstyle="round")
            ax.scatter([t.auprc], [y], s=40, color=c, zorder=3, edgecolor="white", lw=1.2)
            ax.text(t.auprc_ci_high + 0.004, y, f"{t.auprc:+.3f}", va="center", fontsize=7.5,
                    color=INK)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlim(-0.01, 0.17)
        ax.set_title("Red display class" if label == "red" else "TAGLN-high", loc="left",
                     fontsize=10, weight="bold")
        ax.set_xlabel("Difference in test AUPRC")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ys, [PAIR_NAME[p] for p in PAIR_ORDER])
    pdf.savefig(fig)
    plt.close(fig)


def test_summary():
    def a(label, m):
        return T(label, m).auprc

    def d(label, p):
        t = T(label, f"Δ {p}")
        return f"{t.auprc:+.3f} [{t.auprc_ci_low:+.3f}, {t.auprc_ci_high:+.3f}]"

    pr, pt = T("red", "GNN-A").prevalence, T("tagln_high", "GNN-A").prevalence
    rb, tb = T("red", "GNN-B no junction"), T("tagln_high", "GNN-B no junction")
    leak = ""
    if ARGS.leakage_log and "ALL CHECKS PASSED" in ARGS.leakage_log.read_text():
        n_pass = ARGS.leakage_log.read_text().count("PASS ")
        leak = (f"Leakage checks on the trained models: all {n_pass} passed (the target "
                f"cell's own markers and labels\nchange its prediction by exactly 0; raw "
                f"output at the end of this report).")
    blocks = [
        ("p", "Per-cell binary classification on Carver's ExN dataset (319 fields, 1.23M "
              "cells), split by well:\n60% train, 20% validation, 20% test. No model ever "
              "sees the target cell's own markers or labels.\nTest wells were scored once "
              "(2026-11-26), after every model was final.\n" + leak),
        ("h", "Headline (test wells)"),
        ("p", f"Red display class: GNN-B with touching neighbours' VE-cad hidden reaches "
              f"AUPRC {rb.auprc:.3f}\n[{rb.auprc_ci_low:.3f}, {rb.auprc_ci_high:.3f}], "
              f"{rb.auprc / pr:.1f}× the random-guess score of {pr:.3f}; within-field AUROC "
              f"{rb.within_field_auroc:.3f}."),
        ("p", f"TAGLN-high: the same model reaches AUPRC {tb.auprc:.3f} [{tb.auprc_ci_low:.3f}, "
              f"{tb.auprc_ci_high:.3f}], {tb.auprc / pt:.1f}× the random-guess\nscore of "
              f"{pt:.3f}; within-field AUROC {tb.within_field_auroc:.3f}."),
        ("h", "What the numbers say"),
        ("p", f"1. The GNN beats gradient-boosted trees given the same information, for both "
              f"labels and versions:\n   red A {d('red', 'GNN-A - HGB nbr avg (A)')}, "
              f"red B {d('red', 'GNN-B - HGB nbr avg (B)')};\n   TAGLN-high A "
              f"{d('tagln_high', 'GNN-A - HGB nbr avg (A)')}, TAGLN-high B "
              f"{d('tagln_high', 'GNN-B - HGB nbr avg (B)')}."),
        ("p", f"2. Neighbours' markers and labels add the most (GNN-B − GNN-A): red "
              f"{d('red', 'GNN-B - GNN-A')},\n   TAGLN-high {d('tagln_high', 'GNN-B - GNN-A')}."),
        ("p", f"3. The junction a target shares with touching neighbours explains little of "
              f"that gain: hiding their\n   VE-cad costs red {d('red', 'GNN-B - GNN-B no junction')} "
              f"and TAGLN-high {d('tagln_high', 'GNN-B - GNN-B no junction')}.\n"
              f"   The no-junction GNN-B is the conservative headline number."),
        ("p", "4. AUPRC cannot be compared across labels (about 5% of cells are red, about 11% "
              "TAGLN-high).\n   Relative to chance, and ranked within a field, red is "
              "predicted better than TAGLN-high."),
        ("p", f"5. Test agrees with validation: the models rank in the same order. GNN-B AUPRC, "
              f"validation → test:\n   red {r['red_B']:.3f} → {a('red', 'GNN-B'):.3f}; "
              f"TAGLN-high {r['tagln_high_B']:.3f} → {a('tagln_high', 'GNN-B'):.3f}. TAGLN-high "
              f"is lower on test for every\n   model, baselines included, so its test wells are "
              f"harder rather than the GNN overfitting."),
        ("p", "6. Open question for Carver: TAGLN-high leans heavily on neighbours' TAGLN calls. "
              "Check whether\n   the way TAGLN-high was called couples a cell's call to its "
              "neighbours' before reading it as biology."),
    ]
    return blocks


r = {k: v["final_auprc"] for k, v in runs.items()}
summary = [
    ("p", "All 6 GNN runs and all 10 baselines finished on 2026-11-26 on Lightning (GPU). Scores below\n"
          "are on the validation wells; the test wells have not been used. These are single-number\n"
          "scores without confidence intervals; score_exn.py --split val adds 95% intervals\n"
          "(resampling wells) and paired differences."),
    ("h", "What the numbers say"),
    ("p", f"1. Every GNN beats the baseline given the same information. Red: +{gap('red','A','nbrA_hgb'):.3f} "
          f"(A), +{gap('red','B','nbrB_hgb'):.3f} (B), +{gap('red','B_nojunction','nbrB_nojunction_hgb'):.3f} "
          f"(B, no junction).\n   TAGLN-high: +{gap('tagln_high','A','nbrA_hgb'):.3f} (A), "
          f"+{gap('tagln_high','B','nbrB_hgb'):.3f} (B), +{gap('tagln_high','B_nojunction','nbrB_nojunction_hgb'):.3f} "
          f"(B, no junction). AUPRC, higher is better."),
    ("p", f"2. Neighbours' markers and labels help (Version B over A): red {r['red_A']:.3f} → {r['red_B']:.3f}; "
          f"TAGLN-high {r['tagln_high_A']:.3f} → {r['tagln_high_B']:.3f}."),
    ("p", f"3. Hiding the VE-cad values of neighbours whose masks touch the target costs little:\n"
          f"   red {r['red_B']:.3f} → {r['red_B_nojunction']:.3f}; TAGLN-high {r['tagln_high_B']:.3f} → "
          f"{r['tagln_high_B_nojunction']:.3f}. So Version B's gain does not come mainly from\n"
          f"   the cell–cell junction the target shares with its touching neighbours."),
    ("p", "4. AUPRC cannot be compared across the two labels: about 11% of cells are TAGLN-high and\n"
          "   about 5% are red, so a random guess already scores twice as high on TAGLN-high. The\n"
          "   '× chance' column puts both on the same footing: red is predicted further above chance."),
    ("p", "5. For TAGLN-high, adding neighbours' markers to the tree baseline raises AUPRC more than it\n"
          "   does for red. Most of that seems to come from neighbours' own TAGLN calls (TAGLN-high cells\n"
          "   sit next to each other). That may be real spatial clustering, or a side effect of how the\n"
          "   TAGLN-high calls were made; worth asking Carver before interpreting it."),
    ("h", "Still to run"),
    ("p", "check_exn_leakage.py --trained (should end ALL CHECKS PASSED), then score_exn.py --split val.\n"
          "Test split only after the validation results are reviewed and the models are final."),
]

with PdfPages(out) as pdf:
    if test is not None:
        text_page(pdf, "ExN cell classification: final results", test_summary())
        test_tables_page(pdf)
        test_diff_page(pdf)
    else:
        text_page(pdf, "ExN cell classification: training results (validation)", summary)
    tables_page(pdf)
    gap_page(pdf)
    curve_page(pdf)
    if ARGS.leakage_log:
        log_pages(pdf, ARGS.leakage_log, "Raw log: leakage_trained.log")
    log_pages(pdf, base_log, "Raw log: baselines.log")
    log_pages(pdf, train_log, "Raw log: train_all.log")
print("wrote", out)
