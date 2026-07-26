#!/usr/bin/env python3
"""Stage 3 — QC: confound probe + UMAP over the cell embeddings.

Answers the decisive question from the methodology (§9, §11): did the
experimental condition leak into the embeddings? If a simple classifier can
predict `condition` from the frozen embeddings well above chance, the confound
leaked and the adversarial scrubbing head should be added; if it can't, the
contrastive-only encoder is fine and the adversary can be skipped.

Also renders UMAPs colored by condition (leakage check) and by EndMT score
(biology sanity), and a quick k-means preview (cluster-ability check).

Inputs:
    <root>/stage3/embeddings.parquet   (from embed_cells.py)
    <root>/master_table.csv            (for endmt_score per cell)

Outputs:
    <root>/stage3/probe_report.json         probe accuracies + verdict
    <root>/stage3/umap_by_condition.png
    <root>/stage3/umap_by_endmt.png

Usage:
    python probe_embeddings.py
    python probe_embeddings.py --max-cells 100000 --push-to-drive
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

DRIVE_ROOT = "Fusion AI/Prof Huang Project/Cellpose feature extractions"


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
    raise SystemExit(f"embeddings not found: {path} (or .csv). Run embed_cells.py first.")


def main() -> None:
    root = _default_local_root()
    ap = argparse.ArgumentParser(description="Confound probe + UMAP QC (Stage 3).")
    ap.add_argument("--embeddings", type=Path, default=root / "stage3" / "embeddings.parquet")
    ap.add_argument("--master-table", type=Path, default=root / "master_table.csv")
    ap.add_argument("--out-dir", type=Path, default=root / "stage3")
    ap.add_argument("--max-cells", type=int, default=100000,
                    help="subsample this many cells for probe/UMAP (speed)")
    ap.add_argument("--push-to-drive", action="store_true")
    args = ap.parse_args()

    df = load_embeddings(args.embeddings)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise SystemExit("no emb_* columns found in embeddings file")
    print(f"Loaded {len(df)} cell embeddings ({len(emb_cols)} dims)")

    # subsample for speed (stratified-ish by condition)
    if len(df) > args.max_cells:
        df = df.sample(args.max_cells, random_state=0).reset_index(drop=True)
        print(f"  subsampled to {len(df)} cells")

    X = df[emb_cols].to_numpy(dtype=np.float32)
    y_names = df["condition"].to_numpy()          # string labels (UMAP legend)
    conditions = sorted(pd.unique(y_names))
    # integer codes for the probes: MLPClassifier(early_stopping=True) cannot
    # score string targets (it runs np.isnan on the predictions).
    y = np.searchsorted(np.array(conditions), y_names)
    n_classes = len(conditions)
    chance = 1.0 / n_classes
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- CONFOUND PROBE: can a classifier read `condition` off the embedding? ----
    # Three fixes over the naive version:
    #  1. GROUPED CV by (condition, image_id) — cells from one image are highly
    #     correlated, so plain k-fold lets a probe memorise image-specific quirks
    #     and reports inflated leakage. No image spans train and test here.
    #  2. An MLP probe, reported as the STRONGEST (honest upper bound on how much
    #     condition information is recoverable at all).
    #  3. Every probe reported separately — no max() collapsing. A weak probe
    #     scoring low means that probe underfit, not that the information is gone
    #     (e.g. axis-aligned RF trees struggle with a linear boundary in 64-d).
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score, GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    # group key must include condition: image_ids repeat across conditions
    groups = (df["condition"].astype(str) + "/" + df["image_id"].astype(str)).to_numpy()
    n_groups = len(np.unique(groups))
    cv = GroupKFold(n_splits=min(3, n_groups))
    print(f"\nConfound probe (predict condition from embeddings)")
    print(f"  grouped {min(3, n_groups)}-fold CV over {n_groups} images "
          f"(no image spans train/test)")

    probes = [
        ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
        ("random_forest", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)),
        ("mlp", make_pipeline(StandardScaler(),
                              MLPClassifier(hidden_layer_sizes=(128,), max_iter=300,
                                            early_stopping=True, random_state=0))),
    ]
    results = {}
    for name, clf in probes:
        scores = cross_val_score(clf, X, y, cv=cv, groups=groups,
                                 scoring="accuracy", n_jobs=-1)
        results[name] = {"accuracy": float(scores.mean()), "std": float(scores.std())}
        print(f"  {name:<14} acc={scores.mean():.3f} (chance={chance:.3f})")

    # The STRONGEST probe is the honest measure of recoverable information.
    strongest = max(results.items(), key=lambda kv: kv[1]["accuracy"])
    strongest_acc = strongest[1]["accuracy"]
    print(f"  -> strongest probe: {strongest[0]} ({strongest_acc:.3f}) "
          f"= upper bound on recoverable condition info")

    # ---- EndMT-RETENTION: how well can endmt_score be recovered from h? ----
    # High R2 = biology preserved in the embedding. Read alongside the leak: the
    # goal is LOW condition-accuracy AND HIGH endmt-R2 (methodology §D).
    # Same grouped CV, and again the strongest regressor is the honest number.
    endmt_r2 = None
    try:
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.neural_network import MLPRegressor
        master = pd.read_csv(args.master_table,
                             usecols=["image_id", "condition", "cell_id", "endmt_score"])
        merged = df.merge(master, on=["image_id", "condition", "cell_id"], how="left")
        yv = merged["endmt_score"].to_numpy(dtype=np.float32)
        ok = np.isfinite(yv)
        if ok.sum() > 100:
            print("\nEndMT-retention (predict endmt_score from embeddings)...")
            endmt_r2 = {}
            for name, reg in [
                ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
                ("random_forest", RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)),
                ("mlp", make_pipeline(StandardScaler(),
                                      MLPRegressor(hidden_layer_sizes=(128,), max_iter=300,
                                                   early_stopping=True, random_state=0))),
            ]:
                r2 = float(cross_val_score(reg, X[ok], yv[ok], cv=cv, groups=groups[ok],
                                           scoring="r2", n_jobs=-1).mean())
                endmt_r2[name] = r2
                print(f"  {name:<14} R2={r2:.3f}")
    except Exception as e:
        print(f"[endmt-retention] skipped ({e})")

    # Verdict is based on the STRONGEST probe (upper bound), not a max over
    # probes of differing capacity.
    ratio = strongest_acc / chance
    if ratio >= 2.0 and strongest_acc >= 0.5:
        verdict = ("LEAK — condition is strongly recoverable. NOTE: some of this may be "
                   "legitimate stiffness biology; decompose into date vs stiffness before "
                   "scrubbing harder.")
    elif ratio >= 1.5:
        verdict = "PARTIAL — some condition signal present"
    else:
        verdict = "CLEAN — condition not recoverable beyond chance"
    print(f"\nVerdict: {verdict}")

    report = {
        "n_cells_probed": int(len(df)),
        "n_images": int(n_groups),
        "cv": f"GroupKFold({min(3, n_groups)}) by (condition, image_id)",
        "n_conditions": n_classes,
        "conditions": conditions,
        "chance_accuracy": chance,
        "condition_probe": results,
        "strongest_probe": {"name": strongest[0], "accuracy": strongest_acc},
        "accuracy_over_chance": ratio,
        "endmt_retention_r2": endmt_r2,
        "verdict": verdict,
    }
    if endmt_r2 is not None:
        best_r2 = max(endmt_r2.items(), key=lambda kv: kv[1])
        print(f"EndMT-retention R2 (strongest: {best_r2[0]}): {best_r2[1]:.3f}  "
              f"(higher = biology better preserved)")
    print(f"\nSummary — condition leak (strongest {strongest[0]}): {strongest_acc:.3f} "
          f"vs chance {chance:.3f}"
          + (f"  |  EndMT retention: {max(endmt_r2.values()):.3f}" if endmt_r2 else ""))
    (args.out_dir / "probe_report.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.out_dir / 'probe_report.json'}")

    # ---- UMAPs (leakage view + biology view) ----
    try:
        import umap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        print("\nComputing UMAP (this can take a minute)...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0)
        XY = reducer.fit_transform(X)

        # by condition (leakage): interleaved = good; separated = leak
        fig, ax = plt.subplots(figsize=(8, 7))
        for cond in conditions:
            m = y_names == cond
            ax.scatter(XY[m, 0], XY[m, 1], s=3, alpha=0.5, label=cond)
        ax.legend(markerscale=4, fontsize=8, loc="best")
        ax.set_title("UMAP of cell embeddings — colored by CONDITION\n"
                     "(interleaved = no leak; separated = confound leak)")
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(); fig.savefig(args.out_dir / "umap_by_condition.png", dpi=130)
        plt.close(fig)

        # by EndMT score (biology sanity): expect a gradient if signal captured
        master = pd.read_csv(args.master_table,
                             usecols=["image_id", "condition", "cell_id", "endmt_score"])
        merged = df.merge(master, on=["image_id", "condition", "cell_id"], how="left")
        fig, ax = plt.subplots(figsize=(8, 7))
        sc = ax.scatter(XY[:, 0], XY[:, 1], s=3, alpha=0.6,
                        c=merged["endmt_score"].to_numpy(), cmap="coolwarm", vmin=0, vmax=1)
        fig.colorbar(sc, label="EndMT score")
        ax.set_title("UMAP of cell embeddings — colored by EndMT score\n"
                     "(a gradient here = encoder captured real biology)")
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(); fig.savefig(args.out_dir / "umap_by_endmt.png", dpi=130)
        plt.close(fig)
        print(f"Wrote UMAP figures -> {args.out_dir}")
    except ImportError:
        print("\n[skip] UMAP/matplotlib not installed; skipping figures. "
              "Install with: pip install umap-learn matplotlib")

    if args.push_to_drive:
        dest = f"gdrive:{DRIVE_ROOT}/stage3"
        print("Pushing QC outputs to Drive...")
        r = subprocess.run(["rclone", "copy", str(args.out_dir), dest,
                            "--include", "probe_report.json", "--include", "umap_*.png",
                            "--transfers=8"])
        if r.returncode != 0:
            raise RuntimeError(f"rclone failed (exit {r.returncode})")


if __name__ == "__main__":
    main()
