#!/usr/bin/env python3
"""Step 1: fix the train / val / test split by well and build the training cache.

Reads the extractor's outputs (exn_nodes.csv, exn_edges.csv, exn_schema.json)
once and writes everything the GNN, the baselines and the scorer need:

    <work>/split/wells.csv     one row per well: batch, well, split, counts
    <work>/split/rules.json    the RULES it was built with + sha256 of wells.csv
    <work>/cache/nodes.npz     transformed node columns, labels, split, field, well
    <work>/cache/edges.npz     CSR adjacency over all 1.2M cells + edge features
    <work>/cache/meta.json     column names, standardisation stats (train wells only)

The split is made once. A second run reuses split/wells.csv and only rebuilds
the cache; pass --new-split to draw a new split (every model trained on the old
one is then invalid, and the later steps will refuse to mix them).

Usage
-----
    python exn_pipeline/prepare_exn_training.py \
        --features-dir ~/exn_features --work-dir ~/exn_training
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exn_columns import TEXT_TWINS  # noqa: E402
from exn_data import (EDGE_COLS, LOG1P_COLS, LOG_COLS, RULES, SPLIT_CODE,  # noqa: E402
                      default_features_dir, default_work_dir, is_binary, sha256_file)

KEY_SHIFT = 20          # node key = field_code << 20 | cell_id
INPUT_ROLES = ("feature", "marker", "label", "condition")


def transform(col: str, v: pd.Series, d_max: pd.Series | None = None) -> np.ndarray:
    v = pd.to_numeric(v, errors="coerce").astype(np.float64)
    if col == "mean_neighbor_dist_px" and d_max is not None:
        v = v.fillna(d_max)                 # isolated cell: as far as a neighbour can be
    if col == "elongation":
        v = v.fillna(1.0)                   # undefined ellipse: treat as round
    if col in LOG1P_COLS:
        return np.log1p(v.clip(lower=0)).to_numpy()
    if col in LOG_COLS:
        return np.log(v.clip(lower=1e-6)).to_numpy()
    return v.to_numpy()


def make_split(nodes: pd.DataFrame, seed: int) -> pd.DataFrame:
    w = (nodes.groupby("wellkey", sort=True)
         .agg(batch=("batch", "first"), well=("well", "first"),
              n_fields=("folder", "nunique"), n_cells=("cell_id", "size"),
              n_targets=("target_ok", "sum"), n_red=("label_red", "sum"),
              n_tagln_high=("label_tagln_high", "sum"))
         .reset_index())
    rng = np.random.default_rng(seed)
    w["split"] = ""
    for _, g in w.groupby("batch", sort=True):
        keys = g.wellkey.to_numpy()[rng.permutation(len(g))]
        n = len(keys)
        if n == 1:
            parts = ["train"]
        elif n == 2:
            parts = ["train", "test"]
        else:
            n_test = max(1, int(round(RULES["fractions"]["test"] * n)))
            n_val = max(1, int(round(RULES["fractions"]["val"] * n)))
            parts = ["test"] * n_test + ["val"] * n_val + ["train"] * (n - n_test - n_val)
        w.loc[w.wellkey.isin(keys), "split"] = pd.Series(parts, index=keys).reindex(
            w.loc[w.wellkey.isin(keys), "wellkey"]).to_numpy()
    return w


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features-dir", type=Path, default=default_features_dir())
    ap.add_argument("--work-dir", type=Path, default=default_work_dir())
    ap.add_argument("--new-split", action="store_true",
                    help="draw a new split even if split/wells.csv exists")
    ap.add_argument("--chunk-rows", type=int, default=2_000_000)
    args = ap.parse_args()

    t0 = time.time()
    feat, work = args.features_dir.expanduser(), args.work_dir.expanduser()
    (work / "split").mkdir(parents=True, exist_ok=True)
    (work / "cache").mkdir(parents=True, exist_ok=True)
    print(f"prepare_exn_training.py  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  features : {feat}\n  work dir : {work}")

    schema = json.loads((feat / "exn_schema.json").read_text())
    if "roles" not in schema:
        raise SystemExit("[error] exn_schema.json has no roles; rerun extract_exn_features.py")
    roles = {c: v["role"] for c, v in schema["nodes"].items()}
    input_cols = [c for c, r in roles.items() if r in INPUT_ROLES and c not in TEXT_TWINS]
    label_cols = [c for c, r in roles.items() if r == "label"]
    need = ["folder", "cell_id", "batch", "well", "n_fragments", "field_d_max_px"] + input_cols
    need = list(dict.fromkeys(need))

    # ---- nodes
    print(f"\nReading exn_nodes.csv ({len(need)} columns)...", flush=True)
    nodes = pd.read_csv(feat / "exn_nodes.csv", usecols=need,
                        dtype={"folder": str, "batch": str, "well": str})
    print(f"  {len(nodes):,} cells in {nodes.folder.nunique()} fields  "
          f"({time.time() - t0:.0f} s)")
    if nodes.cell_id.max() >= (1 << KEY_SHIFT):
        raise SystemExit("[error] cell_id too large for the node key")
    bad_well = nodes.batch.isna() | nodes.well.isna()
    if bad_well.any():
        print(f"  [warn] {int(bad_well.sum())} cells have no batch/well; their field is "
              f"used as the well")
    nodes["wellkey"] = np.where(bad_well, "field:" + nodes.folder,
                                nodes.batch.fillna("") + "/" + nodes.well.fillna(""))
    nodes["target_ok"] = nodes.n_fragments.fillna(1).astype(int) == 1
    for c in label_cols:
        nodes[c] = nodes[c].fillna(0).astype(np.int8)

    # ---- split (made once)
    wells_path = work / "split" / "wells.csv"
    if wells_path.exists() and not args.new_split:
        wells = pd.read_csv(wells_path)
        print(f"\nReusing the existing split: {wells_path}")
        unknown = set(nodes.wellkey) - set(wells.wellkey)
        if unknown:
            raise SystemExit(f"[error] {len(unknown)} wells are not in the saved split, "
                             f"e.g. {sorted(unknown)[:3]}; rerun with --new-split")
    else:
        wells = make_split(nodes, RULES["seed"])
        wells.to_csv(wells_path, index=False)
        print(f"\nNew split written: {wells_path}")
    rules = {"rules": RULES, "wells_sha256": sha256_file(wells_path),
             "created": datetime.now().isoformat(timespec="seconds"),
             "features_dir": str(feat)}
    rules_path = work / "split" / "rules.json"
    if rules_path.exists() and not args.new_split:
        old = json.loads(rules_path.read_text())
        if old["wells_sha256"] != rules["wells_sha256"] or old["rules"] != RULES:
            raise SystemExit("[error] split/rules.json disagrees with wells.csv or RULES")
        rules = old
    else:
        rules_path.write_text(json.dumps(rules, indent=2))
    split_of = dict(zip(wells.wellkey, wells.split))
    nodes["split"] = nodes.wellkey.map(split_of)

    # ---- per-split summary
    summ = nodes.groupby("split").agg(
        wells=("wellkey", "nunique"), fields=("folder", "nunique"), cells=("cell_id", "size"),
        targets=("target_ok", "sum"), red_pct=("label_red", "mean"),
        tagln_high_pct=("label_tagln_high", "mean")).reindex(["train", "val", "test"])
    summ[["red_pct", "tagln_high_pct"]] *= 100
    print("\n" + summ.round(2).to_string())
    cover = nodes.groupby("batch").split.agg(lambda s: ",".join(sorted(set(s))))
    thin = cover[~cover.str.contains("test") | ~cover.str.contains("val")]
    if len(thin):
        print("\nBatches not in every split (too few wells to divide):")
        print(thin.to_string())

    # ---- node arrays
    fields = sorted(nodes.folder.unique())
    well_names = sorted(nodes.wellkey.unique())
    fcode = pd.Categorical(nodes.folder, categories=fields).codes.astype(np.int64)
    wcode = pd.Categorical(nodes.wellkey, categories=well_names).codes.astype(np.int32)
    key = (fcode << KEY_SHIFT) | nodes.cell_id.to_numpy(np.int64)
    if len(np.unique(key)) != len(key):
        raise SystemExit("[error] duplicate (field, cell_id) in exn_nodes.csv")
    key_order = np.argsort(key)
    key_sorted = key[key_order]

    X = np.empty((len(nodes), len(input_cols)), dtype=np.float32)
    for j, c in enumerate(input_cols):
        X[:, j] = transform(c, nodes[c], nodes.field_d_max_px)
    split_code = nodes.split.map(SPLIT_CODE).to_numpy(np.int8)
    train = split_code == SPLIT_CODE["train"]
    node_stats = {}
    for j, c in enumerate(input_cols):
        if is_binary(c):
            continue
        v = X[train, j]
        v = v[np.isfinite(v)]
        sd = float(v.std()) if len(v) else 1.0
        node_stats[c] = {"mean": float(v.mean()) if len(v) else 0.0,
                         "std": sd if sd > 1e-8 else 1.0}

    # ---- edges, read in chunks and mapped to global node indices
    print("\nReading exn_edges.csv in chunks...", flush=True)
    ecols = ["folder", "src_cell_id", "dst_cell_id"] + EDGE_COLS
    src_l, dst_l, e_l = [], [], []
    n_read = 0
    for ch in pd.read_csv(feat / "exn_edges.csv", usecols=ecols, chunksize=args.chunk_rows,
                          dtype={"folder": str}):
        fc = pd.Categorical(ch.folder, categories=fields).codes.astype(np.int64)
        if (fc < 0).any():
            raise SystemExit("[error] edges reference a field with no nodes")
        for col, out in (("src_cell_id", src_l), ("dst_cell_id", dst_l)):
            k = (fc << KEY_SHIFT) | ch[col].to_numpy(np.int64)
            pos = np.minimum(np.searchsorted(key_sorted, k), len(key_sorted) - 1)
            if not (key_sorted[pos] == k).all():
                raise SystemExit("[error] edges reference cells missing from exn_nodes.csv")
            out.append(key_order[pos].astype(np.int32))
        e_l.append(np.stack([transform(c, ch[c]) for c in EDGE_COLS], axis=1)
                   .astype(np.float32))
        n_read += len(ch)
        print(f"  {n_read:,} edges  ({time.time() - t0:.0f} s)", flush=True)
    src = np.concatenate(src_l)
    dst = np.concatenate(dst_l)
    E = np.concatenate(e_l)
    del src_l, dst_l, e_l
    if (src == dst).any():
        raise SystemExit("[error] self-loops in exn_edges.csv")
    order = np.argsort(src, kind="stable")
    src, dst, E = src[order], dst[order], E[order]
    indptr = np.zeros(len(nodes) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(np.bincount(src, minlength=len(nodes)))
    # every edge must have its reverse (the extractor writes both directions)
    fwd = src.astype(np.int64) * len(nodes) + dst
    rev = dst.astype(np.int64) * len(nodes) + src
    if not np.array_equal(np.sort(fwd), np.sort(rev)):
        raise SystemExit("[error] edge table is not symmetric")
    del fwd, rev
    etrain = train[src]
    edge_stats = {}
    for j, c in enumerate(EDGE_COLS):
        if is_binary(c):
            continue
        v = E[etrain, j]
        edge_stats[c] = {"mean": float(v.mean()), "std": float(v.std()) or 1.0}

    # ---- write
    np.savez(work / "cache" / "nodes.npz", X=X, field=fcode.astype(np.int32), well=wcode,
             split=split_code, target_ok=nodes.target_ok.to_numpy(),
             cell_id=nodes.cell_id.to_numpy(np.int32),
             **{c: nodes[c].to_numpy(np.int8) for c in label_cols})
    np.savez(work / "cache" / "edges.npz", indptr=indptr, dst=dst.astype(np.int32), E=E)
    meta = {"node_cols": input_cols, "label_cols": label_cols, "node_stats": node_stats,
            "edge_cols": EDGE_COLS, "edge_stats": edge_stats,
            "wells_sha256": rules["wells_sha256"], "fields": fields, "wells": well_names,
            "roles": schema["roles"], "schema_nodes": schema["nodes"],
            "schema_edges": schema["edges"],
            "n_nodes": int(len(nodes)), "n_edges": int(len(dst)),
            "transforms": {"log1p": sorted(LOG1P_COLS), "log": sorted(LOG_COLS),
                           "mean_neighbor_dist_px": "NaN -> field d_max, then log1p",
                           "elongation": "NaN -> 1, then log"},
            "built": datetime.now().isoformat(timespec="seconds")}
    (work / "cache" / "meta.json").write_text(json.dumps(meta, indent=2))

    print("\n" + "=" * 72)
    print(f"Cells {len(nodes):,}  |  directed edges {len(dst):,}  |  "
          f"targets (unfragmented) {int(nodes.target_ok.sum()):,}")
    print(f"Wells {len(well_names)}  |  fields {len(fields)}  |  "
          f"split sha256 {rules['wells_sha256'][:12]}")
    print(f"Wrote {work / 'split'} and {work / 'cache'}  ({(time.time() - t0) / 60:.1f} min)")
    print("=" * 72)


if __name__ == "__main__":
    main()
