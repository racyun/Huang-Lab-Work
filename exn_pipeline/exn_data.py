"""Shared definitions for the ExN cell-classification experiments.

Everything that decides a result lives here, so the GNN, the baselines, the
leakage checks and the scorer cannot drift apart: the split rules, how node
inputs are built for each version, how a target cell's own values are hidden,
and how predictions are graded.

Four experiments = {--label tagln_high, red} x {--version A, B}.

    Version A   inputs = morphology / spatial features of the target cell and
                of every cell in its 2-hop neighbourhood, plus edge features
    Version B   A + the neighbours' marker readouts and labels
                (VEcad_bright_edge_fraction, label_vecad_low, label_tagln_high,
                label_red). The target cell's own copies are zeroed in the
                input, and an is_center flag marks it.

--hide-touching-vecad (Version B only) additionally zeroes the VE-cad-derived
columns of every neighbour whose mask touches the target: those cells share a
junction with the target, so their VE-cad edge fraction partly measures the
target's own membrane.

Layout of the work directory (written by prepare_exn_training.py):

    <work>/split/wells.csv      one row per well (batch/well) and its split
    <work>/split/rules.json     RULES + sha256 of wells.csv
    <work>/cache/nodes.npz      transformed node table, labels, split, field ids
    <work>/cache/edges.npz      CSR adjacency (indptr, dst) + edge features
    <work>/cache/meta.json      column names, standardisation stats (train wells)
    <work>/runs/<run>/          one GNN run
    <work>/baselines/<name>/    one baseline
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exn_columns import LABELS, select_columns  # noqa: E402

# Fixed before any result is seen. Changing one changes the split hash, and
# every later step refuses to run against a split it was not built on.
RULES = {
    "split_unit": "well (batch + well): all fields of one well share a split",
    "fractions": {"train": 0.6, "val": 0.2, "test": 0.2},
    "balanced_by": "batch: wells are shuffled and divided within each batch",
    "small_batches": "1 well -> train; 2 wells -> train + test",
    "seed": 0,
    "hops": 2,
    "target_rule": "n_fragments == 1 (fragmented masks stay in the graph as neighbours)",
    "standardise_with": "training wells only",
    "primary_metric": "AUPRC (early stopping and model choice); AUROC and "
                      "within-field AUROC reported alongside",
}
SPLITS = ("train", "val", "test")
SPLIT_CODE = {s: i for i, s in enumerate(SPLITS)}

# Columns that are 0/1 and are not standardised.
BINARY_PREFIXES = ("label_", "ecm_")
BINARY_COLS = {"near_border", "treatment_nicotine", "is_mutual", "touching"}

# How each numeric column is transformed before standardising. Skewed, positive
# quantities are log1p'd; elongation (>= 1) is logged so 1 maps to 0.
LOG1P_COLS = {"area_px", "perimeter_px", "major_axis_px", "minor_axis_px",
              "mean_neighbor_dist_px", "nn1_dist_px", "dist_to_border_px", "n_contacts",
              "stiffness_kpa", "dist_px", "shared_boundary_px"}
LOG_COLS = {"elongation"}

# Edge features, in this order, everywhere.
EDGE_COLS = ["dist_norm", "dist_px", "is_mutual", "shared_boundary_px", "touching"]
TOUCHING_EDGE_COL = EDGE_COLS.index("touching")

# The neighbour columns that derive from VE-cad, hidden for touching
# neighbours under --hide-touching-vecad.
VECAD_DERIVED = ["VEcad_bright_edge_fraction", "label_vecad_low", "label_red"]


def is_binary(col: str) -> bool:
    return col in BINARY_COLS or col.startswith(BINARY_PREFIXES)


def default_work_dir() -> Path:
    if "EXN_WORK" in os.environ:
        return Path(os.environ["EXN_WORK"]).expanduser()
    studio = Path("/teamspace/studios/this_studio")
    return (studio if studio.exists() else Path.home()) / "exn_training"


def default_features_dir() -> Path:
    studio = Path("/teamspace/studios/this_studio")
    return (studio if studio.exists() else Path.home()) / "exn_features"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_name(label: str, version: str, hide_touching_vecad: bool = False,
             use_conditions: bool = False) -> str:
    name = f"{label}_{version}"
    if hide_touching_vecad:
        name += "_nojunction"
    if use_conditions:
        name += "_cond"
    return name


# ------------------------------------------------------------------- the cache
@dataclass
class Cache:
    work_dir: Path
    meta: dict
    X: np.ndarray            # [n_nodes, n_cols] transformed (not standardised)
    cols: list
    labels: dict             # label column -> int8 [n_nodes]
    field: np.ndarray        # int32 field index
    well: np.ndarray         # int32 well index
    split: np.ndarray        # int8 split code
    target_ok: np.ndarray    # bool
    indptr: np.ndarray       # int64 [n_nodes + 1]
    dst: np.ndarray          # int32 [n_edges]
    E: np.ndarray            # float32 [n_edges, len(EDGE_COLS)] standardised

    @property
    def n_nodes(self) -> int:
        return len(self.field)

    def targets(self, split: str) -> np.ndarray:
        """Global node indices of the cells graded in this split."""
        return np.flatnonzero((self.split == SPLIT_CODE[split]) & self.target_ok)

    def y(self, label: str) -> np.ndarray:
        return self.labels[LABELS[label]]


def verify_split(work_dir: Path) -> dict:
    rules_path = work_dir / "split" / "rules.json"
    if not rules_path.exists():
        raise SystemExit(f"[error] {rules_path} not found. Run prepare_exn_training.py first.")
    saved = json.loads(rules_path.read_text())
    if saved["rules"] != RULES:
        raise SystemExit("[error] split/rules.json does not match RULES in exn_data.py")
    if saved["wells_sha256"] != sha256_file(work_dir / "split" / "wells.csv"):
        raise SystemExit("[error] split/wells.csv changed after the split was saved")
    return saved


def load_cache(work_dir: Path) -> Cache:
    saved = verify_split(work_dir)
    meta = json.loads((work_dir / "cache" / "meta.json").read_text())
    if meta["wells_sha256"] != saved["wells_sha256"]:
        raise SystemExit("[error] the cache was built on a different split; rerun "
                         "prepare_exn_training.py --force")
    nz = np.load(work_dir / "cache" / "nodes.npz")
    ez = np.load(work_dir / "cache" / "edges.npz")
    E = ez["E"].astype(np.float32)
    for j, c in enumerate(EDGE_COLS):
        if not is_binary(c):
            s = meta["edge_stats"][c]
            E[:, j] = (E[:, j] - s["mean"]) / s["std"]
    return Cache(work_dir=work_dir, meta=meta, X=nz["X"], cols=list(meta["node_cols"]),
                 labels={c: nz[c] for c in meta["label_cols"]}, field=nz["field"],
                 well=nz["well"], split=nz["split"], target_ok=nz["target_ok"],
                 indptr=ez["indptr"], dst=ez["dst"], E=np.nan_to_num(E, nan=0.0))


# ------------------------------------------------------------- node inputs
@dataclass
class Inputs:
    """The node matrix one experiment sees, and what to hide at the target."""
    label: str
    version: str
    hide_touching_vecad: bool
    use_conditions: bool
    cols: list               # columns of V, in order
    V: np.ndarray            # [n_nodes, len(cols)] standardised, NaN -> 0
    hide_at_center: np.ndarray   # column indices zeroed for the target cell
    vecad_cols: np.ndarray       # column indices zeroed for touching neighbours
    n_flags: int             # indicator columns appended per subgraph

    @property
    def in_dim(self) -> int:
        return len(self.cols) + self.n_flags

    def describe(self) -> dict:
        return {"label": self.label, "version": self.version,
                "hide_touching_vecad": self.hide_touching_vecad,
                "use_conditions": self.use_conditions, "node_cols": self.cols,
                "hidden_at_target": [self.cols[i] for i in self.hide_at_center],
                "flags": (["is_center"] if self.version == "B" else []) +
                         (["vecad_hidden"] if self.hide_touching_vecad else [])}


def build_inputs(cache: Cache, label: str, version: str, hide_touching_vecad: bool = False,
                 use_conditions: bool = False) -> Inputs:
    if hide_touching_vecad and version != "B":
        raise ValueError("--hide-touching-vecad only applies to Version B")
    schema = {"roles": cache.meta["roles"], "nodes": cache.meta["schema_nodes"],
              "edges": cache.meta["schema_edges"]}
    plan = select_columns(schema, label, version, use_conditions)
    cols = list(dict.fromkeys(plan.self_inputs + plan.neighbor_inputs))
    missing = [c for c in cols if c not in cache.cols]
    if missing:
        raise SystemExit(f"[error] columns missing from the cache: {missing}")
    idx = [cache.cols.index(c) for c in cols]
    V = cache.X[:, idx].astype(np.float32, copy=True)
    for j, c in enumerate(cols):
        if not is_binary(c):
            s = cache.meta["node_stats"][c]
            V[:, j] = (V[:, j] - s["mean"]) / s["std"]
    V = np.nan_to_num(V, nan=0.0)

    # Anything a neighbour may show but the target may not: its own markers and labels.
    hide = [j for j, c in enumerate(cols) if c not in plan.self_inputs]
    if label and LABELS[label] in cols and cols.index(LABELS[label]) not in hide:
        raise AssertionError("target label would be visible at the target cell")
    if version == "A" and hide:
        raise AssertionError(f"Version A should hide nothing, got {[cols[j] for j in hide]}")
    vec = [cols.index(c) for c in VECAD_DERIVED if c in cols] if hide_touching_vecad else []
    n_flags = (1 if version == "B" else 0) + (1 if hide_touching_vecad else 0)
    return Inputs(label, version, hide_touching_vecad, use_conditions, cols, V,
                  np.array(hide, dtype=np.int64), np.array(vec, dtype=np.int64), n_flags)


# ------------------------------------------------------------ ego graphs
def ego_arrays(cache: Cache, inputs: Inputs, center: int, hops: int = RULES["hops"],
               V: np.ndarray | None = None):
    """The target's k-hop neighbourhood as arrays, with hidden values removed.

    Returns (x [m, in_dim], edge_index [2, e], edge_attr [e, 5], center_local).
    The subgraph is the induced subgraph on all cells within `hops` edges, so a
    `hops`-layer GNN computes exactly what it would on the whole field.
    Hiding happens on the input, before any message passing, so the target's
    own values cannot reach its prediction by way of a neighbour and back.
    """
    V = inputs.V if V is None else V
    ip, dst = cache.indptr, cache.dst
    frontier = np.array([center], dtype=np.int64)
    seen = frontier
    for _ in range(hops):
        if len(frontier) == 0:
            break
        nb = np.concatenate([dst[ip[u]:ip[u + 1]] for u in frontier]).astype(np.int64)
        frontier = np.setdiff1d(nb, seen)
        seen = np.union1d(seen, frontier)
    sub = seen                                          # sorted global ids
    c = int(np.searchsorted(sub, center))

    starts, ends = ip[sub], ip[sub + 1]
    lens = (ends - starts).astype(np.int64)
    total = int(lens.sum())
    if total:
        base = np.repeat(starts - np.concatenate(([0], np.cumsum(lens)[:-1])), lens)
        eidx = base + np.arange(total)
        d = dst[eidx].astype(np.int64)
        pos = np.minimum(np.searchsorted(sub, d), len(sub) - 1)
        keep = sub[pos] == d
        src_local = np.repeat(np.arange(len(sub)), lens)[keep]
        edge_index = np.stack([src_local, pos[keep]])
        edge_attr = cache.E[eidx[keep]]
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, len(EDGE_COLS)), dtype=np.float32)

    x = V[sub].copy()
    flags = []
    if inputs.version == "B":
        if len(inputs.hide_at_center):
            x[c, inputs.hide_at_center] = 0.0
        is_center = np.zeros((len(sub), 1), dtype=np.float32)
        is_center[c] = 1.0
        flags.append(is_center)
    if inputs.hide_touching_vecad:
        # first-hop neighbours whose mask touches the target
        e0, e1 = ip[center], ip[center + 1]
        touch = dst[e0:e1][cache.E[e0:e1, TOUCHING_EDGE_COL] > _touch_threshold(cache)]
        rows = np.searchsorted(sub, touch.astype(np.int64))
        hidden = np.zeros((len(sub), 1), dtype=np.float32)
        if len(rows) and len(inputs.vecad_cols):
            x[np.ix_(rows, inputs.vecad_cols)] = 0.0
            hidden[rows] = 1.0
        flags.append(hidden)
    if flags:
        x = np.concatenate([x] + flags, axis=1)
    return x.astype(np.float32), edge_index, edge_attr, c


def _touch_threshold(cache: Cache) -> float:
    # touching is binary and not standardised; anything above 0.5 is a contact
    return 0.5


def ego_graph(cache: Cache, inputs: Inputs, center: int, y: float = float("nan"),
              V: np.ndarray | None = None):
    import torch
    from torch_geometric.data import Data

    x, ei, ea, c = ego_arrays(cache, inputs, center, V=V)
    return Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(ei),
                edge_attr=torch.from_numpy(np.ascontiguousarray(ea)),
                center_index=torch.tensor([c], dtype=torch.long),
                y=torch.tensor([y], dtype=torch.float32),
                node_id=torch.tensor([center], dtype=torch.long))


class TargetCells:
    """torch Dataset of ego graphs, one per target cell."""

    def __init__(self, cache: Cache, inputs: Inputs, nodes: np.ndarray, y: np.ndarray):
        self.cache, self.inputs = cache, inputs
        self.nodes = np.asarray(nodes, dtype=np.int64)
        self.yv = np.asarray(y, dtype=np.float32)

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, i):
        return ego_graph(self.cache, self.inputs, int(self.nodes[i]), float(self.yv[i]))


# ---------------------------------------------------------------- grading
def _auc_parts(y: np.ndarray, p: np.ndarray, group: np.ndarray):
    """Per-group Mann-Whitney U and positive x negative pair counts."""
    import pandas as pd

    df = pd.DataFrame({"y": y.astype(np.int64), "p": p, "g": group})
    df["r"] = df.groupby("g")["p"].rank(method="average")
    df["ry"] = df.r * df.y
    agg = df.groupby("g").agg(n=("y", "size"), npos=("y", "sum"), rsum=("ry", "sum"))
    agg["nneg"] = agg.n - agg.npos
    agg["U"] = agg.rsum - agg.npos * (agg.npos + 1) / 2.0
    agg["pairs"] = (agg.npos * agg.nneg).astype(float)
    return agg


def within_field_auroc(y, p, field) -> float:
    """AUROC computed inside each field, pooled over fields by pair count.

    Positive rates differ by field and condition, so overall AUROC can be won by
    recognising the field. Pooling within-field AUROCs only credits a model for
    ranking cells of the same field correctly.
    """
    a = _auc_parts(np.asarray(y), np.asarray(p), np.asarray(field))
    return float(a.U.sum() / a.pairs.sum()) if a.pairs.sum() > 0 else float("nan")


def metrics(y, p, field) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    return {"auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "within_field_auroc": within_field_auroc(y, p, field),
            "prevalence": float(y.mean()), "n_cells": int(len(y)),
            "n_positive": int(y.sum())}


def best_f1_threshold(y, p) -> float:
    from sklearn.metrics import precision_recall_curve

    prec, rec, thr = precision_recall_curve(np.asarray(y).astype(int), p)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(thr[int(np.argmax(f1[:-1]))]) if len(thr) else 0.5


def threshold_metrics(y, p, thr: float) -> dict:
    y = np.asarray(y).astype(bool)
    yhat = np.asarray(p) >= thr
    tp, fp = int((yhat & y).sum()), int((yhat & ~y).sum())
    fn, tn = int((~yhat & y).sum()), int((~yhat & ~y).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {"threshold": float(thr), "precision": prec, "recall": rec,
            "specificity": spec, "f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "balanced_accuracy": (rec + spec) / 2, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def bootstrap(y, preds: dict, field, well, n_boot: int = 300, seed: int = 0,
              pairs: list | None = None) -> dict:
    """95% CIs that resample WELLS, for each model and for paired differences.

    Cells and fields within one well are not independent, so resampling cells
    would make the intervals too narrow.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y).astype(int)
    well = np.asarray(well)
    field = np.asarray(field)
    wells, inv = np.unique(well, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(wells) + 1))
    members = [order[bounds[i]:bounds[i + 1]] for i in range(len(wells))]
    rng = np.random.default_rng(seed)
    draws = {k: {"auroc": [], "auprc": [], "within_field_auroc": []} for k in preds}
    for _ in range(n_boot):
        pick = rng.integers(0, len(wells), len(wells))
        idx = np.concatenate([members[i] for i in pick])
        # a field drawn twice must count as two fields for the within-field AUROC
        rep = np.repeat(np.arange(len(pick)), [len(members[i]) for i in pick])
        fkey = field[idx].astype(np.int64) * (len(pick) + 1) + rep
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        for k, p in preds.items():
            pp = np.asarray(p)[idx]
            draws[k]["auroc"].append(roc_auc_score(yy, pp))
            draws[k]["auprc"].append(average_precision_score(yy, pp))
            draws[k]["within_field_auroc"].append(within_field_auroc(yy, pp, fkey))
    out = {"models": {}, "differences": {}, "n_boot": n_boot, "unit": "well"}
    for k, d in draws.items():
        out["models"][k] = {m: _ci(v) for m, v in d.items()}
    for a, b in (pairs or []):
        if a in draws and b in draws:
            out["differences"][f"{a} - {b}"] = {
                m: _ci(np.array(draws[a][m]) - np.array(draws[b][m])) for m in draws[a]}
    return out


def _ci(v) -> dict:
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return {"low": float("nan"), "high": float("nan")}
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {"low": float(lo), "high": float(hi), "frac_above_0": float((v > 0).mean())}
