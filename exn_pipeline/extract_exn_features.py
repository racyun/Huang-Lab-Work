#!/usr/bin/env python3
"""ExN dataset -> master node table + master edge table for graph classification.

Reads the pre-segmented ExN35 / ExN39 / ExN41 fields (Carver's Cellpose masks
plus per-cell marker calls) and writes:

    <out>/exn_nodes.csv     one row per cell, every experiment and field
    <out>/exn_edges.csv     one row per directed neighbour relation
    <out>/exn_fields_qc.csv one row per field: counts, graph stats, status
    <out>/exn_issues.csv    one row per problem found (missing files, ID
                            mismatches, invalid values, label inconsistencies)
    <out>/exn_run_log.txt   full console transcript
    <out>/exn_schema.json   every column of both tables, with its meaning and its
                            role (id / label / marker / feature / condition /
                            group / qc); exn_columns.py turns roles plus the
                            --label and --version flags into model inputs

Nothing is re-segmented and nothing already in cells.csv is recomputed: the
existing measurements (TAGLN_high, VEcad_bright_edge_fraction, area_px,
display_class) are carried through unchanged and merged by cell_id. The script
only ADDS what cells.csv lacks — centroid, elongation and other shape
descriptors, border position, contact lengths, and the spatial graph.

Expected layout (recursive; any depth below each ExN folder works):

    <data-root>/
        ExN_field_metrics.csv
        ExN35/<field>/{cells.csv, cell_ids.tif, class_codes.png, percentages.csv, ...}
        ExN39/<field>/...
        ExN41/<field>/...

A field is any directory containing cells.csv. Its identifier is its path
relative to <data-root> (e.g. "ExN35/260929_500kPa_control_C1+C3+F+L_D12_rep2"),
which is exactly the `folder` column of ExN_field_metrics.csv and is unique
across the dataset. A cell's global identifier is "<folder>#<cell_id>".

Neighbourhood definition (unchanged from motifs/stage2_graphs/build_graphs.py):
  * candidate edges = k-nearest neighbours on centroids (k=8), symmetrised by
    union; is_mutual = both cells list each other among their k nearest
  * prune edges longer than d_max = 3 x the field's median nearest-neighbour
    distance, so the cutoff adapts to each field's density
  * dist_norm = dist / d_max
  * near_border = centroid within d_max of the image edge (neighbourhood is
    truncated by the field of view)

Usage (Lightning):
    python exn_pipeline/extract_exn_features.py \
        --data-root /teamspace/studios/this_studio/exn_data \
        --out-dir   /teamspace/studios/this_studio/exn_features
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table

REQUIRED_COLS = ["cell_id", "TAGLN_high", "VEcad_bright_edge_fraction",
                 "area_px", "display_class"]
METRICS_NAME = "ExN_field_metrics.csv"
RED_PREFIX = "Red"
# class_codes.png encoding, verified on the sample field: 0 background, 1 gray, 2 red
CODE_BG, CODE_GRAY, CODE_RED = 0, 1, 2


# ----------------------------------------------------------------------------- paths
def _default_root(sub: str) -> Path:
    studio = Path("/teamspace/studios/this_studio")
    if "EXN_ROOT" in os.environ:
        return Path(os.environ["EXN_ROOT"]).expanduser() / sub
    return (studio if studio.exists() else Path.home()) / sub


class _Tee:
    def __init__(self, path: Path):
        self.f = open(path, "w")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- geometry
def contact_lengths(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared boundary between every pair of touching cells.

    Counts 4-connected pixel pairs whose two pixels belong to different cells.
    Returns (cell_a, cell_b, n_pixel_pairs) with cell_a < cell_b.
    """
    m = mask.astype(np.int64)
    base = int(m.max()) + 1
    keys = []
    for a, b in ((m[:, :-1], m[:, 1:]), (m[:-1, :], m[1:, :])):
        sel = (a != b) & (a > 0) & (b > 0)
        x, y = a[sel], b[sel]
        keys.append(np.minimum(x, y) * base + np.maximum(x, y))
    if not keys or sum(k.size for k in keys) == 0:
        e = np.empty(0, dtype=np.int64)
        return e, e, e
    k, cnt = np.unique(np.concatenate(keys), return_counts=True)
    return k // base, k % base, cnt


def knn_graph(pos: np.ndarray, k: int, d_max_mult: float):
    """k-NN union graph pruned at d_max; matches build_graphs.py edge_rule='knn'.

    Returns (i, j, dist, is_mutual) for undirected edges i < j, plus d_max and the
    per-cell nearest-neighbour distance.
    """
    n = len(pos)
    empty = (np.empty(0, int), np.empty(0, int), np.empty(0), np.empty(0, bool))
    if n < 2:
        return (*empty, float("nan"), np.full(n, np.nan))
    kq = min(k, n - 1)
    dist, idx = cKDTree(pos).query(pos, k=kq + 1)          # column 0 is self
    nn1 = dist[:, 1]
    d_max = d_max_mult * float(np.median(nn1))

    i = np.repeat(np.arange(n), kq)
    j = idx[:, 1:].ravel()
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    key = lo * n + hi
    uniq, counts = np.unique(key, return_counts=True)       # count 2 => both directions
    a, b = uniq // n, uniq % n
    mutual = counts == 2
    d = np.linalg.norm(pos[a] - pos[b], axis=1)
    keep = d <= d_max
    return a[keep], b[keep], d[keep], mutual[keep], d_max, nn1


# ----------------------------------------------------------------------------- one field
def process_field(args) -> dict:
    field_dir, root, k, d_max_mult, vecad_gate, verify_sha, metrics_row = args
    folder = field_dir.relative_to(root).as_posix()
    issues: list[dict] = []

    def issue(sev, check, detail):
        issues.append({"folder": folder, "severity": sev, "check": check, "detail": detail})

    qc = {"folder": folder, "status": "ok"}
    out = {"folder": folder, "nodes": None, "edges": None, "issues": issues, "qc": qc}

    def fail(check, detail):
        issue("error", check, detail)
        qc["status"] = f"skipped: {check}"
        return out

    # ---- required files
    p_cells, p_mask = field_dir / "cells.csv", field_dir / "cell_ids.tif"
    p_codes, p_pct = field_dir / "class_codes.png", field_dir / "percentages.csv"
    for p in (p_cells, p_mask):
        if not p.exists():
            return fail("missing_file", p.name)
    for p in (p_codes, p_pct, field_dir / "original.png", field_dir / "red_gray.png"):
        if not p.exists():
            issue("warning", "missing_file", p.name)

    # ---- cells.csv
    try:
        cells = pd.read_csv(p_cells)
    except Exception as e:
        return fail("unreadable_cells_csv", str(e))
    missing = [c for c in REQUIRED_COLS if c not in cells.columns]
    if missing:
        return fail("missing_columns", ",".join(missing))
    if not cells.cell_id.is_unique:
        return fail("duplicate_cell_id", f"{int(cells.cell_id.duplicated().sum())} duplicates")
    extra = [c for c in cells.columns if c not in REQUIRED_COLS]
    if extra:
        issue("info", "extra_columns_carried", ",".join(extra))

    # ---- mask
    try:
        mask = tifffile.imread(str(p_mask))
    except Exception as e:
        return fail("unreadable_mask", str(e))
    if mask.ndim != 2:
        return fail("mask_not_2d", str(mask.shape))
    H, W = mask.shape
    qc.update(image_h=H, image_w=W)

    # ---- cell IDs: the mask and cells.csv must describe the same set of cells
    mask_ids = np.unique(mask)
    mask_ids = mask_ids[mask_ids > 0]
    csv_ids = cells.cell_id.to_numpy()
    only_mask = np.setdiff1d(mask_ids, csv_ids)
    only_csv = np.setdiff1d(csv_ids, mask_ids)
    if only_mask.size or only_csv.size:
        return fail("cell_id_mismatch",
                    f"{only_mask.size} ids only in mask, {only_csv.size} only in cells.csv "
                    f"(e.g. mask {only_mask[:5].tolist()}, csv {only_csv[:5].tolist()})")

    # ---- the mask's pixel count must equal the reported area for every cell
    cells = cells.set_index("cell_id").loc[np.sort(csv_ids)]
    ids = cells.index.to_numpy()
    area_mask = np.bincount(mask.ravel().astype(np.int64), minlength=int(ids.max()) + 1)[ids]
    bad_area = area_mask != cells.area_px.to_numpy()
    if bad_area.any():
        return fail("area_mismatch", f"{int(bad_area.sum())} cells; area_px != mask pixel "
                                     "count, so cell_id may not map to the same object")

    # ---- value sanity
    frac = cells.VEcad_bright_edge_fraction
    n_bad = int((frac.isna() | (frac < 0) | (frac > 1)).sum())
    if n_bad:
        issue("warning", "vecad_fraction_out_of_range", f"{n_bad} cells NaN or outside [0, 1]")
    if (cells.area_px <= 0).any():
        issue("warning", "nonpositive_area", f"{int((cells.area_px <= 0).sum())} cells")
    tag = cells.TAGLN_high
    if tag.dtype != bool:
        tag = tag.astype(str).str.strip().str.lower().map({"true": True, "false": False})
        if tag.isna().any():
            issue("warning", "tagln_high_unparseable", f"{int(tag.isna().sum())} cells")
    classes = set(cells.display_class.unique())
    known = {c for c in classes if c.startswith(RED_PREFIX) or c.startswith("Gray")}
    if classes - known:
        issue("warning", "unknown_display_class", "; ".join(sorted(classes - known)))
    red = cells.display_class.str.startswith(RED_PREFIX).to_numpy()

    # ---- class_codes.png must agree with display_class, cell by cell
    if p_codes.exists():
        codes = np.array(Image.open(p_codes))
        if codes.shape != mask.shape:
            issue("warning", "class_codes_shape", f"{codes.shape} vs mask {mask.shape}")
        else:
            fg = mask > 0
            lab, cc = mask[fg].astype(np.int64), codes[fg].astype(np.int64)
            n_red_px = np.bincount(lab, weights=(cc == CODE_RED), minlength=int(ids.max()) + 1)[ids]
            frac_red = n_red_px / area_mask
            mixed = int(((frac_red > 0) & (frac_red < 1)).sum())
            if mixed:
                issue("warning", "class_code_mixed_within_cell", f"{mixed} cells")
            disagree = int(((frac_red >= 0.5) != red).sum())
            if disagree:
                issue("warning", "class_code_vs_display_class", f"{disagree} cells disagree")
            if np.isin(np.unique(codes[~fg]), [CODE_BG], invert=True).any():
                issue("warning", "class_code_on_background", "nonzero code outside cells")

    # ---- red must equal the documented gate: TAGLN_high AND fraction <= gate
    gate = tag.fillna(False).to_numpy() & (frac.to_numpy() <= vecad_gate)
    if (gate != red).any():
        issue("warning", "red_gate_mismatch",
              f"{int((gate != red).sum())} cells where red != TAGLN_high & "
              f"VEcad_bright_edge_fraction <= {vecad_gate}")

    # ---- per-field metadata: percentages.csv first, field_metrics.csv as fallback
    meta = {}
    if p_pct.exists():
        try:
            meta = pd.read_csv(p_pct).iloc[0].to_dict()
        except Exception as e:
            issue("warning", "unreadable_percentages", str(e))
    if metrics_row is not None:
        for key, val in metrics_row.items():
            meta.setdefault(key, val)
    else:
        issue("warning", "not_in_field_metrics", "folder absent from ExN_field_metrics.csv")
    if meta.get("folder") not in (None, folder):
        issue("warning", "folder_name_mismatch", f"percentages says {meta.get('folder')}")
    for key, got in (("total_cells", len(cells)), ("red_cells", int(red.sum())),
                     ("red_cell_area_px", int(cells.area_px[red].sum()))):
        want = meta.get(key)
        if want is not None and not pd.isna(want) and int(want) != got:
            issue("warning", f"{key}_mismatch", f"metadata {int(want)} vs files {got}")
    if verify_sha:
        for fname, key in (("cell_ids.tif", "instance_tiff_sha256"),
                           ("original.png", "source_sha256")):
            p, want = field_dir / fname, meta.get(key)
            if p.exists() and isinstance(want, str) and _sha256(p) != want:
                issue("warning", "checksum_mismatch", f"{fname} differs from metadata {key}")

    # ---- shape descriptors (only what cells.csv lacks)
    props = pd.DataFrame(regionprops_table(
        mask, properties=("label", "centroid", "axis_major_length", "axis_minor_length",
                          "perimeter", "solidity", "bbox"))).set_index("label").loc[ids]
    cy, cx = props["centroid-0"].to_numpy(), props["centroid-1"].to_numpy()
    major, minor = props.axis_major_length.to_numpy(), props.axis_minor_length.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        elong = np.where(minor > 0, major / minor, np.nan)
    if np.isnan(elong).any():
        issue("info", "elongation_undefined", f"{int(np.isnan(elong).sum())} cells (minor axis 0)")
    touches = ((props["bbox-0"] == 0) | (props["bbox-1"] == 0) |
               (props["bbox-2"] == H) | (props["bbox-3"] == W)).to_numpy()
    dist_border = np.minimum.reduce([cx, cy, W - cx, H - cy])

    # cells whose mask is several disconnected pieces: centroid and shape are of the union
    n_frag = np.ones(len(ids), dtype=int)
    objs = ndimage.find_objects(mask)
    id_to_row = {int(c): r for r, c in enumerate(ids)}
    for lbl, sl in enumerate(objs, start=1):
        if sl is None or lbl not in id_to_row:
            continue
        _, nf = ndimage.label(mask[sl] == lbl)
        n_frag[id_to_row[lbl]] = nf
    if (n_frag > 1).any():
        issue("info", "fragmented_masks", f"{int((n_frag > 1).sum())} cells split into >1 piece")

    # ---- spatial graph
    pos = np.column_stack([cx, cy])
    a, b, d, mutual, d_max, nn1 = knn_graph(pos, k, d_max_mult)
    near_border = dist_border < d_max if np.isfinite(d_max) else np.zeros(len(ids), bool)

    # contact lengths, keyed by cell-id pair
    ca, cb, clen = contact_lengths(mask)
    contact = {(int(x), int(y)): int(n) for x, y, n in zip(ca, cb, clen)}
    n_contacts = np.zeros(len(ids), dtype=int)
    for (x, y) in contact:
        n_contacts[id_to_row[x]] += 1
        n_contacts[id_to_row[y]] += 1
    ida, idb = ids[a], ids[b]
    shared = np.array([contact.get((int(min(p, q)), int(max(p, q))), 0)
                       for p, q in zip(ida, idb)], dtype=int)
    graph_pairs = {(int(min(p, q)), int(max(p, q))) for p, q in zip(ida, idb)}
    missed = sum(1 for pr in contact if pr not in graph_pairs)

    deg = np.bincount(np.concatenate([a, b]), minlength=len(ids))
    dsum = np.bincount(np.concatenate([a, b]), weights=np.concatenate([d, d]), minlength=len(ids))
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_nd = np.where(deg > 0, dsum / deg, np.nan)
    if (deg == 0).any():
        issue("info", "isolated_cells", f"{int((deg == 0).sum())} cells with no neighbour")

    qc.update(n_cells=len(ids), n_red=int(red.sum()), n_tagln_high=int(tag.fillna(False).sum()),
              n_edges_undirected=int(len(a)), mean_degree=float(deg.mean()),
              d_max_px=d_max, median_nn_px=float(np.median(nn1)) if len(nn1) else np.nan,
              n_near_border=int(near_border.sum()), n_touch_border=int(touches.sum()),
              n_isolated=int((deg == 0).sum()), n_contact_pairs=len(contact),
              contact_pairs_not_in_graph=missed, n_fragmented=int((n_frag > 1).sum()))

    # ---- node rows
    gid = [f"{folder}#{c}" for c in ids]
    nodes = pd.DataFrame({
        "global_cell_id": gid, "folder": folder, "cell_id": ids,
        # existing measurements, unchanged
        **{c: cells[c].to_numpy() for c in cells.columns if c not in ("TAGLN_high",)},
        "TAGLN_high": tag.to_numpy(),
        # labels (derived from the existing columns, not recomputed from pixels)
        "label_tagln_high": tag.fillna(False).astype(int).to_numpy(),
        "label_vecad_low": (frac.to_numpy() <= vecad_gate).astype(int),
        "label_red": red.astype(int),
        # new morphology
        "centroid_x": cx, "centroid_y": cy, "elongation": elong,
        "major_axis_px": major, "minor_axis_px": minor,
        "perimeter_px": props.perimeter.to_numpy(), "solidity": props.solidity.to_numpy(),
        "n_fragments": n_frag,
        # new spatial
        "neighbor_count": deg, "mean_neighbor_dist_px": mean_nd, "nn1_dist_px": nn1,
        "n_contacts": n_contacts, "touches_border": touches.astype(int),
        "near_border": near_border.astype(int), "dist_to_border_px": dist_border,
        "image_h": H, "image_w": W, "field_d_max_px": d_max,
    })
    for key in ("dataset", "batch", "stiffness", "treatment", "ECM", "well", "field",
                "source_id", "drive_field_folder"):
        nodes[key] = meta.get(key, np.nan)

    # ---- edge rows, both directions so "neighbours of X" is one filter on src
    e = pd.DataFrame({
        "src_global_id": [gid[i] for i in a] + [gid[j] for j in b],
        "dst_global_id": [gid[j] for j in b] + [gid[i] for i in a],
        "folder": folder,
        "src_cell_id": np.concatenate([ida, idb]), "dst_cell_id": np.concatenate([idb, ida]),
        "dist_px": np.concatenate([d, d]),
        "dist_norm": np.concatenate([d, d]) / d_max if np.isfinite(d_max) else np.nan,
        "is_mutual": np.concatenate([mutual, mutual]).astype(int),
        "shared_boundary_px": np.concatenate([shared, shared]),
    })
    e["touching"] = (e.shared_boundary_px > 0).astype(int)

    out["nodes"], out["edges"] = nodes, e
    if any(i["severity"] == "warning" for i in issues):
        qc["status"] = "ok_with_warnings"
    return out


# ----------------------------------------------------------------------------- main
# Every column carries a role. Model scripts pick their inputs by role (see
# exn_columns.py), never by hand, so a label or marker column cannot slip into
# the inputs of its own cell.
ROLES = {
    "id":        "identifier; never a model input",
    "label":     "candidate training target, chosen at model time with --label. The "
                 "target cell's own value is never an input; a neighbour's value is a "
                 "Version B input only",
    "marker":    "marker readout of the cell (TAGLN / VE-cad). The target cell's own "
                 "value is never an input, because both labels are defined from these "
                 "columns; a neighbour's value is a Version B input only",
    "feature":   "morphology or spatial-graph descriptor; an input in Versions A and B, "
                 "for the target cell and its neighbours",
    "condition": "experimental condition (stiffness, treatment, ECM); optional input, "
                 "chosen with --use-conditions",
    "group":     "batch / well / field bookkeeping for splits and confounding checks; "
                 "never a model input",
    "qc":        "quality-control or geometry bookkeeping; never a model input",
    "unreviewed": "column found in cells.csv that this script does not know; never a "
                  "model input until someone assigns it a role",
}


def _c(role: str, desc: str) -> dict:
    assert role in ROLES, role
    return {"role": role, "description": desc}


SCHEMA = {
    "nodes": {
        "global_cell_id": _c("id", "unique cell id: '<folder>#<cell_id>'"),
        "folder": _c("id", "field identifier: path relative to data root; "
                           "matches ExN_field_metrics.folder"),
        "cell_id": _c("id", "cell id within its field; equals the label value in cell_ids.tif"),
        # cells.csv, carried through unchanged
        "TAGLN_high": _c("marker", "cells.csv, unchanged: TAGLN-dominance call (True/False)"),
        "VEcad_bright_edge_fraction": _c("marker", "cells.csv, unchanged: fraction of the "
                                                   "cell edge that is VE-cad bright"),
        "display_class": _c("marker", "cells.csv, unchanged: red/gray class text"),
        "area_px": _c("feature", "cells.csv, unchanged (verified equal to the mask pixel count)"),
        # 0/1 recodings of the cells.csv columns above
        "label_tagln_high": _c("label", "1 if TAGLN_high is True (--label tagln_high)"),
        "label_red": _c("label", "1 if display_class is Red = TAGLN_high AND VE-cad-low "
                                 "(--label red; what class_codes.png and red_gray.png encode)"),
        "label_vecad_low": _c("marker", "1 if VEcad_bright_edge_fraction <= gate (default 0.2); "
                                        "half of the red definition, so not offered as a target"),
        # new morphology
        "elongation": _c("feature", "major / minor axis of the equal-moment ellipse (as before)"),
        "major_axis_px": _c("feature", "equal-moment ellipse major axis, px"),
        "minor_axis_px": _c("feature", "equal-moment ellipse minor axis, px"),
        "perimeter_px": _c("feature", "mask perimeter, px"),
        "solidity": _c("feature", "area / convex-hull area (1 = convex)"),
        # new spatial
        "neighbor_count": _c("feature", "graph degree after the d_max prune"),
        "mean_neighbor_dist_px": _c("feature", "mean centroid distance to graph neighbours, px"),
        "nn1_dist_px": _c("feature", "distance to the nearest other centroid, px"),
        "n_contacts": _c("feature", "number of other masks this mask physically touches"),
        "near_border": _c("feature", "1 if centroid within d_max of the image edge (previous "
                                     "on_border); tells the model neighbor_count is truncated"),
        "dist_to_border_px": _c("feature", "centroid distance to the nearest image edge, px"),
        # geometry and segmentation bookkeeping
        "centroid_x": _c("qc", "column coordinate of the mask centroid, px (absolute position "
                               "in the field carries no biology)"),
        "centroid_y": _c("qc", "row coordinate of the mask centroid, px"),
        "n_fragments": _c("qc", "connected pieces in this cell's mask (1 normally)"),
        "touches_border": _c("qc", "1 if any mask pixel lies on the image edge"),
        "image_h": _c("qc", "field height, px"),
        "image_w": _c("qc", "field width, px"),
        "field_d_max_px": _c("qc", "this field's edge-length cutoff, px"),
        # metadata
        "stiffness": _c("group", "substrate stiffness as text, e.g. '500kPa' "
                                 "(use stiffness_kpa as the input)"),
        "stiffness_kpa": _c("condition", "numeric stiffness parsed from `stiffness`, kPa"),
        "treatment": _c("group", "control / nicotine as text (use treatment_nicotine "
                                 "as the input)"),
        "treatment_nicotine": _c("condition", "1 if treatment is nicotine, 0 if control"),
        "ECM": _c("group", "ECM composition string (use the ecm_<X> columns as inputs)"),
        "dataset": _c("group", "ExN35 / ExN39 / ExN41"),
        "batch": _c("group", "imaging batch; split and confounding checks only"),
        "well": _c("group", "well; split unit"),
        "field": _c("group", "field within the well"),
        "source_id": _c("group", "metadata"),
        "drive_field_folder": _c("group", "metadata"),
        # "ecm_<X>" columns are added in main() once the ECM components are known
    },
    "edges": {
        "src_global_id": _c("id", "target cell"),
        "dst_global_id": _c("id", "neighbour cell"),
        "folder": _c("id", "field"),
        "src_cell_id": _c("id", "target cell_id"),
        "dst_cell_id": _c("id", "neighbour cell_id"),
        "dist_px": _c("feature", "centroid distance, px"),
        "dist_norm": _c("feature", "dist_px / field_d_max_px"),
        "is_mutual": _c("feature", "1 if each cell is in the other's k nearest neighbours"),
        "shared_boundary_px": _c("feature", "4-connected pixel pairs on the shared mask "
                                            "boundary (0 = not touching)"),
        "touching": _c("feature", "1 if the two masks share a boundary"),
    },
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", type=Path, default=_default_root("exn_data"))
    ap.add_argument("--out-dir", type=Path, default=_default_root("exn_features"))
    ap.add_argument("--metrics", type=Path, default=None,
                    help=f"path to {METRICS_NAME} (default: searched for under --data-root)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--d-max-mult", type=float, default=3.0)
    ap.add_argument("--vecad-gate", type=float, default=0.2,
                    help="VEcad_bright_edge_fraction at or below which a cell is VE-cad-low; "
                         "0.2 reproduces every red call in the sample field")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--no-sha", action="store_true", help="skip checksum verification")
    ap.add_argument("--parquet", action="store_true", help="also write .parquet copies")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N fields")
    args = ap.parse_args()

    root = args.data_root.expanduser().resolve()
    out = args.out_dir.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    tee = _Tee(out / "exn_run_log.txt")
    sys.stdout = tee
    t0 = time.time()
    try:
        print(f"extract_exn_features.py  {datetime.now():%Y-%m-%d %H:%M:%S}")
        print(f"  data root : {root}\n  out dir   : {out}")
        print(f"  k={args.k}  d_max_mult={args.d_max_mult}  vecad_gate={args.vecad_gate}  "
              f"workers={args.workers}  sha={'off' if args.no_sha else 'on'}")
        if not root.exists():
            raise SystemExit(f"[error] data root does not exist: {root}")

        # ---- metadata table
        mpath = args.metrics
        if mpath is None:
            hits = sorted(root.glob(METRICS_NAME)) + sorted(root.glob(f"*/{METRICS_NAME}"))
            mpath = hits[0] if hits else None
        metrics = pd.DataFrame()
        if mpath and Path(mpath).exists():
            metrics = pd.read_csv(mpath)
            print(f"  metrics   : {mpath}  ({len(metrics)} fields listed)")
        else:
            print(f"  [warn] {METRICS_NAME} not found; using per-field percentages.csv only")
        mrows = {r["folder"]: r for r in metrics.to_dict("records")} if len(metrics) else {}

        # ---- discover fields
        fields = sorted({p.parent for p in root.rglob("cells.csv")})
        if args.limit:
            fields = fields[:args.limit]
        found = {f.relative_to(root).as_posix() for f in fields}
        print(f"\nDiscovered {len(fields)} field folder(s) containing cells.csv")
        by_ds = pd.Series([f.split("/")[0] for f in found]).value_counts()
        for ds, n in by_ds.items():
            print(f"  {ds}: {n}")
        pre_issues = []
        if mrows and not args.limit:
            for f in sorted(set(mrows) - found):
                pre_issues.append({"folder": f, "severity": "error", "check": "field_not_on_disk",
                                   "detail": "listed in ExN_field_metrics.csv, no cells.csv found"})
            if pre_issues:
                print(f"  [warn] {len(pre_issues)} field(s) listed in metrics but missing on disk")

        # ---- process
        jobs = [(f, root, args.k, args.d_max_mult, args.vecad_gate, not args.no_sha,
                 mrows.get(f.relative_to(root).as_posix())) for f in fields]
        results = []
        print(f"\nProcessing {len(jobs)} fields...")
        with Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(process_field, jobs), 1):
                results.append(r)
                if i % 10 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)} fields  ({100 * i / len(jobs):.0f}%)  "
                          f"{(time.time() - t0) / 60:.1f} min", flush=True)

        results.sort(key=lambda r: r["folder"])
        nodes = pd.concat([r["nodes"] for r in results if r["nodes"] is not None], ignore_index=True)
        edges = pd.concat([r["edges"] for r in results if r["edges"] is not None], ignore_index=True)
        qc = pd.DataFrame([r["qc"] for r in results])
        issues = pd.DataFrame(pre_issues + [i for r in results for i in r["issues"]],
                              columns=["folder", "severity", "check", "detail"])

        # ---- derived metadata columns
        nodes["stiffness_kpa"] = pd.to_numeric(
            nodes["stiffness"].astype(str).str.extract(r"([\d.]+)")[0], errors="coerce")
        tr = nodes["treatment"].astype(str).str.strip().str.lower()
        nodes["treatment_nicotine"] = tr.map({"nicotine": 1, "control": 0})   # NaN if unknown
        unknown = nodes["treatment_nicotine"].isna() & nodes["treatment"].notna()
        if unknown.any():
            odd = sorted(tr[unknown].unique())
            print(f"  [warn] treatment values not control/nicotine: {odd} "
                  f"(treatment_nicotine left blank)")
        ecm_src = metrics["ECM"] if "ECM" in metrics else nodes["ECM"]
        tokens = sorted({t for s in ecm_src.dropna().astype(str) for t in s.split("+") if t})
        for t in tokens:
            nodes[f"ecm_{t}"] = nodes["ECM"].astype(str).str.split("+").apply(lambda s: int(t in s))

        # ---- integrity checks on the assembled tables
        if not nodes.global_cell_id.is_unique:
            raise SystemExit("[error] global_cell_id not unique — refusing to write")
        known = set(nodes.global_cell_id)
        dangling = (~edges.src_global_id.isin(known)) | (~edges.dst_global_id.isin(known))
        if dangling.any():
            raise SystemExit(f"[error] {int(dangling.sum())} edges reference unknown cells")

        # ---- schema for this run: every written column gets exactly one role
        schema = {"roles": ROLES, "nodes": dict(SCHEMA["nodes"]), "edges": dict(SCHEMA["edges"]),
                  "run": {"vecad_gate": args.vecad_gate, "k": args.k,
                          "d_max_mult": args.d_max_mult}}
        for t in tokens:
            schema["nodes"][f"ecm_{t}"] = _c("condition", f"1 if ECM component {t} is present")
        for col in nodes.columns:
            if col not in schema["nodes"]:
                schema["nodes"][col] = _c("unreviewed", "extra cells.csv column, carried "
                                                        "through unchanged")
                print(f"  [warn] column '{col}' has no role; tagged 'unreviewed' "
                      f"(excluded from model inputs)")
        for table, df in (("nodes", nodes), ("edges", edges)):
            missing = [c for c in schema[table] if c not in df.columns]
            if missing:
                raise SystemExit(f"[error] schema lists {table} columns not written: {missing}")
            schema[table] = {c: schema[table][c] for c in df.columns}   # file column order

        # ---- write
        nodes.to_csv(out / "exn_nodes.csv", index=False)
        edges.to_csv(out / "exn_edges.csv", index=False)
        qc.to_csv(out / "exn_fields_qc.csv", index=False)
        issues.to_csv(out / "exn_issues.csv", index=False)
        (out / "exn_schema.json").write_text(json.dumps(schema, indent=2))
        if args.parquet:
            nodes.to_parquet(out / "exn_nodes.parquet", index=False)
            edges.to_parquet(out / "exn_edges.parquet", index=False)

        # ---- summary
        ok = qc.status.str.startswith("ok")
        print("\n" + "=" * 72)
        print(f"Fields processed : {int(ok.sum())} / {len(qc)}  "
              f"(skipped {int((~ok).sum())})")
        print(f"Cells            : {len(nodes):,}")
        print(f"Edges (directed) : {len(edges):,}  ({len(edges) // 2:,} undirected)")
        print(f"label_red        : {nodes.label_red.sum():,}  ({100 * nodes.label_red.mean():.2f}%)")
        print(f"label_tagln_high : {nodes.label_tagln_high.sum():,}  "
              f"({100 * nodes.label_tagln_high.mean():.2f}%)")
        print(f"neighbor_count   : median {nodes.neighbor_count.median():.0f}, "
              f"isolated {int((nodes.neighbor_count == 0).sum()):,}")
        print(f"near_border      : {100 * nodes.near_border.mean():.1f}% of cells  |  "
              f"touches_border: {100 * nodes.touches_border.mean():.2f}%")
        if len(issues):
            print("\nIssues by check:")
            print(issues.groupby(["severity", "check"]).size().rename("fields_or_rows")
                  .to_string())
        else:
            print("\nNo issues found.")
        print("=" * 72)
        for f in ("exn_nodes.csv", "exn_edges.csv", "exn_fields_qc.csv", "exn_issues.csv",
                  "exn_schema.json", "exn_run_log.txt"):
            print(f"  wrote {out / f}")
        print(f"Done in {(time.time() - t0) / 60:.1f} min")
    finally:
        sys.stdout = tee.out
        tee.f.close()


if __name__ == "__main__":
    main()
