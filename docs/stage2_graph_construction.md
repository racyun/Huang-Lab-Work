# Stage 2 — Spatial Graph Construction

**Goal:** convert each per-image table of cells (the output of Stage 1) into a
**spatial graph** the GNN can operate on. One graph per image. Nodes are cells
carrying their Stage 1 feature vectors; edges encode **physical adjacency**
(who is spatially next to whom), derived from centroid coordinates only.

This document merges the graph-construction plan with the gap analysis against
the current Stage 1 output, and spells out exactly what to change before coding.

---

## 0. The one principle to keep straight

**Features ride on nodes; geometry defines edges. They never mix during
construction.**

| What | Built from | Meaning |
|---|---|---|
| **Node features** (`x`) | intensities, morphology (+ EndMT score later) | *what* each cell is |
| **Edges** (`edge_index`) | centroid coordinates only | *who* is a spatial neighbor of whom |
| **Edge features** (`edge_attr`) | centroid distances | *how far apart* connected cells are |

k-NN is run **on centroids**, so "nearest" means *physically closest in the
image* — never feature-similar. Raw `(x, y)` must **not** go into the node
feature vector (the GNN would learn absolute position instead of relative
structure). Spatial information lives in the edges only.

---

## 1. Inputs (from Stage 1) — and the gaps to close first

The graph builder consumes a single long table, one row per cell:

| column | type | notes |
|---|---|---|
| `image_id` | str | which image/well the cell came from — **NEW, must add** |
| `condition` | str | folder name, e.g. `260514_900kPa` — **NEW, must add** |
| `cell_id` | int | per-image mask label (resets each image) |
| `centroid_x`, `centroid_y` | float | centroid in pixels |
| `feat_0 … feat_n` | float | the per-cell features |

### Gaps between this and the current Stage 1 CSVs

The current per-image CSV (`cellwise_metadata/<condition>/<name>_metadata.csv`)
has columns:

```
cell_id, area_px, elongation,
ch1_cellwise_mean_intensity,            # DAPI (nuclei, reference)
ch2_cellwise_mean_membrane_intensity,  # VE-cadherin (endothelial, membrane band)
ch3_cellwise_mean_intensity,           # TAGLN (mesenchymal, whole-cell)
endmt_score,                           # relative-ratio EndMT index, [0, 1]
centroid_x, centroid_y
```

**Channel map (from Carver):** CH1 = DAPI, CH2 = GFP/VE-cadherin, CH3 =
AF594/TAGLN. The EndMT score (relative-ratio form, bounded [0, 1], per the lab
master doc) uses E = VE-cadherin membrane signal (CH2) and M = TAGLN whole-cell
signal (CH3), each **background-subtracted** via the median of that channel over
non-cell (`mask == 0`) pixels and clamped at 0, then `endmt_score = M / (E + M)`:
~0 = endothelial, ~1 = mesenchymal. DAPI is used only to identify cells, not in
the score. Cells with no membrane band or no signal in either marker → `NaN`.

| Gap | Resolution |
|---|---|
| **No `image_id` / `condition`** — image identity lives only in the filename/folder | Inject both when concatenating: `image_id` from the filename stem, `condition` from the parent folder name. Without them you can't keep graphs separate or tag them downstream. |
| **`cell_id` is not globally unique** — it resets to 1,2,3… per image | Fine *within* a graph. Globally, disambiguate with the pair (`image_id`, `cell_id`). No need to renumber. |
| **Well-level metadata (stiffness / nicotine / ECM)** | **Deliberately NOT node features.** They are constant within an image (no within-graph signal) and would defeat the Stage 3 adversarial scrubbing. Carry **only** `condition` as a **graph-level label** on the `Data` object for downstream stratification / adversarial training — never as a column of `x`. |
| **Feature vector is thin** (6 features incl. EndMT score) | Not blocking. Motifs will only be as rich as the features. `endmt_score` is already a calibrated [0, 1] quantity (background-subtracted relative ratio); keep it **raw** — do not z-score it with the other features (that would destroy the "0 = endothelial, 1 = mesenchymal" meaning). |
| **Coordinate column names** | Already `centroid_x` / `centroid_y` (the plan calls them `x` / `y`). Trivial rename in the builder. |

### Prerequisite step: assemble the master table

Between Stage 1 and Stage 2, add a small step that:

1. Concatenates every `cellwise_metadata/<condition>/*.csv` into one table.
2. Adds `image_id` (from filename) and `condition` (from folder).
3. **Z-scores each feature column globally** (mean/std across *all* cells in
   *all* conditions, computed once). This is the cross-image normalization the
   plan implies — without it, large-magnitude columns (`area_px`) drown out the
   rest in both the GNN and the contrastive loss.
4. **Leaves `endmt_score` as-is.** It is computed per image in Stage 1 with
   per-image background subtraction and is already a calibrated [0, 1] ratio
   (self-normalizing for multiplicative per-cell effects). Do **not** z-score it
   in step 3 — exclude it from the global standardization so its [0, 1]
   semantics survive. (If cross-batch drift in the score is later observed in
   QC, apply a per-batch robust rescale of the score only, downstream.)

> Note: well-level metadata (stiffness, nicotine, ECM) is dropped from the
> cellwise features entirely — it was never in the per-cell vector and should
> stay out. Only `condition` is retained, and only as a graph-level tag.

---

## 2. Method

**Primary: k-NN on centroids, symmetrized (union), with a max-distance prune.**

For each image independently:

1. Build a KD-tree on `(centroid_x, centroid_y)`.
2. For each cell, find its `k` nearest cells.
3. **Symmetrize by union:** keep edge (A, B) if B is in A's top-k *or* A is in
   B's top-k. (Mutual-kNN — *and* instead of *or* — is the stricter, sparser
   alternative. **Union is the safer default** for GNN message passing because
   it keeps the graph connected and undirected.)
4. **Prune** any edge longer than `d_max`. Stops a cell in a sparse region from
   being "neighbors" with something biologically far away just because it's the
   k-th closest.
5. **Drop self-loops.**

### Alternatives (keep in back pocket; swap in if kNN motifs look density-driven)

- **Delaunay triangulation + distance prune** — parameter-free tessellation,
  closer to "who's physically touching whom" (squidpy's default). Needs
  aggressive pruning of long boundary edges.
- **Radius graph (r-ball)** — connect everything within distance `r`. Node
  degree then varies with local density, which can be a useful signal or a
  confound.

**Decision: build kNN-union first** (the debuggable baseline). Delaunay is ~10
lines behind a flag — add it as an ablation later, don't let it delay Stage 2.

---

## 3. Parameters & recommended defaults

| param | default | rationale |
|---|---|---|
| `k` | **8** | Mid-range of the 6–10 in the methodology doc. Sweep {6, 8, 10} *later* and check motif stability — don't sweep prematurely. |
| `d_max` | **~3× median nearest-neighbor distance, computed per-image** | Adaptive to each image's cell density. Compute median NN distance first, then set the cutoff. |
| symmetrization | **union** | Keeps the graph connected; GNNs expect undirected. |
| self-loops | **off** | PyG's GAT adds them internally if needed. |
| coords as node features? | **No** | Spatial info lives in edges only (see §0). |

---

## 4. Edge features

Store on each edge:

- `dist` — Euclidean distance between centroids (pixels; µm if calibrated).
- `dist_norm` — `dist / d_max` ∈ (0, 1], so it is scale-free across images.
- *(optional)* `inv_dist` — `1 / (dist + ε)`, for use as an attention prior /
  edge weight later.
- *(optional)* `is_mutual` — 1 if the edge was in **both** cells' top-k, 0 if
  only one direction. A soft confidence signal.

Skip relative angle/orientation unless there's reason to think directionality
matters in this tissue.

> **Implementation caveat:** the union-symmetrization trick (`flip + unique`)
> does **not** compute `is_mutual` for free — it discards which direction(s) an
> edge came from. If you want `is_mutual`, compute it *before* the union by
> recording which edges appeared in both directions.

---

## 5. Implementation sketch (PyTorch Geometric)

One `Data` object per image:

```python
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from scipy.spatial import cKDTree

def build_graph(df_image, feat_cols, k=8, d_max_mult=3.0):
    pos = torch.tensor(df_image[["centroid_x", "centroid_y"]].values, dtype=torch.float)
    x   = torch.tensor(df_image[feat_cols].values, dtype=torch.float)

    # adaptive d_max from this image's median nearest-neighbor distance
    tree = cKDTree(pos.numpy())
    nn_dist, _ = tree.query(pos.numpy(), k=2)        # k=2 -> self + 1st neighbor
    median_nn = float(np.median(nn_dist[:, 1]))
    d_max = d_max_mult * median_nn

    # kNN, then symmetrize via union (loop=False -> no self edges)
    edge_index = knn_graph(pos, k=k, loop=False, flow="target_to_source")
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)

    # edge attrs + distance prune
    src, dst = edge_index
    dist = (pos[src] - pos[dst]).norm(dim=1)
    keep = dist <= d_max
    edge_index, dist = edge_index[:, keep], dist[keep]
    edge_attr = torch.stack([dist, dist / d_max], dim=1)

    return Data(
        x=x, pos=pos,
        edge_index=edge_index, edge_attr=edge_attr,
        image_id=df_image["image_id"].iloc[0],
        condition=df_image["condition"].iloc[0],   # graph-level label, NOT a feature
    )
```

Save each `Data` to `graphs/{condition}/{image_id}.pt`. Downstream, load with
`torch_geometric.data.Batch.from_data_list(...)` — PyG handles the disjoint-union
batching so no edges leak across images, and graph-level fields (`condition`)
stay one-per-graph.

> If the rest of the stack is scanpy/AnnData, `squidpy.gr.spatial_neighbors`
> does kNN/Delaunay/radius in one call and stores the graph in `adata.obsp`.

---

## 6. QC — run before moving to Stage 3

**Per image:**

- **Degree distribution.** After symmetrize+prune, expect a tight-ish
  distribution around `k`–`2k`. Long right tail → density artifacts; lots of
  degree-0/1 → `d_max` too aggressive or segmentation gaps.
- **Edge-length histogram.** Should fall off well before `d_max`. A spike at
  `d_max` means the cutoff is doing real work — inspect those cases.
- **Connected components.** Ideally one big component per image. A few small
  islands are fine (genuinely isolated clusters); many singletons are not.
- **Overlay plot.** For ~5 images per condition, plot the graph on top of the
  segmentation mask. Eyeball: do the edges look like "neighbors" to a human?

**Across the dataset:**

- **Cells/image, edges/image, mean degree — tabulated by condition.** Large
  systematic differences here will leak into motifs later (a confound), so it's
  important to know now.

---

## 7. Edge cases

- **Tiny images (< k+1 cells):** skip, or fall back to a fully-connected graph.
  Log and review.
- **Border cells:** their neighborhood is physically truncated by the field of
  view. Add a boolean node feature `on_border` (centroid within `d_max` of a
  true image edge) so the model can learn to discount them — and so you can
  exclude them from motif-frequency stats later. **Flag-and-keep, don't drop.**
  All images are 682×682, so `build_graphs.py --image-size 682` (the default)
  uses the real field-of-view edges; pass `--image-size 0` to fall back to the
  centroid bounding box if dimensions ever vary.
- **Segmentation merges/splits:** not fixable here, but over/under-segmentation
  directly corrupts the graph. If Stage 1 segmentation is shaky, that's the
  higher-leverage thing to fix.

---

## 8. Output spec

- `graphs/{condition}/{image_id}.pt` — one PyG `Data` per image with `x`, `pos`,
  `edge_index`, `edge_attr`, `image_id`, `condition`.
- `graph_qc.csv` — one row per image: `n_nodes`, `n_edges`, mean/median degree,
  `n_components`, median edge length, `d_max` used.
- `graph_overlays/` — a handful of PNGs for the eyeball check.

---

## 9. Locked decisions

1. **Pixel vs µm:** use **pixels** if every image is the same magnification
   (all tiles are 10× focus stacks → likely consistent). Only convert to µm if
   magnification varies across wells.
2. **`k`:** **fixed at 8** now; sweep {6, 8, 10} after motifs exist.
3. **kNN-union (default) vs Delaunay+prune:** build **kNN first**; Delaunay
   behind a flag as a later ablation.
4. **Border cells:** **flag-and-keep** (`on_border` node feature).
5. **Well-level metadata (stiffness / nicotine / ECM):** **not node features.**
   Only `condition` is carried, as a **graph-level label** for Stage 3
   adversarial training and motif stratification.

---

## 10. Change checklist (to close the gaps before coding the builder)

- [x] Add an **assemble-master-table** step: concatenate all Stage 1 CSVs, add
      `image_id` (from filename) + `condition` (from folder). → `assemble_master_table.py`
- [x] **Z-score feature columns globally** across the whole dataset
      (exclude `endmt_score` — keep it raw [0, 1]). → `assemble_master_table.py`
- [x] Keep `condition` as a **graph-level tag**; exclude all well-level
      metadata (stiffness / ECM / nicotine) from the node feature vector. → `build_graphs.py`
- [x] (Stage 1) **EndMT score added** to the per-cell extractor (`endmt_score`).
- [x] Write `build_graphs.py` implementing §2–§5, emitting the §8 outputs.
- [x] Add `on_border` node feature (§7). → `build_graphs.py`
- [ ] Run the §6 QC suite and review overlays before starting Stage 3.

## 11. How to run (Stage 2)

```bash
pip install torch torch_geometric scipy scikit-image pandas matplotlib

# 1. assemble + globally normalize the master cell table
python motifs/stage2_graphs/assemble_master_table.py   # reads <root>/cellwise_metadata/

# 2. build one spatial graph per image (+ QC csv + overlays)
python motifs/stage2_graphs/build_graphs.py --k 8 --d-max-mult 3.0

# Add --push-to-drive to either script to rclone the outputs up to
# gdrive:<DRIVE_ROOT> when done (Lightning Studios are ephemeral — back them up):
python motifs/stage2_graphs/assemble_master_table.py --push-to-drive
python motifs/stage2_graphs/build_graphs.py --push-to-drive
```

Graphs are the **direct input to Stage 3** (the GNN reads them via
`Batch.from_data_list`). Load a saved graph with
`torch.load(path, weights_only=False)` (required on torch >= 2.6).

`<root>` follows the Stage 1 convention (`CELLPOSE_LOCAL_ROOT`, else the
Lightning Studio path, else `~/cellpose_work`). Outputs land in
`<root>/master_table.csv`, `<root>/graphs/<condition>/<image_id>.pt`,
`<root>/graph_qc.csv`, and `<root>/graph_overlays/`.

Node-feature order (`graphs/feature_names.json`): the 5 globally z-scored
morphology/intensity columns, then raw `endmt_score`, then `on_border`.
Edge attributes: `[dist, dist_norm, inv_dist, is_mutual]`.
