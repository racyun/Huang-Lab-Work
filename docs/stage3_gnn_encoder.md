# Stage 3 — GNN Encoder + Self-Supervised Contrastive Training

**Goal:** train a graph neural network that compresses each cell's spatial
**neighborhood** into a fixed-length **embedding vector**, such that structurally
similar neighborhoods land close together and dissimilar ones land far apart —
*without any motif labels*. These embeddings are the input to Stage 4 clustering,
where each cluster becomes a motif.

This covers Steps 3–6 of the project doc (neighborhood sampling → GNN encoder →
contrastive training → the encoder ready for clustering). It builds directly on
the Stage 2 graphs (`graphs/<condition>/<image_id>.pt`).

---

## 0. Where this fits

```
Stage 2 (done)                 Stage 3 (this doc)                    Stage 4
──────────────                 ──────────────────                    ───────
one graph per image     →   sample 2-hop neighborhood per cell  →   cluster
nodes = cells               GAT encoder -> neighborhood embedding     embeddings
edges = spatial adj.        contrastive (+ optional adversarial)      (Leiden)
                            train with no labels                      = motifs
                                    │
                            OUTPUT: trained encoder + one embedding per cell
```

**The one principle:** the encoder learns *neighborhood similarity*, not cell
identity. Two augmented views of the same neighborhood must embed close; two
different neighborhoods must embed apart. Everything below serves that.

---

## 1. Inputs (from Stage 2)

- `graphs/<condition>/<image_id>.pt` — PyG `Data` per image with `x` [N, 7]
  (z-scored morphology/intensity + raw `endmt_score` + `on_border`),
  `edge_index`, `edge_attr` [E, 4] (`dist, dist_norm, inv_dist, is_mutual`),
  `pos`, and the graph-level label `condition`.
- `graphs/feature_names.json`, `master_table.csv`, `normalization_stats.json`.

**Gap to note:** the graphs currently carry only `condition` (the 6 stiffness
folders) as a graph-level label. Scrubbing nicotine/ECM in the adversarial step
would need a `condition → (stiffness, nicotine, ecm)` lookup joined in (not yet
built). For the first pass, the adversary (if used) scrubs `condition`.

---

## 2. Neighborhood subgraph sampling (Step 3)

A motif is about a cell's **local context**, so the unit of analysis is a
**2-hop ego-subgraph**: a center cell + its neighbors + their neighbors
(~15–30 cells). One per cell → ~1.27M subgraphs across the dataset.

- **Hop count: 2** (the doc's sweet spot). 1-hop = too little context;
  3-hop+ = tissue-scale, blurs local distinctions.
- Extract with PyG `k_hop_subgraph(node_idx, num_hops=2, edge_index)` per cell,
  carrying `x` and `edge_attr` for the induced subgraph. Mark the center node.
- **Efficiency:** don't materialize 1.27M subgraphs to disk. Two viable routes:
  - **(A) On-the-fly ego sampling** (matches the doc literally): each training
    step samples a batch of center cells, extracts their 2-hop subgraphs live,
    augments, encodes, pools to a neighborhood embedding. Sample a subset of
    cells per epoch (e.g. 50k) rather than all 1.27M.
  - **(B) Full-graph node-level encoding** (more efficient): run the GAT over
    the whole image graph; after L layers each node's embedding already
    summarizes its L-hop neighborhood. Contrast node embeddings across two
    augmented views of the graph (GRACE-style). No explicit subgraph extraction.

**Decision: start with (A)** — it matches the doc's framing, makes the
"neighborhood embedding" explicit, and is easy to reason about. Keep (B) as a
scaling fallback if ego-sampling is too slow on the full dataset.

---

## 3. Encoder architecture (Step 4)

A **Graph Attention Network (GAT)** — neighbors are weighted by learned
attention, not averaged equally.

| component | choice | notes |
|---|---|---|
| layers | **3** GAT layers | doc-specified; captures up to 3-hop context |
| heads | 4 attention heads (concat in hidden layers, mean on last) | standard GAT |
| hidden dim | 128 | per-layer node dim |
| activation | ELU + dropout (0.2) | between layers |
| edge features | feed `edge_attr` into attention (GATv2 `edge_dim=4`) | uses `dist_norm`, `is_mutual` as edge priors |
| readout | attention pool (or mean pool) over subgraph nodes | → one neighborhood vector |
| projection | embedding dim **64**, + a 2-layer MLP projection head for the contrastive loss | 64 matches the doc's clustering step |

- **Use GATv2** (`torch_geometric.nn.GATv2Conv`) — strictly more expressive than
  GATv1 and accepts `edge_dim` so our edge attributes inform attention.
- **Readout** produces the neighborhood-level embedding assigned to the center
  cell. Attention pooling (`GlobalAttention`) weights informative cells more;
  mean pooling is the simpler baseline — start with mean, try attention later.
- **Two-vector convention (SimCLR):** the encoder produces an embedding `h`
  (64-d, kept for clustering); a small **projection head** `g(h)=z` is used
  *only* for the contrastive loss and discarded afterward. Clustering uses `h`,
  not `z`.

---

## 4. Augmentations (Step 5, the "two views")

For each neighborhood, generate two stochastic views (`A′`, `A″`) by composing:

- **Edge dropping** — randomly remove a fraction of edges (p ≈ 0.2).
- **Feature masking** — zero out a random subset of node-feature dimensions
  (p ≈ 0.1).
- **Node dropping** — drop a few *outer* (non-center) cells (p ≈ 0.1); never
  drop the center.
- **Feature noise** — small Gaussian noise on the (already z-scored) features.

These define what "the same neighborhood" means: invariances the encoder must
learn through. Keep them mild — too aggressive and you destroy the structure
you're trying to encode.

---

## 5. Contrastive objective (Step 5)

**NT-Xent (normalized temperature-scaled cross-entropy)**, SimCLR-style:

- For a batch of `B` neighborhoods, make `2B` views. Each view's positive is its
  sibling (same neighborhood); all other `2B−2` are negatives.
- Maximize cosine similarity of positives, minimize it for negatives, scaled by
  temperature `τ ≈ 0.2`.
- Larger batches give more negatives → better embeddings; use the largest batch
  the GPU allows (gradient accumulation if needed).

```
L_contrastive = NT-Xent({z_i}, temperature=τ)
```

---

## 6. Adversarial confound scrubbing (optional — measure first)

The risk (discussed at length): the encoder may organize the embedding space by
**experimental condition** (stiffness) rather than structure, so clusters just
re-discover the conditions. The fix is an adversary that tries to predict
`condition` from the embedding, with a **gradient-reversal layer (GRL)** that
trains the encoder to make that prediction *impossible*.

```
L_total = L_contrastive  −  λ · L_adversary        (GRL flips the adv. gradient)
```

- Adversary = small MLP on the pooled embedding `h`, predicting `condition`.
- `λ` **ramped from 0** over training so the encoder learns structure first,
  then is pressured to drop the confound.
- To also scrub nicotine/ECM later: one GRL+head per variable (needs the
  metadata lookup from §1).

**Decision: don't build the adversary first.** Train contrastive-only, then run
the **confound probe** (§9). If `condition` is *not* recoverable from the
embeddings → no leak, skip the adversary entirely. If it *is* easily recoverable
→ add the GRL head. This keeps the first build simple and only pays the
complexity cost if the data proves it's needed.

---

## 7. Training loop

- **Batching:** `torch_geometric.loader.DataLoader` over the augmented
  subgraph views; `Batch.from_data_list` disjoint-union batches them so no
  message passing leaks across neighborhoods.
- **Optimizer:** AdamW, lr ≈ 1e-3, cosine decay, weight decay 1e-4.
- **Epochs:** start ~50–100; watch the contrastive loss plateau.
- **Hardware:** **GPU** (this is real NN training — unlike Stages 1-feature/2).
  A T4 is fine.
- **Checkpoint** best/last encoder; log loss (and adversary accuracy if used) to
  W&B or a JSONL, mirroring the existing pipeline's logging.

---

## 8. Outputs

- `stage3/encoder.pt` — trained GAT encoder weights (the deliverable).
- `embeddings.parquet` (or `.npy` + index) — one **64-d embedding per cell**,
  keyed by (`condition`, `image_id`, `cell_id`). This is Stage 4's input.
- `stage3/train_log.jsonl` — loss curves (+ adversary metrics if used).
- `stage3/config.json` — hyperparameters used (reproducibility).

---

## 9. Validation / QC (before Stage 4)

1. **Augmentation invariance** — positive pairs should have high cosine
   similarity; sanity-check a few.
2. **Confound probe (the key check):** freeze the encoder, train a simple
   classifier to predict `condition` from the embeddings. Also color a UMAP of
   the embeddings by `condition`.
   - Can't predict / conditions interleaved → **no leak, skip the adversary.**
   - Easily predicted / cleanly separated by condition → **add the adversary**
     (§6) and retrain.
3. **Embedding sanity** — color the UMAP by `endmt_score`; an EndMT gradient
   should appear if the encoder captured biology.
4. **Don't over-scrub** — if the adversary is used, confirm motifs remain
   interpretable afterward (structure not erased along with the confound).

---

## 10. Locked decisions

1. **Hops:** 2-hop ego-subgraphs (doc sweet spot).
2. **Encoder:** 3-layer GATv2, 4 heads, hidden 128, embedding **64**, edge_dim=4.
3. **Readout:** mean pool first; attention pool as an upgrade.
4. **Objective:** NT-Xent contrastive, τ≈0.2, SimCLR projection head (discarded
   for clustering).
5. **Augmentations:** edge drop + feature mask + outer-node drop + feature noise
   (mild).
6. **Adversary:** measure-first — contrastive-only, add GRL only if the confound
   probe shows leakage.
7. **Embedding used downstream = `h`** (encoder output), not the projection `z`.

---

## 11. Implementation sketch (PyTorch Geometric)

```python
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class NeighborhoodEncoder(nn.Module):
    def __init__(self, in_dim=7, hid=128, emb=64, heads=4, edge_dim=4, p=0.2):
        super().__init__()
        self.g1 = GATv2Conv(in_dim, hid, heads=heads, edge_dim=edge_dim, dropout=p)
        self.g2 = GATv2Conv(hid*heads, hid, heads=heads, edge_dim=edge_dim, dropout=p)
        self.g3 = GATv2Conv(hid*heads, emb, heads=1, edge_dim=edge_dim, dropout=p)
        self.proj = nn.Sequential(nn.Linear(emb, emb), nn.ReLU(), nn.Linear(emb, emb))

    def encode(self, x, edge_index, edge_attr, batch):
        x = F.elu(self.g1(x, edge_index, edge_attr))
        x = F.elu(self.g2(x, edge_index, edge_attr))
        x = self.g3(x, edge_index, edge_attr)
        return global_mean_pool(x, batch)          # h: [B, emb] neighborhood embedding

    def forward(self, data):                        # returns (h for clustering, z for loss)
        h = self.encode(data.x, data.edge_index, data.edge_attr, data.batch)
        return h, self.proj(h)

def nt_xent(z1, z2, tau=0.2):
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.t() / tau
    n = z1.size(0)
    targets = torch.arange(n, device=z.device)
    targets = torch.cat([targets + n, targets])
    sim.fill_diagonal_(-1e9)
    return F.cross_entropy(sim, targets)
```

(GRL + adversary head added only if §9 shows leakage.)

---

## 12. How to run (planned scripts)

```bash
pip install torch torch_geometric pandas pyarrow umap-learn scikit-learn

# 1. train the encoder (contrastive; --adversarial only if the probe says so)
python train_encoder.py --epochs 100 --emb-dim 64 --hops 2

# 2. embed every cell with the frozen encoder -> embeddings.parquet
python embed_cells.py --encoder stage3/encoder.pt

# 3. confound probe + UMAPs (QC §9)
python probe_embeddings.py
```

Add `--push-to-drive` to back up `encoder.pt` + `embeddings.parquet` (Studios
are ephemeral), matching the Stage 2 scripts.

---

## 13. Build checklist

- [ ] `models/neighborhood_encoder.py` — GATv2 encoder + projection head (§3, §11).
- [ ] `augment.py` — edge drop / feature mask / node drop / noise (§4).
- [ ] `train_encoder.py` — ego-sampling loader + NT-Xent loop + checkpointing (§5, §7).
- [ ] `embed_cells.py` — frozen encoder → one 64-d embedding per cell → parquet (§8).
- [ ] `probe_embeddings.py` — confound probe + UMAP QC (§9).
- [ ] Decide on adversary from the probe result; add GRL head only if needed (§6).
- [ ] (later) `condition → (stiffness, nicotine, ecm)` lookup to scrub all three.
