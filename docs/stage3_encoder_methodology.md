# Stage 3 — GNN Encoder: Full Methodology & Design Rationale

This document is the deep-dive companion to
[stage3_gnn_encoder.md](stage3_gnn_encoder.md). For every architectural choice it
states **what** it does, **why** it's the right choice here, and **what
alternatives it beats**. The goal is a methodology that holds up in a lab
meeting or a methods section.

---

## 0. What the encoder must achieve (the design contract)

The encoder takes one **neighborhood subgraph** (a center cell + its 2-hop
neighbors, as tensors `x`, `edge_index`, `edge_attr`) and outputs **one
fixed-length vector** (the embedding) such that:

1. **Structurally similar neighborhoods → similar vectors.** Two "transition
   fronts" should land near each other even if they're in different images.
2. **Different neighborhoods → distant vectors.** A "mesenchymal cluster" should
   land far from an "endothelial sheet."
3. **Fixed length regardless of input size.** Subgraphs vary (15–30 cells); the
   output must be a constant-dimensional vector so we can cluster them.
4. **No labels required.** We have no ground-truth motifs, so learning must be
   self-supervised.
5. **Invariant to nuisances** (small perturbations; ideally experimental
   condition) so motifs reflect biology, not noise or batch.

Every choice below serves one of these five requirements.

---

## 1. Why a graph neural network at all

The data is relational: cells (nodes) + spatial adjacency (edges) + per-cell
features. The signal we want — *spatial arrangement of cell types* — lives in the
relationships, not in any single cell.

- **vs. an MLP on a flattened feature table:** an MLP has no notion of
  neighbors; it can't express "this cell is surrounded by mesenchymal cells."
  It would treat the neighborhood as a bag of numbers with no structure. ✗
- **vs. a CNN on image patches:** a CNN operates on a fixed pixel grid. Cells
  aren't on a grid — they're irregularly placed points with variable neighbor
  counts. A CNN also reintroduces pixel-level nuisances (staining, focus) we
  deliberately distilled away in Stages 1–2, and can't produce clean per-cell
  motif assignments. ✗
- **GNN:** operates natively on irregular graphs, respects the adjacency we
  built, handles variable-sized neighborhoods, and produces per-node/neighborhood
  representations. ✓ This is the natural inductive bias for spatial-omics data and
  the standard in the field (squidpy, CellCharter, etc.).

**The GNN's core operation — message passing — is exactly the computation we
want:** each cell updates itself using its neighbors, which *is* "characterize a
cell by its local context."

---

## 2. Why GAT (graph attention), specifically GATv2

Among GNN layer types, the choice is between treating neighbors **equally** vs.
**weighting** them. Options:

| Layer | Neighbor aggregation | Limitation here |
|---|---|---|
| **GCN** | fixed, degree-normalized average | every neighbor counts equally — can't downweight an irrelevant cell |
| **GraphSAGE** | mean/max/LSTM pool | better, but still no learned per-neighbor importance |
| **GIN** | sum + MLP (max discriminative power for graph isomorphism) | powerful for *distinguishing* graphs, but sum-aggregation is sensitive to size/degree and gives no interpretable neighbor weighting |
| **GAT / GATv2** | **learned attention weights per neighbor** | ✓ best fit — see below |

**Why attention is optimal here.** In a real neighborhood, not all neighbors are
equally informative. When characterizing a transitioning cell, an adjacent
mesenchymal cell may matter more than a distant endothelial one. **GAT learns how
much to weight each neighbor** — exactly the behavior the project doc describes
("the model decides neighbor B is more informative than neighbor C"). This gives:

- **Better representations** — the model focuses on the structurally meaningful
  neighbors.
- **Interpretability** — attention weights are inspectable ("which neighbors
  drove this cell's embedding?"), valuable for biologist trust.
- **A natural slot for edge features** — attention can be conditioned on
  `edge_attr` (distance, mutuality), so closeness modulates influence.

**Why GATv2 over the original GAT.** The original GAT computes a *static*
attention that, due to the order of its linear+nonlinearity, can collapse to a
ranking that ignores the query node ("static attention" — proven limited by
Brody et al. 2021). **GATv2 fixes this with dynamic attention** (swap the order so
the scoring MLP is applied after concatenation), giving strictly more expressive
attention at the same cost. GATv2 also cleanly accepts `edge_dim`, letting us
feed our 4-d edge attributes into the attention computation. **No downside vs.
GATv1, strict upside → GATv2.**

---

## 3. Layer-by-layer architecture and why each piece

```
input x [N,7]
   │  (optional input Linear 7→128 to lift features before attention)
   ▼
GATv2Conv layer 1   (heads=4, edge_dim=4)  → ELU → dropout
   ▼
GATv2Conv layer 2   (heads=4, edge_dim=4)  → ELU → dropout      [+ residual]
   ▼
GATv2Conv layer 3   (heads=1, → emb dim)   → (no activation)
   ▼
READOUT: pool all node vectors → 1 neighborhood vector  h [64]
   ▼
PROJECTION HEAD g(h)=z  (used only for the contrastive loss)
```

### 3a. Number of layers = 3 → and why not more

Each GAT layer passes information **one hop**. With 3 layers, the center cell
receives information from up to 3 hops away — matching (and slightly exceeding)
our 2-hop subgraph so every cell in the subgraph can inform the center.

- **Too few (1 layer):** the center only sees immediate neighbors — too little
  context to distinguish motifs (the doc's "1-hop is too small" point).
- **Too many (5+ layers):** GNNs suffer **oversmoothing** — after many rounds of
  averaging, all node vectors converge to the same value and become
  indistinguishable. 3 layers is the empirically robust sweet spot for
  neighborhood-scale tasks and aligns with the 2-hop subgraph radius. ✓

### 3b. Multi-head attention (4 heads) → why

Each "head" learns an independent attention pattern; their outputs are
concatenated. This is the standard transformer/GAT trick: multiple heads let the
model attend to **different relational aspects simultaneously** (e.g. one head
keys on distance, another on marker similarity) and **stabilizes training**
(averages out noisy single-head attention). 4 heads is a good
capacity/compute balance for graphs this small; the final layer uses 1 head (or
averages heads) to produce the embedding dimension cleanly.

### 3c. Hidden dimension = 128 → why

The node features start at 7 dims — too narrow for the model to compute rich
intermediate representations. Lifting to 128 hidden units gives the attention
layers room to encode combinations of features and neighborhood patterns, without
being so large that it overfits ~1.27M small graphs. 64–256 is the typical range;
128 is a balanced default to tune.

### 3d. Activation = ELU, plus dropout → why

- **ELU** (exponential linear unit) is the GAT paper's default — smooth,
  non-saturating for negatives, trains stably. ReLU is a fine alternative; ELU
  empirically edges it for attention nets.
- **Dropout (0.2)** on attention weights and features is **regularization** —
  randomly zeroing some attention/features prevents the model from over-relying
  on any single neighbor and improves generalization across the dataset.

### 3e. Residual connection (layer 2→3) → why

Adding the input of a layer to its output (`h = h + layer(h)`) gives gradients a
shortcut path, **mitigating oversmoothing and vanishing gradients** in deeper
GNNs. With only 3 layers it's a minor but free stabilizer; include it on the
middle layer.

### 3f. Feeding edge features into attention (`edge_dim=4`) → why

Our edges carry `[dist, dist_norm, inv_dist, is_mutual]`. Passing them into
GATv2's attention lets the model **modulate neighbor influence by geometry** — a
cell 12px away can be weighted differently from one 70px away, and a mutual k-NN
edge (both cells chose each other) can be treated as higher-confidence. This
injects spatial scale into the attention, which pure node-only message passing
would ignore. ✓

---

## 4. Readout (graph pooling) → mean first, attention as upgrade

After the conv layers, every cell has a vector; we need **one vector for the
whole neighborhood**. That's the readout.

| Readout | Behavior | Trade-off |
|---|---|---|
| **Mean pool** | average all node vectors | simple, stable, size-invariant; the robust default |
| **Sum pool** | add node vectors | encodes size/count, but scales with neighborhood size → sensitive to cell-count differences (a confound) |
| **Max pool** | per-dim max | captures "is there any cell like X," loses distribution |
| **Attention pool** | learned weighted average | can emphasize the center/important cells; more expressive, more params |

**Why mean first:** it's **size-invariant** (a 20-cell and 28-cell version of the
same motif pool to comparable vectors), stable, and parameter-free — the right
baseline. **Why attention pool as an upgrade:** since the *center* cell defines
the neighborhood, a learned pool (or concatenating the center's own final vector
with the mean) can sharpen the embedding. Start mean, try center-aware/attention
pooling as an ablation. Avoid sum here because neighborhood size correlates with
local density, which can leak as a confound.

---

## 5. The projection head (SimCLR design) → why a throwaway MLP

The encoder outputs `h` (64-d). For the contrastive loss we pass `h` through a
small 2-layer MLP to get `z`, compute the loss on `z`, and **then discard the
projection head** — clustering uses `h`, not `z`.

**Why this indirection (from SimCLR):** the contrastive loss forces `z` to throw
away information that distinguishes augmentations (it must make two views
identical). If we applied the loss directly to `h`, we'd damage `h` by stripping
useful detail. The projection head **absorbs that information loss**, letting `h`
stay rich while `z` does the contrastive work. Empirically this yields markedly
better downstream representations. ✓

---

## 6. Embedding dimension = 64 → why

64 dims is enough to encode the variety of neighborhood structures while staying
low enough to cluster well (clustering degrades in very high dimensions — the
"curse of dimensionality"). It matches the doc's Stage 4 plan (UMAP from 64→~10
before Leiden). 32 may underfit; 128 adds little and slows clustering. 64 is the
field-standard default to start, tunable later.

---

## 7. The learning objective: self-supervised contrastive (NT-Xent)

### 7a. Why self-supervised / contrastive at all

We have **no motif labels**, so supervised learning is impossible. The options
for label-free representation learning:

| Approach | Idea | Why not / why |
|---|---|---|
| **Graph autoencoder** | reconstruct the graph from the embedding | tends to prioritize reconstructing low-level details (exact features/edges) over capturing semantic structure; embeddings less cluster-friendly |
| **DGI (Deep Graph Infomax)** | maximize mutual info between node & graph summary | strong, but coarser; less control over what "similar" means |
| **BGRL / bootstrapped** | no negatives, momentum target | great at scale, no negatives needed, but more moving parts; a fine alternative if negatives become a bottleneck |
| **Contrastive (SimCLR/GRACE) + NT-Xent** | same neighborhood (augmented) = similar; others = different | ✓ directly encodes our exact requirement; well-understood; the project doc specifies it |

**Contrastive learning directly operationalizes requirement #1–2:** "different
versions of the same neighborhood should be close; unrelated ones far." That's
precisely the rule the doc describes, so it's the natural choice.

### 7b. NT-Xent loss — what it does

For a batch of `B` neighborhoods we make `2B` augmented views. For each view, its
**positive** is its sibling (same source neighborhood); the other `2B−2` are
**negatives**. NT-Xent maximizes cosine similarity to the positive and minimizes
it to negatives, scaled by **temperature τ** (≈0.2):

- **Temperature** controls how sharply the model separates positives from
  negatives. Lower τ → harder penalties on close negatives (sharper clusters),
  but too low destabilizes. ~0.1–0.3 is standard.
- **Cosine similarity** (not Euclidean) makes the loss scale-invariant — only the
  *direction* of embeddings matters, which clusters more robustly.

### 7c. Why batch size matters

Negatives come from *the rest of the batch*. More negatives = a harder, more
informative task = better embeddings. So use the **largest batch the GPU allows**
(gradient accumulation or a memory bank if needed). This is the main lever on
contrastive quality.

---

## 8. Augmentations — defining "the same neighborhood"

Contrastive learning is only as good as its augmentations: they *define the
invariances* the encoder learns. We make two views by composing mild random
perturbations. Each has a biological/robustness rationale:

| Augmentation | What | Why (invariance it teaches) | Risk if too strong |
|---|---|---|---|
| **Edge dropping** (p≈0.2) | randomly remove edges | robustness to the *exact* k-NN wiring (which is itself an approximation) → motif shouldn't hinge on one edge | graph fragments; loses structure |
| **Feature masking** (p≈0.1) | zero random feature dims | robustness to a missing/uncertain measurement (e.g. a noisy channel) | erases the signal being learned |
| **Node dropping** (p≈0.1, outer only) | drop a few non-center cells | robustness to segmentation hits/misses at the edge of the FOV | drop too many → different neighborhood |
| **Feature noise** | small Gaussian on (z-scored) features | robustness to measurement noise | swamps real differences |

**Why mild:** augmentations must preserve the neighborhood's *identity* while
varying its *incidentals*. Too aggressive and A′/A″ are no longer "the same
neighborhood," and the loss teaches nonsense. **Never drop the center cell** — it
defines the neighborhood. These four are the standard graph-contrastive set
(GraphCL/GRACE) and map cleanly onto real sources of variation in this data.

---

## 9. Adversarial confound scrubbing — measure first, then maybe add

**The problem:** stiffness (and nicotine/ECM) genuinely changes cell appearance,
so the encoder might organize embeddings by *condition* rather than *structure* —
and clustering would just rediscover the experimental groups. This is the central
risk to the whole project's validity.

**The mechanism (if used):** an adversary MLP tries to predict `condition` from
the embedding; a **gradient-reversal layer (GRL)** sits between encoder and
adversary. Forward pass: identity. Backward pass: the GRL **multiplies the
gradient by −λ**, so the adversary's "make condition predictable" signal becomes
"make condition *un*predictable" for the encoder. Net objective:

```
L = L_contrastive  −  λ · L_adversary
```

At equilibrium the adversary can't beat chance → the embedding carries no
condition information → motifs are structure-defined. `λ` is **ramped from 0** so
the encoder learns structure first, then is pressured to drop the confound.

**Why "measure first" is optimal, not "always include it":**
- The adversary is the **most finicky** component (λ tuning, head balancing,
  risk of over-scrubbing real biology).
- Whether it's *needed* is an **empirical question** answerable cheaply: train
  contrastive-only, then probe whether `condition` is recoverable from `h`.
  - Not recoverable → no leak → adversary adds complexity for nothing. **Skip.**
  - Easily recoverable → leak confirmed → **add it.**
- This sequences risk correctly: get a working baseline, prove the problem
  exists, then solve it. (See QC §11.)

**To scrub nicotine/ECM too** requires the `condition → (stiffness, nicotine,
ecm)` lookup (not yet built); each variable gets its own GRL+head, or one
multi-task head.

---

## 10. Training procedure & why each setting

| Setting | Choice | Why |
|---|---|---|
| **Subgraph source** | on-the-fly 2-hop ego from image graphs | fresh augmentations each epoch; no 1.27M-file storage; subgraphs are cheap views |
| **Batching** | PyG `DataLoader` (disjoint-union) | batches many subgraphs without edges leaking across them |
| **Optimizer** | AdamW | decoupled weight decay → better generalization than Adam for this |
| **LR** | ~1e-3, cosine decay | standard for contrastive GNN; cosine anneals smoothly to a good minimum |
| **Weight decay** | 1e-4 | mild L2 regularization |
| **Warmup** | a few epochs | stabilizes early attention training |
| **λ ramp** (if adversary) | 0 → target over epochs | learn structure before scrubbing |
| **Epochs** | 50–100, watch loss plateau | contrastive needs enough steps to separate the space |
| **Hardware** | GPU (T4 fine) | real NN training — unlike Stages 1-feature/2 which were CPU |
| **Checkpoint** | best + last | resumability + pick best by a proxy (loss / probe) |

---

## 11. Validation / QC — how we know it worked (before Stage 4)

1. **Augmentation invariance:** positive pairs should have high cosine similarity.
   Sanity-check a sample.
2. **Confound probe (the decisive test):** freeze `h`, train a simple classifier
   to predict `condition`. Color a UMAP of `h` by condition.
   - Can't predict / interleaved → no leak (skip adversary).
   - Easily predicted / separated → leak (add adversary, retrain).
3. **Biology sanity:** color the UMAP by `endmt_score` — an EndMT gradient should
   appear if the encoder captured real signal.
4. **Cluster preview:** a quick k-means/Leiden on `h` should yield a handful of
   coherent groups (not one blob, not 500 fragments) — a green light for Stage 4.
5. **Don't over-scrub:** if the adversary is on, confirm motifs stay
   interpretable (structure not erased with the confound).

---

## 12. Hyperparameter reference (defaults + rationale + sweep)

| Param | Default | Rationale | Sweep later |
|---|---|---|---|
| hops (subgraph radius) | 2 | local context, not tissue-scale | {1, 2, 3} |
| GNN layers | 3 | ~match 2-hop; avoid oversmoothing | {2, 3, 4} |
| heads | 4 | multi-aspect attention, stable | {2, 4, 8} |
| hidden dim | 128 | capacity vs overfit | {64, 128, 256} |
| embedding dim | 64 | cluster-friendly, matches Stage 4 | {32, 64, 128} |
| dropout | 0.2 | regularization | {0.1–0.3} |
| τ (temperature) | 0.2 | separation sharpness | {0.1, 0.2, 0.5} |
| batch size | as large as fits | more negatives = better | maximize |
| lr | 1e-3 | standard, cosine decay | {3e-4, 1e-3} |
| λ (adversary) | ramp 0→~1 | scrub strength (if used) | tune via probe |
| augment ps | edge .2 / feat .1 / node .1 | mild, identity-preserving | small grid |

---

## 13. Failure modes & mitigations

- **Representation collapse** (all embeddings identical): caused by too-strong
  augmentations or no/weak negatives. Mitigate: milder augs, bigger batch,
  check τ. (BGRL is an alternative if negatives are the issue.)
- **Oversmoothing** (deep GNN blurs nodes): keep 3 layers, residuals.
- **Confound leakage** (clusters = conditions): the probe catches it; add
  adversary.
- **Over-scrubbing** (adversary erases biology): cap λ, verify interpretability.
- **Size/density confound** (motifs track cell count): use mean (not sum)
  readout; tabulate degree/density by condition (Stage 2 QC) and watch for it.
- **Thin features** (motifs not rich): only 7 node features now — enrich later
  (more morphology, better EndMT panel) for finer motifs.

---

## 14. One-paragraph summary

A 3-layer **GATv2** encoder consumes each 2-hop neighborhood subgraph, using
**edge-feature-conditioned attention** to weight neighbors by relevance and
geometry, then a **mean readout** to produce a 64-d neighborhood embedding.
It's trained **self-supervised with NT-Xent contrastive loss** over two mildly
augmented views per neighborhood (edge/feature/node perturbations), via a
discarded **SimCLR projection head**, on GPU with AdamW + cosine LR. An
**adversarial gradient-reversal head is held in reserve** — added only if a
post-hoc probe shows experimental condition leaking into the embeddings. The
output — one 64-d embedding per cell — feeds Stage 4 clustering, where each
cluster becomes a motif. Every choice (GATv2, attention, 3 layers, mean pool,
projection head, contrastive objective, measure-first adversary) is selected to
satisfy the five design requirements: fixed-length, size-invariant, label-free,
similarity-preserving, and nuisance-invariant embeddings.
