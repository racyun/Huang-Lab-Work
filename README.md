# Huang Lab — AI for Cardiovascular Tissue-on-a-Chip Microscopy

Machine-learning tools for fluorescence microscopy of endothelial cells and
cardiomyocytes cultured on substrates of different stiffness, developed in
Prof. Ngan Huang's lab. The repository holds two research tracks that share a
codebase:

| Track | Question | Status |
|---|---|---|
| **B — EndMT spatial-motif discovery** (`motifs/`) | Does substrate stiffness change *how endothelial-to-mesenchymal transition (EndMT) is organised in space* — are there recurring neighbourhood patterns invisible to the eye? | **Active.** Built end-to-end (segmentation → graphs → GNN encoder → motif clustering, validation, figures). |
| **A — Cardiomyocyte detection** (`config/`, `data/`, `models/`, `training/`, `scripts/`) | Can a stiffness-conditioned masked autoencoder (MAE) pretrained on unlabelled z-stacks improve Deformable-DETR cell detection? | **Parked.** Infrastructure complete and tested; MAE→DETR backbone wiring and full-data training not done. |

Track B is the current focus and is described below. Track A is documented in
full in [docs/mae_detr_pipeline.md](docs/mae_detr_pipeline.md).

---

## Contents

1. [Track B — EndMT spatial-motif pipeline](#1-track-b--endmt-spatial-motif-pipeline)
   - [Aims and task definition](#aims-and-task-definition)
   - [How the pipeline is used](#how-the-pipeline-is-used)
   - [Why this design](#why-this-design)
   - [What is new](#what-is-new)
   - [Pipeline overview](#pipeline-overview)
   - [Data](#data)
   - [Stage 1 — Segmentation and per-cell features](#stage-1--segmentation-and-per-cell-features)
   - [Stage 2 — Master table and spatial graphs](#stage-2--master-table-and-spatial-graphs)
   - [Stage 3 — Self-supervised neighbourhood encoder](#stage-3--self-supervised-neighbourhood-encoder)
   - [Stage 4 — Motif clustering, validation and figures](#stage-4--motif-clustering-validation-and-figures)
   - [What labels are (and are not) needed](#what-labels-are-and-are-not-needed)
   - [Running the pipeline](#running-the-pipeline)
   - [Results so far](#results-so-far)
   - [Open questions](#open-questions)
2. [Track A — MAE + Deformable-DETR detection](#2-track-a--mae--deformable-detr-detection)
3. [Repository layout](#3-repository-layout)
4. [Setup](#4-setup)
5. [Documentation](#5-documentation)
6. [Acknowledgements](#6-acknowledgements)

---

## 1. Track B — EndMT spatial-motif pipeline

### Aims and task definition

**Big-picture goal.** We have thousands of microscope images of endothelial
cells on substrates of different stiffness, and we want to find **recurring
spatial patterns in how cells are arranged near each other** — and how those
patterns shift as stiffness changes. A computer cannot simply look at the images
and "find patterns"; we have to tell it what to look at. We do that by turning
each image into a **graph** whose nodes are cells and whose edges connect
spatial neighbours, training a neural network to recognise which local
neighbourhoods look alike, and grouping similar neighbourhoods together. Each
group is one **motif**.

**Why neighbourhoods, not cells.** EndMT is usually scored one cell at a time
(how mesenchymal is *this* cell?). That discards the spatial context — whether a
transitioning cell sits alone inside an endothelial sheet, clusters with others,
or lines a front between endothelial and mesenchymal regions. Motifs capture
exactly that context. The kinds of motifs we expect the method to surface, and
that a biologist could name, look like:

| Expected motif | Neighbourhood composition |
|---|---|
| Endothelial sheet | all endothelial, no transition |
| Transition focus | one transitioning cell surrounded by endothelial cells |
| Isolated transition | a lone transitioning cell with few neighbours |
| Mixed interface | endothelial, transitioning and mesenchymal cells side by side |
| Transition front | a line of transitioning cells between the two states |
| Mesenchymal cluster | all mesenchymal |

(These are the hypothesised catalogue from the project plan; the motifs the
pipeline actually discovers are named *after* clustering by looking at examples
— see [Results so far](#results-so-far).)

**What a result looks like.** Two headline outputs:

- **Motif maps** — the original tile with every cell coloured by the motif its
  neighbourhood was assigned to. One dot = one segmented cell at its real
  position; the colour = the motif label for that cell's *surroundings*.
  Neighbouring cells usually share a motif because they share most of the same
  neighbourhood, so the maps show contiguous coloured domains rather than
  salt-and-pepper noise.
- **Motif frequency across stiffness** — stacked bars of the proportion of each
  condition's cells in each motif. This is the scientific claim: if, say,
  mesenchymal-cluster motifs grow and endothelial-sheet motifs shrink as the
  substrate stiffens, that is a spatial signature of stiffness-driven EndMT.

### How the pipeline is used

1. **Training phase** — images from *all* stiffness conditions are pooled into
   one training set and the GNN encoder is trained with a self-supervised
   contrastive objective (no labels). The result is an encoder that maps any
   neighbourhood to an embedding vector.
2. **Motif-discovery phase** — the trained encoder is run over every
   neighbourhood from every condition, and the combined embedding space is
   clustered. Those clusters are the motifs; each motif is defined by
   neighbourhoods drawn from across all stiffnesses.
3. **Inference phase** — given a new image at any stiffness, run it through the
   encoder and assign each cell to its nearest motif centroid.

The point of pooling and of the de-confounding machinery in Stage 3 is that a
neighbourhood that structurally looks like "transition focus" lands near the
transition-focus cluster *regardless of which condition it came from*. One
model, one motif vocabulary, applied uniformly to every image — so motif
frequencies are comparable across conditions.

### Why this design

- **Why a GNN rather than a CNN on image crops?** A CNN on a patch around each
  cell learns from raw pixels but has no notion of which cells are neighbours or
  what their properties are, and it re-introduces staining and focus nuisances
  we deliberately distilled away in Stages 1–2. The GNN operates directly on
  the graph of cells with known features, so it learns from spatial
  relationships between cells with known properties — far more biologically
  meaningful and interpretable.
- **Why attention (GAT)?** When updating a cell's representation the model
  learns how much weight to give each neighbour, and those attention weights
  can be inspected afterwards to see which neighbours mattered — useful for
  characterising motifs.
- **Why self-supervised contrastive rather than supervised?** We cannot label
  motifs because we do not know what they are — discovering them is the task.
  Contrastive learning builds a meaningful embedding space from the structure of
  the data alone.
- **Why cluster after training rather than jointly?** Joint representation
  learning + clustering exists but is harder to train and less stable.
  Decoupling them — learn a good embedding space first, then cluster — is more
  robust and much easier to validate.

### What is new

Spatial-neighbourhood analysis exists in spatial omics, but the existing tools
do not fit this problem:

1. They were built for **multiplexed** spatial omics (CODEX, IMC, spatial
   transcriptomics) with 20–40 markers per cell. Here the same ideas are applied
   to **conventional fluorescence imaging with 3 markers** — a genuinely
   different input format.
2. They are built around static tissue snapshots for cancer-vs-healthy
   comparison, not for studying a **transition process** such as EndMT.
3. None of them handle **substrate stiffness** as an experimental axis; they are
   designed for clinical / anatomical comparisons.

### Pipeline overview

```
Stage 1                 Stage 2                  Stage 3                    Stage 4
───────                 ───────                  ───────                    ───────
tiles ─► Cellpose-SAM   per-image CSVs ─►        graphs ─► 2-hop ego        embeddings ─► UMAP + Leiden
      ─► masks               master table          subgraphs ─► GATv2         ─► motif per cell
      ─► per-cell            (global z-score)      encoder (NT-Xent,          ─► validation / robustness
         features       ─► kNN graph per image     optional adversary)        ─► motif maps, catalog,
         + EndMT score     (nodes=cells,          ─► 64-d embedding              frequency-by-condition
                            edges=adjacency)         per cell
```

In plain language, the six steps the code implements:

1. **Per-cell feature vectors** — for every cell in every image, record its
   brightness in each channel, size, elongation, and a 0-to-1 score for how
   "endothelial vs. mesenchymal" it looks. Each cell becomes one row of numbers
   so cells can be compared quantitatively.
2. **Graph construction** — in each image, draw a line from every cell to its
   *k* = 8 closest cells (by centre-to-centre distance) and store the distance
   on each line. Keeping *k* small (6–10) is deliberate: connect cells that are
   too far apart and motifs end up reflecting whole-image composition rather
   than the immediate microenvironment.
3. **Neighbourhood subgraphs** — for every cell, cut out the cell, its
   neighbours, and its neighbours' neighbours (2 hops, roughly 15–30 cells).
   This is the unit that gets a motif. 1 hop is too little context; 3+ hops
   blurs into tissue scale.
4. **GNN encoder** — three graph-attention layers compress a subgraph of any
   size into one fixed-length vector. Each layer lets every cell update its
   representation from its neighbours, weighting them by learned attention;
   after three layers each cell has "seen" three hops out. A readout pools all
   cells into a single neighbourhood embedding.
5. **Contrastive training** — take a neighbourhood, make two slightly perturbed
   copies (drop a few edges, hide some features, drop outer cells, add noise),
   and train the encoder to embed the two copies close together while pushing
   other neighbourhoods away (NT-Xent loss). The encoder learns which properties
   survive perturbation — cell composition, spatial organisation, local
   interaction patterns — and those become the basis of similarity.
6. **Clustering** — embed every cell, optionally UMAP-reduce 64 → ~10 dims,
   build a kNN similarity graph over embeddings, run Leiden. Each cluster is a
   motif; every cell gets a label; the labels feed the motif catalogue, motif
   maps and cross-stiffness statistics.

The one design principle carried through every stage: **features ride on
nodes, geometry defines edges.** Nearest neighbours are computed on centroids,
never on features, and raw (x, y) is never a node feature. Condition labels
(stiffness, nicotine, ECM, imaging date) are graph-level tags only, so the
encoder cannot trivially learn them.

### Data

- Three-channel fluorescence tiles (682 × 682 px) of endothelial monolayers.
  Tiles are stored in RGB-plane order **R = TAGLN** (mesenchymal marker),
  **G = VE-cadherin** (endothelial junction marker), **B = DAPI** (nuclei).
- Conditions: substrates of **5, 150, 500 (two imaging dates), and 900 kPa**
  plus tissue-culture plastic; ~6 000 tiles, ~1.27 M cells in total.
- Images, masks and per-cell CSVs live on Google Drive
  (`Fusion AI/Prof Huang Project/Cellpose feature extractions/`) and are
  synced to a Lightning AI Studio with `rclone`. Every script takes a
  `--push-to-drive` flag to back its outputs up, since Studios are ephemeral.
- Per-well metadata (stiffness, nicotine / no nicotine, ECM coating, imaging
  date) is carried as a graph-level `condition` tag, never as a node feature.
- Nothing large is committed to git: no images, masks, graphs, checkpoints or
  run outputs.

**File-format caveat.** The tiles are standard **8-bit RGB TIFFs**, not
scientific multi-channel TIFFs: the original acquisitions were almost certainly
12/16-bit per channel, have been compressed to 0–255 and pseudo-coloured, and
carry no embedded metadata saying which channel is which marker (the channel
map above comes from the lab). Dynamic range has been lost. If raw acquisition
files (16-bit per channel with OME metadata) become available, the pipeline
should be re-run on them.

The local working root is resolved in this order: `$CELLPOSE_LOCAL_ROOT`, the
Lightning path `/teamspace/studios/this_studio/cellpose_work`, else
`~/cellpose_work`.

### Stage 1 — Segmentation and per-cell features

`motifs/stage1_segmentation/`

| File | What it does |
|---|---|
| `lightning_cellpose_batch.py` | Runs **Cellpose-SAM** over every tile, writes one label mask per tile. Resumable; skips tiles that already have masks. |
| `automated_cellwise_feature_extraction.py` | For every cell in every mask: area, elongation, DAPI mean, VE-cadherin **membrane-band** mean, TAGLN whole-cell mean, centroid, and the **EndMT score** `M / (E + M)` with `E` = background-subtracted VE-cadherin, `M` = background-subtracted TAGLN, so 0 ≈ endothelial and 1 ≈ mesenchymal. |
| `colab_cellpose.ipynb`, `lightning_colab_cellpose.ipynb` | Notebook forms of the segmentation step (Colab and Lightning). |

Output: `cellwise_metadata/<condition>/<tile>_metadata.csv`, one row per cell.

### Stage 2 — Master table and spatial graphs

`motifs/stage2_graphs/` — design in [docs/stage2_graph_construction.md](docs/stage2_graph_construction.md).

| File | What it does |
|---|---|
| `assemble_master_table.py` | Concatenates every Stage 1 CSV, adds `image_id` and `condition`, and **z-scores the five morphology/intensity features globally** across the whole dataset. `endmt_score` is kept raw so its meaning survives. Writes `master_table.csv` + `normalization_stats.json`. |
| `build_graphs.py` | One PyTorch Geometric `Data` graph per image. Nodes carry `[5 z-scored features, endmt_score, on_border]`; edges come from **k-NN on centroids (k = 8), symmetrised by union and pruned by an adaptive per-image distance cap**; edge attributes are `[dist, dist_norm, inv_dist, is_mutual]`. Also writes a per-image QC table and overlay PNGs for eyeballing. |

Output: `graphs/<condition>/<image_id>.pt`, `graphs/feature_names.json`,
`graph_qc.csv`, `graph_overlays/`.

### Stage 3 — Self-supervised neighbourhood encoder

`motifs/stage3_encoder/` and `models/neighborhood_encoder.py` — design in
[docs/stage3_gnn_encoder.md](docs/stage3_gnn_encoder.md); every architectural
choice and the alternatives it beats are argued in
[docs/stage3_encoder_methodology.md](docs/stage3_encoder_methodology.md).

| File | What it does |
|---|---|
| `sample_subgraphs.py` | Extracts a cell's **2-hop ego-subgraph** (the unit a motif is assigned to) and writes an inspection sample with preview PNGs. The same extractor runs on-the-fly during training — the ~1.27 M subgraphs are never materialised. |
| `models/neighborhood_encoder.py` | **GATv2** encoder (3 layers, edge-feature-conditioned attention, multi-head) + mean-pool + projection head; NT-Xent loss; optional gradient-reversal **condition adversary** and an EndMT-retention head. |
| `train_encoder.py` | Contrastive training: each subgraph gets two mild augmented views (edge drop, feature mask, outer-node drop, feature noise) that must embed close together. AdamW + cosine LR, checkpointing, optional W&B. Flags for de-confounding: `--adversarial --adv-target {condition,date,stiffness}`, `--intensity-jitter`, `--batch-norm-features`, `--endmt-head`. |
| `embed_cells.py` | Runs the frozen encoder over every cell's real neighbourhood and writes one **64-d embedding per cell** (`embeddings.parquet`). Uses the encoder output `h`, not the projection `z`. |
| `probe_embeddings.py` | **Confound probe**: can a classifier predict condition from the embeddings well above chance (grouped CV, MLP probe)? `--decompose` splits any leak into a **batch (imaging-date)** component vs a **biology (stiffness)** component. Also renders UMAPs coloured by condition and by EndMT score. |

Output: `stage3/encoder.pt`, `stage3/embeddings.parquet`, `stage3/probe_report.json`, UMAP PNGs.

### Stage 4 — Motif clustering, validation and figures

`motifs/stage4_motifs/`

| File | What it does |
|---|---|
| `cluster_motifs.py` | UMAP-reduce the embeddings, fit **Leiden** on a kNN graph of a subsample (k-means fallback), assign every cell to the nearest centroid. Writes `motifs.parquet`, a per-motif feature profile (what each motif *is*), motif × condition and motif × imaging-date tables — the latter is the batch-artifact check. |
| `validate_motifs.py` | Are the motifs real? **Spatial coherence** (edge concordance vs a within-image permutation null — ratio ≈ 1 means no spatial information), **recurrence** across images, **stability** across seeds (ARI), and **separation** (silhouette). |
| `robustness_motifs.py` | How much can each motif be trusted? Per-cell **assignment confidence** (centroid margin), **bootstrap stability** per motif, and **encoder reproducibility** (agreement between two encoders trained with different seeds). |
| `baseline_no_neighbors.py` | Control: k-means on the same six per-cell features with **no neighbourhood information**, written in the same format so the validation and robustness scripts run unchanged. If it matches the GNN, spatial context isn't earning its keep. |
| `plot_motif_maps.py` | Paints motif labels back onto tissue: per-image motif map beside the original tile, paired grids across conditions, per-motif EndMT breakdowns, a **motif catalog** and per-motif **galleries** of ego-subgraphs. |
| `plot_motif_frequency.py` | The headline figure: stacked bars of motif proportion per condition ordered by stiffness, plus grouped bars with ± SEM across images. `--merge-by-stiffness` pools the two 500 kPa dates; `--labels` names the motifs. |

### What labels are (and are not) needed

Motif discovery is unsupervised, so almost nothing has to be hand-annotated.

| Tier | What | Needed for | How obtained |
|---|---|---|---|
| **1 — required** | Stiffness / condition per image | graph-level tag; adversary target | experimental metadata, already known |
| **1 — required** | Instance segmentation mask per image | defines the graph nodes | automated (Cellpose-SAM); no biological labelling, just "where are the cells" |
| **2 — strongly recommended** | Marker intensity per cell (VE-cadherin, TAGLN, DAPI) | node features; makes motifs interpretable by marker composition | computed from mask + image, not annotated |
| **2 — strongly recommended** | EndMT score per cell | node feature; the biological readout | derived from marker intensities by a formula — the "labelling" work is choosing the formula/thresholds, not labelling cells |
| **3 — validation only** | ~300–500 manually validated cell-state calls, spread across conditions | confirm the marker-based EndMT score agrees with expert review | manual review (not used in training) |
| **3 — validation only** | ~30–50 expert-identified example neighbourhoods | sanity check that discovered motifs include patterns experts already recognise | pointed out by Prof. Huang / Carver (not used in training) |

Without Tier 2 the encoder still trains and motifs still emerge, but they would
be defined purely by morphology and arrangement, with no biological context —
the EndMT story would be much weaker. Tier 3 is what lets us *make claims*
about the results rather than just produce them.

### Running the pipeline

All commands are run from the repository root. Stage 1 needs a GPU for
Cellpose; Stage 3 training is GPU-recommended; everything else is CPU.

```bash
pip install -r requirements-motifs.txt

# Stage 1 — on Lightning AI (or Colab via the notebooks)
python motifs/stage1_segmentation/lightning_cellpose_batch.py
python motifs/stage1_segmentation/automated_cellwise_feature_extraction.py

# Stage 2
python motifs/stage2_graphs/assemble_master_table.py --push-to-drive
python motifs/stage2_graphs/build_graphs.py --k 8 --d-max-mult 3.0 --push-to-drive

# Stage 3
python motifs/stage3_encoder/sample_subgraphs.py                      # eyeball 2-hop neighbourhoods first
python motifs/stage3_encoder/train_encoder.py --epochs 100 --batch-size 256 --seed 0
python motifs/stage3_encoder/embed_cells.py --encoder <root>/stage3/encoder.pt
python motifs/stage3_encoder/probe_embeddings.py --decompose            # did condition leak? batch or biology?

# If the probe shows a batch leak, retrain scrubbing the imaging date:
python motifs/stage3_encoder/train_encoder.py --epochs 100 --adversarial --adv-target date --intensity-jitter

# Stage 4
python motifs/stage4_motifs/cluster_motifs.py    --embeddings <root>/stage3/embeddings.parquet --out-dir <root>/stage4
python motifs/stage4_motifs/validate_motifs.py   --motifs <root>/stage4/motifs.parquet --embeddings <root>/stage3/embeddings.parquet
python motifs/stage4_motifs/robustness_motifs.py --motifs <root>/stage4/motifs.parquet --embeddings <root>/stage3/embeddings.parquet
python motifs/stage4_motifs/baseline_no_neighbors.py --master-table <root>/master_table.csv --out-dir <root>/baseline_cells --n-clusters 6
python motifs/stage4_motifs/plot_motif_maps.py      --motifs <root>/stage4/motifs.parquet --graphs-dir <root>/graphs --imgs-dir <root>/imgs --catalog --galleries
python motifs/stage4_motifs/plot_motif_frequency.py --motifs <root>/stage4/motifs.parquet --merge-by-stiffness
```

Every script has `--help`; each writes a JSON report and/or a log alongside its
figures so a run is self-describing.

### Results so far

Findings from the latest full run (internally "v4"); the run artefacts live on
Drive / Lightning, not in git.

- The confound probe's `--decompose` analysis showed the condition leak in the
  embeddings was a **batch (imaging-date) effect, not a stiffness effect** —
  which is why the adversary targets `date` and intensity jitter is used
  rather than scrubbing stiffness (the biology we want to keep).
- Leiden clustering yields **6 motifs** over ~1.27 M cells, silhouette ≈ 0.36.
  Motifs are spatially coherent and recur across images.
- **Open concern:** five of the six motifs have near-identical mean EndMT
  (≈ 0.39); motifs currently separate on VE-cadherin intensity more than on
  EndMT state. Candidate fixes under consideration: Delaunay / radius graphs
  in Stage 2 instead of kNN, and stronger EndMT retention in Stage 3.
- Next steps: interpret the motif-frequency figure across stiffness, run the
  cross-seed encoder-reproducibility test, compare against the no-neighbour
  baseline, and name the motifs with the lab (pick representative cells nearest
  each centroid, report each motif's mean composition, and have a biologist
  look at the examples).

### Open questions

- **Marker panel.** Three channels (VE-cadherin, TAGLN, DAPI) are enough to
  score EndMT, but CD31, vimentin or N-cadherin would add confidence to the
  cell-state calls and richer node features.
- **Raw data.** The current tiles are 8-bit RGB exports (see
  [Data](#data)); 16-bit multi-channel acquisitions would restore dynamic range
  and remove the need for a hand-maintained channel map.
- **Graph definition.** kNN (k = 8) is the current edge rule. Delaunay
  triangulation or a radius graph would connect only cells that share a tissue
  boundary and is a candidate fix for the flat-EndMT motifs above.
- **Validation labels.** The Tier 3 manual cell-state calls and expert example
  neighbourhoods have not yet been collected.

---

## 2. Track A — MAE + Deformable-DETR detection

A two-stage detector for cardiomyocytes on 5 kPa vs 900 kPa substrates
(444 wells, each with a 35-plane × 4-channel z-stack, a focus-stacked image and
a hybrid projection):

1. **Self-supervised pretraining** — three masked autoencoders (2D focused, 2D
   hybrid, 3D tubelet-patched z-stack) trained jointly, all conditioned on
   substrate stiffness through a shared `StiffnessMLP`.
2. **Detection fine-tuning** — Deformable-DETR (`SenseTime/deformable-detr`)
   on bounding-box labels, with mAP / AP50 / AP75 / mean-IoU evaluation.

The full write-up — data layout, architecture, configuration reference,
metrics and commands — is in [docs/mae_detr_pipeline.md](docs/mae_detr_pipeline.md).

```bash
pip install -r requirements.txt
python scripts/train_pretrain.py --smoke     # forward/backward on random tensors, ~10 s on CPU
pytest tests/ -v                             # 20 synthetic-data integration tests
```

**Status:** the pipeline runs end-to-end and the tests pass, but the detector
still uses the stock ResNet-50 backbone — wiring the pretrained MAE ViT into
the DETR feature pyramid (`models/weight_loaders.py`,
`detection.mae_encoder_ckpt`) is unfinished, and no full-dataset mAP has been
recorded.

---

## 3. Repository layout

```
Huang-Lab-Work/
├── README.md                     ← this file
├── requirements.txt              Track A dependencies
├── requirements-motifs.txt       Track B dependencies
│
├── motifs/                       TRACK B — EndMT spatial-motif pipeline
│   ├── stage1_segmentation/      Cellpose-SAM + per-cell features + EndMT score
│   ├── stage2_graphs/            master table + kNN spatial graphs
│   ├── stage3_encoder/           subgraph sampling, GATv2 training, embedding, confound probe
│   └── stage4_motifs/            clustering, validation, robustness, baseline, figures
│
├── models/                       Shared model code
│   ├── neighborhood_encoder.py   Track B: GATv2 encoder, NT-Xent, adversary, EndMT head
│   └── mae.py, mae_volume.py, multi_mae.py, stiffness.py, pos_embed.py, weight_loaders.py   (Track A)
│
├── config/   data/   training/   scripts/   utils/   tests/     TRACK A packages
├── notebooks/colab_train.ipynb   Track A Colab GPU training notebook
│
├── docs/
│   ├── mae_detr_pipeline.md              Track A: full documentation
│   ├── stage2_graph_construction.md      Track B: graph design + QC
│   ├── stage3_gnn_encoder.md             Track B: encoder + contrastive training plan
│   └── stage3_encoder_methodology.md     Track B: design rationale, alternatives considered
│
└── archive/                      Read-only reference: original Colab dataloaders and the
                                  vendored facebookresearch/mae tree (CC-BY-NC-4.0)
```

## 4. Setup

```bash
git clone https://github.com/racyun/Huang-Lab-Work.git
cd Huang-Lab-Work
python -m venv .venv && source .venv/bin/activate

# PyTorch first, matching your CUDA version: https://pytorch.org/get-started/locally/
pip install torch torchvision

pip install -r requirements-motifs.txt   # Track B
pip install -r requirements.txt          # Track A (optional)
```

Track B scripts read and write under `<root>` (see [Data](#data)); Track A
takes machine-specific paths from `config/local.yaml`
(`cp config/local.yaml.example config/local.yaml`, gitignored).

For Lightning AI, the one-time `rclone` setup for the Drive remote is
documented at the top of `motifs/stage1_segmentation/lightning_cellpose_batch.py`.

## 5. Documentation

| Document | Covers |
|---|---|
| [docs/stage2_graph_construction.md](docs/stage2_graph_construction.md) | Why kNN-on-centroids, the node/edge feature contract, QC suite, gap analysis against Stage 1 |
| [docs/stage3_gnn_encoder.md](docs/stage3_gnn_encoder.md) | Ego-subgraph sampling, GATv2 encoder, augmentations, NT-Xent, adversary, how to run |
| [docs/stage3_encoder_methodology.md](docs/stage3_encoder_methodology.md) | Every design decision with the alternatives it beats (GNN vs MLP/CNN, GAT vs GCN, contrastive vs autoencoding, …) |
| [docs/mae_detr_pipeline.md](docs/mae_detr_pipeline.md) | Track A end to end: data, MAE variants, stiffness conditioning, DETR fine-tuning, config reference |

## 6. Acknowledgements

- **Cellpose-SAM** — Pachitariu, M., Rariden, M. & Stringer, C. (2025). Cellpose-SAM: superhuman generalization for cellular segmentation.
- **GATv2** — Brody, S., Alon, U. & Yahav, E. (2022). How Attentive are Graph Attention Networks? (ICLR). Via PyTorch Geometric.
- **NT-Xent / SimCLR** — Chen, T., Kornblith, S., Norouzi, M. & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations.
- **Leiden** — Traag, V. A., Waltman, L. & van Eck, N. J. (2019). From Louvain to Leiden. Via `leidenalg` / `python-igraph`.
- **UMAP** — McInnes, L., Healy, J. & Melville, J. (2018).
- **MAE** — He, K. et al. (2021). Masked Autoencoders Are Scalable Vision Learners. The vendored codebase in `archive/legacy_facebook_mae/` is CC-BY-NC-4.0, Meta Platforms, Inc.
- **VideoMAE** — Tong, Z. et al. (2022); inspiration for the 3D tubelet masking.
- **Deformable DETR** — Zhu, X. et al. (2020); weights from the HuggingFace `SenseTime/deformable-detr` checkpoint.
- **timm** — Wightman, R. (2019).

Developed in Prof. Ngan Huang's lab; channel maps and the EndMT scoring
convention follow the lab's protocol.
