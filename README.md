# Huang Lab — Tissue-on-a-Chip Cardiovascular Imaging Pipeline

This repository implements an end-to-end machine learning pipeline for automatically detecting and localizing cells in microscopy images of cardiovascular tissue-on-a-chip experiments. The pipeline has two stages: first, it learns rich visual representations from unlabelled microscopy images using a technique called **Masked Autoencoding (MAE)**; then it uses those representations to fine-tune a **Deformable-DETR** object detector to predict bounding boxes around individual cells. The data comes from chips seeded with cardiomyocytes grown on substrates of two different mechanical stiffnesses (5 kPa and 900 kPa), and the model is designed to handle both conditions simultaneously.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Data](#2-data)
3. [Model Architecture](#3-model-architecture)
   - [What is MAE?](#what-is-mae-masked-autoencoding)
   - [Focused Encoder (2D MAE)](#focused-encoder-2d-mae)
   - [Hybrid Encoder (2D MAE)](#hybrid-encoder-2d-mae)
   - [Volume Encoder (3D MAE)](#volume-encoder-3d-mae)
   - [Stiffness Conditioning](#stiffness-conditioning)
   - [Multi-Encoder Wrapper](#multi-encoder-wrapper)
   - [What is Deformable-DETR?](#what-is-deformable-detr)
   - [Detection Fine-tuning](#detection-fine-tuning)
4. [Training Pipeline](#4-training-pipeline)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Project Structure](#6-project-structure)
7. [Setup and Running](#7-setup-and-running)
8. [Configuration](#8-configuration)
9. [Results So Far](#9-results-so-far)
10. [Acknowledgements](#10-acknowledgements)

---

## 1. Overview

Tissue-on-a-chip experiments produce large volumes of microscopy images. Manually annotating every cell in every image is time-consuming and does not scale. This project automates that process with a two-stage deep learning pipeline:

```
Stage 1: Self-supervised Pretraining (MAE)
─────────────────────────────────────────
Raw microscopy images (no labels needed)
        │
        ▼
  Three parallel encoders learn to
  reconstruct masked image patches
        │
        ▼
  Encoders now "understand" cell
  structure and tissue morphology

Stage 2: Detection Fine-tuning (Deformable-DETR)
─────────────────────────────────────────────────
Pretrained encoder weights
        │
        ▼
  Fine-tune on labelled bounding boxes
        │
        ▼
  Model predicts cell locations
  in new images
```

A key design feature is **stiffness conditioning**: the model knows whether each image comes from a 5 kPa (soft) or 900 kPa (stiff) substrate and uses that information when processing images. This matters because cells grown on substrates of different stiffnesses can look and behave differently.

---

## 2. Data

### Experimental Setup

Each experiment well contains cardiomyocytes imaged at multiple focal planes (z-slices) using fluorescence microscopy. There are 222 wells per stiffness condition, giving **444 total samples** across two conditions:

- **900 kPa** — stiff substrate (mimics scar tissue stiffness)
- **5 kPa** — soft substrate (closer to healthy heart tissue stiffness)

### Three Image Modalities

For each well, three different image representations are available:

| Modality | Description | Shape |
|---|---|---|
| **Z-stack** | Full 3D volume: 35 focal planes × 4 fluorescence channels = 140 TIFF files | `[140, H, W]` |
| **Focused** | A single sharp 2D image computed by merging the sharpest parts of each z-level across the stack | `[C, H, W]` |
| **Hybrid** | A projection that combines fluorescence channels into a single composite view | `[C, H, W]` |

Think of the z-stack as a full 3D scan of the tissue, the focused image as the "best 2D photo" you could take of it, and the hybrid image as a specially processed composite view emphasizing different biological structures.

### Data Directory Layout

All data lives under a parent directory (`250918_Deepmind_CV_Collaboration/`) with this structure:

```
250918_Deepmind_CV_Collaboration/
│
├── 250811_Athchip_noninflam_900kPa/              # Raw z-stack images, 900 kPa
│   ├── W001/P00001/
│   │   ├── 10X_W001_P00001_Z001_CH1.tif          # Z-slice 1, channel 1
│   │   ├── 10X_W001_P00001_Z001_CH2.tif          # Z-slice 1, channel 2
│   │   ├── 10X_W001_P00001_Z001_CH3.tif
│   │   ├── 10X_W001_P00001_Z001_Overlay.tif
│   │   └── ... (35 z-levels × 4 channels = 140 files per well)
│   ├── W002/P00001/
│   └── ... W222/P00001/
│
├── 250814_Athchip_non-inflam_5kPa/               # Raw z-stack images, 5 kPa
│   └── (same structure)
│
├── 20251027_2123__FocusStack_250811_..._900kPa/  # Focus-stacked images, 900 kPa
│   ├── ..._W001_P00001_focus_stacked.tif         # One file per well (flat folder)
│   └── ...
│
├── 20251027_2215__FocusStack_250814_..._5kPa/    # Focus-stacked images, 5 kPa
│   └── (same structure)
│
├── 250811_Athchip_noninflam_900kPa_HybridResults/  # Hybrid projections, 900 kPa
│   ├── hybrid_results_W001/
│   │   └── W001_hybrid_projection.tif
│   └── ... hybrid_results_W222/
│
├── 250814_Athchip_non-inflam_5kPa_HybridResults/   # Hybrid projections, 5 kPa
│   └── (same structure)
│
└── bbox_txt_for_training/                          # Bounding box labels
    ├── 900kPa/
    │   ├── W001.txt    # One line per box: x1,y1,x2,y2
    │   └── ... W222.txt
    └── 5kPa/
        └── ...
```

### Bounding Box Labels

Label files contain one bounding box per line in pixel coordinates:

```
x1,y1,x2,y2
10.0,20.0,50.0,60.0
130.5,80.0,200.0,145.0
```

The detection pipeline automatically converts these to normalized center-x, center-y, width, height format (values between 0 and 1) as required by Deformable-DETR.

### Well Discovery and Alignment

The data loader (`data/combined.py`) builds the well list by **intersection**: a well is only included in training if it has all four components — a z-stack folder, a focused image, a hybrid image, and a label file. This handles cases where a small number of wells may be missing from one modality without manual intervention.

---

## 3. Model Architecture

### What is MAE? (Masked Autoencoding)

Masked Autoencoding is a self-supervised learning technique — meaning the model trains itself without needing human-annotated labels. The idea is borrowed from masked language modeling in NLP (like how BERT learns by predicting missing words in a sentence), but applied to images.

Here is how it works:

```
Original image patches:
[ A ][ B ][ C ][ D ][ E ][ F ][ G ][ H ]

After random masking (75% masked):
[ A ][   ][   ][ D ][   ][   ][ G ][   ]

Encoder sees only unmasked patches (25%):
[ A ][ D ][ G ]  →  Encoder  →  Representations

Decoder tries to reconstruct everything:
  Representations + mask tokens  →  Decoder  →  [ A ][ B* ][ C* ][ D ][ E* ][ F* ][ G ][ H* ]

Loss: MSE between B* vs B, C* vs C, etc. (masked patches only)
```

The reason this works is that to fill in the missing 75% of an image, the model must learn to understand the actual structure and content of the image — it cannot just memorize the input. After pretraining, the encoder has learned rich, generalizable visual representations without ever seeing a single label.

### The Full Architecture at a Glance

```
                     ┌─────────────────────────────────────────────┐
                     │         STAGE 1: MAE Pretraining             │
                     │                                               │
  Focused image  ──► │  Focused Encoder (2D ViT)  ──► Decoder ──►  │ loss_focused
  Hybrid image   ──► │  Hybrid Encoder  (2D ViT)  ──► Decoder ──►  │ loss_hybrid
  Z-stack volume ──► │  Volume Encoder  (3D ViT)  ──► Decoder ──►  │ loss_volume
  Stiffness kPa  ──► │  StiffnessMLP ──────────────────────────►   │
                     │              (injected into all encoders)     │
                     │                                               │
                     │  Total loss = loss_f + loss_h + loss_v        │
                     └─────────────────────────────────────────────┘
                                         │
                              Pretrained focused encoder
                                         │
                     ┌─────────────────────────────────────────────┐
                     │       STAGE 2: Detection Fine-tuning         │
                     │                                               │
  Focused image  ──► │  Backbone (initialized from pretrained enc.) │
                     │  ──► Deformable-DETR neck + detection head   │
                     │  ──► 300 object queries                       │
                     │  ──► Predicted bounding boxes + class scores  │
                     └─────────────────────────────────────────────┘
```

### Focused Encoder (2D MAE)

**File:** `models/mae.py` — `MaskedAutoencoderViT` / `MAEViTEncoder`

This encoder processes the focus-stacked image — a single sharp 2D image representing the full tissue in one plane.

The encoder is a **Vision Transformer (ViT)**. Here is what that means in plain terms:

1. **Patch Embedding**: The image is divided into a grid of non-overlapping 16×16 pixel patches. Each patch is linearly projected into a vector of size 384 (the `embed_dim`). A 224×224 image produces (224/16)² = 196 patches.

2. **Positional Encoding**: Because transformers process all patches simultaneously (not in order), each patch vector gets a unique positional signal added to it so the model knows where each patch came from in the image. These are fixed sinusoidal encodings, not learned.

3. **Stiffness Injection**: The stiffness embedding (see below) is added to every patch vector before the transformer processes them.

4. **Random Masking**: 75% of patch tokens are randomly removed. Only the remaining 25% are passed to the transformer.

5. **Transformer Encoder**: A stack of self-attention blocks processes the visible patches. Each attention block lets every patch "look at" every other visible patch, allowing the model to reason about context.

6. **Decoder**: A smaller transformer decoder takes the encoder outputs plus learnable "mask" tokens (placeholders for the missing patches) and reconstructs the full image. The loss is the MSE between predicted and actual pixel values for the masked patches only.

### Hybrid Encoder (2D MAE)

**File:** `models/mae.py`

Identical architecture to the focused encoder, but trained on the hybrid projection image. The two encoders have **completely independent weights** — the focused encoder specializes in one view of the tissue, the hybrid encoder specializes in the other. They share the same decoder design but not decoder weights.

### Volume Encoder (3D MAE)

**File:** `models/mae_volume.py` — `MaskedAutoencoderVolume`

This encoder processes the full z-stack as a 3D volume rather than a single 2D image. The key challenge is that the z-stack is 140 slices deep — a much larger input than a single 2D image.

The solution is **tubelet patching** (borrowed from VideoMAE, a technique originally developed for video understanding):

```
Z-stack: 140 slices × 64 × 64 pixels

Tubelet patching with tublet_z=4, patch_xy=16:

Along Z:         140 / 4  = 35 tubes
Along H:          64 / 16 = 4 positions
Along W:          64 / 16 = 4 positions
─────────────────────────────────────
Total tokens:   35 × 4 × 4 = 560 tokens per sample
```

Each "tube" is a 4×16×16 voxel block. The `VolumeTubeEmbed` module extracts these tubes using a 3D convolution with kernel and stride `(4, 16, 16)`. A group of 4 consecutive z-slices corresponds to one complete focal level with all 4 fluorescence channels — so each tube has full spectral information at one spatial position in depth.

After tubelet extraction, the rest follows the same pattern as the 2D MAE: positional encoding, stiffness injection, masking (75%), transformer encoder, decoder, MSE loss on masked tubes.

### Stiffness Conditioning

**File:** `models/stiffness.py` — `StiffnessMLP`

The substrate stiffness (5 or 900 kPa) is a single number per sample. This MLP converts it into a vector that matches the patch embedding dimension, which is then added to every patch token in all three encoders:

```
kPa value (e.g. 900.0)
       │
       ▼
  Normalize: 900.0 / 900.0 = 1.0   (or use log1p mode)
       │
       ▼
  MLP: Linear(1 → 64) → GELU → Linear(64 → 384)
       │
       ▼
  stiffness_embedding: [batch, 384]
       │
       ▼
  Added to ALL patch tokens in ALL three encoders
  (broadcast over the sequence length dimension)
```

This design means the stiffness information is woven into every patch at every transformer layer via the attention mechanism. A single `StiffnessMLP` is shared across all three encoders — it learns one universal way to represent substrate stiffness.

**Normalization modes** (set via `kpa_input_mode` in config):

| Mode | Formula | 5 kPa → | 900 kPa → |
|---|---|---|---|
| `divide` (default) | `kpa / 900.0` | 0.0056 | 1.0 |
| `log1p` | `log(1 + kpa)` | 1.79 | 6.80 |
| `identity` | raw kPa | 5.0 | 900.0 (not recommended) |

### Multi-Encoder Wrapper

**File:** `models/multi_mae.py` — `MultiEncoderMAE`

This module ties the three encoders and the stiffness MLP together into a single trainable unit:

```python
# Conceptually what MultiEncoderMAE.forward() does:

stiff_emb = stiffness_mlp(batch["stiffness"])   # shared embedding

loss_f, _, _ = mae_focused(batch["focused"], stiff_emb)
loss_h, _, _ = mae_hybrid(batch["hybrid"],   stiff_emb)
loss_v, _, _ = mae_volume(batch["zstack"],   stiff_emb)

total_loss = w_f * loss_f + w_h * loss_h + w_v * loss_v
```

All three encoders train jointly in one backward pass. The optimizer uses **four separate parameter groups** — one for each encoder and one for the stiffness MLP — each with an independent learning rate multiplier. This allows, for example, the slower-to-converge volume encoder to use a smaller learning rate than the 2D encoders.

### What is Deformable-DETR?

**DETR** stands for DEtection TRansformer. Unlike traditional object detectors (like YOLO or Faster R-CNN) that use anchor boxes and region proposals, DETR frames object detection as a **direct set prediction problem**: it predicts all bounding boxes simultaneously using a set of learned "object queries."

Think of it this way: DETR starts with 300 blank question-marks, each representing a potential object. Through cross-attention with the image features, each question-mark either finds an object to describe (outputting a box + class score) or stays empty (outputting "no object").

**Deformable-DETR** is an improved version that is faster and handles multi-scale features better. Instead of every query attending to every pixel, it attends to a small set of key sampling points that move deformably around regions of interest. We use the HuggingFace checkpoint `SenseTime/deformable-detr` as the starting point.

### Detection Fine-tuning

**Files:** `training/main_detect.py`, `training/engine_detect.py`

The detection model uses the focused image as input (configurable to hybrid via `detection.image_mode`). Before training:

1. Images are resized so the short side is 800 pixels (preserving aspect ratio)
2. Bounding boxes in `x1,y1,x2,y2` pixel format are converted to normalized `cx,cy,w,h` (center-x, center-y, width, height, all divided by image dimensions) as expected by DETR
3. A boolean `pixel_mask` is added to each image indicating which pixels are real vs. padding (needed when batching images of different sizes)

After each training epoch, the model is evaluated on the full training set to compute detection metrics (see Section 5).

The intended future upgrade is to replace the stock ResNet-50 backbone in Deformable-DETR with the pretrained `MAEViTEncoder`. The infrastructure for this is already in place (`models/weight_loaders.py`, `detection.mae_encoder_ckpt` config key), but the ViT-to-DETR feature pyramid wiring has not yet been implemented.

---

## 4. Training Pipeline

### Stage 1: MAE Pretraining

**Script:** `scripts/train_pretrain.py`

**What happens each epoch:**

```
For each batch:
  1. Load focused image, hybrid image, z-stack, stiffness kPa
  2. Compute stiffness embedding (StiffnessMLP)
  3. Forward pass through all three MAEs (with stiffness conditioning)
  4. Compute weighted reconstruction losses: loss = loss_f + loss_h + loss_v
  5. Backward pass (with optional AMP mixed precision)
  6. Optional gradient clipping (grad_clip in config)
  7. Optimizer step (AdamW, four param groups)
  8. Learning rate schedule step

After each epoch:
  - Log per-head losses to outputs/pretrain_log.jsonl
  - Log metrics to Weights & Biases (if enabled)
  - Save checkpoint every 10% of total epochs
```

**Learning rate schedule:**

```
LR
 │         /‾‾‾‾‾‾‾‾‾‾‾\
 │        /              \
 │       /                \
 │      /                  \_____________
 │─────/
 │
 0   warmup   peak          end
     (10 ep)  (1.5e-4)
```

Linear warmup for `warmup_epochs` epochs, then cosine decay to `min_lr`. Each of the four parameter groups follows the same schedule shape but scaled by its `lr_mult_*` multiplier.

### Stage 2: Detection Fine-tuning

**Script:** `scripts/train_detect.py`

Downloads the `SenseTime/deformable-detr` checkpoint from HuggingFace on first run (~160 MB). Trains with AdamW for `detection.epochs` epochs. After each epoch, runs an evaluation pass to compute mAP and related metrics.

### Disk Caching

Loading 140 TIFF files per well per batch is slow. The `CachedTissueChipDataset` wrapper saves each fully processed sample as a `.pt` file on the first read:

- Key: `sha256(split_name + well_id)[:24]` — stable, collision-resistant
- Location: `dataset.cache_dir` in config (set to a fast local disk or SSD)
- Effect: First epoch is slow (reads all TIFFs, ~10 seconds per well). All subsequent epochs load in ~1 second per well — roughly a 10x speedup.

Set `dataset.cache_dir` to `null` to disable caching (useful when debugging data loading).

### Weights and Biases Integration

All training can optionally log to [Weights & Biases](https://wandb.ai) for experiment tracking. The integration is fully optional — every W&B function is wrapped in no-ops that activate only when `wandb.enabled: true` in config.

Enable it:
```bash
pip install wandb
wandb login
# then add --wandb to the training command
```

Metrics logged during pretraining:

| Metric | Description |
|---|---|
| `train/step_loss` | Loss every N gradient steps |
| `train/loss` | Total epoch loss |
| `train/loss_focused` | Focused-head reconstruction loss |
| `train/loss_hybrid` | Hybrid-head reconstruction loss |
| `train/loss_volume` | Volume-head reconstruction loss |
| `lr/focused` | Learning rate for focused encoder |
| `lr/hybrid` | Learning rate for hybrid encoder |
| `lr/volume` | Learning rate for volume encoder |
| `lr/stiffness` | Learning rate for StiffnessMLP |

Metrics logged during detection fine-tuning:

| Metric | Description |
|---|---|
| `detect/step_loss` | Loss every N gradient steps |
| `detect/loss` | Epoch detection loss |
| `detect/lr` | Learning rate |
| `eval/mAP` | Mean Average Precision @ IoU [0.5:0.95] |
| `eval/AP50` | Average Precision at IoU 0.50 |
| `eval/AP75` | Average Precision at IoU 0.75 |
| `eval/mean_iou` | Mean best-match IoU per ground-truth box |

---

## 5. Evaluation Metrics

Detection quality is measured with standard object detection metrics. Here is what each one means in plain terms:

### IoU (Intersection over Union)

The most fundamental metric. For a single predicted box and a ground-truth box:

```
         ┌──────────────┐
         │   Ground     │
         │   Truth      │
         │      ┌───────┼──────┐
         │      │  ///  │      │
         │      │ inter-│      │
         └──────┼───────┘      │
                │  Prediction  │
                └──────────────┘

IoU = Area of intersection / Area of union
```

IoU = 1.0 means perfect overlap. IoU = 0 means the boxes do not touch at all. A prediction is typically considered a "hit" (true positive) if IoU > 0.5.

### AP (Average Precision)

For a given IoU threshold, AP summarizes the precision-recall curve into a single number. Precision measures "how often your predictions are correct"; recall measures "how many ground-truth cells you found." AP is high when the model is both precise and comprehensive.

### mAP (mean Average Precision)

**`eval/mAP`** — Averaged over multiple IoU thresholds from 0.5 to 0.95 in steps of 0.05. This is the standard COCO metric. A score of 1.0 means perfect detection at all strictness levels; 0.0 means no detections were correct.

**`eval/AP50`** — AP at the single threshold IoU = 0.50. This is a lenient metric: a box just needs to overlap with the ground truth by more than half.

**`eval/AP75`** — AP at IoU = 0.75. This is stricter — predictions must be substantially more accurate in their localization.

### Mean IoU

**`eval/mean_iou`** — For each ground-truth box, find the predicted box with the highest IoU and record that value. Average across all ground-truth boxes in the dataset. This gives an intuitive sense of "how well-localized the typical cell detection is."

These metrics are computed using `torchmetrics.detection.MeanAveragePrecision` with the `faster_coco_eval` backend for speed.

---

## 6. Project Structure

```
Huang-Lab-Work/
│
├── config/
│   ├── __init__.py            # load_config(): deep-merges default + local YAML
│   ├── settings.py            # Python dataclasses (FullConfig, MultiMAEConfig, etc.)
│   ├── default.yaml           # All default hyperparameters (committed to repo)
│   └── local.yaml             # Machine-specific data paths (gitignored)
│
├── data/
│   ├── modalities.py          # ZStackModalDataset, FocusedModalDataset, HybridModalDataset
│   ├── boxes.py               # load_boxes_txt(): parse x1,y1,x2,y2 label files
│   ├── combined.py            # TissueChipDataset: joins all modalities, filters by intersection
│   ├── collate.py             # tissue_chip_collate(), pretrain_collate()
│   ├── pretrain_dataset.py    # TissueChipPretrainDataset: resize/crop for MAE input
│   ├── detection_dataset.py   # TissueChipDetectionDataset: COCO-style targets for DETR
│   └── cache.py               # CachedTissueChipDataset: per-sample .pt disk cache
│
├── models/
│   ├── pos_embed.py           # 2D and 3D sinusoidal positional embeddings
│   ├── stiffness.py           # StiffnessMLP: kPa scalar → embedding vector
│   ├── mae.py                 # MaskedAutoencoderViT + MAEViTEncoder (2D)
│   ├── mae_volume.py          # VolumeTubeEmbed + MaskedAutoencoderVolume (3D)
│   ├── multi_mae.py           # MultiEncoderMAE: combines all three encoders
│   └── weight_loaders.py      # Utilities to transfer MAE weights to detection backbone
│
├── training/
│   ├── lr_sched.py            # Linear warmup + cosine decay schedule
│   ├── engine_pretrain.py     # train_one_epoch() for MAE pretraining
│   ├── engine_detect.py       # train_one_epoch_detect() + eval_one_epoch_detect()
│   ├── main_pretrain.py       # run_pretrain(): full loop with AdamW + checkpointing
│   └── main_detect.py         # run_detect(): Deformable-DETR fine-tuning loop
│
├── scripts/
│   ├── train_pretrain.py      # CLI: --smoke | --inspect-data | --train | --wandb
│   └── train_detect.py        # CLI: --train | --wandb | --wandb-run-name
│
├── utils/
│   ├── checkpoint.py          # save_checkpoint(), load_checkpoint()
│   ├── wandb_utils.py         # W&B wrapper (all no-ops if wandb.enabled: false)
│   └── logging_utils.py       # General logging helpers
│
├── tests/
│   └── test_pipeline.py       # 20 integration tests using synthetic data
│
├── colab_train.ipynb          # Google Colab notebook for GPU training
├── requirements.txt
├── .gitignore
└── archive/                   # Legacy code (read-only reference)
    ├── legacy_colab_dataloaders/   # Original Colab-based data loading scripts
    └── legacy_facebook_mae/        # Original Meta MAE codebase (CC-BY-NC-4.0)
```

---

## 7. Setup and Running

### Prerequisites

- Python 3.9+
- For GPU training: CUDA 11.8 or 12.x with a compatible PyTorch wheel

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd Huang-Lab-Work

# 2. Install PyTorch for your CUDA version
# See https://pytorch.org/get-started/locally/ for the right command.
# Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install all other dependencies
pip install -r requirements.txt
```

### Quick Sanity Check (No Data Needed)

```bash
# Runs the full model forward/backward pass on random tensors.
# Should complete in ~10 seconds on CPU.
python scripts/train_pretrain.py --smoke
```

### Running the Tests

```bash
# All 20 integration tests, uses synthetic data, no GPU needed (~10 sec)
pytest tests/ -v

# Run a single test
pytest tests/test_pipeline.py::test_multi_encoder_forward -v
```

### Configuring Data Paths

Copy the example local config and fill in your paths:

```bash
cp config/local.yaml.example config/local.yaml
# Edit config/local.yaml with your actual data directory paths
```

### Verifying Data Loads Correctly

```bash
python scripts/train_pretrain.py --inspect-data --local-config config/local.yaml
```

This loads and prints the shape of one real batch without running any training. Useful for catching path or format issues before a long training run.

### Pretraining

```bash
# CPU (small subset, for development)
python scripts/train_pretrain.py \
    --config config/default.yaml \
    --local-config config/local.yaml \
    --train

# With W&B logging
python scripts/train_pretrain.py \
    --config config/default.yaml \
    --local-config config/local.yaml \
    --train \
    --wandb

# Resume from a checkpoint
python scripts/train_pretrain.py \
    --config config/default.yaml \
    --local-config config/local.yaml \
    --train \
    --resume outputs/multimae_epoch_50.pth
```

All scripts must be run from the **repo root**.

### Detection Fine-tuning

```bash
python scripts/train_detect.py \
    --config config/default.yaml \
    --local-config config/local.yaml \
    --wandb \
    --wandb-run-name "run-001" \
    --wandb-project "huang-lab-tissue-chip"
```

The HuggingFace `SenseTime/deformable-detr` checkpoint (~160 MB) is downloaded automatically on first run.

### Google Colab (GPU Training)

Open `colab_train.ipynb` in Google Colab. The notebook:
1. Mounts Google Drive
2. Clones this repository
3. Writes a `local.yaml` config pointing to your Drive-hosted data
4. Runs the full pretraining and detection pipeline with GPU

---

## 8. Configuration

The config system uses two YAML files that are deep-merged at load time:

- `config/default.yaml` — committed to the repo; contains all default hyperparameters
- `config/local.yaml` — gitignored; contains machine-specific paths and overrides

Any key in `local.yaml` overrides the same key in `default.yaml`. Tilde paths (`~/path/to/data`) are expanded automatically.

### Example `local.yaml`

```yaml
dataset:
  well_count: 222
  cache_dir: "~/.huang_lab_cache"       # set to null to disable caching
  resize:
    zstack: [64, 64]                    # resize z-stack frames to 64×64 (CPU default)
    focused: [224, 224]
    hybrid: [224, 224]
  splits:
    - name: "900kpa"
      stiffness_kpa: 900.0
      zstack_root:  "~/path/to/900kPa/zstacks"
      focused_root: "~/path/to/900kPa/focused"
      hybrid_root:  "~/path/to/900kPa/hybrid"
      labels_root:  "~/path/to/900kPa/labels"
    - name: "5kpa"
      stiffness_kpa: 5.0
      zstack_root:  "~/path/to/5kPa/zstacks"
      focused_root: "~/path/to/5kPa/focused"
      hybrid_root:  "~/path/to/5kPa/hybrid"
      labels_root:  "~/path/to/5kPa/labels"

training:
  batch_size: 1
  amp: false          # set to true on GPU

wandb:
  enabled: true
  project: "huang-lab-tissue-chip"
  entity: "your-wandb-username"
```

### Key Hyperparameters Reference

**`dataset` section**

| Key | Default | Description |
|---|---|---|
| `well_count` | 222 | Maximum well index to scan per condition |
| `expected_z_slices` | 140 | Warns if a well has a different slice count |
| `cache_dir` | `null` | Path for `.pt` cache files; `null` = disabled |
| `resize` | all `null` | Optional `[H, W]` resize per modality |

**`multi_mae` section**

| Key | Default | Description |
|---|---|---|
| `image_size` | 224 | Input size for 2D MAEs (H = W) |
| `patch_xy_2d` | 16 | Patch size in pixels for 2D MAEs |
| `volume_z` | 140 | Number of z-slices in volume MAE |
| `volume_h` / `volume_w` | 224 | Spatial dimensions for volume MAE |
| `tublet_z` | 4 | Z-slices per tube (must divide `volume_z`) |
| `embed_dim` | 384 | Patch embedding dimension |
| `stiffness_hidden` | 64 | Hidden dimension of StiffnessMLP |
| `kpa_input_mode` | `"divide"` | kPa normalization: `divide` / `log1p` / `identity` |
| `mask_ratio_focused` | 0.75 | Masking ratio for focused encoder |
| `mask_ratio_hybrid` | 0.75 | Masking ratio for hybrid encoder |
| `mask_ratio_volume` | 0.75 | Masking ratio for volume encoder |
| `loss_weight_focused` | 1.0 | Weight for focused head in total loss |
| `loss_weight_hybrid` | 1.0 | Weight for hybrid head in total loss |
| `loss_weight_volume` | 1.0 | Weight for volume head in total loss |
| `lr_mult_focused` | 1.0 | LR multiplier for focused encoder |
| `lr_mult_hybrid` | 1.0 | LR multiplier for hybrid encoder |
| `lr_mult_volume` | 1.0 | LR multiplier for volume encoder |

**`training` section**

| Key | Default | Description |
|---|---|---|
| `lr` | 1.5e-4 | Base (peak) learning rate |
| `warmup_epochs` | 10 | Number of linear warmup epochs |
| `epochs` | 100 | Total pretraining epochs |
| `batch_size` | 2 | Samples per batch |
| `amp` | `true` | Mixed precision training (CUDA only) |
| `grad_clip` | `null` | Max gradient norm; `null` = no clipping |

**`detection` section**

| Key | Default | Description |
|---|---|---|
| `hf_model` | `SenseTime/deformable-detr` | HuggingFace model ID |
| `num_classes` | 2 | Number of classes (including background) |
| `train_image_short_side` | 800 | Resize short side of images to this value |
| `image_mode` | `"focused"` | Which modality to feed to DETR: `focused` or `hybrid` |
| `mae_encoder_ckpt` | `null` | Path to multi-MAE checkpoint for backbone init (future) |
| `epochs` | 50 | Detection fine-tuning epochs |

---

## 9. Results So Far

The pipeline infrastructure is complete and all integration tests pass. The following are the current states of each component:

**Pretraining**
- Full multi-encoder MAE forward and backward pass verified on synthetic data
- Stiffness conditioning confirmed to produce different encoder outputs for 5 kPa vs 900 kPa inputs
- Disk cache yields ~10x speedup on repeated epochs compared to raw TIFF loading
- Learning rate warmup and cosine decay verified in tests

**Detection**
- Deformable-DETR fine-tuning loop runs end-to-end
- Detection metrics (mAP, AP50, AP75, mean IoU) computed after each epoch
- Bounding box format conversion (xyxy → normalized cxcywh) verified in tests

**Not Yet Completed**
- MAE encoder → Deformable-DETR backbone transfer: the pretrained `MAEViTEncoder` is not yet wired as the backbone in the detection model. The current detection pipeline uses the stock ResNet-50 backbone from the `SenseTime/deformable-detr` checkpoint. The weight transfer utilities exist in `models/weight_loaders.py` and the config hook (`detection.mae_encoder_ckpt`) is in place, but connecting the ViT output to the DETR feature pyramid neck is outstanding work.
- Full-dataset training numbers are not yet available. The pipeline is ready to run on the full 444-sample dataset on GPU; final mAP scores will be recorded here after training.

---

## 10. Acknowledgements

**MAE (Masked Autoencoders Are Scalable Vision Learners)**
He, K., Chen, X., Xie, S., Li, Y., Dollar, P., & Girshick, R. (2021).
The original Facebook MAE codebase is archived under `archive/legacy_facebook_mae/` and is licensed CC-BY-NC-4.0 by Meta Platforms, Inc. The `models/mae.py` and `models/pos_embed.py` files in this repo are derived from that work.

**VideoMAE (VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training)**
Tong, Z., Song, Y., Wang, J., & Wang, L. (2022).
Inspiration for the tubelet-based 3D masking strategy used in `models/mae_volume.py`.

**Deformable DETR (Deformable DETR: Deformable Transformers for End-to-End Object Detection)**
Zhu, X., Su, W., Lu, L., Li, B., Wang, X., & Dai, J. (2020).
Detection backbone provided via the HuggingFace `SenseTime/deformable-detr` checkpoint.

**timm (PyTorch Image Models)**
Wightman, R. (2019). https://github.com/huggingface/pytorch-image-models
Used for `PatchEmbed`, `Block`, and related ViT components.

**Huang Lab, University of [Institution]**
This pipeline was developed in collaboration with the Huang Lab for cardiovascular tissue-on-a-chip research.
