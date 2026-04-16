# Huang Lab — Tissue-Chip Vision Pipeline

End-to-end ML pipeline for cardiovascular tissue-on-a-chip imaging:
**multi-encoder MAE pretraining → Deformable-DETR bounding-box detection**.

---

## Table of contents

1. [Project background](#1-project-background)
2. [Repository layout](#2-repository-layout)
3. [Quick start](#3-quick-start)
4. [Data format](#4-data-format)
5. [Architecture](#5-architecture)
   - [Stiffness conditioning](#stiffness-conditioning)
   - [2D MAE — focused & hybrid](#2d-mae--focused--hybrid)
   - [3D volume MAE — z-stack](#3d-volume-mae--z-stack)
   - [Multi-encoder wrapper](#multi-encoder-wrapper)
   - [Detection (Deformable-DETR)](#detection-deformable-detr)
6. [Configuration reference](#6-configuration-reference)
7. [Training](#7-training)
   - [Pretraining](#pretraining)
   - [Detection fine-tuning](#detection-fine-tuning)
8. [Caching & I/O](#8-caching--io)
9. [Tests](#9-tests)
10. [Known gaps & next steps](#10-known-gaps--next-steps)
11. [Dependencies](#11-dependencies)
12. [Repository audit](#12-repository-audit)
13. [License](#13-license)

---

## 1. Project background

Each experiment well produces four types of data:

| Modality | Shape | Notes |
|---|---|---|
| **Z-stack** | `[140, C, H, W]` | 140 confocal z-slices per well |
| **Focused** | `[C, H, W]` | Single focus-stacked composite image |
| **Hybrid** | `[C, H, W]` | Computationally fused image |
| **Stiffness** | `[1]` (scalar) | Substrate stiffness in kPa (5 or 900) |

Each well also has **bounding-box labels** (`W###.txt`) for cell detection.

The goal is to pretrain modality-specific encoders via masked autoencoding — learning rich representations from unlabelled structure — then fine-tune a Deformable-DETR detector on the labelled boxes.

---

## 2. Repository layout

```
├── config/
│   ├── __init__.py            # load_config() merging default + local YAML
│   ├── settings.py            # dataclasses: FullConfig, MultiMAEConfig, DetectionConfig, …
│   ├── default.yaml           # canonical defaults (committed)
│   └── local.yaml.example     # template — copy to local.yaml and fill in paths
│
├── data/
│   ├── modalities.py          # ZStackModalDataset, FocusedModalDataset, HybridModalDataset
│   ├── boxes.py               # load_boxes_txt() — x1,y1,x2,y2 per line
│   ├── combined.py            # TissueChipDataset — intersection-filtered well list
│   ├── collate.py             # tissue_chip_collate, pretrain_collate
│   ├── pretrain_dataset.py    # TissueChipPretrainDataset — resize/crop to model dims
│   ├── detection_dataset.py   # TissueChipDetectionDataset — COCO-style targets
│   └── cache.py               # CachedTissueChipDataset — per-sample .pt disk cache
│
├── models/
│   ├── pos_embed.py           # 2D & 3D sin-cos positional embeddings
│   ├── stiffness.py           # StiffnessMLP, normalize_kpa()
│   ├── mae.py                 # MAEViTEncoder, MaskedAutoencoderViT (2D)
│   ├── mae_volume.py          # VolumeTubeEmbed, MaskedAutoencoderVolume (3D)
│   ├── multi_mae.py           # MultiEncoderMAE, build_multi_encoder_mae()
│   └── weight_loaders.py      # Helpers for transferring MAE weights to detection backbone
│
├── training/
│   ├── lr_sched.py            # Linear warmup + cosine decay
│   ├── engine_pretrain.py     # train_one_epoch() — AMP, grad clip, per-head losses
│   ├── main_pretrain.py       # run_pretrain() — full loop with AdamW + checkpointing
│   ├── engine_detect.py       # train_one_epoch_detect()
│   └── main_detect.py         # run_detect() — HuggingFace Deformable-DETR loop
│
├── scripts/
│   ├── train_pretrain.py      # CLI: --smoke | --inspect-data | --train
│   └── train_detect.py        # CLI: --train
│
├── utils/
│   ├── checkpoint.py          # save_checkpoint(), load_checkpoint()
│   └── logging_utils.py
│
├── tests/
│   └── test_pipeline.py       # Integration tests (no real data needed)
│
├── requirements.txt
├── .gitignore
└── archive/                   # Legacy code (read-only reference)
    ├── README.md
    ├── legacy_colab_dataloaders/   # Old Colab notebooks with Drive paths
    └── legacy_facebook_mae/        # Original Meta MAE codebase
```

---

## 3. Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Sanity-check the pipeline on random tensors (no data needed, ~10 sec)
python scripts/train_pretrain.py --smoke

# 3. Run tests
python -m pytest tests/ -v

# 4. Configure your data paths
cp config/local.yaml.example config/local.yaml
#    → edit config/local.yaml: fill in zstack_root, focused_root, hybrid_root, labels_root

# 5. Verify a real batch loads correctly
python scripts/train_pretrain.py --inspect-data --local-config config/local.yaml

# 6. Run pretraining
python scripts/train_pretrain.py --train --local-config config/local.yaml

# 7. Run detection fine-tuning
python scripts/train_detect.py --local-config config/local.yaml
```

All scripts must be run from the **repo root** — they add it to `sys.path` automatically.

---

## 4. Data format

### Directory layout

The loader expects the following structure for each stiffness condition (split):

```
<zstack_root>/
    W001/P00001/*.tif          # 140 z-slice images
    W002/P00001/*.tif
    ...

<focused_root>/
    W001/<name>_focus_stacked.tif   # one focused image per well
    ...  (discovered by glob + W### regex)

<hybrid_root>/
    hybrid_results_W001/*.tif      # one hybrid image per well folder
    ...

<labels_root>/
    W001.txt                   # one bounding box per line: x1,y1,x2,y2
    W002.txt
    ...
```

### Well discovery and alignment

`data/combined.py` builds the well list by **intersection**: a well is included only if it has a z-stack directory with `.tif` files, a hybrid folder with `.tif` files, a focus image matching `W###` in its path, and a label `.txt` file. This automatically handles the ~220 focus images vs 222 z/hybrid wells discrepancy without manual alignment.

### Bounding box format

Label files (`W###.txt`) contain one box per line with four comma-separated floats:

```
x1,y1,x2,y2
10.0,20.0,50.0,60.0
```

Assumed to be **pixel coordinates in xyxy format**. The detection dataset converts these to normalized cxcywh (DETR convention) at load time. If your labels use a different format (e.g. YOLO xywh normalized, class-prefixed), update `data/boxes.py` and `data/detection_dataset.py`.

---

## 5. Architecture

### Stiffness conditioning

`models/stiffness.py` — `StiffnessMLP`

The substrate stiffness (kPa) is a single scalar per well. It is embedded and injected into all three encoders as a **patch-level additive bias**:

```
kpa (scalar) → normalize → MLP(1 → 64 → embed_dim) → stiff_emb [B, D]
                                                              ↓
patch_tokens [B, L, D]  +  stiff_emb.unsqueeze(1)  →  conditioned tokens
```

This happens **after** spatial positional encoding and **before** masking, so stiffness information propagates through all transformer layers via attention. One `StiffnessMLP` is shared across all three encoders.

**Normalization modes** (`kpa_input_mode` in config):
- `"divide"` (default): `kpa / 900.0` → 5 kPa becomes 0.0056, 900 kPa becomes 1.0
- `"log1p"`: `log(1 + kpa)`
- `"identity"`: raw kPa value (not recommended — scale mismatch)

### 2D MAE — focused & hybrid

`models/mae.py` — `MaskedAutoencoderViT` / `MAEViTEncoder`

Standard Vision Transformer MAE applied to single 2D images (`[B, C, H, W]`).

**Encoder** (`MAEViTEncoder`):
1. Patch embedding via `timm.layers.PatchEmbed` (Conv2d projection)
2. 2D sin-cos positional encoding (fixed, not learned)
3. Stiffness injection (additive per-sample bias to all patch tokens)
4. Random masking — removes `mask_ratio` fraction of patches
5. CLS token prepended
6. ViT transformer blocks

**Decoder** (in `MaskedAutoencoderViT`):
1. Linear projection to `decoder_embed_dim`
2. Masked tokens (learnable) re-inserted at their original positions
3. Decoder pos embed added
4. Lightweight ViT blocks
5. Linear prediction head → pixel patch reconstruction
6. MSE loss on masked patches only (optionally normalized per-patch)

`MAEViTEncoder.forward_dense()` runs the encoder **without masking**, returning the full patch sequence. This is intended for extracting features for the detection backbone (future work).

Two separate `MaskedAutoencoderViT` instances are used — one for focused images, one for hybrid — with **independent weights** but identical hyperparameters.

### 3D volume MAE — z-stack

`models/mae_volume.py` — `MaskedAutoencoderVolume`

VideoMAE-style encoder operating on the full z-stack as a 3D volume `[B, C, Z, H, W]`.

**Tube embedding** (`VolumeTubeEmbed`): a single `Conv3d` with kernel `(tublet_z, patch_xy, patch_xy)` and matching stride. With the defaults `tublet_z=4`, `patch_xy=16`, `Z=140`, `H=W=64`:
- Along Z: 140 / 4 = **35 tubes**
- Spatially: 64 / 16 = **4 × 4 = 16 positions**
- Total tokens: **35 × 4 × 4 = 560 per sample**

**Positional encoding**: flat 1D sin-cos embedding over the lexicographic index of all tubes (Z × H × W order). This is simpler than factored 3D embeddings and works for any even `embed_dim`.

Everything else (masking, decoder, loss) mirrors the 2D MAE.

### Multi-encoder wrapper

`models/multi_mae.py` — `MultiEncoderMAE`

```
batch["focused"]   → MaskedAutoencoderViT (focused) → loss_f
batch["hybrid"]    → MaskedAutoencoderViT (hybrid)  → loss_h
batch["zstack"]    → MaskedAutoencoderVolume         → loss_z
batch["stiffness"] → StiffnessMLP → stiff_emb  (shared across all three)

total_loss = w_f * loss_f + w_h * loss_h + w_z * loss_z
```

**Separate learning rates per head**: `param_groups_with_head_lrs()` returns four optimizer parameter groups (focused, hybrid, volume, stiffness MLP), each with an independent LR multiplier set in config (`multi_mae.lr_mult_*`). This allows, for example, the volume encoder to train slower if gradients are noisier.

### Detection (Deformable-DETR)

`training/main_detect.py` / `training/engine_detect.py`

Fine-tuning uses HuggingFace `transformers.AutoModelForObjectDetection` with the `SenseTime/deformable-detr` checkpoint (ResNet-50 backbone). The detection dataset:

- Uses the **focused** image by default (`detection.image_mode`; switch to `"hybrid"` in config)
- Resizes by **short side** to `train_image_short_side` (default 800)
- Converts `xyxy` pixel boxes → **normalized `cxcywh`** (DETR convention, values in [0,1])
- Pads images within a batch to a common size with a boolean `pixel_mask`

**MAE encoder → detection backbone (future work)**: `detection.mae_encoder_ckpt` is reserved for when you replace the ResNet backbone with `MAEViTEncoder`. Use `models/weight_loaders.load_mae_encoder_from_multimae_ckpt()` to strip the `mae_focused.encoder.` prefix from a multi-MAE checkpoint and load it into your ViT backbone module. The stock checkpoint currently uses ResNet and this wiring is not yet built.

---

## 6. Configuration reference

All settings live in `config/default.yaml`. Machine-specific paths go in `config/local.yaml` (gitignored). `load_config(default, local)` deep-merges local on top of default.

### Key sections

**`dataset`**
| Key | Default | Description |
|---|---|---|
| `well_count` | 222 | Maximum well index to scan |
| `expected_z_slices` | 140 | Warn if a well has a different count |
| `focus_filename_glob` | `*focus_stacked*.tif` | Glob pattern for focus images |
| `hybrid_folder_template` | `hybrid_results_{well_id}` | Folder name template |
| `cache_dir` | `null` | Path for per-sample `.pt` cache; `null` = no cache |
| `resize` | all `null` | Optional `[H, W]` resize per modality |
| `splits` | (empty) | List of `{name, stiffness_kpa, zstack_root, focused_root, hybrid_root, labels_root}` |

**`multi_mae`**
| Key | Default | Description |
|---|---|---|
| `image_size` | 224 | H=W for 2D MAE inputs |
| `patch_xy_2d` | 16 | 2D patch size |
| `volume_z/h/w` | 140/224/224 | Volume spatial dims fed to 3D MAE |
| `tublet_z` | 4 | Z-slices per tube (must divide `volume_z`) |
| `embed_dim` | 384 | Shared encoder embedding dimension |
| `stiffness_hidden` | 64 | Hidden dim of StiffnessMLP |
| `kpa_input_mode` | `"divide"` | Normalization: `divide` / `log1p` / `identity` |
| `loss_weight_*` | 1.0 | Per-head loss weights |
| `mask_ratio_*` | 0.75 | Per-head masking ratios |
| `lr_mult_*` | 1.0 | Per-head LR multipliers |

**`training`**
| Key | Default | Description |
|---|---|---|
| `lr` | 1.5e-4 | Base learning rate (peak after warmup) |
| `warmup_epochs` | 10 | Linear warmup length |
| `epochs` | 100 | Total training epochs |
| `amp` | `true` | Mixed precision (CUDA only) |
| `grad_clip` | `null` | Max gradient norm (`null` = disabled) |
| `batch_size` | 2 | Per-GPU batch size |

**`detection`**
| Key | Default | Description |
|---|---|---|
| `hf_model` | `SenseTime/deformable-detr` | HuggingFace model ID |
| `num_classes` | 2 | Number of detection classes (including background) |
| `train_image_short_side` | 800 | Resize images to this short side |
| `image_mode` | `"focused"` | Which modality to use: `focused` or `hybrid` |
| `mae_encoder_ckpt` | `null` | Path to multi-MAE checkpoint for backbone init |

---

## 7. Training

### Pretraining

```bash
# Quick sanity check — random 224×224 tensors, no data needed
python scripts/train_pretrain.py --smoke

# Inspect one real batch (shapes, no training)
python scripts/train_pretrain.py --inspect-data --local-config config/local.yaml

# Full pretrain
python scripts/train_pretrain.py --train --local-config config/local.yaml \
    [--config config/default.yaml] [--resume outputs/multimae_epoch_50.pth] [--device cuda]
```

**What the loop does:**
1. Builds `MultiEncoderMAE` from `multi_mae` config
2. Creates four AdamW param groups (one per component) with per-head LR multipliers
3. Applies linear warmup + cosine decay LR schedule
4. Each step: forward all three MAEs conditioned on stiffness → weighted loss sum → backward → optional grad clip → optimizer step
5. Logs per-head losses to `outputs/pretrain_log.jsonl` every epoch
6. Saves checkpoints every 10% of total epochs (and always at the final epoch)

### Detection fine-tuning

```bash
python scripts/train_detect.py --local-config config/local.yaml \
    [--config config/default.yaml] [--device cuda]
```

Downloads `SenseTime/deformable-detr` from HuggingFace on first run (~160 MB). Trains for `detection.epochs` epochs with AdamW.

---

## 8. Caching & I/O

Loading 140 TIFF slices per well per batch is slow. Set `dataset.cache_dir` to a fast local disk:

```yaml
# config/local.yaml
dataset:
  cache_dir: /scratch/my_project/cache
```

`CachedTissueChipDataset` writes each processed sample as a `.pt` file keyed by `sha256(split+well_id)[:24]` on first access, then loads from disk on subsequent accesses. This avoids the full-dataset numpy export (which filled the disk in earlier iterations) while still eliminating repeated TIFF decoding.

---

## 9. Tests

Tests live in `tests/test_pipeline.py` and use a **synthetic 3-well dataset** (64×64 images, tiny model dimensions). No real data or GPU required.

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test
python -m pytest tests/test_pipeline.py::test_multi_encoder_forward -v
```

**Test coverage:**

| Test | What it verifies |
|---|---|
| `test_config_loads` | Config merging, field values |
| `test_dataset_length_and_keys` | Well intersection, sample dict keys |
| `test_dataset_tensor_shapes` | Correct shapes for all modalities |
| `test_pretrain_collate_shapes` | zstack permuted to `[B,C,Z,H,W]` |
| `test_stiffness_values` | Only 5.0 or 900.0 kPa present |
| `test_boxes_load` / `test_boxes_empty` | Box loading, empty-file edge case |
| `test_multi_encoder_forward` | Loss > 0, all three heads return values |
| `test_multi_encoder_backward` | Gradients flow to all parameters |
| `test_stiffness_conditions_output` | Different kPa → different model output |
| `test_smoke_script` | `--smoke` script exits 0 |
| `test_checkpoint_save_load` | Weights identical after save/reload |
| `test_run_pretrain_produces_checkpoints` | Full loop writes `.pth` + JSONL |
| `test_run_pretrain_resume` | Resume from checkpoint doesn't raise |
| `test_detection_dataset_cxcywh` | Boxes normalized, in [0, 1] |
| `test_detection_collate_pixel_mask` | Batch has `pixel_mask` bool tensor |
| `test_disk_cache` | Cache file written; tensors match on reload |
| `test_lr_warmup_increases` | LR strictly increases during warmup |
| `test_lr_cosine_decays` | LR decreases after warmup peak |
| `test_lr_reaches_min` | LR never goes below `min_lr` |

---

## 10. Known gaps & next steps

### MAE encoder → Deformable-DETR backbone (not yet wired)

The current detection pipeline uses Deformable-DETR's **stock ResNet-50 backbone**. The intended flow is:

1. Take a pretrained `MultiEncoderMAE` checkpoint
2. Extract `mae_focused.encoder.*` weights using `models/weight_loaders.load_mae_encoder_from_multimae_ckpt()`
3. Build a custom `nn.Module` that wraps `MAEViTEncoder.forward_dense()` and produces FPN-style multi-scale feature maps
4. Swap it in as the backbone for Deformable-DETR

This requires matching channel dimensions and strides between the ViT and the DETR neck. `detection.mae_encoder_ckpt` in config is the hook for step 2 — currently emits a warning if set.

### Confirm bounding box format

`data/boxes.py` assumes `x1,y1,x2,y2` in pixel coordinates. Verify against the actual label files before running detection training. If the format differs (YOLO normalized xywh, class-prefixed, etc.), update `load_boxes_txt()` and the scaling logic in `TissueChipDetectionDataset.__getitem__()`.

### Multi-GPU / distributed training

The training loop is single-process. For multi-GPU, wrap `MultiEncoderMAE` in `torch.nn.parallel.DistributedDataParallel` and add a `DistributedSampler` to the DataLoader. The param groups in `param_groups_with_head_lrs()` are compatible with DDP.

---

## 11. Dependencies

```
torch >= 2.1, < 2.7
torchvision >= 0.16, < 0.22
timm >= 1.0.12, < 2.0        # PatchEmbed/Block from timm.layers (1.x API)
transformers >= 4.36, < 5.0  # Deformable-DETR via AutoModelForObjectDetection
accelerate >= 0.25
PyYAML >= 6.0.1
Pillow >= 10.0
numpy >= 1.24, < 3.0
```

Install the `torch` wheel for your CUDA version from [pytorch.org](https://pytorch.org) if needed, then `pip install -r requirements.txt`.

---

## 12. Repository audit

### Active code (source of truth)

Everything in `config/`, `data/`, `models/`, `training/`, `scripts/`, `utils/`, `tests/`, `requirements.txt`, `.gitignore`, `README.md`.

### Archived legacy code

| Path | What it was | Superseded by |
|---|---|---|
| `archive/legacy_colab_dataloaders/disk_cached_dataloader.py` | Colab disk-cache dataloader | `data/cache.py` + `data/combined.py` |
| `archive/legacy_colab_dataloaders/working_zstack+focused+hybrid_dataloader.py` | Working Colab dataloader | `data/modalities.py` + `data/combined.py` |
| `archive/legacy_colab_dataloaders/ten_sample_disk_cached_dataloader_working.py` | 10-sample test version | `tests/test_pipeline.py` |
| `archive/legacy_colab_dataloaders/Dataloaders.md` | Notes on caching issues | `data/cache.py` + this README §8 |
| `archive/legacy_facebook_mae/models_mae.py` | Original single-image MAE | `models/mae.py` |
| `archive/legacy_facebook_mae/models_vit.py` | ViT classification backbone | `models/mae.py` (`MAEViTEncoder`) |
| `archive/legacy_facebook_mae/main_pretrain.py` | ImageFolder-based pretrain | `training/main_pretrain.py` |
| `archive/legacy_facebook_mae/main_finetune.py` | Image **classification** finetune | `training/main_detect.py` |
| `archive/legacy_facebook_mae/engine_*.py` | Training engines | `training/engine_pretrain.py`, `training/engine_detect.py` |
| `archive/legacy_facebook_mae/util/` | MAE utilities | `models/pos_embed.py`, `utils/checkpoint.py` |
| `archive/legacy_facebook_mae/submitit_*.py` | Slurm launchers | Not yet ported |

---

## 13. License

The original MAE code (archived under `archive/legacy_facebook_mae/`) is licensed **CC-BY-NC-4.0** by Meta Platforms, Inc. New project files (`data/`, `models/`, `training/`, etc.) follow the same academic non-commercial pattern unless your lab specifies otherwise.
