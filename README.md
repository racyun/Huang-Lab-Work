# Huang Lab — tissue-chip vision (MAE → detection)

This repository is being refactored for **multi-encoder MAE pretraining** and **Deformable-DETR-style bounding-box detection** on cardiovascular tissue-on-chip data (z-stack, focused stack, hybrid, stiffness, and bbox labels).

## Layout (source of truth)

| Path | Role |
|------|------|
| `config/` | YAML defaults + dataclass schema; use `config/local.yaml` (gitignored) for machine-specific paths. |
| `data/` | `TissueChipDataset`, pretrain transforms, optional disk cache, detection dataset + COCO-style boxes. |
| `models/` | 2D MAE (`mae.py`), volume MAE (`mae_volume.py`), `MultiEncoderMAE` + stiffness MLP (`multi_mae.py`, `stiffness.py`). |
| `training/` | `main_pretrain.py` / `engine_pretrain.py`, `main_detect.py` / `engine_detect.py`, LR schedule. |
| `scripts/` | `train_pretrain.py` (`--smoke`, `--train`, `--inspect-data`), `train_detect.py`. |
| `utils/` | Checkpoint I/O, logging helpers. |
| `requirements.txt` | Pinned dependency ranges (see below). |
| `archive/` | **Legacy only** — old Colab dataloaders and full Meta MAE clone (see `archive/README.md`). |

Run scripts from the repo root (they add the root to `sys.path`):

```bash
pip install -r requirements.txt
# Random-tensor sanity check (multi-encoder + stiffness)
python scripts/train_pretrain.py --smoke
# Full pretrain (needs data paths in config/local.yaml)
python scripts/train_pretrain.py --train --local-config config/local.yaml
# One batch shapes from real data
python scripts/train_pretrain.py --inspect-data --local-config config/local.yaml
# Deformable-DETR fine-tune (HuggingFace; downloads SenseTime/deformable-detr by default)
python scripts/train_detect.py --local-config config/local.yaml
```

`--inspect-data` / `--train` need valid `dataset.splits[*]` roots. Set `dataset.cache_dir` to cache per-sample `.pt` dicts and limit repeated I/O.

## Model overview

- **Focused & hybrid**: standard 2D MAE (`MaskedAutoencoderViT`) on `224×224` (configurable).
- **Z-stack**: **VideoMAE-style** volume MAE (`MaskedAutoencoderVolume`) on `[C,Z,H,W]` with 3D tube embedding `(tublet_z, patch_xy, patch_xy)` and flat sin-cos positions over tube indices.
- **Stiffness**: shared `StiffnessMLP` maps normalized kPa `[B,1] → [B,D]` and the result is **added to every patch token** (after spatial pos embed, before masking) for all three encoders.
- **Pretrain loss**: `w_f·L_focus + w_h·L_hybrid + w_z·L_volume` with **separate optimizer param groups** and per-head LR multipliers (`multi_mae.lr_mult_*`).
- **Detection**: `training/main_detect.py` loads **`SenseTime/deformable-detr`** via `transformers` and trains on **normalized cxcywh** boxes. The default backbone is **ResNet**; `detection.mae_encoder_ckpt` is reserved for when you attach a **ViT encoder** — use `models/weight_loaders.load_mae_encoder_from_multimae_ckpt` on your backbone module.

### Indexing / focus count mismatch

Wells are indexed by **intersection**: z-stack + hybrid + label + a focus file whose path contains `W###`. Wells that appear in z/hybrid but lack a focus image are **skipped**, so counts can be below 222 without manual alignment.

## Science / data (brief)

Project context: AI/ML for synthetic cardiovascular tissue-on-chip images; targets include ECM stiffness and morphology. Per-sample inputs: **140 z-slices**, one **focused** image, one **hybrid** image, **stiffness (kPa)**, and **bounding-box labels** (comma-separated lines in `W###.txt` files). Dataset layout details were documented in earlier commits; see `archive/legacy_colab_dataloaders/` for Colab-era notes.

## Configuration

1. Copy `config/local.yaml.example` → `config/local.yaml`.
2. Set `dataset.splits[*].{zstack_root,focused_root,hybrid_root,labels_root}` to your drive or NFS paths (Colab, cluster, or laptop — only paths change).
3. Optional: adjust `dataset.resize` so each modality has consistent `H×W` within the batch (native resolution works if all images in a modality already match).

### Ambiguities you may need to resolve

- **Focus images**: discovery uses `dataset.focus_filename_glob` (default `*focus_stacked*.tif`) and regex `W\d{3}` in the **full path**. If your naming differs, change the glob or extend `discover_focus_paths` in `data/modalities.py`.
- **Hybrid folder template**: default `hybrid_results_{well_id}`; override `hybrid_folder_template` in YAML if your tree differs.
- **Z-slice count**: default expected count is 140; a warning is emitted on mismatch (non-fatal).
- **Bounding-box format**: one box per line, four comma-separated floats (assumed `x1,y1,x2,y2`). If labels use another convention, update `data/boxes.py` **after confirming with your lab**.

## Dependencies

Versions are chosen for **timm ≥ 1.0** (`PatchEmbed` / `Block` from `timm.layers` with fallbacks). A typical stack:

- `torch` / `torchvision`: install the wheel that matches your CUDA runtime from [pytorch.org](https://pytorch.org) if needed.
- `timm>=1.0.12`
- `transformers`, `accelerate` (detection)
- `PyYAML`, `Pillow`, `numpy`

## Repository audit (post-refactor)

### Active files (canonical)

- `README.md`, `requirements.txt`, `.gitignore`
- `config/default.yaml`, `config/local.yaml.example`, `config/__init__.py`, `config/settings.py`
- `data/*.py` (incl. `pretrain_dataset.py`, `detection_dataset.py`, `cache.py`)
- `models/mae.py`, `mae_volume.py`, `multi_mae.py`, `stiffness.py`, `weight_loaders.py`, `pos_embed.py`
- `training/*.py`
- `utils/__init__.py`, `utils/checkpoint.py`, `utils/logging_utils.py`
- `scripts/train_pretrain.py`, `scripts/train_detect.py`
- `archive/README.md` + everything under `archive/legacy_*` (frozen reference)

### Archived / duplicate legacy (not source of truth)

| Location | Notes |
|----------|--------|
| `archive/legacy_colab_dataloaders/` | Three overlapping Colab exports (`disk_cached_*`, `working_*`, `ten_sample_*`), Google Drive paths, `google.colab` imports. **Superseded by** `data/`. |
| `archive/legacy_facebook_mae/` | Full MAE repo: `main_finetune.py` (classification), `main_linprobe.py`, Slurm `submitit_*`, `util/datasets.py` (ImageFolder), `models_vit.py`, demo notebook, docs. **Encoder/MAE pieces reimplemented in** `models/`; training loop trimmed to `scripts/train_pretrain.py`. |

### Removed from “active” surface (by design)

- ImageNet `ImageFolder` pretrain/finetune drivers (remain in `archive/` only).
- ViT **classification** head (`models_vit.py` in archive) — superseded by HuggingFace detection fine-tuning + optional ViT backbone wiring.

## License

The original MAE code is CC-BY-NC (see `archive/legacy_facebook_mae/LICENSE`). New project files follow the same academic use pattern unless your lab specifies otherwise.
