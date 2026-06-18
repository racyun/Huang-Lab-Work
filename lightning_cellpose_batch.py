#!/usr/bin/env python3
"""Cellpose-SAM batch segmentation (Lightning AI version).

Runs Cellpose-SAM on ~6000 cell images stored in Google Drive and saves masks
back to Drive. This is the script form of `lightning_colab_cellpose.ipynb`.

How this differs from the Colab version:
  - Uses `rclone` (not `google.colab.drive`) to access Google Drive.
  - Per-folder workflow: sync images down to Studio disk -> run Cellpose ->
    sync masks back up to Drive.
  - Paths are Lightning-local (`/teamspace/studios/this_studio/...`).

One-time setup (run in the Lightning terminal, NOT here):
  1. Install rclone:    curl https://rclone.org/install.sh | sudo bash
  2. Configure a Google Drive remote named exactly `gdrive`:    rclone config
       n -> new remote; name: gdrive; storage: drive; client_id/secret blank;
       scope: 1 (full access); service_account_file blank; advanced: n;
       auto config: n  (IMPORTANT, no browser on Lightning) -- open the URL it
       prints on your laptop and paste the auth code back; team drive: n; y, q.
  3. Verify:   rclone lsd gdrive:

Dependencies (install once, into the Studio's environment):
    pip install "numpy<2.0" cellpose tifffile tqdm

Resumable: if the Studio disconnects, just re-run. Already-processed masks are
skipped (SKIP_EXISTING) and only un-uploaded masks get pushed (rclone copy).

Usage:
    python lightning_cellpose_batch.py
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# Path inside your Google Drive (relative to gdrive: remote root, i.e. 'My Drive')
DRIVE_ROOT = 'Fusion AI/Prof Huang Project/Cellpose feature extractions'

# Local disk path where images are cached during processing.
# Defaults to the Lightning Studio path; override with the CELLPOSE_LOCAL_ROOT
# env var (e.g. on a Mac:  export CELLPOSE_LOCAL_ROOT=~/cellpose_work ).
# Falls back to ~/cellpose_work automatically if the Lightning path is absent.
_default_root = Path('/teamspace/studios/this_studio/cellpose_work')
if 'CELLPOSE_LOCAL_ROOT' in os.environ:
    LOCAL_ROOT = Path(os.environ['CELLPOSE_LOCAL_ROOT']).expanduser()
elif _default_root.parent.parent.exists():  # /teamspace/studios exists -> on Lightning
    LOCAL_ROOT = _default_root
else:
    LOCAL_ROOT = Path.home() / 'cellpose_work'

# Stiffness conditions — one per folder
CONDITIONS = [
    '260513_TC_Level',
    '260514_900kPa',
    '260516_5kPa',
    '260516_500kPa',
    '260521_150kPa',
    '260522_500kPa',
]

MODEL_TYPE    = 'cpsam'  # Cellpose-SAM
DIAMETER      = None     # None = auto-estimate per image
CHANNELS      = [0, 0]   # [0,0] = use all channels for SAM
SKIP_EXISTING = True     # set False to reprocess everything from scratch
BATCH_SIZE    = 8        # images processed per model.eval() call
                         # increase if you have more RAM/GPU; decrease on OOM
IMG_EXTS      = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}


# --------------------------------------------------------------------------- #
# rclone sync helpers (Drive <-> Studio disk)
# --------------------------------------------------------------------------- #

def verify_rclone() -> None:
    """Confirm rclone is configured with a `gdrive:` remote, else raise."""
    result = subprocess.run(['rclone', 'lsd', 'gdrive:'], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'rclone not configured. Error:\n{result.stderr}')
    print('rclone OK. Top-level Drive folders:')
    print(result.stdout)


def _run_rclone(args: list) -> None:
    """Run an rclone command, streaming output, raising on failure."""
    print(f'  $ rclone {" ".join(args)}')
    result = subprocess.run(['rclone', *args, '--progress', '--transfers=8', '--checkers=16'],
                            capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f'rclone failed (exit {result.returncode})')


def pull_images_from_drive(condition: str) -> Path:
    """Download `imgs/<condition>/tiles/` from Drive to local disk. Returns local path."""
    remote = f'gdrive:{DRIVE_ROOT}/imgs/{condition}/tiles'
    local  = LOCAL_ROOT / 'imgs' / condition / 'tiles'
    local.mkdir(parents=True, exist_ok=True)
    _run_rclone(['copy', remote, str(local)])
    return local


def push_masks_to_drive(condition: str) -> None:
    """Upload local masks for a condition back to Drive (only new/changed files)."""
    local  = LOCAL_ROOT / 'masks' / condition
    remote = f'gdrive:{DRIVE_ROOT}/masks/{condition}'
    _run_rclone(['copy', str(local), remote])


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #

def _load_image(img_path: Path) -> np.ndarray:
    """Load image and ensure shape is (H, W, 3)."""
    img = tifffile.imread(str(img_path))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[0] in (1, 3):  # (C,H,W) -> (H,W,C)
        img = np.moveaxis(img, 0, -1)
        if img.shape[-1] == 1:
            img = np.concatenate([img, img, img], axis=-1)
    return img


def segment_folder(in_dir: Path, out_dir: Path, model) -> dict:
    """Process all images in in_dir in batches, writing masks to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    all_paths = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not all_paths:
        print(f'  [WARN] No images found in {in_dir}')
        return {'processed': 0, 'skipped': 0, 'failed': 0}

    todo, skipped_paths = [], []
    for p in all_paths:
        out_path = out_dir / f'{p.stem}_masks.tif'
        if SKIP_EXISTING and out_path.exists():
            skipped_paths.append(p)
        else:
            todo.append(p)

    processed = failed = 0
    skipped = len(skipped_paths)
    failed_files = []

    chunks = [todo[i:i+BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    pbar = tqdm(total=len(todo), desc=in_dir.parent.name, unit='img')

    for chunk in chunks:
        imgs, paths = [], []
        for p in chunk:
            try:
                imgs.append(_load_image(p))
                paths.append(p)
            except Exception as e:
                print(f'  [FAIL load] {p.name}: {e}')
                failed_files.append(p.name)
                failed += 1
                pbar.update(1)

        if not imgs:
            continue

        try:
            masks_list, _, _ = model.eval(
                imgs,
                diameter=DIAMETER,
                channels=CHANNELS,
                normalize=True,
            )
        except Exception as e:
            print(f'  [FAIL batch] {[p.name for p in paths]}: {e}')
            failed += len(paths)
            failed_files += [p.name for p in paths]
            pbar.update(len(paths))
            continue

        for p, mask in zip(paths, masks_list):
            out_path = out_dir / f'{p.stem}_masks.tif'
            try:
                tifffile.imwrite(str(out_path), mask.astype(np.uint16), compression='lzw')
                processed += 1
            except Exception as e:
                print(f'  [FAIL save] {p.name}: {e}')
                failed += 1
                failed_files.append(p.name)
            pbar.update(1)

    pbar.close()
    if failed_files:
        print(f'  Failed files: {failed_files}')
    return {'processed': processed, 'skipped': skipped, 'failed': failed}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    import torch
    from cellpose import models

    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. Check runtime
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print(f'GPU detected: {torch.cuda.get_device_name(0)}')
        print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
        print('Expected: a few sec/image — all 6000 in well under an hour.')
    else:
        print('No GPU detected — running on CPU.')
        print('Expected: ~30-120 sec/image. For 6000 images, plan multiple sessions.')
        print('Tip: open a GPU Studio in Lightning (Settings -> Compute) for faster runs.')
        print('Skip-if-exists is enabled, so each new session resumes where the last left off.')

    # 2. Verify rclone
    verify_rclone()

    print(f'Local working dir: {LOCAL_ROOT}')
    print(f'Drive root        : gdrive:{DRIVE_ROOT}')
    print(f'Conditions        : {len(CONDITIONS)}')

    # 3. Load model once (reused across all images)
    model = models.CellposeModel(gpu=use_gpu, model_type=MODEL_TYPE)
    print(f'Model loaded: {MODEL_TYPE}  |  GPU={model.gpu}')

    # 4. Process each condition: pull -> segment -> push
    grand_total = {'processed': 0, 'skipped': 0, 'failed': 0}
    t_start = time.time()

    for condition in CONDITIONS:
        print(f'\n========== {condition} ==========')

        print('[1/3] Pulling images from Drive...')
        in_dir = pull_images_from_drive(condition)

        print('[2/3] Running Cellpose...')
        out_dir = LOCAL_ROOT / 'masks' / condition
        t0 = time.time()
        stats = segment_folder(in_dir, out_dir, model)
        elapsed = time.time() - t0
        print(f'  processed={stats["processed"]}  skipped={stats["skipped"]}  '
              f'failed={stats["failed"]}  ({elapsed/60:.1f} min)')

        print('[3/3] Pushing masks to Drive...')
        push_masks_to_drive(condition)

        for k in grand_total:
            grand_total[k] += stats[k]

    total_elapsed = time.time() - t_start
    print('\n========== DONE ==========')
    print(f'Total processed : {grand_total["processed"]}')
    print(f'Total skipped   : {grand_total["skipped"]}')
    print(f'Total failed    : {grand_total["failed"]}')
    print(f'Wall time       : {total_elapsed/60:.1f} min')


if __name__ == '__main__':
    main()
