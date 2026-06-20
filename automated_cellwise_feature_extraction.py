# -*- coding: utf-8 -*-
"""automated_cellwise_feature_extraction (Lightning AI version).

Per-cell feature extraction over a nested, per-condition Drive layout:

    Cellpose feature extractions/
    ├── imgs/<condition>/tiles/<name>.tif
    ├── masks/<condition>/<name>_masks.tif
    └── cellwise_metadata/<condition>/<name>_metadata.csv   (output)

Conditions are discovered automatically under imgs/ and processed one at a
time. The per-cell feature-extraction logic is unchanged from the Colab
version. This version runs on Lightning AI instead of Colab:
  - Uses `rclone` (not `google.colab.drive`) to access Google Drive.
  - Per-condition workflow: pull imgs+masks down to Studio disk -> extract
    features locally -> push the metadata CSVs back up to Drive.

One-time setup (run in the Lightning terminal): install rclone and configure
a Google Drive remote named exactly `gdrive` (see lightning_cellpose_batch.py
for the full rclone steps). Then:
    export RCLONE_CONFIG=~/rclone-config/rclone.conf   # if your config lives there

Dependencies:
    pip install numpy pandas scipy scikit-image tifffile

Usage:
    python automated_cellwise_feature_extraction.py

This is pure CPU work (numpy/scipy/skimage) — no GPU required.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import tifffile
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.measure import regionprops_table

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# Path inside your Google Drive (relative to gdrive: remote root, i.e. 'My Drive')
DRIVE_ROOT = 'Fusion AI/Prof Huang Project/Cellpose feature extractions'

# Local disk path where data is cached during processing.
# Defaults to the Lightning Studio path; override with CELLPOSE_LOCAL_ROOT
# (e.g. on a Mac:  export CELLPOSE_LOCAL_ROOT=~/cellpose_work ). Falls back to
# ~/cellpose_work automatically if the Lightning path is absent.
_default_root = Path('/teamspace/studios/this_studio/cellpose_work')
if 'CELLPOSE_LOCAL_ROOT' in os.environ:
    LOCAL_ROOT = Path(os.environ['CELLPOSE_LOCAL_ROOT']).expanduser()
elif _default_root.parent.parent.exists():  # /teamspace/studios exists -> on Lightning
    LOCAL_ROOT = _default_root
else:
    LOCAL_ROOT = Path.home() / 'cellpose_work'

# Stiffness conditions — leave empty to auto-discover from Drive (imgs/ listing)
CONDITIONS: list[str] = []

# Feature-extraction params (same as the single-image notebook)
BAND_WIDTH = 2
RESTRICT_OUTWARD_TO_BG = True
IMG_EXTENSIONS = {'.tif', '.tiff'}
SKIP_EXISTING = True   # skip pairs whose metadata CSV already exists locally


def process_image_mask_pair(image_path, mask_path, output_csv_path,
                            band_width=BAND_WIDTH,
                            restrict_outward_to_bg=RESTRICT_OUTWARD_TO_BG):
    """Run the existing feature-extraction pipeline on one (image, mask) pair
    and save the resulting per-cell dataframe as CSV. Returns the dataframe."""
    mask  = tifffile.imread(str(mask_path))
    image = tifffile.imread(str(image_path))

    if image.ndim == 2:
        image = image[..., None]

    cell_ids = np.unique(mask)
    cell_ids = cell_ids[cell_ids != 0]

    # Empty-mask short-circuit so we still write a (header-only) CSV
    if cell_ids.size == 0:
        df = pd.DataFrame(columns=[
            'area_px', 'elongation',
            'ch1_cellwise_mean_intensity',
            'ch2_cellwise_mean_membrane_intensity',
            'ch3_cellwise_mean_intensity',
            'centroid_x', 'centroid_y',
        ])
        df.index.name = 'cell_id'
        df.to_csv(output_csv_path)
        return df

    # ----- Channels 1 & 3 (axis indices 0 and 2): whole-cell mean -----
    whole_cell_channels = [0, 2]
    whole_cell_means = {
        c: ndimage.mean(image[..., c], labels=mask, index=cell_ids)
        for c in whole_cell_channels
    }

    # ----- Channel 2 (axis index 1, VE-cadherin): membrane-band mean -----
    ve_cad = image[..., 1]
    slices = ndimage.find_objects(mask)
    membrane_means = np.full(len(cell_ids), np.nan)
    for i, cid in enumerate(cell_ids):
        sl = slices[cid - 1]
        if sl is None:
            continue
        padded = tuple(
            slice(max(s.start - band_width, 0),
                  min(s.stop  + band_width, mask.shape[d]))
            for d, s in enumerate(sl)
        )
        sub_mask = mask[padded]
        cell     = (sub_mask == cid)
        dilated  = ndimage.binary_dilation(cell, iterations=band_width)
        eroded   = ndimage.binary_erosion(cell, iterations=band_width)
        band     = dilated & ~eroded
        if restrict_outward_to_bg:
            band &= (sub_mask == cid) | (sub_mask == 0)
        px = ve_cad[padded][band]
        if px.size:
            membrane_means[i] = px.mean()

    # ----- Per-cell pixel area -----
    areas = np.bincount(mask.ravel())[cell_ids]

    # ----- Per-cell elongation (major / minor axis of the equivalent ellipse) -----
    # ----- and centroid coordinates (needed for Stage 2 graph construction) -----
    props = pd.DataFrame(
        regionprops_table(
            mask,
            properties=('label', 'axis_major_length', 'axis_minor_length', 'centroid'),
        )
    ).set_index('label')
    elongation = np.where(
        props['axis_minor_length'] > 0,
        props['axis_major_length'] / props['axis_minor_length'],
        np.nan,
    )
    elongation = pd.Series(elongation, index=props.index).reindex(cell_ids).to_numpy()
    # regionprops names centroid columns 'centroid-0' (row/y) and 'centroid-1' (col/x)
    centroid_y = props['centroid-0'].reindex(cell_ids).to_numpy()
    centroid_x = props['centroid-1'].reindex(cell_ids).to_numpy()

    # ----- Combine and save -----
    df = pd.DataFrame(
        {
            'area_px':                              areas,
            'elongation':                           elongation,
            'ch1_cellwise_mean_intensity':          whole_cell_means[0],
            'ch2_cellwise_mean_membrane_intensity': membrane_means,
            'ch3_cellwise_mean_intensity':          whole_cell_means[2],
            'centroid_x':                           centroid_x,
            'centroid_y':                           centroid_y,
        },
        index=cell_ids,
    )
    df.index.name = 'cell_id'
    df.to_csv(output_csv_path)
    return df


# --------------------------------------------------------------------------- #
# rclone sync helpers (Drive <-> Studio disk)
# --------------------------------------------------------------------------- #

def verify_rclone() -> None:
    """Confirm rclone is configured with a `gdrive:` remote, else raise."""
    result = subprocess.run(['rclone', 'lsd', 'gdrive:'], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'rclone not configured. Error:\n{result.stderr}')
    print('rclone OK.')


def _run_rclone(args: list) -> None:
    print(f'  $ rclone {" ".join(args)}')
    result = subprocess.run(['rclone', *args, '--progress', '--transfers=8', '--checkers=16'],
                            capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(f'rclone failed (exit {result.returncode})')


def discover_conditions_from_drive() -> list[str]:
    """List condition folder names under Drive's imgs/ directory."""
    remote = f'gdrive:{DRIVE_ROOT}/imgs'
    result = subprocess.run(['rclone', 'lsf', '--dirs-only', remote],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f'rclone lsf failed for {remote}:\n{result.stderr}')
    # lsf prints one dir per line with a trailing slash
    return sorted(line.rstrip('/') for line in result.stdout.splitlines() if line.strip())


def pull_condition_inputs(condition: str) -> tuple[Path, Path]:
    """Download imgs/<condition>/tiles and masks/<condition> to local disk."""
    tiles_remote = f'gdrive:{DRIVE_ROOT}/imgs/{condition}/tiles'
    masks_remote = f'gdrive:{DRIVE_ROOT}/masks/{condition}'
    tiles_local  = LOCAL_ROOT / 'imgs' / condition / 'tiles'
    masks_local  = LOCAL_ROOT / 'masks' / condition
    tiles_local.mkdir(parents=True, exist_ok=True)
    masks_local.mkdir(parents=True, exist_ok=True)
    print('  pulling images...')
    _run_rclone(['copy', tiles_remote, str(tiles_local)])
    print('  pulling masks...')
    _run_rclone(['copy', masks_remote, str(masks_local)])
    return tiles_local, masks_local


def push_condition_outputs(condition: str) -> None:
    """Upload local metadata CSVs for a condition back to Drive."""
    local  = LOCAL_ROOT / 'cellwise_metadata' / condition
    remote = f'gdrive:{DRIVE_ROOT}/cellwise_metadata/{condition}'
    _run_rclone(['copy', str(local), remote])


# --------------------------------------------------------------------------- #
# Per-condition processing
# --------------------------------------------------------------------------- #

def process_condition(condition: str, tiles_dir: Path, mask_dir: Path) -> dict:
    """Process every image/mask pair for one condition. Returns a stats dict."""
    out_dir = LOCAL_ROOT / 'cellwise_metadata' / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        p for p in tiles_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    )
    total = len(image_files)
    print(f'  Found {total} image(s) in {tiles_dir}')

    processed = skipped = failed = 0
    done = 0

    def _report() -> None:
        pct = (done / total * 100.0) if total else 100.0
        print(f'  {condition}: {done}/{total} pairs ({pct:.1f}%)', flush=True)

    for image_path in image_files:
        mask_name = f'{image_path.stem}_masks{image_path.suffix}'
        mask_path = mask_dir / mask_name
        out_path  = out_dir / f'{image_path.stem}_metadata.csv'

        if SKIP_EXISTING and out_path.exists():
            skipped += 1
            done += 1
            if done % 10 == 0:
                _report()
            continue

        if not mask_path.exists():
            print(f'  [skip] no mask for {image_path.name} (expected {mask_name})')
            skipped += 1
            done += 1
            if done % 10 == 0:
                _report()
            continue

        try:
            process_image_mask_pair(image_path, mask_path, out_path)
            processed += 1
        except Exception as e:
            print(f'  [fail] {image_path.name}: {e}')
            failed += 1
        done += 1
        if done % 10 == 0:
            _report()

    if done % 10 != 0:
        _report()

    return {
        'condition': condition,
        'total':     total,
        'processed': processed,
        'skipped':   skipped,
        'failed':    failed,
    }


# --------------------------------------------------------------------------- #
# Main: pull -> extract -> push, condition-by-condition
# --------------------------------------------------------------------------- #

def main() -> None:
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    verify_rclone()

    conditions = CONDITIONS or discover_conditions_from_drive()
    print(f'Local working dir: {LOCAL_ROOT}')
    print(f'Drive root        : gdrive:{DRIVE_ROOT}')
    print(f'Discovered {len(conditions)} condition(s): {conditions}\n')

    all_stats = []
    t_start = time.time()

    for i, condition in enumerate(conditions, start=1):
        print(f'========== [{i}/{len(conditions)}] {condition} ==========')

        print('[1/3] Pulling inputs from Drive...')
        tiles_dir, mask_dir = pull_condition_inputs(condition)

        print('[2/3] Extracting features...')
        t0 = time.time()
        stats = process_condition(condition, tiles_dir, mask_dir)
        elapsed = time.time() - t0
        print(f'  -> processed={stats["processed"]}  skipped={stats["skipped"]}  '
              f'failed={stats["failed"]}  (total={stats["total"]}, {elapsed/60:.1f} min)')

        print('[3/3] Pushing metadata to Drive...')
        push_condition_outputs(condition)
        all_stats.append(stats)

    # ----- Grand summary across all conditions -----
    g_processed = sum(s['processed'] for s in all_stats)
    g_skipped   = sum(s['skipped']   for s in all_stats)
    g_failed    = sum(s['failed']    for s in all_stats)
    g_total     = sum(s['total']     for s in all_stats)
    total_elapsed = time.time() - t_start

    print('\n========== SUMMARY ==========')
    print(f'{"condition":<22} {"processed":>9} {"skipped":>8} {"failed":>7} {"total":>7}')
    for s in all_stats:
        print(f'{s["condition"]:<22} {s["processed"]:>9} {s["skipped"]:>8} '
              f'{s["failed"]:>7} {s["total"]:>7}')
    print('-' * 56)
    print(f'{"ALL":<22} {g_processed:>9} {g_skipped:>8} {g_failed:>7} {g_total:>7}')
    print(f'Wall time: {total_elapsed/60:.1f} min')


if __name__ == '__main__':
    main()
