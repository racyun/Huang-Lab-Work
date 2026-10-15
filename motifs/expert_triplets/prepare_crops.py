#!/usr/bin/env python3
"""Expert triplets, step 1 — freeze the image split and render a crop buffer.

The annotation app (annotate_app.py) builds triplets on the fly from crops
rendered here. Nothing in this script caps how many crops get annotated: run
it again with a new --n-test / --n-train to append another batch whenever the
experts get close to the end of the buffer. A crop only becomes part of the
train or test set once the app actually shows it in a triplet.

What is fixed, and why:

  split.csv           every (condition, image_id) is assigned to train or test
                      ONCE, stratified by condition (= imaging date x stiffness).
                      Splitting by image, not by crop or triplet, keeps
                      neighbourhoods that share cells on one side only. The
                      file is never rewritten; images added later need
                      --assign-new, which only appends.

  render_config.json  crop size, contrast mode and marker style are frozen on
                      the first run so every crop is shown at the same scale
                      (cell size is something experts judge). Later batches
                      must match it.

Each crop is a square window centred on one cell's centroid, sized to cover a
typical 2-hop neighbourhood (the encoder's receptive field), with four tick marks
pointing at the centre cell. Cells whose window would run off the image are
skipped, and sampled centres in the same image are kept at least half a crop
apart so two crops never show mostly the same cells.

Outputs (in --out-dir, default <root>/expert_triplets/):
    split.csv, render_config.json
    crops.csv            one row per rendered crop (append-only)
    crops/<split>/*.png  the images the app serves

Usage:
    python prepare_crops.py --n-test 600 --n-train 1200
    python prepare_crops.py --n-test 300            # top up the test buffer
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stage4_motifs"))
from plot_motif_maps import _composite, _default_local_root, load_tile  # noqa: E402

KEY = ["condition", "image_id"]
CROP_COLS = ["crop_id", "split", "condition", "image_id", "cell_id",
             "centroid_x", "centroid_y", "x0", "y0", "size_px",
             "footprint_cells", "footprint_inside", "batch", "seed",
             "render_version", "file", "sha256", "created_at"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------- split ----
def make_or_load_split(path: Path, images: pd.DataFrame, test_frac: float,
                       seed: int, assign_new: bool) -> pd.DataFrame:
    """Stratified-by-condition image split, written once and then frozen."""
    if path.exists():
        split = pd.read_csv(path, dtype={"image_id": str})
        known = split.set_index(KEY).index
        new = images[~images.set_index(KEY).index.isin(known)]
        if len(new):
            if not assign_new:
                raise SystemExit(
                    f"{len(new)} image(s) in the master table are not in "
                    f"{path.name}. Re-run with --assign-new to append them "
                    f"(existing assignments never change).")
            add = _assign(new, test_frac, seed + len(split))
            split = pd.concat([split, add], ignore_index=True)
            split.to_csv(path, index=False)
            print(f"  appended {len(add)} new image(s) to {path.name}")
        return split
    split = _assign(images, test_frac, seed)
    split.to_csv(path, index=False)
    print(f"  wrote {path.name}: "
          f"{(split.split == 'test').sum()} test / "
          f"{(split.split == 'train').sum()} train images")
    return split


def _assign(images: pd.DataFrame, test_frac: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cond, grp in images.groupby("condition", sort=True):
        ids = grp["image_id"].tolist()
        rng.shuffle(ids)
        n_test = int(round(len(ids) * test_frac))
        if len(ids) >= 2:
            n_test = min(max(n_test, 1), len(ids) - 1)   # both sides get some
        for i, img in enumerate(ids):
            rows.append({"condition": cond, "image_id": img,
                         "split": "test" if i < n_test else "train",
                         "assigned_at": now(), "seed": seed,
                         "test_frac": test_frac})
    return pd.DataFrame(rows)


# ------------------------------------------------------------ footprint ----
def two_hop(pos: np.ndarray, k: int) -> list[np.ndarray]:
    """Per cell, the indices of its 2-hop k-NN neighbourhood (union kNN, the
    stage-2 default rule; the d_max pruning there is ignored, which only makes
    the footprint slightly larger)."""
    n = len(pos)
    kq = min(k, n - 1)
    if kq < 1:
        return [np.array([i]) for i in range(n)]
    _, idx = cKDTree(pos).query(pos, k=kq + 1)
    nbrs = [set(row[1:]) for row in idx]
    for i in range(n):                          # make symmetric (union kNN)
        for j in list(nbrs[i]):
            nbrs[j].add(i)
    out = []
    for i in range(n):
        hop1 = nbrs[i]
        hop2 = set().union(*(nbrs[j] for j in hop1)) if hop1 else set()
        out.append(np.fromiter(hop1 | hop2 | {i}, dtype=int))
    return out


def footprint_extent(pos: np.ndarray, members: np.ndarray, i: int) -> float:
    """Half-width of the square around cell i that holds its whole footprint."""
    return float(np.abs(pos[members] - pos[i]).max())


# ------------------------------------------------------------- rendering ----
def render(rgb: np.ndarray, cx: float, cy: float, half: int, marker_r: float,
           nbr_xy: np.ndarray | None) -> tuple[Image.Image, int, int]:
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    win = rgb[y0:y0 + 2 * half, x0:x0 + 2 * half]
    im = Image.fromarray((np.clip(win, 0, 1) * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    lx, ly = cx - x0, cy - y0
    if nbr_xy is not None:
        for nx, ny in nbr_xy:
            px, py = nx - x0, ny - y0
            d.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(255, 255, 255))
    # Four ticks pointing at the centre cell from just outside it, so the cell
    # itself is never covered. Dark halo first so they read on any background.
    w = max(2, int(round(marker_r / 10)))
    tick = max(6.0, marker_r * 0.6)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        seg = [lx + dx * marker_r, ly + dy * marker_r,
               lx + dx * (marker_r + tick), ly + dy * (marker_r + tick)]
        d.line(seg, fill=(0, 0, 0), width=w + 3)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        seg = [lx + dx * marker_r, ly + dy * marker_r,
               lx + dx * (marker_r + tick), ly + dy * (marker_r + tick)]
        d.line(seg, fill=(255, 255, 255), width=w)
    return im, x0, y0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def safe_name(*parts) -> str:
    raw = "__".join(str(p) for p in parts)
    clean = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in raw)
    return f"{clean[:80]}_{hashlib.sha1(raw.encode()).hexdigest()[:8]}"


# ------------------------------------------------------------------ main ----
def main() -> None:
    root = _default_local_root()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--master-table", type=Path, default=root / "master_table.csv")
    ap.add_argument("--imgs-dir", type=Path, default=root / "imgs")
    ap.add_argument("--out-dir", type=Path, default=root / "expert_triplets")
    ap.add_argument("--n-test", type=int, default=0, help="test crops to add")
    ap.add_argument("--n-train", type=int, default=0, help="train crops to add")
    ap.add_argument("--test-frac", type=float, default=1 / 3,
                    help="share of images per condition in test (first run only)")
    ap.add_argument("--k", type=int, default=8, help="k-NN k, as in stage 2")
    ap.add_argument("--crop-size", type=int, default=0,
                    help="crop width in px; 0 = auto from the 90th percentile "
                         "of 2-hop footprints (first run only, then frozen)")
    ap.add_argument("--contrast", choices=["tile", "global"], default="tile",
                    help="tile = stretch each image on its own percentiles (as "
                         "in the other expert sheets); global = one stretch for "
                         "all images, which keeps cross-image brightness "
                         "differences visible (first run only)")
    ap.add_argument("--mark-neighbors", action="store_true",
                    help="also dot the 2-hop neighbours' centroids")
    ap.add_argument("--assign-new", action="store_true",
                    help="append images missing from split.csv")
    ap.add_argument("--seed", type=int, default=None,
                    help="sampling seed for this batch (default: time-based)")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    seed = args.seed if args.seed is not None else int(datetime.now().timestamp())
    rng = np.random.default_rng(seed)

    master = pd.read_csv(args.master_table,
                         usecols=KEY + ["cell_id", "centroid_x", "centroid_y"],
                         dtype={"image_id": str})
    images = master[KEY].drop_duplicates()
    split = make_or_load_split(out / "split.csv", images, args.test_frac,
                               seed=0, assign_new=args.assign_new)

    # ---- frozen render settings ----
    cfg_path = out / "render_config.json"
    groups = {k: g for k, g in master.groupby(KEY, sort=False)}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if args.crop_size and args.crop_size != cfg["crop_size"]:
            raise SystemExit(f"crop size is frozen at {cfg['crop_size']} px "
                             f"(render_config.json); delete it only before any "
                             f"annotation exists")
    else:
        sizes, nn = [], []
        sample_keys = images.sample(min(40, len(images)), random_state=0)
        for key in sample_keys.itertuples(index=False):
            pos = groups[tuple(key)][["centroid_x", "centroid_y"]].to_numpy(float)
            if len(pos) < 3:
                continue
            fp = two_hop(pos, args.k)
            pick = rng.choice(len(pos), size=min(50, len(pos)), replace=False)
            sizes += [2 * footprint_extent(pos, fp[i], i) for i in pick]
            nn.append(np.median(cKDTree(pos).query(pos, k=2)[0][:, 1]))
        size = args.crop_size or int(np.ceil(np.percentile(sizes, 90) * 1.1 / 2) * 2)
        cfg = {"crop_size": size, "contrast": args.contrast, "k": args.k,
               "marker_radius": round(float(np.median(nn)) * 0.7, 1),
               "mark_neighbors": bool(args.mark_neighbors),
               "render_version": f"2hop-k{args.k}-{size}px-{args.contrast}"
                                 f"{'-nbr' if args.mark_neighbors else ''}-v1",
               "source_master_table": str(args.master_table.resolve()),
               "source_imgs_dir": str(args.imgs_dir.resolve()),
               "created_at": now()}
        cfg_path.write_text(json.dumps(cfg, indent=2))
        print(f"  froze render settings: {cfg['crop_size']} px crops "
              f"({cfg['contrast']} contrast), centre marker at r={cfg['marker_radius']} px")
    half = cfg["crop_size"] // 2

    # ---- existing crops: avoid reuse and crowding ----
    crops_path = out / "crops.csv"
    old = (pd.read_csv(crops_path, dtype={"image_id": str})
           if crops_path.exists() else pd.DataFrame(columns=CROP_COLS))
    taken: dict = {}
    for r in old.itertuples(index=False):
        taken.setdefault((r.condition, r.image_id), []).append(
            (r.centroid_x, r.centroid_y))

    gstretch = None
    if cfg["contrast"] == "global":
        gstretch = global_stretch(split, args.imgs_dir, out)

    batch = datetime.now().strftime("%Y%m%d-%H%M%S")
    new_rows = []
    for which, n_want in (("test", args.n_test), ("train", args.n_train)):
        if n_want <= 0:
            continue
        pool = split[split.split == which]
        print(f"\n{which}: rendering {n_want} crop(s) from {len(pool)} image(s)")
        (out / "crops" / which).mkdir(parents=True, exist_ok=True)
        made, full = 0, set()
        # An image runs out once every eligible cell is used or crowded; its
        # unfilled share is redrawn from the images that still have room.
        while made < n_want:
            avail = pool[~pool.set_index(KEY).index.isin(list(full))]
            if avail.empty:
                break
            for key, n_img in plan_images(avail, n_want - made, rng).items():
                got = render_image(*key, n_img, which, groups, taken, args,
                                   cfg, half, gstretch, rng, out, batch, seed,
                                   new_rows)
                made += got
                if got < n_img:
                    full.add(key)
        print(f"  {which}: {made} crop(s) written")
        if made < n_want:
            print(f"  [short] only {made} of {n_want}: every {which} image is "
                  f"out of eligible cells at {cfg['crop_size']} px crops spaced "
                  f"{half} px apart")

    if new_rows:
        new = pd.DataFrame(new_rows, columns=CROP_COLS)
        both = new if old.empty else pd.concat([old, new], ignore_index=True)
        dupes = both["crop_id"].duplicated().sum()
        assert dupes == 0, f"{dupes} duplicate crop_id(s); refusing to write"
        # Write-then-rename, so a running annotate_app never reads a
        # half-written file while a new batch is being appended.
        tmp = crops_path.with_suffix(".csv.tmp")
        both.to_csv(tmp, index=False)
        tmp.replace(crops_path)
    counts = (pd.read_csv(crops_path).split.value_counts().to_dict()
              if crops_path.exists() else {})
    print(f"\ncrops.csv now holds {counts} (batch {batch}, seed {seed})")


def plan_images(pool: pd.DataFrame, n_want: int, rng) -> dict:
    """Balance draws over conditions first, then images within a condition."""
    conds = sorted(pool.condition.unique())
    plan: dict = {}
    for _ in range(n_want):
        cond = conds[rng.integers(len(conds))]
        imgs = pool[pool.condition == cond].image_id.tolist()
        key = (cond, imgs[rng.integers(len(imgs))])
        plan[key] = plan.get(key, 0) + 1
    return plan


def render_image(cond, img, n_img, which, groups, taken, args, cfg, half,
                 gstretch, rng, out, batch, seed, new_rows) -> int:
    cells = groups.get((cond, img))
    if cells is None or len(cells) < 3:
        return 0
    tile = load_tile(args.imgs_dir, cond, img)
    if tile is None:
        return 0
    rgb = _composite(tile) if gstretch is None else apply_stretch(tile, gstretch)
    h, w = rgb.shape[:2]
    pos = cells[["centroid_x", "centroid_y"]].to_numpy(float)
    ids = cells["cell_id"].to_numpy()
    fp = two_hop(pos, cfg["k"])
    prior = list(taken.get((cond, img), []))
    order = rng.permutation(len(pos))
    made = 0
    for i in order:
        if made >= n_img:
            break
        cx, cy = pos[i]
        x0, y0 = int(round(cx)) - half, int(round(cy)) - half
        if x0 < 0 or y0 < 0 or x0 + 2 * half > w or y0 + 2 * half > h:
            continue                                     # window off the image
        if prior and np.min(np.hypot(*(np.array(prior) - (cx, cy)).T)) < half:
            continue                                     # overlaps another crop
        members = fp[i]
        inside = np.mean((np.abs(pos[members] - pos[i]) <= half).all(axis=1))
        nbr_xy = pos[members[members != i]] if cfg["mark_neighbors"] else None
        im, x0, y0 = render(rgb, cx, cy, half, cfg["marker_radius"], nbr_xy)
        name = safe_name(cond, img, ids[i]) + ".png"
        rel = Path("crops") / which / name
        im.save(out / rel)
        new_rows.append({
            "crop_id": f"{cond}/{img}/{ids[i]}", "split": which,
            "condition": cond, "image_id": img, "cell_id": int(ids[i]),
            "centroid_x": round(float(cx), 2), "centroid_y": round(float(cy), 2),
            "x0": x0, "y0": y0, "size_px": 2 * half,
            "footprint_cells": int(len(members)),
            "footprint_inside": round(float(inside), 3),
            "batch": batch, "seed": seed,
            "render_version": cfg["render_version"], "file": str(rel),
            "sha256": sha256(out / rel), "created_at": now()})
        prior.append((cx, cy))
        made += 1
    taken[(cond, img)] = prior
    return made


def global_stretch(split, imgs_dir, out) -> dict:
    """1st/99th percentiles per channel pooled over a sample of images; saved
    so later batches reuse the same stretch."""
    path = out / "global_stretch.json"
    if path.exists():
        return json.loads(path.read_text())
    from plot_motif_maps import DAPI_PLANE, TAGLN_PLANE, VECAD_PLANE
    vals = {p: [] for p in (TAGLN_PLANE, VECAD_PLANE, DAPI_PLANE)}
    for r in split.sample(min(20, len(split)), random_state=0).itertuples():
        t = load_tile(imgs_dir, r.condition, r.image_id)
        if t is None or t.ndim != 3:
            continue
        for p in vals:
            vals[p].append(np.random.default_rng(0).choice(t[..., p].ravel(), 20000))
    st = {str(p): [float(np.percentile(np.concatenate(v), 1)),
                   float(np.percentile(np.concatenate(v), 99))]
          for p, v in vals.items() if v}
    path.write_text(json.dumps(st, indent=2))
    return st


def apply_stretch(tile, st: dict) -> np.ndarray:
    from plot_motif_maps import DAPI_PLANE, TAGLN_PLANE, VECAD_PLANE
    rgb = np.zeros((*tile.shape[:2], 3), dtype=float)
    for out_c, p in enumerate((TAGLN_PLANE, VECAD_PLANE, DAPI_PLANE)):
        lo, hi = st[str(p)]
        rgb[..., out_c] = np.clip((tile[..., p].astype(float) - lo) / (hi - lo + 1e-6), 0, 1)
    return rgb


if __name__ == "__main__":
    main()
