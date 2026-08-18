#!/usr/bin/env python3
"""Stage 4 visualisation — paint motif labels back onto the tissue.

Produces three kinds of figure:

  1. PER-IMAGE TWO-PANEL   motif_map_<cond>__<img>.png
     LEFT = motif map (cells coloured by motif), RIGHT = the original tile.

  2. PAIRED GRIDS          motif_map_grid_part<N>.png
     3 conditions per figure, 2 rows x 3 columns: top row the motif maps,
     bottom row the matching original images directly beneath each map.

  3. PER-MOTIF BREAKDOWN   motif_breakdown_<cond>__<img>.png
     One panel per motif for a single image, showing only that motif's cells,
     coloured by EndMT score (other cells shown faint grey for context).

Motif colours are fixed by motif id across every panel and image.

Usage:
    python plot_motif_maps.py --motifs <stage4>/motifs.parquet \
        --graphs-dir <root>/graphs --imgs-dir <root>/imgs
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

VERSION = "2026-07-27 two-panel + paired-grid + per-motif-EndMT"
IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def _default_local_root() -> Path:
    default = Path("/teamspace/studios/this_studio/cellpose_work")
    if "CELLPOSE_LOCAL_ROOT" in os.environ:
        return Path(os.environ["CELLPOSE_LOCAL_ROOT"]).expanduser()
    if default.parent.parent.exists():
        return default
    return Path.home() / "cellpose_work"


def motif_colors(n_motifs: int):
    base = plt.cm.tab10 if n_motifs <= 10 else plt.cm.tab20
    return [base(i % base.N) for i in range(n_motifs)]


def _stretch(ch):
    lo, hi = np.percentile(ch, [1, 99])
    return np.clip((ch.astype(float) - lo) / (hi - lo + 1e-6), 0, 1)


def _composite(im):
    """R=TAGLN(ch3), G=VE-cad(ch2), B=DAPI(ch1)."""
    if im.ndim == 2:
        s = _stretch(im)
        return np.dstack([s, s, s])
    rgb = np.zeros((*im.shape[:2], 3), dtype=float)
    rgb[..., 0] = _stretch(im[..., 2])
    rgb[..., 1] = _stretch(im[..., 1])
    rgb[..., 2] = _stretch(im[..., 0])
    return rgb


def _norm(s: str) -> str:
    """Loose key for matching names that differ only in case/punctuation."""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


_TILE_INDEX: dict[str, dict] = {}


def build_tile_index(imgs_dir: Path, cond: str, verbose: bool = True) -> dict:
    """Recursively index EVERY image file for a condition, at any depth.

    Layout-agnostic on purpose: works whether tiles live in
    <imgs>/<cond>/tiles/, <imgs>/<cond>/, or somewhere deeper. Cached per
    condition so the (slow, over Drive) walk happens once.
    """
    if cond in _TILE_INDEX:
        return _TILE_INDEX[cond]

    roots = [imgs_dir / cond, imgs_dir]
    exact: dict[str, Path] = {}
    loose: dict[str, Path] = {}
    scanned_root = None
    for root in roots:
        if not root.exists():
            continue
        scanned_root = root
        try:
            for p in root.rglob("*"):
                if p.suffix.lower() in IMG_EXTS:
                    exact.setdefault(p.stem, p)
                    loose.setdefault(_norm(p.stem), p)
        except Exception as e:
            print(f"    [index error] {root}: {e}")
        if exact:
            break                                   # found tiles; stop widening

    idx = {"exact": exact, "loose": loose, "root": scanned_root}
    _TILE_INDEX[cond] = idx
    if verbose:
        if exact:
            sample = list(exact)[:3]
            print(f"    indexed {len(exact)} tile(s) under {scanned_root}  "
                  f"e.g. {sample}")
        else:
            print(f"    [NO TILES INDEXED] nothing with extensions {IMG_EXTS} "
                  f"found under {roots[0]} or {roots[1]}")
    return idx


def find_tile(imgs_dir: Path, cond: str, img_id: str, diagnose: bool = True):
    """Look the tile up in the recursive index; report clearly on failure."""
    idx = build_tile_index(imgs_dir, cond, verbose=diagnose)
    p = idx["exact"].get(img_id) or idx["loose"].get(_norm(img_id))
    if p is not None:
        return p
    # last resort: unique substring match (handles extra prefixes/suffixes)
    hits = [v for k, v in idx["loose"].items() if _norm(img_id) in k]
    if len(hits) == 1:
        return hits[0]
    if diagnose:
        print(f"    [tile NOT FOUND] image_id='{img_id}' in condition '{cond}'")
        if idx["exact"]:
            print(f"      {len(idx['exact'])} tiles are indexed; example stems: "
                  f"{list(idx['exact'])[:5]}")
            print("      -> the tile names do not match the image_id. Compare the "
                  "stems above with the image_id and tell me the pattern.")
        if len(hits) > 1:
            print(f"      ambiguous substring match ({len(hits)} candidates)")
    return None


def load_tile(imgs_dir: Path, cond: str, img_id: str):
    p = find_tile(imgs_dir, cond, img_id)
    if p is None:
        return None
    try:
        import tifffile
        img = tifffile.imread(str(p))
        print(f"    tile OK: {p.name}  shape={img.shape}")
        return img
    except Exception as e:
        print(f"    [tile READ FAILED] {p}: {e}")
        return None


def draw_map(ax, g, motifs_for_image, colors, overlay_img=None, node_size=28,
             show_edges=True, color_by="motif", endmt=None,
             within_motif_edges_only=False):
    """Draw an image's graph.

    color_by='motif'  -> nodes take their motif colour
    color_by='endmt'  -> nodes take the coolwarm EndMT colour map (0..1), the
                         same mapping used by the neighbourhood-subgraph viewers
    within_motif_edges_only -> keep only edges whose two endpoints share a motif
    """
    pos = g.pos.numpy()
    if overlay_img is not None:
        ax.imshow(_composite(overlay_img))
    if show_edges:
        ei = g.edge_index.numpy()
        if within_motif_edges_only:
            keep = motifs_for_image[ei[0]] == motifs_for_image[ei[1]]
            ei = ei[:, keep]
        edge_col = "white" if overlay_img is not None else "gray"
        alpha = 0.35 if overlay_img is not None else 0.4
        for s, d in ei.T:
            ax.plot([pos[s, 0], pos[d, 0]], [pos[s, 1], pos[d, 1]],
                    lw=0.35, color=edge_col, alpha=alpha, zorder=1)
    if color_by == "endmt" and endmt is not None:
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=endmt, cmap="coolwarm",
                        vmin=0, vmax=1, s=node_size, zorder=2,
                        edgecolors="black", linewidths=0.25)
    else:
        sc = ax.scatter(pos[:, 0], pos[:, 1],
                        c=[colors[m] for m in motifs_for_image],
                        s=node_size, zorder=2, edgecolors="black", linewidths=0.25)
    if overlay_img is None:
        ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    return sc


def draw_endmt_for_motif(ax, g, motifs_for_image, endmt, motif_id, node_size=26):
    """One motif in isolation, its cells coloured by EndMT score."""
    pos = g.pos.numpy()
    sel = motifs_for_image == motif_id
    ax.scatter(pos[~sel, 0], pos[~sel, 1], s=node_size * 0.35, color="lightgrey",
               alpha=0.55, zorder=1)                       # context
    sc = ax.scatter(pos[sel, 0], pos[sel, 1], c=endmt[sel], cmap="coolwarm",
                    vmin=0, vmax=1, s=node_size, zorder=2,
                    edgecolors="black", linewidths=0.3)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    return sc, int(sel.sum())


def build_motif_catalog(mdf, graphs_dir: Path, n_motifs: int, colors,
                        e_idx, out_dir: Path, max_images: int, seed: int):
    """One catalog image per unique motif.

    For each motif we scan images, take the LARGEST connected patch of that
    motif (its cells plus only the edges between them) as the exemplar to draw,
    and colour its nodes by EndMT score.

    "Occurrences" is the total number of cells assigned to that motif across
    every image in the dataset (exact — taken from motifs.parquet, not the
    scanned sample). The number of distinct spatial patches found while
    scanning is reported separately, since one contiguous patch contains many
    cells.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    cat_dir = out_dir / "Motif catalog"
    cat_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nBuilding motif catalog -> {cat_dir}")
    print(f"  will write {n_motifs} catalog image(s) — one per unique motif")

    # exact occurrence counts over the WHOLE dataset
    cell_counts = mdf["motif"].value_counts().to_dict()
    imgs_present = (mdf.groupby("motif")["image_id"].nunique()).to_dict()
    n_images_total = mdf[["condition", "image_id"]].drop_duplicates().shape[0]

    # scan a sample of images for the best exemplar patch per motif
    pairs = mdf[["condition", "image_id"]].drop_duplicates()
    if len(pairs) > max_images:
        pairs = pairs.sample(max_images, random_state=seed)
    best = {m: None for m in range(n_motifs)}      # (size, cond, img, node_ids)
    patches = {m: 0 for m in range(n_motifs)}
    print(f"  scanning {len(pairs)} of {n_images_total} source images to pick "
          f"each motif's exemplar patch...")

    for cond, img in pairs.itertuples(index=False):
        gp = graphs_dir / cond / f"{img}.pt"
        if not gp.exists():
            continue
        g = torch.load(str(gp), weights_only=False)
        sub = mdf[(mdf.condition == cond) & (mdf.image_id == img)]
        if len(sub) != g.num_nodes or g.edge_index.shape[1] == 0:
            continue
        lab = sub["motif"].to_numpy()
        src, dst = g.edge_index.numpy()
        same = lab[src] == lab[dst]                # within-motif edges only
        for m in range(n_motifs):
            idx = np.flatnonzero(lab == m)
            if idx.size < 2:
                continue
            remap = -np.ones(g.num_nodes, dtype=int)
            remap[idx] = np.arange(idx.size)
            em = same & (lab[src] == m)
            s2, d2 = remap[src[em]], remap[dst[em]]
            if s2.size == 0:
                continue
            adj = coo_matrix((np.ones(s2.size), (s2, d2)),
                             shape=(idx.size, idx.size))
            ncomp, cl = connected_components(adj, directed=False)
            sizes = np.bincount(cl)
            patches[m] += int((sizes >= 2).sum())
            k = int(sizes.argmax())
            if best[m] is None or sizes[k] > best[m][0]:
                best[m] = (int(sizes[k]), cond, img, idx[cl == k])

    # draw one figure per motif
    for m in range(n_motifs):
        if best[m] is None:
            print(f"  [skip] motif {m}: no connected patch found")
            continue
        size, cond, img, node_ids = best[m]
        g = torch.load(str(graphs_dir / cond / f"{img}.pt"), weights_only=False)
        pos = g.pos.numpy()[node_ids]
        endmt = g.x[node_ids, e_idx].numpy() if e_idx is not None else None

        keep = set(node_ids.tolist())
        src, dst = g.edge_index.numpy()
        remap = {n: i for i, n in enumerate(node_ids.tolist())}
        edges = [(remap[s], remap[d]) for s, d in zip(src, dst)
                 if s in keep and d in keep]

        fig, ax = plt.subplots(figsize=(8, 8.8))
        for s, d in edges:
            ax.plot([pos[s, 0], pos[d, 0]], [pos[s, 1], pos[d, 1]],
                    lw=0.9, color="gray", alpha=0.6, zorder=1)
        if endmt is not None:
            sc = ax.scatter(pos[:, 0], pos[:, 1], c=endmt, cmap="coolwarm",
                            vmin=0, vmax=1, s=90, zorder=2,
                            edgecolors="black", linewidths=0.4)
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("EndMT score (0 = endothelial, 1 = mesenchymal)")
        else:
            ax.scatter(pos[:, 0], pos[:, 1], color=colors[m], s=90, zorder=2,
                       edgecolors="black", linewidths=0.4)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
        ax.set_title(f"Motif {m}", fontsize=17, color=colors[m], fontweight="bold")
        occ = cell_counts.get(m, 0)
        fig.text(0.5, 0.045, f"Occurrences: {occ:,}", ha="center", fontsize=15,
                 fontweight="bold")
        fig.text(0.5, 0.012,
                 f"present in {imgs_present.get(m, 0):,} / {n_images_total:,} images"
                 f"   |   {patches[m]:,} distinct patches in {len(pairs)} scanned"
                 f"   |   exemplar: {cond}/{img} ({size} cells)",
                 ha="center", fontsize=9, color="dimgrey")
        fp = cat_dir / f"motif_{m:02d}.png"
        fig.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {fp.name}  (Occurrences: {occ:,}, exemplar {size} cells)")


def build_motif_galleries(mdf, graphs_dir: Path, n_motifs: int, e_idx,
                          out_dir: Path, scan_images: int, per_motif_limit: int,
                          min_patch: int, seed: int):
    """One folder per motif, holding many instances of that motif.

    Each image shows one connected patch of the motif in its local context:
    every cell/edge in the surrounding window is drawn grey, while the patch's
    own cells and the edges between them are coloured by EndMT score (edges use
    the mean EndMT of their two endpoints) — the same coolwarm 0..1 mapping as
    the motif catalog and the neighbourhood-subgraph viewers.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    cmap = plt.cm.coolwarm
    dirs = {}
    for m in range(n_motifs):
        d = out_dir / f"Motif {m:02d}"
        d.mkdir(parents=True, exist_ok=True)
        dirs[m] = d

    pairs = mdf[["condition", "image_id"]].drop_duplicates()
    if len(pairs) > scan_images:
        pairs = pairs.sample(scan_images, random_state=seed)
    print(f"\nBuilding per-motif galleries -> {out_dir}")
    print(f"  {n_motifs} folders, up to {per_motif_limit} instances each")
    print(f"  scanning {len(pairs)} source images (patches of >= {min_patch} cells)")

    saved = {m: 0 for m in range(n_motifs)}
    for cond, img in pairs.itertuples(index=False):
        if all(saved[m] >= per_motif_limit for m in range(n_motifs)):
            break
        gp = graphs_dir / cond / f"{img}.pt"
        if not gp.exists():
            continue
        g = torch.load(str(gp), weights_only=False)
        sub = mdf[(mdf.condition == cond) & (mdf.image_id == img)]
        if len(sub) != g.num_nodes or g.edge_index.shape[1] == 0:
            continue
        lab = sub["motif"].to_numpy()
        pos = g.pos.numpy()
        endmt = g.x[:, e_idx].numpy() if e_idx is not None else np.zeros(g.num_nodes)
        src, dst = g.edge_index.numpy()

        for m in range(n_motifs):
            if saved[m] >= per_motif_limit:
                continue
            idx = np.flatnonzero(lab == m)
            if idx.size < min_patch:
                continue
            remap = -np.ones(g.num_nodes, dtype=int)
            remap[idx] = np.arange(idx.size)
            em = (lab[src] == m) & (lab[dst] == m)
            s2, d2 = remap[src[em]], remap[dst[em]]
            if s2.size == 0:
                continue
            adj = coo_matrix((np.ones(s2.size), (s2, d2)), shape=(idx.size, idx.size))
            _, cl = connected_components(adj, directed=False)
            for c in range(cl.max() + 1):
                if saved[m] >= per_motif_limit:
                    break
                nodes = idx[cl == c]
                if nodes.size < min_patch:
                    continue
                # zoom window around this patch
                pad = 60
                x0, y0 = pos[nodes].min(0) - pad
                x1, y1 = pos[nodes].max(0) + pad
                inwin = ((pos[:, 0] >= x0) & (pos[:, 0] <= x1) &
                         (pos[:, 1] >= y0) & (pos[:, 1] <= y1))

                fig, ax = plt.subplots(figsize=(7, 7.6))
                # grey context: everything in the window
                wm = inwin[src] & inwin[dst]
                for s, d in zip(src[wm], dst[wm]):
                    ax.plot([pos[s, 0], pos[d, 0]], [pos[s, 1], pos[d, 1]],
                            lw=0.5, color="lightgrey", alpha=0.8, zorder=1)
                ax.scatter(pos[inwin, 0], pos[inwin, 1], s=26, color="lightgrey",
                           edgecolors="grey", linewidths=0.2, zorder=2)
                # the motif patch: edges coloured by mean EndMT of endpoints
                nodeset = set(nodes.tolist())
                pm = np.array([s in nodeset and d in nodeset
                               for s, d in zip(src, dst)])
                for s, d in zip(src[pm], dst[pm]):
                    ax.plot([pos[s, 0], pos[d, 0]], [pos[s, 1], pos[d, 1]],
                            lw=1.8, color=cmap(float(np.clip(
                                (endmt[s] + endmt[d]) / 2.0, 0, 1))),
                            alpha=0.95, zorder=3)
                sc = ax.scatter(pos[nodes, 0], pos[nodes, 1], c=endmt[nodes],
                                cmap="coolwarm", vmin=0, vmax=1, s=95, zorder=4,
                                edgecolors="black", linewidths=0.5)
                cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
                cb.set_label("EndMT score (0 = endothelial, 1 = mesenchymal)")
                ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
                ax.set_aspect("equal"); ax.axis("off")
                ax.set_title(f"Motif {m} — instance {saved[m] + 1}\n"
                             f"{cond} / {img}   ({nodes.size} cells)", fontsize=11)
                fp = dirs[m] / f"motif{m:02d}_{saved[m] + 1:04d}_{cond}__{img}.png"
                fig.savefig(fp, dpi=120, bbox_inches="tight")
                plt.close(fig)
                saved[m] += 1

    print("  instances saved per motif:")
    for m in range(n_motifs):
        print(f"    Motif {m:02d}: {saved[m]}  -> {dirs[m].name}/")
    return saved


def endmt_index(graphs_dir: Path):
    fn = graphs_dir / "feature_names.json"
    if fn.exists():
        names = json.loads(fn.read_text())
        if "endmt_score" in names:
            return names.index("endmt_score")
    return None


def main() -> None:
    root = _default_local_root()
    ap = argparse.ArgumentParser(description="Plot motif cluster maps (Stage 4).")
    ap.add_argument("--motifs", type=Path, default=root / "stage4" / "motifs.parquet")
    ap.add_argument("--graphs-dir", type=Path, default=root / "graphs")
    ap.add_argument("--imgs-dir", type=Path, default=root / "imgs")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--images-per-condition", type=int, default=1)
    ap.add_argument("--images", nargs="*", default=None,
                    help="explicit 'condition/image_id' entries")
    ap.add_argument("--conditions-per-grid", type=int, default=3,
                    help="conditions per paired-grid figure (part1, part2, ...)")
    ap.add_argument("--overlay", action="store_true",
                    help="draw the motif map on top of the tile as well")
    ap.add_argument("--no-edges", action="store_true")
    ap.add_argument("--catalog", action="store_true",
                    help="also build the 'Motif catalog' subfolder: one exemplar "
                         "image per unique motif, nodes coloured by EndMT score")
    ap.add_argument("--catalog-scan-images", type=int, default=300,
                    help="how many SOURCE images to scan when searching for each "
                         "motif's exemplar patch. This does NOT set the number of "
                         "catalog images — that is always one per motif. Scanning "
                         "more is slower but may find a larger exemplar patch.")
    ap.add_argument("--galleries", action="store_true",
                    help="build one folder per motif ('Motif 00', 'Motif 01', ...) "
                         "holding many instances of that motif, context in grey and "
                         "the motif's own cells/edges coloured by EndMT score")
    ap.add_argument("--per-motif-limit", type=int, default=100,
                    help="max instance images saved per motif folder (default 100). "
                         "Raise for a fuller gallery; every patch would be tens of "
                         "thousands of images.")
    ap.add_argument("--gallery-scan-images", type=int, default=300,
                    help="source images scanned when collecting motif instances")
    ap.add_argument("--min-patch", type=int, default=4,
                    help="ignore motif patches smaller than this many cells")
    ap.add_argument("--debug-tiles", action="store_true",
                    help="report which tiles can be found for the selected images, "
                         "then exit without plotting")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"plot_motif_maps.py  [{VERSION}]")
    out_dir = args.out_dir or args.motifs.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    mdf = pd.read_parquet(args.motifs)
    n_motifs = int(mdf["motif"].max()) + 1
    colors = motif_colors(n_motifs)
    e_idx = endmt_index(args.graphs_dir)
    print(f"Loaded {len(mdf)} motif labels across {n_motifs} motifs")
    print(f"imgs-dir: {args.imgs_dir}  (exists: {args.imgs_dir.exists()})")
    if e_idx is None:
        print("  [warn] endmt_score not in feature_names.json -> per-motif "
              "breakdown will be skipped")

    # ---- choose images ----
    if args.images:
        picks = [tuple(s.split("/", 1)) for s in args.images]
    else:
        rng = np.random.default_rng(args.seed)
        picks = []
        for cond, sub in mdf.groupby("condition"):
            imgs = sub["image_id"].unique()
            take = rng.choice(imgs, size=min(args.images_per_condition, len(imgs)),
                              replace=False)
            picks += [(cond, i) for i in take]
    print(f"Drawing {len(picks)} image(s)\n")

    # ---- tile-only diagnostic: answer "where are the tiles?" and stop ----
    if args.debug_tiles:
        print("=" * 70)
        print("TILE DIAGNOSTIC")
        print("=" * 70)
        for cond, img_id in picks:
            print(f"\n{cond} / {img_id}")
            p = find_tile(args.imgs_dir, cond, img_id)
            print(f"  -> {'FOUND: ' + str(p) if p else 'NOT FOUND'}")
        print("\n" + "=" * 70)
        return

    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[m],
                      markeredgecolor="black", markersize=9, label=f"motif {m}")
               for m in range(n_motifs)]

    panels, n_with_tile = [], 0
    for cond, img_id in picks:
        gpath = args.graphs_dir / cond / f"{img_id}.pt"
        if not gpath.exists():
            print(f"  [skip] no graph for {cond}/{img_id}")
            continue
        g = torch.load(str(gpath), weights_only=False)
        sub = mdf[(mdf.condition == cond) & (mdf.image_id == img_id)]
        if len(sub) != g.num_nodes:
            print(f"  [skip] {cond}/{img_id}: {len(sub)} labels vs {g.num_nodes} nodes")
            continue
        mfi = sub["motif"].to_numpy()
        print(f"  {cond}/{img_id}")
        raw = load_tile(args.imgs_dir, cond, img_id)
        if raw is not None:
            n_with_tile += 1

        # ---- 1. per-image FOUR-panel figure ----
        #   1 clustering map | 2 original | 3 full overlay | 4 within-motif
        #   overlay, nodes coloured by EndMT
        endmt = g.x[:, e_idx].numpy() if e_idx is not None else None
        if raw is not None:
            fig, axes4 = plt.subplots(1, 4, figsize=(34, 8.5))
            draw_map(axes4[0], g, mfi, colors, None, node_size=30,
                     show_edges=not args.no_edges)
            axes4[0].set_title(f"1. Clustering map — {g.num_nodes} cells", fontsize=12)

            axes4[1].imshow(_composite(raw))
            axes4[1].set_title("2. Original image\n(R=TAGLN, G=VE-cad, B=DAPI)",
                               fontsize=12)
            axes4[1].axis("off")

            draw_map(axes4[2], g, mfi, colors, raw, node_size=30,
                     show_edges=not args.no_edges)
            axes4[2].set_title("3. Full overlay\n(all edges)", fontsize=12)

            sc4 = draw_map(axes4[3], g, mfi, colors, raw, node_size=30,
                           show_edges=not args.no_edges, color_by="endmt",
                           endmt=endmt, within_motif_edges_only=True)
            axes4[3].set_title("4. Within-motif overlay\n(same-motif edges only, "
                               "nodes = EndMT)", fontsize=12)
            if endmt is not None:
                cb = fig.colorbar(sc4, ax=axes4[3], fraction=0.046, pad=0.02)
                cb.set_label("EndMT score", fontsize=9)

            axes4[0].legend(handles=handles, loc="upper left",
                            bbox_to_anchor=(0, -0.02), ncol=min(n_motifs, 6),
                            fontsize=9, frameon=False)
            fig.suptitle(f"{cond} / {img_id}", fontsize=14)
        else:
            fig, axL = plt.subplots(figsize=(9, 9))
            draw_map(axL, g, mfi, colors, None, node_size=36,
                     show_edges=not args.no_edges)
            axL.set_title(f"{cond} / {img_id}   ({g.num_nodes} cells)", fontsize=11)
            axL.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                       fontsize=9, frameon=False)
        fig.tight_layout()
        fp = out_dir / f"motif_map_{cond}__{img_id}.png"
        fig.savefig(fp, dpi=140, bbox_inches="tight")
        plt.close(fig)
        from PIL import Image as _Im
        w, h = _Im.open(fp).size
        print(f"    wrote {fp.name}  {w}x{h}  "
              f"({'FOUR-PANEL' if raw is not None else 'single panel (no tile)'})")

        # ---- 3. per-motif breakdown, coloured by EndMT ----
        if e_idx is not None:
            endmt = g.x[:, e_idx].numpy()
            ncol = int(np.ceil(np.sqrt(n_motifs)))
            nrow = int(np.ceil(n_motifs / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5 * nrow),
                                     squeeze=False)
            for ax in axes.flat:
                ax.axis("off")
            sc = None
            for m in range(n_motifs):
                ax = axes[m // ncol][m % ncol]
                sc, cnt = draw_endmt_for_motif(ax, g, mfi, endmt, m)
                ax.set_title(f"motif {m}  ({cnt} cells, "
                             f"{100 * cnt / g.num_nodes:.1f}%)", fontsize=11,
                             color=colors[m])
            if sc is not None:
                cbar = fig.colorbar(sc, ax=axes, fraction=0.02, pad=0.02)
                cbar.set_label("EndMT score (0 = endothelial, 1 = mesenchymal)")
            fig.suptitle(f"Per-motif breakdown — {cond} / {img_id}\n"
                         f"each panel shows one motif, cells coloured by EndMT score",
                         fontsize=13)
            fig.savefig(out_dir / f"motif_breakdown_{cond}__{img_id}.png", dpi=140,
                        bbox_inches="tight")
            plt.close(fig)

        panels.append((cond, img_id, g, mfi, raw))

    print(f"\nTiles found for {n_with_tile}/{len(panels)} image(s)")
    if n_with_tile == 0 and panels:
        print("  !! No original images were located — the paths above show where "
              "we looked. Fix --imgs-dir (expected <imgs-dir>/<condition>/tiles/"
              "<image_id>.tif) and re-run.")

    # ---- 2. paired grids: one figure per group of conditions ----
    by_cond: dict[str, tuple] = {}
    for p in panels:                                   # first image per condition
        by_cond.setdefault(p[0], p)
    conds = sorted(by_cond)
    per = max(1, args.conditions_per_grid)
    for part, i in enumerate(range(0, len(conds), per), start=1):
        chunk = [by_cond[c] for c in conds[i:i + per]]
        ncol = len(chunk)
        fig, axes = plt.subplots(2, ncol, figsize=(6 * ncol, 12), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for k, (cond, img_id, g, mfi, raw) in enumerate(chunk):
            draw_map(axes[0][k], g, mfi, colors, raw if args.overlay else None,
                     node_size=18, show_edges=not args.no_edges)
            axes[0][k].set_title(f"{cond}\n{img_id}", fontsize=11)
            if raw is not None:
                axes[1][k].imshow(_composite(raw))
            else:
                axes[1][k].text(0.5, 0.5, "(original image\nnot found)",
                                ha="center", va="center", fontsize=11, color="grey")
            axes[1][k].set_title("original image", fontsize=10)
        fig.legend(handles=handles, loc="lower center", ncol=min(n_motifs, 8),
                   fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.01))
        fig.suptitle(f"Motif maps (top) and original images (bottom) — part {part}",
                     fontsize=14)
        fig.tight_layout()
        fname = out_dir / f"motif_map_grid_part{part}.png"
        fig.savefig(fname, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {fname.name}  ({ncol} conditions)")

    # ---- 4. motif catalog ----
    if args.catalog:
        build_motif_catalog(mdf, args.graphs_dir, n_motifs, colors, e_idx,
                            out_dir, args.catalog_scan_images, args.seed)

    # ---- 5. per-motif galleries ----
    if args.galleries:
        build_motif_galleries(mdf, args.graphs_dir, n_motifs, e_idx, out_dir,
                              args.gallery_scan_images, args.per_motif_limit,
                              args.min_patch, args.seed)


if __name__ == "__main__":
    main()
