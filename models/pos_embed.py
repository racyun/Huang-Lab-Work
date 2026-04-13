# Copyright (c) Meta Platforms, Inc. and affiliates. (original MAE util)
# SPDX-License-Identifier: CC-BY-NC-4.0
"""2D sine-cosine position embeddings and checkpoint interpolation."""

from __future__ import annotations

import numpy as np
import torch


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim], dtype=np.float32), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return emb.astype(np.float32)


def interpolate_pos_embed(model: torch.nn.Module, checkpoint_model: dict) -> None:
    """Resize spatial pos_embed in *checkpoint_model* when patch grid changes (in-place)."""
    if "pos_embed" not in checkpoint_model:
        return
    pos_embed_checkpoint = checkpoint_model["pos_embed"]
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    new_size = int(num_patches**0.5)
    if orig_size != new_size:
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        checkpoint_model["pos_embed"] = torch.cat((extra_tokens, pos_tokens), dim=1)


def get_3d_sincos_pos_embed(
    embed_dim: int, grid_z: int, grid_h: int, grid_w: int, cls_token: bool = False
) -> np.ndarray:
    """Sin-cos positions for VideoMAE-style tubes: lexicographic flat index over (Z,H,W)."""
    assert embed_dim % 2 == 0
    L = grid_z * grid_h * grid_w
    pos = np.arange(L, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim], dtype=np.float32), pos_embed], axis=0)
    return pos_embed
