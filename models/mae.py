# Copyright (c) Meta Platforms, Inc. and affiliates. (derived from MAE)
# SPDX-License-Identifier: CC-BY-NC-4.0
"""MAE building blocks: ViT encoder, masking, patchify; full MAE for single-image pretraining."""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.pos_embed import get_2d_sincos_pos_embed


def _patch_embed_cls():
    try:
        from timm.layers import PatchEmbed

        return PatchEmbed
    except ImportError:
        from timm.models.layers import PatchEmbed

        return PatchEmbed


def _vit_block_cls():
    try:
        from timm.layers import Block

        return Block
    except ImportError:
        from timm.models.vision_transformer import Block

        return Block


def random_masking(x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random patch masking. x: [N, L, D]. Returns masked sequence (subset), binary mask [N,L], ids_restore."""
    N, L, D = x.shape
    len_keep = int(L * (1.0 - mask_ratio))
    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    mask = torch.ones((N, L), device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore


def patchify(imgs: torch.Tensor, patch_size: int, in_chans: int) -> torch.Tensor:
    """imgs: [N, C, H, W] -> [N, L, patch_size**2 * C]"""
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], in_chans, h, p, w, p))
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * in_chans))
    return x


def unpatchify(x: torch.Tensor, patch_size: int, in_chans: int) -> torch.Tensor:
    """[N, L, p*p*C] -> [N, C, H, W]"""
    p = patch_size
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]
    x = x.reshape(shape=(x.shape[0], h, w, p, p, in_chans))
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(shape=(x.shape[0], in_chans, h * p, w * p))


class MAEViTEncoder(nn.Module):
    """ViT encoder with patch embedding, fixed sin-cos pos enc, and random masking (MAE-style)."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        PatchEmbed = _patch_embed_cls()
        Block = _vit_block_cls()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self._init_pos_embed()

    def _init_pos_embed(self) -> None:
        grid = int(self.patch_embed.num_patches**0.5)
        pe = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        patch_stiffness: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        if patch_stiffness is not None:
            x = x + patch_stiffness.unsqueeze(1)
        x, mask, ids_restore = random_masking(x, mask_ratio)
        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore

    def forward_dense(self, x: torch.Tensor, patch_stiffness: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Full patch sequence (no masking) for detection-style feature extraction."""
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        if patch_stiffness is not None:
            x = x + patch_stiffness.unsqueeze(1)
        cls = self.cls_token + self.pos_embed[:, :1, :]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class MaskedAutoencoderViT(nn.Module):
    """Single-image MAE: composes ``MAEViTEncoder`` + lightweight decoder (for reference pretraining)."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        norm_pix_loss: bool = False,
    ):
        super().__init__()
        Block = _vit_block_cls()

        self.encoder = MAEViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
        )
        num_patches = self.encoder.patch_embed.num_patches
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
        self.norm_pix_loss = norm_pix_loss
        self.in_chans = in_chans
        self.patch_size = patch_size

        self._init_decoder_pos()
        self.apply(self._init_weights)
        nn.init.normal_(self.encoder.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        w = self.encoder.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _init_decoder_pos(self) -> None:
        grid = int(self.encoder.patch_embed.num_patches**0.5)
        dec = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], grid, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(dec).float().unsqueeze(0))

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def patch_embed(self) -> nn.Module:
        return self.encoder.patch_embed

    @property
    def pos_embed(self) -> nn.Parameter:
        return self.encoder.pos_embed

    def forward_encoder(
        self,
        x: torch.Tensor,
        mask_ratio: float,
        patch_stiffness: Optional[torch.Tensor] = None,
    ):
        return self.encoder(x, mask_ratio, patch_stiffness=patch_stiffness)

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x[:, 1:, :]

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = patchify(imgs, self.patch_size, self.in_chans)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        return (loss * mask).sum() / mask.sum()

    def forward(
        self,
        imgs: torch.Tensor,
        mask_ratio: float = 0.75,
        patch_stiffness: Optional[torch.Tensor] = None,
    ):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, patch_stiffness=patch_stiffness)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16(**kwargs) -> MaskedAutoencoderViT:
    return MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def mae_vit_large_patch16(**kwargs) -> MaskedAutoencoderViT:
    return MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def mae_vit_huge_patch14(**kwargs) -> MaskedAutoencoderViT:
    return MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


MODEL_REGISTRY = {
    "mae_vit_base_patch16": mae_vit_base_patch16,
    "mae_vit_large_patch16": mae_vit_large_patch16,
    "mae_vit_huge_patch14": mae_vit_huge_patch14,
}
