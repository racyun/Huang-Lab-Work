"""VideoMAE-style masked autoencoder on volumes [B, C, Z, H, W]."""

from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.mae import random_masking
from models.pos_embed import get_3d_sincos_pos_embed


def _vit_block_cls():
    try:
        from timm.layers import Block

        return Block
    except ImportError:
        from timm.models.vision_transformer import Block

        return Block


class VolumeTubeEmbed(nn.Module):
    """3D conv patch embedding: tubes of shape (tublet_z, patch_xy, patch_xy)."""

    def __init__(
        self,
        z_size: int,
        h_size: int,
        w_size: int,
        tublet_z: int,
        patch_xy: int,
        in_chans: int,
        embed_dim: int,
    ):
        super().__init__()
        if z_size % tublet_z or h_size % patch_xy or w_size % patch_xy:
            raise ValueError(
                f"Volume ({z_size},{h_size},{w_size}) must be divisible by "
                f"tube ({tublet_z},{patch_xy},{patch_xy})"
            )
        self.tublet_z = tublet_z
        self.patch_xy = patch_xy
        self.grid_z = z_size // tublet_z
        self.grid_h = h_size // patch_xy
        self.grid_w = w_size // patch_xy
        self.num_patches = self.grid_z * self.grid_h * self.grid_w
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tublet_z, patch_xy, patch_xy),
            stride=(tublet_z, patch_xy, patch_xy),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


def patchify_volume(x: torch.Tensor, tublet_z: int, patch_xy: int) -> torch.Tensor:
    """[B, C, Z, H, W] -> [B, L, tublet_z * patch_xy**2 * C]"""
    B, C, Z, H, W = x.shape
    tz, p = tublet_z, patch_xy
    zt, ht, wt = Z // tz, H // p, W // p
    x = x.reshape(B, C, zt, tz, ht, p, wt, p)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
    return x.reshape(B, zt * ht * wt, tz * p * p * C)


class MAEVolumeEncoder(nn.Module):
    def __init__(
        self,
        z_size: int,
        h_size: int,
        w_size: int,
        tublet_z: int,
        patch_xy: int,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        Block = _vit_block_cls()
        self.patch_embed = VolumeTubeEmbed(z_size, h_size, w_size, tublet_z, patch_xy, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        gz, gh, gw = self.patch_embed.grid_z, self.patch_embed.grid_h, self.patch_embed.grid_w
        pe = get_3d_sincos_pos_embed(embed_dim, gz, gh, gw, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

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
        self.tublet_z = tublet_z
        self.patch_xy = patch_xy
        self.in_chans = in_chans

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


class MaskedAutoencoderVolume(nn.Module):
    def __init__(
        self,
        z_size: int,
        h_size: int,
        w_size: int,
        tublet_z: int,
        patch_xy: int,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        norm_pix_loss: bool = False,
    ):
        super().__init__()
        Block = _vit_block_cls()
        self.encoder = MAEVolumeEncoder(
            z_size=z_size,
            h_size=h_size,
            w_size=w_size,
            tublet_z=tublet_z,
            patch_xy=patch_xy,
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
        gz, gh, gw = self.encoder.patch_embed.grid_z, self.encoder.patch_embed.grid_h, self.encoder.patch_embed.grid_w
        dec = get_3d_sincos_pos_embed(decoder_embed_dim, gz, gh, gw, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(dec).float().unsqueeze(0))

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
        tz, p = tublet_z, patch_xy
        self.decoder_pred = nn.Linear(decoder_embed_dim, tz * p * p * in_chans, bias=True)
        self.norm_pix_loss = norm_pix_loss
        self.in_chans = in_chans
        self.tublet_z = tublet_z
        self.patch_xy = patch_xy

        nn.init.normal_(self.encoder.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.xavier_uniform_(self.encoder.patch_embed.proj.weight)
        if self.encoder.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.encoder.patch_embed.proj.bias)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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

    def forward_loss(self, vol: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        target = patchify_volume(vol, self.tublet_z, self.patch_xy)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        return (loss * mask).sum() / mask.sum()

    def forward(
        self,
        vol: torch.Tensor,
        mask_ratio: float = 0.75,
        patch_stiffness: Optional[torch.Tensor] = None,
    ):
        latent, mask, ids_restore = self.encoder(vol, mask_ratio, patch_stiffness=patch_stiffness)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(vol, pred, mask)
        return loss, pred, mask


def mae_volume_small(**kwargs) -> MaskedAutoencoderVolume:
    return MaskedAutoencoderVolume(
        z_size=kwargs.pop("z_size", 140),
        h_size=kwargs.pop("h_size", 224),
        w_size=kwargs.pop("w_size", 224),
        tublet_z=kwargs.pop("tublet_z", 4),
        patch_xy=kwargs.pop("patch_xy", 16),
        in_chans=kwargs.pop("in_chans", 3),
        embed_dim=kwargs.pop("embed_dim", 384),
        depth=kwargs.pop("depth", 8),
        num_heads=kwargs.pop("num_heads", 6),
        decoder_embed_dim=kwargs.pop("decoder_embed_dim", 384),
        decoder_depth=kwargs.pop("decoder_depth", 4),
        decoder_num_heads=kwargs.pop("decoder_num_heads", 8),
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
