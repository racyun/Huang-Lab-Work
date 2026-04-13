"""Three MAE heads (focused, hybrid, z-stack volume) with shared stiffness MLP and separate losses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn

from models.mae import MaskedAutoencoderViT
from models.mae_volume import MaskedAutoencoderVolume
from models.stiffness import StiffnessMLP, normalize_kpa

if TYPE_CHECKING:
    from config.settings import MultiMAEConfig


class MultiEncoderMAE(nn.Module):
    def __init__(
        self,
        mae_focused: MaskedAutoencoderViT,
        mae_hybrid: MaskedAutoencoderViT,
        mae_volume: MaskedAutoencoderVolume,
        embed_dim: int,
        stiffness_hidden: int = 64,
        kpa_input_mode: str = "divide",
        kpa_divisor: float = 900.0,
        loss_weight_focused: float = 1.0,
        loss_weight_hybrid: float = 1.0,
        loss_weight_volume: float = 1.0,
    ):
        super().__init__()
        self.mae_focused = mae_focused
        self.mae_hybrid = mae_hybrid
        self.mae_volume = mae_volume
        self.stiffness_mlp = StiffnessMLP(embed_dim, hidden_dim=stiffness_hidden)
        self.kpa_input_mode = kpa_input_mode
        self.kpa_divisor = kpa_divisor
        self.w_f = loss_weight_focused
        self.w_h = loss_weight_hybrid
        self.w_z = loss_weight_volume

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        mask_ratio_focused: float = 0.75,
        mask_ratio_hybrid: float = 0.75,
        mask_ratio_volume: float = 0.75,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        kpa = normalize_kpa(batch["stiffness"], self.kpa_input_mode, self.kpa_divisor)
        stiff_emb = self.stiffness_mlp(kpa)

        lf, _, _ = self.mae_focused(batch["focused"], mask_ratio_focused, patch_stiffness=stiff_emb)
        lh, _, _ = self.mae_hybrid(batch["hybrid"], mask_ratio_hybrid, patch_stiffness=stiff_emb)
        lz, _, _ = self.mae_volume(batch["zstack"], mask_ratio_volume, patch_stiffness=stiff_emb)

        total = self.w_f * lf + self.w_h * lh + self.w_z * lz
        parts = {"loss_focused": lf.detach(), "loss_hybrid": lh.detach(), "loss_volume": lz.detach()}
        return total, parts


def build_multi_encoder_mae(cfg: "MultiMAEConfig") -> MultiEncoderMAE:
    """Build default multi-MAE from ``MultiMAEConfig``."""
    p = cfg
    vol = MaskedAutoencoderVolume(
        z_size=p.volume_z,
        h_size=p.volume_h,
        w_size=p.volume_w,
        tublet_z=p.tublet_z,
        patch_xy=p.volume_patch_xy,
        in_chans=p.in_chans,
        embed_dim=p.embed_dim,
        depth=p.vol_depth,
        num_heads=p.vol_num_heads,
        decoder_embed_dim=p.decoder_embed_dim,
        decoder_depth=p.vol_dec_depth,
        decoder_num_heads=p.vol_dec_num_heads,
        norm_pix_loss=p.norm_pix_loss,
    )
    common = dict(
        img_size=p.image_size,
        patch_size=p.patch_xy_2d,
        in_chans=p.in_chans,
        embed_dim=p.embed_dim,
        depth=p.twod_depth,
        num_heads=p.twod_num_heads,
        decoder_embed_dim=p.decoder_embed_dim,
        decoder_depth=p.twod_dec_depth,
        decoder_num_heads=p.twod_dec_num_heads,
        norm_pix_loss=p.norm_pix_loss,
    )
    mf = MaskedAutoencoderViT(**common)
    mh = MaskedAutoencoderViT(**common)
    return MultiEncoderMAE(
        mf,
        mh,
        vol,
        embed_dim=p.embed_dim,
        stiffness_hidden=p.stiffness_hidden,
        kpa_input_mode=p.kpa_input_mode,
        kpa_divisor=p.kpa_divisor,
        loss_weight_focused=p.loss_weight_focused,
        loss_weight_hybrid=p.loss_weight_hybrid,
        loss_weight_volume=p.loss_weight_volume,
    )


def param_groups_with_head_lrs(
    model: MultiEncoderMAE,
    base_lr: float,
    lr_mult_focused: float,
    lr_mult_hybrid: float,
    lr_mult_volume: float,
    lr_mult_stiffness: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    return [
        {"params": model.mae_focused.parameters(), "lr": base_lr * lr_mult_focused, "weight_decay": weight_decay},
        {"params": model.mae_hybrid.parameters(), "lr": base_lr * lr_mult_hybrid, "weight_decay": weight_decay},
        {"params": model.mae_volume.parameters(), "lr": base_lr * lr_mult_volume, "weight_decay": weight_decay},
        {"params": model.stiffness_mlp.parameters(), "lr": base_lr * lr_mult_stiffness, "weight_decay": weight_decay},
    ]
