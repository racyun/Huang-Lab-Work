"""Visualize pretrain MAE reconstruction: original → masked → reconstructed.

Loads a saved pretrain checkpoint (multimae_epoch_*.pth) and runs a forward
pass on a few samples to produce side-by-side figures showing the focused and
hybrid modality reconstructions.  No retraining required.

Usage
-----
    python3 scripts/visualize_pretrain.py \\
        --checkpoint /content/outputs/pretrain/multimae_epoch_50.pth \\
        --config config/default.yaml \\
        --local-config config/colab.yaml \\
        --num-samples 6 \\
        --mask-ratio 0.75 \\
        --output-dir /content/drive/MyDrive/pretrain_reconstructions
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import load_config
from data.collate import pretrain_collate
from data.pretrain_dataset import build_pretrain_dataset
from models.mae import patchify, unpatchify
from models.multi_mae import build_multi_encoder_mae
from models.stiffness import normalize_kpa
from utils.checkpoint import load_checkpoint


def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _build_masked_vis(imgs: torch.Tensor, mask: torch.Tensor, patch_size: int, in_chans: int) -> torch.Tensor:
    """Return the original image with masked patches set to 0.5 (mid-gray)."""
    patches = patchify(imgs, patch_size, in_chans)           # [N, L, p²C]
    mask_exp = mask.unsqueeze(-1).expand_as(patches)         # [N, L, p²C]
    patches_vis = patches.clone()
    patches_vis[mask_exp.bool()] = 0.5
    return unpatchify(patches_vis, patch_size, in_chans)     # [N, C, H, W]


def _build_recon_vis(
    imgs: torch.Tensor,
    pred: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int,
    in_chans: int,
    norm_pix_loss: bool,
) -> torch.Tensor:
    """Paste reconstructed patches onto original (visible patches kept from original)."""
    patches_orig = patchify(imgs, patch_size, in_chans)      # [N, L, p²C]

    # Un-normalise pred if norm_pix_loss was used during training
    if norm_pix_loss:
        mean = patches_orig.mean(dim=-1, keepdim=True)
        var = patches_orig.var(dim=-1, keepdim=True)
        pred = pred * (var + 1e-6) ** 0.5 + mean

    mask_exp = mask.unsqueeze(-1).expand_as(patches_orig)
    recon_patches = torch.where(mask_exp.bool(), pred, patches_orig)
    return unpatchify(recon_patches, patch_size, in_chans).clamp(0, 1)


def _show_row(axes, imgs: torch.Tensor, masked: torch.Tensor, recon: torch.Tensor, label: str) -> None:
    """Fill three pre-created axes with original / masked / reconstructed."""
    for ax, t, title in zip(
        axes,
        [imgs, masked, recon],
        [f"{label}\noriginal", "masked input", "reconstructed"],
    ):
        ax.imshow(t.permute(1, 2, 0).clamp(0, 1).cpu().numpy())
        ax.set_title(title, fontsize=9)
        ax.axis("off")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to multimae_epoch_*.pth")
    p.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    p.add_argument("--local-config", type=Path, default=None)
    p.add_argument("--num-samples", type=int, default=6)
    p.add_argument("--mask-ratio", type=float, default=0.75,
                   help="Masking ratio (should match training, default 0.75)")
    p.add_argument("--output-dir", type=Path,
                   default=Path("/content/outputs/pretrain/visualizations"))
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Loading config from {args.config}...")
    cfg = load_config(args.config, args.local_config)

    print("Building pretrain dataset...")
    ds = build_pretrain_dataset(cfg)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=pretrain_collate)

    print("Building model...")
    model = build_multi_encoder_mae(cfg.multi_mae).to(device)

    print(f"Loading weights from {args.checkpoint}...")
    load_checkpoint(args.checkpoint, model, strict=True)
    model.eval()

    mm = cfg.multi_mae
    norm_pix = mm.norm_pix_loss
    patch_size = mm.patch_xy_2d
    in_chans = mm.in_chans

    print(f"Rendering {args.num_samples} samples → {args.output_dir}")

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= args.num_samples:
                break

            batch = _to_device(batch, device)
            kpa = normalize_kpa(batch["stiffness"], mm.kpa_input_mode, mm.kpa_divisor)
            stiff_emb = model.stiffness_mlp(kpa)

            focused_img = batch["focused"]   # [1, 3, H, W]
            hybrid_img = batch["hybrid"]     # [1, 3, H, W]

            # Forward through each 2D MAE head
            _, pred_f, mask_f = model.mae_focused(
                focused_img, args.mask_ratio, patch_stiffness=stiff_emb
            )
            _, pred_h, mask_h = model.mae_hybrid(
                hybrid_img, args.mask_ratio, patch_stiffness=stiff_emb
            )

            # Build visualisation tensors (all on CPU, single batch item [0])
            f_orig   = focused_img[0].cpu()
            h_orig   = hybrid_img[0].cpu()
            f_masked = _build_masked_vis(focused_img, mask_f, patch_size, in_chans)[0].cpu()
            h_masked = _build_masked_vis(hybrid_img,  mask_h, patch_size, in_chans)[0].cpu()
            f_recon  = _build_recon_vis(focused_img, pred_f, mask_f, patch_size, in_chans, norm_pix)[0].cpu()
            h_recon  = _build_recon_vis(hybrid_img,  pred_h, mask_h, patch_size, in_chans, norm_pix)[0].cpu()

            # 2 rows (focused / hybrid) × 3 columns (original / masked / recon)
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            well_id = f"sample_{idx}"
            fig.suptitle(f"Sample {idx}  |  well={well_id}  |  mask_ratio={args.mask_ratio}",
                         fontsize=11)

            _show_row(axes[0], f_orig, f_masked, f_recon, "focused")
            _show_row(axes[1], h_orig, h_masked, h_recon, "hybrid")

            plt.tight_layout()
            out_path = args.output_dir / f"sample_{idx:03d}.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  [{idx + 1}/{args.num_samples}] {out_path}")

    print(f"\nDone. {args.num_samples} images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
