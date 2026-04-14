#!/usr/bin/env python3
"""Multi-encoder MAE pretraining (focused + hybrid 2D MAE, z-stack volume MAE, stiffness MLP)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from config import load_config
from data.collate import pretrain_collate
from data.pretrain_dataset import build_pretrain_dataset
from models.mae import MODEL_REGISTRY, MaskedAutoencoderViT
from models.multi_mae import build_multi_encoder_mae
from training.main_pretrain import run_pretrain
from utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config" / "default.yaml")
    p.add_argument("--local-config", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None, help="Resume from multimae checkpoint .pth")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Single backward pass on random tensors shaped like the multi-encoder inputs.",
    )
    p.add_argument(
        "--inspect-data",
        action="store_true",
        help="Print one pretrain batch from config (requires valid data paths).",
    )
    p.add_argument(
        "--train",
        action="store_true",
        help="Run full pretrain loop from config (epochs, dataloader, logging).",
    )
    p.add_argument("--mae-model", type=str, default="mae_vit_base_patch16", choices=list(MODEL_REGISTRY.keys()))
    # W&B overrides (also settable via config/local.yaml wandb: section)
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument("--wandb-project", type=str, default=None, help="W&B project name.")
    p.add_argument("--wandb-run-name", type=str, default=None, help="W&B run display name.")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = load_config(args.config, args.local_config)

    if args.inspect_data:
        ds = build_pretrain_dataset(cfg)
        from torch.utils.data import DataLoader

        loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=pretrain_collate, num_workers=0)
        batch = next(iter(loader))
        print("zstack (B,C,Z,H,W):", batch["zstack"].shape)
        print("focused:", batch["focused"].shape)
        print("hybrid:", batch["hybrid"].shape)
        print("stiffness:", batch["stiffness"].shape)
        return

    if args.smoke:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        mm = cfg.multi_mae
        model = build_multi_encoder_mae(mm).to(device)
        B = 2
        batch = {
            "zstack": torch.randn(B, mm.in_chans, mm.volume_z, mm.volume_h, mm.volume_w, device=device),
            "focused": torch.randn(B, mm.in_chans, mm.image_size, mm.image_size, device=device),
            "hybrid": torch.randn(B, mm.in_chans, mm.image_size, mm.image_size, device=device),
            "stiffness": torch.tensor([[900.0], [5.0]], device=device),
        }
        loss, parts = model(batch)
        loss.backward()
        print("multi-mae smoke OK loss=", float(loss.detach()), "parts=", {k: float(v) for k, v in parts.items()})
        return

    if args.train:
        if args.wandb:
            cfg.wandb.enabled = True
        run_pretrain(
            args.config,
            args.local_config,
            args.resume,
            args.device,
            wandb_run_name=args.wandb_run_name,
            wandb_project=args.wandb_project,
        )
        return

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_ctor = MODEL_REGISTRY[args.mae_model]
    model: MaskedAutoencoderViT = model_ctor()
    model.to(device)
    x = torch.randn(2, 3, 224, 224, device=device)
    loss, _, _ = model(x, mask_ratio=cfg.pretrain.mask_ratio)
    loss.backward()
    print("single-mae baseline smoke OK loss=", float(loss.detach()))


if __name__ == "__main__":
    main()
