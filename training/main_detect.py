from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from config import load_config
from data.detection_dataset import build_detection_dataset, collate_detection_batch
from training.engine_detect import train_one_epoch_detect
from utils.checkpoint import save_checkpoint
from utils.wandb_utils import finish_wandb, init_wandb, log_detect_epoch


def run_detect(
    config_path: Path,
    local_config: Optional[Path],
    device_str: Optional[str],
    wandb_run_name: Optional[str] = None,
    wandb_project: Optional[str] = None,
) -> None:
    try:
        from transformers import AutoModelForObjectDetection
    except ImportError as e:
        raise SystemExit(
            "Install transformers for detection: pip install transformers accelerate"
        ) from e

    cfg = load_config(config_path, local_config)

    if wandb_project:
        cfg.wandb.project = wandb_project

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    det = cfg.detection

    if det.mae_encoder_ckpt:
        warnings.warn(
            "detection.mae_encoder_ckpt is not applied to the default ResNet backbone. "
            "Use load_mae_encoder_from_multimae_ckpt on a custom ViT backbone module when you wire it in.",
            UserWarning,
            stacklevel=1,
        )

    model = AutoModelForObjectDetection.from_pretrained(
        det.hf_model,
        num_labels=det.num_classes,
        ignore_mismatched_sizes=True,
    ).to(device)

    ds = build_detection_dataset(cfg)
    loader = DataLoader(
        ds,
        batch_size=det.batch_size,
        shuffle=True,
        num_workers=det.num_workers,
        collate_fn=collate_detection_batch,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=det.lr, weight_decay=det.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=det.amp and device.type == "cuda")

    out_dir = Path(cfg.training.output_dir) / "detect"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── W&B init ──────────────────────────────────────────────────────────────
    init_wandb(cfg, mode="detect", run_name_override=wandb_run_name)

    global_step = 0

    try:
        for epoch in range(det.epochs):
            epoch_loss, global_step = train_one_epoch_detect(
                model, loader, optimizer, device, cfg, scaler, global_step
            )
            current_lr = optimizer.param_groups[0]["lr"]

            row = {"epoch": epoch, "loss": epoch_loss}
            print(json.dumps(row))
            with open(out_dir / "detect_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            log_detect_epoch(epoch, epoch_loss, current_lr, global_step)
    finally:
        finish_wandb()

    save_checkpoint(out_dir / "detector_final.pth", model=model, optimizer=optimizer, epoch=det.epochs - 1)
    print("Detection training finished.", file=sys.stderr)
