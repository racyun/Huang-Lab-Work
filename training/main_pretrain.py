from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from config import load_config
from data.collate import pretrain_collate
from data.pretrain_dataset import build_pretrain_dataset
from models.multi_mae import build_multi_encoder_mae, param_groups_with_head_lrs
from training.engine_pretrain import train_one_epoch
from training.lr_sched import set_epoch_learning_rates
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.wandb_utils import (
    finish_wandb,
    init_wandb,
    log_pretrain_epoch,
    watch_model,
)


def run_pretrain(
    config_path: Path,
    local_config: Optional[Path],
    resume: Optional[Path],
    device_str: Optional[str],
    wandb_run_name: Optional[str] = None,
    wandb_project: Optional[str] = None,
) -> None:
    cfg = load_config(config_path, local_config)

    # Allow CLI overrides for W&B settings
    if wandb_project:
        cfg.wandb.project = wandb_project

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.training.seed)

    ds = build_pretrain_dataset(cfg)
    loader = DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=pretrain_collate,
        drop_last=True,
    )

    model = build_multi_encoder_mae(cfg.multi_mae).to(device)
    if resume is not None:
        load_checkpoint(resume, model, strict=False)

    groups = param_groups_with_head_lrs(
        model,
        base_lr=cfg.training.lr,
        lr_mult_focused=cfg.multi_mae.lr_mult_focused,
        lr_mult_hybrid=cfg.multi_mae.lr_mult_hybrid,
        lr_mult_volume=cfg.multi_mae.lr_mult_volume,
        lr_mult_stiffness=cfg.multi_mae.lr_mult_stiffness,
        weight_decay=cfg.training.weight_decay,
    )
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.training.amp and device.type == "cuda")

    out_dir = Path(cfg.training.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── W&B init ──────────────────────────────────────────────────────────────
    init_wandb(cfg, mode="pretrain", run_name_override=wandb_run_name)
    if cfg.wandb.watch_model:
        watch_model(model, log_freq=cfg.wandb.log_freq * 10)

    global_step = 0

    try:
        for epoch in range(cfg.training.epochs):
            set_epoch_learning_rates(
                optimizer,
                epoch,
                epochs=cfg.training.epochs,
                warmup_epochs=cfg.training.warmup_epochs,
                min_lr=cfg.training.min_lr,
                peak_lr=cfg.training.lr,
            )

            stats, global_step = train_one_epoch(
                model, loader, optimizer, device, cfg, scaler, epoch, global_step
            )

            # Console + JSONL log
            print(json.dumps({"train": stats}, indent=2))
            with open(out_dir / "pretrain_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(stats) + "\n")

            # W&B epoch log
            log_pretrain_epoch(epoch, stats, optimizer, global_step)

            if (epoch + 1) % max(1, cfg.training.epochs // 10) == 0 or epoch + 1 == cfg.training.epochs:
                save_checkpoint(
                    out_dir / f"multimae_epoch_{epoch+1}.pth",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    extra={"config": str(config_path)},
                )
    finally:
        finish_wandb()

    print("Pretrain finished.", file=sys.stderr)
