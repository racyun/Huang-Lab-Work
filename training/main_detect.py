from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from config import load_config
from data.detection_dataset import build_detection_dataset, collate_detection_batch
from training.engine_detect import eval_one_epoch_detect, train_one_epoch_detect
from utils.checkpoint import save_checkpoint
from utils.wandb_utils import finish_wandb, init_wandb, log_detect_combined


def _log(msg: str) -> None:
    """Timestamped heartbeat print so users see progress during long startup phases."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_detect(
    config_path: Path,
    local_config: Optional[Path],
    device_str: Optional[str],
    wandb_run_name: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_enabled: bool = False,
) -> None:
    try:
        from transformers import AutoModelForObjectDetection
    except ImportError as e:
        raise SystemExit(
            "Install transformers for detection: pip install transformers accelerate"
        ) from e

    _log("Loading config...")
    cfg = load_config(config_path, local_config)

    if wandb_enabled:
        cfg.wandb.enabled = True
    if wandb_project:
        cfg.wandb.project = wandb_project

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    det = cfg.detection
    _log(f"Device: {device}  |  batch_size={det.batch_size}  num_workers={det.num_workers}  epochs={det.epochs}")

    if det.mae_encoder_ckpt:
        warnings.warn(
            "detection.mae_encoder_ckpt is not applied to the default ResNet backbone. "
            "Use load_mae_encoder_from_multimae_ckpt on a custom ViT backbone module when you wire it in.",
            UserWarning,
            stacklevel=1,
        )

    _log(f"Loading detection model from HuggingFace: {det.hf_model} (downloads ~150 MB on first run)...")
    t0 = time.time()
    model = AutoModelForObjectDetection.from_pretrained(
        det.hf_model,
        num_labels=det.num_classes,
        ignore_mismatched_sizes=True,
    ).to(device)
    _log(f"Model loaded in {time.time() - t0:.1f}s")

    _log("Building detection dataset (scanning wells/files; uses 'detect' cache subdir)...")
    t0 = time.time()
    ds = build_detection_dataset(cfg)
    _log(f"Dataset built: {len(ds)} samples in {time.time() - t0:.1f}s")

    loader = DataLoader(
        ds,
        batch_size=det.batch_size,
        shuffle=True,
        num_workers=det.num_workers,
        collate_fn=collate_detection_batch,
        pin_memory=device.type == "cuda",
    )
    _log(f"DataLoader ready: {len(loader)} steps/epoch")

    optimizer = torch.optim.AdamW(model.parameters(), lr=det.lr, weight_decay=det.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=det.amp and device.type == "cuda")

    out_dir = Path(cfg.training.output_dir) / "detect"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── W&B init ──────────────────────────────────────────────────────────────
    _log("Initialising W&B...")
    init_wandb(cfg, mode="detect", run_name_override=wandb_run_name)
    _log("W&B ready.")

    start_epoch = 0
    global_step = 0
    if det.resume_ckpt:
        from utils.checkpoint import load_checkpoint
        ckpt = load_checkpoint(Path(det.resume_ckpt), model, optimizer)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        print(f"Resumed from {det.resume_ckpt} — starting at epoch {start_epoch}", file=sys.stderr)

    conf_threshold = getattr(det, "conf_threshold", 0.5)

    try:
        for epoch in range(start_epoch, det.epochs):
            _log(f"=== Epoch {epoch+1}/{det.epochs} starting (train) ===")
            t_epoch = time.time()
            epoch_loss, global_step = train_one_epoch_detect(
                model, loader, optimizer, device, cfg, scaler, global_step
            )
            _log(f"=== Epoch {epoch+1} train done in {time.time() - t_epoch:.1f}s, loss={epoch_loss:.4f} ===")
            current_lr = optimizer.param_groups[0]["lr"]

            # ── Evaluation ────────────────────────────────────────────────────
            _log(f"--- Epoch {epoch+1} eval starting ---")
            t_eval = time.time()
            eval_metrics = eval_one_epoch_detect(model, loader, device, conf_threshold)
            _log(f"--- Epoch {epoch+1} eval done in {time.time() - t_eval:.1f}s "
                 f"AP50={eval_metrics.get('AP50', 0):.4f} mAP={eval_metrics.get('mAP', 0):.4f} ---")

            row = {"epoch": epoch, "loss": epoch_loss, **eval_metrics}
            print(json.dumps(row), flush=True)
            with open(out_dir / "detect_log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            # Single W&B call so all metrics share the same step point
            log_detect_combined(epoch, epoch_loss, current_lr, eval_metrics, global_step)

            # Save every epoch for first 5, then every epochs//10
            save_every = max(1, det.epochs // 10)
            if epoch < 5 or (epoch + 1) % save_every == 0 or epoch + 1 == det.epochs:
                save_checkpoint(
                    out_dir / f"detector_epoch_{epoch+1}.pth",
                    model=model, optimizer=optimizer, epoch=epoch,
                    extra={"global_step": global_step},
                )
    finally:
        finish_wandb()

    save_checkpoint(out_dir / "detector_final.pth", model=model, optimizer=optimizer, epoch=det.epochs - 1)
    print("Detection training finished.", file=sys.stderr)
