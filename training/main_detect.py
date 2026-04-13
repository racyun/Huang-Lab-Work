from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import load_config
from data.detection_dataset import build_detection_dataset, collate_detection_batch
from training.engine_detect import train_one_epoch_detect
from utils.checkpoint import save_checkpoint


def run_detect(
    config_path: Path,
    local_config: Path | None,
    device_str: str | None,
) -> None:
    try:
        from transformers import AutoModelForObjectDetection
    except ImportError as e:
        raise SystemExit(
            "Install transformers for detection: pip install transformers accelerate"
        ) from e

    cfg = load_config(config_path, local_config)
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

    for epoch in range(det.epochs):
        loss = train_one_epoch_detect(model, loader, optimizer, device, cfg, scaler)
        row = {"epoch": epoch, "loss": loss}
        print(json.dumps(row))
        with open(out_dir / "detect_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    save_checkpoint(out_dir / "detector_final.pth", model=model, optimizer=optimizer, epoch=det.epochs - 1)
    print("Detection training finished.", file=sys.stderr)
