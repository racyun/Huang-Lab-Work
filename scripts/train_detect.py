#!/usr/bin/env python3
"""Fine-tune HuggingFace Deformable-DETR on tissue-chip boxes (COCO-style targets)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.main_detect import run_detect
from utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config" / "default.yaml")
    p.add_argument("--local-config", type=Path, default=None)
    p.add_argument("--device", type=str, default=None)
    # W&B overrides
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    p.add_argument("--wandb-project", type=str, default=None, help="W&B project name.")
    p.add_argument("--wandb-run-name", type=str, default=None, help="W&B run display name.")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    if args.wandb:
        from config import load_config
        cfg = load_config(args.config, args.local_config)
        cfg.wandb.enabled = True

    run_detect(
        args.config,
        args.local_config,
        args.device,
        wandb_run_name=args.wandb_run_name,
        wandb_project=args.wandb_project,
    )


if __name__ == "__main__":
    main()
