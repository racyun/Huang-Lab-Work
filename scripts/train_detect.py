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
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    run_detect(args.config, args.local_config, args.device)


if __name__ == "__main__":
    main()
