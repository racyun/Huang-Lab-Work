from __future__ import annotations

from pathlib import Path

import torch


def load_boxes_txt(path: Path) -> torch.Tensor:
    """Load bounding boxes from a comma-separated text file (one box per line: x1,y1,x2,y2)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Label file not found: {path}")
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        return torch.zeros((0, 4), dtype=torch.float32)
    rows = []
    for line in lines:
        parts = [float(x) for x in line.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Expected 4 comma-separated values per line in {path}, got {len(parts)}")
        rows.append(parts)
    return torch.tensor(rows, dtype=torch.float32)
