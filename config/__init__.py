"""Configuration loaders (YAML + dataclasses)."""

from pathlib import Path
from typing import Any, Optional

import yaml

from config.settings import (
    DatasetConfig,
    DetectionConfig,
    FullConfig,
    MultiMAEConfig,
    PretrainConfig,
    SplitConfig,
    TrainingConfig,
)


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def load_config(
    path: Path,
    local_override: Optional[Path] = None,
) -> FullConfig:
    raw = load_yaml(path)
    if local_override is not None and local_override.is_file():
        raw = merge_dict(raw, load_yaml(local_override))
    return FullConfig.from_dict(raw)
