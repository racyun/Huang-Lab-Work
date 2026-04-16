"""Optional per-sample disk cache (partial materialization without full-dataset npy trees)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch.utils.data import Dataset


def _safe_key(split: str, well_id: str) -> str:
    raw = f"{split}_{well_id}".encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class DiskTensorCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.root / f"{key}.pt"

    def get(self, key: str, factory: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        p = self.path(key)
        if p.is_file():
            return _torch_load(p)
        data = factory()
        torch.save(data, p)
        return data


class CachedTissueChipDataset(Dataset):
    """Caches ``__getitem__`` dicts (tensors + small metadata strings) to ``.pt`` files."""

    def __init__(
        self,
        inner: Dataset,
        cache_dir: Path,
        key_from_sample: Callable[[dict[str, Any]], str],
    ):
        self.inner = inner
        self.cache = DiskTensorCache(cache_dir)
        self.key_from_sample = key_from_sample

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raw = self.inner[idx]

        def factory() -> dict[str, Any]:
            return raw

        key = self.key_from_sample(raw)
        return self.cache.get(key, factory)


def default_cache_key(sample: dict[str, Any]) -> str:
    return _safe_key(str(sample.get("split", "na")), str(sample.get("well_id", "na")))
