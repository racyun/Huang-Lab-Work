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
    """Caches ``__getitem__`` dicts (tensors + small metadata strings) to ``.pt`` files.

    If ``key_from_idx`` is provided, the cache is checked **before** the inner
    dataset is touched, so a cache hit avoids the (potentially very slow) inner
    load entirely. Without it, the cache only avoids re-running post-processing —
    the inner load still runs every time.
    """

    def __init__(
        self,
        inner: Dataset,
        cache_dir: Path,
        key_from_sample: Callable[[dict[str, Any]], str],
        key_from_idx: Optional[Callable[[int], str]] = None,
    ):
        self.inner = inner
        self.cache = DiskTensorCache(cache_dir)
        self.key_from_sample = key_from_sample
        self.key_from_idx = key_from_idx

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Fast path: cache hit without touching the inner dataset.
        if self.key_from_idx is not None:
            key = self.key_from_idx(idx)
            cache_path = self.cache.path(key)
            if cache_path.is_file():
                return _torch_load(cache_path)
            # Cache miss: fall through to load + write below.
            raw = self.inner[idx]
            torch.save(raw, cache_path)
            return raw

        # Legacy slow path: always loads from inner, only saves post-processing.
        raw = self.inner[idx]
        key = self.key_from_sample(raw)
        return self.cache.get(key, lambda: raw)


def default_cache_key(sample: dict[str, Any]) -> str:
    return _safe_key(str(sample.get("split", "na")), str(sample.get("well_id", "na")))
