"""
Integration tests for the multi-encoder MAE + detection pipeline.

All tests use a small synthetic dataset (3 wells, 64x64 images, tiny model)
so they run in ~1-2 minutes on CPU with no real data required.

Run from the repo root:
    python -m pytest tests/ -v
or:
    python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_tif(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _make_fake_dataset(root: Path) -> None:
    """Write a minimal fake well layout: 2 wells at 900 kPa, 1 at 5 kPa."""
    for kpa, wids in [("900kpa", ["W001", "W002"]), ("5kpa", ["W003"])]:
        for wid in wids:
            for z in range(140):
                _make_tif(root / kpa / "zstack" / wid / "P00001" / f"z{z:03d}.tif")
            _make_tif(root / kpa / "focused" / wid / f"{wid}_focus_stacked.tif")
            _make_tif(root / kpa / "hybrid" / f"hybrid_results_{wid}" / "hybrid.tif")
            lbl = root / kpa / "labels" / f"{wid}.txt"
            lbl.parent.mkdir(parents=True, exist_ok=True)
            lbl.write_text("10.0,20.0,50.0,60.0\n5.0,5.0,30.0,30.0\n")


def _make_cfg_dict(root: Path, output_dir: Path, cache_dir: Path | None = None) -> dict:
    return {
        "dataset": {
            "well_prefix": "W",
            "well_count": 10,
            "zstack_subdir": "P00001",
            "expected_z_slices": 140,
            "focus_filename_glob": "*focus_stacked*.tif",
            "hybrid_folder_template": "hybrid_results_{well_id}",
            "resize": {"zstack": None, "focused": None, "hybrid": None},
            "cache_dir": str(cache_dir) if cache_dir else None,
            "splits": [
                {
                    "name": "900kpa",
                    "stiffness_kpa": 900.0,
                    "zstack_root": str(root / "900kpa" / "zstack"),
                    "focused_root": str(root / "900kpa" / "focused"),
                    "hybrid_root": str(root / "900kpa" / "hybrid"),
                    "labels_root": str(root / "900kpa" / "labels"),
                },
                {
                    "name": "5kpa",
                    "stiffness_kpa": 5.0,
                    "zstack_root": str(root / "5kpa" / "zstack"),
                    "focused_root": str(root / "5kpa" / "focused"),
                    "hybrid_root": str(root / "5kpa" / "hybrid"),
                    "labels_root": str(root / "5kpa" / "labels"),
                },
            ],
        },
        "training": {
            "output_dir": str(output_dir),
            "seed": 0,
            "batch_size": 2,
            "epochs": 2,
            "num_workers": 0,
            "lr": 1.5e-4,
            "weight_decay": 0.05,
            "warmup_epochs": 1,
            "amp": False,
            "grad_clip": 1.0,
            "min_lr": 0.0,
        },
        "pretrain": {"image_size": 64, "mask_ratio": 0.75},
        "multi_mae": {
            "image_size": 64,
            "patch_xy_2d": 16,
            "volume_z": 140,
            "volume_h": 64,
            "volume_w": 64,
            "tublet_z": 4,
            "volume_patch_xy": 16,
            "in_chans": 3,
            "embed_dim": 96,
            "twod_depth": 2,
            "twod_num_heads": 3,
            "vol_depth": 2,
            "vol_num_heads": 3,
            "decoder_embed_dim": 96,
            "twod_dec_depth": 2,
            "twod_dec_num_heads": 3,
            "vol_dec_depth": 2,
            "vol_dec_num_heads": 3,
            "norm_pix_loss": False,
            "stiffness_hidden": 32,
            "kpa_input_mode": "divide",
            "kpa_divisor": 900.0,
            "loss_weight_focused": 1.0,
            "loss_weight_hybrid": 1.0,
            "loss_weight_volume": 1.0,
            "mask_ratio_focused": 0.75,
            "mask_ratio_hybrid": 0.75,
            "mask_ratio_volume": 0.75,
            "lr_mult_focused": 1.0,
            "lr_mult_hybrid": 1.0,
            "lr_mult_volume": 1.0,
            "lr_mult_stiffness": 1.0,
        },
        "detection": {
            "hf_model": "SenseTime/deformable-detr",
            "num_classes": 2,
            "train_image_short_side": 64,
            "max_train_samples": 3,
            "image_mode": "focused",
            "lr": 2e-5,
            "weight_decay": 1e-4,
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "amp": False,
            "mae_encoder_ckpt": None,
            "mae_encoder_prefix": "mae_focused.encoder.",
        },
    }


# Shared fixtures
@pytest.fixture(scope="module")
def fake_data():
    """Create fake dataset once for all tests in this module."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_fake_dataset(root)
        yield root


@pytest.fixture(scope="module")
def cfg_path(fake_data, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    cfg_dict = _make_cfg_dict(fake_data, out)
    p = out / "test_cfg.yaml"
    p.write_text(yaml.dump(cfg_dict))
    return p


# ── 1. config ─────────────────────────────────────────────────────────────────

def test_config_loads(cfg_path):
    from config import load_config
    cfg = load_config(cfg_path)
    assert cfg.multi_mae.embed_dim == 96
    assert len(cfg.dataset.splits) == 2
    assert cfg.dataset.splits[0].stiffness_kpa == 900.0


# ── 2. dataset ────────────────────────────────────────────────────────────────

def test_dataset_length_and_keys(cfg_path):
    from config import load_config
    from data.pretrain_dataset import build_pretrain_dataset

    cfg = load_config(cfg_path)
    ds = build_pretrain_dataset(cfg)
    assert len(ds) == 3  # 2 wells at 900 kPa + 1 at 5 kPa
    sample = ds[0]
    assert set(sample.keys()) >= {"zstack", "focused", "hybrid", "stiffness"}


def test_dataset_tensor_shapes(cfg_path):
    from config import load_config
    from data.pretrain_dataset import build_pretrain_dataset

    cfg = load_config(cfg_path)
    ds = build_pretrain_dataset(cfg)
    s = ds[0]
    # zstack: [Z, C, H, W]
    assert s["zstack"].shape == (140, 3, 64, 64)
    # 2D views: [C, H, W]
    assert s["focused"].shape == (3, 64, 64)
    assert s["hybrid"].shape == (3, 64, 64)
    # stiffness scalar
    assert s["stiffness"].shape == (1,)


def test_pretrain_collate_shapes(cfg_path):
    """DataLoader with pretrain_collate should produce [B,C,Z,H,W] for zstack."""
    from config import load_config
    from data.collate import pretrain_collate
    from data.pretrain_dataset import build_pretrain_dataset
    from torch.utils.data import DataLoader

    cfg = load_config(cfg_path)
    ds = build_pretrain_dataset(cfg)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0,
                        collate_fn=pretrain_collate)
    batch = next(iter(loader))
    assert batch["zstack"].shape == (2, 3, 140, 64, 64)
    assert batch["focused"].shape == (2, 3, 64, 64)
    assert batch["hybrid"].shape == (2, 3, 64, 64)
    assert batch["stiffness"].shape == (2, 1)


def test_stiffness_values(cfg_path):
    """Stiffness tensors should only contain 5.0 or 900.0 kPa."""
    from config import load_config
    from data.pretrain_dataset import build_pretrain_dataset

    cfg = load_config(cfg_path)
    ds = build_pretrain_dataset(cfg)
    kpas = {ds[i]["stiffness"].item() for i in range(len(ds))}
    assert kpas <= {5.0, 900.0}


# ── 3. boxes ──────────────────────────────────────────────────────────────────

def test_boxes_load():
    from data.boxes import load_boxes_txt

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("10.0,20.0,50.0,60.0\n5.0,5.0,30.0,30.0\n")
        p = Path(f.name)
    boxes = load_boxes_txt(p)
    assert boxes.shape == (2, 4)
    assert boxes[0].tolist() == [10.0, 20.0, 50.0, 60.0]
    p.unlink()


def test_boxes_empty():
    from data.boxes import load_boxes_txt

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("")
        p = Path(f.name)
    boxes = load_boxes_txt(p)
    assert boxes.shape == (0, 4)
    p.unlink()


# ── 4. model ──────────────────────────────────────────────────────────────────

def test_multi_encoder_forward(cfg_path):
    from config import load_config
    from data.collate import pretrain_collate
    from data.pretrain_dataset import build_pretrain_dataset
    from models.multi_mae import build_multi_encoder_mae
    from torch.utils.data import DataLoader

    cfg = load_config(cfg_path)
    ds = build_pretrain_dataset(cfg)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0,
                        collate_fn=pretrain_collate)
    batch = next(iter(loader))

    model = build_multi_encoder_mae(cfg.multi_mae)
    loss, parts = model(batch)
    assert loss.item() > 0
    assert set(parts.keys()) == {"loss_focused", "loss_hybrid", "loss_volume"}
    for v in parts.values():
        assert v.item() > 0


def test_multi_encoder_backward(cfg_path):
    from config import load_config
    from data.collate import pretrain_collate
    from data.pretrain_dataset import build_pretrain_dataset
    from models.multi_mae import build_multi_encoder_mae
    from torch.utils.data import DataLoader

    cfg = load_config(cfg_path)
    ds = build_pretrain_dataset(cfg)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0,
                        collate_fn=pretrain_collate)
    batch = next(iter(loader))

    model = build_multi_encoder_mae(cfg.multi_mae)
    loss, _ = model(batch)
    loss.backward()
    # At least some parameters should have gradients
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad


def test_stiffness_conditions_output(cfg_path):
    """Same image with different stiffness values should produce different losses."""
    from config import load_config
    from data.collate import pretrain_collate
    from data.pretrain_dataset import build_pretrain_dataset
    from models.multi_mae import build_multi_encoder_mae
    from torch.utils.data import DataLoader

    cfg = load_config(cfg_path)
    ds = build_pretrain_dataset(cfg)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=pretrain_collate)
    batch = next(iter(loader))

    model = build_multi_encoder_mae(cfg.multi_mae)
    model.eval()
    with torch.no_grad():
        batch_low = {**batch, "stiffness": torch.tensor([[5.0]])}
        batch_high = {**batch, "stiffness": torch.tensor([[900.0]])}
        loss_low, _ = model(batch_low)
        loss_high, _ = model(batch_high)
    # Losses should differ when stiffness differs (stiffness MLP has non-zero effect)
    assert loss_low.item() != loss_high.item()


def test_smoke_script():
    """--smoke flag runs without error (no real data needed)."""
    import subprocess
    repo = Path(__file__).parent.parent
    r = subprocess.run(
        [sys.executable, "scripts/train_pretrain.py", "--smoke"],
        capture_output=True,
        text=True,
        cwd=str(repo),
    )
    assert r.returncode == 0, r.stderr
    assert "smoke OK" in r.stdout


# ── 5. checkpoint ─────────────────────────────────────────────────────────────

def test_checkpoint_save_load(cfg_path, tmp_path):
    from config import load_config
    from models.multi_mae import build_multi_encoder_mae
    from utils.checkpoint import load_checkpoint, save_checkpoint

    cfg = load_config(cfg_path)
    model = build_multi_encoder_mae(cfg.multi_mae)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    ckpt_path = tmp_path / "model.pth"
    save_checkpoint(ckpt_path, model=model, optimizer=opt, epoch=3)

    model2 = build_multi_encoder_mae(cfg.multi_mae)
    ckpt = load_checkpoint(ckpt_path, model2, opt)
    assert ckpt["epoch"] == 3

    # Weights should match after reload
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.allclose(p1, p2), f"Mismatch in {n1}"


# ── 6. full training loop ─────────────────────────────────────────────────────

def test_run_pretrain_produces_checkpoints(fake_data, tmp_path):
    from config import load_config
    from training.main_pretrain import run_pretrain

    cfg_dict = _make_cfg_dict(fake_data, tmp_path / "out")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(cfg_dict))

    run_pretrain(cfg_path, local_config=None, resume=None, device_str="cpu")

    ckpts = list((tmp_path / "out").glob("*.pth"))
    assert len(ckpts) > 0, "No checkpoints written"
    log = (tmp_path / "out" / "pretrain_log.jsonl")
    assert log.exists(), "No JSONL log written"
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2  # one per epoch


def test_run_pretrain_resume(fake_data, tmp_path):
    from config import load_config
    from training.main_pretrain import run_pretrain

    cfg_dict = _make_cfg_dict(fake_data, tmp_path / "out")
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(cfg_dict))

    run_pretrain(cfg_path, local_config=None, resume=None, device_str="cpu")
    ckpt = sorted((tmp_path / "out").glob("*.pth"))[-1]
    # Should not raise when resuming
    run_pretrain(cfg_path, local_config=None, resume=ckpt, device_str="cpu")


# ── 7. detection dataset ──────────────────────────────────────────────────────

def test_detection_dataset_cxcywh(cfg_path):
    """Boxes should be normalized cxcywh in [0, 1]."""
    from config import load_config
    from data.detection_dataset import build_detection_dataset

    cfg = load_config(cfg_path)
    ds = build_detection_dataset(cfg)
    sample = ds[0]
    boxes = sample["boxes"]
    assert boxes.shape[-1] == 4
    # normalized coords must be in (0, 1]
    assert boxes.max() <= 1.0 + 1e-5
    assert boxes.min() >= 0.0 - 1e-5


def test_detection_collate_pixel_mask(cfg_path):
    from config import load_config
    from data.detection_dataset import build_detection_dataset, collate_detection_batch
    from torch.utils.data import DataLoader

    cfg = load_config(cfg_path)
    ds = build_detection_dataset(cfg)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0,
                        collate_fn=collate_detection_batch)
    batch = next(iter(loader))
    assert "pixel_values" in batch
    assert "pixel_mask" in batch
    assert "labels" in batch
    assert batch["pixel_mask"].dtype == torch.bool
    assert len(batch["labels"]) == 2


# ── 8. disk cache ─────────────────────────────────────────────────────────────

def test_disk_cache(fake_data, tmp_path):
    from config import load_config
    from data.pretrain_dataset import build_pretrain_dataset

    cache_dir = tmp_path / "cache"
    cfg_dict = _make_cfg_dict(fake_data, tmp_path / "out", cache_dir=cache_dir)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.dump(cfg_dict))

    cfg = load_config(cfg_path)
    ds = build_pretrain_dataset(cfg)

    # First access writes cache
    s1 = ds[0]
    pt_files = list(cache_dir.glob("*.pt"))
    assert len(pt_files) == 1

    # Second access reads from cache; tensors should be identical
    s2 = ds[0]
    assert torch.allclose(s1["focused"], s2["focused"])


# ── 9. lr scheduler ───────────────────────────────────────────────────────────

def test_lr_warmup_increases():
    from training.lr_sched import set_epoch_learning_rates

    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    lrs = [set_epoch_learning_rates(opt, e, epochs=10, warmup_epochs=3,
                                    min_lr=1e-5, peak_lr=1e-3) for e in range(10)]
    # warmup should strictly increase
    assert lrs[0] < lrs[1] < lrs[2]


def test_lr_cosine_decays():
    from training.lr_sched import set_epoch_learning_rates

    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    lrs = [set_epoch_learning_rates(opt, e, epochs=10, warmup_epochs=2,
                                    min_lr=1e-5, peak_lr=1e-3) for e in range(10)]
    # cosine tail should decay
    assert lrs[3] > lrs[6] > lrs[9]


def test_lr_reaches_min():
    from training.lr_sched import set_epoch_learning_rates

    min_lr = 1e-5
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    final_lr = set_epoch_learning_rates(opt, 9, epochs=10, warmup_epochs=2,
                                         min_lr=min_lr, peak_lr=1e-3)
    assert final_lr >= min_lr - 1e-9


# ── standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
