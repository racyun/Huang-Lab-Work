from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SplitConfig:
    name: str
    stiffness_kpa: float
    zstack_root: str
    focused_root: str
    hybrid_root: str
    labels_root: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SplitConfig:
        return cls(
            name=str(d["name"]),
            stiffness_kpa=float(d["stiffness_kpa"]),
            zstack_root=str(d.get("zstack_root", "")),
            focused_root=str(d.get("focused_root", "")),
            hybrid_root=str(d.get("hybrid_root", "")),
            labels_root=str(d.get("labels_root", "")),
        )


@dataclass
class DatasetConfig:
    well_prefix: str = "W"
    well_count: int = 222
    zstack_subdir: str = "P00001"
    expected_z_slices: Optional[int] = 140
    focus_filename_glob: str = "*focus_stacked*.tif"
    hybrid_folder_template: str = "hybrid_results_{well_id}"
    resize: dict[str, Optional[list[int]]] = field(
        default_factory=lambda: {"zstack": None, "focused": None, "hybrid": None}
    )
    splits: list[SplitConfig] = field(default_factory=list)
    cache_dir: Optional[str] = None
    cache_only_modality: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DatasetConfig:
        resize = d.get("resize") or {}
        return cls(
            well_prefix=str(d.get("well_prefix", "W")),
            well_count=int(d.get("well_count", 222)),
            zstack_subdir=str(d.get("zstack_subdir", "P00001")),
            expected_z_slices=(
                int(d["expected_z_slices"]) if d.get("expected_z_slices") is not None else None
            ),
            focus_filename_glob=str(d.get("focus_filename_glob", "*focus_stacked*.tif")),
            hybrid_folder_template=str(d.get("hybrid_folder_template", "hybrid_results_{well_id}")),
            resize={
                "zstack": resize.get("zstack"),
                "focused": resize.get("focused"),
                "hybrid": resize.get("hybrid"),
            },
            splits=[SplitConfig.from_dict(s) for s in d.get("splits", [])],
            cache_dir=d.get("cache_dir"),
            cache_only_modality=d.get("cache_only_modality"),
        )


@dataclass
class TrainingConfig:
    output_dir: str = "./outputs"
    seed: int = 0
    batch_size: int = 2
    epochs: int = 100
    num_workers: int = 4
    lr: float = 1.5e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    amp: bool = True
    grad_clip: Optional[float] = None
    min_lr: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingConfig:
        return cls(
            output_dir=str(d.get("output_dir", "./outputs")),
            seed=int(d.get("seed", 0)),
            batch_size=int(d.get("batch_size", 2)),
            epochs=int(d.get("epochs", 100)),
            num_workers=int(d.get("num_workers", 4)),
            lr=float(d.get("lr", 1.5e-4)),
            weight_decay=float(d.get("weight_decay", 0.05)),
            warmup_epochs=int(d.get("warmup_epochs", 10)),
            amp=bool(d.get("amp", True)),
            grad_clip=float(d["grad_clip"]) if d.get("grad_clip") is not None else None,
            min_lr=float(d.get("min_lr", 0.0)),
        )


@dataclass
class PretrainConfig:
    image_size: int = 224
    mask_ratio: float = 0.75

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PretrainConfig:
        return cls(
            image_size=int(d.get("image_size", 224)),
            mask_ratio=float(d.get("mask_ratio", 0.75)),
        )


@dataclass
class MultiMAEConfig:
    image_size: int = 224
    patch_xy_2d: int = 16
    volume_z: int = 140
    volume_h: int = 224
    volume_w: int = 224
    tublet_z: int = 4
    volume_patch_xy: int = 16
    in_chans: int = 3
    embed_dim: int = 384
    twod_depth: int = 6
    twod_num_heads: int = 6
    vol_depth: int = 6
    vol_num_heads: int = 6
    decoder_embed_dim: int = 512
    twod_dec_depth: int = 4
    twod_dec_num_heads: int = 8
    vol_dec_depth: int = 4
    vol_dec_num_heads: int = 8
    norm_pix_loss: bool = False
    stiffness_hidden: int = 64
    kpa_input_mode: str = "divide"
    kpa_divisor: float = 900.0
    loss_weight_focused: float = 1.0
    loss_weight_hybrid: float = 1.0
    loss_weight_volume: float = 1.0
    mask_ratio_focused: float = 0.75
    mask_ratio_hybrid: float = 0.75
    mask_ratio_volume: float = 0.75
    lr_mult_focused: float = 1.0
    lr_mult_hybrid: float = 1.0
    lr_mult_volume: float = 1.0
    lr_mult_stiffness: float = 1.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MultiMAEConfig:
        return cls(
            image_size=int(d.get("image_size", 224)),
            patch_xy_2d=int(d.get("patch_xy_2d", 16)),
            volume_z=int(d.get("volume_z", 140)),
            volume_h=int(d.get("volume_h", 224)),
            volume_w=int(d.get("volume_w", 224)),
            tublet_z=int(d.get("tublet_z", 4)),
            volume_patch_xy=int(d.get("volume_patch_xy", 16)),
            in_chans=int(d.get("in_chans", 3)),
            embed_dim=int(d.get("embed_dim", 384)),
            twod_depth=int(d.get("twod_depth", 6)),
            twod_num_heads=int(d.get("twod_num_heads", 6)),
            vol_depth=int(d.get("vol_depth", 6)),
            vol_num_heads=int(d.get("vol_num_heads", 6)),
            decoder_embed_dim=int(d.get("decoder_embed_dim", 512)),
            twod_dec_depth=int(d.get("twod_dec_depth", 4)),
            twod_dec_num_heads=int(d.get("twod_dec_num_heads", 8)),
            vol_dec_depth=int(d.get("vol_dec_depth", 4)),
            vol_dec_num_heads=int(d.get("vol_dec_num_heads", 8)),
            norm_pix_loss=bool(d.get("norm_pix_loss", False)),
            stiffness_hidden=int(d.get("stiffness_hidden", 64)),
            kpa_input_mode=str(d.get("kpa_input_mode", "divide")),
            kpa_divisor=float(d.get("kpa_divisor", 900.0)),
            loss_weight_focused=float(d.get("loss_weight_focused", 1.0)),
            loss_weight_hybrid=float(d.get("loss_weight_hybrid", 1.0)),
            loss_weight_volume=float(d.get("loss_weight_volume", 1.0)),
            mask_ratio_focused=float(d.get("mask_ratio_focused", 0.75)),
            mask_ratio_hybrid=float(d.get("mask_ratio_hybrid", 0.75)),
            mask_ratio_volume=float(d.get("mask_ratio_volume", 0.75)),
            lr_mult_focused=float(d.get("lr_mult_focused", 1.0)),
            lr_mult_hybrid=float(d.get("lr_mult_hybrid", 1.0)),
            lr_mult_volume=float(d.get("lr_mult_volume", 1.0)),
            lr_mult_stiffness=float(d.get("lr_mult_stiffness", 1.0)),
        )


@dataclass
class DetectionConfig:
    hf_model: str = "SenseTime/deformable-detr"
    num_classes: int = 2
    train_image_short_side: int = 800
    max_train_samples: Optional[int] = None
    image_mode: str = "focused"
    lr: float = 2e-5
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 2
    num_workers: int = 2
    amp: bool = True
    mae_encoder_ckpt: Optional[str] = None
    mae_encoder_prefix: str = "mae_focused.encoder."

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DetectionConfig:
        return cls(
            hf_model=str(d.get("hf_model", "facebook/deformable-detr-resnet-50")),
            num_classes=int(d.get("num_classes", 2)),
            train_image_short_side=int(d.get("train_image_short_side", 800)),
            max_train_samples=d.get("max_train_samples"),
            image_mode=str(d.get("image_mode", "focused")),
            lr=float(d.get("lr", 2e-5)),
            weight_decay=float(d.get("weight_decay", 1e-4)),
            epochs=int(d.get("epochs", 50)),
            batch_size=int(d.get("batch_size", 2)),
            num_workers=int(d.get("num_workers", 2)),
            amp=bool(d.get("amp", True)),
            mae_encoder_ckpt=d.get("mae_encoder_ckpt"),
            mae_encoder_prefix=str(d.get("mae_encoder_prefix", "mae_focused.encoder.")),
        )


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "huang-lab-tissue-chip"
    entity: Optional[str] = None        # W&B username or team name
    run_name: Optional[str] = None      # auto-generated if None
    tags: list = field(default_factory=list)
    notes: str = ""
    log_freq: int = 10                  # log step-level metrics every N steps
    watch_model: bool = False           # wandb.watch() — shows gradient histograms

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WandbConfig:
        return cls(
            enabled=bool(d.get("enabled", False)),
            project=str(d.get("project", "huang-lab-tissue-chip")),
            entity=d.get("entity") or None,
            run_name=d.get("run_name") or None,
            tags=list(d.get("tags", [])),
            notes=str(d.get("notes", "")),
            log_freq=int(d.get("log_freq", 10)),
            watch_model=bool(d.get("watch_model", False)),
        )


@dataclass
class FullConfig:
    dataset: DatasetConfig
    training: TrainingConfig
    pretrain: PretrainConfig
    multi_mae: MultiMAEConfig
    detection: DetectionConfig
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FullConfig:
        mm = d.get("multi_mae") or {}
        det = d.get("detection") or {}
        wb = d.get("wandb") or {}
        return cls(
            dataset=DatasetConfig.from_dict(d.get("dataset", {})),
            training=TrainingConfig.from_dict(d.get("training", {})),
            pretrain=PretrainConfig.from_dict(d.get("pretrain", {})),
            multi_mae=MultiMAEConfig.from_dict(mm),
            detection=DetectionConfig.from_dict(det),
            wandb=WandbConfig.from_dict(wb),
        )
