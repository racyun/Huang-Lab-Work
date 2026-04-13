"""Single 2D view + COCO-style targets for HuggingFace DETR / Deformable-DETR."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from config.settings import DetectionConfig, FullConfig
from data.combined import TissueChipDataset


def xyxy_pixel_to_cxcywh_norm(
    boxes_xyxy: torch.Tensor, image_width: int, image_height: int
) -> torch.Tensor:
    """Pixel xyxy -> normalized cxcywh in [0,1] (DETR-style)."""
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy.reshape(0, 4)
    x1, y1, x2, y2 = boxes_xyxy.unbind(dim=-1)
    cx = (x1 + x2) / 2.0 / image_width
    cy = (y1 + y2) / 2.0 / image_height
    w = (x2 - x1).clamp(min=1e-6) / image_width
    h = (y2 - y1).clamp(min=1e-6) / image_height
    return torch.stack([cx, cy, w, h], dim=-1)


class TissueChipDetectionDataset(Dataset):
    def __init__(self, base: TissueChipDataset, det: DetectionConfig):
        self.base = base
        self.det = det
        self.short = det.train_image_short_side

    def __len__(self) -> int:
        n = len(self.base)
        if self.det.max_train_samples is not None:
            return min(n, self.det.max_train_samples)
        return n

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.base[idx]
        if self.det.image_mode == "hybrid":
            img = s["hybrid"]
        else:
            img = s["focused"]
        _, h0, w0 = img.shape
        scale = self.short / max(h0, w0)
        new_h = max(1, int(round(h0 * scale)))
        new_w = max(1, int(round(w0 * scale)))
        img_r = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
        boxes = s["boxes"]
        if boxes.numel() > 0:
            boxes = boxes * scale
        boxes_norm = xyxy_pixel_to_cxcywh_norm(boxes, new_w, new_h)
        labels = torch.zeros((boxes_norm.shape[0],), dtype=torch.long)
        return {
            "pixel_values": img_r,
            "class_labels": labels,
            "boxes": boxes_norm,
            "orig_size": torch.tensor([h0, w0]),
            "size": torch.tensor([new_h, new_w]),
        }


def build_detection_dataset(cfg: FullConfig) -> TissueChipDetectionDataset:
    from data.combined import build_tissue_chip_dataset

    base = build_tissue_chip_dataset(cfg.dataset)
    return TissueChipDetectionDataset(base, cfg.detection)


def collate_detection_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad images to a common size within batch; boolean ``pixel_mask`` for valid pixels."""
    imgs = [b["pixel_values"] for b in batch]
    max_h = max(x.shape[-2] for x in imgs)
    max_w = max(x.shape[-1] for x in imgs)
    out_imgs = []
    pixel_mask = torch.zeros(len(imgs), max_h, max_w, dtype=torch.bool)
    for i, x in enumerate(imgs):
        _, h, w = x.shape
        pad_h, pad_w = max_h - h, max_w - w
        out_imgs.append(F.pad(x, (0, pad_w, 0, pad_h), value=0.0))
        pixel_mask[i, :h, :w] = True
    pixel_values = torch.stack(out_imgs, dim=0)
    labels = []
    for b in batch:
        labels.append(
            {
                "class_labels": b["class_labels"],
                "boxes": b["boxes"],
            }
        )
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "labels": labels}
