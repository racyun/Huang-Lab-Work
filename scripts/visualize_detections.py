"""Visualize detection model predictions vs ground truth bounding boxes.

Loads a saved detection checkpoint, runs inference on a few samples, and
saves PNGs with both ground-truth boxes (lime green) and predicted boxes
(red, with confidence score and class) overlaid on the image.

Usage
-----
    python3 scripts/visualize_detections.py \\
        --checkpoint /content/outputs/detect/detector_epoch_50.pth \\
        --config config/default.yaml \\
        --local-config config/colab.yaml \\
        --num-samples 8 \\
        --score-threshold 0.3 \\
        --output-dir /content/outputs/detect/visualizations
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make project root importable when run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_convert

from config import load_config
from data.detection_dataset import build_detection_dataset, collate_detection_batch
from utils.checkpoint import load_checkpoint




def _postprocess_predictions(
    pred_logits: torch.Tensor,
    pred_boxes_norm: torch.Tensor,
    img_h: int,
    img_w: int,
    score_threshold: float,
    top_k: int,
) -> dict[str, torch.Tensor]:
    """Sigmoid + flat top-K post-processing — matches engine_detect.py."""
    num_classes = pred_logits.shape[-1]
    probs = pred_logits.sigmoid()                  # [Q, C]
    flat = probs.flatten(0, 1)                     # [Q*C]
    k = min(top_k, flat.numel())
    topk_scores, topk_idx = flat.topk(k)
    query_idx = topk_idx // num_classes
    class_ids = topk_idx % num_classes

    keep = topk_scores >= score_threshold
    scores = topk_scores[keep]
    boxes_cxcywh = pred_boxes_norm[query_idx[keep]]
    labels = class_ids[keep]

    boxes_xyxy = box_convert(boxes_cxcywh, "cxcywh", "xyxy")
    boxes_xyxy[:, [0, 2]] *= img_w
    boxes_xyxy[:, [1, 3]] *= img_h

    return {"boxes": boxes_xyxy.cpu(), "scores": scores.cpu(), "labels": labels.cpu()}


def _draw(
    ax,
    img: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    title: str,
) -> None:
    """Draw image + GT (lime) + predictions (red) on a matplotlib axis."""
    ax.imshow(img.permute(1, 2, 0).numpy())

    # Ground truth — lime green, thicker stroke
    for box, lbl in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = box.tolist()
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=2.0, edgecolor="lime", facecolor="none"
        ))
        ax.text(x1, max(0, y1 - 4), f"GT cls={int(lbl)}",
                color="lime", fontsize=8,
                bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))

    # Predictions — red, thinner stroke, with score
    for box, score, lbl in zip(pred_boxes, pred_scores, pred_labels):
        x1, y1, x2, y2 = box.tolist()
        ax.add_patch(patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, linewidth=1.5, edgecolor="red", facecolor="none"
        ))
        ax.text(x1, y2 + 12, f"{float(score):.2f} cls={int(lbl)}",
                color="red", fontsize=7,
                bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))

    ax.set_title(title, fontsize=10)
    ax.axis("off")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to detector_epoch_*.pth")
    p.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    p.add_argument("--local-config", type=Path, default=None)
    p.add_argument("--num-samples", type=int, default=8,
                   help="How many images to render")
    p.add_argument("--score-threshold", type=float, default=0.3,
                   help="Drop predictions below this confidence")
    p.add_argument("--top-k", type=int, default=50,
                   help="Top-K candidate boxes considered per image before threshold")
    p.add_argument("--output-dir", type=Path,
                   default=Path("/content/outputs/detect/visualizations"))
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Loading config from {args.config}...")
    cfg = load_config(args.config, args.local_config)

    print("Building detection dataset...")
    ds = build_detection_dataset(cfg)
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=collate_detection_batch,
    )

    print(f"Loading detection model ({cfg.detection.hf_model})...")
    from transformers import AutoModelForObjectDetection
    model = AutoModelForObjectDetection.from_pretrained(
        cfg.detection.hf_model,
        num_labels=cfg.detection.num_classes,
        ignore_mismatched_sizes=True,
    ).to(device)

    print(f"Loading weights from {args.checkpoint}...")
    load_checkpoint(args.checkpoint, model, strict=False)
    model.eval()

    print(f"Rendering {args.num_samples} samples to {args.output_dir}...")
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= args.num_samples:
                break

            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            _, _, H, W = pixel_values.shape
            preds = _postprocess_predictions(
                outputs.logits[0], outputs.pred_boxes[0], H, W,
                score_threshold=args.score_threshold, top_k=args.top_k,
            )

            img = pixel_values[0].cpu().clamp(0, 1)

            # Ground-truth boxes
            gt = batch["labels"][0]
            gt_norm = gt["boxes"]
            gt_cls = gt["class_labels"]
            if gt_norm.numel() > 0:
                gt_xyxy = box_convert(gt_norm, "cxcywh", "xyxy")
                gt_xyxy[:, [0, 2]] *= W
                gt_xyxy[:, [1, 3]] *= H
            else:
                gt_xyxy = torch.zeros(0, 4)

            fig, ax = plt.subplots(figsize=(10, 10))
            title = (f"Sample {idx}  |  GT (lime): {int(len(gt_cls))} boxes  "
                     f"|  Pred (red): {int(len(preds['scores']))} boxes "
                     f"(score≥{args.score_threshold})")
            _draw(ax, img, gt_xyxy, gt_cls,
                  preds["boxes"], preds["scores"], preds["labels"],
                  title=title)

            out_path = args.output_dir / f"sample_{idx:03d}.png"
            fig.savefig(out_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"  [{idx + 1}/{args.num_samples}] {out_path}")

    print(f"\nDone. {args.num_samples} images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
