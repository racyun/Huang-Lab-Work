from __future__ import annotations

import time

import torch
from torchvision.ops import box_convert, box_iou

from config.settings import FullConfig
from utils.wandb_utils import log_detect_step


# Console step heartbeat every N steps, independent of W&B log_freq.
_STDOUT_LOG_EVERY = 10


def _has_faster_coco() -> bool:
    try:
        import faster_coco_eval  # noqa: F401
        return True
    except ImportError:
        return False


def train_one_epoch_detect(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: FullConfig,
    scaler: torch.amp.GradScaler | None,
    global_step: int = 0,
) -> tuple[float, int]:
    """
    Run one epoch of Deformable-DETR detection training.

    Returns
    -------
    mean_loss : float
    global_step : int
        Updated optimizer step counter.
    """
    model.train()
    total, n = 0.0, 0
    use_amp = cfg.detection.amp and device.type == "cuda"

    n_steps = len(data_loader)
    t_step = time.time()
    print(f"[engine] detect train loop started — {n_steps} steps; first batch may be slow on cold cache", flush=True)

    for step_in_epoch, batch in enumerate(data_loader):
        if step_in_epoch == 0:
            print(f"[engine] first batch loaded in {time.time() - t_step:.1f}s", flush=True)
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        pixel_mask = batch["pixel_mask"].to(device, non_blocking=True)
        labels: list[dict[str, torch.Tensor]] = [
            {k: v.to(device, non_blocking=True) for k, v in lab.items()}
            for lab in batch["labels"]
        ]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = out.loss

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        step_loss = float(loss.detach())
        log_detect_step(global_step, step_loss, cfg.wandb.log_freq)

        if global_step % _STDOUT_LOG_EVERY == 0:
            dt = time.time() - t_step
            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"detect step {step_in_epoch+1}/{n_steps} "
                  f"global_step={global_step} loss={step_loss:.4f} "
                  f"({dt:.1f}s since last)", flush=True)
            t_step = time.time()

        bs = pixel_values.shape[0]
        total += step_loss * bs
        n += bs
        global_step += 1

    return total / max(1, n), global_step


def eval_one_epoch_detect(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    conf_threshold: float = 0.5,
) -> dict[str, float]:
    """
    Run inference on all batches and compute detection metrics.

    Returns a dict with keys:
        mAP, AP50, AP75, mean_iou, precision, recall
    """
    try:
        from torchmetrics.detection import MeanAveragePrecision
    except ImportError:
        raise SystemExit("Install torchmetrics: pip install torchmetrics[detection]")

    # Use faster-coco-eval backend if available for ~10× speedup on CPU
    _backend = "faster_coco_eval" if _has_faster_coco() else "pycocotools"
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        backend=_backend,
    )
    metric.to(device)

    # Top-K per image: always keep best K predictions regardless of threshold,
    # so mAP is non-zero even for untrained models. After training, confident
    # predictions will naturally outscore random ones.
    TOP_K = 50

    model.eval()
    all_ious: list[float] = []

    with torch.no_grad():
        for batch in data_loader:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            pixel_mask = batch["pixel_mask"].to(device, non_blocking=True)
            gt_labels: list[dict] = batch["labels"]

            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            # pred_logits: [B, num_queries, num_classes]
            # pred_boxes:  [B, num_queries, 4] — cxcywh normalised
            pred_logits = outputs.logits          # [B, Q, C]
            pred_boxes_norm = outputs.pred_boxes  # [B, Q, 4] cxcywh

            B, _, img_h, img_w = pixel_values.shape

            preds_list = []
            targets_list = []

            for i in range(B):
                num_classes = pred_logits.shape[-1]

                # HuggingFace Deformable-DETR uses SIGMOID focal loss, not
                # softmax — every class is foreground (no implicit background
                # slot). Standard post-processing: sigmoid each (query, class)
                # logit independently, then take top-K over the flattened
                # (Q*C) score matrix. A single query can contribute multiple
                # high-confidence (class, box) predictions if needed.
                probs = pred_logits[i].sigmoid()              # [Q, C]
                flat_scores = probs.flatten(0, 1)             # [Q*C]
                k = min(TOP_K, flat_scores.numel())
                topk_scores, topk_indices = flat_scores.topk(k)
                # Decompose flat indices back into (query_idx, class_id)
                query_idx = topk_indices // num_classes
                class_ids = topk_indices % num_classes

                boxes_cxcywh = pred_boxes_norm[i][query_idx]  # [K, 4]
                # Convert cxcywh normalised → xyxy pixel
                boxes_xyxy = box_convert(boxes_cxcywh, "cxcywh", "xyxy")
                boxes_xyxy[:, [0, 2]] *= img_w
                boxes_xyxy[:, [1, 3]] *= img_h

                preds_list.append({
                    "boxes": boxes_xyxy,
                    "scores": topk_scores,
                    "labels": class_ids,
                })

                # Ground-truth: cxcywh normalised → xyxy pixel
                gt = gt_labels[i]
                gt_boxes_norm = gt["boxes"].to(device)     # [N, 4] cxcywh norm
                gt_cls = gt["class_labels"].to(device)     # [N]
                if gt_boxes_norm.numel() > 0:
                    gt_xyxy = box_convert(gt_boxes_norm, "cxcywh", "xyxy")
                    gt_xyxy[:, [0, 2]] *= img_w
                    gt_xyxy[:, [1, 3]] *= img_h
                else:
                    gt_xyxy = torch.zeros((0, 4), device=device)

                targets_list.append({"boxes": gt_xyxy, "labels": gt_cls})

                # Per-image mean IoU (best match per GT box)
                if boxes_xyxy.shape[0] > 0 and gt_xyxy.shape[0] > 0:
                    iou_mat = box_iou(gt_xyxy, boxes_xyxy)  # [N_gt, K_pred]
                    best_iou, _ = iou_mat.max(dim=1)        # [N_gt]
                    all_ious.extend(best_iou.cpu().tolist())

            metric.update(preds_list, targets_list)

    result = metric.compute()
    mean_iou = float(sum(all_ious) / len(all_ious)) if all_ious else 0.0

    return {
        "mAP":      float(result["map"]),
        "AP50":     float(result["map_50"]),
        "AP75":     float(result["map_75"]),
        "mean_iou": mean_iou,
    }
