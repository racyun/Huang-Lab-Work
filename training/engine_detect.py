from __future__ import annotations

import torch
from torch.cuda.amp import GradScaler, autocast

from config.settings import FullConfig


def train_one_epoch_detect(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: FullConfig,
    scaler: GradScaler | None,
) -> float:
    model.train()
    total, n = 0.0, 0
    use_amp = cfg.detection.amp and device.type == "cuda"

    for batch in data_loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        pixel_mask = batch["pixel_mask"].to(device, non_blocking=True)
        labels: list[dict[str, torch.Tensor]] = []
        for lab in batch["labels"]:
            labels.append({k: v.to(device, non_blocking=True) for k, v in lab.items()})

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            out = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
            loss = out.loss

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = pixel_values.shape[0]
        total += float(loss.detach()) * bs
        n += bs
    return total / max(1, n)
