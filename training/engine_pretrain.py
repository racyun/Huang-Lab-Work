from __future__ import annotations

import time
from typing import Any

import torch

from config.settings import FullConfig
from utils.wandb_utils import log_pretrain_step


# Console step-loss print every N steps, independent of W&B log_freq.
# Lets you see the run is alive even if W&B isn't logging yet.
_STDOUT_LOG_EVERY = 10


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: FullConfig,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    global_step: int = 0,
) -> tuple[dict[str, float], int]:
    """
    Run one full epoch of multi-encoder MAE pretraining.

    Returns
    -------
    stats : dict
        Epoch-averaged metrics: loss, loss_focused, loss_hybrid, loss_volume, epoch.
    global_step : int
        Updated optimizer step counter (for W&B x-axis alignment).
    """
    model.train()
    total = 0.0
    n = 0
    sums = {"loss_focused": 0.0, "loss_hybrid": 0.0, "loss_volume": 0.0}

    use_amp = cfg.training.amp and device.type == "cuda"
    mm = cfg.multi_mae

    n_steps = len(data_loader)
    t_step = time.time()
    print(f"[engine] epoch {epoch} loop started — {n_steps} steps; first batch may take 30–60s "
          f"(Drive read + cache write)", flush=True)

    for step_in_epoch, batch in enumerate(data_loader):
        if step_in_epoch == 0:
            dt_first = time.time() - t_step
            print(f"[engine] first batch loaded in {dt_first:.1f}s", flush=True)
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            loss, parts = model(
                batch,
                mask_ratio_focused=mm.mask_ratio_focused,
                mask_ratio_hybrid=mm.mask_ratio_hybrid,
                mask_ratio_volume=mm.mask_ratio_volume,
            )

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            if cfg.training.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.training.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()

        step_loss = float(loss.detach())
        step_parts = {k: float(v) for k, v in parts.items()}
        log_pretrain_step(global_step, step_loss, step_parts, cfg.wandb.log_freq)

        # Stdout heartbeat: shows the loop is alive even if W&B hasn't flushed.
        if global_step % _STDOUT_LOG_EVERY == 0:
            dt = time.time() - t_step
            print(f"[{time.strftime('%H:%M:%S')}] "
                  f"epoch {epoch} step {step_in_epoch+1}/{n_steps} "
                  f"global_step={global_step} loss={step_loss:.4f} "
                  f"({dt:.1f}s since last)", flush=True)
            t_step = time.time()

        bs = batch["zstack"].shape[0]
        total += step_loss * bs
        n += bs
        for k in sums:
            sums[k] += step_parts[k] * bs

        global_step += 1

    out = {"loss": total / max(1, n), "epoch": float(epoch)}
    for k in sums:
        out[k] = sums[k] / max(1, n)
    return out, global_step
