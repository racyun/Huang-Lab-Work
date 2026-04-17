"""
Weights & Biases integration helpers.

All public functions are safe no-ops when:
  - wandb is not installed
  - cfg.wandb.enabled is False
  - a W&B run has not been initialised

Usage
-----
In a training entrypoint::

    from utils.wandb_utils import init_wandb, log_metrics, finish_wandb

    run = init_wandb(cfg, mode="pretrain")
    # ... training loop ...
    log_metrics({"train/loss": 0.5, "epoch": 1}, step=100)
    finish_wandb()
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings import FullConfig


def _wandb():
    """Return the wandb module or None if not installed."""
    try:
        import wandb
        return wandb
    except ImportError:
        return None


def init_wandb(
    cfg: "FullConfig",
    *,
    mode: str = "pretrain",
    run_name_override: Optional[str] = None,
) -> Optional[Any]:
    """
    Initialise a W&B run and return it, or return None if disabled.

    Parameters
    ----------
    cfg:
        Full training config (reads cfg.wandb for all W&B settings).
    mode:
        Short string added to the auto-generated run name, e.g. "pretrain"
        or "detect".
    run_name_override:
        If provided, overrides cfg.wandb.run_name (useful for CLI --run-name).
    """
    wb = _wandb()
    if wb is None or not cfg.wandb.enabled:
        return None

    run_name = run_name_override or cfg.wandb.run_name

    # Flatten FullConfig to a plain dict for W&B config panel
    config_dict = _flatten_config(cfg)

    run = wb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        name=run_name,
        tags=list(cfg.wandb.tags) + [mode],
        notes=cfg.wandb.notes or None,
        config=config_dict,
        reinit=True,
    )
    return run


def watch_model(model: Any, log_freq: int = 100) -> None:
    """Call wandb.watch() on the model to log gradient histograms."""
    wb = _wandb()
    if wb is None or wb.run is None:
        return
    wb.watch(model, log="gradients", log_freq=log_freq)


def log_metrics(metrics: dict[str, Any], step: Optional[int] = None) -> None:
    """
    Log a dict of metrics to the active W&B run.
    No-op if W&B is not running.
    """
    wb = _wandb()
    if wb is None or wb.run is None:
        return
    wb.log(metrics, step=step)


def log_pretrain_step(
    step: int,
    loss: float,
    parts: dict[str, float],
    log_freq: int = 10,
) -> None:
    """Log per-optimizer-step pretrain metrics (throttled by log_freq)."""
    if step % log_freq != 0:
        return
    log_metrics(
        {
            "train/step_loss": loss,
            "train/step_loss_focused": parts.get("loss_focused", 0.0),
            "train/step_loss_hybrid": parts.get("loss_hybrid", 0.0),
            "train/step_loss_volume": parts.get("loss_volume", 0.0),
        },
        step=step,
    )


def log_pretrain_epoch(
    epoch: int,
    stats: dict[str, float],
    optimizer: Any,
    step: int,
) -> None:
    """Log per-epoch pretrain metrics plus one LR value per param group."""
    lrs = {
        f"lr/{name}": g["lr"]
        for name, g in zip(
            ["focused", "hybrid", "volume", "stiffness"],
            optimizer.param_groups,
        )
    }
    log_metrics(
        {
            "train/loss": stats["loss"],
            "train/loss_focused": stats["loss_focused"],
            "train/loss_hybrid": stats["loss_hybrid"],
            "train/loss_volume": stats["loss_volume"],
            "epoch": epoch,
            **lrs,
        },
        step=step,
    )


def log_detect_step(step: int, loss: float, log_freq: int = 10) -> None:
    """Log per-step detection loss (throttled by log_freq)."""
    if step % log_freq != 0:
        return
    log_metrics({"detect/step_loss": loss}, step=step)


def log_detect_epoch(epoch: int, loss: float, lr: float, step: int) -> None:
    """Log per-epoch detection metrics."""
    log_metrics(
        {"detect/loss": loss, "detect/lr": lr, "epoch": epoch},
        step=step,
    )


def log_detect_eval(epoch: int, metrics: dict[str, float], step: int) -> None:
    """Log per-epoch detection evaluation metrics (mAP, IoU, etc.)."""
    log_metrics(
        {f"eval/{k}": v for k, v in metrics.items()} | {"epoch": epoch},
        step=step,
    )


def log_detect_combined(
    epoch: int,
    loss: float,
    lr: float,
    eval_metrics: dict[str, float],
    step: int,
) -> None:
    """Log all per-epoch detection metrics in a single W&B call.

    Merging into one call ensures loss, LR, and eval metrics (AP50, mAP,
    mean_iou) all appear on the same x-axis point in W&B, giving clean
    curves when plotted against epoch or step.
    """
    log_metrics(
        {
            "detect/loss": loss,
            "detect/lr": lr,
            "eval/AP50": eval_metrics.get("AP50", 0.0),
            "eval/mAP": eval_metrics.get("mAP", 0.0),
            "eval/AP75": eval_metrics.get("AP75", 0.0),
            "eval/mean_iou": eval_metrics.get("mean_iou", 0.0),
            "epoch": epoch,
        },
        step=step,
    )


def finish_wandb() -> None:
    """Mark the active W&B run as finished. No-op if not running."""
    wb = _wandb()
    if wb is None or wb.run is None:
        return
    wb.finish()


# ── helpers ───────────────────────────────────────────────────────────────────

def _flatten_config(cfg: "FullConfig") -> dict[str, Any]:
    """Convert FullConfig dataclasses to a flat dict for the W&B config panel."""
    out: dict[str, Any] = {}
    for section_name, section in dataclasses.asdict(cfg).items():
        if isinstance(section, dict):
            for k, v in section.items():
                out[f"{section_name}/{k}"] = v
        else:
            out[section_name] = section
    return out
