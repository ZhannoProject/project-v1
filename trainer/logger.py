"""
Logging helpers: TensorBoard integration and console output for learning
rates and per-epoch scalars.
"""

from pathlib import Path

import torch.nn as nn


def create_tb_writer(save_dir: Path, use_tensorboard: bool):
    """Initialize a TensorBoard SummaryWriter if requested (returns None otherwise)."""
    if not use_tensorboard:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=str(save_dir / "tb"))
    except ImportError:
        print("Tensorboard not found on the device.")
        return None


def log_lr(optimizer, model: nn.Module):
    """Print the current learning rate for each named param group."""
    parts = []
    for g in optimizer.param_groups:
        name = g.get("name", "?")
        parts.append(f"{name}={g['lr']:.2e}")
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    suffix = f"  ({frozen} params frozen)" if frozen > 0 else ""
    print(f"  → lr: {', '.join(parts)}{suffix}")


def log_scalars(writer, train_loss, val_loss, val_metrics, epoch):
    """Write train/val loss and validation metrics to TensorBoard."""
    if writer is None:
        return
    writer.add_scalar("loss/train", train_loss, epoch)
    writer.add_scalar("loss/val", val_loss, epoch)
    for k, v in val_metrics.items():
        writer.add_scalar(f"metric/{k}", v, epoch)
