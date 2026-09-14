"""
Model checkpointing: saves and restores model weights, optimizer state,
scheduler state, and training history so that training can be resumed or
the best model can be loaded for evaluation.
"""

import json
from pathlib import Path

import torch


def save_checkpoint(save_dir: Path, epoch, model, optimizer, scheduler, val_loss, history, is_best):
    """
    Saves model weights, optimizer state, and training history.

    Always writes ``last.pt`` (latest epoch).  When *is_best* is True an
    additional ``best.pt`` is written so the best-performing snapshot is
    always available for evaluation.
    """
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "model_config": getattr(model, "model_config", {}),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "val_loss": val_loss,
        "history": history,
    }
    torch.save(ckpt, save_dir / "last.pt")

    if is_best:
        torch.save(ckpt, save_dir / "best.pt")
        print(f"New best model is found and it has been saved.")


def save_history(save_dir: Path, history):
    """Dump the full training history (losses + metrics per epoch) to JSON."""
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
