"""
Model checkpointing: saves and restores model weights, optimizer state,
scheduler state, and training history so that training can be resumed or
the best model can be loaded for evaluation.
"""

import json
from datetime import datetime
from pathlib import Path

import torch


#: Files that mean "a finished or in-flight run already lives here".
RUN_FILES = ("best.pt", "last.pt", "history.json")


def recorded_seed(save_dir: Path):
    """
    The ``master_seed`` of the run already sitting in ``save_dir``.

    Returns ``(seed, files)``: the seed if one was recorded, else None, and the
    run files that are present.  ``history.json`` is consulted first because it
    is cheap and is rewritten every epoch; the checkpoints carry the same value
    under ``run_config`` and answer for runs that died before the first write.

    Runs trained before the trainer recorded its config have no seed at all —
    those come back as ``(None, [...])``, which is not the same as "no run
    here" and the caller has to tell the two apart.
    """
    save_dir = Path(save_dir)
    files = [f for f in RUN_FILES if (save_dir / f).is_file()]
    if not files:
        return None, []

    hist = save_dir / "history.json"
    if hist.is_file():
        try:
            cfg = (json.loads(hist.read_text()) or {}).get("config") or {}
            if cfg.get("master_seed") is not None:
                return int(cfg["master_seed"]), files
        except (OSError, ValueError):
            pass

    for name in ("last.pt", "best.pt"):
        f = save_dir / name
        if not f.is_file():
            continue
        try:
            ckpt = torch.load(f, map_location="cpu", weights_only=False)
            seed = (ckpt.get("run_config") or {}).get("master_seed")
            if seed is not None:
                return int(seed), files
        except Exception:
            continue

    return None, files


def ensure_writable(save_dir, seed, overwrite: bool = False):
    """
    Refuse to start a run on top of someone else's.

    A checkpoint directory is named after the variant and artifact, so two runs
    of the same model with different seeds target the same folder and the
    second silently destroys the first — best.pt, last.pt and history.json are
    all rewritten in place, and nothing else on disk remembers what was there.
    That is how a set of per-seed repeats turns into a single run without
    anyone noticing.

    So a directory that already holds a run is only reused when the run in it
    used the *same* seed (the ordinary "train it again" loop), or when
    ``overwrite`` says to replace it on purpose.  A run whose seed was never
    recorded cannot be shown to be the same one, so it counts as someone
    else's too.

    Raises ``FileExistsError`` with the ways out; returns quietly otherwise.
    """
    save_dir = Path(save_dir)
    existing_seed, files = recorded_seed(save_dir)
    if not files:
        return

    if existing_seed is not None and seed is not None and existing_seed == seed:
        print(f"[Trainer] {save_dir} already holds a run at master_seed={seed} "
              f"({', '.join(files)}) — it will be overwritten.")
        return

    if overwrite:
        was = "no recorded seed" if existing_seed is None else f"master_seed={existing_seed}"
        print(f"[Trainer] --overwrite: replacing the run in {save_dir} ({was}).")
        return

    held = ("a run whose master_seed was never recorded" if existing_seed is None
            else f"a run at master_seed={existing_seed}")
    mine = "this run has no seed" if seed is None else f"this run uses {seed}"
    raise FileExistsError(
        f"{save_dir} already holds {held} ({', '.join(files)}), and {mine}.\n"
        f"  Starting here would overwrite it, and the weights are not recoverable.\n"
        f"  Either:\n"
        f"    - drop save_dir from the config so the seed picks the folder\n"
        f"      (seeds other than the default get their own runs/..._s<seed>)\n"
        f"    - point --save_dir somewhere else\n"
        f"    - pass --overwrite to replace the existing run on purpose"
    )


def save_checkpoint(save_dir: Path, epoch, model, optimizer, scheduler, val_loss, history, is_best):
    """
    Saves model weights, optimizer state, and training history.

    Always writes ``last.pt`` (latest epoch).  When *is_best* is True an
    additional ``best.pt`` is written so the best-performing snapshot is
    always available for evaluation.
    """
    ckpt = {
        "epoch": epoch,
        # When this epoch was written.  A file mtime answers the same question
        # until the directory is copied, restored or synced, and then it quietly
        # answers a different one.
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "model_state": model.state_dict(),
        "model_config": getattr(model, "model_config", {}),
        # What produced this number — without it val_loss cannot be compared across runs.
        "run_config": (history or {}).get("config", {}),
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
