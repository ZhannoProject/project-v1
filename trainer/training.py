"""
Core Trainer class: orchestrates the training and validation loops,
delegates optimizer setup to ``optimizer.py``, checkpointing to
``checkpoint.py``, and logging to ``logger.py``.
"""

import os
import sys

# Ensure the project root is on sys.path so bare modules (metrics, etc.)
# are importable when this file is loaded from any working directory.
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import time
import inspect
import threading
from functools import wraps
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import metrics
from utils.utils import get_device
from .optimizer import (
    build_optimizer_and_scheduler,
    freeze_decoder,
    unfreeze_all,
)
from trainer.checkpoint import ensure_writable, save_checkpoint, save_history
from trainer.logger import create_tb_writer, log_lr, log_scalars
from trainer.losses import build_loss
from data.protocol import dataset_protocol
from experiment.config import defaults, resolve_config, STAGED_API_DEFAULTS
from experiment.storage import RunLease, SCHEMA_VERSION, atomic_json, archive_existing
from trainer.checkpoint import RUN_FILES
from experiment.methodology import aligned, validate_model

API_DEFAULTS = defaults("api")


def training_session(method):
    """All public training entry points own the output directory until cleanup."""
    @wraps(method)
    def run(self, *args, **kwargs):
        if method.__name__ == "staged_fit" and self.history["config"].get("protocol") == "paper":
            raise ValueError("Paper protocol uses one ordinary 200-epoch AdamW schedule")
        if getattr(self, "_session_owner", None) == threading.get_ident():
            return method(self, *args, **kwargs)
        with RunLease(self.save_dir) as lease:
            self._guard_save_dir(self.history["config"].get("master_seed"), self.overwrite)
            archive_existing(self.save_dir, RUN_FILES, lease.run_id)
            if method.__name__ == "staged_fit":
                bound = inspect.signature(method).bind(self, *args, **kwargs)
                bound.apply_defaults()
                self.history["config"].update({k: v for k, v in bound.arguments.items() if k != "self"})
                self.history["config"]["staged"] = True
            self.history.update(schema_version=SCHEMA_VERSION, run_id=lease.run_id)
            self.model.run_config = dict(self.history["config"])
            atomic_json(self.save_dir / "resolved_config.json",
                        dict(schema_version=SCHEMA_VERSION, run_id=lease.run_id,
                             model_config=self.history["config"]["arch"],
                             run_config=self.history["config"]))
            self._session_active = True
            self._session_owner = threading.get_ident()
            try:
                self.writer = create_tb_writer(self.save_dir, self.use_tensorboard)
                return method(self, *args, **kwargs)
            finally:
                self._session_active = False
                self._session_owner = None
                self._cleanup_training()
    return run


class Trainer:
    """
    Encapsulates training / validation loops, model checkpointing,
    and metric logging for EEG causal denoising models.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_dir: str = API_DEFAULTS["save_dir"],
        learning_rate: float = API_DEFAULTS["learning_rate"],
        attn_lr_scale: float = API_DEFAULTS["attn_lr_scale"],
        weight_decay: float = API_DEFAULTS["weight_decay"],
        max_epochs: int = API_DEFAULTS["max_epochs"],
        log_every: int = API_DEFAULTS["log_every"],
        sched_warmup_epochs: int = API_DEFAULTS["sched_warmup_epochs"],
        eta_min: float = API_DEFAULTS["eta_min"],
        device: Optional[torch.device] = None,
        use_tensorboard: bool = API_DEFAULTS["tensorboard"],
        loss_fn: str = API_DEFAULTS["loss_fn"],
        run_config: Optional[dict] = None,
        overwrite: bool = API_DEFAULTS["overwrite"],
    ):
        run_config = dict(run_config or {})
        if any(isinstance(layer, nn.BatchNorm1d) for layer in model.modules()):
            raise ValueError("Build a fresh model with tcn_norm=none or channel for new training.")
        resolve_config(dict(profile="api", learning_rate=learning_rate, attn_lr_scale=attn_lr_scale,
                            weight_decay=weight_decay, max_epochs=max_epochs, log_every=log_every,
                            sched_warmup_epochs=sched_warmup_epochs, eta_min=eta_min, loss_fn=loss_fn))
        dataset = getattr(train_loader, "dataset", None)
        if hasattr(dataset, "normalization"):
            # Save the actual fitted scale and split, not an unresolved CLI default.
            run_config.update(dataset_protocol(dataset))
        if run_config.get("protocol") == "paper":
            resolve_config({**run_config, "max_epochs": max_epochs, "loss_fn": loss_fn})
            n = int(dataset.clean.shape[-1]) if hasattr(dataset, "clean") else 512
            validate_model(model, run_config, n)
        # Before anything is built or any directory is made: refusing early
        # means a clash costs a second, not a model and an optimizer.
        self.save_dir = Path(save_dir)
        self.overwrite = overwrite
        self.use_tensorboard = use_tensorboard
        self._guard_save_dir((run_config or {}).get("master_seed"), overwrite)

        self.device = device or get_device()
        print(f"[Trainer] Device: {self.device}")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = build_loss(loss_fn).to(self.device)
        print(f"[Trainer] Loss function: {loss_fn}")
        self.max_epochs = max_epochs
        self.log_every = log_every

        # Build AdamW + CosineAnnealingLR with linear warmup
        self.optimizer, self.scheduler = self._build_optimizer(
            lr=learning_rate,
            attn_lr_scale=attn_lr_scale,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            warmup_epochs=sched_warmup_epochs,
            eta_min=eta_min,
        )

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        # ``config`` records what produced these numbers.  val_loss is only comparable
        # across runs that share loss_fn, artifact, snr_range_db and master_seed.
        #
        # ``arch`` is the shape of the model, read off the model itself rather than
        # off the config that was meant to build it.  Without it a run records the
        # training recipe and not the thing trained, and the two can disagree: a CLI
        # override, a config edited after the run, a variant whose name pins a value.
        # This project has the scar — every MLA run here was trained with 4 attention
        # layers and 2 heads while its YAML said 2 and 4, so the headline MHA/MLA
        # ablation differs in depth, head count and parameter count, and nothing in
        # any run said so.  Taken from the built model, it cannot drift from the
        # weights beside it.
        self.history = {"train_loss": [], "val_loss": [], "rrmse_t": [], "rrmse_s": [],
                        "acc": [], "config": {}}

        # Keep a complete record, including direct Python API arguments that
        # used to be absent when no CLI run_config was supplied.
        self.history["config"].update(run_config)
        self.history["config"].update(
            profile=run_config.get("profile", "api"), save_dir=str(self.save_dir),
            learning_rate=learning_rate, attn_lr_scale=attn_lr_scale,
            weight_decay=weight_decay, max_epochs=max_epochs, log_every=log_every,
            sched_warmup_epochs=sched_warmup_epochs, eta_min=eta_min,
            device=str(self.device), tensorboard=use_tensorboard, overwrite=overwrite,
            loss_fn=loss_fn, optimizer=type(self.optimizer).__name__,
            selection_metric="gt_mse",
            arch=self._architecture(), n_params=sum(p.numel() for p in self.model.parameters()))
        self.writer = None
        self.model.run_config = dict(self.history["config"])

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Trainer] Trainable params: {n_params:,}")

    def _architecture(self) -> dict:
        """
        The model's own account of its shape, for ``history.json``.

        Falls back to whatever ``model_config`` holds if the variant cannot be
        resolved (a hand-built model passed straight to the Trainer, which the
        tests do): an incomplete record beats refusing to start a run over one.
        """
        try:
            from experiment.checkpoints import complete_architecture
            return complete_architecture(self.model)
        except Exception:                      # pragma: no cover - defensive
            return dict(getattr(self.model, "model_config", {}) or {})

    def _extra_trainable(self):
        """Trainable tensors outside ``self.model`` (none here; KDTrainer adds its projectors)."""
        return []

    def _build_optimizer(self, **kwargs):
        """Fresh optimizer + scheduler over the model and ``_extra_trainable()``."""
        return build_optimizer_and_scheduler(
            self.model, extra_params=self._extra_trainable(), **kwargs)

    # ------------------------------------------------------------------
    # Core loops
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int):
        """Executes one full pass over the training dataset."""
        self.model.train()  # enables dropout / batchnorm updates
        running_loss = 0.0
        n_seen = 0
        t0 = time.time()

        for step, batch in enumerate(self.train_loader):
            noisy = batch["noisy"].to(self.device)
            clean = batch["clean"].to(self.device)

            # Forward pass
            pred = self.model(noisy)
            pred, clean = aligned(pred, clean, self.model)
            loss = self.criterion(pred, clean)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track loss (weighted by batch size for correct averaging)
            bs = noisy.size(0)
            running_loss += loss.item() * bs
            n_seen += bs

            if (step + 1) % self.log_every == 0:
                avg = running_loss / n_seen
                print(f"  [E{epoch:02d}] step {step+1:4d}/{len(self.train_loader)}  "
                      f"loss={avg:.4f}")

        avg_loss = running_loss / n_seen
        elapsed = time.time() - t0
        print(f"  → train_loss = {avg_loss:.4f}  ({elapsed:.1f}s)")
        return avg_loss

    def _guard_save_dir(self, seed, overwrite: bool):
        """Refuse to start on top of someone else's run — see ensure_writable."""
        ensure_writable(self.save_dir, seed, overwrite)

    @torch.no_grad()  # disable gradient calculation to save memory / compute
    def _validate(self, epoch: int, model: Optional[nn.Module] = None):
        """
        Evaluates the model on the validation dataset.

        ``model`` defaults to the model being trained; KDTrainer passes the
        frozen teacher to report its reference metrics on the same split.
        """
        model = self.model if model is None else model
        model.eval()  # disables dropout, uses population stats for batchnorm
        running_loss = 0.0
        agg_metrics = {"rrmse_t": 0.0, "rrmse_s": 0.0, "acc": 0.0}
        n_seen = 0

        for batch in self.val_loader:
            noisy = batch["noisy"].to(self.device)
            clean = batch["clean"].to(self.device)

            pred = model(noisy)
            # Teacher references use the student's common target positions,
            # with the teacher's own delay alignment.
            cfg = {**self.history["config"],
                   "output_delay": getattr(model, "run_config", {}).get("output_delay", 0)}
            pred, clean = aligned(pred, clean, cfg)
            loss = torch.nn.functional.mse_loss(pred, clean)

            bs = noisy.size(0)
            running_loss += loss.item() * bs

            metricsA = metrics.compute_all_metrics(pred, clean)
            for k, v in metricsA.items():
                agg_metrics[k] += v * bs
            n_seen += bs

        avg_loss = running_loss / n_seen
        for k in agg_metrics:
            agg_metrics[k] /= max(n_seen, 1)  # prevent division by zero

        print(f"  → val_loss   = {avg_loss:.4f}  "
              f"RRMSE_t={agg_metrics['rrmse_t']:.3f}  "
              f"RRMSE_f={agg_metrics['rrmse_s']:.3f}  "
              f"CC={agg_metrics['acc']:.3f}")
        if model is self.model:
            log_lr(self.optimizer, self.model)
        return avg_loss, agg_metrics

    # ------------------------------------------------------------------
    # Epoch driver
    # ------------------------------------------------------------------

    def _run_epochs(self, start_epoch: int, end_epoch: int, phase_label: str):
        """Shared epoch loop used by both fit() and staged_fit()."""
        for epoch in range(start_epoch, end_epoch + 1):
            print(f"\n[{phase_label} | Epoch {epoch}/{self.max_epochs}]")

            train_loss = self._train_one_epoch(epoch)
            val_loss, val_metrics = self._validate(epoch)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            for k, v in val_metrics.items():
                self.history[k].append(v)

            log_scalars(self.writer, train_loss, val_loss, val_metrics, epoch)

            # Step the cosine annealing scheduler (epoch-based, no val_loss needed)
            self.scheduler.step()

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            save_checkpoint(
                self.save_dir, epoch, self.model,
                self.optimizer, self.scheduler,
                val_loss, self.history, is_best,
            )
            # Written every epoch (not only at the end): an interrupted run then still
            # leaves a history.json that describes the best.pt sitting next to it.
            save_history(self.save_dir, self.history)

    def _finish(self):
        save_history(self.save_dir, self.history)
        print(f"\nTraining completed. best_val_loss = {self.best_val_loss:.4f}")
        print(f"The model will be kept at {self.save_dir}")
        if self.writer:
            self.writer.close()

    def _cleanup_training(self):
        if self.writer:
            self.writer.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @training_session
    def fit(self):
        """
        Standard training: all parameters are trained jointly from epoch 1.
        Use this for TCN-only or when staged training is not needed.
        """
        print(f"\n{'='*60}")
        print(f"Training started, {self.max_epochs} epochs in total")
        print(f"{'='*60}")
        self._run_epochs(1, self.max_epochs, phase_label="fit")
        self._finish()

    @training_session
    def staged_fit(
        self,
        warmup_epochs: int = STAGED_API_DEFAULTS["warmup_epochs"],
        warmup_lr: float = STAGED_API_DEFAULTS["warmup_lr"],
        finetune_lr: float = STAGED_API_DEFAULTS["finetune_lr"],
        attn_lr_scale: float = STAGED_API_DEFAULTS["attn_lr_scale"],
        weight_decay: float = STAGED_API_DEFAULTS["weight_decay"],
        sched_warmup_epochs: int = STAGED_API_DEFAULTS["sched_warmup_epochs"],
        eta_min: float = STAGED_API_DEFAULTS["eta_min"],
    ):
        """
        Two-phase training for sandwich models (TCN + attention).

        Phase 1 — TCN warm-up (warmup_epochs):
            Freeze the attention decoder. Only TCN (pt1, pt2, head) is
            trainable. The TCN learns to extract and reconstruct EEG
            features first, providing a stable feature space for the
            attention decoder to build on in Phase 2.

        Phase 2 — Joint fine-tuning (remaining epochs):
            Unfreeze everything. A fresh optimizer is created so the
            momentum/variance states don't carry stale frozen-param info.
            Both TCN and attention co-adapt with the (optionally different)
            learning rates controlled by attn_lr_scale.

        Each phase gets its own CosineAnnealingLR + linear warmup scheduler,
        sized to that phase's epoch count.

        Only applicable to models with a ``decoder`` attribute.
        Falls back to fit() for TCN-only models.
        """
        if not hasattr(self.model, "decoder"):
            print("[Trainer] No decoder found — falling back to standard fit()")
            self.fit()
            return

        assert warmup_epochs < self.max_epochs, (
            f"warmup_epochs ({warmup_epochs}) must be less than "
            f"max_epochs ({self.max_epochs})"
        )

        phase2_epochs = self.max_epochs - warmup_epochs

        # Phase 1: freeze attention decoder, train TCN only
        print(f"\n{'='*60}")
        print(f"Phase 1: TCN warm-up ({warmup_epochs} epochs)")
        print(f"{'='*60}")
        freeze_decoder(self.model)
        self.optimizer, self.scheduler = self._build_optimizer(
            lr=warmup_lr, attn_lr_scale=1.0,
            weight_decay=weight_decay,
            max_epochs=warmup_epochs,
            warmup_epochs=min(sched_warmup_epochs, warmup_epochs // 2),
            eta_min=eta_min,
        )
        self._run_epochs(1, warmup_epochs, phase_label="Phase1-warmup")

        # Phase 2: unfreeze all, joint fine-tuning
        print(f"\n{'='*60}")
        print(f"Phase 2: Joint fine-tuning ({phase2_epochs} epochs)")
        print(f"{'='*60}")
        unfreeze_all(self.model)
        self.optimizer, self.scheduler = self._build_optimizer(
            lr=finetune_lr, attn_lr_scale=attn_lr_scale,
            weight_decay=weight_decay,
            max_epochs=phase2_epochs,
            warmup_epochs=sched_warmup_epochs,
            eta_min=eta_min,
        )
        self._run_epochs(warmup_epochs + 1, self.max_epochs, phase_label="Phase2-finetune")
        self._finish()
