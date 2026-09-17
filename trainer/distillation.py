"""
Knowledge distillation: bidirectional (non-causal) teacher → causal student.

The deployable models are strictly causal — at sample t they only see
[0..t].  Their bidirectional twins (``tcn_*_bi``) see the whole 512-sample
window and therefore denoise better, but they cannot run online.  KD lets
the causal student borrow from the teacher's hindsight: the student is
trained to reproduce, from past-only context, what the teacher produced
with full context.

    noisy ──► teacher (frozen, causal=False) ──► teacher_out, teacher_feats
      │                                              │           │
      └─────► student (trained, causal=True) ──► student_out, student_feats
                                                     │           │
    L = (1 - α) · task(student_out, clean)           │           │
      +      α  · output(student_out, teacher_out) ◄─┘           │
      +      β  · feature(student_feats, teacher_feats) ◄────────┘

Three terms
-----------
  task     the usual denoising loss against the clean signal (same
           ``loss_fn`` as a normal run)
  output   the same waveform loss, but against the teacher's output — a
           "soft target" that is smoother and easier to fit than the
           ground truth because the teacher already resolved the ambiguity
           that only future samples can settle
  feature  MSE between intermediate features of the two models.  The
           default tap is ``decoder`` — the post-attention (B, H', T)
           tensor, i.e. exactly the block that differs between the two
           models.  Both models share every shape, so no projector is
           needed; any dotted submodule path works (``decoder.blocks.0``).

α = 0 and β = 0 recovers ordinary training.  Because teacher and student
have identical state dicts, the student can also be *initialised* from
the teacher (``init_from_teacher``), which usually gives a head start.

Usage
-----
    # 1. train the teacher (any *_bi preset)
    python -m trainer --config configs/tcn_mla_bi_eog.yaml
    # 2. distil the causal student from it
    python -m trainer --config configs/tcn_mla_eog_kd.yaml
    # or on any preset:
    python -m trainer --config configs/tcn_mla_eog.yaml \\
        --teacher_ckpt runs/tcn_mla_bi_eog/best.pt --kd_alpha 0.5 \\
        --kd_feature_weight 0.1 --save_dir runs/tcn_mla_eog_kd

The student is validated and checkpointed exactly like a normal run
(task loss + RRMSE_t / RRMSE_s / ACC against the clean signal), so its
``best.pt`` is a drop-in for eval.py and deploy/export_onnx.py.
``history.json`` additionally records the per-epoch KD terms
(``kd_task``, ``kd_output``, ``kd_feature``) and the teacher's own
validation metrics under ``teacher`` for reference.
"""

import os
import sys

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.variants import build_model_from_config, variant_is_causal
from trainer.training import Trainer
from trainer.losses import build_loss


# ============================================================
# Teacher loading
# ============================================================
def load_teacher(
    ckpt_path: str | Path,
    config_path: str | Path | None = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Rebuild a trained model from its checkpoint and freeze it.

    The architecture comes from the checkpoint's ``model_config`` (written
    by every run of this trainer).  ``config_path`` supplies a YAML for
    legacy checkpoints that predate ``model_config``; failing both, the
    architecture is inferred from the weight shapes.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state"]

    cfg: dict = {}
    if config_path:
        from configs import load_config
        cfg.update(load_config(str(config_path)))
    cfg.update(ckpt.get("model_config") or {})      # what the checkpoint recorded wins
    if "variant" not in cfg:
        # Legacy checkpoint without model_config and no YAML: the weights are
        # the ground truth for every shape-determined hyperparameter.
        from deploy.export_onnx import infer_arch_from_state_dict
        cfg.update(infer_arch_from_state_dict(sd))

    teacher = build_model_from_config(cfg)
    try:
        teacher.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"Teacher weights in {ckpt_path} do not fit the architecture {cfg}. "
            f"Pass --teacher_config with the YAML the teacher was trained with."
        ) from e

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    variant = teacher.model_config.get("variant", "?")
    n_params = sum(p.numel() for p in teacher.parameters())
    print(f"[KD] Teacher: {ckpt_path}  (variant={variant}, epoch={ckpt.get('epoch', '?')}, "
          f"val_loss={ckpt.get('val_loss', float('nan')):.4f}, params={n_params:,})")
    if variant_is_causal(variant):
        print("[KD] Note: the teacher is causal. Distillation still works (e.g. MHA → MLA), "
              "but the intended teacher is a bidirectional *_bi run.")
    if device is not None:
        teacher = teacher.to(device)
    return teacher


def init_student_from_teacher(student: nn.Module, teacher: nn.Module) -> Tuple[int, int]:
    """
    Copy every teacher tensor whose name and shape exist in the student.

    For a ``tcn_mla_bi`` → ``tcn_mla`` pair this is the whole state dict
    (the causal mask is a buffer with identical contents in both).
    Returns (n_copied, n_skipped).
    """
    t_sd = teacher.state_dict()
    s_sd = student.state_dict()
    compatible = {k: v for k, v in t_sd.items()
                  if k in s_sd and s_sd[k].shape == v.shape}
    student.load_state_dict(compatible, strict=False)
    n_copied, n_skipped = len(compatible), len(s_sd) - len(compatible)
    print(f"[KD] Student initialised from teacher: {n_copied} tensors copied, "
          f"{n_skipped} left at random init")
    return n_copied, n_skipped


# ============================================================
# Feature taps (forward hooks on named submodules)
# ============================================================
class FeatureTap:
    """
    Record the forward output of named submodules.

        tap = FeatureTap(model, ["decoder", "decoder.blocks.0"])
        model(x)
        tap.features["decoder"]        # (B, H', T), still attached to the graph
        tap.remove()

    Names are dotted paths as in ``model.named_modules()``.
    """

    def __init__(self, model: nn.Module, layer_names: Sequence[str]):
        self.features: Dict[str, torch.Tensor] = {}
        self._handles = []
        for name in layer_names:
            try:
                module = model.get_submodule(name)
            except AttributeError as e:
                available = [n for n, _ in model.named_modules() if n and "." not in n]
                raise ValueError(
                    f"kd_feature_layers: '{name}' is not a submodule of "
                    f"{type(model).__name__}; top-level modules are {available}"
                ) from e
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            self.features[name] = output
        return hook

    def clear(self):
        self.features = {}

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# ============================================================
# Loss
# ============================================================
class DistillationLoss(nn.Module):
    """
    L = (1 - alpha) * task(student, clean)
      +      alpha  * output(student, teacher)
      + feature_weight * mean_l feature(student_feat_l, teacher_feat_l)

    ``task_loss`` and ``output_loss`` are ordinary ``(pred, target) -> scalar``
    modules from trainer/losses.py.  Returns ``(total, parts)`` where
    ``parts`` holds the detached ``task`` / ``output`` / ``feature`` values
    for logging.
    """

    def __init__(
        self,
        task_loss: nn.Module,
        output_loss: nn.Module,
        alpha: float = 0.5,
        feature_weight: float = 0.0,
        feature_loss: str = "mse",
    ):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"kd_alpha must be in [0, 1], got {alpha}")
        if feature_weight < 0.0:
            raise ValueError(f"kd_feature_weight must be >= 0, got {feature_weight}")
        if feature_loss not in ("mse", "l1"):
            raise ValueError("feature_loss must be 'mse' or 'l1'")
        self.task_loss = task_loss
        self.output_loss = output_loss
        self.alpha = alpha
        self.feature_weight = feature_weight
        self.feature_loss = nn.MSELoss() if feature_loss == "mse" else nn.L1Loss()

    def forward(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
        clean: torch.Tensor,
        student_feats: Optional[Dict[str, torch.Tensor]] = None,
        teacher_feats: Optional[Dict[str, torch.Tensor]] = None,
    ):
        task = self.task_loss(student_out, clean)
        output = self.output_loss(student_out, teacher_out.detach())
        total = (1.0 - self.alpha) * task + self.alpha * output

        feature = student_out.new_zeros(())
        if self.feature_weight > 0.0:
            if not student_feats or not teacher_feats:
                raise ValueError("feature_weight > 0 but no features were tapped")
            per_layer = []
            for name, t_feat in teacher_feats.items():
                s_feat = student_feats[name]
                if s_feat.shape != t_feat.shape:
                    raise ValueError(
                        f"feature '{name}': student {tuple(s_feat.shape)} vs teacher "
                        f"{tuple(t_feat.shape)} — feature KD needs matching shapes "
                        f"(same hidden_channels / bottleneck_ratio / topology)"
                    )
                per_layer.append(self.feature_loss(s_feat, t_feat.detach()))
            feature = torch.stack(per_layer).mean()
            total = total + self.feature_weight * feature

        parts = {"task": task.detach(), "output": output.detach(), "feature": feature.detach()}
        return total, parts


# ============================================================
# Trainer
# ============================================================
class KDTrainer(Trainer):
    """
    Trainer with a frozen teacher running alongside the student.

    Everything except the training-step loss is inherited: validation,
    checkpointing, LR schedule, staged_fit.  The student is validated on
    the clean signal (not the teacher), so ``best.pt`` is picked by real
    denoising quality.
    """

    def __init__(
        self,
        model: nn.Module,
        teacher: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        kd_alpha: float = 0.5,
        kd_feature_weight: float = 0.0,
        kd_feature_layers: Sequence[str] = ("decoder",),
        kd_output_loss: Optional[str] = None,
        kd_feature_loss: str = "mse",
        init_from_teacher: bool = False,
        teacher_ckpt: Optional[str] = None,
        loss_fn: str = "l1_combined",
        **trainer_kwargs,
    ):
        if init_from_teacher:
            init_student_from_teacher(model, teacher)

        super().__init__(model=model, train_loader=train_loader, val_loader=val_loader,
                         loss_fn=loss_fn, **trainer_kwargs)

        self.teacher = teacher.to(self.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        output_loss = build_loss(kd_output_loss or loss_fn).to(self.device)
        self.kd_criterion = DistillationLoss(
            task_loss=self.criterion, output_loss=output_loss,
            alpha=kd_alpha, feature_weight=kd_feature_weight, feature_loss=kd_feature_loss,
        ).to(self.device)

        # Feature taps are only installed when the feature term is active
        layers = tuple(kd_feature_layers) if kd_feature_weight > 0.0 else ()
        self._student_tap = FeatureTap(self.model, layers) if layers else None
        self._teacher_tap = FeatureTap(self.teacher, layers) if layers else None

        print(f"[KD] alpha={kd_alpha} (teacher output) / {1 - kd_alpha:g} (ground truth), "
              f"output_loss={kd_output_loss or loss_fn}, feature_weight={kd_feature_weight}"
              + (f" on {list(layers)} ({kd_feature_loss})" if layers else ""))

        for key in ("kd_task", "kd_output", "kd_feature"):
            self.history[key] = []

        # Teacher reference on the same validation split (once, up front)
        print("[KD] Teacher on the validation split:")
        t_loss, t_metrics = self._validate(0, model=self.teacher)
        self.history["teacher"] = {
            "checkpoint": teacher_ckpt,
            "variant": teacher.model_config.get("variant") if hasattr(teacher, "model_config") else None,
            "val_loss": t_loss, **t_metrics,
        }

    def _train_one_epoch(self, epoch: int):
        """One pass over the training set with the distillation loss."""
        self.model.train()
        self.teacher.eval()
        running = {"total": 0.0, "task": 0.0, "output": 0.0, "feature": 0.0}
        n_seen = 0
        t0 = time.time()

        for step, batch in enumerate(self.train_loader):
            noisy = batch["noisy"].to(self.device)
            clean = batch["clean"].to(self.device)

            # Teacher sees the same window with future context; no gradient
            with torch.no_grad():
                teacher_out = self.teacher(noisy)
            teacher_feats = dict(self._teacher_tap.features) if self._teacher_tap else None

            pred = self.model(noisy)
            student_feats = dict(self._student_tap.features) if self._student_tap else None

            loss, parts = self.kd_criterion(pred, teacher_out, clean, student_feats, teacher_feats)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self._student_tap:
                self._student_tap.clear()
                self._teacher_tap.clear()

            bs = noisy.size(0)
            running["total"] += loss.item() * bs
            for k in ("task", "output", "feature"):
                running[k] += parts[k].item() * bs
            n_seen += bs

            if (step + 1) % self.log_every == 0:
                print(f"  [E{epoch:02d}] step {step+1:4d}/{len(self.train_loader)}  "
                      f"loss={running['total'] / n_seen:.4f}  "
                      f"(task={running['task'] / n_seen:.4f} "
                      f"out={running['output'] / n_seen:.4f} "
                      f"feat={running['feature'] / n_seen:.4f})")

        avg = {k: v / n_seen for k, v in running.items()}
        elapsed = time.time() - t0
        print(f"  → train_loss = {avg['total']:.4f}  "
              f"[task={avg['task']:.4f}  output={avg['output']:.4f}  "
              f"feature={avg['feature']:.4f}]  ({elapsed:.1f}s)")

        for k in ("task", "output", "feature"):
            self.history[f"kd_{k}"].append(avg[k])
        if self.writer:
            for k in ("task", "output", "feature"):
                self.writer.add_scalar(f"kd/{k}", avg[k], epoch)
        return avg["total"]

    def _finish(self):
        if self._student_tap:
            self._student_tap.remove()
            self._teacher_tap.remove()
        super()._finish()
        t = self.history.get("teacher", {})
        if t:
            print(f"Teacher reference: val_loss={t['val_loss']:.4f}  "
                  f"RRMSE_t={t['rrmse_t']:.3f}  RRMSE_s={t['rrmse_s']:.3f}  ACC={t['acc']:.3f}")
