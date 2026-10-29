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

Compressing the student (optional)
----------------------------------
The student does not have to be the teacher's twin.  Its size comes from
its own model keys (``hidden_channels``, ``n_attn_layers``, ``kv_lora_rank``,
``bottleneck_ratio``, even the variant), so a KD preset can distil a
64-channel / 10-layer teacher into a 32-channel / 2-layer student.  The
three loss terms then work as follows:

  task, output   unchanged — both models emit (B, 1, T)
  feature        a learnable ``FeatureProjector`` (a FitNets regressor:
                 one Linear along the mismatched channel axis, pointwise in
                 time and therefore causal) maps each student feature to
                 the teacher's width.  Projectors are trained with the
                 student and discarded afterwards — they are not part of
                 the checkpoint.  Depth mismatch is handled by naming pairs
                 in ``kd_feature_layers``: ``decoder.blocks.1:decoder.blocks.9``
                 taps student block 1 against teacher block 9.
  init_from_teacher
                 same-name tensors of different shape receive the teacher's
                 leading slice (weight inheritance: the first C_s of C_t
                 channels, the first n of N blocks); everything else keeps
                 its random init.

With identical sizes nothing changes: every projector is the identity and
every tensor copies exactly.

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
(task loss + RRMSE_t / RRMSE_f / CC against the clean signal), so its
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
from utils.utils import get_device


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
    Copy every teacher tensor whose name exists in the student.

    Same shape → copied whole.  For a ``tcn_mla_bi`` → ``tcn_mla`` pair this
    is the entire state dict (the causal mask is a buffer with identical
    contents in both).

    Different shape, same rank (a narrower or shallower student) → the
    overlapping leading block is copied, e.g. the first C_s of the teacher's
    C_t channels; the rest of the student tensor keeps its random init.
    Tensors the student lacks (deeper teacher blocks) and tensors the teacher
    lacks are skipped.  Returns (n_copied, n_skipped), where copied counts
    both exact and sliced tensors.
    """
    t_sd = teacher.state_dict()
    s_sd = student.state_dict()
    new_sd, n_exact, n_sliced = {}, 0, 0
    for k, s_t in s_sd.items():
        t_t = t_sd.get(k)
        if t_t is None or t_t.dim() != s_t.dim():
            continue
        if t_t.shape == s_t.shape:
            new_sd[k] = t_t
            n_exact += 1
        else:
            block = tuple(slice(0, min(a, b)) for a, b in zip(t_t.shape, s_t.shape))
            merged = s_t.clone()
            merged[block] = t_t[block].to(merged.dtype)
            new_sd[k] = merged
            n_sliced += 1
    student.load_state_dict(new_sd, strict=False)
    n_copied, n_skipped = n_exact + n_sliced, len(s_sd) - len(new_sd)
    print(f"[KD] Student initialised from teacher: {n_exact} tensors copied"
          + (f", {n_sliced} copied as a leading slice (student is a different size)" if n_sliced else "")
          + f", {n_skipped} left at random init")
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

    def __init__(self, model: nn.Module, layer_names: Sequence[str], role: str = "model"):
        self.features: Dict[str, torch.Tensor] = {}
        self._handles = []
        for name in layer_names:
            try:
                module = model.get_submodule(name)
            except AttributeError as e:
                available = [n for n, _ in model.named_modules() if n and "." not in n]
                raise ValueError(
                    f"kd_feature_layers: the {role} ({type(model).__name__}) has no "
                    f"submodule '{name}'; its top-level modules are {available}"
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


def default_feature_layers(model) -> list:
    """
    Where to match features when the caller did not say.

    ``decoder`` is the interesting place in the sandwich and parallel
    variants: it is the attention stack, and the part a bidirectional teacher
    actually does differently. The TCN families have no decoder -- naming one
    raises -- so they match at ``backbone``, which is the whole convolutional
    trunk and the only representation they have.

    Only the default. A layer the caller names is used as given, and still
    fails loudly if the model has no such module.
    """
    for name in ("decoder", "backbone", "tcn_pt1"):
        if isinstance(getattr(model, name, None), nn.Module):
            return [name]
    raise ValueError(
        f"{type(model).__name__} has no module this knows to match features at; "
        f"name one with --kd_feature_layers (top-level modules: "
        f"{[n for n, _ in model.named_children()]})")


def parse_feature_layers(layer_names: Sequence[str]) -> list:
    """
    ``["decoder", "decoder.blocks.1:decoder.blocks.9"]`` →
    ``[("decoder", "decoder", "decoder"),
       ("decoder.blocks.1:decoder.blocks.9", "decoder.blocks.1", "decoder.blocks.9")]``

    Each entry is ``student_path[:teacher_path]``; the label (first element)
    is the raw entry and keys the logged features.
    """
    pairs = []
    for entry in layer_names:
        student_name, _, teacher_name = str(entry).partition(":")
        student_name, teacher_name = student_name.strip(), teacher_name.strip()
        if not student_name:
            raise ValueError(f"kd_feature_layers: empty student path in {entry!r}")
        pairs.append((str(entry), student_name, teacher_name or student_name))
    return pairs


class FeatureProjector(nn.Module):
    """
    Map a student feature onto the teacher's shape (a FitNets regressor).

    Shapes must have the same rank and differ in at most one non-batch axis;
    a single ``Linear`` acts along that axis and nothing mixes along the
    others, so a (B, C, T) or (B, T, C) feature stays causal in T.  With
    identical shapes it is the identity and owns no parameters.
    """

    def __init__(self, student_shape: Sequence[int], teacher_shape: Sequence[int]):
        super().__init__()
        student_shape, teacher_shape = tuple(student_shape), tuple(teacher_shape)
        if len(student_shape) != len(teacher_shape):
            raise ValueError(f"feature rank mismatch: student {student_shape} vs teacher {teacher_shape}")
        diff = [i for i in range(1, len(student_shape)) if student_shape[i] != teacher_shape[i]]
        if len(diff) > 1:
            raise ValueError(
                f"features differ in more than one axis: student {student_shape} vs teacher "
                f"{teacher_shape}; a projector can only bridge a channel-width difference"
            )
        self.axis = diff[0] if diff else None
        self.proj = nn.Linear(student_shape[self.axis], teacher_shape[self.axis]) if diff else None
        self.student_shape, self.teacher_shape = student_shape[1:], teacher_shape[1:]

    @property
    def is_identity(self) -> bool:
        return self.proj is None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.proj is None:
            return x
        return self.proj(x.movedim(self.axis, -1)).movedim(-1, self.axis)

    def extra_repr(self) -> str:
        return "identity" if self.proj is None else \
            f"axis={self.axis}, {self.student_shape} -> {self.teacher_shape}"


@torch.no_grad()
def _check_tappable(student: nn.Module, teacher: nn.Module, pairs) -> None:
    """
    Refuse a feature tap that cannot exist, and say what to do instead.

    The feature term matches a named submodule of the student against one of
    the teacher, and ``kd_feature_layers`` defaults to ``["decoder"]`` --
    which only the MHA/MLA sandwich variants have.  Distilling across
    families is where this bites: a ``tcn_se`` teacher is a backbone and a
    head, with no attention decoder for a sandwich student's ``decoder`` to
    be matched against.  torch's own message for that is "X has no attribute
    `decoder`", which says nothing about which model, or that three perfectly
    good ways out exist.
    """
    tops = lambda m: [n for n, _ in m.named_modules() if n and "." not in n]  # noqa: E731
    for label, s_name, t_name in pairs:
        for role, model, name in (("student", student, s_name), ("teacher", teacher, t_name)):
            try:
                model.get_submodule(name)
            except AttributeError as e:
                other = "teacher" if role == "student" else "student"
                other_model = teacher if role == "student" else student
                raise ValueError(
                    f"kd_feature_layers: the {role} ({type(model).__name__}) has no "
                    f"submodule '{name}'.\n"
                    f"  {role} top-level modules: {tops(model)}\n"
                    f"  {other} ({type(other_model).__name__}): {tops(other_model)}\n"
                    f"  The feature term matches one named module against another, so it "
                    f"needs a module both models have. Either:\n"
                    f"    kd_feature_weight: 0      turn the feature term off and distil on "
                    f"the outputs alone, which needs no shared module;\n"
                    f"    kd_feature_layers: [...]  name a module both have;\n"
                    f"    kd_feature_layers: ['student_path:teacher_path']  pair two that "
                    f"are named differently."
                ) from e


def probe_feature_shapes(student: nn.Module, teacher: nn.Module, pairs, seq_len: int) -> dict:
    """
    One dry forward of each model on a zero window → ``{label: (student_shape,
    teacher_shape)}`` for the tapped submodules (batch axis excluded).
    """
    _check_tappable(student, teacher, pairs)
    dev = next(student.parameters()).device
    x = torch.zeros(1, 1, seq_len, device=dev)
    s_tap = FeatureTap(student, [s for _, s, _ in pairs], "student")
    t_tap = FeatureTap(teacher, [t for _, _, t in pairs], "teacher")
    was_training = student.training
    try:
        student.eval()(x)
        teacher(x.to(next(teacher.parameters()).device))
    finally:
        student.train(was_training)
        s_tap.remove()
        t_tap.remove()
    return {label: (tuple(s_tap.features[s].shape[1:]), tuple(t_tap.features[t].shape[1:]))
            for label, s, t in pairs}


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
                        f"{tuple(t_feat.shape)} — shapes must match after projection "
                        f"(KDTrainer builds a FeatureProjector for a channel-width difference)"
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

    The student may be smaller than the teacher.  When the feature term is
    on, one ``FeatureProjector`` per tapped layer is built from a dry
    forward (identity when the shapes already agree) and trained with the
    student through the optimizer's ``extra`` group; ``kd_feature_layers``
    entries may be ``student_path:teacher_path`` pairs.  Projectors are
    never written to the checkpoint.
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

        # Feature taps and projectors are only needed when the feature term is active.
        # They must exist before super().__init__ builds the optimizer (see _extra_trainable).
        self._pairs = parse_feature_layers(kd_feature_layers) if kd_feature_weight > 0.0 else []
        self._projectors: Dict[str, FeatureProjector] = {}
        self.kd_projectors = nn.ModuleList()          # the non-identity ones, for parameters()
        if self._pairs:
            seq_len = int(getattr(model, "model_config", {}).get("max_seq_len", 512))
            shapes = probe_feature_shapes(model, teacher, self._pairs, seq_len)
            for label, (s_shape, t_shape) in shapes.items():
                proj = FeatureProjector((1, *s_shape), (1, *t_shape))
                self._projectors[label] = proj
                if not proj.is_identity:
                    self.kd_projectors.append(proj)
            device = trainer_kwargs.get("device") or get_device()
            self.kd_projectors.to(device)

        super().__init__(model=model, train_loader=train_loader, val_loader=val_loader,
                         loss_fn=loss_fn, **trainer_kwargs)

        self.teacher = teacher.to(self.device).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        n_student = sum(p.numel() for p in self.model.parameters())
        n_teacher = sum(p.numel() for p in self.teacher.parameters())
        print(f"[KD] Student {n_student:,} params / teacher {n_teacher:,} "
              f"({n_student / n_teacher:.1%} of the teacher"
              + (", same size" if n_student == n_teacher else f", {n_teacher / n_student:.2f}x smaller")
              + ")")

        output_loss = build_loss(kd_output_loss or loss_fn).to(self.device)
        self.kd_criterion = DistillationLoss(
            task_loss=self.criterion, output_loss=output_loss,
            alpha=kd_alpha, feature_weight=kd_feature_weight, feature_loss=kd_feature_loss,
        ).to(self.device)

        pairs = self._pairs
        self._student_tap = FeatureTap(self.model, [s for _, s, _ in pairs]) if pairs else None
        self._teacher_tap = FeatureTap(self.teacher, [t for _, _, t in pairs]) if pairs else None

        print(f"[KD] alpha={kd_alpha} (teacher output) / {1 - kd_alpha:g} (ground truth), "
              f"output_loss={kd_output_loss or loss_fn}, feature_weight={kd_feature_weight}"
              + (f" on {[label for label, _, _ in pairs]} ({kd_feature_loss})" if pairs else ""))
        for label, proj in self._projectors.items():
            if not proj.is_identity:
                n = sum(p.numel() for p in proj.parameters())
                print(f"[KD] Feature projector for '{label}': {proj.extra_repr()}  ({n:,} params, not saved)")

        for key in ("kd_task", "kd_output", "kd_feature"):
            self.history[key] = []
        self.history["student"] = {
            "variant": self.model.model_config.get("variant") if hasattr(self.model, "model_config") else None,
            "n_params": n_student,
        }

        # Teacher reference on the same validation split (once, up front)
        print("[KD] Teacher on the validation split:")
        t_loss, t_metrics = self._validate(0, model=self.teacher)
        self.history["teacher"] = {
            "checkpoint": teacher_ckpt,
            "variant": teacher.model_config.get("variant") if hasattr(teacher, "model_config") else None,
            "n_params": n_teacher,
            "val_loss": t_loss, **t_metrics,
        }

    def _extra_trainable(self):
        return list(self.kd_projectors.parameters())

    def _student_features(self) -> Optional[Dict[str, torch.Tensor]]:
        """Tapped student features, projected onto the teacher's shapes, keyed by label."""
        if not self._student_tap:
            return None
        return {label: self._projectors[label](self._student_tap.features[s])
                for label, s, _ in self._pairs}

    def _teacher_features(self) -> Optional[Dict[str, torch.Tensor]]:
        if not self._teacher_tap:
            return None
        return {label: self._teacher_tap.features[t] for label, _, t in self._pairs}

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
            teacher_feats = self._teacher_features()

            pred = self.model(noisy)
            student_feats = self._student_features()

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
                  f"RRMSE_t={t['rrmse_t']:.3f}  RRMSE_f={t['rrmse_s']:.3f}  CC={t['acc']:.3f}")
