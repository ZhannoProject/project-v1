"""
Optimizer construction, parameter grouping, and freeze / unfreeze helpers.

Sandwich models (TCN + MHA / TCN + MLA) benefit from per-component learning
rates: the TCN backbone can converge too fast via its residual paths and
starve the attention decoder of gradients, so we allow a separate LR scale
for the decoder group.

Scheduler: CosineAnnealingLR with linear warmup.
  Phase 1 — Linear warmup: LR ramps from ~0 to base_lr over warmup_epochs.
  Phase 2 — Cosine annealing: LR decays smoothly from base_lr to eta_min
            over the remaining epochs.
"""

import torch
import torch.nn as nn


def build_param_groups(
    model: nn.Module,
    base_lr: float,
    attn_lr_scale: float,
    weight_decay: float,
    extra_params=None,
):
    """
    Split model parameters into optimizer groups with different learning rates.

    If the model has a `decoder` attribute (sandwich architecture), returns
    two groups:
      - 'tcn'  : everything except the decoder  -> lr = base_lr
      - 'attn' : the decoder                    -> lr = base_lr * attn_lr_scale

    Otherwise returns a single group with base_lr for all parameters.

    ``extra_params`` are trainable tensors that live outside the model but
    must be optimised alongside it (the KD feature projectors); they form
    a third group 'extra' at base_lr.
    """
    groups = _model_param_groups(model, base_lr, attn_lr_scale, weight_decay)
    extra = [p for p in (extra_params or []) if p.requires_grad]
    if extra:
        groups.append({"params": extra, "lr": base_lr, "weight_decay": weight_decay, "name": "extra"})
    return groups


def _model_param_groups(model, base_lr, attn_lr_scale, weight_decay):
    if hasattr(model, "decoder") and attn_lr_scale != 1.0:
        decoder_ids = {id(p) for p in model.decoder.parameters()}
        tcn_params = [p for p in model.parameters() if id(p) not in decoder_ids]
        attn_params = list(model.decoder.parameters())

        attn_lr = base_lr * attn_lr_scale
        print(f"[Trainer] Adaptive LR: TCN lr={base_lr:.1e}, Decoder lr={attn_lr:.1e} "
              f"(scale={attn_lr_scale}x)")
        return [
            {"params": tcn_params,  "lr": base_lr, "weight_decay": weight_decay, "name": "tcn"},
            {"params": attn_params, "lr": attn_lr, "weight_decay": weight_decay, "name": "decoder"},
        ]
    else:
        # Single group — could be TCN-only or decoder-only during warm-up.
        # Detect which params are trainable to pick a meaningful group name.
        trainable = [p for p in model.parameters() if p.requires_grad]
        if hasattr(model, "decoder"):
            decoder_ids = {id(p) for p in model.decoder.parameters()}
            all_decoder = all(id(p) in decoder_ids for p in trainable)
            name = "decoder" if all_decoder else "all"
        else:
            name = "all"
        return [{"params": trainable, "lr": base_lr, "weight_decay": weight_decay, "name": name}]


def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float,
    attn_lr_scale: float,
    weight_decay: float,
    max_epochs: int,
    warmup_epochs: int = 5,
    eta_min: float = 1e-6,
    extra_params=None,
):
    """
    Create a fresh AdamW optimizer + CosineAnnealingLR with linear warmup.

    The scheduler chains two phases via ``SequentialLR``:
      1. **Linear warmup** (``warmup_epochs`` epochs):
         LR ramps linearly from ``eta_min`` to ``base_lr``.
      2. **Cosine annealing** (remaining epochs):
         LR decays smoothly from ``base_lr`` to ``eta_min``.

    Used at the start of each training phase so that momentum / variance
    states don't carry stale frozen-param info across phases.

    Parameters
    ----------
    max_epochs : int
        Total number of epochs for this training phase.
    warmup_epochs : int
        Number of epochs for the linear LR warmup ramp.
    eta_min : float
        Minimum learning rate at the end of cosine decay.
    """
    param_groups = build_param_groups(model, lr, attn_lr_scale, weight_decay, extra_params)
    optimizer = torch.optim.AdamW(param_groups)

    # Guard against edge cases
    warmup_epochs = max(warmup_epochs, 0)
    cosine_epochs = max(max_epochs - warmup_epochs, 1)

    if warmup_epochs > 0:
        # Phase 1: linear warmup from ~0 → base_lr
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,   # start at 0.1% of base_lr
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        # Phase 2: cosine annealing from base_lr → eta_min
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=eta_min,
        )
        # Chain the two schedulers
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
        print(f"[Scheduler] CosineAnnealing + {warmup_epochs}-epoch linear warmup  "
              f"(T_max={cosine_epochs}, eta_min={eta_min:.1e})")
    else:
        # No warmup — pure cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=eta_min,
        )
        print(f"[Scheduler] CosineAnnealing (T_max={cosine_epochs}, eta_min={eta_min:.1e})")

    return optimizer, scheduler


def freeze_tcn(model: nn.Module):
    """Freeze all parameters except the attention decoder."""
    for name, param in model.named_parameters():
        if not name.startswith("decoder"):
            param.requires_grad = False
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"[Trainer] Frozen {frozen} params, {trainable} trainable (decoder only)")


def freeze_decoder(model: nn.Module):
    """Freeze the attention decoder, leave TCN (pt1, pt2, head) trainable."""
    for name, param in model.named_parameters():
        if name.startswith("decoder"):
            param.requires_grad = False
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"[Trainer] Frozen {frozen} params, {trainable} trainable (TCN only)")


def unfreeze_all(model: nn.Module):
    """Unfreeze every parameter in the model."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"[Trainer] Unfrozen all — {trainable} trainable params")
