"""
CLI entry point for training.

Usage:
    python -m trainer --config configs/tcn_mha_eog.yaml
    python -m trainer --variant tcn --artifact EOG --max_epochs 50

Knowledge distillation (bidirectional teacher → causal student) is enabled
by ``--teacher_ckpt`` / ``teacher_ckpt:`` in the YAML; see configs/*_kd.yaml
and trainer/distillation.py.

All arguments are optional — a YAML config provides defaults, and any CLI
flag overrides the corresponding YAML key.  Priority chain:
    CLI flag  >  YAML file  >  hardcoded fallbacks below
"""

import os
import sys
from pathlib import Path
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from data.Dataloader import build_dataloaders
from models.variants import (build_model_from_config, VARIANT_REGISTRY, variant_is_causal,
                             pinned_kwargs)
from configs import load_config, merge_cli_overrides
from experiment.config import resolve_config, resolve_save_dir, defaults, PROFILES
from trainer.checkpoint import ensure_writable, recorded_seed
from trainer.distillation import default_feature_layers
from trainer.training import Trainer
from trainer.distillation import KDTrainer, load_teacher
from utils.utils import seed_everything, resolve_device, DEVICE_CHOICES

DEFAULT_DATA_DIR = defaults("cli")["data_dir"]


#: The default split and training seed, also used by the sweep planners.
DEFAULT_MASTER_SEED = defaults("cli")["master_seed"]


def build_parser():
    # --- CLI args (all optional — YAML config provides defaults) ---
    parser = argparse.ArgumentParser(description="Train an EEG causal denoising model")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (e.g. configs/tcn_mha_eog.yaml)")
    parser.add_argument("--profile", choices=list(PROFILES), default=None)
    parser.add_argument("--protocol", choices=["exploratory", "paper"], default=None)
    parser.add_argument("--output_delay", type=int, default=None,
                        help="Estimate clean[t] at acquisition t+ell; architecture stays unchanged")
    parser.add_argument("--valid_history_samples", type=int, default=None)
    parser.add_argument("--valid_future_samples", type=int, default=None)
    parser.add_argument("--hop", type=int, default=None)
    parser.add_argument("--tcn_norm", choices=["none", "channel"], default=None)
    parser.add_argument("--se_window", type=int, default=None)
    parser.add_argument("--attention_window", type=int, default=None)
    parser.add_argument("--resampling", choices=["offline", "causal"], default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--sampling", choices=["legacy", "balanced"], default=None)
    parser.add_argument("--split_ratios", type=float, nargs=3, default=None)
    parser.add_argument("--variant", default=None, choices=list(VARIANT_REGISTRY.keys()))
    parser.add_argument("--artifact", default=None, choices=["EOG", "EMG", "BOTH"])
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--save_dir", default=None,
                        help="Checkpoint directory "
                             "(default: runs/<variant>_<artifact>[_kd]_<rate>hz[_s<seed>])")
    parser.add_argument("--overwrite", action="store_true", default=None,
                        help="Replace the run already in save_dir. Without it, training "
                             "refuses to replace any existing run")
    parser.add_argument("--hidden_channels", type=int, default=None)
    parser.add_argument("--n_attn_layers", type=int, default=None,
                        help="Number of transformer layers (e.g. 2, 10, 20, 50)")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--attn_lr_scale", type=float, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--tensorboard", action="store_true", default=None)
    parser.add_argument("--staged", action="store_true", default=None,
                        help="Use two-phase staged training (freeze decoder → joint finetune)")
    parser.add_argument("--warmup_epochs", type=int, default=None,
                        help="Phase 1 epochs (TCN-only warm-up)")
    parser.add_argument("--warmup_lr", type=float, default=None,
                        help="Learning rate for Phase 1")
    parser.add_argument("--finetune_lr", type=float, default=None,
                        help="Base learning rate for Phase 2")
    parser.add_argument("--sched_warmup_epochs", type=int, default=None,
                        help="Epochs of linear LR warmup before cosine decay")
    parser.add_argument("--eta_min", type=float, default=None,
                        help="Minimum LR for cosine annealing (default: 1e-6)")
    parser.add_argument("--loss_fn", type=str, default=None,
                        choices=["mse", "l1", "stft", "l2_combined", "l1_combined"],
                        help="Loss function: 'l1_combined' (default), 'l2_combined', 'mse', 'l1', or 'stft'")
    parser.add_argument("--se_reduction", type=int, default=None,
                        help="Squeeze-and-Excitation ratio for the tcn_se variants: "
                             "C -> C/r -> C, gated from a running mean so it stays causal")
    parser.add_argument("--kv_lora_rank", type=int, default=None,
                        help="KV low-rank bottleneck dim for MLA variants (default 16). "
                             "Ignored for the tcn_mla_r{N} aliases, which fix the rank themselves.")
    parser.add_argument("--bottleneck_ratio", type=float, default=None,
                        help="Channel squeeze ratio between TCN and attention "
                             "(e.g. 0.5 → attn_dim = hidden_channels * 0.5). "
                             "Default: 0.5 for sandwich MHA/MLA variants.")
    parser.add_argument("--skip_bottleneck_ratio", type=float, default=None,
                        help="Replace the identity skip around each attention block with a "
                             "low-rank BottleneckSkip of this ratio (e.g. 0.5), so attention "
                             "cannot be bypassed. Default: off (identity skip).")
    parser.add_argument("--target_fs", type=float, default=None,
                        help="resample the EEGdenoiseNet epochs to this rate before "
                             "training (e.g. 250 for EEG-IO's rate; 512-sample epochs "
                             "become 500). Default: leave them at 256 Hz.")
    parser.add_argument("--master_seed", type=int, default=None,
                        help="Seed for data splits, SNR draws, weight init, dropout and "
                             "batch order (default 42)")
    parser.add_argument("--normalization", choices=["fixed", "epoch"], default=None,
                        help="fixed (default): a training-calibrated scale, suitable for online inference; "
                             "epoch: legacy offline full-window normalization")
    parser.add_argument("--normalization_scale", type=float, default=None,
                        help="Fixed input scale; omitted means calibrate on the training EEG pool")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default=None,
                        help="auto (default: cuda > mps > cpu). Use cpu for bit-exact "
                             "reproducibility: MPS is not deterministic even when seeded.")
    parser.add_argument("--turboquant", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable optional TurboQuant K/V quantization")
    parser.add_argument("--turboquant_bits", type=int, choices=range(2, 9), default=None)
    parser.add_argument("--turboquant_key_mode", choices=["prod", "mse"], default=None)
    parser.add_argument("--turboquant_train_mode", choices=["off", "ste"], default=None)
    parser.add_argument("--turboquant_seed", type=int, default=None)
    # --- Knowledge distillation (see trainer/distillation.py) ---
    parser.add_argument("--teacher_ckpt", type=str, default=None,
                        help="Teacher checkpoint (normally a *_bi run). Enables distillation.")
    parser.add_argument("--teacher_config", type=str, default=None,
                        help="YAML of the teacher, only needed for legacy checkpoints without model_config")
    parser.add_argument("--kd_alpha", type=float, default=None,
                        help="Weight of the teacher-output term; the ground-truth term gets "
                             "1 - kd_alpha (default 0.5)")
    parser.add_argument("--kd_feature_weight", type=float, default=None,
                        help="Weight of the feature-matching term on --kd_feature_layers "
                             "(default 0 = off)")
    parser.add_argument("--kd_feature_layers", nargs="+", default=None,
                        help="Submodules whose outputs are matched, e.g. decoder decoder.blocks.0 "
                             "(default: decoder). student:teacher pairs map different depths, "
                             "e.g. decoder.blocks.1:decoder.blocks.9")
    parser.add_argument("--kd_output_loss", type=str, default=None,
                        choices=["mse", "l1", "stft", "l2_combined", "l1_combined"],
                        help="Loss between student and teacher outputs (default: same as --loss_fn)")
    parser.add_argument("--kd_feature_loss", type=str, default=None, choices=["mse", "l1"],
                        help="Loss between student and teacher features (default: mse)")
    parser.add_argument("--init_from_teacher", action="store_true", default=None,
                        help="Initialise the student from the teacher's weights before training "
                             "(a smaller student receives the leading slice of each tensor)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # --- Build effective config: YAML defaults -> CLI overrides ---
    if args.config:
        cfg = load_config(args.config)
        cfg = merge_cli_overrides(cfg, args)
    else:
        cfg = {k: v for k, v in vars(args).items() if v is not None}

    try:
        cfg = resolve_save_dir(resolve_config(cfg))
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    distil = cfg["teacher_ckpt"] is not None
    if distil and not variant_is_causal(cfg["variant"]):
        # Not refused: MHA -> MLA between two bidirectional twins is a real
        # architecture experiment. But the usual reason to distil here is to
        # give a causal student the hindsight a non-causal teacher has, and a
        # non-causal student already has it -- so say what is being given up
        # rather than just noting the flag.
        print(f"[Warning] Student '{cfg['variant']}' is not causal. Distillation here "
              f"normally transfers hindsight from a non-causal teacher to a causal "
              f"student; a student that already sees the future has nothing to gain "
              f"from that, and the run cannot be deployed or exported either. "
              f"Continuing anyway.")

    # Checked here as well as in the Trainer, so a clash costs a second rather
    # than the time it takes to load the dataset and build the model.  It is a
    # mistake in the command, not a crash, so it exits like one.
    try:
        ensure_writable(cfg["save_dir"], cfg["master_seed"], cfg.get("overwrite", False))
    except FileExistsError as e:
        print(f"\n[Trainer] refusing to start:\n  {e}\n", file=sys.stderr)
        raise SystemExit(2)

    print(f"[Config] {cfg}")

    # One seed for everything: splits / SNR (numpy, inside the dataset), batch
    # order (DataLoader generator), weight init and dropout (torch global RNG).
    seed_everything(cfg["master_seed"])
    device = resolve_device(cfg["device"])
    print(f"[Seed] master_seed = {cfg['master_seed']}  device = {device}"
          + ("  (MPS is not bit-deterministic; use --device cpu for exact repeats)"
             if device.type == "mps" else ""))

    # Data
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=cfg["data_dir"],
        artifact_type=cfg["artifact"],
        batch_size=cfg["batch_size"],
        split_ratios=tuple(cfg["split_ratios"]),
        sampling=cfg["sampling"],
        snr_range_db=tuple(cfg["snr_range_db"]),
        master_seed=cfg["master_seed"],
        target_fs=cfg["target_fs"],
        normalization=cfg["normalization"],
        normalization_scale=cfg["normalization_scale"],
        resampling=cfg["resampling"],
    )

    # Model settings, including optional quantization, are saved in checkpoints.
    model = build_model_from_config(cfg)

    # Train
    trainer_kwargs = dict(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=cfg["save_dir"],
        learning_rate=cfg["learning_rate"],
        attn_lr_scale=cfg["attn_lr_scale"],
        weight_decay=cfg["weight_decay"],
        max_epochs=cfg["max_epochs"],
        log_every=cfg["log_every"],
        sched_warmup_epochs=cfg["sched_warmup_epochs"],
        eta_min=cfg["eta_min"],
        use_tensorboard=cfg["tensorboard"],
        loss_fn=cfg["loss_fn"],
        device=device,
        run_config=cfg,
        overwrite=cfg.get("overwrite", False),
    )
    if distil:
        teacher = load_teacher(cfg["teacher_ckpt"], cfg["teacher_config"])
        trainer = KDTrainer(
            teacher=teacher,
            teacher_ckpt=cfg["teacher_ckpt"],
            kd_alpha=cfg["kd_alpha"],
            kd_feature_weight=cfg["kd_feature_weight"],
            kd_feature_layers=(cfg["kd_feature_layers"]
                               or default_feature_layers(model)),
            kd_output_loss=cfg["kd_output_loss"],
            kd_feature_loss=cfg["kd_feature_loss"],
            init_from_teacher=cfg["init_from_teacher"],
            **trainer_kwargs,
        )
    else:
        trainer = Trainer(**trainer_kwargs)

    if cfg["staged"]:
        trainer.staged_fit(
            warmup_epochs=cfg["warmup_epochs"],
            warmup_lr=cfg["warmup_lr"],
            finetune_lr=cfg["finetune_lr"],
            attn_lr_scale=cfg["attn_lr_scale"],
            weight_decay=cfg["weight_decay"],
            sched_warmup_epochs=cfg["sched_warmup_epochs"],
            eta_min=cfg["eta_min"],
        )
    else:
        trainer.fit()


if __name__ == "__main__":
    main()
