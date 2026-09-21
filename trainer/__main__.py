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
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from data.Dataloader import build_dataloaders
from models.variants import (build_model_from_config, VARIANT_REGISTRY, variant_is_causal,
                             pinned_kwargs)
from configs import load_config, merge_cli_overrides
from trainer.training import Trainer
from trainer.distillation import KDTrainer, load_teacher
from utils.utils import seed_everything, resolve_device, DEVICE_CHOICES

DEFAULT_DATA_DIR = "./data/EEGdenoisenet/data/"


def main():
    # --- CLI args (all optional — YAML config provides defaults) ---
    parser = argparse.ArgumentParser(description="Train an EEG causal denoising model")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (e.g. configs/tcn_mha_eog.yaml)")
    parser.add_argument("--variant", default=None, choices=list(VARIANT_REGISTRY.keys()))
    parser.add_argument("--artifact", default=None, choices=["EOG", "EMG"])
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--save_dir", default=None,
                        help="Checkpoint directory (default: runs/<variant>_<artifact>)")
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
                        help="Use two-phase staged training (freeze TCN → joint finetune)")
    parser.add_argument("--warmup_epochs", type=int, default=None,
                        help="Phase 1 epochs (attention-only warm-up)")
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
    parser.add_argument("--master_seed", type=int, default=None,
                        help="Seed for data splits, SNR draws, weight init, dropout and "
                             "batch order (default 42)")
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
    args = parser.parse_args()

    # --- Build effective config: YAML defaults -> CLI overrides ---
    if args.config:
        cfg = load_config(args.config)
        cfg = merge_cli_overrides(cfg, args)
    else:
        cfg = {k: v for k, v in vars(args).items() if v is not None}

    # Apply fallback defaults for anything still missing
    cfg.setdefault("variant",            "tcn_mla")
    cfg.setdefault("artifact",           "BOTH")
    cfg.setdefault("data_dir",           DEFAULT_DATA_DIR)
    cfg.setdefault("hidden_channels",    64)
    cfg.setdefault("batch_size",         32)
    cfg.setdefault("learning_rate",      5e-3)
    cfg.setdefault("attn_lr_scale",      0.1)
    cfg.setdefault("weight_decay",       0.0)
    cfg.setdefault("max_epochs",         150)
    cfg.setdefault("log_every",          20)
    cfg.setdefault("sched_warmup_epochs", 5)
    cfg.setdefault("eta_min",            1e-6)
    cfg.setdefault("tensorboard",        False)
    cfg.setdefault("staged",             False)
    cfg.setdefault("warmup_epochs",      5)
    cfg.setdefault("warmup_lr",          1e-3)
    cfg.setdefault("finetune_lr",        5e-4)
    cfg.setdefault("loss_fn",            "mse")
    cfg.setdefault("bottleneck_ratio",  0.5)
    cfg.setdefault("skip_bottleneck_ratio", None)
    cfg.setdefault("kv_lora_rank",      16)
    cfg.setdefault("snr_range_db",      [-7.0, 2.0])
    cfg.setdefault("master_seed",       42)
    cfg.setdefault("device",            "auto")
    cfg.setdefault("teacher_ckpt",      None)
    cfg.setdefault("teacher_config",    None)
    cfg.setdefault("kd_alpha",          0.5)
    cfg.setdefault("kd_feature_weight", 0.0)
    cfg.setdefault("kd_feature_layers", ["decoder"])
    cfg.setdefault("kd_output_loss",    None)
    cfg.setdefault("kd_feature_loss",   "mse")
    cfg.setdefault("init_from_teacher", False)

    cfg.setdefault("n_attn_layers", 2)
    if type(cfg["n_attn_layers"]) is not int or cfg["n_attn_layers"] < 1:
        parser.error("n_attn_layers must be a positive integer")

    # A rank alias (tcn_mla_r8 → kv_lora_rank=8) outranks the generic
    # kv_lora_rank that the YAML / the default above carry for tcn_mla, so the
    # printed config, the checkpoint and the model built from it all agree.
    for key, value in pinned_kwargs(cfg["variant"]).items():
        if cfg.get(key) not in (None, value):
            print(f"[Config] variant '{cfg['variant']}' fixes {key}={value}; "
                  f"overriding {key}={cfg[key]}")
        cfg[key] = value

    distil = cfg["teacher_ckpt"] is not None
    if cfg.get("save_dir") is None:
        cfg["save_dir"] = f"./runs/{cfg['variant']}_{cfg['artifact'].lower()}" \
            + ("_kd" if distil else "")
    if distil and not variant_is_causal(cfg["variant"]):
        print(f"[Warning] Student '{cfg['variant']}' is bidirectional; distillation is meant "
              f"to teach a causal student. Continuing anyway.")

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
        snr_range_db=tuple(cfg["snr_range_db"]),
        master_seed=cfg["master_seed"],
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
    )
    if distil:
        teacher = load_teacher(cfg["teacher_ckpt"], cfg["teacher_config"])
        trainer = KDTrainer(
            teacher=teacher,
            teacher_ckpt=cfg["teacher_ckpt"],
            kd_alpha=cfg["kd_alpha"],
            kd_feature_weight=cfg["kd_feature_weight"],
            kd_feature_layers=cfg["kd_feature_layers"],
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
