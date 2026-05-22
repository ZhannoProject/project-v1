"""
CLI entry point for training.

Usage:
    python -m trainer --config configs/tcn_mha_eog.yaml
    python -m trainer --variant tcn --artifact EOG --max_epochs 50

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
from models.variants import build_model, VARIANT_REGISTRY
from configs import load_config, merge_cli_overrides
from trainer.training import Trainer

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
                        help="KV low-rank bottleneck dim for MLA variants (default: attn_dim // 4)")
    parser.add_argument("--bottleneck_ratio", type=float, default=None,
                        help="Channel squeeze ratio between TCN and attention "
                             "(e.g. 0.5 → attn_dim = hidden_channels * 0.5). "
                             "Default: 0.5 for sandwich MHA/MLA variants.")
    args = parser.parse_args()

    # --- Build effective config: YAML defaults -> CLI overrides ---
    if args.config:
        cfg = load_config(args.config)
        cfg = merge_cli_overrides(cfg, args)
    else:
        cfg = {k: v for k, v in vars(args).items() if v is not None}

    # Apply fallback defaults for anything still missing
    cfg.setdefault("variant",            "tcn_mla")
    cfg.setdefault("artifact",           "EOG")
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
    cfg.setdefault("kv_lora_rank",      8)

    if cfg.get("save_dir") is None:
        cfg["save_dir"] = f"./runs/{cfg['variant']}_{cfg['artifact'].lower()}"

    print(f"[Config] {cfg}")

    # Data
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=cfg["data_dir"],
        artifact_type=cfg["artifact"],
        batch_size=cfg["batch_size"],
    )

    # Model — collect architecture kwargs, only forward those the variant accepts
    import inspect
    cls = VARIANT_REGISTRY[cfg["variant"]]
    sig = inspect.signature(cls)

    model_kwargs = {"hidden_channels": cfg["hidden_channels"]}
    if "kv_lora_rank" in sig.parameters and cfg.get("kv_lora_rank") is not None:
        model_kwargs["kv_lora_rank"] = cfg["kv_lora_rank"]
    if "bottleneck_ratio" in sig.parameters and cfg.get("bottleneck_ratio") is not None:
        model_kwargs["bottleneck_ratio"] = cfg["bottleneck_ratio"]
    model = build_model(cfg["variant"], **model_kwargs)

    # Train
    trainer = Trainer(
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
    )

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
