"""One schema with named compatibility profiles for CLI, GUI and Python callers."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from pathlib import Path


@dataclass
class ModelConfig:
    variant: str = "tcn_mla"
    hidden_channels: int = 64
    n_heads: int = 4
    n_attn_layers: int = 2
    max_seq_len: int = 512
    dropout: float = 0.1
    kv_lora_rank: int = 16
    bottleneck_ratio: float = 0.5
    skip_bottleneck_ratio: float | None = None
    se_reduction: int | None = None
    dilations: list | None = None
    causal: bool | None = None
    causal_tcn: bool | None = None
    lookahead: int | None = None
    turboquant: bool = False
    turboquant_bits: int = 3
    turboquant_key_mode: str = "prod"
    turboquant_train_mode: str = "off"
    turboquant_seed: int = 42
    tcn_norm: str = "channel"
    se_window: int | None = None
    attention_window: int | None = None


@dataclass
class DataConfig:
    artifact: str = "BOTH"
    data_dir: str = "./data/EEGdenoisenet/data/"
    batch_size: int = 32
    snr_range_db: tuple = (-7., 2.)
    split_ratios: tuple = (.7, .15, .15)
    sampling: str = "legacy"
    target_fs: float | None = None
    normalization: str = "fixed"
    normalization_scale: float | None = None
    master_seed: int = 42
    resampling: str = "offline"


@dataclass
class TrainConfig:
    learning_rate: float = 5e-3
    attn_lr_scale: float = .1
    weight_decay: float = 0.
    max_epochs: int = 150
    loss_fn: str = "l1_combined"
    sched_warmup_epochs: int = 5
    eta_min: float = 1e-6
    staged: bool = False
    warmup_epochs: int = 5
    warmup_lr: float = 1e-3
    finetune_lr: float = 5e-4
    teacher_ckpt: str | None = None
    teacher_config: str | None = None
    kd_alpha: float = .5
    kd_feature_weight: float = 0.
    kd_feature_layers: list | None = None
    kd_output_loss: str | None = None
    kd_feature_loss: str = "mse"
    init_from_teacher: bool = False
    protocol: str = "exploratory"
    output_delay: int = 0
    valid_history_samples: int | None = None
    valid_future_samples: int = 0
    hop: int = 4
    selection_metric: str = "gt_mse"


@dataclass
class RuntimeConfig:
    save_dir: str | None = None
    device: str = "auto"
    overwrite: bool = False
    log_every: int = 20
    tensorboard: bool = False


# Preserve existing launch behavior explicitly instead of letting defaults drift
# in parser definitions, GUI field declarations and Trainer signatures.
PROFILES = {
    "cli": {},
    "gui": dict(artifact="EOG", learning_rate=5e-4, attn_lr_scale=3., max_epochs=100,
                warmup_epochs=20),
    "api": dict(learning_rate=5e-4, attn_lr_scale=1., max_epochs=50, warmup_epochs=20,
                save_dir="./runs/exp1"),
    "paper": dict(variant="tcn_se", learning_rate=5e-4, attn_lr_scale=1.,
                  max_epochs=200, loss_fn="mse", protocol="paper",
                  tcn_norm="none", se_window=16, attention_window=32, resampling="causal",
                  valid_history_samples=200, valid_future_samples=102),
}


def defaults(profile="cli"):
    if profile not in PROFILES:
        raise ValueError(f"Unknown configuration profile: {profile}")
    cfg = {}
    for group in (ModelConfig, DataConfig, TrainConfig, RuntimeConfig):
        cfg.update(asdict(group()))
    cfg.update(PROFILES[profile])
    cfg["profile"] = profile
    return deepcopy(cfg)


STAGED_API_DEFAULTS = {**defaults("api"), "sched_warmup_epochs": 3}


def resolve_config(values=None, profile="cli"):
    values = dict(values or {})
    cfg = defaults(values.get("profile", profile))
    unknown = set(values) - set(cfg) - {"rank_sweep"}
    if unknown:
        raise ValueError(f"Unknown experiment setting(s): {', '.join(sorted(unknown))}")
    cfg.update(values)
    from models.variants import VARIANT_REGISTRY, pinned_kwargs, variant_is_causal
    if cfg["variant"] not in VARIANT_REGISTRY:
        raise ValueError(f"Unknown variant: {cfg['variant']}")
    cfg.update(pinned_kwargs(cfg["variant"]))
    if cfg["selection_metric"] != "gt_mse":
        raise ValueError("Checkpoint selection must use pure validation GT MSE")
    if cfg["protocol"] not in ("exploratory", "paper"):
        raise ValueError("protocol must be exploratory or paper")
    for name in ("output_delay", "valid_future_samples"):
        if type(cfg[name]) is not int or cfg[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for name in ("se_window", "attention_window", "valid_history_samples"):
        if cfg[name] is not None and (type(cfg[name]) is not int or cfg[name] < 1):
            raise ValueError(f"{name} must be a positive integer")
    if type(cfg["hop"]) is not int or cfg["hop"] < 1:
        raise ValueError("hop must be a positive integer")
    if cfg["tcn_norm"] not in ("none", "channel"):
        raise ValueError("New training accepts tcn_norm=none or channel. BatchNorm is supported for historical checkpoints.")
    if cfg["resampling"] not in ("offline", "causal"):
        raise ValueError("Unknown resampling method")
    if cfg["protocol"] == "paper":
        if cfg["turboquant"]:
            raise ValueError("Experimental TurboQuant is outside the validated paper streaming protocol")
        if cfg["max_epochs"] != 200 or cfg["loss_fn"] != "mse" or cfg["staged"]:
            raise ValueError("Paper protocol requires 200 epochs, GT MSE and ordinary AdamW training")
        if cfg["normalization"] != "fixed":
            raise ValueError("Paper protocol requires fixed input scale")
        if cfg["resampling"] != "causal" or cfg["hop"] != 4:
            raise ValueError("Paper protocol requires causal resampling and hop H=4")
        if cfg["valid_history_samples"] is None:
            raise ValueError("Paper protocol requires a shared valid_history_samples")
        if cfg["lookahead"] or "_la" in cfg["variant"]:
            raise ValueError("Use output_delay with an unchanged causal model, not architectural lookahead")
        if cfg["output_delay"] and not variant_is_causal(cfg["variant"]):
            raise ValueError("Delayed output requires a causal architecture")
        if cfg["output_delay"] > cfg["valid_future_samples"]:
            raise ValueError("output_delay exceeds the common future exclusion")
    for name in ("hidden_channels", "n_heads", "n_attn_layers", "max_seq_len", "batch_size",
                 "max_epochs", "log_every", "kv_lora_rank"):
        if type(cfg[name]) is not int or cfg[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(cfg["master_seed"]) is not int or cfg["master_seed"] < 0:
        raise ValueError("master_seed must be a nonnegative integer")
    for name in ("learning_rate", "attn_lr_scale", "warmup_lr", "finetune_lr"):
        if type(cfg[name]) not in (int, float) or not math.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("weight_decay", "eta_min", "kd_feature_weight"):
        if type(cfg[name]) not in (int, float) or not math.isfinite(cfg[name]) or cfg[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name in ("staged", "tensorboard", "overwrite", "init_from_teacher", "turboquant"):
        if type(cfg[name]) is not bool:
            raise ValueError(f"{name} must be a boolean")
    if cfg["artifact"] not in ("EOG", "EMG", "BOTH"):
        raise ValueError("artifact must be EOG, EMG or BOTH")
    if cfg["sampling"] not in ("legacy", "balanced"):
        raise ValueError("sampling must be legacy or balanced")
    for name in ("sched_warmup_epochs", "warmup_epochs"):
        if type(cfg[name]) is not int or cfg[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not 0 <= cfg["dropout"] < 1:
        raise ValueError("dropout must be in [0, 1)")
    for name in ("bottleneck_ratio", "skip_bottleneck_ratio"):
        if cfg[name] is not None and not 0 < cfg[name] <= 1:
            raise ValueError(f"{name} must be in (0, 1]")
    if cfg["se_reduction"] is not None and (type(cfg["se_reduction"]) is not int or cfg["se_reduction"] < 1):
        raise ValueError("se_reduction must be a positive integer")
    if cfg["turboquant_bits"] not in range(2, 9) or type(cfg["turboquant_bits"]) is not int:
        raise ValueError("turboquant_bits must be an integer in [2, 8]")
    if cfg["turboquant_key_mode"] not in ("prod", "mse") or cfg["turboquant_train_mode"] not in ("off", "ste"):
        raise ValueError("Unknown TurboQuant mode")
    if cfg["kd_feature_loss"] not in ("mse", "l1"):
        raise ValueError("kd_feature_loss must be mse or l1")
    if cfg["normalization"] not in ("fixed", "epoch"):
        raise ValueError("normalization must be fixed or epoch")
    if cfg["normalization_scale"] is not None:
        from data.protocol import normalization_config
        normalization_config(cfg)
    if cfg["target_fs"] is not None and (not math.isfinite(cfg["target_fs"]) or cfg["target_fs"] <= 0):
        raise ValueError("target_fs must be finite and positive")
    if cfg["device"] not in ("auto", "cpu", "cuda", "mps"):
        raise ValueError("Unknown device")
    ratios = cfg["split_ratios"]
    if len(ratios) != 3 or any(not math.isfinite(x) or x <= 0 for x in ratios) or abs(sum(ratios)-1) > 1e-6:
        raise ValueError("split_ratios must contain three positive values summing to 1")
    snr = cfg["snr_range_db"]
    if len(snr) != 2 or not all(math.isfinite(x) for x in snr) or snr[0] > snr[1]:
        raise ValueError("snr_range_db must be an increasing finite pair")
    if not 0 <= cfg["kd_alpha"] <= 1:
        raise ValueError("kd_alpha must be in [0, 1]")
    from trainer.losses import LOSS_REGISTRY
    if cfg["loss_fn"] not in LOSS_REGISTRY or cfg["kd_output_loss"] not in (None, *LOSS_REGISTRY):
        raise ValueError("Unknown task or KD output loss")
    return cfg


def resolve_save_dir(cfg, cwd=Path(".")):
    """Separate fixed-scale runs from legacy epoch-scale paths and seeds."""
    if cfg.get("save_dir") is not None:
        return cfg
    from data.Dataloader import NATIVE_FS
    from trainer.checkpoint import recorded_seed
    rate = str(float(cfg.get("target_fs") or NATIVE_FS)).removesuffix(".0").replace("+", "")
    base = f"./runs/{cfg['variant']}_{cfg['artifact'].lower()}" + ("_kd" if cfg.get("teacher_ckpt") else "")
    base += f"_{rate}hz"
    # Older paths did not record input normalization in their names. Keep
    # epoch-scale naming compatible, but never send a new fixed-scale run
    # into that same namespace. Explicit save_dir values remain authoritative.
    if cfg.get("normalization") == "fixed":
        base += "_fixed"
    seeded = f"{base}_s{cfg['master_seed']}"
    held, files = recorded_seed(Path(cwd) / base)
    cfg["save_dir"] = (seeded if cfg["master_seed"] != 42 or
                       (files and held is not None and held != cfg["master_seed"]) else base)
    return cfg
