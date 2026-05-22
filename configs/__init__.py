"""
Configuration loader for YAML experiment configs.

Usage:
    from configs import load_config
    cfg = load_config("configs/tcn_mha_eog.yaml")
    # cfg is a plain dict: cfg["learning_rate"], cfg["variant"], etc.

CLI overrides take precedence over the YAML file — this lets you
share one config across similar runs while tweaking a single knob
from the command line:

    python trainer/trainer.py --config configs/tcn_mha_eog.yaml --epochs 200
"""

import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    """
    Load a YAML config file and return its contents as a flat dict.

    Raises FileNotFoundError if the path doesn't exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_cli_overrides(cfg: dict, cli_args) -> dict:
    """
    Merge CLI argparse values into the config dict.

    Rules:
      - CLI args that were explicitly provided (not None) override the YAML.
      - YAML keys not present in CLI remain untouched.
      - CLI args not present in YAML are added (forward-compatible).

    This gives a priority chain:  CLI flag > YAML file > Trainer defaults.
    """
    merged = dict(cfg)
    for key, val in vars(cli_args).items():
        # Skip the config path itself
        if key == "config":
            continue
        # Only override if the CLI arg was explicitly set
        if val is not None:
            merged[key] = val
    return merged
