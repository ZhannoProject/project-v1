"""
Configuration loader for YAML experiment configs.

Usage:
    from configs import load_config
    cfg = load_config("configs/tcn_mha_eog.yaml")
    # cfg is a plain dict: cfg["learning_rate"], cfg["variant"], etc.

CLI overrides take precedence over the YAML file — this lets you
share one config across similar runs while tweaking a single knob
from the command line:

    python -m trainer --config configs/tcn_mha_eog.yaml --max_epochs 200
"""

import yaml
import json
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
        # PyYAML treats JSON exponent-only numbers such as 1e-06 as strings.
        # Frozen GUI commands and run snapshots must preserve numeric types.
        cfg = json.load(f) if p.suffix.lower() == ".json" else yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {p}")
    if "schema_version" in cfg and "run_config" in cfg:
        from experiment.storage import SCHEMA_VERSION
        if type(cfg["schema_version"]) is not int or cfg["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"Unsupported configuration snapshot version: {cfg['schema_version']}")
        saved = dict(cfg["run_config"])
        for key in ("arch", "optimizer", "n_params"):
            saved.pop(key, None)
        saved.update(cfg.get("model_config") or {})
        # A saved execution permission is not permission for a later replay.
        saved["overwrite"] = False
        cfg = saved
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
