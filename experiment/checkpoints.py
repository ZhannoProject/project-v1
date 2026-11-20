"""Versioned checkpoint reading and model reconstruction shared by every entry point."""

from collections.abc import Mapping
import json
from pathlib import Path

from experiment.storage import SCHEMA_VERSION


FIRST_TCN_CONVS = ("tcn_pt1.network.0.conv_block.0.conv.weight",
                   "backbone.network.0.conv_block.0.conv.weight")


def infer_tcn_norm(state):
    """Read the normalization from TCN tensors, including historical weights."""
    first_conv = next((key for key in FIRST_TCN_CONVS if key in state), None)
    if first_conv is None:
        return None
    prefix = first_conv.removesuffix("0.conv.weight") + "2."
    if prefix + "running_mean" in state:
        return "batch"
    if prefix + "norm.weight" in state:
        return "channel"
    if not any(key.startswith(prefix) for key in state):
        return "none"
    return None


def normalize_checkpoint(raw):
    if not isinstance(raw, Mapping) or not isinstance(raw.get("model_state"), Mapping):
        raise ValueError("Checkpoint must contain a model_state mapping")
    version = raw.get("schema_version", 0)
    if type(version) is not int or version not in (0, SCHEMA_VERSION):
        raise ValueError(f"Unsupported checkpoint schema_version: {version!r}")
    out = dict(raw)
    cfg = dict((out.get("history") or {}).get("config") or {})
    cfg.update(out.get("run_config") or {})
    arch = dict(cfg.get("arch") or {})
    arch.update(out.get("model_config") or {})
    if arch.get("tcn_norm") is None:
        inferred_norm = infer_tcn_norm(out["model_state"])
        if inferred_norm is not None:
            arch["tcn_norm"] = inferred_norm
    out.update(schema_version=SCHEMA_VERSION, source_schema_version=version,
               model_config=arch, run_config=cfg)
    return out


def read_checkpoint(path):
    import torch
    return normalize_checkpoint(torch.load(path, map_location="cpu", weights_only=False))


def infer_arch_from_state_dict(sd: dict) -> dict:
    """
    Recover every shape-determined constructor argument from the weights.

    Returns a dict accepted by build_model(**cfg) minus ``n_heads``, which
    is not recoverable (all head counts give identical tensor shapes).
    """
    cfg = {}

    # First conv of the first TCN block: (hidden_channels, in_channels=1, k)
    first_conv = next((key for key in FIRST_TCN_CONVS if key in sd), None)
    if first_conv is None:
        raise ValueError("Unrecognised state dict: no TCN first-layer weight found")
    hidden = sd[first_conv].shape[0]
    cfg["hidden_channels"] = int(hidden)
    norm = infer_tcn_norm(sd)
    if norm is not None:
        cfg["tcn_norm"] = norm

    # The SE gate is visible in the weights: a tcn_se checkpoint carries
    # se.fc1/se.fc2 that a plain tcn does not. Without this it would be read
    # back as "tcn", and the strict load would fail on the extra keys -- or
    # worse, a future non-strict path would quietly drop the gate.
    se_key = next((k for k in sd if k.endswith("se.fc1.weight")), None)
    if se_key is not None:
        # (C/r, C, 1): the squeeze ratio is the shape, not a stored number.
        hidden_se, chans = sd[se_key].shape[0], sd[se_key].shape[1]
        cfg["se_reduction"] = max(1, int(round(chans / hidden_se)))

    block_ids = {int(k.split(".")[2]) for k in sd if k.startswith("decoder.blocks.")}
    if not block_ids:
        cfg["variant"] = "tcn_se" if se_key is not None else "tcn"
        return cfg
    cfg["n_attn_layers"] = max(block_ids) + 1

    attn = "decoder.blocks.0.attn."
    is_mla = attn + "kv_down_proj.weight" in sd
    is_par = "squeeze.conv.weight" not in sd
    cfg["variant"] = {(False, False): "tcn_mha", (True, False): "tcn_mla",
                      (False, True): "tcn_mha_par", (True, True): "tcn_mla_par"}[(is_mla, is_par)]

    if not is_par:
        attn_dim = sd["squeeze.conv.weight"].shape[0]
        # The model computes int(hidden * ratio); (attn_dim + 0.5) / hidden
        # guarantees that truncation lands exactly on attn_dim even when the
        # ratio is not representable in binary floating point.
        cfg["bottleneck_ratio"] = (attn_dim + 0.5) / hidden
    if is_mla:
        cfg["kv_lora_rank"] = int(sd[attn + "kv_down_proj.weight"].shape[0])
    skip_w = "decoder.blocks.0.attn_skip.net.0.weight"        # (bottleneck_dim, d_model)
    if skip_w in sd:                                          # optional BottleneckSkip
        skip_dim, d_model = sd[skip_w].shape
        cfg["skip_bottleneck_ratio"] = (skip_dim + 0.5) / d_model   # same truncation trick
    cfg["max_seq_len"] = int(sd[attn + "causal_mask"].shape[-1])
    return cfg


def complete_architecture(model):
    from models.variants import resolved_arch
    arch = resolved_arch(model)
    try:
        inferred = infer_arch_from_state_dict(model.state_dict())
    except ValueError:  # hand-built models outside the registry
        return arch
    for key, value in inferred.items():
        if arch.get(key) is None:
            arch[key] = value
    if hasattr(model, "dilations"):
        arch["dilations"] = list(model.dilations)
    return arch


def load_model(source, config=None, overrides=None, device="cpu", for_export=False, freeze=False):
    """Load with checkpoint metadata first and explicit export/evaluation policy."""
    from configs import load_config
    from data.protocol import bind_run_config
    from models.variants import (base_variant, pinned_kwargs, variant_is_causal,
                                 build_model_from_config, resolved_arch)
    path = None if isinstance(source, Mapping) else Path(source)
    raw = normalize_checkpoint(source) if path is None else read_checkpoint(path)
    fallback = {}
    if path is not None:
        for name in ("config.json", "resolved_config.json"):
            candidate = path.parent / name
            if candidate.is_file():
                saved = json.loads(candidate.read_text())
                if name == "resolved_config.json":
                    if not raw.get("run_id") or saved.get("run_id") != raw["run_id"]:
                        continue
                    saved = saved.get("run_config") or {}
                fallback.update(saved)
    if config is not None:
        fallback.update(load_config(str(config)) if not isinstance(config, Mapping) else config)
    sd = raw["model_state"]
    stored = raw["model_config"]
    inferred = infer_arch_from_state_dict(sd)
    label = stored.get("variant") or fallback.get("variant")
    family = base_variant(label) if label else None
    optional_se = family == "tcn" and inferred["variant"] == "tcn_se" and (
        stored.get("se_reduction") or fallback.get("se_reduction"))
    if label and family != inferred["variant"] and not optional_se:
        raise ValueError(f"Checkpoint model_config says variant={label} but the weights imply "
                         f"{inferred['variant']}; refusing to guess.")
    for key, value in (pinned_kwargs(label) if label else {}).items():
        if key in inferred and inferred[key] != value:
            raise ValueError(f"Checkpoint is labelled '{label}', which fixes {key}={value}, "
                             f"but the weights imply {key}={inferred[key]}")
    for key in ("hidden_channels", "n_attn_layers", "kv_lora_rank", "max_seq_len", "tcn_norm"):
        if stored.get(key) is not None and key in inferred and stored[key] != inferred[key]:
            raise ValueError(f"Checkpoint model_config says {key}={stored[key]} but the weights "
                             f"imply {key}={inferred[key]}; refusing to guess.")
    cfg = {**fallback, **inferred,
           **{key: value for key, value in stored.items() if value is not None or key not in inferred}}
    cfg["variant"] = label or inferred["variant"]
    if "n_attn_layers" in inferred:
        cfg.setdefault("n_heads", 4)   # historical head count cannot be inferred from tensor shapes
    changes = {k: v for k, v in (overrides or {}).items() if v is not None}
    if "variant" in changes and changes["variant"] != cfg["variant"]:
        raise ValueError("Explicit variant differs from checkpoint variant")
    cfg.update(changes)
    if for_export:
        # A bounded-lookahead rung is deployable as a causal system whose output
        # is delayed by its L samples; the exporter verifies that contract. Only
        # an unbounded (bidirectional) model cannot run online at all.
        bounded = isinstance(cfg.get("lookahead"), int) and cfg["lookahead"] > 0
        if not variant_is_causal(cfg["variant"]) and not bounded:
            raise ValueError(f"{source} is a bidirectional teacher "
                             f"({cfg['variant']}); it cannot run online. Export a causal student "
                             f"or a bounded-lookahead model instead.")
        cfg["turboquant"] = False
    model = build_model_from_config(cfg)
    model.load_state_dict(sd, strict=True)
    bind_run_config(model, raw, fallback)
    if model.run_config.get("protocol") == "paper":
        from experiment.methodology import validate_model
        import math
        rate = model.run_config.get("target_fs") or 256.
        validate_model(model, model.run_config, math.ceil(512 * rate / 256.))
    raw["run_config"] = dict(model.run_config)
    raw["model_config"] = complete_architecture(model)
    model.eval().to(device or "cpu")
    if freeze:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return model, complete_architecture(model), raw
