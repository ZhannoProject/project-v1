"""
Where a variant's parameters go.

The Runs tab prints one number per checkpoint ("319k params"), which is
enough to notice that two models differ and not enough to see why.  This
module builds the model the form currently describes and takes it apart:
per stage, and inside the attention decoder per projection, with the
per-layer cost next to the total.

Two totals are reported and they are not the same number:

  trainable    sum of ``model.parameters()`` — the weights being optimised
  checkpoint   what the Runs tab counts — every tensor in ``model_state``
               except the ``causal_mask`` buffers.  Larger than ``trainable``
               by the BatchNorm running statistics, which are saved but
               never optimised.

Nothing is cached: a few-hundred-thousand-parameter model builds on CPU in
milliseconds, so the page can recompute on every keystroke.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import re
from typing import Any, Dict, List, Optional

from gui.config_io import SCHEMA, RUNS_DIR

# Sections of SCHEMA that reach the model constructor.  Everything else
# (data, optimiser, loss, KD) cannot change a parameter count.
MODEL_SECTIONS = ("Model", "TurboQuant")
BYTES_FP32 = 4

# Ceiling on what the counter will allocate.  This runs on every keystroke, so
# one stray digit in hidden_channels (640 → 6400) would otherwise ask torch for
# tens of gigabytes and take the GUI down with it.  200M parameters is ~800 MB
# of fp32 and roughly a thousand times the largest variant in the study, so
# nothing real trips it.
MAX_PARAMS = 200_000_000


def model_defaults() -> Dict[str, Any]:
    """Schema defaults for the keys that build the model (mirrors trainer/__main__.py)."""
    return {f.key: f.default
            for sec in SCHEMA if sec.title in MODEL_SECTIONS
            for f in sec.fields}


def resolve(values: Optional[dict]) -> Dict[str, Any]:
    """
    Form values laid over the schema defaults, keeping only model keys.

    This runs on every keystroke, so a field mid-edit is the normal case, not
    an error to be surprised by: an empty depth box arrives as 0 or None and
    must come back as a sentence the page can show, never a traceback.  The
    constructors catch most of it; the two checks here cover what they do not.
    """
    if values is not None and not isinstance(values, dict):
        raise ValueError(f"values must be an object, got {type(values).__name__}")
    cfg = model_defaults()
    for key, value in (values or {}).items():
        if key in cfg:
            cfg[key] = value
    if not cfg.get("variant"):
        cfg["variant"] = "tcn_mla"
    if not isinstance(cfg["variant"], str):
        raise ValueError(f"variant must be a name, got {type(cfg['variant']).__name__}")
    # A decoder with no blocks builds without complaint and then has nothing to
    # describe.  The trainer rejects it too, in the same words.
    depth = cfg.get("n_attn_layers")
    if depth is not None and (type(depth) is not int or depth < 1):
        raise ValueError(f"n_attn_layers must be a positive integer, got {depth!r}")
    return cfg


def _notes(cfg: Dict[str, Any]) -> List[str]:
    """
    Settings that are on screen but do nothing for this variant.

    Two ways that happens: the name pins the value (``tcn_mla_r16`` fixes the
    rank, ``*_bi`` fixes causality), or the constructor has no such argument
    at all — ``bottleneck_ratio`` is a sandwich-only knob, so the parallel
    models run attention at the full TCN width whatever the form says.
    Only keys moved away from their default are worth a line; the rest are
    defaults nobody chose.
    """
    from models.variants import VARIANT_REGISTRY, pinned_kwargs

    variant = cfg["variant"]
    if variant not in VARIANT_REGISTRY:
        raise ValueError(f"Unknown variant {variant!r}. "
                         f"Available: {', '.join(VARIANT_REGISTRY)}")
    accepted = set(inspect.signature(VARIANT_REGISTRY[variant]).parameters)
    pinned = pinned_kwargs(variant)
    defaults = model_defaults()
    notes = []
    for key, value in cfg.items():
        if key == "variant" or key.startswith("turboquant"):
            continue
        if key in pinned:
            if value is not None and value != pinned[key]:
                notes.append(f"{variant} fixes {key} = {pinned[key]!r}; "
                             f"the form's {value!r} is ignored.")
        elif key not in accepted and value != defaults[key]:
            notes.append(f"{key} = {value!r} is not an argument of {variant} "
                         f"and changes nothing here.")
    return notes


# ============================================================
# Breakdown
# ============================================================
def _estimate(cfg: Dict[str, Any]) -> int:
    """
    Rough upper bound on the parameter count, computed from the numbers alone.

    Deliberately crude — it only has to be the right order of magnitude to stop
    an absurd config before ``torch`` tries to allocate it.  A residual block is
    two k=3 convolutions (~6H² each) and there are at most six of them; a
    transformer layer is ~4d² of attention plus ~8d² of FFN.
    """
    def number(key, absent, unusable):
        """``absent`` when the key is unset, ``unusable`` when it is not a number.

        A value this cannot read must never switch the ceiling off — a string
        ``bottleneck_ratio`` alongside a mistyped width would then let the very
        allocation this exists to prevent straight through.  Each unreadable
        field falls back to the value that keeps the bound conservative.
        """
        v = cfg.get(key)
        if v is None:
            return absent
        # bool is an int in Python and is never a width or a depth.
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return unusable
        return v

    # A non-numeric width is the one case that is safe to skip: Conv1d rejects it
    # before reserving anything, so there is nothing to protect against.
    h = number("hidden_channels", 0, None)
    if h is None:
        return 0
    layers = number("n_attn_layers", 0, 0)      # a decoder that cannot build costs nothing
    ratio = number("bottleneck_ratio", 1, 1)    # unreadable ratio → assume full width
    d = h * (ratio if ratio > 0 else 1)
    return int(12 * 6 * h * h + 12 * layers * d * d)


def _count(module) -> int:
    return sum(p.numel() for p in module.parameters())


def _describe(module) -> str:
    """A one-line shape for the modules that carry weights; class name otherwise."""
    from models.TCN_backbone import CausalConv1d

    if isinstance(module, CausalConv1d):
        conv = module.conv
        return f"1D conv · {conv.in_channels}→{conv.out_channels} · k={conv.kernel_size[0]}"
    cls = type(module).__name__
    if cls == "Conv1d":
        return f"conv · {module.in_channels}→{module.out_channels} · k={module.kernel_size[0]}"
    if cls == "Linear":
        return f"linear · {module.in_features}→{module.out_features}"
    if cls == "LayerNorm":
        return f"LayerNorm · {module.normalized_shape[0]}"
    return cls


# Order the decoder rows the way the forward pass runs, not the way
# named_parameters() happens to yield them.
_DECODER_ORDER = ["attn.qkv_proj", "attn.q_proj", "attn.kv_down_proj", "attn.kv_norm",
                  "attn.k_up_proj", "attn.v_up_proj", "attn.out_proj",
                  "attn_skip", "norms", "ffn", "ln_f"]


def _decoder_labels(decoder) -> Dict[str, str]:
    """Row labels carrying the actual dimensions, so they cannot go stale."""
    block = decoder.blocks[0]
    d = block.ln_1.normalized_shape[0]
    r = getattr(block.attn, "kv_lora_rank", None)
    skip = getattr(decoder, "skip_bottleneck_ratio", None)
    return {
        "attn.qkv_proj":     f"qkv_proj — fused Q, K, V · {d}→{3 * d}",
        "attn.q_proj":       f"q_proj — queries · {d}→{d}",
        "attn.kv_down_proj": f"kv_down_proj — compress · {d}→{r}",
        "attn.kv_norm":      f"kv_norm — LayerNorm on the {r}-d latent",
        "attn.k_up_proj":    f"k_up_proj — decompress · {r}→{d}",
        "attn.v_up_proj":    f"v_up_proj — decompress · {r}→{d}",
        "attn.out_proj":     f"out_proj · {d}→{d}",
        "attn_skip":         f"attention skip bottleneck · ratio {skip}",
        "norms":             "block LayerNorms · ln_1 + ln_2",
        "ffn":               f"FFN — 4× expansion · {d}→{4 * d}→{d}",
        "ln_f":              "final LayerNorm",
    }


def _decoder_group(name: str, decoder) -> dict:
    """
    One row per projection, summed over the block stack.

    Every block holds an independent copy of the same shapes, so a bucket's
    total divides exactly by the number of layers — the per-layer figure is
    what the reader needs to see how depth scales the model.
    """
    n_layers = len(decoder.blocks)
    if not n_layers:                       # resolve() rejects this; belt and braces
        return {"key": name, "label": name, "note": "no blocks",
                "params": _count(decoder), "items": [], "subtotals": {}}
    buckets: Dict[str, int] = {}
    for pname, p in decoder.named_parameters():
        parts = pname.split(".")
        if parts[0] == "blocks":                      # blocks.<i>.<module>...
            sub = parts[2:]
            if sub[0] in ("ln_1", "ln_2"):
                key = "norms"
            elif sub[0] == "attn" and len(sub) > 2:
                key = "attn." + sub[1]
            else:
                key = sub[0]                          # ffn, attn_skip
        else:
            key = parts[0]                            # ln_f
        buckets[key] = buckets.get(key, 0) + p.numel()

    labels = _decoder_labels(decoder)
    rank = _DECODER_ORDER.index
    order = sorted(buckets, key=lambda k: rank(k) if k in _DECODER_ORDER else len(_DECODER_ORDER))
    items = [{"key": k, "label": labels.get(k, k), "params": buckets[k],
              "per_layer": buckets[k] // n_layers if k != "ln_f" and n_layers else None}
             for k in order]

    d = decoder.blocks[0].ln_1.normalized_shape[0]
    kind = decoder.attention_type.upper()
    return {"key": name, "label": name,
            "note": f"{n_layers} × {kind} block · d_model {d}"
                    f"{'' if decoder.causal else ' · bidirectional'}",
            "params": _count(decoder), "items": items,
            "subtotals": {
                "attention": sum(v for k, v in buckets.items() if k.startswith("attn")),
                "ffn": buckets.get("ffn", 0),
                "norms": buckets.get("norms", 0) + buckets.get("ln_f", 0),
            }}


def _groups(model) -> List[dict]:
    """One entry per top-level stage, in forward order (named_children preserves it)."""
    from models.TCN_backbone import CausalTCNBackbone, CausalConv1d
    from models.AttentionBlocks.attention import CausalTransformerDecoder

    out = []
    for name, child in model.named_children():
        if isinstance(child, CausalTransformerDecoder):
            out.append(_decoder_group(name, child))
        elif isinstance(child, CausalTCNBackbone):
            items = [{"key": f"network.{i}", "params": _count(block),
                      "label": f"residual block · dilation {child.dilations[i]}"}
                     for i, block in enumerate(child.network)]
            out.append({"key": name, "label": name,
                        "note": f"dilations {', '.join(str(d) for d in child.dilations)}"
                                f" · width {child.hidden_channels}",
                        "params": _count(child), "items": items})
        elif isinstance(child, CausalConv1d):
            out.append({"key": name, "label": name, "note": _describe(child),
                        "params": _count(child), "items": []})
        else:
            # Anything else (the TCN-only head is a two-conv Sequential): one
            # row per child that holds weights, so the shapes are still visible.
            items = [{"key": f"{sub}", "label": f"{sub} · {_describe(grandchild)}",
                      "params": _count(grandchild)}
                     for sub, grandchild in child.named_children() if _count(grandchild)]
            out.append({"key": name, "label": name,
                        "note": _describe(child) if not items else
                                f"{type(child).__name__} of {len(items)}",
                        "params": _count(child), "items": items})
    return out


def _attention_facts(model) -> Optional[dict]:
    """
    What the attention stack actually runs at, and what it costs to stream.

    ``kv_per_step`` is the state an online decoder must keep for one new
    sample: MHA caches K and V at full width, MLA caches only the latent.
    Counted in fp32 elements from the architecture alone — TurboQuant
    quantises those tensors further and is not folded in here.
    """
    from models.AttentionBlocks.attention import CausalTransformerDecoder

    decoder = next((m for m in model.modules()
                    if isinstance(m, CausalTransformerDecoder)), None)
    if decoder is None or not len(decoder.blocks):
        return None
    attn = decoder.blocks[0].attn
    d = decoder.blocks[0].ln_1.normalized_shape[0]
    n_layers = len(decoder.blocks)
    rank = getattr(attn, "kv_lora_rank", None)
    per_layer = rank if rank is not None else 2 * d
    return {
        "attention_type": decoder.attention_type,
        "causal": bool(decoder.causal),
        "d_model": d,
        "n_heads": attn.n_heads,
        "head_dim": attn.head_dim,
        "n_layers": n_layers,
        "kv_lora_rank": rank,
        "kv_per_step": per_layer * n_layers,
        "kv_bytes_per_step": per_layer * n_layers * BYTES_FP32,
        # How much smaller the cache is than the MHA it replaces, at this width
        "kv_compression": round(2 * d / rank, 2) if rank else 1.0,
        "turboquant_bits": (getattr(attn.turboquant, "bits", None)
                            if getattr(attn, "turboquant", None) is not None else None),
    }


def report(values: Optional[dict] = None, *, config: Optional[dict] = None,
           source: str = "form") -> dict:
    """
    Build the model and describe its parameter count.

    ``values`` are form values (merged over the schema defaults); ``config``
    is a checkpoint's recorded ``model_config``, used verbatim so a run's row
    reports the architecture that actually trained rather than a
    reconstruction of it.
    """
    from models.variants import build_model_from_config

    cfg = dict(config) if config is not None else resolve(values)
    notes = [] if config is not None else _notes(cfg)

    # A half-typed form is the normal case here (n_heads 3 against d_model 64,
    # a width of 0 while the digit is being replaced).  The constructor says
    # exactly what is wrong; the page wants that sentence, not a traceback.
    size = _estimate(cfg)
    if size > MAX_PARAMS:
        raise ValueError(
            f"that is roughly a {size / 1e6:,.0f}M-parameter model; the counter stops at "
            f"{MAX_PARAMS // 1_000_000}M so a mistyped width cannot exhaust memory")
    try:
        # build_model_from_config narrates what a variant name overrides.  That
        # belongs in the trainer's log, not once per keystroke on the server
        # console — the same information comes back in ``notes``.
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_model_from_config(cfg)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"{cfg.get('variant')}: {exc}") from exc
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    # Match the Runs tab exactly: state_dict minus the causal_mask buffers.
    saved = total + sum(b.numel() for name, b in model.named_buffers()
                        if not name.endswith("causal_mask"))
    masks = sum(b.numel() for name, b in model.named_buffers()
                if name.endswith("causal_mask"))

    return {
        "source": source,
        "variant": cfg.get("variant"),
        "config": {k: v for k, v in cfg.items() if not k.startswith("turboquant")},
        "notes": notes,
        # Two independent axes.  ``causal`` is the attention mask (and, for the
        # attention-free baseline, the whole model); ``causal_tcn`` is the
        # convolution padding.  TCNOnly has only the first, so it answers for
        # both — tcn_bi is non-causal throughout.
        "causal": bool(getattr(model, "causal", True)),
        "causal_tcn": bool(getattr(model, "causal_tcn", getattr(model, "causal", True))),
        "trainable": trainable,
        "frozen": total - trainable,
        "checkpoint": saved,
        "buffers": saved - total,
        "mask_elements": masks,
        "bytes_fp32": trainable * BYTES_FP32,
        "groups": _groups(model),
        "attention": _attention_facts(model),
    }


def report_for_run(name: str) -> dict:
    """
    The same breakdown for a finished run, from the architecture its
    checkpoint recorded.  Runs saved before ``model_config`` existed cannot
    be rebuilt this way and say so rather than guessing from the name.
    """
    # Same guard as config_io.read_run_history: a name-shaped string, and the
    # resolved directory really sitting directly inside runs/.  The regex alone
    # admits ".." — only the second check rules it out.
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name or ""):
        raise ValueError(f"invalid run name {name!r}")
    d = (RUNS_DIR / name).resolve()
    if d.parent != RUNS_DIR.resolve():
        raise ValueError("run must be a directory directly inside runs/")
    if not d.is_dir():
        # Kept separate from the sandbox check: a typo deserves "no such run",
        # not a lecture about where runs live.
        raise FileNotFoundError(f"no run directory at runs/{name}")
    path = d / "best.pt"
    if not path.is_file():
        raise FileNotFoundError(f"no checkpoint at runs/{name}/best.pt")
    import torch
    cfg = dict((torch.load(path, map_location="cpu",
                           weights_only=False).get("model_config") or {}))
    if not cfg.get("variant"):
        raise ValueError(f"runs/{name} predates model_config; its architecture "
                         f"is not recorded in the checkpoint")
    out = report(config=cfg, source=f"runs/{name}")
    out["run"] = name
    return out
