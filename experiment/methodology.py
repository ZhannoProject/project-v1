"""Shared alignment, finite-history contracts and publication checks.

Output at acquisition index t+ell estimates target t. No padding or model
parameter is changed to obtain ell. Every loss/metric uses the same crop.
"""

import math
import torch
from torch import nn


def context(model, *, windowed=False):
    """Architectural extents, with global operations scoped to a supplied window."""
    from models.TCN_backbone import CausalConv1d, CausalTCNBackbone, CausalSE, ResidualBlock
    from models.AttentionBlocks.attention import CausalMHA, CausalMLA

    def extent(module):
        if isinstance(module, CausalConv1d):
            return module.pad_left, module.pad_right
        if isinstance(module, CausalSE):
            if not module.pool_window:
                if windowed:
                    return 0, module.lookahead or 0
                raise ValueError("Cumulative/global SE has no fixed complete window; set se_window")
            left = module.pool_window - 1 if module.causal else (module.pool_window - 1) // 2
            return left, module.pool_window - 1 - left
        if isinstance(module, (CausalMHA, CausalMLA)):
            window = getattr(module, "attention_window", None)
            if window is None:
                if windowed:
                    return 0, 0
                raise ValueError("Set attention_window to bound batch and cached attention identically")
            return window - 1, 0 if module.causal else window - 1
        if isinstance(module, ResidualBlock):
            a = extent(module.conv_block)
            b = extent(module.se) if module.se is not None else (0, 0)
            return a[0] + b[0], a[1] + b[1]
        if isinstance(module, CausalTCNBackbone):
            return extent(module.network)
        if isinstance(module, nn.Sequential):
            terms = [extent(child) for child in module]
            return tuple(sum(v[i] for v in terms) for i in (0, 1))
        if hasattr(module, "blocks"):
            terms = [extent(block.attn) for block in module.blocks]
            return tuple(sum(v[i] for v in terms) for i in (0, 1))
        # All remaining pieces in registered topologies are pointwise.
        return 0, 0

    if hasattr(model, "backbone"):
        return extent(model.backbone)
    a, b, c = extent(model.tcn_pt1), extent(model.decoder), extent(model.tcn_pt2)
    if hasattr(model, "squeeze"):
        return tuple(a[i] + b[i] + c[i] for i in (0, 1))
    return tuple(a[i] + max(b[i], c[i]) for i in (0, 1))


def settings(model_or_config):
    return (model_or_config if isinstance(model_or_config, dict)
            else getattr(model_or_config, "run_config", {})) or {}


def target_interval(n, config):
    cfg = settings(config)
    delay = cfg.get("output_delay", 0)
    start = (cfg.get("valid_history_samples") or 1) - 1
    stop = n - max(cfg.get("valid_future_samples", 0), delay)
    if min(start, delay) < 0 or start >= stop:
        raise ValueError(f"Empty valid target interval [{start}, {stop}) for {n} samples")
    return start, stop


def aligned(pred, clean, model_or_config):
    cfg = settings(model_or_config)
    start, stop = target_interval(clean.shape[-1], cfg)
    delay = cfg.get("output_delay", 0)
    if pred.shape[-1] != clean.shape[-1]:
        raise ValueError("Model output and full target lengths differ before delay alignment")
    return pred[..., start + delay:stop + delay], clean[..., start:stop]


def aligned_output(pred, model_or_config, n=None):
    """Target-aligned trace; NaN marks unavailable warm-up/tail, never zero-fill."""
    cfg = settings(model_or_config)
    n = pred.shape[-1] if n is None else n
    start, stop = target_interval(n, cfg)
    out = torch.full_like(pred, float("nan"))
    delay = cfg.get("output_delay", 0)
    out[..., start:stop] = pred[..., start + delay:stop + delay]
    return out


def validate_model(model, cfg, n):
    target_interval(n, cfg)
    if cfg.get("protocol") != "paper":
        return
    past, future = context(model)
    from data.resampling import resampling_info
    warmup = resampling_info(256., cfg.get("target_fs") or 256.)["warmup_samples"]
    if cfg["valid_history_samples"] - 1 < past + warmup:
        raise ValueError(f"Shared history needs at least {past + warmup + 1} samples for this model")
    if cfg["valid_future_samples"] < future + cfg.get("output_delay", 0):
        raise ValueError(f"Shared future exclusion needs at least {future + cfg.get('output_delay', 0)} samples")
    if any(isinstance(m, nn.BatchNorm1d) for m in model.modules()):
        raise ValueError("Time-axis BatchNorm is not permitted in paper training")


def latency(config, fs):
    cfg = settings(config)
    hop, delay = cfg.get("hop", 4), cfg.get("output_delay", 0)
    return dict(sample_rate_hz=fs, hop_samples=hop, output_delay_samples=delay,
                future_delay_ms=1000 * delay / fs,
                algorithmic_budget_ms=1000 * (hop + delay) / fs,
                measured=False)


@torch.no_grad()
def parity(model, signal, hop=4, tolerance=1e-4, *, normalized=False):
    """Compare batch and persistent state on raw or already normalized input."""
    from models.incremental import IncrementalModel
    cfg = settings(model)
    if hop != 4:
        raise ValueError("The methodology parity operating point is H=4")
    scale = 1.0 if normalized else cfg.get("normalization_scale")
    if not normalized and (cfg.get("normalization") != "fixed" or scale is None):
        raise ValueError("Raw-input parity requires saved fixed calibration")
    model.eval()
    x = torch.as_tensor(signal, device=next(model.parameters()).device, dtype=torch.float32).flatten()
    if not torch.isfinite(x).all() or not math.isfinite(float(scale)) or scale <= 0:
        raise ValueError("Parity input and calibration must be finite")
    filters = not normalized and cfg.get("protocol") == "paper"
    stream_filter = None
    if filters:
        from deploy.preprocessing import CausalFilter
        fs = cfg.get("target_fs") or 256.
        reference_input = torch.as_tensor(CausalFilter(fs).process(x.cpu().numpy()), device=x.device)
        stream_filter = CausalFilter(fs)
    else:
        reference_input = x
    reference = model((reference_input / scale).view(1, 1, -1))[0, 0] * scale
    stream = IncrementalModel(model)
    got = []
    stream.reset()
    for start in range(0, x.numel(), hop):
        chunk = x[start:start + hop]
        if stream_filter is not None:
            chunk = torch.as_tensor(stream_filter.process(chunk.cpu().numpy()), device=x.device)
        for sample in chunk / scale:
            got.append(stream.step(sample.reshape(1, 1)).reshape(()) * scale)
    got = torch.stack(got)
    start, stop = target_interval(x.numel(), cfg)
    delay = cfg.get("output_delay", 0)
    a, b = got[start + delay:stop + delay], reference[start + delay:stop + delay]
    delta = float((a - b).abs().max())
    if not math.isfinite(delta):
        raise ValueError("Nonfinite parity difference")
    return dict(max_abs_delta=delta, tolerance=tolerance, passed=delta < tolerance,
                n_samples=x.numel(), domain="model_input" if normalized else "raw_amplitude",
                includes_causal_filters=filters,
                normalization_scale=float(scale), **latency(cfg, cfg.get("target_fs") or 256.))


@torch.no_grad()
def model_stream_checks(model, n_samples=512):
    """Probe the loaded graph and its persistent-state implementation once."""
    from models.variants import variant_is_causal

    cfg = settings(model)
    arch = getattr(model, "model_config", {})
    n_samples = min(n_samples, arch.get("max_seq_len") or n_samples)
    try:
        start, stop = target_interval(n_samples, cfg)
    except ValueError as exc:
        unavailable = {"passed": None, "reason": str(exc)}
        return dict(domain="model_input", n_samples=n_samples,
                    parity=unavailable, causality=unavailable)
    delay = cfg.get("output_delay", 0)
    first = start + delay
    cut = first + max(1, (stop - start) // 2)
    generator = torch.Generator().manual_seed(0)
    param = next(model.parameters())
    signal = torch.randn(1, 1, n_samples, generator=generator).to(param)
    model.eval()

    causal = {"passed": None, "reason": "Probe has no samples after the valid prefix"}
    if cut < n_samples:
        changed = signal.clone()
        changed[..., cut:] = 1 - 2 * changed[..., cut:]
        before = model(signal)[..., first:cut]
        after = model(changed)[..., first:cut]
        delta = float((before - after).abs().max())
        causal = dict(passed=math.isfinite(delta) and delta < 1e-6,
                      max_abs_delta=delta if math.isfinite(delta) else None,
                      tolerance=1e-6, changed_at_sample=cut, checked_targets=cut - first,
                      output_delay_samples=delay)

    variant = arch.get("variant", "")
    step = {"passed": None, "reason": "This model has no past-only incremental path"}
    if variant_is_causal(variant) and getattr(model, "causal", True) and getattr(model, "causal_tcn", True):
        try:
            step = parity(model, signal, normalized=True)
        except (RuntimeError, TypeError, ValueError) as exc:
            step = {"passed": False, "reason": str(exc)}
    return dict(domain="model_input", n_samples=n_samples, parity=step, causality=causal)
