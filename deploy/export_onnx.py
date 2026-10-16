"""
Export a trained checkpoint (typically the 2-layer TCN + MLA sandwich) to a
single self-contained ONNX file for Raspberry Pi 5 deployment, and verify it.

    python -m deploy.export_onnx --ckpt runs/tcn_mla_eog/best.pt
    python -m deploy.export_onnx --ckpt runs/tcn_mla_eog/best.pt \
        --data_dir data/EEGdenoisenet/data --artifact EOG

What the export produces (in --out_dir, default deploy/artifacts/):
    <name>.onnx   fp32 graph, opset 18, weights embedded, dynamic batch axis,
                  constants folded (standard ai.onnx ops only, ~1.2 MB)
    <name>.json   sidecar: architecture, I/O contract, preprocessing, and
                  the verification results

Why fp32 and not INT8: dynamic INT8 quantization was evaluated and rejected.
It picks activation scales per tensor at run time, so a change at sample t
alters the scale used for every earlier sample -- the graph stops being
causal (past_diff 0.32 on the reference checkpoint).  It was also slower
(the DynamicQuantizeLinear overhead dominates these tiny matmuls) and less
accurate.  The fp32 graph runs a 512-sample window in a few ms on a Pi 5.

Architecture resolution
-----------------------
Older checkpoints (e.g. runs/tcn_mla_eog/best.pt) carry no ``model_config``
and were trained with a kv_lora_rank that differs from the YAML preset.  The
weights are the ground truth, so every shape-determined hyperparameter
(hidden_channels, bottleneck_ratio, skip_bottleneck_ratio, n_attn_layers,
kv_lora_rank, max_seq_len, variant) is inferred from the state dict.  ``n_heads`` is the one setting that
leaves no trace in the weights: it comes from --n_heads, then the checkpoint's
model_config, then --config, then the project default of 4.  A rank alias
(tcn_mla_r4) has the same weights layout as tcn_mla; its checkpoint is accepted
when the rank its name promises matches the weights, and the alias is kept as
the ``variant`` in the sidecar.

Deployment contract (also written to the sidecar JSON)
------------------------------------------------------
    input   noisy_eeg     float32 (batch, 1, seq_len)
    output  denoised_eeg  float32 (batch, 1, seq_len)
    Each window must be divided by its own standard deviation before the
    model and multiplied back afterwards (this mirrors data/Dataloader.py).
    The graph is strictly causal, so a live stream can be denoised by
    feeding the most recent seq_len samples and keeping the newest outputs
    (see deploy/raspi_infer.py).

TurboQuant is a fake-quantization accuracy experiment with no runtime
benefit, so it is always stripped before export; the ONNX graph is the
full-precision network.  The graph has no KV cache: every call recomputes
attention over the whole window (T=512, d_model=32 -> negligible on a Pi 5).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.variants import build_model, variant_is_causal, base_variant, pinned_kwargs  # noqa: E402
from models.AttentionBlocks.attention import CausalMHA, CausalMLA  # noqa: E402

INPUT_NAME = "noisy_eeg"
OUTPUT_NAME = "denoised_eeg"
DEFAULT_OPSET = 18
DEFAULT_N_HEADS = 4
SAMPLE_RATE_HZ = 256          # EEGdenoiseNet: 512-sample epochs = 2 s


# ============================================================
# 1. Rebuild the model from a checkpoint
# ============================================================
def infer_arch_from_state_dict(sd: dict) -> dict:
    """
    Recover every shape-determined constructor argument from the weights.

    Returns a dict accepted by build_model(**cfg) minus ``n_heads``, which
    is not recoverable (all head counts give identical tensor shapes).
    """
    cfg = {}

    # First conv of the first TCN block: (hidden_channels, in_channels=1, k)
    first_conv = next(
        (k for k in ("tcn_pt1.network.0.conv_block.0.conv.weight",
                     "backbone.network.0.conv_block.0.conv.weight") if k in sd),
        None,
    )
    if first_conv is None:
        raise ValueError("Unrecognised state dict: no TCN first-layer weight found")
    hidden = sd[first_conv].shape[0]
    cfg["hidden_channels"] = int(hidden)

    block_ids = {int(k.split(".")[2]) for k in sd if k.startswith("decoder.blocks.")}
    if not block_ids:
        cfg["variant"] = "tcn"
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


def load_checkpoint_model(ckpt_path: str | Path, config_path: str | Path | None = None,
                          n_heads: int | None = None) -> tuple[torch.nn.Module, dict, dict]:
    """
    Build the model whose shapes match the checkpoint and load its weights.

    Returns (model in eval mode, resolved architecture dict, raw checkpoint).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state"]
    stored = dict(ckpt.get("model_config", {}))

    # Bidirectional teachers (tcn_*_bi) have the same weights as their causal
    # twins, so only model_config can tell them apart.  They need future
    # samples and must never be deployed for streaming.
    if not variant_is_causal(stored.get("variant", "")):
        raise ValueError(
            f"{ckpt_path} is a bidirectional teacher ({stored['variant']}); it attends to "
            f"future samples and cannot run online. Export the distilled causal student instead."
        )

    arch = infer_arch_from_state_dict(sd)

    yaml_cfg = {}
    if config_path:
        from configs import load_config
        yaml_cfg = load_config(config_path)

    # n_heads priority: CLI > checkpoint model_config > YAML > default
    arch["n_heads"] = int(
        n_heads if n_heads is not None
        else stored.get("n_heads", yaml_cfg.get("n_heads", DEFAULT_N_HEADS))
    )
    if arch["variant"] == "tcn":
        arch.pop("n_heads")

    # Sanity: anything the checkpoint claims about itself must agree with the weights.
    # The weights can only reveal the base architecture, so a rank alias such as
    # tcn_mla_r4 is compared as tcn_mla — and what its name promises (rank 4) is
    # then checked against the weights explicitly.
    label = stored.get("variant")
    if label is not None and base_variant(label) != arch["variant"]:
        raise ValueError(
            f"Checkpoint model_config says variant={label} but the weights imply "
            f"{arch['variant']}; refusing to guess."
        )
    for key, value in (pinned_kwargs(label) if label is not None else {}).items():
        if key in arch and arch[key] != value:
            raise ValueError(
                f"Checkpoint is labelled '{label}', which fixes {key}={value}, but the weights "
                f"imply {key}={arch[key]}; refusing to export a model under a name that misdescribes it."
            )
    for key in ("hidden_channels", "n_attn_layers", "kv_lora_rank", "max_seq_len"):
        if key in stored and key in arch and stored[key] != arch[key]:
            raise ValueError(
                f"Checkpoint model_config says {key}={stored[key]} but the weights imply "
                f"{key}={arch[key]}; refusing to guess."
            )
    if label is not None:
        arch["variant"] = label          # keep the run's own name in the sidecar
    if stored.get("turboquant"):
        print("[export] Checkpoint was trained with TurboQuant enabled; exporting the "
              "full-precision network (TurboQuant is a fake-quant experiment only).")

    model = build_model(**arch, turboquant=False)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, arch, ckpt


# ============================================================
# 2. Export
# ============================================================
def _prepare_for_export(model: torch.nn.Module) -> None:
    """Strip runtime-only hooks that must not end up in the graph."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (CausalMHA, CausalMLA)):
            m.turboquant = None
            m.capture_weights(False)


def export_to_onnx(model: torch.nn.Module, onnx_path: str | Path, seq_len: int = 512,
                   opset: int = DEFAULT_OPSET, dynamic_batch: bool = True) -> dict:
    """
    Write ``model`` to ``onnx_path`` as one self-contained file.

    Uses the torch.export-based ("dynamo") exporter, which folds the causal
    left-padding into Conv pads and the mask slice into constants.  Falls
    back to the TorchScript exporter (opset <= 17) on torch versions that
    lack it.  Returns a dict describing which path was taken.
    """
    import onnx

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    _prepare_for_export(model)

    max_len = getattr(model.decoder.blocks[0].attn, "causal_mask", None) if hasattr(model, "decoder") else None
    if max_len is not None and seq_len > max_len.shape[-1]:
        raise ValueError(f"seq_len={seq_len} exceeds the model's causal mask ({max_len.shape[-1]})")

    dummy = torch.zeros(1, 1, seq_len, dtype=torch.float32)
    common = dict(input_names=[INPUT_NAME], output_names=[OUTPUT_NAME])
    info = {"opset": opset, "dynamic_batch": dynamic_batch, "seq_len": seq_len}

    try:
        dynamic_shapes = {"x": {0: torch.export.Dim("batch")}} if dynamic_batch else None
        torch.onnx.export(model, (dummy,), str(onnx_path), dynamo=True, external_data=False,
                          opset_version=opset, dynamic_shapes=dynamic_shapes, verbose=False,
                          **common)
        info["exporter"] = "dynamo"
    except Exception as err:  # older torch, or an op the new exporter cannot lower
        print(f"[export] dynamo exporter unavailable ({type(err).__name__}: {str(err)[:120]}); "
              f"falling back to the TorchScript exporter")
        legacy_opset = min(opset, 17)
        dynamic_axes = {INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}} if dynamic_batch else None
        torch.onnx.export(model, (dummy,), str(onnx_path), dynamo=False,
                          opset_version=legacy_opset, dynamic_axes=dynamic_axes,
                          do_constant_folding=True, **common)
        info.update(exporter="torchscript", opset=legacy_opset)

    info["folded"] = _fold_constants(onnx_path)
    onnx.checker.check_model(str(onnx_path), full_check=True)
    proto = onnx.load(str(onnx_path))
    info.update(
        ir_version=proto.ir_version,
        n_nodes=len(proto.graph.node),
        op_types=sorted({n.op_type for n in proto.graph.node}),
        file_size_kb=round(onnx_path.stat().st_size / 1024, 1),
    )
    return info


STANDARD_DOMAINS = {"", "ai.onnx"}


def _fold_constants(onnx_path: Path) -> bool:
    """
    Rewrite the file with onnxruntime's *basic* (hardware-independent)
    optimisations applied offline.  The main effect here is folding the
    ``Equal(causal_mask, 0)`` nodes into bool constants, which drops the
    two dead 1 MB float masks and shrinks the file ~2.5x.  Only the basic
    level is used because higher levels emit hardware-specific fused ops.
    Returns False (file untouched) if any non-standard op appeared.
    """
    import onnx
    import onnxruntime as ort

    folded = onnx_path.with_suffix(".folded.tmp.onnx")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.optimized_model_filepath = str(folded)
    ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    proto = onnx.load(str(folded))
    folded.unlink()
    if any(n.domain not in STANDARD_DOMAINS for n in proto.graph.node):
        return False
    # ORT registers every domain it knows as an import; keep only the ones used
    used = {n.domain for n in proto.graph.node} | {""}
    keep = [o for o in proto.opset_import if o.domain in used]
    del proto.opset_import[:]
    proto.opset_import.extend(keep)
    onnx.save(proto, str(onnx_path))
    return True


# ============================================================
# 3. Verify
# ============================================================
def _session(onnx_path: str | Path, n_threads: int = 4):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = n_threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])


def _torch_forward(model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(x)).numpy()


def verify_onnx(model: torch.nn.Module, onnx_path: str | Path, seq_len: int = 512,
                dynamic_batch: bool = True, atol: float = 1e-4, data_dir: str | None = None,
                master_seed: int = 42,
                artifact: str = "EOG", max_batches: int | None = None, seed: int = 0) -> dict:
    """
    Check the ONNX graph against the PyTorch model.

      agreement  max |onnx - torch| on random windows (batch 1, and 3 if dynamic)
      causality  perturbing sample T/2 must not change any output before T/2
      latency    onnxruntime batch-1 wall time on THIS machine (not the Pi)
      metrics    RRMSE_t / RRMSE_f / CC of both backends on the real test
                 split, when data_dir is given

    Raises AssertionError on a failed agreement or causality check.
    """
    sess = _session(onnx_path)
    rng = np.random.default_rng(seed)
    run = lambda x: sess.run([OUTPUT_NAME], {INPUT_NAME: x})[0]
    report: dict = {}

    # ---- agreement ----
    batches = [1, 3] if dynamic_batch else [1]
    worst = 0.0
    for b in batches:
        x = rng.standard_normal((b, 1, seq_len), dtype=np.float32)
        worst = max(worst, float(np.abs(run(x) - _torch_forward(model, x)).max()))
    report["max_abs_diff_vs_torch"] = worst
    assert worst < atol, f"ONNX/PyTorch disagree: max |diff| = {worst:.3e} (atol {atol})"

    # ---- causality ----
    t0 = seq_len // 2
    x1 = rng.standard_normal((1, 1, seq_len), dtype=np.float32)
    x2 = x1.copy()
    x2[0, 0, t0] += 10.0
    y1, y2 = run(x1), run(x2)
    past = float(np.abs(y1[..., :t0] - y2[..., :t0]).max())
    future = float(np.abs(y1[..., t0:] - y2[..., t0:]).max())
    report["causality_past_diff"] = past
    report["causality_future_diff"] = future
    assert past < 1e-5, f"Causality violated in ONNX graph: past_diff = {past:.3e}"

    # ---- latency (host) ----
    x = rng.standard_normal((1, 1, seq_len), dtype=np.float32)
    for _ in range(10):
        run(x)
    times = []
    for _ in range(50):
        t = time.perf_counter()
        run(x)
        times.append(time.perf_counter() - t)
    report["host_latency_ms_median"] = round(1e3 * float(np.median(times)), 3)
    report["host"] = f"{platform.machine()} / {platform.system()}"

    # ---- benchmark metrics on the real test split ----
    if data_dir:
        report["master_seed"] = master_seed
        report["test_metrics"] = evaluate_backends(
            {"torch": lambda x: _torch_forward(model, x), Path(onnx_path).name: run},
            data_dir, artifact, master_seed, max_batches,
        )
    return report


def evaluate_backends(backends: dict, data_dir: str, artifact: str = "EOG",
                      master_seed: int = 42,
                      max_batches: int | None = None, batch_size: int = 32) -> dict:
    """
    Run each ``name -> f(np.ndarray) -> np.ndarray`` backend over the test
    split and return {name: {rrmse_t, rrmse_s, acc}} averaged over batches.
    Inputs are already sigma-normalised by the dataset, exactly as in training.
    """
    from data.Dataloader import build_dataloaders
    from metrics import compute_all_metrics

    # The split is a function of the seed, so it has to be the seed the
    # checkpoint was trained with -- otherwise these "test" metrics are
    # measured on epochs the model was trained on.
    _, _, test_loader = build_dataloaders(data_dir, artifact_type=artifact,
                                          batch_size=batch_size, master_seed=master_seed)
    sums = {name: {"rrmse_t": 0.0, "rrmse_s": 0.0, "acc": 0.0} for name in backends}
    n = 0
    for i, batch in enumerate(test_loader):
        if max_batches is not None and i >= max_batches:
            break
        noisy = batch["noisy"].numpy().astype(np.float32)
        clean = batch["clean"]
        for name, fn in backends.items():
            m = compute_all_metrics(torch.from_numpy(fn(noisy)), clean)
            for k in sums[name]:
                sums[name][k] += m[k]
        n += 1
    return {name: {k: round(v / n, 4) for k, v in s.items()} for name, s in sums.items()}


# ============================================================
# 4. One-call pipeline + CLI
# ============================================================
def export_checkpoint(ckpt: str | Path, out_dir: str | Path = PROJECT_ROOT / "deploy" / "artifacts",
                      name: str | None = None, config: str | None = None, n_heads: int | None = None,
                      seq_len: int | None = None, opset: int = DEFAULT_OPSET, dynamic_batch: bool = True,
                      data_dir: str | None = None, artifact: str = "EOG",
                      master_seed: int = 42,
                      max_batches: int | None = None) -> dict:
    """
    Checkpoint -> verified ONNX -> sidecar JSON.
    Returns the sidecar dict.  This is the function to call from other code.
    """
    import onnx
    import onnxruntime

    ckpt = Path(ckpt)
    out_dir = Path(out_dir)
    name = name or ckpt.parent.name or ckpt.stem
    model, arch, raw = load_checkpoint_model(ckpt, config, n_heads)
    seq_len = seq_len or arch.get("max_seq_len", 512)

    print(f"[export] checkpoint : {ckpt}  (epoch {raw.get('epoch', '?')}, val_loss {raw.get('val_loss', float('nan')):.4f})")
    print(f"[export] architecture: {arch}")
    print(f"[export] params      : {sum(p.numel() for p in model.parameters()):,}")

    fp32_path = out_dir / f"{name}.onnx"
    export_info = export_to_onnx(model, fp32_path, seq_len, opset, dynamic_batch)
    print(f"[export] wrote {fp32_path}  ({export_info['file_size_kb']} KB, {export_info['exporter']} exporter, "
          f"opset {export_info['opset']}, {export_info['n_nodes']} nodes, folded={export_info['folded']})")

    print("[verify] onnxruntime vs PyTorch ...")
    fp32_report = verify_onnx(model, fp32_path, seq_len, dynamic_batch,
                              data_dir=data_dir, artifact=artifact,
                              master_seed=master_seed, max_batches=max_batches)
    _print_report(fp32_report)

    sidecar = {
        "model": {"checkpoint": str(ckpt), "epoch": raw.get("epoch"), "val_loss": raw.get("val_loss"),
                  **arch, "n_params": sum(p.numel() for p in model.parameters())},
        "export": {**export_info,
                   "torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__},
        "io": {"input": {"name": INPUT_NAME, "shape": ["batch" if dynamic_batch else 1, 1, seq_len], "dtype": "float32"},
               "output": {"name": OUTPUT_NAME, "shape": ["batch" if dynamic_batch else 1, 1, seq_len], "dtype": "float32"}},
        "preprocessing": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "window_samples": seq_len,
            "artifact_trained_on": artifact,
            "normalise": "x_in = x / max(std(x), 1e-8) per window; x_out = model(x_in) * std(x)",
            "causal": True,
            "streaming": "feed the latest window_samples samples, keep the newest `hop` outputs",
        },
        "verification": fp32_report,
    }

    json_path = out_dir / f"{name}.json"
    json_path.write_text(json.dumps(sidecar, indent=2))
    print(f"[export] wrote {json_path}")
    return sidecar


def _print_report(r: dict) -> None:
    print(f"         max|onnx-torch| = {r['max_abs_diff_vs_torch']:.2e}   "
          f"causality past/future = {r['causality_past_diff']:.1e} / {r['causality_future_diff']:.2e}   "
          f"host latency = {r['host_latency_ms_median']} ms")
    if "test_metrics" in r:
        for name, m in r["test_metrics"].items():
            print(f"         {name:<28s} RRMSE_t={m['rrmse_t']:.4f}  RRMSE_f={m['rrmse_s']:.4f}  CC={m['acc']:.4f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="checkpoint, e.g. runs/tcn_mla_eog/best.pt")
    p.add_argument("--out_dir", default=str(PROJECT_ROOT / "deploy" / "artifacts"))
    p.add_argument("--name", default=None, help="output stem (default: checkpoint's run directory name)")
    p.add_argument("--config", default=None, help="YAML used to train (only consulted for n_heads)")
    p.add_argument("--n_heads", type=int, default=None, help="override; not recoverable from weights")
    p.add_argument("--seq_len", type=int, default=None, help="window length to bake in (default: max_seq_len)")
    p.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    p.add_argument("--static_batch", action="store_true", help="fix batch=1 instead of a dynamic batch axis")
    p.add_argument("--data_dir", default=None, help="EEGdenoiseNet .npy dir; enables test-set metrics")
    p.add_argument("--artifact", default="EOG", choices=["EOG", "EMG", "BOTH"])
    p.add_argument("--max_batches", type=int, default=None, help="limit test batches for a quick check")
    p.add_argument("--master_seed", type=int, default=42,
                   help="the split the test metrics are measured on; it must be the seed "
                        "the checkpoint was trained with, or they are measured on epochs "
                        "the model was trained on")
    a = p.parse_args(argv)
    # Keywords past the positional block: a parameter inserted into the middle
    # of the signature would otherwise silently shift what follows it.
    export_checkpoint(a.ckpt, a.out_dir, a.name, a.config, a.n_heads, a.seq_len, a.opset,
                      not a.static_batch, data_dir=a.data_dir, artifact=a.artifact,
                      master_seed=a.master_seed, max_batches=a.max_batches)


if __name__ == "__main__":
    main()
