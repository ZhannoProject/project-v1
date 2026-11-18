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

from models.AttentionBlocks.attention import CausalMHA, CausalMLA  # noqa: E402

INPUT_NAME = "noisy_eeg"
OUTPUT_NAME = "denoised_eeg"
DEFAULT_OPSET = 18
DEFAULT_N_HEADS = 4
SAMPLE_RATE_HZ = 256          # EEGdenoiseNet: 512-sample epochs = 2 s


# ============================================================
# 1. Rebuild the model from a checkpoint
# ============================================================
# Public compatibility imports; reconstruction lives outside deployment.
from experiment.checkpoints import infer_arch_from_state_dict, load_model


def load_checkpoint_model(ckpt_path, config_path=None, n_heads=None):
    return load_model(ckpt_path, config=config_path, overrides={"n_heads": n_heads},
                      for_export=True)


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


def export_step_to_onnx(model: torch.nn.Module, onnx_path: str | Path,
                        opset: int = DEFAULT_OPSET) -> dict:
    """
    Export one incremental step: ``(x_t, state...) -> (y_t, new_state...)``.

    The windowed export recomputes a whole 512-sample window per call and
    carries no state, which is why its sidecar has to say "the graph has no
    KV cache".  This one declares the cache as graph inputs and outputs, so
    onnxruntime allocates it on the board and the compression MLA was chosen
    for is finally a property of something running rather than an estimate.

    Exported through the TorchScript path rather than dynamo: the step is
    twenty-odd state tensors in and out, and the older exporter handles that
    signature without the dynamic-shape bookkeeping the new one wants for a
    graph that has no dynamic shapes at all.

    Verified against ``models/incremental.py`` by the caller, which is itself
    pinned to the batch model -- so a step graph is never trusted because it
    exported cleanly.
    """
    import onnx
    from models.step_graph import StepGraph

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    _prepare_for_export(model)

    graph = StepGraph(model).eval()
    spec = graph.state_spec
    state = graph.initial_state(1)
    names = [s["name"] for s in spec]

    with torch.no_grad():
        torch.onnx.export(
            graph, (torch.zeros(1, 1, 1), *state), str(onnx_path),
            input_names=["x_t", *names],
            output_names=["y_t", *(f"{n}_out" for n in names)],
            opset_version=opset, dynamo=False, verbose=False)

    loaded = onnx.load(str(onnx_path))
    return {
        "opset": opset,
        "exporter": "torchscript",
        "mode": "incremental",
        "n_nodes": len(loaded.graph.node),
        "file_size_kb": round(onnx_path.stat().st_size / 1024, 1),
        "window": graph.window,
        "topology": graph.topology,
        "state": spec,
        "state_tensors": len(spec),
        "state_floats": int(sum(int(torch.tensor(s["shape"]).prod()) for s in spec)),
        "mask_fill": __import__("models.step_graph", fromlist=["MASK_FILL"]).MASK_FILL,
    }


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
                master_seed: int = None,
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
        from data.protocol import evaluation_data
        data_options = evaluation_data(model, master_seed, artifact)
        report["master_seed"] = data_options["master_seed"]
        report["test_metrics"] = evaluate_backends(
            {"torch": lambda x: _torch_forward(model, x), Path(onnx_path).name: run},
            data_dir, max_batches=max_batches, data_options=data_options,
            score_config=getattr(model, "run_config", {}),
        )
    return report


def evaluate_backends(backends: dict, data_dir: str, artifact: str = "EOG",
                      master_seed: int = 42,
                      max_batches: int | None = None, batch_size: int = 32,
                      data_options: dict | None = None, score_config: dict | None = None) -> dict:
    """
    Run each ``name -> f(np.ndarray) -> np.ndarray`` backend over the test
    split and return {name: {rrmse_t, rrmse_s, acc}} averaged over batches.
    Inputs are already sigma-normalised by the dataset, exactly as in training.
    """
    from data.Dataloader import build_dataloaders
    from metrics import compute_all_metrics
    from experiment.methodology import aligned

    # The split is a function of the seed, so it has to be the seed the
    # checkpoint was trained with -- otherwise these "test" metrics are
    # measured on epochs the model was trained on.
    options = data_options or dict(artifact_type=artifact, master_seed=master_seed)
    _, _, test_loader = build_dataloaders(data_dir, batch_size=batch_size, **options)
    sums = {name: {"rrmse_t": 0.0, "rrmse_s": 0.0, "acc": 0.0} for name in backends}
    n = 0
    for i, batch in enumerate(test_loader):
        if max_batches is not None and i >= max_batches:
            break
        noisy = batch["noisy"].numpy().astype(np.float32)
        clean = batch["clean"]
        size = clean.shape[0]
        for name, fn in backends.items():
            pred, target = aligned(torch.from_numpy(fn(noisy)), clean, score_config or {})
            m = compute_all_metrics(pred, target)
            for k in sums[name]:
                sums[name][k] += m[k] * size
        n += size
    return {name: {k: round(v / n, 4) for k, v in s.items()} for name, s in sums.items()}


# ============================================================
# 4. One-call pipeline + CLI
# ============================================================
def _preprocessing(raw, window, artifact, streaming):
    from data.protocol import normalization_config
    cfg = raw.get("run_config") or {}
    norm = normalization_config(cfg)
    fixed = norm["normalization"] == "fixed"
    from experiment.methodology import latency
    return dict(sample_rate_hz=cfg.get("target_fs") or SAMPLE_RATE_HZ,
                window_samples=window, artifact_trained_on=artifact, **norm,
                normalise=("x_in = x / normalization_scale; x_out = model(x_in) * normalization_scale"
                           if fixed else "offline only: x_in = x / max(std(x), 1e-8) per epoch; restore scale"),
                causal=fixed, model_causal=True, streaming=streaming,
                output_delay_samples=cfg.get("output_delay", 0),
                output_semantics="output at acquisition t estimates target t-output_delay_samples",
                valid_history_samples=cfg.get("valid_history_samples"),
                valid_future_samples=cfg.get("valid_future_samples", 0),
                latency_budget=latency(cfg, cfg.get("target_fs") or SAMPLE_RATE_HZ))


def _export_step_checkpoint(model, arch, raw, ckpt, out_dir, name, artifact, opset,
                            trained_artifact):
    """
    The incremental export, and the check that it is the same function.

    A step graph is verified against ``models/incremental.py`` rather than
    against the batch model directly: that wrapper is already pinned to the
    batch pass from both ends, so it is the stricter oracle, and it is the
    thing this graph is supposed to be a port of.  Both warmup and the steady
    state are covered -- the run is longer than the attention window, so the
    validity mask is exercised filling up and full.
    """
    import numpy as np
    import onnxruntime
    from models.incremental import IncrementalModel
    from models.step_graph import StepGraph

    path = out_dir / f"{name}_step.onnx"
    info = export_step_to_onnx(model, path, opset)
    print(f"[export] wrote {path}  ({info['file_size_kb']} KB, {info['n_nodes']} nodes, "
          f"{info['state_tensors']} state tensors, {info['state_floats']:,} floats of state)")

    print("[verify] onnxruntime step loop vs models/incremental.py ...")
    torch.manual_seed(0)
    cfg = raw.get("run_config") or {}
    probe_n = max(int(info["window"]) + 128 if info["window"] else 400,
                  (cfg.get("valid_history_samples") or 1) + cfg.get("valid_future_samples", 0) + 128)
    probe = torch.randn(probe_n)
    reference = IncrementalModel(model).run(probe).numpy()

    sess = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]
    state = {n: t.numpy() for n, t in zip(in_names[1:], StepGraph(model).initial_state(1))}
    got = np.empty(len(probe), dtype=np.float32)
    for t in range(len(probe)):
        out = sess.run(None, {"x_t": probe[t].view(1, 1, 1).numpy(), **state})
        got[t] = out[0].reshape(())
        state = dict(zip(in_names[1:], out[1:]))
    max_diff = float(np.abs(reference - got).max())
    if cfg.get("protocol") == "paper":
        from experiment.methodology import target_interval
        start, stop = target_interval(len(probe), cfg)
        delay = cfg.get("output_delay", 0)
        max_diff = float(np.abs(reference[start+delay:stop+delay] - got[start+delay:stop+delay]).max()
                         * cfg["normalization_scale"])
    print(f"         max|onnx-incremental| = {max_diff:.2e} over {len(probe)} steps")
    if max_diff >= 1e-4:
        raise ValueError(
            f"the exported step graph does not reproduce models/incremental.py "
            f"(max|diff| = {max_diff:.2e}); refusing to ship a graph that computes "
            f"something else")

    sidecar = {
        "model": {"checkpoint": str(ckpt), "epoch": raw.get("epoch"),
                  "val_loss": raw.get("val_loss"), **arch,
                  "n_params": sum(p.numel() for p in model.parameters())},
        "export": {**info, "torch": torch.__version__},
        "data_protocol": dict(raw.get("run_config") or {}),
        "io": {"input": {"name": "x_t", "shape": [1, 1, 1], "dtype": "float32"},
               "output": {"name": "y_t", "shape": [1, 1, 1], "dtype": "float32"},
               "state": info["state"]},
        "provenance": {
            "trained_master_seed": (raw.get("run_config") or {}).get("master_seed"),
            "trained_artifact": trained_artifact or artifact,
        },
        "preprocessing": _preprocessing(raw, info["window"], artifact,
                                        "feed one sample and the state; keep the output and new state"),
        "verification": {"max_abs_diff_vs_incremental": max_diff, "n_steps": len(probe),
                         "domain": "raw_amplitude" if cfg.get("protocol") == "paper" else "model_input"},
    }
    json_path = out_dir / f"{name}_step.json"
    json_path.write_text(json.dumps(sidecar, indent=2))
    print(f"[export] wrote {json_path}")
    return sidecar


def _export_replay_tail(model, full_path, seq_len, hop, opset):
    """Export a shorter graph and verify the retained outputs against the full graph."""
    from deploy.eegio_replay import FAST_WINDOW_BATCH
    from experiment.methodology import context

    try:
        past, future = context(model)
    except ValueError:
        return None
    tail_len = past + hop
    if future or tail_len >= seq_len:
        return None

    tail_path = full_path.with_name(f"{full_path.stem}_tail.onnx")
    try:
        export_to_onnx(model, tail_path, seq_len=tail_len, opset=opset)
        full_session, tail_session = _session(full_path), _session(tail_path)
        probe = np.random.default_rng(0).standard_normal((FAST_WINDOW_BATCH, 1, seq_len), dtype=np.float32)
        reference = np.concatenate([
            full_session.run([OUTPUT_NAME], {INPUT_NAME: window[None]})[0][..., -hop:]
            for window in probe
        ])
        suffixes = np.ascontiguousarray(probe[..., -tail_len:])
        actual = tail_session.run([OUTPUT_NAME], {INPUT_NAME: suffixes})[0][..., -hop:]
        error = float(np.max(np.abs(actual - reference)))
        if not np.isfinite(error) or error >= 1e-4:
            raise ValueError(f"retained outputs differ by {error:.3g}")
    except Exception as exc:
        print(f"[export] Fast replay will use full windows: {exc}")
        return None

    print(f"[export] Fast replay uses {tail_len} inference samples; max normalized difference {error:.2e}")
    return {"file": tail_path.name, "input_samples": tail_len,
            "batch_windows": FAST_WINDOW_BATCH, "max_abs_diff": error}


def export_checkpoint(ckpt: str | Path, out_dir: str | Path = PROJECT_ROOT / "deploy" / "artifacts",
                      name: str | None = None, config: str | None = None, n_heads: int | None = None,
                      seq_len: int | None = None, opset: int = DEFAULT_OPSET, dynamic_batch: bool = True,
                      data_dir: str | None = None, artifact: str = "EOG",
                      master_seed: int = None,
                      max_batches: int | None = None,
                      incremental: bool = False, *, fast_replay_hop: int | None = None) -> dict:
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
    cfg = getattr(model, "run_config", {})
    seq_len = seq_len or (round(512 * cfg["target_fs"] / SAMPLE_RATE_HZ)
                          if cfg.get("target_fs") else arch.get("max_seq_len", 512))

    # What the model was actually trained on beats what the command line said.
    # --artifact only ever chose the verification batch, but its value was
    # written into the sidecar as "artifact_trained_on", so exporting a BOTH
    # model without the flag stamped it "EOG" and nothing noticed. The
    # checkpoint knows; ask it, and say so when the two disagree.
    trained_artifact = (raw.get("run_config") or {}).get("artifact")
    if trained_artifact and trained_artifact != artifact:
        print(f"[export] checkpoint was trained on {trained_artifact}, not the --artifact "
              f"{artifact} given; using {trained_artifact} for the sidecar and the "
              f"verification batch")
        artifact = trained_artifact

    print(f"[export] checkpoint : {ckpt}  (epoch {raw.get('epoch', '?')}, val_loss {raw.get('val_loss', float('nan')):.4f})")
    print(f"[export] architecture: {arch}")
    print(f"[export] params      : {sum(p.numel() for p in model.parameters()):,}")

    if incremental:
        return _export_step_checkpoint(model, arch, raw, ckpt, out_dir, name,
                                       artifact, opset, trained_artifact)

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
        "data_protocol": dict(raw.get("run_config") or {}),
        "io": {"input": {"name": INPUT_NAME, "shape": ["batch" if dynamic_batch else 1, 1, seq_len], "dtype": "float32"},
               "output": {"name": OUTPUT_NAME, "shape": ["batch" if dynamic_batch else 1, 1, seq_len], "dtype": "float32"}},
        # What split this model was fitted on. The scorer compares it against the
        # provenance stamped into a test set, because "test" is only test
        # relative to a seed: EEGDenoiseDataset permutes by master_seed, so a set
        # cut at 42 is training data for a run trained at 67. Taken from the
        # checkpoint's own run_config, not from the CLI --master_seed (which
        # only picks the verification batch) and not from the run's name.
        "provenance": {
            "trained_master_seed": (raw.get("run_config") or {}).get("master_seed"),
            "trained_artifact": (raw.get("run_config") or {}).get("artifact"),
            "note": "null means the run predates run_config; a test set cannot be "
                    "checked against it and is scored with a warning",
        },
        "preprocessing": _preprocessing(raw, seq_len, artifact,
                                        "feed the latest window_samples samples, keep the newest hop outputs"),
        "verification": fp32_report,
    }

    if fast_replay_hop is not None:
        tail = _export_replay_tail(model, fp32_path, seq_len, fast_replay_hop, opset)
        if tail is not None:
            sidecar["fast_replay"] = tail

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
    p.add_argument("--incremental", action="store_true",
                   help="export the one-sample step graph instead: state in, state out, "
                        "a real KV cache. Runs on the Pi with onnxruntime alone, and is the "
                        "only exported form whose cache size is a property of the graph.")
    p.add_argument("--max_batches", type=int, default=None, help="limit test batches for a quick check")
    p.add_argument("--master_seed", type=int, default=None,
                   help="the split the test metrics are measured on; it must be the seed "
                        "the checkpoint was trained with, or they are measured on epochs "
                        "the model was trained on")
    a = p.parse_args(argv)
    # Keywords past the positional block: a parameter inserted into the middle
    # of the signature would otherwise silently shift what follows it.
    export_checkpoint(a.ckpt, a.out_dir, a.name, a.config, a.n_heads, a.seq_len, a.opset,
                      not a.static_batch, data_dir=a.data_dir, artifact=a.artifact,
                      master_seed=a.master_seed, max_batches=a.max_batches,
                      incremental=a.incremental)


if __name__ == "__main__":
    main()
