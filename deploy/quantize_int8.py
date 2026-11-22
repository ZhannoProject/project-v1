"""
Static INT8 quantization of an exported graph, and the causality test it has
to pass.

This project rejected INT8 once already, and the reason is worth restating
because it is exactly what this module is built around.  ``quantize_dynamic``
derives each activation's scale **at run time, from the tensor it is about to
quantize**.  A 512-sample window is one tensor, so a sample at *t* takes part
in the scale applied to every sample before it: perturb sample 256 and the
output at sample 0 moves.  ``deploy/export_onnx.py`` measured
``past_diff = 0.32`` on the reference checkpoint and refused it.  That is not
a tuning problem, it is what dynamic quantization *is*.

Static quantization does not have the defect.  Scales are computed once,
offline, on a calibration set, and are written into the graph as
**constants**.  Nothing at run time depends on the window's contents, so a
perturbation at *t* cannot reach an output before *t*: the graph is causal
for the same reason the fp32 graph is.  That is a claim about the
construction, so this module asserts it on the produced graph rather than
arguing it -- the same perturb-and-look-back probe ``verify_onnx`` runs, at
the same tolerance, on every graph written here.

Two properties of this project make the result better than it usually is:
the model is six residual blocks of 1-D convolution with no attention in the
baseline, and its activations are already scale-normalised (every window is
divided by its own standard deviation before the model), so the calibration
range is stable across windows rather than per-recording.

Calibration draws from the **train** split at the checkpoint's own
``master_seed``.  That is not a detail: calibrating on val or test would fit
the quantization scales to the data the model is then scored on, which is
the same leakage the epoch-level split exists to prevent, one level down.

    python -m deploy.quantize_int8 --onnx deploy/artifacts/tcn_both.onnx \
        --data_dir data/EEGdenoisenet/data --artifact BOTH --master_seed 67

Writes ``<name>_int8.onnx`` and a ``<name>_int8.json`` sidecar recording the
calibration set, the quantization settings, the causality probe, the
accuracy of both graphs on the test split, and the size and latency of each.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys                                                        # noqa: E402

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.export_onnx import (                                  # noqa: E402
    INPUT_NAME, OUTPUT_NAME, SAMPLE_RATE_HZ, _session, evaluate_backends,
)

#: Activation/weight dtype pairs onnxruntime supports on CPU.  The right one
#: is a property of the target CPU, not of the model: x86-64 with VNNI wants
#: unsigned activations, ARM64 (the Pi 5 is Cortex-A76) is documented the
#: other way.  Both are measured rather than assumed -- see ``--activation``.
ACTIVATION_TYPES = ("uint8", "int8")

CALIBRATORS = ("minmax", "percentile", "entropy")

#: Quantize the convolutions and leave everything else alone.  This is the
#: setting the attention variants need, and the reason is structural rather
#: than empirical: a causal attention graph carries a scalar ``-inf``
#: initializer -- the mask value fed to ``Where`` before ``Softmax`` -- so a
#: histogram-based calibrator is handed a range of [-inf, inf] and refuses to
#: run at all. Quantizing only ``Conv`` keeps the masked scores, the softmax
#: and the layer norms in fp32, which is also where most of the accuracy loss
#: came from when everything was quantized.
CONV_ONLY = ["Conv"]

#: How far a perturbation may reach backwards before the graph is refused.
#: The same bound ``verify_onnx`` uses, deliberately: an INT8 graph that is
#: "nearly causal" is a graph that reads the future, and the whole premise of
#: this project is that it does not.
CAUSALITY_TOL = 1e-5


# ============================================================
# Calibration
# ============================================================
class TrainSplitCalibration:
    """
    Feed ``quantize_static`` real windows from the model's own train split.

    ``onnxruntime.quantization.CalibrationDataReader`` is an interface with
    one method; subclassing it directly would make this module import the
    quantization stack at import time, which the tests do not need.

    The batches are materialised in ``__init__`` and replayed, because some
    calibrators ask for the data more than once and a generator exhausted on
    the first pass silently calibrates on nothing.
    """

    def __init__(self, data_dir: str, artifact: str, master_seed: int,
                 n_batches: int = 16, batch_size: int = 32,
                 input_name: str = INPUT_NAME, data_options: dict | None = None):
        from data.Dataloader import build_dataloaders

        options = data_options or dict(artifact_type=artifact, master_seed=master_seed)
        train_loader, _, _ = build_dataloaders(data_dir, batch_size=batch_size, **options)
        self.batches: List[Dict[str, np.ndarray]] = []
        for i, batch in enumerate(train_loader):
            if i >= n_batches:
                break
            self.batches.append(
                {input_name: batch["noisy"].numpy().astype(np.float32)})
        if not self.batches:
            raise ValueError("calibration set is empty")
        self.n_windows = sum(b[input_name].shape[0] for b in self.batches)
        self._i = 0

    def get_next(self):
        if self._i >= len(self.batches):
            return None
        self._i += 1
        return self.batches[self._i - 1]

    def rewind(self):
        self._i = 0


def _synthetic_calibration(n_batches: int = 8, batch_size: int = 32,
                           seq_len: int = 512, seed: int = 0,
                           input_name: str = INPUT_NAME):
    """
    A stand-in calibration set of sigma-normalised Gaussian windows.

    For tests and for a smoke run without the dataset.  It calibrates to the
    right *scale* -- the dataset normalises every window to unit standard
    deviation, so a standard normal has the same range -- but not to the real
    distribution's tails, so a graph calibrated this way is not the one to
    deploy.  Recorded as ``synthetic`` in the sidecar for that reason.
    """
    rng = np.random.default_rng(seed)

    class _Reader:
        def __init__(self):
            self.batches = [
                {input_name: rng.standard_normal((batch_size, 1, seq_len),
                                                 dtype=np.float32)}
                for _ in range(n_batches)
            ]
            self.n_windows = n_batches * batch_size
            self._i = 0

        def get_next(self):
            if self._i >= len(self.batches):
                return None
            self._i += 1
            return self.batches[self._i - 1]

        def rewind(self):
            self._i = 0

    return _Reader()


# ============================================================
# Quantization
# ============================================================
def _refuse_histogram_over_a_mask(onnx_path: Path, calibrate_method: str,
                                  op_types: Optional[List[str]]) -> None:
    """
    Stop before onnxruntime reports a range of [-inf, inf] and leaves the
    reader to work out why.

    A causal attention graph holds a scalar ``-inf`` constant: it is the mask
    value fed to ``Where`` ahead of ``Softmax``, i.e. the very thing that
    makes the model causal.  ``percentile`` and ``entropy`` build a histogram
    over each tensor they calibrate, and a histogram over a tensor containing
    -inf has no finite range, so they abort.  Quantizing only the
    convolutions avoids the masked tensors entirely, which is the setting
    these graphs want anyway.
    """
    if calibrate_method == "minmax" or op_types:
        return
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path), load_external_data=False)
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        if arr.dtype.kind == "f" and not np.isfinite(arr).all():
            raise ValueError(
                f"{onnx_path.name} holds a non-finite constant ({init.name}), which "
                f"is the causal attention mask. A '{calibrate_method}' calibrator "
                f"builds a histogram per tensor and cannot take a range of "
                f"[-inf, inf].\n"
                f"  Quantize the convolutions only (op_types=CONV_ONLY, or "
                f"--conv_only on the CLI): on this project's attention graphs that "
                f"is both more accurate and faster than quantizing everything."
            )



def quantize_graph(
    fp32_onnx: str | Path,
    out_path: str | Path,
    calibration,
    activation: str = "uint8",
    per_channel: bool = True,
    calibrate_method: str = "percentile",
    reduce_range: bool = False,
    op_types: Optional[List[str]] = None,
) -> dict:
    """
    Write a statically quantized QDQ copy of ``fp32_onnx``.

    QDQ rather than QOperator because it is the format onnxruntime optimises
    on both of the architectures this project targets, and because the
    Quantize/Dequantize pairs keep the graph readable: the scales are visible
    initializers, which is what makes the causality argument checkable by
    inspection as well as by probe.

    ``op_types`` restricts quantization to those op types, which is how the
    attention variants are made to behave: see ``CONV_ONLY``.  Everything
    else stays fp32, so the result is a mixed-precision graph.
    """
    from onnxruntime.quantization import (CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    if activation not in ACTIVATION_TYPES:
        raise ValueError(f"activation must be one of {ACTIVATION_TYPES}")
    if calibrate_method not in CALIBRATORS:
        raise ValueError(f"calibrate_method must be one of {CALIBRATORS}")

    fp32_onnx, out_path = Path(fp32_onnx), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _refuse_histogram_over_a_mask(fp32_onnx, calibrate_method, op_types)

    act = {"uint8": QuantType.QUInt8, "int8": QuantType.QInt8}[activation]
    method = {"minmax": CalibrationMethod.MinMax,
              "percentile": CalibrationMethod.Percentile,
              "entropy": CalibrationMethod.Entropy}[calibrate_method]

    with tempfile.TemporaryDirectory() as tmp:
        # Shape inference and folding first: this is what turns the graph's
        # BatchNormalization nodes into the Conv they belong to, so the
        # quantizer sees one affine op per block instead of two.
        prepped = Path(tmp) / "prepped.onnx"
        quant_pre_process(str(fp32_onnx), str(prepped), skip_symbolic_shape=False)

        calibration.rewind()
        quantize_static(
            model_input=str(prepped),
            model_output=str(out_path),
            calibration_data_reader=calibration,
            quant_format=QuantFormat.QDQ,
            activation_type=act,
            weight_type=QuantType.QInt8,     # weights are always signed
            per_channel=per_channel,
            reduce_range=reduce_range,
            calibrate_method=method,
            op_types_to_quantize=list(op_types) if op_types else None,
        )

    return {
        "quant_format": "QDQ",
        "activation_type": activation,
        "weight_type": "int8",
        "per_channel": per_channel,
        "reduce_range": reduce_range,
        "calibrate_method": calibrate_method,
        "op_types_to_quantize": list(op_types) if op_types else "all supported",
        "calibration_windows": getattr(calibration, "n_windows", None),
        "file_size_kb": round(out_path.stat().st_size / 1024, 1),
    }


# ============================================================
# Verification
# ============================================================
def probe_causality(run: Callable[[np.ndarray], np.ndarray], seq_len: int = 512,
                    amplitude: float = 3.0, seed: int = 0) -> dict:
    """
    Perturb one sample and measure how far the output moves backwards.

    ``past`` must be zero: an output before *t* that responds to *t* is a
    graph reading the future.  ``future`` must *not* be zero, or the probe
    proved nothing -- a perturbation large enough to saturate the input
    quantizer would clip to the same code as the original and the test would
    pass vacuously.  The amplitude is therefore kept inside the calibrated
    range (the dataset normalises each window to unit standard deviation, so
    3.0 is a few sigma and well short of the MinMax range).
    """
    rng = np.random.default_rng(seed)
    t0 = seq_len // 2
    x1 = rng.standard_normal((1, 1, seq_len), dtype=np.float32)
    x2 = x1.copy()
    x2[0, 0, t0] += amplitude
    y1, y2 = run(x1), run(x2)
    return {
        "past_diff": float(np.abs(y1[..., :t0] - y2[..., :t0]).max()),
        "future_diff": float(np.abs(y1[..., t0:] - y2[..., t0:]).max()),
        "perturbation": amplitude,
        "at_sample": t0,
    }


def _latency_ms(run: Callable[[np.ndarray], np.ndarray], seq_len: int = 512,
                n: int = 50, seed: int = 1) -> dict:
    x = np.random.default_rng(seed).standard_normal((1, 1, seq_len), dtype=np.float32)
    for _ in range(10):
        run(x)
    times = []
    for _ in range(n):
        t = time.perf_counter()
        run(x)
        times.append(time.perf_counter() - t)
    times = np.asarray(times) * 1e3
    return {"median_ms": round(float(np.median(times)), 3),
            "p95_ms": round(float(np.percentile(times, 95)), 3)}


def _source_data_options(source, master_seed=None, artifact=None):
    from types import SimpleNamespace
    from data.protocol import evaluation_data
    cfg = dict(source.get("data_protocol") or {})
    prov, prep = source.get("provenance") or {}, source.get("preprocessing") or {}
    cfg.setdefault("master_seed", prov.get("trained_master_seed"))
    cfg.setdefault("artifact", prov.get("trained_artifact") or prep.get("artifact_trained_on"))
    cfg.setdefault("target_fs", prep.get("sample_rate_hz"))
    cfg.setdefault("normalization", prep.get("normalization", "epoch"))
    cfg.setdefault("normalization_scale", prep.get("normalization_scale"))
    return evaluation_data(SimpleNamespace(run_config=cfg, checkpoint_loaded=True),
                           master_seed, artifact)


def verify_int8(fp32_onnx: str | Path, int8_onnx: str | Path, seq_len: int = 512,
                data_dir: Optional[str] = None, artifact: str = None,
                master_seed: int = None, max_batches: Optional[int] = None,
                n_threads: int = 2, data_options: dict | None = None) -> dict:
    """
    Everything that has to be true of the quantized graph before it is used.

    Raises AssertionError if the INT8 graph is not causal, or if the
    perturbation probe did not actually perturb anything.  Accuracy is
    *reported*, not asserted: how much RRMSE an 8-bit graph may give up is a
    judgement for whoever deploys it, and the point of this report is to put
    the number in front of them.
    """
    fp32_sess, int8_sess = _session(fp32_onnx, n_threads), _session(int8_onnx, n_threads)
    run32 = lambda x: fp32_sess.run([OUTPUT_NAME], {INPUT_NAME: x})[0]   # noqa: E731
    run8 = lambda x: int8_sess.run([OUTPUT_NAME], {INPUT_NAME: x})[0]    # noqa: E731

    report: dict = {"host": f"{platform.machine()} / {platform.system()}",
                    "threads": n_threads}

    # ---- causality: the reason this module exists ----
    causal = probe_causality(run8, seq_len)
    report["causality"] = causal
    assert causal["past_diff"] < CAUSALITY_TOL, (
        f"INT8 graph is not causal: a perturbation at sample {causal['at_sample']} "
        f"moved an earlier output by {causal['past_diff']:.3e} "
        f"(tolerance {CAUSALITY_TOL:g}). Static calibration should make this exactly "
        f"zero; a non-zero value means a scale is still being derived at run time."
    )
    assert causal["future_diff"] > 0.0, (
        "the causality probe changed nothing downstream either, so it proved "
        "nothing -- the perturbation was probably clipped by the input quantizer"
    )
    report["causality"]["fp32_reference"] = probe_causality(run32, seq_len)

    # ---- agreement: reported against the signal, not as an absolute ----
    rng = np.random.default_rng(7)
    x = rng.standard_normal((8, 1, seq_len), dtype=np.float32)
    y32, y8 = run32(x), run8(x)
    rms = float(np.sqrt((y32 ** 2).mean()))
    report["agreement"] = {
        "max_abs_diff": float(np.abs(y32 - y8).max()),
        "rms_diff": float(np.sqrt(((y32 - y8) ** 2).mean())),
        "output_rms": rms,
        "rms_diff_rel": float(np.sqrt(((y32 - y8) ** 2).mean()) / rms) if rms else None,
    }

    # ---- size and speed, both graphs, same machine and thread count ----
    report["size_kb"] = {
        "fp32": round(Path(fp32_onnx).stat().st_size / 1024, 1),
        "int8": round(Path(int8_onnx).stat().st_size / 1024, 1),
    }
    report["size_kb"]["ratio"] = round(
        report["size_kb"]["fp32"] / report["size_kb"]["int8"], 2)
    report["latency"] = {"fp32": _latency_ms(run32, seq_len),
                         "int8": _latency_ms(run8, seq_len)}
    report["latency"]["speedup_median"] = round(
        report["latency"]["fp32"]["median_ms"]
        / max(report["latency"]["int8"]["median_ms"], 1e-9), 2)

    # ---- what it costs in accuracy, on the real test split ----
    if data_dir:
        source_json = Path(fp32_onnx).with_suffix(".json")
        source = json.loads(source_json.read_text()) if source_json.exists() else {}
        if data_options is None:
            data_options = _source_data_options(source, master_seed, artifact)
        report["master_seed"] = data_options["master_seed"]
        report["test_metrics"] = evaluate_backends(
            {"fp32": run32, "int8": run8},
            data_dir, artifact, master_seed, max_batches,
            data_options=data_options,
            score_config=source.get("data_protocol") or {},
        )
        m = report["test_metrics"]
        report["test_metrics"]["delta_int8_minus_fp32"] = {
            k: round(m["int8"][k] - m["fp32"][k], 4) for k in m["fp32"]
        }
    return report


# ============================================================
# One call
# ============================================================
def quantize_checkpoint_graph(
    fp32_onnx: str | Path,
    out_dir: Optional[str | Path] = None,
    data_dir: Optional[str] = None,
    artifact: str = None,
    master_seed: int = None,
    seq_len: int = None,
    activation: str = "uint8",
    per_channel: bool = True,
    calibrate_method: str = "percentile",
    reduce_range: bool = False,
    op_types: Optional[List[str]] = None,
    n_calib_batches: int = 16,
    max_batches: Optional[int] = None,
    n_threads: int = 2,
) -> dict:
    """
    fp32 graph in, verified INT8 graph plus sidecar out.

    ``data_dir`` is what makes the calibration real; without it the graph is
    calibrated on synthetic windows and the sidecar says so, because a
    calibration set is part of what a quantized model *is* and a reader
    should not have to guess which one was used.
    """
    fp32_onnx = Path(fp32_onnx)
    out_dir = Path(out_dir) if out_dir else fp32_onnx.parent
    stem = fp32_onnx.stem
    suffix = "_int8" if not op_types else "_int8conv"
    int8_path = out_dir / f"{stem}{suffix}.onnx"
    source_json = fp32_onnx.with_suffix(".json")
    source = json.loads(source_json.read_text()) if source_json.exists() else {}
    prep = source.get("preprocessing") or {}
    seq_len = seq_len or prep.get("window_samples") or 512
    options = None
    if data_dir:
        options = _source_data_options(source, master_seed, artifact)
        master_seed, artifact = options["master_seed"], options["artifact_type"]

    if data_dir:
        calib = TrainSplitCalibration(data_dir, artifact, master_seed,
                                      n_batches=n_calib_batches, data_options=options)
        calib_desc = {"source": "train split", "artifact": artifact,
                      "master_seed": master_seed, "windows": calib.n_windows}
    else:
        calib = _synthetic_calibration(seq_len=seq_len)
        calib_desc = {"source": "synthetic", "windows": calib.n_windows,
                      "warning": "not calibrated on real data; do not deploy"}

    quant = quantize_graph(fp32_onnx, int8_path, calib, activation=activation,
                           per_channel=per_channel,
                           calibrate_method=calibrate_method,
                           reduce_range=reduce_range, op_types=op_types)
    verification = verify_int8(fp32_onnx, int8_path, seq_len=seq_len,
                               data_dir=data_dir, artifact=artifact,
                               master_seed=master_seed, max_batches=max_batches,
                               n_threads=n_threads, data_options=options)

    sidecar = {
        "source_graph": str(fp32_onnx.relative_to(PROJECT_ROOT)
                            if fp32_onnx.is_relative_to(PROJECT_ROOT) else fp32_onnx),
        "quantization": {**quant, "calibration": calib_desc,
                         "scheme": "static (scales are graph constants)"},
        "io": {"input": {"name": INPUT_NAME, "shape": ["batch", 1, seq_len],
                         "dtype": "float32"},
               "output": {"name": OUTPUT_NAME, "shape": ["batch", 1, seq_len],
                          "dtype": "float32"}},
        "preprocessing": prep or {"sample_rate_hz": SAMPLE_RATE_HZ,
                          "window_samples": seq_len,
                          "normalise": "x / max(std(x), 1e-8) per window, "
                                       "rescaled afterwards (as in fp32)"},
        "provenance": source.get("provenance", {}),
        "data_protocol": source.get("data_protocol", {}),
        "verification": verification,
    }
    json_path = out_dir / f"{stem}{suffix}.json"
    json_path.write_text(json.dumps(sidecar, indent=2))
    return {"onnx": int8_path, "json": json_path, **sidecar}


def _print_report(r: dict) -> None:
    v, q = r["verification"], r["quantization"]
    c, f = v["causality"], v["causality"]["fp32_reference"]
    print(f"\n  wrote {r['onnx']}")
    print(f"  scheme     : {q['scheme']}, {q['quant_format']}, "
          f"act={q['activation_type']} w={q['weight_type']}, "
          f"per_channel={q['per_channel']}, calib={q['calibrate_method']}")
    print(f"  ops        : {q['op_types_to_quantize']}")
    print(f"  calibration: {q['calibration']['source']}, "
          f"{q['calibration']['windows']} windows")
    print(f"  causality  : past {c['past_diff']:.3e}  future {c['future_diff']:.4f}"
          f"   (fp32: past {f['past_diff']:.3e}  future {f['future_diff']:.4f})")
    a = v["agreement"]
    print(f"  agreement  : rms diff {a['rms_diff']:.4f} on output rms "
          f"{a['output_rms']:.4f}  ({100 * a['rms_diff_rel']:.1f}% relative)")
    s, lat = v["size_kb"], v["latency"]
    print(f"  size       : {s['fp32']} kB -> {s['int8']} kB  ({s['ratio']}x smaller)")
    print(f"  latency    : {lat['fp32']['median_ms']} ms -> {lat['int8']['median_ms']} ms "
          f"median  ({lat['speedup_median']}x)   [{v['host']}, {v['threads']} threads]")
    if "test_metrics" in v:
        m = v["test_metrics"]
        print(f"  {'':<13}{'RRMSE_t':>9}{'RRMSE_f':>9}{'CC':>9}")
        for name in ("fp32", "int8"):
            print(f"  {name:<13}{m[name]['rrmse_t']:>9.4f}{m[name]['rrmse_s']:>9.4f}"
                  f"{m[name]['acc']:>9.4f}")
        d = m["delta_int8_minus_fp32"]
        print(f"  {'delta':<13}{d['rrmse_t']:>+9.4f}{d['rrmse_s']:>+9.4f}{d['acc']:>+9.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Statically quantize an exported graph to INT8 and verify it")
    ap.add_argument("--onnx", required=True, help="fp32 graph from deploy/export_onnx.py")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--data_dir", default=None,
                    help="EEGdenoiseNet .npy directory; calibration and metrics "
                         "need it. Without it the graph is calibrated on synthetic "
                         "windows and marked as such.")
    ap.add_argument("--artifact", default=None, choices=["EOG", "EMG", "BOTH"])
    ap.add_argument("--master_seed", type=int, default=None,
                    help="the seed the checkpoint was trained with; it picks the "
                         "split the calibration and the metrics come from")
    ap.add_argument("--seq_len", type=int, default=None)
    ap.add_argument("--activation", default="uint8", choices=list(ACTIVATION_TYPES))
    ap.add_argument("--calibrate_method", default="percentile", choices=list(CALIBRATORS),
                    help="percentile is the measured best here; minmax is the one "
                         "that tolerates a graph containing the causal mask's -inf")
    ap.add_argument("--per_channel", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--conv_only", action="store_true",
                    help="quantize Conv nodes only and leave attention in fp32; "
                         "what the MHA/MLA graphs need (see CONV_ONLY)")
    ap.add_argument("--reduce_range", action="store_true",
                    help="7-bit weights; for pre-VNNI x86 where INT8 accumulation "
                         "can saturate")
    ap.add_argument("--n_calib_batches", type=int, default=16)
    ap.add_argument("--max_batches", type=int, default=None,
                    help="cap the test-split evaluation, for a quick run")
    ap.add_argument("--threads", type=int, default=2,
                    help="onnxruntime intra-op threads for the latency comparison")
    a = ap.parse_args(argv)

    r = quantize_checkpoint_graph(
        a.onnx, a.out_dir, a.data_dir, a.artifact, a.master_seed, a.seq_len,
        activation=a.activation, per_channel=a.per_channel,
        calibrate_method=a.calibrate_method, reduce_range=a.reduce_range,
        op_types=CONV_ONLY if a.conv_only else None,
        n_calib_batches=a.n_calib_batches, max_batches=a.max_batches,
        n_threads=a.threads,
    )
    _print_report(r)
    return r


if __name__ == "__main__":
    main()
