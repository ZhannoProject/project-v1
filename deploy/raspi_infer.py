"""
Run an exported EEG denoiser on a Raspberry Pi 5 (or any CPU) with only
numpy + onnxruntime.  Copy this file, the .onnx and its .json sidecar to the
Pi; nothing else from the repo is needed.

    python raspi_infer.py tcn_mla_eog.onnx --benchmark
    python raspi_infer.py tcn_mla_eog.onnx --testdata EOG_test100.npz
    python raspi_infer.py tcn_mla_eog.onnx --input noisy.npy --output clean.npy
    python raspi_infer.py tcn_mla_eog.onnx --input noisy.npy --output clean.npy --stream --hop 32
    python raspi_infer.py tcn_mla_eog.onnx --testdata EOG_test100.npz --latency --latency_out lat
    python raspi_infer.py tcn_mla_eog.onnx --input rec.npy --stream --realtime --latency --latency_every 50
    python raspi_infer.py tcn_mla_eog.onnx --testdata EOG_test100.npz --duty_cycle 64,32,16,8,4

Two ways to use the model
-------------------------
denoise_epochs(x)   x is (T,) or (N, T) with T == the exported window length.
                    Each row is normalised by its own std, denoised, and
                    rescaled -- the same contract as the training data loader.

denoise_stream(x)   x is an arbitrarily long 1-D signal.  Because the network
                    is strictly causal, the output for sample t only needs
                    samples <= t, so the stream is processed by sliding a
                    window of the last T samples forward by `hop` samples and
                    keeping the newest `hop` outputs of each call.  The first
                    window (samples 0..T-1) is emitted whole; every position
                    then has exactly the same amount of left-context it had
                    during training.  Latency is one forward pass per hop:
                    at 256 Hz a hop of 32 samples is a 125 ms cadence.

StreamingDenoiser   the same thing as an object you push chunks into, for a
                    live acquisition loop.

LatencyMonitor      per-hop timing for either streaming path: onnxruntime time,
                    total hop time, deadline misses against the hop budget
                    (hop / fs), throughput, SoC temperature, and -- when the
                    signal is replayed at the sample rate with --realtime /
                    stream_realtime() -- the true end-to-end latency from a
                    sample being available to its denoised value being emitted.
                    --latency turns it on; --latency_out writes .json + .csv.

--duty_cycle        real-time applicability check: paced replay for a list of
                    hop sizes, reporting compute / period (duty cycle), e2e
                    latency and its drift, deadline misses and an OK / MARGINAL /
                    FAIL verdict per hop, plus the smallest hop that is safe.

--cpus / preflight  every timing run first pins the process (--cpus 0-1) and
                    reads the state of the machine -- CPU governor, affinity,
                    thread count, throttle flag, load -- through
                    deploy/preflight.py.  The verdict is printed, written into
                    every trace this run saves, and with --require-locked a
                    machine that is not held still refuses to be measured.

--testdata          scores a .npz cut by deploy/make_testdata.py (noisy, clean,
                    snr_db) with the EEGdenoiseNet metrics, in both epoch and
                    streaming mode, next to the untouched noisy input as a
                    baseline.  Running the same command on the host and on the
                    Pi should give the same numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Copied to the Pi next to this file; it is stdlib-only and never optional for
# a timing run, because a latency number without the state of the machine it
# was taken on is not a measurement anyone can repeat.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import preflight                                                 # noqa: E402

EPS = 1e-8
#: The rate the epochs were cut at. Only a fallback: a sidecar that records
#: its own sample_rate_hz wins, because the model is what decides it.
SAMPLE_RATE_HZ = 256


def normalization_scale(meta, signal=None, sigma=None, online=False):
    """No signal-dependent scale is ever inferred by an online entry point."""
    cfg = meta.get("preprocessing") or {}
    mode = cfg.get("normalization", "epoch")
    if mode not in ("epoch", "fixed"):
        raise ValueError(f"Unknown normalization: {mode!r}")
    if mode == "fixed":
        recorded = cfg.get("normalization_scale")
        if recorded is None:
            raise ValueError("Fixed normalization metadata has no normalization_scale")
        if sigma is not None and not np.isclose(float(sigma), float(recorded), rtol=1e-7, atol=0):
            raise ValueError("sigma differs from the checkpoint normalization_scale")
        sigma = recorded
    if sigma is not None:
        sigma = float(sigma)
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        return sigma
    if online:
        raise ValueError("Legacy epoch normalization is offline only. Online inference requires "
                         "a fixed sigma calibrated before this signal, or a fixed-normalization checkpoint.")
    return np.maximum(np.asarray(signal).std(axis=-1, keepdims=True), EPS)


class EEGDenoiser:
    """
    One exported model, with its threading pinned rather than defaulted.

    onnxruntime's thread model is not the OpenMP one, and three details of it
    decide what a latency number here means:

    * ``intra_op_num_threads = N`` is **N in total, including the calling
      thread** -- ORT starts N-1 workers and runs the rest on the caller.
      (Measured: N-1 extra OS threads, none at N=1.  ORT asserts the same
      thing, refusing ``intra_op_thread_affinities`` unless it is given
      exactly N-1 entries.)  So "one core per thread" means N cores for N,
      which is what ``--cpus`` pins.
    * the default is **0**, which means one thread per physical core -- an
      unset thread count is not "single-threaded", it is "as many as the board
      has", and on a Pi 5 that is the 4-thread row of the table below, the
      worst one.
    * the pool **spins** before sleeping.  That buys latency and costs CPU
      while the loop is idle between hops: measured over a paced replay of one
      window per 125 ms, spinning takes 31% of a core against 18% without, for
      13.0 ms per hop against 14.0.  It does not change whether the loop keeps
      up; it changes how much of the board is left for everything else.

    Modern wheels carry no OpenMP at all (``get_build_info()`` says so), so
    ``OMP_NUM_THREADS`` governs none of this -- it is still set for numpy, and
    the preflight check reports it as that rather than as the control.

    Thread counts measured on a Pi 5 (tcn_mla_eog, 512-sample window): 1 thread
    24 ms, 2 threads 18 ms with a tight tail (p95 19 ms), 4 threads 20 ms
    median but p95 30 ms / max 40 ms from contending with the OS for every core.
    """

    lookahead = 0                    # causal unless the sidecar declares L (set in __init__)

    def __init__(self, onnx_path: str | Path, n_threads: int = 2, spin: bool = True):
        onnx_path = Path(onnx_path)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = n_threads
        # Pinned rather than left at 0 so the trace records a number the run
        # really used.  Sequential execution never touches the inter-op pool;
        # it is set explicitly for the same reason.
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.add_session_config_entry("session.intra_op.allow_spinning", "1" if spin else "0")
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self.threads = {
            "intra_op": n_threads,
            "workers": max(0, n_threads - 1),      # ORT counts the caller as one of them
            "inter_op": 1,
            "execution_mode": "sequential",
            "allow_spinning": bool(spin),
            "openmp_in_build": "openmp" in ort.get_build_info().lower(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "onnxruntime": ort.__version__,
        }

        inp, out = self.sess.get_inputs()[0], self.sess.get_outputs()[0]
        self.input_name, self.output_name = inp.name, out.name
        self.window = int(inp.shape[-1])
        self.dynamic_batch = not isinstance(inp.shape[0], int)

        sidecar = onnx_path.with_suffix(".json")
        self.meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        self.sample_rate = self.meta.get("preprocessing", {}).get("sample_rate_hz",
                                                                   SAMPLE_RATE_HZ)
        # Architectural lookahead L: output index t reads inputs up to t+L, so
        # a stream may only emit it once sample t+L has arrived. 0 = causal.
        self.lookahead = int(self.meta.get("preprocessing", {}).get("lookahead_samples") or 0)
        if self.lookahead >= self.window:
            raise ValueError(f"lookahead {self.lookahead} does not fit a {self.window}-sample window")

        self.last_run_s = 0.0        # wall time of the most recent sess.run (read by LatencyMonitor)

    # ---- core call: (N, 1, T) normalised -> (N, 1, T) normalised ----
    def _run(self, x: np.ndarray) -> np.ndarray:
        if not self.dynamic_batch and x.shape[0] != 1:
            return np.concatenate([self._run(row[None]) for row in x])
        t0 = time.perf_counter()
        y = self.sess.run([self.output_name], {self.input_name: x})[0]
        self.last_run_s = time.perf_counter() - t0
        return y

    def denoise_epochs(self, x: np.ndarray, sigma: float | None = None) -> np.ndarray:
        """(T,) or (N, T) raw-scale windows -> same shape, denoised, raw scale."""
        x = np.asarray(x, dtype=np.float32)
        squeeze = x.ndim == 1
        x = x.reshape(-1, self.window)
        sigma = normalization_scale(self.meta, x, sigma)
        y = self._run((x / sigma)[:, None, :])[:, 0, :] * sigma
        return y[0] if squeeze else y

    def denoise_stream(self, signal: np.ndarray, hop: int = 32,
                       monitor: "LatencyMonitor | None" = None,
                       sigma: float | None = None) -> np.ndarray:
        """
        Causally denoise a long 1-D signal; output is sample-aligned with the input.
        ``monitor`` (a LatencyMonitor) records the cost of every hop.

        With lookahead L, each hop keeps the outputs that end L samples before
        the window end -- the newest ones whose L future inputs are real -- so
        out[t] is exactly what a live loop emits once sample t+L arrives. The
        final L samples have no such window; they are filled from the last one
        and lie outside the scored interval.
        """
        signal = np.asarray(signal, dtype=np.float32)
        sigma = normalization_scale(self.meta, sigma=sigma, online=True)
        T, L = self.window, self.lookahead
        if monitor is not None:
            monitor.lookahead = L
        if L and len(signal) >= T:
            out = np.empty_like(signal)
            t0 = time.perf_counter()
            out[:T] = self.denoise_epochs(signal[:T], sigma=sigma)
            if monitor is not None:
                monitor.record_first_window(time.perf_counter() - t0)
            for end in range(T + hop, len(signal) + hop, hop):
                end = min(end, len(signal))
                t0 = time.perf_counter()
                y = self.denoise_epochs(signal[end - T:end], sigma=sigma)
                out[end - hop - L:end - L] = y[T - hop - L:T - L]
                out[end - L:end] = y[T - L:]          # provisional; replaced by the next hop
                if monitor is not None:
                    monitor.record_hop(self.last_run_s, time.perf_counter() - t0)
            return out
        if len(signal) < T:
            # Right-pad with zeros: causality guarantees the padding cannot
            # influence the real samples, but normalise on real samples only.
            xn = np.zeros((1, 1, T), dtype=np.float32)
            xn[0, 0, : len(signal)] = signal / sigma
            return self._run(xn)[0, 0, : len(signal)] * sigma
        out = np.empty_like(signal)
        t0 = time.perf_counter()
        out[:T] = self.denoise_epochs(signal[:T], sigma=sigma)
        if monitor is not None:
            monitor.record_first_window(time.perf_counter() - t0)
        for end in range(T + hop, len(signal) + hop, hop):
            end = min(end, len(signal))
            t0 = time.perf_counter()
            out[end - hop:end] = self.denoise_epochs(signal[end - T:end], sigma=sigma)[-hop:]
            if monitor is not None:
                monitor.record_hop(self.last_run_s, time.perf_counter() - t0)
        return out


class IncrementalDenoiser:
    """
    The step graph, driven one sample at a time with numpy and onnxruntime.

    ``EEGDenoiser`` above runs the windowed export: a whole 512-sample window
    per call, no state, so each hop recomputes everything and the KV cache
    MLA was chosen for does not exist on the board at all.  This runs
    ``<model>_step.onnx``, which declares its state as graph inputs and hands
    it back as outputs.  The cache is then a tensor onnxruntime allocates
    here, and its size is a property of the model rather than an estimate
    made about it on a laptop.

    No torch: the state buffers are read straight off the graph's own input
    shapes, so this needs nothing the Pi does not already have.

    What it buys is latency, not throughput.  A windowed run at hop 32 emits
    nothing until 32 samples have arrived -- 125 ms at 256 Hz -- and divides
    one call across all of them. This emits on every sample, and pays a whole
    graph call to do it. At hop 32 the windowed path is therefore the cheaper
    one; the step graph wins as the hop shrinks, and at a hop of one sample it
    is the only one that keeps up.
    """

    lookahead = 0                    # a step graph is causal-only (the exporter refuses L>0)

    def __init__(self, onnx_path: str | Path, n_threads: int = 2, spin: bool = True):
        onnx_path = Path(onnx_path)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(n_threads)
        opts.add_session_config_entry("session.intra_op.allow_spinning", "1" if spin else "0")
        self.sess = ort.InferenceSession(str(onnx_path), opts,
                                         providers=["CPUExecutionProvider"])
        self.threads = {"intra_op": int(n_threads), "workers": max(0, int(n_threads) - 1),
                        "allow_spinning": bool(spin)}

        inputs = self.sess.get_inputs()
        self.x_name = inputs[0].name
        self.state_names = [i.name for i in inputs[1:]]
        self.state_shapes = [tuple(i.shape) for i in inputs[1:]]
        self.out_names = [o.name for o in self.sess.get_outputs()]

        sidecar = onnx_path.with_suffix(".json")
        self.meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        self.sample_rate = ((self.meta.get("preprocessing") or {}).get("sample_rate_hz")
                            or SAMPLE_RATE_HZ)
        self.window = int((self.meta.get("preprocessing") or {}).get("window_samples") or 0)
        self.last_run_s = 0.0

        # The mask starts fully closed: no position is valid until the stream
        # has fed one. Matching models/incremental.py exactly, which slices
        # its ring to what it has written.
        self.mask_fill = float(((self.meta.get("export") or {}).get("mask_fill")) or -1e9)

    # -- state ------------------------------------------------
    def initial_state(self) -> dict:
        """Zeroed buffers and a closed mask, named as the graph expects them."""
        state = {}
        for name, shape in zip(self.state_names, self.state_shapes):
            fill = self.mask_fill if name.startswith("attn_mask") else 0.0
            state[name] = np.full(shape, fill, dtype=np.float32)
        return state

    def state_bytes(self) -> int:
        return int(sum(np.prod(s) for s in self.state_shapes)) * 4

    def state_summary(self) -> dict:
        """
        What the graph carries between steps, split by what it is.

        The KV half is the part MLA shrinks; the convolution half is the same
        in every variant and does not, so an embedded budget needs both
        numbers rather than the ratio of the first one alone.
        """
        kv = conv = 0
        window = max(1, self.window)
        for name, shape in zip(self.state_names, self.state_shapes):
            n = int(np.prod(shape))
            if name.startswith("kv_"):
                kv += n
            elif not name.startswith("attn_mask"):
                conv += n
        return {"kv_floats_per_step": kv // window if kv else 0,
                "kv_cache_bytes": kv * 4,
                "conv_state_bytes": conv * 4,
                "total_state_bytes": self.state_bytes(),
                "state_tensors": len(self.state_names)}

    # -- stepping ---------------------------------------------
    def step(self, x_t: float, state: dict) -> tuple[float, dict]:
        """One sample in, one sample out, and the state that follows it."""
        feed = {self.x_name: np.array([[[x_t]]], dtype=np.float32), **state}
        t0 = time.perf_counter()
        out = self.sess.run(None, feed)
        self.last_run_s = time.perf_counter() - t0
        return float(out[0].reshape(())), dict(zip(self.state_names, out[1:]))

    def denoise_epochs(self, x: np.ndarray, monitor: "LatencyMonitor | None" = None) -> np.ndarray:
        """
        ``(N, T)`` raw-scale epochs -> the same, denoised.

        The state is reset per epoch because the epochs are independent 2 s
        segments: carrying state across them would feed one recording's
        history into another's first samples. That is also what makes this
        directly comparable with the windowed path's epoch mode.
        """
        x = np.asarray(x, dtype=np.float32)
        single = x.ndim == 1
        x = x.reshape(1, -1) if single else x
        out = np.empty_like(x)
        for e in range(x.shape[0]):
            sigma = float(np.asarray(normalization_scale(self.meta, x[e])).reshape(()))
            state = self.initial_state()
            for t in range(x.shape[1]):
                t0 = time.perf_counter()
                y, state = self.step(float(x[e, t]) / sigma, state)
                if monitor is not None:
                    monitor.record_hop(self.last_run_s, time.perf_counter() - t0)
                out[e, t] = y * sigma
        return out.reshape(-1) if single else out

    def denoise_stream(self, signal: np.ndarray, sigma: float | None = None,
                       monitor: "LatencyMonitor | None" = None) -> np.ndarray:
        """
        A continuous 1-D signal, one sample at a time.

        Use the checkpoint's fixed scale, or an explicitly pre-calibrated
        ``sigma`` for legacy models. Never estimate it from this signal.
        """
        signal = np.asarray(signal, dtype=np.float32).reshape(-1)
        sigma = normalization_scale(self.meta, sigma=sigma, online=True)
        state = self.initial_state()
        out = np.empty_like(signal)
        for t in range(signal.shape[0]):
            t0 = time.perf_counter()
            y, state = self.step(float(signal[t]) / sigma, state)
            if monitor is not None:
                monitor.record_hop(self.last_run_s, time.perf_counter() - t0)
            out[t] = y * sigma
        return out

    def stream_realtime(self, signal: np.ndarray, sigma: float | None = None,
                        monitor: "LatencyMonitor | None" = None) -> np.ndarray:
        """
        Paced replay for the step graph: each sample is fed when it would have
        been acquired (t0 + (t+1)/fs) and stepped at once, so the monitor's
        end-to-end figure is acquisition -> estimate for every single sample.
        There is no window to fill first; the state starts empty.
        """
        signal = np.asarray(signal, dtype=np.float32).reshape(-1)
        sigma = normalization_scale(self.meta, sigma=sigma, online=True)
        state = self.initial_state()
        out = np.empty_like(signal)
        start = time.perf_counter()
        for t in range(signal.shape[0]):
            due = start + (t + 1) / self.sample_rate
            now = time.perf_counter()
            if due > now:
                time.sleep(due - now)
            t0 = time.perf_counter()
            y, state = self.step(float(signal[t]) / sigma, state)
            if monitor is not None:
                monitor.record_hop(self.last_run_s, time.perf_counter() - t0)
            out[t] = y * sigma
        if monitor is not None:
            monitor.attach_schedule(start, 0)
        return out


class LatencyMonitor:
    """
    Per-hop latency recorder for the streaming path -- what a live BCI loop
    on the Pi needs to know about itself.

    For every hop it stores
        infer_ms   the onnxruntime call alone (EEGDenoiser.last_run_s)
        total_ms   the whole hop: buffer roll + normalise + run + rescale + slice
        t_s        when the hop finished (perf_counter), for cadence / e2e latency
        temp_c     SoC temperature from /sys/class/thermal (Raspberry Pi OS; NaN elsewhere)
    and derives median / p95 / p99 / max, jitter, deadline misses against the
    hop budget (hop / fs), real-time headroom and throughput.  With a schedule
    attached (see stream_realtime) it also reports end-to-end latency: the
    time from the last sample of a hop being *available* to its denoised
    samples being *emitted*, which is the number that matters for a live loop.

    Pass one to StreamingDenoiser(..., monitor=m) or denoise_stream(..., monitor=m);
    ``report_every=N`` prints a rolling-window line every N hops; ``summary()``
    returns a dict; ``save("stem")`` writes stem.json + stem.csv.
    """

    THERMAL = Path("/sys/class/thermal/thermal_zone0/temp")

    def __init__(self, hop: int, sample_rate: float, report_every: int = 0,
                 window: int = 64, label: str = "latency",
                 environment: dict | None = None, model_window: int | None = None,
                 mode: str = "windowed", kv_cache: bool | None = None,
                 state: dict | None = None):
        self.mode = mode
        # Whether this model carries a KV cache at all. The baseline has
        # convolution state and no attention, so "incremental" must not be
        # taken to imply one -- the whole point of recording it is that the
        # cache is the thing being claimed.
        self.kv_cache = kv_cache
        # How big that cache actually is, read off the graph's own state
        # tensors. This is the MLA claim's number, and the board is where it
        # is worth having: measured here, not computed about the model
        # somewhere else.
        self.state = state
        self.hop, self.fs = int(hop), float(sample_rate)
        # The model's window, as distinct from ``window`` above, which is the
        # rolling size of the printed average. Recorded so the trace can say
        # how the numbers were produced.
        self.model_window = None if model_window is None else int(model_window)
        self.budget_ms = 1e3 * self.hop / self.fs
        self.report_every, self.window, self.label = int(report_every), int(window), label
        # The state of the machine these numbers were taken on, so the trace
        # file carries the conditions and not only the result.
        self.environment = environment
        # What the model scored on the test set during this same run. Latency
        # and accuracy are two halves of one measurement -- "15 ms a hop" only
        # means something next to what the model got for it -- and they were
        # being printed and dropped while the timings were saved.
        self.scores: dict | None = None
        self.infer_ms: list[float] = []
        self.total_ms: list[float] = []
        self.t_s: list[float] = []
        self.temp_c: list[float] = []
        self.first_window_ms: float | None = None
        self.available_s: np.ndarray | None = None     # set by attach_schedule()
        # Lookahead L of the model being timed (set by the streaming paths). The
        # emitted samples of a hop estimate targets L samples older than its
        # newest input, so their target-to-output latency includes L / fs.
        self.lookahead = 0
        self.t_start: float | None = None              # start of the first recorded forward pass
        self._thermal_ok = self.THERMAL.exists()

    # ---- recording ----
    def _temp(self) -> float:
        if not self._thermal_ok:
            return float("nan")
        try:
            return int(self.THERMAL.read_text()) / 1000.0
        except (OSError, ValueError):
            self._thermal_ok = False
            return float("nan")

    def record_first_window(self, seconds: float) -> None:
        """The one whole-window forward pass that ends warm-up (not a hop)."""
        self.first_window_ms = 1e3 * seconds
        if self.t_start is None:
            self.t_start = time.perf_counter() - seconds

    def record_hop(self, infer_s: float, total_s: float) -> None:
        now = time.perf_counter()
        if self.t_start is None:
            self.t_start = now - total_s
        self.infer_ms.append(1e3 * infer_s)
        self.total_ms.append(1e3 * total_s)
        self.t_s.append(now)
        self.temp_c.append(self._temp())
        n = len(self.total_ms)
        if self.report_every and n % self.report_every == 0:
            print(self.live_line())

    def attach_schedule(self, t0_s: float, window: int) -> None:
        """
        Declare when each hop's data became available: sample i arrived at
        t0_s + i / fs, so hop k (k = 1, 2, ...) was complete at
        t0_s + (window + k * hop) / fs.  Enables the e2e_latency_ms summary.
        """
        k = np.arange(1, len(self.total_ms) + 1)
        self.available_s = t0_s + (window + k * self.hop) / self.fs

    # ---- reporting ----
    @staticmethod
    def _stats(x) -> dict:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return {}
        return {"median": round(float(np.median(x)), 3), "mean": round(float(x.mean()), 3),
                "p95": round(float(np.percentile(x, 95)), 3), "p99": round(float(np.percentile(x, 99)), 3),
                "max": round(float(x.max()), 3), "min": round(float(x.min()), 3),
                "std": round(float(x.std()), 3)}

    def live_line(self) -> str:
        n = len(self.total_ms)
        w = np.asarray(self.total_ms[-self.window:])
        misses = int(np.sum(np.asarray(self.total_ms) > self.budget_ms))
        temp = f" | {self.temp_c[-1]:.1f}°C" if not np.isnan(self.temp_c[-1]) else ""
        return (f"[{self.label}] hop {n:6d} | last {self.total_ms[-1]:6.2f} ms (run {self.infer_ms[-1]:6.2f}) | "
                f"last {len(w)}: median {np.median(w):6.2f}  p95 {np.percentile(w, 95):6.2f}  max {w.max():6.2f} ms | "
                f"misses {misses}/{n} of {self.budget_ms:.1f} ms budget{temp}")

    def summary(self) -> dict:
        total = np.asarray(self.total_ms)
        n = len(total)
        wall = (self.t_s[-1] - self.t_start) if n else 0.0     # first forward start -> last hop end
        s = {"hop": self.hop, "sample_rate_hz": self.fs, "hop_budget_ms": round(self.budget_ms, 3),
             "n_hops": n, "wall_s": round(wall, 3),
             "first_window_ms": None if self.first_window_ms is None else round(self.first_window_ms, 3),
             "infer_ms": self._stats(self.infer_ms), "total_ms": self._stats(total),
             "overhead_ms": self._stats(total - np.asarray(self.infer_ms)) if n else {}}
        if n:
            p95 = float(np.percentile(total, 95))
            s.update(deadline_misses=int(np.sum(total > self.budget_ms)),
                     miss_rate=round(float(np.mean(total > self.budget_ms)), 4),
                     realtime_headroom_x=round(self.budget_ms / p95, 2) if p95 > 0 else None,
                     throughput_hops_per_s=round(n / wall, 2) if wall > 0 else None,
                     jitter_ms=round(p95 - float(np.median(total)), 3),
                     # Fraction of each hop period spent computing; > 1 cannot keep up.
                     duty_cycle={"mean": round(float(total.mean()) / self.budget_ms, 4),
                                 "p95": round(p95 / self.budget_ms, 4),
                                 "p99": round(float(np.percentile(total, 99)) / self.budget_ms, 4)})
        if n > 1:
            s["interval_ms"] = self._stats(1e3 * np.diff(self.t_s))
        if self.available_s is not None and n:
            e2e = 1e3 * (np.asarray(self.t_s) - self.available_s)
            s["e2e_latency_ms"] = self._stats(e2e)
            if self.lookahead:
                # Newest input available -> output emitted is e2e; the newest
                # emitted *target* was acquired L samples before that input.
                wait = 1e3 * self.lookahead / self.fs
                s["lookahead_samples"] = self.lookahead
                s["lookahead_wait_ms"] = round(wait, 3)
                s["target_e2e_latency_ms"] = self._stats(e2e + wait)
            # Backlog detector: a loop that cannot keep up emits later and later.
            q = max(1, n // 4)
            s["e2e_drift_ms"] = round(float(np.median(e2e[-q:]) - np.median(e2e[:q])), 3)
        temps = np.asarray(self.temp_c)
        if n and not np.all(np.isnan(temps)):
            s["temp_c"] = {"start": round(float(temps[0]), 1), "end": round(float(temps[-1]), 1),
                           "max": round(float(np.nanmax(temps)), 1)}
        # How these milliseconds were produced, not only what they were. The
        # environment block exists so a latency figure cannot be separated
        # from the machine; this is the same argument for the algorithm.
        # Without it a reader has to know which files reach the board to work
        # out that this is the hop-and-recompute path and not a cached one.
        incremental = self.mode == "incremental"
        s["inference"] = {
            "mode": self.mode,
            "runtime": "onnxruntime",
            "window": self.model_window,
            "hop": self.hop,
            "kv_cache": (self.kv_cache if self.kv_cache is not None else False),
            **(self.state or {}),
            "recompute_factor": (1.0 if incremental else
                                 (round(self.model_window / self.hop, 2)
                                  if self.model_window and self.hop else None)),
            "note": ("one sample per call, state carried in and out of the graph; nothing "
                     "is recomputed and the KV cache is a tensor onnxruntime allocates"
                     if incremental else
                     "each hop re-runs the whole window; the exported graph has no cache "
                     "inputs or outputs, so no state carries between calls"),
        }
        if self.environment is not None:
            s["environment"] = self.environment
        if self.scores is not None:
            s["scores"] = self.scores
        s["memory"] = memory_block()
        return s

    def print_summary(self) -> dict:
        s = self.summary()
        if not s["n_hops"]:
            print(f"[{self.label}] no hops recorded")
            return s
        t, r = s["total_ms"], s["infer_ms"]
        print(f"[{self.label}] {s['n_hops']} hops of {s['hop']} samples ({s['hop_budget_ms']} ms budget each), "
              f"{s['wall_s']} s wall, first window {s['first_window_ms']} ms")
        print(f"           per hop   : median {t['median']} | p95 {t['p95']} | p99 {t['p99']} | max {t['max']} ms  "
              f"(jitter p95-median {s['jitter_ms']} ms)")
        print(f"           of which  : onnxruntime {r['median']} ms median / {r['p95']} p95;  "
              f"pre/post-processing {s['overhead_ms']['median']} ms median")
        dc = s["duty_cycle"]
        print(f"           real time : {s['deadline_misses']} deadline misses ({100 * s['miss_rate']:.2f}%), "
              f"{s['realtime_headroom_x']}x headroom at p95, {s['throughput_hops_per_s']} hops/s, "
              f"duty cycle {100 * dc['mean']:.1f}% mean / {100 * dc['p99']:.1f}% p99")
        if "e2e_latency_ms" in s:
            e = s["e2e_latency_ms"]
            print(f"           end-to-end: sample available -> output emitted  median {e['median']} | "
                  f"p95 {e['p95']} | max {e['max']} ms  (drift over run {s['e2e_drift_ms']:+.2f} ms)")
            if "target_e2e_latency_ms" in s:
                g = s["target_e2e_latency_ms"]
                print(f"           lookahead : L = {s['lookahead_samples']} samples, "
                      f"+{s['lookahead_wait_ms']} ms wait; target acquired -> output emitted  "
                      f"median {g['median']} | p95 {g['p95']} | p99 {g['p99']} | max {g['max']} ms")
        if "temp_c" in s:
            c = s["temp_c"]
            print(f"           SoC temp  : {c['start']} -> {c['end']} °C (peak {c['max']})")
        mem = s.get("memory") or {}
        if mem.get("peak_rss_mib") is not None:
            print(f"           memory    : peak {mem['peak_rss_mib']} MiB resident"
                  + (f" ({mem['added_by_session_mib']} MiB above the {mem['rss_before_session_mib']} MiB"
                     f" before the session)" if mem.get("added_by_session_mib") is not None else ""))
        return s

    def save(self, stem: str | Path) -> tuple[Path, Path]:
        """Write <stem>.json (summary) and <stem>.csv (one row per hop)."""
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        js, csv = stem.with_suffix(".json"), stem.with_suffix(".csv")
        js.write_text(json.dumps(self.summary(), indent=2))
        n = len(self.total_ms)
        t0 = self.t_start if self.t_start is not None else 0.0
        cols = [np.arange(1, n + 1), self.infer_ms, self.total_ms, np.asarray(self.t_s) - t0, self.temp_c]
        header, fmt = "hop,infer_ms,total_ms,t_s,temp_c", ["%d", "%.4f", "%.4f", "%.4f", "%.1f"]
        if self.available_s is not None:
            cols.append(1e3 * (np.asarray(self.t_s) - self.available_s))
            header, fmt = header + ",e2e_latency_ms", fmt + ["%.4f"]
        np.savetxt(csv, np.column_stack(cols) if n else np.zeros((0, len(cols))),
                   delimiter=",", header=header, comments="", fmt=fmt)
        return js, csv


class StreamingDenoiser:
    """
    Push-based wrapper for a live acquisition loop: feed chunks as they are
    acquired, get denoised samples back.  Produces exactly the same output
    as EEGDenoiser.denoise_stream() on the concatenated chunks.

    Nothing is emitted until one full window has arrived (2 s at 256 Hz for
    a 512-sample window); that first window is then emitted whole and every
    following `hop` samples are emitted after one forward pass each.  Running
    the model on a zero-filled buffer during warm-up would be cheaper but is
    not what the network saw in training (attention would attend to padding).
    """

    def __init__(self, denoiser: EEGDenoiser, hop: int = 32, monitor: LatencyMonitor | None = None,
                 sigma: float | None = None):
        self.d, self.hop, self.monitor = denoiser, hop, monitor
        if monitor is not None:
            monitor.lookahead = denoiser.lookahead
        self.sigma = normalization_scale(denoiser.meta, sigma=sigma, online=True)
        self.buf = np.zeros(denoiser.window, dtype=np.float32)
        self.filled = 0                                   # real samples in buf
        self.pending = np.zeros(0, dtype=np.float32)      # not yet processed

    def push(self, chunk: np.ndarray) -> np.ndarray:
        """Returns the denoised samples that became available (may be empty)."""
        self.pending = np.concatenate([self.pending, np.asarray(chunk, dtype=np.float32).reshape(-1)])
        W, out = self.d.window, []

        if self.filled < W:                               # warm-up: fill the first window
            take = min(W - self.filled, len(self.pending))
            self.buf[self.filled:self.filled + take] = self.pending[:take]
            self.pending, self.filled = self.pending[take:], self.filled + take
            if self.filled < W:
                return np.zeros(0, dtype=np.float32)
            t0 = time.perf_counter()
            # With lookahead L the newest L outputs still lack their future
            # inputs, so they wait for the next hop: every emitted sample is
            # final, and emission trails acquisition by exactly L samples.
            L = self.d.lookahead
            out.append(self.d.denoise_epochs(self.buf, sigma=self.sigma)[:W - L])
            if self.monitor is not None:
                self.monitor.record_first_window(time.perf_counter() - t0)

        while len(self.pending) >= self.hop:              # steady state: one forward pass per hop
            t0 = time.perf_counter()
            self.buf = np.roll(self.buf, -self.hop)
            self.buf[-self.hop:], self.pending = self.pending[:self.hop], self.pending[self.hop:]
            L = self.d.lookahead
            out.append(self.d.denoise_epochs(self.buf, sigma=self.sigma)[W - self.hop - L:W - L])
            if self.monitor is not None:
                self.monitor.record_hop(self.d.last_run_s, time.perf_counter() - t0)
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def stream_realtime(d: EEGDenoiser, signal: np.ndarray, hop: int = 32,
                    monitor: LatencyMonitor | None = None, chunk: int | None = None,
                    sigma: float | None = None) -> np.ndarray:
    """
    Replay ``signal`` through a StreamingDenoiser at the sample rate, the way
    an acquisition loop would deliver it: a chunk of ``chunk`` samples
    (default: one hop) is pushed only once its last sample "has arrived".
    Takes len(signal) / fs seconds of wall time.  The monitor then measures
    what a live loop would see -- including scheduler jitter -- and its
    e2e_latency_ms is the real sample-available -> output-emitted delay.
    Output is identical to denoise_stream(signal, hop).
    """
    signal = np.asarray(signal, dtype=np.float32)
    chunk = chunk or hop
    s = StreamingDenoiser(d, hop, monitor, sigma=sigma)
    out = []
    t0 = time.perf_counter()
    # Chunk boundaries start at the window, not at sample 0. Cut from 0, a
    # window that is not a multiple of the hop (500 samples at 250 Hz, against
    # 16/32/64) left every hop's data waiting for the remainder of a chunk --
    # 12 samples, 48 ms -- which the e2e figure then reported as latency.
    W = min(d.window, len(signal))
    bounds = [0, W] + list(range(W + chunk, len(signal) + chunk, chunk))
    for a, b in zip(bounds[:-1], bounds[1:]):
        b = min(b, len(signal))
        if a >= b:
            continue
        due = t0 + b / d.sample_rate                                # last sample of this chunk arrives
        now = time.perf_counter()
        if due > now:
            time.sleep(due - now)
        out.append(s.push(signal[a:b]))
    if monitor is not None:
        monitor.attach_schedule(t0, d.window)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def duty_cycle_check(d: EEGDenoiser, signal: np.ndarray, hops=(64, 32, 16, 8, 4), seconds: float = 20.0,
                     report_every: int = 0, environment: dict | None = None,
                     sigma: float | None = None) -> list[dict]:
    """
    Real-time applicability sweep.  For each hop size the first ``seconds`` of
    ``signal`` are replayed at the sample rate (stream_realtime), so every row
    is what a live loop would see on this board:

        duty cycle   compute per hop / hop period (hop / fs).  Each hop costs a
                     full-window forward pass, so halving the hop doubles it.
        e2e latency  sample available -> output emitted, plus its drift over the
                     run: a loop that cannot keep up emits later and later.
        verdict      OK        p99 duty cycle <= 0.7, no misses, no drift
                     MARGINAL  no misses but p99 duty cycle > 0.7, or drift > one hop period / 4
                     FAIL      deadline misses, or drift > one hop period (backlog)

    The total algorithmic latency of the causal stream is the buffering delay
    (hop / fs) plus the compute; the table reports both so the hop can be
    chosen against a latency target with a known safety margin.
    """
    signal = np.asarray(signal, dtype=np.float32)
    n = int(seconds * d.sample_rate)
    if len(signal) < n:
        signal = np.tile(signal, int(np.ceil(n / len(signal))))
    signal = signal[:n]
    rows = []
    print(f"[duty] {seconds:.0f} s paced replay per hop at {d.sample_rate:.0f} Hz, window {d.window}, "
          f"{d.sess.get_session_options().intra_op_num_threads} onnxruntime threads")
    print(f"       {'hop':>4s} {'period':>8s} {'hops/s':>6s} | {'compute ms':>21s} | {'duty %':>15s} | "
          f"{'e2e ms':>15s} {'drift':>7s} | {'misses':>6s} | verdict")
    print(f"       {'':>4s} {'ms':>8s} {'':>6s} | {'median':>6s} {'p99':>6s} {'max':>6s} | {'mean':>6s} {'p99':>7s} | "
          f"{'median':>7s} {'max':>7s} {'ms':>7s} | {'':>6s} |")
    for hop in hops:
        m = LatencyMonitor(hop, d.sample_rate, report_every, label=f"duty/hop{hop}",
                           environment=environment)
        stream_realtime(d, signal, hop, m, sigma=sigma)
        s = m.summary()
        if not s["n_hops"]:
            continue
        dc, e2e, drift = s["duty_cycle"], s["e2e_latency_ms"], s["e2e_drift_ms"]
        if s["deadline_misses"] or drift > s["hop_budget_ms"]:
            verdict = "FAIL"
        elif dc["p99"] > 0.7 or drift > s["hop_budget_ms"] / 4:
            verdict = "MARGINAL"
        else:
            verdict = "OK"
        row = {"hop": hop, "hop_budget_ms": s["hop_budget_ms"], "hops_per_s": round(d.sample_rate / hop, 1),
               "n_hops": s["n_hops"], "compute_median_ms": s["total_ms"]["median"],
               "compute_p99_ms": s["total_ms"]["p99"], "compute_max_ms": s["total_ms"]["max"],
               "duty_mean": dc["mean"], "duty_p99": dc["p99"], "e2e_median_ms": e2e["median"],
               "e2e_max_ms": e2e["max"], "e2e_drift_ms": drift, "deadline_misses": s["deadline_misses"],
               "lookahead_wait_ms": s.get("lookahead_wait_ms", 0.0),
               "total_latency_median_ms": round(s["hop_budget_ms"] + e2e["median"]
                                                + s.get("lookahead_wait_ms", 0.0), 2),
               "temp_c": s.get("temp_c"), "peak_rss_mib": peak_rss_mib(), "verdict": verdict}
        rows.append(row)
        print(f"       {hop:>4d} {row['hop_budget_ms']:>8.2f} {row['hops_per_s']:>6.1f} | "
              f"{row['compute_median_ms']:>6.2f} {row['compute_p99_ms']:>6.2f} {row['compute_max_ms']:>6.2f} | "
              f"{100 * dc['mean']:>6.1f} {100 * dc['p99']:>7.1f} | "
              f"{e2e['median']:>7.2f} {e2e['max']:>7.2f} {drift:>+7.2f} | {s['deadline_misses']:>6d} | {verdict}")
    ok = [r for r in rows if r["verdict"] == "OK"]
    if ok:
        best = min(ok, key=lambda r: r["hop"])
        print(f"[duty] smallest hop that is comfortably real-time: {best['hop']} samples = "
              f"{best['hop_budget_ms']:.1f} ms buffering + "
              + (f"{best['lookahead_wait_ms']:.1f} ms lookahead + " if best.get("lookahead_wait_ms") else "")
              + f"{best['e2e_median_ms']:.1f} ms compute "
              f"= {best['total_latency_median_ms']:.1f} ms total latency at {100 * best['duty_mean']:.0f}% duty cycle")
    else:
        print("[duty] no hop in the sweep is comfortably real-time on this board")
    mem = memory_block()
    if mem["peak_rss_mib"] is not None:
        print(f"[duty] peak process memory {mem['peak_rss_mib']} MiB resident over the sweep")
    return rows


def save_duty_rows(rows: list[dict], stem: str | Path,
                   environment: dict | None = None) -> tuple[Path, Path]:
    """
    Write the sweep as ``{"rows": [...], "environment": {...}}``.

    Earlier sweeps were a bare list; readers accept both, but a sweep written
    from now on carries the state of the machine it was measured on, because a
    duty cycle is only a number about the model if the clock was held still.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    js, csv = stem.with_suffix(".json"), stem.with_suffix(".csv")
    js.write_text(json.dumps({"rows": rows, "environment": environment, "memory": memory_block()},
                             indent=2))
    keys = [k for k in rows[0] if k != "temp_c"] if rows else []
    lines = [",".join(keys)] + [",".join(str(r[k]) for k in keys) for r in rows]
    csv.write_text("\n".join(lines) + "\n")
    return js, csv


# ============================================================
# Process memory
# ============================================================
#: Resident memory just before the onnxruntime session is built, set by main().
#: The interpreter, numpy and onnxruntime are already loaded by then, so peak
#: minus this is what the session, the model and the replayed signal added.
_RSS_BEFORE_SESSION_MIB: float | None = None
_RSS_AFTER_SESSION_MIB: float | None = None


def rss_now_mib() -> float | None:
    """Current resident set size (Linux /proc/self/status VmRSS); None elsewhere."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)        # kB -> MiB
    except OSError:
        pass
    return None


def peak_rss_mib() -> float | None:
    """
    Peak resident set size of this process over its whole life, in MiB.

    getrusage's ru_maxrss: kilobytes on Linux, bytes on macOS. It is the
    kernel's high-water mark, so it catches a transient peak between two
    samples that polling would miss, and it never goes down.
    """
    try:
        import resource
    except ImportError:                                  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)


def memory_block() -> dict:
    """What a trace records about memory, read at the moment it is written."""
    peak = peak_rss_mib()
    before = _RSS_BEFORE_SESSION_MIB
    return {"peak_rss_mib": peak,
            "rss_before_session_mib": before,
            "rss_after_session_mib": _RSS_AFTER_SESSION_MIB,
            "added_by_session_mib": (round(peak - before, 1)
                                     if peak is not None and before is not None else None),
            "source": "getrusage ru_maxrss (process high-water mark, MiB)"}


# ============================================================
# EEGdenoiseNet metrics (numpy replicas of metrics.py)
# ============================================================
def _rms(x):
    return np.sqrt(np.mean(x ** 2, axis=-1) + 1e-12)


def rrmse_temporal(pred, clean):
    return _rms(pred - clean) / _rms(clean)


def rrmse_spectral(pred, clean):
    p = np.abs(np.fft.rfft(pred, axis=-1)) ** 2
    c = np.abs(np.fft.rfft(clean, axis=-1)) ** 2
    return _rms(p - c) / _rms(c)


def acc(pred, clean):
    pc = pred - pred.mean(axis=-1, keepdims=True)
    cc = clean - clean.mean(axis=-1, keepdims=True)
    cov = (pc * cc).mean(axis=-1)
    return cov / (np.sqrt((pc ** 2).mean(axis=-1) * (cc ** 2).mean(axis=-1)) + 1e-12)


def metrics(pred, clean):
    """Per-epoch (N,) arrays for (N, T) inputs."""
    pred, clean = np.asarray(pred, np.float32), np.asarray(clean, np.float32)
    return {"rrmse_t": rrmse_temporal(pred, clean), "rrmse_s": rrmse_spectral(pred, clean),
            "acc": acc(pred, clean)}


#: Filenames deploy/make_testdata.py writes: <ARTIFACT>_test<N>[_s<SEED>].npz
_TESTDATA_NAME = re.compile(r"^(EOG|EMG|BOTH)_test(\d+)(?:_s(\d+))?(?:_[\d.]+hz)?\.npz$")


def testdata_provenance(npz_path: str | Path, data=None) -> dict:
    """
    Which split a test set was cut from, and how confidently that is known.

    ``source`` is ``"recorded"`` when the arrays carry it (make_testdata stamps
    ``master_seed`` / ``artifact`` / ``split`` into the file), ``"filename"``
    when it had to be read off the name, and ``"unknown"`` when neither works.
    Only ``recorded`` is a guarantee: a name is a convention, and renaming a
    file changes what it claims without changing what it holds.
    """
    path = Path(npz_path)
    data = np.load(path) if data is None else data
    keys = set(getattr(data, "files", []) or data.keys())

    if "master_seed" in keys:
        return {"source": "recorded",
                "master_seed": int(data["master_seed"]),
                "artifact": str(data["artifact"]) if "artifact" in keys else None,
                "split": str(data["split"]) if "split" in keys else "test",
                "split_ratios": data["split_ratios"].tolist() if "split_ratios" in keys else None,
                "sampling": str(data["sampling"]) if "sampling" in keys else None,
                "resampling": str(data["resampling"]) if "resampling" in keys else None,
                "fs": float(data["fs"]) if "fs" in keys else None}

    m = _TESTDATA_NAME.match(path.name)
    if m:
        # No suffix means the default seed, which is the convention
        # make_testdata writes under.
        return {"source": "filename", "artifact": m.group(1),
                "master_seed": int(m.group(3)) if m.group(3) else 42, "split": "test"}
    return {"source": "unknown", "artifact": None, "master_seed": None, "split": None}


def check_split(meta: dict, prov: dict, *, strict: bool = True) -> list[str]:
    """
    Refuse to score a model on epochs it may have trained on.

    "Test set" is not a property of a file on its own.  ``EEGDenoiseDataset``
    permutes the epoch pool with ``master_seed`` and slices train/val/test out
    of that permutation, so a set cut at seed 42 is 70% *training data* for a
    run trained at seed 67.  Scoring across that boundary does not fail, warn,
    or look odd — it quietly returns a better number than the model earns,
    which is the worst way for a measurement to be wrong.

    Returns the problems found.  With ``strict`` a seed mismatch raises, since
    the resulting figure is not a test score at all; everything else is
    reported and left to the caller, because a deliberate cross-artifact probe
    is a legitimate thing to run.
    """
    problems, warnings = [], []
    model = (meta or {}).get("provenance") or {}
    cfg = (meta or {}).get("data_protocol") or {}
    if cfg.get("protocol") == "paper":
        if not strict or prov.get("source") != "recorded":
            raise ValueError("Paper scoring requires recorded provenance and strict split checks")
        for key in ("sampling", "resampling", "split_ratios"):
            if prov.get(key) is None:
                raise ValueError(f"Test data lacks {key}; regenerate with --ckpt")
            expected = cfg.get(key, [0.7, 0.15, 0.15] if key == "split_ratios" else None)
            match = np.allclose(prov[key], expected) if key == "split_ratios" else prov[key] == expected
            if not match:
                raise ValueError(f"Test data {key} differs from checkpoint protocol")
    trained_seed = model.get("trained_master_seed")
    trained_artifact = (model.get("trained_artifact")
                        or ((meta or {}).get("preprocessing") or {}).get("artifact_trained_on"))

    if prov["split"] not in (None, "test"):
        problems.append(f"this file is the '{prov['split']}' split, not the test split")

    if prov["master_seed"] is None or trained_seed is None:
        why = ("the model's checkpoint predates run_config, so the seed it trained at was not recorded"
               if trained_seed is None else
               f"{Path(prov.get('path', 'the test set')).name} carries no provenance")
        warnings.append(f"cannot verify the split: {why}. If the two seeds differ, "
                        f"these epochs are the model's training data and the score is meaningless.")
    elif int(prov["master_seed"]) != int(trained_seed):
        problems.append(
            f"LEAK: the test set was cut from the master_seed {prov['master_seed']} split, but this "
            f"model was trained at master_seed {trained_seed}. The splits are different permutations "
            f"of the same pool, so most of these epochs are in that model's training set and the "
            f"score would be inflated. Cut a matching set with: python -m deploy.make_testdata "
            f"--artifact {prov['artifact'] or 'EOG'} --master_seed {trained_seed}")

    if prov["artifact"] and trained_artifact and prov["artifact"] != trained_artifact:
        warnings.append(f"the model was trained on {trained_artifact} but this set is "
                        f"{prov['artifact']} — a cross-artifact probe, not a like-for-like score")

    if prov["source"] == "filename":
        warnings.append(f"split read from the filename, not from the file — rename it and this "
                        f"check is wrong. Re-cut it to stamp the provenance in.")

    for w in warnings:
        print(f"[test] WARNING: {w}")
    if problems:
        joined = "\n       ".join(problems)
        if strict:
            raise ValueError(f"refusing to score this pair:\n       {joined}\n       "
                             f"pass --allow-split-mismatch to score it anyway")
        for x in problems:
            print(f"[test] WARNING (forced): {x}")
    return problems + warnings


def evaluate_testdata(d: EEGDenoiser, npz_path: str | Path, hop: int = 32, per_snr: bool = True,
                      monitor: LatencyMonitor | None = None, strict_split: bool = True) -> dict:
    data = np.load(npz_path)
    # Before anything is scored: are these epochs this model has never seen?
    prov = testdata_provenance(npz_path, data)
    prov["path"] = str(npz_path)
    check_split(d.meta, prov, strict=strict_split)
    if "fs" in data and not np.isclose(float(data["fs"]), d.sample_rate):
        raise ValueError(f"test data rate {float(data['fs']):g} Hz differs from model rate {d.sample_rate:g} Hz")
    noisy, clean, snr = data["noisy"], data["clean"], data["snr_db"]
    n, T = noisy.shape
    cfg = d.meta.get("data_protocol") or {}
    paper = cfg.get("protocol") == "paper"
    if paper and hop != 4:
        raise ValueError("Paper evaluation requires H=4")
    stepping = isinstance(d, IncrementalDenoiser)
    if not stepping and T != d.window:
        # The windowed graph's input is a fixed 512 wide, so an epoch of any
        # other length cannot be fed to it at all. A step graph takes one
        # sample at a time and will consume an epoch of any length -- its
        # ``window`` is how far back an output reaches, not a required shape.
        raise ValueError(f"test data has {T}-sample epochs but the model window is {d.window}")
    if stepping and T < d.window:
        print(f"[test] note: {T}-sample epochs are shorter than this model's {d.window}-sample "
              f"history, so no output in them sees a full receptive field")

    t0 = time.perf_counter()
    y_epoch = d.denoise_epochs(noisy) if not stepping else d.denoise_epochs(noisy, monitor)
    t_epoch = time.perf_counter() - t0

    if stepping:
        # There is no second mode to compare: a step graph emits on every
        # sample, so "epoch mode" and "stream mode" are the same computation.
        # Reporting a separate stream row would be the same numbers twice.
        #
        # Named for what produced it. The windowed path's "epoch mode" is one
        # batched 512-wide forward per epoch; this never does that, it steps a
        # sample at a time and resets its state between epochs. The scoring
        # protocol is the same -- independent epochs, each normalised by its
        # own sigma -- which is what keeps the two numbers comparable, and the
        # suffix is there so the mechanism is not read off the label wrongly.
        t_stream = 0.0
        modes = {"noisy input (no model)": noisy, "epoch mode (stepped)": y_epoch}
    elif (d.meta.get("preprocessing") or {}).get("normalization", "epoch") == "fixed":
        t0 = time.perf_counter()
        y_stream = (np.stack([d.denoise_stream(row, hop, monitor) for row in noisy]) if paper
                    else d.denoise_stream(noisy.reshape(-1), hop, monitor).reshape(n, T))
        t_stream = time.perf_counter() - t0
        modes = {"noisy input (no model)": noisy, "epoch mode": y_epoch,
                 f"stream mode (hop {hop})": y_stream}
    else:
        t_stream = 0.0
        modes = {"noisy input (no model)": noisy, "epoch mode (offline)": y_epoch}
        print("[test] legacy epoch normalization: reporting offline metrics only")
    start = (cfg.get("valid_history_samples") or 1) - 1
    delay = cfg.get("output_delay", 0)
    # The last L outputs of an epoch read past its end, so no streamed output
    # for them is ever final; they are excluded from every row alike.
    stop = T - max(cfg.get("valid_future_samples", 0), delay, getattr(d, "lookahead", 0))
    if start >= stop:
        raise ValueError("No common complete target interval in test data")
    results = {name: metrics(y[..., start + (0 if name.startswith('noisy') else delay):
                                  stop + (0 if name.startswith('noisy') else delay)], clean[..., start:stop])
               for name, y in modes.items()}

    print(f"[test] {Path(npz_path).name}: {n} epochs x {T} samples, SNR {snr.min():+.1f}..{snr.max():+.1f} dB"
          + (f", {prov['artifact']} test split at master_seed {prov['master_seed']}"
             f" ({prov['source']})" if prov["master_seed"] is not None else ""))
    print(f"       {'':<26s} {'RRMSE_t':>8s} {'RRMSE_f':>8s} {'CC':>7s}")
    for name, m in results.items():
        print(f"       {name:<26s} {m['rrmse_t'].mean():8.4f} {m['rrmse_s'].mean():8.4f} {m['acc'].mean():7.4f}")
    if stepping:
        print(f"       stepped {n * T:,} samples in {t_epoch:.2f} s "
              f"({1e3 * t_epoch / (n * T):.3f} ms/sample, "
              f"{n * T / d.sample_rate:.0f} s of signal); state reset per epoch")
    elif any(k.startswith("stream mode") for k in modes):
        print(f"       epoch mode: {1e3 * t_epoch / n:.2f} ms/epoch (batched)   "
              f"stream mode: {1e3 * t_stream / (n * T / hop):.2f} ms/hop, {t_stream:.2f} s for {n * T / d.sample_rate:.0f} s of signal")
        print("       note: the stream row concatenates independent epochs, so every window straddles two unrelated "
              "signals at different scales; it is a pessimistic bound, not what a continuous recording would give.")

    if per_snr:
        # Whichever epoch row this run produced: "epoch mode" from the
        # windowed path, "epoch mode (stepped)" from the step graph.
        epoch_label = next(k for k in results if k.startswith("epoch mode"))
        m = results[epoch_label]
        print(f"       per SNR ({epoch_label}):  {'SNR dB':>7s} {'n':>3s} "
              f"{'RRMSE_t':>8s} {'RRMSE_f':>8s} {'CC':>7s}")
        for level in np.unique(np.round(snr, 2)):
            sel = np.isclose(np.round(snr, 2), level)
            print(f"       {'':<22s} {level:+7.1f} {sel.sum():3d} {m['rrmse_t'][sel].mean():8.4f} "
                  f"{m['rrmse_s'][sel].mean():8.4f} {m['acc'][sel].mean():7.4f}")
    return {
        "testdata": Path(npz_path).name,
        "n_epochs": int(n),
        "hop": int(hop),
        "split": {k: prov.get(k) for k in ("artifact", "master_seed", "source")},
        "modes": {name: {k: float(v.mean()) for k, v in m.items()}
                  for name, m in results.items()},
    }


def benchmark_steps(d: "IncrementalDenoiser", n_iter: int = 2000) -> dict:
    """
    Time one step of the incremental graph, warmed up.

    The state is advanced for a while first: a step graph's cost is flat in
    how full its cache is (every call attends over the whole window, masked),
    but the allocator and the thread pool are not warm on the first calls and
    reporting those as the cost would be unfair to it.
    """
    state = d.initial_state()
    for _ in range(min(200, n_iter // 4)):
        _, state = d.step(0.0, state)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        _, state = d.step(0.0, state)
        times.append((time.perf_counter() - t0) * 1e3)
    t = np.asarray(times)
    budget = 1e3 / d.sample_rate
    med = float(np.median(t))
    return {"n_iter": n_iter, "median_ms": round(med, 4),
            "p95_ms": round(float(np.percentile(t, 95)), 4),
            "max_ms": round(float(t.max()), 4),
            "budget_ms": round(budget, 3), "sample_rate_hz": d.sample_rate,
            "realtime_headroom_x": round(budget / med, 1) if med else None,
            "max_fs_hz": round(1e3 / med, 1) if med else None,
            "state_bytes": d.state_bytes()}


def benchmark(d: EEGDenoiser, n_iter: int = 200, hop: int = 32) -> dict:
    x = np.random.default_rng(0).standard_normal((1, 1, d.window), dtype=np.float32)
    for _ in range(20):
        d._run(x)
    t = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        d._run(x)
        t.append(time.perf_counter() - t0)
    t = 1e3 * np.asarray(t)
    budget_ms = 1e3 * hop / d.sample_rate
    return {
        "window": d.window, "median_ms": round(float(np.median(t)), 2),
        "p95_ms": round(float(np.percentile(t, 95)), 2), "max_ms": round(float(t.max()), 2),
        "hop": hop, "hop_budget_ms": round(budget_ms, 1),
        "realtime_headroom_x": round(budget_ms / float(np.percentile(t, 95)), 1),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("onnx", help="exported model, e.g. tcn_mla_eog.onnx (sidecar .json optional)")
    p.add_argument("--input", help=".npy: (T,) / (N, T) epochs, or a long 1-D signal with --stream")
    p.add_argument("--output", help=".npy to write the denoised result to")
    p.add_argument("--stream", action="store_true", help="treat --input as one continuous signal")
    p.add_argument("--sigma", type=float, default=None,
                   help="Fixed scale calibrated before the input stream (required for legacy online models)")
    p.add_argument("--hop", type=int, default=32, help="stream hop in samples (latency = hop / fs)")
    p.add_argument("--threads", type=int, default=2, help="onnxruntime intra-op threads (2 is best on a Pi 5)")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--testdata", help=".npz from deploy/make_testdata.py; prints benchmark metrics")
    p.add_argument("--incremental", action="store_true",
                   help="drive <model>_step.onnx one sample at a time, carrying its state, "
                        "instead of re-running a whole window per hop. The only path with a "
                        "real KV cache; emits on every sample, so it has no hop buffering.")
    p.add_argument("--allow-split-mismatch", action="store_true",
                   help="score even when the test set was cut from a different split than the "
                        "model trained on. The number is then not a test score — the epochs are "
                        "in the model's training set. For debugging only.")
    p.add_argument("--latency", action="store_true",
                   help="per-hop latency monitor for --stream / --testdata: onnxruntime time, total hop time, "
                        "deadline misses against hop/fs, temperature; prints a summary")
    p.add_argument("--latency_every", type=int, default=0, metavar="N",
                   help="with --latency: print a rolling-window line every N hops (default: summary only)")
    p.add_argument("--latency_out", metavar="STEM",
                   help="with --latency: write STEM.json (summary) and STEM.csv (one row per hop)")
    p.add_argument("--realtime", action="store_true",
                   help="with --stream: replay the input at the sample rate through StreamingDenoiser "
                        "(takes len/fs seconds) so --latency reports true end-to-end latency")
    p.add_argument("--duty_cycle", metavar="HOPS",
                   help="real-time applicability sweep over comma-separated hop sizes, e.g. 64,32,16,8,4: "
                        "paced replay per hop, duty cycle + e2e latency + verdict. --input / --testdata "
                        "then only supply the signal (no scoring / denoising output)")
    p.add_argument("--duty_seconds", type=float, default=20.0, help="paced seconds per hop in --duty_cycle")
    p.add_argument("--cpus", metavar="LIST",
                   help="pin this process to these CPUs before measuring, e.g. 0-1 "
                        "(Linux only; one CPU per thread)")
    p.add_argument("--require-locked", action="store_true",
                   help="refuse to measure unless the environment check passes: governor "
                        "at performance, pinned CPUs, explicit thread count, no throttling")
    p.add_argument("--no-preflight", action="store_true",
                   help="skip the environment check — the traces then record nothing "
                        "about the conditions the numbers were taken under")
    p.add_argument("--no-spin", action="store_true",
                   help="stop onnxruntime's thread pool busy-waiting between inferences. "
                        "Measured over a paced replay: spinning costs ~31%% of a core "
                        "against ~18%% and buys ~1 ms per hop, so it changes how much of "
                        "the board is left for the rest of the system, not whether the "
                        "loop keeps up. Recorded either way")
    a = p.parse_args(argv)

    # Order matters here.  Pinning first means onnxruntime's worker threads
    # inherit the affinity mask when the session spawns them; pinning after
    # would leave them wherever the scheduler had already put them.  The
    # session is then built -- cheap, and nothing is timed yet -- so that the
    # check can report the threading the session really has rather than the
    # number that was asked for.  Only then can a run be refused.
    if a.cpus:
        try:
            print(f"[preflight] pinned to CPU {preflight.format_cpu_list(preflight.pin(a.cpus))}")
        except (ValueError, RuntimeError) as e:
            p.error(f"--cpus {a.cpus}: {e}")

    global _RSS_BEFORE_SESSION_MIB, _RSS_AFTER_SESSION_MIB
    _RSS_BEFORE_SESSION_MIB = rss_now_mib()
    if a.incremental:
        d = IncrementalDenoiser(a.onnx, a.threads, spin=not a.no_spin)
        # Same shape as the windowed line above, deliberately: the GUI's log
        # parser matches one pattern, and the step-graph detail goes in the
        # trailing part it already passes through rather than in the middle
        # of the fields it reads.
        print(f"[raspi] {a.onnx}: window={d.window} fs={d.sample_rate} Hz  "
              f"providers={d.sess.get_providers()}  "
              f"intra_op={d.threads['intra_op']}  "
              f"spinning={'on' if d.threads['allow_spinning'] else 'off'}  "
              f"incremental step graph, {len(d.state_names)} state tensors, "
              f"{d.state_bytes():,} B")
    else:
        d = EEGDenoiser(a.onnx, a.threads, spin=not a.no_spin)
        print(f"[raspi] {a.onnx}: window={d.window} fs={d.sample_rate} Hz  "
              f"providers={d.sess.get_providers()}  "
              f"intra_op={d.threads['intra_op']} ({d.threads['workers']} "
              f"worker{'s' if d.threads['workers'] != 1 else ''} + the caller)  "
              f"spinning={'on' if d.threads['allow_spinning'] else 'off'}"
              + (f"  lookahead={d.lookahead} ({1000 * d.lookahead / d.sample_rate:g} ms, "
                 f"outputs emitted {d.lookahead} samples late)" if d.lookahead else ""))

    _RSS_AFTER_SESSION_MIB = rss_now_mib()

    env = None
    if not a.no_preflight:
        env = preflight.check(a.threads, runtime=d.threads)
        preflight.emit(env)
        if a.require_locked and env["verdict"] != preflight.LOCKED:
            for reason in env["blockers"]:
                print(f"[preflight]   {reason}", file=sys.stderr)
            raise SystemExit(
                f"[preflight] refusing to measure: the machine is {env['verdict']} and "
                "this run was asked to be reportable (drop --require-locked to measure anyway)")

    def new_monitor(label):
        # Stepping emits on every sample, so its deadline is one sample
        # period rather than a hop of them. Recording hop=1 is what makes the
        # duty cycle comparable with a windowed run at any hop.
        return LatencyMonitor(1 if a.incremental else a.hop,
                              d.sample_rate, a.latency_every, label=label,
                              environment=env,
                              model_window=d.window,
                              mode="incremental" if a.incremental else "windowed",
                              kv_cache=(any(n.startswith("kv_") for n in d.state_names)
                                        if a.incremental else False),
                              state=(d.state_summary() if a.incremental else None),
                              ) if a.latency else None

    def finish_monitor(m, stem_suffix):
        if m is None:
            return
        m.print_summary()
        if a.latency_out:
            js, csv = m.save(a.latency_out + stem_suffix)
            print(f"[{m.label}] wrote {js} and {csv}")

    if a.benchmark and isinstance(d, IncrementalDenoiser):
        # The windowed benchmark times one forward over a whole window, which
        # a step graph has no equivalent of: it advances one sample. Timing
        # the step is the comparable thing, and the per-sample figure is what
        # the two paths are judged on anyway.
        r = benchmark_steps(d)
        print(f"[bench] median {r['median_ms']} ms | p95 {r['p95_ms']} ms | max {r['max_ms']} ms "
              f"per sample;  one sample = {r['budget_ms']} ms at {r['sample_rate_hz']:g} Hz "
              f"-> {r['realtime_headroom_x']}x real-time headroom, "
              f"{r['max_fs_hz']:,.0f} Hz ceiling")
    elif a.benchmark:
        r = benchmark(d, hop=a.hop)
        print(f"[bench] median {r['median_ms']} ms | p95 {r['p95_ms']} ms | max {r['max_ms']} ms per "
              f"{r['window']}-sample window;  hop {r['hop']} = {r['hop_budget_ms']} ms budget "
              f"-> {r['realtime_headroom_x']}x real-time headroom")

    if a.duty_cycle and a.incremental:
        # The sweep answers "which hop can this keep up at", and a step graph
        # has no hop to sweep: it emits on every sample, so its hop is 1 and
        # its duty cycle is one number the latency run already reports.
        #
        # Worse than meaningless, it was wrong. duty_cycle_check wraps the
        # denoiser in the windowed StreamingDenoiser, which feeds it a whole
        # window per hop -- so the step graph re-stepped 125 samples for every
        # emitted hop, recomputing exactly what carrying state exists to
        # avoid, and the result came out as a clean OK/FAIL table.
        print("[duty] ERROR: --duty_cycle sweeps hop sizes, and --incremental has no hop: it "
              "emits on every sample. Its duty cycle is in the latency run (hop 1). Run the "
              "sweep against the windowed graph instead.", file=sys.stderr)
        return 2

    if a.duty_cycle:
        if a.input:
            sig = np.load(a.input).reshape(-1)
        elif a.testdata:
            sig = np.load(a.testdata)["noisy"].reshape(-1)
        else:
            p.error("--duty_cycle needs --input (1-D .npy) or --testdata (.npz) as the signal")
        hops = [int(h) for h in a.duty_cycle.split(",")]
        rows = duty_cycle_check(d, sig, hops, a.duty_seconds, a.latency_every, environment=env, sigma=a.sigma)
        if a.latency_out and rows:
            js, csv = save_duty_rows(rows, a.latency_out + "_duty", environment=env)
            print(f"[duty] wrote {js} and {csv}")
        return                       # --input / --testdata were only the signal source

    if a.testdata:
        m = new_monitor("latency/testdata")
        try:
            scores = evaluate_testdata(d, a.testdata, a.hop, monitor=m,
                                       strict_split=not a.allow_split_mismatch)
            if m is not None:
                m.scores = scores
        except ValueError as exc:
            # This runs over SSH on a board; a traceback here is noise around a
            # sentence the operator needs to read. The GUI tails these lines too.
            print(f"[test] ERROR: {exc}", file=sys.stderr)
            return 1
        finish_monitor(m, "_testdata" if a.input and a.stream else "")

    if a.input:
        x = np.load(a.input)
        if a.stream:
            m = new_monitor("latency/realtime" if a.realtime else "latency/stream")
            if a.realtime:
                print(f"[raspi] replaying {x.size / d.sample_rate:.1f} s of signal at {d.sample_rate} Hz ...")
                y = (d.stream_realtime(x.reshape(-1), sigma=a.sigma, monitor=m)
                     if isinstance(d, IncrementalDenoiser)
                     else stream_realtime(d, x.reshape(-1), a.hop, m, sigma=a.sigma))
            else:
                if isinstance(d, IncrementalDenoiser):
                    y = d.denoise_stream(x.reshape(-1), sigma=a.sigma, monitor=m)
                else:
                    y = d.denoise_stream(x.reshape(-1), a.hop, m, sigma=a.sigma)
            finish_monitor(m, "_stream" if a.testdata else "")
        else:
            y = d.denoise_epochs(x)
        print(f"[raspi] denoised {x.shape} -> {y.shape}")
        if a.output:
            np.save(a.output, y.astype(np.float32))
            print(f"[raspi] wrote {a.output}")


if __name__ == "__main__":
    raise SystemExit(main())
