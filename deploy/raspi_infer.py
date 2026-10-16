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
        self.sample_rate = self.meta.get("preprocessing", {}).get("sample_rate_hz", 256)

        self.last_run_s = 0.0        # wall time of the most recent sess.run (read by LatencyMonitor)

    # ---- core call: (N, 1, T) normalised -> (N, 1, T) normalised ----
    def _run(self, x: np.ndarray) -> np.ndarray:
        if not self.dynamic_batch and x.shape[0] != 1:
            return np.concatenate([self._run(row[None]) for row in x])
        t0 = time.perf_counter()
        y = self.sess.run([self.output_name], {self.input_name: x})[0]
        self.last_run_s = time.perf_counter() - t0
        return y

    def denoise_epochs(self, x: np.ndarray) -> np.ndarray:
        """(T,) or (N, T) raw-scale windows -> same shape, denoised, raw scale."""
        x = np.asarray(x, dtype=np.float32)
        squeeze = x.ndim == 1
        x = x.reshape(-1, self.window)
        sigma = np.maximum(x.std(axis=1, keepdims=True), EPS)
        y = self._run((x / sigma)[:, None, :])[:, 0, :] * sigma
        return y[0] if squeeze else y

    def denoise_stream(self, signal: np.ndarray, hop: int = 32,
                       monitor: "LatencyMonitor | None" = None) -> np.ndarray:
        """
        Causally denoise a long 1-D signal; output is sample-aligned with the input.
        ``monitor`` (a LatencyMonitor) records the cost of every hop.
        """
        signal = np.asarray(signal, dtype=np.float32)
        T = self.window
        if len(signal) < T:
            # Right-pad with zeros: causality guarantees the padding cannot
            # influence the real samples, but normalise on real samples only.
            sigma = max(float(signal.std()), EPS)
            xn = np.zeros((1, 1, T), dtype=np.float32)
            xn[0, 0, : len(signal)] = signal / sigma
            return self._run(xn)[0, 0, : len(signal)] * sigma
        out = np.empty_like(signal)
        t0 = time.perf_counter()
        out[:T] = self.denoise_epochs(signal[:T])
        if monitor is not None:
            monitor.record_first_window(time.perf_counter() - t0)
        for end in range(T + hop, len(signal) + hop, hop):
            end = min(end, len(signal))
            t0 = time.perf_counter()
            out[end - hop:end] = self.denoise_epochs(signal[end - T:end])[-hop:]
            if monitor is not None:
                monitor.record_hop(self.last_run_s, time.perf_counter() - t0)
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
                 environment: dict | None = None):
        self.hop, self.fs = int(hop), float(sample_rate)
        self.budget_ms = 1e3 * self.hop / self.fs
        self.report_every, self.window, self.label = int(report_every), int(window), label
        # The state of the machine these numbers were taken on, so the trace
        # file carries the conditions and not only the result.
        self.environment = environment
        self.infer_ms: list[float] = []
        self.total_ms: list[float] = []
        self.t_s: list[float] = []
        self.temp_c: list[float] = []
        self.first_window_ms: float | None = None
        self.available_s: np.ndarray | None = None     # set by attach_schedule()
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
            # Backlog detector: a loop that cannot keep up emits later and later.
            q = max(1, n // 4)
            s["e2e_drift_ms"] = round(float(np.median(e2e[-q:]) - np.median(e2e[:q])), 3)
        temps = np.asarray(self.temp_c)
        if n and not np.all(np.isnan(temps)):
            s["temp_c"] = {"start": round(float(temps[0]), 1), "end": round(float(temps[-1]), 1),
                           "max": round(float(np.nanmax(temps)), 1)}
        if self.environment is not None:
            s["environment"] = self.environment
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
        if "temp_c" in s:
            c = s["temp_c"]
            print(f"           SoC temp  : {c['start']} -> {c['end']} °C (peak {c['max']})")
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

    def __init__(self, denoiser: EEGDenoiser, hop: int = 32, monitor: LatencyMonitor | None = None):
        self.d, self.hop, self.monitor = denoiser, hop, monitor
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
            out.append(self.d.denoise_epochs(self.buf))
            if self.monitor is not None:
                self.monitor.record_first_window(time.perf_counter() - t0)

        while len(self.pending) >= self.hop:              # steady state: one forward pass per hop
            t0 = time.perf_counter()
            self.buf = np.roll(self.buf, -self.hop)
            self.buf[-self.hop:], self.pending = self.pending[:self.hop], self.pending[self.hop:]
            out.append(self.d.denoise_epochs(self.buf)[-self.hop:])
            if self.monitor is not None:
                self.monitor.record_hop(self.d.last_run_s, time.perf_counter() - t0)
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def stream_realtime(d: EEGDenoiser, signal: np.ndarray, hop: int = 32,
                    monitor: LatencyMonitor | None = None, chunk: int | None = None) -> np.ndarray:
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
    s = StreamingDenoiser(d, hop, monitor)
    out = []
    t0 = time.perf_counter()
    for i in range(0, len(signal), chunk):
        due = t0 + min(i + chunk, len(signal)) / d.sample_rate     # last sample of this chunk arrives
        now = time.perf_counter()
        if due > now:
            time.sleep(due - now)
        out.append(s.push(signal[i:i + chunk]))
    if monitor is not None:
        monitor.attach_schedule(t0, d.window)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def duty_cycle_check(d: EEGDenoiser, signal: np.ndarray, hops=(64, 32, 16, 8, 4), seconds: float = 20.0,
                     report_every: int = 0, environment: dict | None = None) -> list[dict]:
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
        stream_realtime(d, signal, hop, m)
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
               "total_latency_median_ms": round(s["hop_budget_ms"] + e2e["median"], 2),
               "temp_c": s.get("temp_c"), "verdict": verdict}
        rows.append(row)
        print(f"       {hop:>4d} {row['hop_budget_ms']:>8.2f} {row['hops_per_s']:>6.1f} | "
              f"{row['compute_median_ms']:>6.2f} {row['compute_p99_ms']:>6.2f} {row['compute_max_ms']:>6.2f} | "
              f"{100 * dc['mean']:>6.1f} {100 * dc['p99']:>7.1f} | "
              f"{e2e['median']:>7.2f} {e2e['max']:>7.2f} {drift:>+7.2f} | {s['deadline_misses']:>6d} | {verdict}")
    ok = [r for r in rows if r["verdict"] == "OK"]
    if ok:
        best = min(ok, key=lambda r: r["hop"])
        print(f"[duty] smallest hop that is comfortably real-time: {best['hop']} samples = "
              f"{best['hop_budget_ms']:.1f} ms buffering + {best['e2e_median_ms']:.1f} ms compute "
              f"= {best['total_latency_median_ms']:.1f} ms total latency at {100 * best['duty_mean']:.0f}% duty cycle")
    else:
        print("[duty] no hop in the sweep is comfortably real-time on this board")
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
    js.write_text(json.dumps({"rows": rows, "environment": environment}, indent=2))
    keys = [k for k in rows[0] if k != "temp_c"] if rows else []
    lines = [",".join(keys)] + [",".join(str(r[k]) for k in keys) for r in rows]
    csv.write_text("\n".join(lines) + "\n")
    return js, csv


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


def evaluate_testdata(d: EEGDenoiser, npz_path: str | Path, hop: int = 32, per_snr: bool = True,
                      monitor: LatencyMonitor | None = None) -> dict:
    data = np.load(npz_path)
    noisy, clean, snr = data["noisy"], data["clean"], data["snr_db"]
    n, T = noisy.shape
    if T != d.window:
        raise ValueError(f"test data has {T}-sample epochs but the model window is {d.window}")

    t0 = time.perf_counter()
    y_epoch = d.denoise_epochs(noisy)
    t_epoch = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_stream = d.denoise_stream(noisy.reshape(-1), hop, monitor).reshape(n, T)
    t_stream = time.perf_counter() - t0

    modes = {"noisy input (no model)": noisy, "epoch mode": y_epoch, f"stream mode (hop {hop})": y_stream}
    results = {name: metrics(y, clean) for name, y in modes.items()}

    print(f"[test] {Path(npz_path).name}: {n} epochs x {T} samples, SNR {snr.min():+.1f}..{snr.max():+.1f} dB")
    print(f"       {'':<26s} {'RRMSE_t':>8s} {'RRMSE_f':>8s} {'CC':>7s}")
    for name, m in results.items():
        print(f"       {name:<26s} {m['rrmse_t'].mean():8.4f} {m['rrmse_s'].mean():8.4f} {m['acc'].mean():7.4f}")
    print(f"       epoch mode: {1e3 * t_epoch / n:.2f} ms/epoch (batched)   "
          f"stream mode: {1e3 * t_stream / (n * T / hop):.2f} ms/hop, {t_stream:.2f} s for {n * T / d.sample_rate:.0f} s of signal")
    print("       note: the stream row concatenates independent epochs, so every window straddles two unrelated "
          "signals at different scales; it is a pessimistic bound, not what a continuous recording would give.")

    if per_snr:
        m = results["epoch mode"]
        print(f"       per SNR (epoch mode):  {'SNR dB':>7s} {'n':>3s} {'RRMSE_t':>8s} {'RRMSE_f':>8s} {'CC':>7s}")
        for level in np.unique(np.round(snr, 2)):
            sel = np.isclose(np.round(snr, 2), level)
            print(f"       {'':<22s} {level:+7.1f} {sel.sum():3d} {m['rrmse_t'][sel].mean():8.4f} "
                  f"{m['rrmse_s'][sel].mean():8.4f} {m['acc'][sel].mean():7.4f}")
    return {name: {k: float(v.mean()) for k, v in m.items()} for name, m in results.items()}


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
    p.add_argument("--hop", type=int, default=32, help="stream hop in samples (latency = hop / fs)")
    p.add_argument("--threads", type=int, default=2, help="onnxruntime intra-op threads (2 is best on a Pi 5)")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--testdata", help=".npz from deploy/make_testdata.py; prints benchmark metrics")
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

    d = EEGDenoiser(a.onnx, a.threads, spin=not a.no_spin)
    print(f"[raspi] {a.onnx}: window={d.window} fs={d.sample_rate} Hz  "
          f"providers={d.sess.get_providers()}  "
          f"intra_op={d.threads['intra_op']} ({d.threads['workers']} "
          f"worker{'s' if d.threads['workers'] != 1 else ''} + the caller)  "
          f"spinning={'on' if d.threads['allow_spinning'] else 'off'}")

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
        return LatencyMonitor(a.hop, d.sample_rate, a.latency_every, label=label,
                              environment=env) if a.latency else None

    def finish_monitor(m, stem_suffix):
        if m is None:
            return
        m.print_summary()
        if a.latency_out:
            js, csv = m.save(a.latency_out + stem_suffix)
            print(f"[{m.label}] wrote {js} and {csv}")

    if a.benchmark:
        r = benchmark(d, hop=a.hop)
        print(f"[bench] median {r['median_ms']} ms | p95 {r['p95_ms']} ms | max {r['max_ms']} ms per "
              f"{r['window']}-sample window;  hop {r['hop']} = {r['hop_budget_ms']} ms budget "
              f"-> {r['realtime_headroom_x']}x real-time headroom")

    if a.duty_cycle:
        if a.input:
            sig = np.load(a.input).reshape(-1)
        elif a.testdata:
            sig = np.load(a.testdata)["noisy"].reshape(-1)
        else:
            p.error("--duty_cycle needs --input (1-D .npy) or --testdata (.npz) as the signal")
        hops = [int(h) for h in a.duty_cycle.split(",")]
        rows = duty_cycle_check(d, sig, hops, a.duty_seconds, a.latency_every, environment=env)
        if a.latency_out and rows:
            js, csv = save_duty_rows(rows, a.latency_out + "_duty", environment=env)
            print(f"[duty] wrote {js} and {csv}")
        return                       # --input / --testdata were only the signal source

    if a.testdata:
        m = new_monitor("latency/testdata")
        evaluate_testdata(d, a.testdata, a.hop, monitor=m)
        finish_monitor(m, "_testdata" if a.input and a.stream else "")

    if a.input:
        x = np.load(a.input)
        if a.stream:
            m = new_monitor("latency/realtime" if a.realtime else "latency/stream")
            if a.realtime:
                print(f"[raspi] replaying {x.size / d.sample_rate:.1f} s of signal at {d.sample_rate} Hz ...")
                y = stream_realtime(d, x.reshape(-1), a.hop, m)
            else:
                y = d.denoise_stream(x.reshape(-1), a.hop, m)
            finish_monitor(m, "_stream" if a.testdata else "")
        else:
            y = d.denoise_epochs(x)
        print(f"[raspi] denoised {x.shape} -> {y.shape}")
        if a.output:
            np.save(a.output, y.astype(np.float32))
            print(f"[raspi] wrote {a.output}")


if __name__ == "__main__":
    main()
