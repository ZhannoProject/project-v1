"""
Run an exported EEG denoiser on a Raspberry Pi 5 (or any CPU) with only
numpy + onnxruntime.  Copy this file, the .onnx and its .json sidecar to the
Pi; nothing else from the repo is needed.

    python raspi_infer.py tcn_mla_eog.onnx --benchmark
    python raspi_infer.py tcn_mla_eog.onnx --testdata EOG_test100.npz
    python raspi_infer.py tcn_mla_eog.onnx --input noisy.npy --output clean.npy
    python raspi_infer.py tcn_mla_eog.onnx --input noisy.npy --output clean.npy --stream --hop 32

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

--testdata          scores a .npz cut by deploy/make_testdata.py (noisy, clean,
                    snr_db) with the EEGdenoiseNet metrics, in both epoch and
                    streaming mode, next to the untouched noisy input as a
                    baseline.  Running the same command on the host and on the
                    Pi should give the same numbers.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

EPS = 1e-8


class EEGDenoiser:
    def __init__(self, onnx_path: str | Path, n_threads: int = 2):
        onnx_path = Path(onnx_path)
        opts = ort.SessionOptions()
        # Measured on a Pi 5 (tcn_mla_eog, 512-sample window): 1 thread 24 ms,
        # 2 threads 18 ms with a tight tail (p95 19 ms), 4 threads 20 ms median
        # but p95 30 ms / max 40 ms from contending with the OS for every core.
        opts.intra_op_num_threads = n_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

        inp, out = self.sess.get_inputs()[0], self.sess.get_outputs()[0]
        self.input_name, self.output_name = inp.name, out.name
        self.window = int(inp.shape[-1])
        self.dynamic_batch = not isinstance(inp.shape[0], int)

        sidecar = onnx_path.with_suffix(".json")
        self.meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        self.sample_rate = self.meta.get("preprocessing", {}).get("sample_rate_hz", 256)

    # ---- core call: (N, 1, T) normalised -> (N, 1, T) normalised ----
    def _run(self, x: np.ndarray) -> np.ndarray:
        if not self.dynamic_batch and x.shape[0] != 1:
            return np.concatenate([self._run(row[None]) for row in x])
        return self.sess.run([self.output_name], {self.input_name: x})[0]

    def denoise_epochs(self, x: np.ndarray) -> np.ndarray:
        """(T,) or (N, T) raw-scale windows -> same shape, denoised, raw scale."""
        x = np.asarray(x, dtype=np.float32)
        squeeze = x.ndim == 1
        x = x.reshape(-1, self.window)
        sigma = np.maximum(x.std(axis=1, keepdims=True), EPS)
        y = self._run((x / sigma)[:, None, :])[:, 0, :] * sigma
        return y[0] if squeeze else y

    def denoise_stream(self, signal: np.ndarray, hop: int = 32) -> np.ndarray:
        """Causally denoise a long 1-D signal; output is sample-aligned with the input."""
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
        out[:T] = self.denoise_epochs(signal[:T])
        for end in range(T + hop, len(signal) + hop, hop):
            end = min(end, len(signal))
            out[end - hop:end] = self.denoise_epochs(signal[end - T:end])[-hop:]
        return out


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

    def __init__(self, denoiser: EEGDenoiser, hop: int = 32):
        self.d, self.hop = denoiser, hop
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
            out.append(self.d.denoise_epochs(self.buf))

        while len(self.pending) >= self.hop:              # steady state
            self.buf = np.roll(self.buf, -self.hop)
            self.buf[-self.hop:], self.pending = self.pending[:self.hop], self.pending[self.hop:]
            out.append(self.d.denoise_epochs(self.buf)[-self.hop:])
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


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


def evaluate_testdata(d: EEGDenoiser, npz_path: str | Path, hop: int = 32, per_snr: bool = True) -> dict:
    data = np.load(npz_path)
    noisy, clean, snr = data["noisy"], data["clean"], data["snr_db"]
    n, T = noisy.shape
    if T != d.window:
        raise ValueError(f"test data has {T}-sample epochs but the model window is {d.window}")

    t0 = time.perf_counter()
    y_epoch = d.denoise_epochs(noisy)
    t_epoch = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_stream = d.denoise_stream(noisy.reshape(-1), hop).reshape(n, T)
    t_stream = time.perf_counter() - t0

    modes = {"noisy input (no model)": noisy, "epoch mode": y_epoch, f"stream mode (hop {hop})": y_stream}
    results = {name: metrics(y, clean) for name, y in modes.items()}

    print(f"[test] {Path(npz_path).name}: {n} epochs x {T} samples, SNR {snr.min():+.1f}..{snr.max():+.1f} dB")
    print(f"       {'':<26s} {'RRMSE_t':>8s} {'RRMSE_s':>8s} {'ACC':>7s}")
    for name, m in results.items():
        print(f"       {name:<26s} {m['rrmse_t'].mean():8.4f} {m['rrmse_s'].mean():8.4f} {m['acc'].mean():7.4f}")
    print(f"       epoch mode: {1e3 * t_epoch / n:.2f} ms/epoch (batched)   "
          f"stream mode: {1e3 * t_stream / (n * T / hop):.2f} ms/hop, {t_stream:.2f} s for {n * T / d.sample_rate:.0f} s of signal")
    print("       note: the stream row concatenates independent epochs, so every window straddles two unrelated "
          "signals at different scales; it is a pessimistic bound, not what a continuous recording would give.")

    if per_snr:
        m = results["epoch mode"]
        print(f"       per SNR (epoch mode):  {'SNR dB':>7s} {'n':>3s} {'RRMSE_t':>8s} {'RRMSE_s':>8s} {'ACC':>7s}")
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
    a = p.parse_args(argv)

    d = EEGDenoiser(a.onnx, a.threads)
    print(f"[raspi] {a.onnx}: window={d.window} fs={d.sample_rate} Hz  "
          f"providers={d.sess.get_providers()}  threads={a.threads}")

    if a.benchmark:
        r = benchmark(d, hop=a.hop)
        print(f"[bench] median {r['median_ms']} ms | p95 {r['p95_ms']} ms | max {r['max_ms']} ms per "
              f"{r['window']}-sample window;  hop {r['hop']} = {r['hop_budget_ms']} ms budget "
              f"-> {r['realtime_headroom_x']}x real-time headroom")

    if a.testdata:
        evaluate_testdata(d, a.testdata, a.hop)

    if a.input:
        x = np.load(a.input)
        y = d.denoise_stream(x.reshape(-1), a.hop) if a.stream else d.denoise_epochs(x)
        print(f"[raspi] denoised {x.shape} -> {y.shape}")
        if a.output:
            np.save(a.output, y.astype(np.float32))
            print(f"[raspi] wrote {a.output}")


if __name__ == "__main__":
    main()
