"""Replay one native 250 Hz EEG-IO channel on a Raspberry Pi 5."""

import argparse
from itertools import chain
import json
from pathlib import Path
import time

import numpy as np

from deploy.preprocessing import CausalFilter


FAST_WINDOW_BATCH = 16


def _result(filtered, output, fs, hop, *, backend, paced, elapsed, processing_ms=None, misses=None):
    if not np.isfinite(filtered).all() or not np.isfinite(output).all():
        raise RuntimeError("Pi replay produced a non-finite signal")
    report = {
        "n_samples": len(filtered), "sample_rate_hz": fs, "hop_samples": hop,
        "backend": backend, "paced": paced, "observation_s": elapsed,
        "steady_hops": len(processing_ms) if processing_ms is not None else None,
        "processing_p50_ms": float(np.percentile(processing_ms, 50)) if processing_ms else None,
        "processing_p99_ms": float(np.percentile(processing_ms, 99)) if processing_ms else None,
        "deadline_misses": misses if paced else None,
        "hop_budget_ms": 1000. * hop / fs if paced else None,
    }
    return filtered, output, report


def _fast_windows(model, tail, signal, fs, hop, input_filter, progress):
    from deploy.raspi_infer import normalization_scale

    start = last_update = time.perf_counter()
    filtered = input_filter.process(signal)
    win, n = model.window, len(filtered)
    output = np.empty_like(filtered)
    first_end = min(win, n)
    first = filtered[:first_end]
    if first_end < win:
        first = np.pad(first, (win - first_end, 0))
    output[:first_end] = model.denoise_epochs(first)[-first_end:]

    if n > win:
        windows = np.lib.stride_tricks.sliding_window_view(filtered, win)
        ends = np.minimum(np.arange(win + hop, n + hop, hop), n)
        for i in range(0, len(ends), FAST_WINDOW_BATCH):
            group = ends[i:i + FAST_WINDOW_BATCH]
            full_windows = windows[group - win]
            # The graph needs only its causal context. Legacy scaling still
            # uses the full original window, independently for every hop.
            scale = normalization_scale(model.meta, full_windows)
            normalized = full_windows[:, -tail.window:] / scale
            predictions = tail._run(normalized[:, None, :])[:, 0, -hop:] * scale
            for end, values in zip(group, predictions):
                keep = min(hop, end - win)
                output[end - keep:end] = values[-keep:]
            now = time.perf_counter()
            if progress and group[-1] < n and now - last_update >= 2.:
                progress(int(group[-1]), n)
                last_update = now
    elapsed = time.perf_counter() - start
    if progress:
        progress(n, n)
    filtered, output, report = _result(filtered, output, fs, hop, backend="onnx_windowed",
                                       paced=False, elapsed=elapsed)
    report.update(inference_window_samples=tail.window, window_batch_size=FAST_WINDOW_BATCH)
    return filtered, output, report


def replay(model, signal, hop, band, notch_hz, *, incremental, paced=True, progress=None,
           tail_model=None):
    """Replay the same sample sequence with optional acquisition timing."""
    from deploy.raspi_infer import normalization_scale

    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1 or not len(signal) or not np.isfinite(signal).all():
        raise ValueError("EEG-IO replay requires a finite single-channel recording")
    fs = float(model.sample_rate)
    if fs != 250:
        raise ValueError("EEG-IO Pi replay requires a model trained at 250 Hz")
    if hop < 1 or (incremental and hop != 4) or (not incremental and hop > model.window):
        raise ValueError("Use H=4 for stateful replay or a hop within the model window")
    protocol = model.meta.get("data_protocol") or {}
    delay = int(protocol.get("output_delay", 0)) if incremental else 0
    score_start = max(3 * int(fs), (protocol.get("valid_history_samples") or 1) - 1)
    input_filter = CausalFilter(fs, band=band, notch_hz=notch_hz)
    if tail_model is not None:
        if paced or incremental or not hop <= tail_model.window < model.window:
            raise ValueError("The short graph requires fast windowed replay")
        return _fast_windows(model, tail_model, signal, fs, hop, input_filter, progress)

    filtered = np.empty_like(signal)
    output = np.empty_like(signal)
    processing_ms = []
    misses = 0
    if incremental:
        scale = normalization_scale(model.meta, online=True)
        state = model.initial_state()
    first_end = min(len(signal), hop if incremental else model.window)
    ends = chain(range(first_end, len(signal), hop), (len(signal),))
    clock_start = time.perf_counter()
    last_update = clock_start
    start = 0
    for end in ends:
        if paced:
            available = clock_start + end / fs
            time.sleep(max(0., available - time.perf_counter()))
        begin = time.perf_counter()
        filtered[start:end] = input_filter.process(signal[start:end])
        if incremental:
            for i in range(start, end):
                y, state = model.step(float(filtered[i]) / scale, state)
                output[i] = y * scale
        else:
            window = filtered[max(0, end - model.window):end]
            if len(window) < model.window:
                window = np.pad(window, (model.window - len(window), 0))
            # Match the historical scorer, which retains up to a full hop
            # from the last window even when it overlaps the preceding hop.
            keep = end if start == 0 else min(hop, end - model.window)
            output[end - keep:end] = model.denoise_epochs(window)[-keep:]
        emitted = time.perf_counter()
        if start - delay >= score_start and end - start == hop:
            processing_ms.append((emitted - begin) * 1000.)
            if paced:
                misses += emitted > available + hop / fs
        if progress and (end == len(signal) or emitted - last_update >= 2.):
            progress(end, len(signal))
            last_update = emitted
        start = end
    return _result(filtered, output, fs, hop,
                   backend="onnx_step" if incremental else "onnx_windowed",
                   paced=paced, elapsed=emitted - clock_start,
                   processing_ms=processing_ms, misses=misses)


def main(argv=None):
    from deploy import preflight
    from deploy.raspi_infer import EEGDenoiser, IncrementalDenoiser

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--hop", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--cpus", required=True)
    parser.add_argument("--band", nargs=2, type=float, required=True)
    parser.add_argument("--notch", type=float, required=True)
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--replay-mode", choices=("paced", "fast"), default="paced",
                        help="Use acquisition timing (paced) or process successive hops immediately (fast)")
    parser.add_argument("--tail-model", help="Verified short ONNX graph for fast windowed replay")
    args = parser.parse_args(argv)

    preflight.pin(args.cpus)
    denoiser = IncrementalDenoiser if args.incremental else EEGDenoiser
    model = denoiser(args.model, args.threads, spin=False)
    tail = EEGDenoiser(args.tail_model, args.threads, spin=False) if args.tail_model else None
    before = preflight.check(args.threads, model.threads)
    if "raspberry pi 5" not in before["board"].lower():
        raise RuntimeError(f"Expected a Raspberry Pi 5, found {before['board']}")

    def progress(done, total):
        print(f"[eegio-pi-samples] {done} {total}", flush=True)

    x, y, report = replay(model, np.load(args.input, allow_pickle=False), args.hop,
                          tuple(args.band), args.notch or None,
                          incremental=args.incremental, paced=args.replay_mode == "paced",
                          progress=progress, tail_model=tail)
    after = preflight.check(args.threads, model.threads)
    report.update(measured_on_device=True, board=before["board"],
                  includes_causal_filters=True, environment_before=before, environment_after=after)
    out = Path(args.out)
    record = json.dumps(report)
    np.savez(out.with_suffix(".npz"), filtered=x, output=y, report=record)
    out.with_suffix(".json").write_text(record, encoding="utf-8")


if __name__ == "__main__":
    main()
