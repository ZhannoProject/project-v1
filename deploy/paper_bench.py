"""Measured Pi 5 hop replay. No host estimates are accepted as device results.

python -m deploy.paper_bench --model MODEL_step.onnx --input recording.npy \
  --out measurements/run --cpus 2-3 --threads 2

First exercise an exported random-weight model; repeat with final weights.
External meter CSVs have time_s,power_mw columns, captured during this replay.
"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

from deploy.memory import peak_ram_mib


def power_summary(active_path, idle_path, duration_s):
    def read(path):
        with open(path, newline="") as stream:
            rows = list(csv.DictReader(stream))
        t = np.asarray([float(r["time_s"]) for r in rows])
        p = np.asarray([float(r["power_mw"]) for r in rows])
        if len(t) < 2 or not np.isfinite(t).all() or not np.isfinite(p).all() or (np.diff(t) <= 0).any() or (p < 0).any():
            raise ValueError("External meter trace needs finite, increasing timestamps and nonnegative mW")
        if t[-1] - t[0] < duration_s:
            raise ValueError("External meter trace is shorter than the measurement")
        # Use the requested observation duration, not a longer surrounding idle interval.
        end = t[0] + duration_s
        keep = t < end
        ts, ps = np.r_[t[keep], end], np.r_[p[keep], np.interp(end, t, p)]
        integrate = getattr(np, "trapezoid", None) or np.trapz
        return float(integrate(ps, ts) / duration_s)
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("Power observation duration must be positive")
    active, idle = read(active_path), read(idle_path)
    return dict(source="external_meter", active_trace=str(active_path), idle_trace=str(idle_path),
                observation_s=duration_s, total_power_mw=active, idle_power_mw=idle,
                incremental_power_mw=active-idle, total_energy_mj=active*duration_s,
                incremental_energy_mj=(active-idle)*duration_s)


def replay(model, signal, hop=4, channels=1, raw_input=True, sigma=None):
    """Paced sample acquisition; every channel owns its own persistent state."""
    from deploy.raspi_infer import normalization_scale
    if isinstance(hop, bool) or not isinstance(hop, (int, np.integer)) or hop < 1 or channels < 1:
        raise ValueError("Replay requires a positive integer hop and at least one channel")
    fs = float(model.sample_rate)
    cfg = model.meta.get("data_protocol") or {}
    delay = int(cfg.get("output_delay", 0))
    sigma = normalization_scale(model.meta, sigma=sigma, online=True)
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1 or not np.isfinite(signal).all() or len(signal) < hop or len(signal) % hop:
        raise ValueError("Input must be a finite 1-D recording with a whole number of hops")
    states = [model.initial_state() for _ in range(channels)]
    from deploy.preprocessing import CausalFilter
    filters = [CausalFilter(fs) if raw_input else None for _ in range(channels)]
    outputs = np.empty((channels, len(signal)), dtype=np.float32)
    rows = []
    start = time.perf_counter()
    for end in range(hop, len(signal) + 1, hop):
        available = start + end / fs
        time.sleep(max(0., available - time.perf_counter()))
        t0 = time.perf_counter()
        infer_s = 0.
        for channel in range(channels):
            chunk = signal[end-hop:end]
            if filters[channel] is not None:
                chunk = filters[channel].process(chunk)
            for index, sample in zip(range(end-hop, end), chunk):
                value, states[channel] = model.step(float(sample) / sigma, states[channel])
                outputs[channel, index] = value * sigma
                infer_s += model.last_run_s
        emitted = time.perf_counter()
        first_target = end-hop-delay
        last_target = end-1-delay
        rows.append(dict(acquisition_end_sample=end-1, target_first_sample=first_target,
                         target_last_sample=last_target,
                         infer_ms=infer_s*1000, processing_ms=(emitted-t0)*1000,
                         emitted_s=emitted-start, available_s=available-start,
                         deadline_miss=emitted > available + hop/fs,
                         # Acquisition at (i+1)/fs, with explicit output alignment.
                         target_first_e2e_ms=(emitted-start-(first_target+1)/fs)*1000,
                         target_last_e2e_ms=(emitted-start-(last_target+1)/fs)*1000))
    return outputs, rows, time.perf_counter()-start


def main(argv=None):
    from deploy import preflight
    from deploy.raspi_infer import IncrementalDenoiser, normalization_scale
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--input_fs", type=float, required=True)
    p.add_argument("--input_stage", choices=["raw", "preprocessed"], default="raw")
    p.add_argument("--out", required=True)
    p.add_argument("--cpus", default="")
    p.add_argument("--target", choices=["host", "pi"], default="pi")
    p.add_argument("--seconds", type=float)
    p.add_argument("--hop", type=int, default=4, help="Samples per replay hop")
    p.add_argument("--sigma", type=float, help="Prior fixed calibration for legacy models")
    p.add_argument("--exploratory", action="store_true", help="Allow exploratory settings such as legacy calibration or a different hop")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--channels", type=int, choices=[1, 2, 8, 16], default=1)
    p.add_argument("--weights", choices=["random", "final"], required=True)
    p.add_argument("--power_csv")
    p.add_argument("--idle_power_csv")
    p.add_argument("--meter", help="External meter model, location and supply arrangement")
    a = p.parse_args(argv)
    if a.hop < 1:
        p.error("Hop must be a positive integer")
    if a.hop != 4 and not a.exploratory:
        p.error("Use --exploratory for hop sizes other than 4")
    out = Path(a.out)
    if out.with_suffix(".json").exists():
        raise FileExistsError("Measurement already exists; choose a fresh output stem")
    if a.cpus:
        preflight.pin(a.cpus)
    model = IncrementalDenoiser(a.model, a.threads, spin=False)
    before = preflight.check(a.threads, model.threads)
    if a.target == "pi" and (before["verdict"] != preflight.LOCKED or "raspberry pi 5" not in before["board"].lower()):
        raise RuntimeError("Paper measurements require a locked Raspberry Pi 5; host estimates are excluded")
    cfg = model.meta.get("data_protocol") or {}
    if cfg.get("protocol") != "paper" and not a.exploratory:
        raise ValueError("Use a paper-protocol export with fixed calibration and delay metadata")
    if a.input_fs != model.sample_rate:
        raise ValueError("Device replay requires the native model rate; never use offline resampling")
    source = np.load(a.input, allow_pickle=False, mmap_mode="r")
    if source.ndim != 1:
        raise ValueError("Channel replication requires one continuous 1-D input recording")
    if a.seconds is not None:
        if not np.isfinite(a.seconds) or a.seconds <= 0:
            raise ValueError("Replay duration must be finite and positive")
        n = int(a.seconds * a.input_fs) // a.hop * a.hop
        if n > len(source) or n < a.hop:
            raise ValueError("Input recording is shorter than the requested replay")
        source = source[:n]
    scale = normalization_scale(model.meta, sigma=a.sigma, online=True)
    output, rows, duration = replay(model, source, hop=a.hop, channels=a.channels,
                                    raw_input=a.input_stage == "raw", sigma=scale)
    after = preflight.check(a.threads, model.threads)
    warmup = cfg.get("valid_history_samples") or model.window or 1
    steady = [r for r in rows if r["target_first_sample"] >= warmup-1]
    processing = np.asarray([r["processing_ms"] for r in steady])
    e2e = np.asarray([r["target_first_e2e_ms"] for r in steady])
    if not len(e2e):
        raise ValueError("Recording has no complete steady-state targets")
    power = None
    if a.power_csv or a.idle_power_csv:
        if not (a.power_csv and a.idle_power_csv and a.meter):
            raise ValueError("Power requires active+idle external traces and meter/supply provenance")
        power = {**power_summary(a.power_csv, a.idle_power_csv, duration), "meter_setup": a.meter}
    hop_budget_ms = 1000 * a.hop / model.sample_rate
    report = dict(measured_on_device=a.target == "pi", where=a.target, weights=a.weights, board=before["board"],
                  cpu_scope="Cortex-A76" if a.target == "pi" else "host diagnostic",
                  protocol=cfg.get("protocol", "legacy"), exploratory=a.exploratory or a.target == "host",
                  channels=a.channels, channel_mode="replicated single-channel",
                  sample_rate_hz=model.sample_rate, hop_samples=a.hop, observation_s=duration,
                  normalization_scale=scale,
                  processing_p50_ms=float(np.percentile(processing, 50)),
                  processing_p99_ms=float(np.percentile(processing, 99)),
                  end_to_end_p50_ms=float(np.percentile(e2e, 50)),
                  end_to_end_p99_ms=float(np.percentile(e2e, 99)),
                  deadline_misses=sum(r["deadline_miss"] for r in steady),
                  deadline_miss_rate=sum(r["deadline_miss"] for r in steady)/len(steady),
                  hop_budget_ms=hop_budget_ms,
                  replica_max_abs_delta=float(np.max(np.abs(output-output[:1]))),
                  steady_hops=len(steady), warmup_hops=len(rows)-len(steady),
                  first_hop_processing_ms=rows[0]["processing_ms"],
                  duty_cycle_percent=float(processing.mean() / hop_budget_ms * 100),
                  process_peak_ram_mib=peak_ram_mib(),
                  environment_before=before, environment_after=after, power=power,
                  measurement_record_complete=a.target == "pi" and not a.exploratory and after["verdict"] == preflight.LOCKED and power is not None,
                  input_stage=a.input_stage, includes_causal_filters=a.input_stage == "raw",
                  model=str(a.model), input=str(a.input), data_protocol=cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    np.save(out.with_suffix(".npy"), output)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
