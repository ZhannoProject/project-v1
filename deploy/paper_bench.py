"""Measured Pi 5 hop replay. No host estimates are accepted as device results.

python -m deploy.paper_bench --model MODEL_step.onnx --input recording.npy \
  --out measurements/run --cpus 2-3 --threads 2

First exercise an exported random-weight model; repeat with final weights.
External meter CSVs have time_s,power_mw columns, captured during this replay.
"""

import argparse
from contextlib import ExitStack
import signal as signals
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


def replay(model, signal, hop=4, channels=1, raw_input=True, sigma=None, reset_samples=(), backend=None):
    """Paced sample acquisition; every channel owns its own persistent state."""
    from deploy.raspi_infer import normalization_scale
    if isinstance(hop, bool) or not isinstance(hop, (int, np.integer)) or hop < 1 or channels < 1:
        raise ValueError("Replay requires a positive integer hop and at least one channel")
    fs = float(model.sample_rate)
    cfg = model.meta.get("data_protocol") or {}
    delay = int(getattr(model, "output_delay", cfg.get("output_delay", 0)))
    sigma = normalization_scale(model.meta, sigma=sigma, online=True)
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 1 or not np.isfinite(signal).all() or len(signal) < hop or len(signal) % hop:
        raise ValueError("Input must be a finite 1-D recording with a whole number of hops")
    reset_samples = list(reset_samples)
    if (any(type(p) is not int or not 0 < p < len(signal) for p in reset_samples)
            or reset_samples != sorted(set(reset_samples))):
        raise ValueError("Subject reset samples must be unique, increasing indices inside the recording")
    from deploy.parallel_replay import ChannelProcessor
    processor = None if backend else ChannelProcessor(model, channels, sigma, raw_input, reset_samples)
    outputs = backend.output if backend else np.empty((channels, len(signal)), dtype=np.float32)
    rows = []
    start = time.perf_counter()
    segment_start, last_progress = 0, start
    for end in range(hop, len(signal) + 1, hop):
        available = start + end / fs
        time.sleep(max(0., available - time.perf_counter()))
        t0 = time.perf_counter()
        first = end-hop
        if backend:
            worker_rows = backend.hop(first, end)
            infer_s = sum(r['infer_s'] for r in worker_rows)
            segment_start = worker_rows[0]['segment_start']
        else:
            outputs[:, first:end], infer_s, segment_start = processor.hop(signal, first, end)
        emitted = time.perf_counter()
        first_target = end-hop-delay
        last_target = end-1-delay
        rows.append(dict(acquisition_end_sample=end-1, target_first_sample=first_target,
                         target_last_sample=last_target,
                         segment_start_sample=segment_start,
                         valid_output=first_target >= segment_start,
                         infer_ms=infer_s*1000, processing_ms=(emitted-t0)*1000,
                         emitted_s=emitted-start, available_s=available-start,
                         deadline_miss=emitted > available + hop/fs,
                         # Acquisition at (i+1)/fs, with explicit output alignment.
                         target_first_e2e_ms=(emitted-start-(first_target+1)/fs)*1000,
                         target_last_e2e_ms=(emitted-start-(last_target+1)/fs)*1000))
        if emitted - last_progress >= 5. or end == len(signal):
            print(f"[replay-progress] {channels} channels · {end/fs:.1f}/{len(signal)/fs:.1f} s "
                  f"· elapsed {emitted-start:.1f} s", flush=True)
            last_progress = emitted
    return outputs, rows, time.perf_counter()-start


def require_device(before, target, out):
    from deploy import preflight
    if target == "pi" and (before["verdict"] != preflight.LOCKED or "raspberry pi 5" not in before["board"].lower()):
        out.parent.mkdir(parents=True, exist_ok=True)
        report_path = out.with_name(out.name + '_preflight.json')
        report_path.write_text(json.dumps(before, indent=2), encoding='utf-8')
        print(f"[preflight] {before['board']} · {before['verdict']}", flush=True)
        reasons = []
        for check in before.get('checks', []):
            print(f"[preflight] {check['status']} · {check['name']}: {check['detail']}", flush=True)
            if check['status'] in (preflight.FAIL, preflight.WARN):
                reasons.append(f"{check['name']}: {check['detail']}")
        if 'raspberry pi 5' not in before['board'].lower():
            reasons.append(f"expected Raspberry Pi 5, found {before['board']}")
        detail = '; '.join(reasons or before.get('blockers') or [before['verdict']])
        raise RuntimeError(f"Device state check failed before replay: {detail}. "
                           f"Full state saved to {report_path}")


def _main(argv, resources):
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
    p.add_argument("--reset_samples", default="", help="Comma-separated subject boundaries; reset states without restarting the replay clock")
    p.add_argument("--sigma", type=float, help="Prior fixed calibration for legacy models")
    p.add_argument("--exploratory", action="store_true", help="Allow exploratory settings such as legacy calibration or a different hop")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--workers", type=int, choices=[1, 2], default=1, help="Independent channel processes; 2 uses one ORT thread per core")
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
    if a.workers == 2 and a.threads != 2:
        p.error("Two channel workers require a total CPU budget of --threads 2")
    count = min(a.workers, a.channels)
    cpus = preflight.parse_cpu_list(a.cpus) if a.cpus else []
    if a.workers == 2 and cpus and len(cpus) != 2:
        p.error("Two channel workers require exactly two selected CPU cores")
    if cpus:
        preflight.pin(preflight.format_cpu_list(cpus[:count] if a.workers == 2 else cpus))
    model = IncrementalDenoiser(a.model, 1 if a.workers == 2 else a.threads, spin=False)
    if a.workers == 1:
        before = preflight.check(a.threads, model.threads)
        require_device(before, a.target, out)
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
        n = int(round(a.seconds * a.input_fs)) // a.hop * a.hop
        if n > len(source) or n < a.hop:
            raise ValueError("Input recording is shorter than the requested replay")
        source = source[:n]
    scale = normalization_scale(model.meta, sigma=a.sigma, online=True)
    resets = [int(p) for p in a.reset_samples.split(',') if p]
    backend = None
    if a.workers == 2:
        from deploy.parallel_replay import ParallelChannels
        backend = resources.enter_context(ParallelChannels(a.model, a.input, len(source), a.channels,
                    count, cpus, scale, a.input_stage == 'raw', resets))
    runtime = backend.runtime if backend else model.threads
    cpu_budget = count if backend else a.threads
    if backend:
        before = preflight.check(cpu_budget, runtime)
    require_device(before, a.target, out)
    print(f"[execution] {count if backend else 1} process(es) · "
          f"{1 if backend else a.threads} ORT thread(s) per process · CPU {a.cpus or 'host scheduler'}", flush=True)
    output, rows, duration = replay(model, source, hop=a.hop, channels=a.channels,
                                    raw_input=a.input_stage == "raw", sigma=scale,
                                    reset_samples=resets, backend=backend)
    if backend:
        backend.finish()
    after = preflight.check(cpu_budget, runtime, phase='after_replay')
    warmup = cfg.get("valid_history_samples") or model.window or 1
    steady = [r for r in rows if r["target_first_sample"] >= r['segment_start_sample']+warmup-1]
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
                  protocol=cfg.get("protocol", "legacy"), exploratory=a.exploratory or a.target == "host" or bool(getattr(model,"experimental_bounded",False)),
                  channels=a.channels, channel_mode="replicated single-channel",
                  subject_reset_samples=resets, state_reset='each_subject_boundary' if resets else 'case_start',
                  replay_duration_s=len(source)/a.input_fs,
                  sample_rate_hz=model.sample_rate, hop_samples=a.hop, observation_s=duration,
                  normalization_scale=scale,
                  lookahead_samples=int(getattr(model,'lookahead',0)),
                  output_delay_samples=int(getattr(model,'output_delay',cfg.get('output_delay',0))),
                  output_alignment='acquisition',
                  inference_semantics=(model.meta.get('provenance') or {}).get('inference_semantics','stateful'),
                  experimental_bounded=bool(getattr(model,'experimental_bounded',False)),
                  valid_target_intervals=[[lo, max(lo, hi-int(getattr(model,'output_delay',cfg.get('output_delay',0))))]
                                          for lo,hi in zip([0,*resets],[*resets,len(source)])],
                  timing_scope='software replay; target timing includes output delay and hop buffering; excludes acquisition driver and transport',

                  execution_mode='channel_processes' if backend else 'serial_channels',
                  requested_workers=a.workers, effective_workers=count if backend else 1,
                  runtime=runtime, timing_includes_worker_ipc=bool(backend),
                  infer_ms_scope='sum of ORT call durations across all channels',
                  worker_measurements=[dict(**w, cpu_percent=100*w['cpu_time_s']/duration)
                                       for w in backend.finished] if backend else [],
                  worker_ram_scope='per-process peak RSS; shared pages may be counted in multiple processes',

                  processing_p50_ms=float(np.percentile(processing, 50)),
                  processing_p99_ms=float(np.percentile(processing, 99)),
                  end_to_end_p50_ms=float(np.percentile(e2e, 50)),
                  end_to_end_p99_ms=float(np.percentile(e2e, 99)),
                  deadline_misses=sum(r["deadline_miss"] for r in steady),
                  deadline_miss_rate=sum(r["deadline_miss"] for r in steady)/len(steady),
                  hop_budget_ms=hop_budget_ms,
                  total_hops=len(rows),
                  all_hop_deadline_misses=sum(r['deadline_miss'] for r in rows),
                  all_hop_deadline_miss_rate=sum(r['deadline_miss'] for r in rows)/len(rows),
                  processing_max_ms=max(r['processing_ms'] for r in rows),
                  max_hop_response_ms=max((r['emitted_s']-r['available_s'])*1000 for r in rows),
                  replay_overrun_s=duration-len(source)/a.input_fs,
                  realtime_no_misses=not any(r['deadline_miss'] for r in rows),
                  replica_max_abs_delta=float(np.max(np.abs(output-output[:1]))),
                  steady_hops=len(steady), warmup_hops=len(rows)-len(steady),
                  first_hop_processing_ms=rows[0]["processing_ms"],
                  duty_cycle_percent=float(processing.mean() / hop_budget_ms * 100),
                  process_peak_ram_mib=peak_ram_mib(),
                  environment_before=before, environment_after=after, power=power,
                  measurement_record_complete=a.target == "pi" and not a.exploratory and not getattr(model,"experimental_bounded",False) and after["verdict"] == preflight.LOCKED and power is not None,
                  input_stage=a.input_stage, includes_causal_filters=a.input_stage == "raw",
                  model=str(a.model), input=str(a.input), data_protocol=cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    np.save(out.with_suffix(".npy"), output)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main(argv=None):
    def interrupted(*_):
        raise KeyboardInterrupt
    previous = {sig: signals.signal(sig, interrupted) for sig in (signals.SIGTERM, signals.SIGINT)}
    try:
        with ExitStack() as resources:
            return _main(argv, resources)
    finally:
        for sig, handler in previous.items():
            signals.signal(sig, handler)


if __name__ == "__main__":
    main()
