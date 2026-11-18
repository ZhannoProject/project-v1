"""Run the EEG-IO comparison with native-rate replay on a Pi 5 over SSH."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4

from deploy.channel_sweep import Remote, fingerprint
from deploy.preflight import parse_cpu_list
from gui import config_io, eegio_tab, pi


def build_command(spec: dict) -> list[str]:
    from data.EEGIO import EEGIO_FS

    if spec.get("offline") or spec.get("non_causal_test"):
        raise ValueError("Offline comparison runs on this computer")
    runs = spec.get("runs")
    if not isinstance(runs, list) or not runs or any(not isinstance(r, str) for r in runs):
        raise ValueError("Choose one or more models to replay on the Pi 5")
    if len(set(runs)) != len(runs):
        raise ValueError("Choose each model at most once")
    if spec.get("zero_phase") or spec.get("target_fs", EEGIO_FS) != EEGIO_FS:
        raise ValueError("Pi 5 replay requires native 250 Hz and causal preprocessing")
    replay_mode = spec.get("replay_mode", "paced")
    if replay_mode not in ("paced", "fast"):
        raise ValueError("Pi replay mode must be paced or fast")
    hop = pi._int(spec, "hop", 4, 1, 512)
    available = {r["name"]: r for r in eegio_tab.available_runs()}
    for run in runs:
        model = available.get(run)
        if not model or model["train_fs"] != EEGIO_FS or not model["causal"]:
            raise ValueError(f"{run} must be a completed causal model trained at 250 Hz")
        if model.get("training_protocol") == "paper" and hop != 4:
            raise ValueError(f"{run} uses the paper protocol and requires hop H=4")
    dataset = eegio_tab.dataset_status()
    if not dataset["present"] or not dataset["subjects"]:
        raise ValueError(dataset.get("message") or "No EEG-IO recordings found")

    ssh = str(spec.get("ssh") or "").strip()
    remote_dir = str(spec.get("remote_dir") or "eeg").strip()
    if not ssh:
        raise ValueError("Enter the Pi 5 SSH address, for example pi@raspberrypi.local")
    # Validate with the same transport that will stage the files.
    Remote(ssh, remote_dir, "validation")
    threads = pi._int(spec, "threads", 2, 1, 4)
    cpus = str(spec.get("cpus") or "2-3").replace(" ", "")
    if not re.fullmatch(r"[0-3](?:-[0-3])?(?:,[0-3](?:-[0-3])?)*", cpus):
        raise ValueError("Choose Pi 5 CPU cores from 0 to 3, for example 2-3")
    if len(parse_cpu_list(cpus)) < threads:
        raise ValueError("Select at least one CPU core per inference thread")

    validated = {
        "runs": runs, "target_fs": EEGIO_FS, "data_dir": dataset["dir"],
        "hop": hop,
        "replay_mode": replay_mode,
        "n_subjects": pi._int(spec, "n_subjects", eegio_tab.DEFAULT_N_SUBJECTS, 1, 20),
        "include_soft": bool(spec.get("include_soft")),
        "ssh": ssh, "remote_dir": remote_dir, "threads": threads, "cpus": cpus,
    }
    return [sys.executable, "-u", "-m", "gui.eegio_pi", json.dumps(validated)]


def check_stopped():
    stop_file = os.environ.get("EEG_GUI_STOP_FILE")
    if stop_file and Path(stop_file).exists():
        raise KeyboardInterrupt


def evaluate_model(spec, data_dir, run, subjects, cfg, *, on_progress=None):
    import numpy as np

    import eegio_eval as E
    from deploy.export_onnx import export_checkpoint
    from experiment.checkpoints import read_checkpoint
    from models.variants import variant_is_causal

    check_stopped()
    runs_dir = config_io.RUNS_DIR.resolve()
    checkpoint = (runs_dir / run / "best.pt").resolve()
    if runs_dir not in checkpoint.parents or not checkpoint.is_file():
        raise ValueError(f"No checkpoint found inside runs/ for {run}")
    digest = fingerprint(checkpoint)
    protocol = read_checkpoint(checkpoint)["run_config"]
    incremental = protocol.get("protocol") == "paper"
    replay_mode = spec.get("replay_mode", "paced")
    if incremental and cfg.hop != 4:
        raise ValueError(f"{run} uses the paper protocol and requires hop H=4")

    with TemporaryDirectory(prefix="eegio-pi-") as directory:
        local = Path(directory)
        sidecar = export_checkpoint(checkpoint, local, name="model", seq_len=cfg.win,
                                    incremental=incremental,
                                    fast_replay_hop=cfg.hop if replay_mode == "fast" and not incremental else None)
        if (sidecar["preprocessing"]["sample_rate_hz"] != E.EEGIO_FS
                or not variant_is_causal(sidecar["model"]["variant"])):
            raise ValueError(f"{run} must be a causal model trained at 250 Hz")
        graph = local / ("model_step.onnx" if incremental else "model.onnx")
        tail = sidecar.get("fast_replay")
        remote = Remote(spec["ssh"], spec["remote_dir"], uuid4().hex, collection="eegio_replays")
        replay_args = ["-m", "deploy.eegio_replay", "--model", "model.onnx",
                       "--input", "recording.npy", "--out", "replay", "--hop", str(cfg.hop),
                       "--replay-mode", replay_mode,
                       "--threads", str(spec["threads"]), "--cpus", spec["cpus"],
                       "--band", *map(str, cfg.band), "--notch", str(cfg.notch_hz or 0)]
        if incremental:
            replay_args.append("--incremental")
        if tail:
            replay_args += ["--tail-model", tail["file"]]

        def replay(rec):
            if rec.fs != E.EEGIO_FS:
                raise ValueError(f"{rec.subject} is not a native 250 Hz recording")
            source = local / "recording.npy"
            np.save(source, rec.signal[cfg.channel])
            remote.upload(source, source.name)
            remote.run_case(replay_args, local, "replay", suffixes=(".npz",))
            with np.load(local / "replay.npz", allow_pickle=False) as traces:
                report = json.loads(traces["report"].item())
                if (report.get("measured_on_device") is not True
                        or "raspberry pi 5" not in report.get("board", "").lower()
                        or report.get("n_samples") != rec.n_samples
                        or report.get("sample_rate_hz") != rec.fs
                        or report.get("hop_samples") != cfg.hop
                        or report.get("paced") is not (replay_mode == "paced")
                        or report.get("backend") != ("onnx_step" if incremental else "onnx_windowed")
                        or report.get("inference_window_samples") != (tail["input_samples"] if tail else None)):
                    raise RuntimeError(f"Incomplete or incompatible Pi replay for {rec.subject}")
                return traces["filtered"], traces["output"], report

        check_stopped()
        remote.ssh(
            "grep -aEqi '^Raspberry Pi 5([[:space:]]|$)' /proc/device-tree/model "
            "/sys/firmware/devicetree/base/model 2>/dev/null || "
            "{ echo 'The SSH target must be a Raspberry Pi 5.' >&2; exit 1; }"
        )
        try:
            remote.stage(graph, runner="eegio_replay.py", model_stem="model")
            if tail:
                remote.upload(local / tail["file"], tail["file"])
            result = E.evaluate_subjects(data_dir, run, subjects, cfg,
                                         on_progress=on_progress, replay_fn=replay)
        finally:
            remote.stop(remove_workspace=True)

    if fingerprint(checkpoint) != digest:
        raise RuntimeError(f"{run} changed during replay. Run the comparison again.")
    result["checkpoint_sha256"] = digest
    result["onnx_verification"] = sidecar["verification"]
    if tail:
        result["fast_replay"] = tail
    return result


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv

    def interrupted(*_):
        raise KeyboardInterrupt

    def progress(update):
        check_stopped()
        update = {**update, "replay_mode": spec.get("replay_mode", "paced")}
        print(f"[eegio-progress] {json.dumps(update)}", flush=True)

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        spec = json.loads(args[0])

        def run_model(*args, **kwargs):
            return evaluate_model(spec, *args, **kwargs)

        out = eegio_tab.evaluate(spec, on_progress=progress, evaluate_model=run_model, target="pi5")
    except KeyboardInterrupt:
        return 130
    except subprocess.CalledProcessError as exc:
        print(f"[eegio-error] Pi replay command failed (exit {exc.returncode}). Check the job output for details.")
        return 1
    except Exception as exc:
        print(f"[eegio-error] {exc}")
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
    print(f"[eegio-result] {out['name']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
