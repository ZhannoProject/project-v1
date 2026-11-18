"""Run identical-weight EEG channel-load cases locally or over SSH."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from uuid import uuid4


ALLOWED_CHANNELS = (1, 2, 8, 16)
SSH_RE = re.compile(r"^[A-Za-z0-9_.\[][A-Za-z0-9_.:%\[\]@-]*$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def run_cancellable(command, check=True, **kwargs):
    """Observe the GUI stop request even on Windows, which has no SIGTERM handler."""
    stop_file = os.environ.get('EEG_GUI_STOP_FILE')
    if os.name == 'nt':
        kwargs.setdefault('creationflags', subprocess.CREATE_NO_WINDOW)
    if not stop_file:
        return subprocess.run(command, check=check, **kwargs)
    if Path(stop_file).exists():
        raise KeyboardInterrupt
    if kwargs.pop('capture_output', False):
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    process = subprocess.Popen(command, **kwargs)
    try:
        while True:
            try:
                stdout, stderr = process.communicate(timeout=.1)
                break
            except subprocess.TimeoutExpired:
                if Path(stop_file).exists():
                    raise KeyboardInterrupt
        if check and process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command, output=stdout, stderr=stderr)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def channel_counts(value):
    values = value if isinstance(value, list) else str(value).replace(" ", "").split(",")
    if not values or any(str(v) not in {str(n) for n in ALLOWED_CHANNELS} for v in values):
        raise ValueError("Choose EEG channel counts from 1, 2, 8, 16")
    if len(set(map(int, values))) != len(values):
        raise ValueError("Channel counts must not repeat")
    return sorted(map(int, values))


def case_order(counts, repeats):
    """Rotate the starting load to reduce a systematic warm-board ordering."""
    return [(c, r+1) for r in range(repeats)
            for c in counts[r % len(counts):] + counts[:r % len(counts)]]


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def eegio_input(data_dir, subject, electrode, fs):
    """Read one continuous, unflagged EEG-IO segment at the replay rate."""
    import numpy as np
    from data.EEGIO import list_subjects, read_recording

    subjects = list_subjects(data_dir)
    subject = subject or subjects[0]
    if subject not in subjects:
        raise ValueError("Choose an available EEG-IO subject")
    recording = read_recording(data_dir, subject)
    if electrode not in recording.channels:
        raise ValueError("Choose an EEG-IO electrode (Fp1 or Fp2)")
    valid = ~recording.corrupt_mask()
    starts, stops = np.flatnonzero(np.diff(np.r_[False, valid, False])).reshape(-1, 2).T
    if not len(starts):
        raise ValueError(f"EEG-IO {subject} has no unflagged recording segment")
    longest = int(np.argmax(stops - starts))
    start, stop = int(starts[longest]), int(stops[longest])
    values = recording.signal[recording.channels.index(electrode), start:stop]
    resampling = None
    if fs != recording.fs:
        from data.resampling import causal_resample, resampling_info
        values = causal_resample(values, recording.fs, fs)
        resampling = resampling_info(recording.fs, fs)
    return values, dict(
        subject=subject, electrode=electrode, native_fs=recording.fs,
        source=str(Path(data_dir).resolve() / f"{subject}_data.csv"),
        start_sample=start, stop_sample=stop, resampling=resampling,
    )


class Remote:
    def __init__(self, target, directory, run_id, *, collection="channel_sweeps"):
        if not SSH_RE.fullmatch(target) or target.count("@") > 1 or not NAME_RE.fullmatch(directory):
            raise ValueError("Invalid Pi address or remote directory")
        if not NAME_RE.fullmatch(run_id) or not NAME_RE.fullmatch(collection):
            raise ValueError("Invalid Pi workspace name")
        self.target = target
        user, sep, host = target.rpartition("@")
        host = host if sep else target
        self.scp_target = ((user+"@") if sep else "") + (f"[{host}]" if ":" in host and not host.startswith("[") else host)
        self.relative = f"{directory}/{collection}/{run_id}"
        self.root = '"$HOME"/' + shlex.quote(self.relative)
        self.python = '"$HOME"/' + shlex.quote(f"{directory}/env/bin/python")
        self.options = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                        "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3"]

    def ssh(self, command, **kwargs):
        return run_cancellable(["ssh", *self.options, self.target, command], **kwargs)

    def upload(self, source, name):
        if not NAME_RE.fullmatch(name):
            raise ValueError("Invalid Pi workspace file name")
        run_cancellable(["scp", "-q", *self.options, str(source),
                        f"{self.scp_target}:{self.relative}/{name}"], check=True)

    def stage(self, model, recording=None, *, runner="paper_bench.py", model_stem="model_step"):
        deploy = Path(__file__).resolve().parent
        self.ssh(f"mkdir -p -- {self.root}/deploy")
        files = [deploy/n for n in ("__init__.py", runner, "preflight.py",
                                   "preprocessing.py", "memory.py", "raspi_infer.py", "requirements-raspi.txt")]
        run_cancellable(["scp", "-q", *self.options, *map(str,files),
                        f"{self.scp_target}:{self.relative}/deploy/"], check=True)
        self.upload(model, f"{model_stem}.onnx")
        self.upload(model.with_suffix(".json"), f"{model_stem}.json")
        if recording is not None:
            self.upload(recording, "recording.npy")
        env = self.python.rsplit("/bin/python",1)[0]
        self.ssh(f"(test -x {self.python} || python3 -m venv {env}) && "
                 f"{self.python} -m pip install --disable-pip-version-check -q -r {self.root}/deploy/requirements-raspi.txt")

    def run_case(self, args, local, stem, *, suffixes=(".json", ".csv")):
        # The PID is scoped to this unique experiment directory. exec keeps it
        # the benchmark's PID, so Stop can address exactly this remote process.
        launch = "import os,sys;open('current.pid','w').write(str(os.getpid()));os.execv(sys.executable,[sys.executable,*sys.argv[1:]])"
        command = (f"cd {self.root} && {self.python} -u -c {shlex.quote(launch)} "
                   + shlex.join(args))
        self.ssh(command)
        for suffix in suffixes:
            run_cancellable(["scp", "-q", *self.options,
                            f"{self.scp_target}:{self.relative}/{stem}{suffix}", str(local)], check=True)

    def stop(self, *, remove_workspace=False):
        # Verify the working directory before using the PID, including after a
        # completed case whose PID file remains. Never kill an unrelated PID.
        command = (f"cd {self.root} 2>/dev/null || exit 0; "
                   "if test -f current.pid; then p=$(cat current.pid); "
                   "if test \"$(readlink /proc/$p/cwd 2>/dev/null)\" = \"$PWD\"; then kill -TERM \"$p\" 2>/dev/null || true; fi; fi")
        if remove_workspace:
            command += f'; cd "$HOME" && rm -rf -- {self.root}'
        try:
            subprocess.run(["ssh", *self.options, self.target, command], timeout=3,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        except (OSError, subprocess.TimeoutExpired):
            pass


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input")
    source.add_argument("--testdata", help="Concatenate test epochs as a timing fixture, not a continuous-recording validation")
    source.add_argument("--eegio_dir", help="Replay a continuous EEG-IO segment")
    p.add_argument("--eegio_subject")
    p.add_argument("--eegio_electrode", choices=["Fp1", "Fp2"], default="Fp1")
    p.add_argument("--input_fs", type=float, required=True)
    p.add_argument("--input_stage", choices=["raw","preprocessed"], default="raw")
    p.add_argument("--channels", default="1,2,8,16")
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--hop", type=int, default=4, help="Samples per replay hop")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--cpus", default="")
    p.add_argument("--weights", choices=["random","final"], default="final")
    p.add_argument("--target", choices=["host","pi"], default="host")
    p.add_argument("--ssh", default="")
    p.add_argument("--remote_dir", default="eeg")
    p.add_argument("--sigma", type=float)
    p.add_argument("--exploratory", action="store_true")
    p.add_argument("--out_dir", required=True)
    a=p.parse_args(argv)
    counts=channel_counts(a.channels)
    if a.hop < 1:
        p.error("Hop must be a positive integer")
    if not 1 <= a.repeats <= 10 or not 1 <= a.threads <= 16 or not 1 <= a.seconds <= 600:
        p.error("Use 1–10 repeats, 1–16 threads and 1–600 seconds per case")
    model = Path(a.model).resolve()
    if not model.is_file() or not model.with_suffix('.json').is_file():
        p.error("Model and sidecar must exist")
    import numpy as np
    if not np.isfinite(a.input_fs) or a.input_fs <= 0:
        p.error("Input rate must be finite and positive")
    n = int(a.seconds * a.input_fs) // a.hop * a.hop
    eegio = None
    if a.eegio_dir:
        if a.input_stage != "raw":
            p.error("EEG-IO replay requires causal filtering of the recording")
        try:
            values, eegio = eegio_input(a.eegio_dir, a.eegio_subject, a.eegio_electrode, a.input_fs)
        except (ValueError, FileNotFoundError) as exc:
            p.error(str(exc))
        source_path, input_kind = eegio["source"], "eegio"
    else:
        recording = Path(a.input or a.testdata).resolve()
        if not recording.is_file():
            p.error("Recording must exist")
        source_path = str(recording)
        input_kind = "testdata_fixture" if a.testdata else "recording"
        if a.testdata:
            with np.load(recording,allow_pickle=False) as data:
                if "fs" not in data or float(data["fs"]) != a.input_fs:
                    p.error("Test-set rate differs from the declared input rate")
                values=np.asarray(data["noisy"],dtype=np.float32).reshape(-1)
        else:
            values=np.load(recording,allow_pickle=False,mmap_mode='r')
            if values.ndim!=1 or values.dtype.kind not in 'fiu':
                p.error("Recording must be a real-valued 1-D EEG signal")
    if n>len(values) or n<a.hop or not np.isfinite(values[:n]).all():
        p.error("Input cannot supply the requested finite replay")
    run_id=datetime.now().strftime("channels_%Y%m%d_%H%M%S_")+uuid4().hex[:8]
    remote=Remote(a.ssh,a.remote_dir,run_id) if a.target=="pi" else None
    root=Path(a.out_dir).resolve()/run_id
    root.mkdir(parents=True, exist_ok=False)
    model_path=str(model)
    model_name=model.stem
    shutil.copyfile(model,root/'model_step.onnx')
    shutil.copyfile(model.with_suffix('.json'),root/'model_step.json')
    model=root/'model_step.onnx'
    recording=root/('timing_fixture.npy' if a.testdata else 'recording.npy')
    np.save(recording,np.asarray(values[:n],dtype=np.float32))
    manifest=dict(name=run_id, status="running", model=model_name, model_source=model_path, model_path=str(model),
                  model_sha256=fingerprint(model), model_metadata_sha256=fingerprint(model.with_suffix('.json')),
                  input=str(recording), input_source=source_path,
                  input_kind=input_kind, input_sha256=fingerprint(recording),
                  target=a.target, sample_rate_hz=a.input_fs, hop_samples=a.hop, counts=counts,
                  repeats=a.repeats, seconds=a.seconds, threads=a.threads, cpus=a.cpus,
                  channel_mode="replicated single-channel", results=[])
    if eegio is not None:
        manifest["eegio"] = eegio
    atomic_json(root/"experiment.json",manifest)
    def interrupted(*_):
        raise KeyboardInterrupt
    previous={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        if remote:
            print("Preparing the Pi runtime and copying the fixed model and recording",flush=True)
            remote.stage(model,recording)
        for c,repeat in case_order(counts,a.repeats):
            stem=f"c{c}_r{repeat}"
            print(f"[channels] {c} EEG channels · repeat {repeat}/{a.repeats} · {a.seconds:g} s",flush=True)
            args=["-m","deploy.paper_bench","--model","model_step.onnx" if remote else str(model),
                  "--input","recording.npy" if remote else str(recording),"--input_fs",str(a.input_fs),
                  "--input_stage",a.input_stage,"--out",stem if remote else str(root/stem),
                  "--target",a.target,"--seconds",str(a.seconds),"--channels",str(c),
                  "--threads",str(a.threads),"--weights",a.weights,"--hop",str(a.hop)]
            if a.cpus: args += ["--cpus",a.cpus]
            if a.sigma is not None: args += ["--sigma",str(a.sigma)]
            if a.exploratory or a.hop != 4: args += ["--exploratory"]
            if remote: remote.run_case(args,root,stem)
            else: run_cancellable([sys.executable,'-X','utf8',"-u",*args])
            item=json.loads((root/f"{stem}.json").read_text())
            manifest["results"].append({"case":stem,"repeat":repeat,**item})
            atomic_json(root/"experiment.json",manifest)
        manifest["status"]="complete"
    except KeyboardInterrupt:
        manifest["status"]="stopped"
        raise SystemExit(130)
    except Exception as exc:
        manifest.update(status="failed",error=str(exc))
        raise
    finally:
        if remote: remote.stop()
        atomic_json(root/"experiment.json",manifest)
        for s, handler in previous.items(): signal.signal(s,handler)
        print(f"[channels-result] {run_id} · {manifest['status']}",flush=True)


if __name__=="__main__":
    main()
