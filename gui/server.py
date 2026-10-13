"""
Local HTTP server behind the training GUI (stdlib only).

    GET  /                       the page (gui/static/index.html)
    GET  /api/meta               schema, choice lists, configs, runs, environment
    GET  /api/config?name=X      one preset: parsed values + raw text
    POST /api/preview            {name, values} → the text a save would write
    POST /api/config             {name, values} → save, returns the new text
    POST /api/config/new         {name, template} → create a preset
    GET  /api/runs               runs/ table (refreshed on demand)
    GET  /api/runs/kinds         complete runs only, grouped by variant x artifact
    GET  /api/runs/times         every run, newest first, grouped by day
    GET  /api/runs/losses        complete runs only, grouped by loss_fn x artifact
    GET  /api/runs/optimizers    complete runs only, grouped by optimizer recipe
    GET  /api/runs/history?name=X    one finished run's per-epoch curve, for the chart
    POST /api/params             {values} → parameter breakdown of that model
    GET  /api/params/run?name=X  the same breakdown for a finished run's checkpoint
    POST /api/train              {name, extra_args} → start `python -m trainer --config configs/X`
    POST /api/train/stop
    GET  /api/train/status?since=N   incremental log lines from index N, plus the
                                 per-epoch validation metrics parsed from them
    POST /api/eval/preview       {mode, run|runs, ...} → the command a run would use
    POST /api/eval               the same, but starts `python eval.py ...`
    POST /api/eval/stop
    GET  /api/eval/status?since=N    incremental log lines, plus the figures the
                                 run has reported writing so far
    GET  /api/eval/figure?path=X one file from figures/ (png / svg / json)

Binds to 127.0.0.1 only; there is no authentication.  One training job and
one evaluation job at a time, each independent of the other; their output is
kept in memory (last 5000 lines) and mirrored to
runs/gui_{train,eval}_<timestamp>.log.
"""

from __future__ import annotations

import collections
import json
import re
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from gui import config_io, params
from gui.config_io import PROJECT_ROOT, RUNS_DIR

STATIC_DIR = Path(__file__).resolve().parent / "static"


# ============================================================
# Job runner
# ============================================================
class Job:
    """
    Runs one subprocess, buffers its output line by line, and serves that
    buffer back incrementally so the page can tail it.

    Subclasses give the lines meaning by overriding ``_scan`` (called with the
    lock held, once per line) and ``_extra_status``.  The GUI keeps one
    instance per kind of job — training and evaluation — so a long evaluation
    does not lock out the Train tab, and neither job can clobber the other's
    log.
    """

    MAX_LINES = 5000
    KIND = "job"

    def __init__(self):
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._lines: collections.deque = collections.deque(maxlen=self.MAX_LINES)
        self._offset = 0                       # index of _lines[0] in the full stream
        self.command: list[str] = []
        self.started: str | None = None
        self.log_path: Path | None = None
        self.returncode: int | None = None
        self._reset_state()

    # ---- subclass hooks ----
    def _reset_state(self):
        """Clear whatever this job accumulates from the log."""

    def _scan(self, line: str):
        """Read meaning out of one log line.  Called with the lock held."""

    def _extra_status(self) -> dict:
        """This job's own status fields.  Called with the lock held."""
        return {}

    # ---- process ----
    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, command: list[str], cwd: Path, log_path: Path | None = None):
        with self._lock:
            if self.running:
                raise RuntimeError(f"a {self.KIND} job is already in progress")
            self._lines.clear()
            self._offset = 0
            self._reset_state()
            self.command = list(command)
            self.started = datetime.now().isoformat(timespec="seconds")
            self.returncode = None
            self.log_path = log_path
            if log_path:
                log_path.parent.mkdir(parents=True, exist_ok=True)
            self._proc = subprocess.Popen(
                command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"},
            )
            threading.Thread(target=self._pump, args=(self._proc,), daemon=True).start()

    def _pump(self, proc: subprocess.Popen):
        log = open(self.log_path, "w") if self.log_path else None
        try:
            for line in proc.stdout:
                line = line.rstrip("\n")
                with self._lock:
                    if len(self._lines) == self._lines.maxlen:
                        self._offset += 1
                    self._lines.append(line)
                    self._scan(line)
                if log:
                    log.write(line + "\n")
                    log.flush()
        finally:
            proc.stdout.close()
            proc.wait()
            self.returncode = proc.returncode
            if log:
                log.write(f"\n[exit code {proc.returncode}]\n")
                log.close()

    def stop(self, grace: float = 5.0):
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return False
        proc.terminate()
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            proc.kill()
        return True

    def status(self, since: int = 0) -> dict:
        with self._lock:
            start = max(since - self._offset, 0)
            lines = list(self._lines)[start:]
            nxt = self._offset + len(self._lines)
            extra = self._extra_status()
        return {
            **extra,
            "kind": self.KIND,
            "running": self.running,
            "returncode": self.returncode,
            "command": " ".join(shlex.quote(c) for c in self.command),
            "started": self.started,
            "log_path": str(self.log_path) if self.log_path else None,
            "since": since, "next": nxt, "lines": lines,
        }


class TrainingJob(Job):
    """
    A trainer subprocess, with the per-epoch validation metrics read back out
    of its output so the page can plot them while the run is going.

    Parsing the log (rather than the checkpoints) keeps the trainer untouched
    and costs nothing: the lines are already streaming through here.
    """

    KIND = "train"

    #   [fit | Epoch 12/40]
    EPOCH_RE = re.compile(r"\[([^\]|]+?)\s*\|\s*Epoch\s+(\d+)\s*/\s*(\d+)\]")
    #     -> train_loss = 0.4234  (24.1s)        (KD adds [task=... output=...])
    TRAIN_RE = re.compile(r"train_loss\s*=\s*(-?[\d.]+(?:[eE][-+]?\d+)?)")
    #     -> val_loss   = 0.4232  RRMSE_t=0.760  RRMSE_f=0.700  CC=0.659
    #   The arrow is required: KDTrainer prints a "Teacher reference: val_loss=..."
    #   summary in the same shape, and that is not an epoch of this run.
    #   RRMSE_s/ACC are the names the trainer printed before they were renamed
    #   for the paper (spectral -> f, and ACC because it reads as "accuracy").
    #   Both spellings are accepted so the logs already in runs/ still plot.
    _NUM = r"(-?[\d.]+(?:[eE][-+]?\d+)?)"
    VAL_RE = re.compile(
        r"→\s*val_loss\s*=\s*" + _NUM + r"\s+"
        r"RRMSE_t=\s*" + _NUM + r"\s+"
        r"RRMSE_[fs]=\s*" + _NUM + r"\s+"
        r"(?:CC|ACC)=\s*" + _NUM)

    def _reset_state(self):
        self._metrics: list[dict] = []         # one record per finished epoch
        self._epoch: dict | None = None        # epoch in progress
        self._max_epochs: int | None = None

    def _scan(self, line: str):
        """Pull epoch boundaries and validation metrics out of one log line.

        Called with the lock held.  A validation line only counts once an epoch
        header has been seen, so the teacher's reference pass that KDTrainer
        runs before training starts is ignored.
        """
        m = self.EPOCH_RE.search(line)
        if m:
            self._epoch = {"epoch": int(m.group(2)), "phase": m.group(1).strip()}
            self._max_epochs = int(m.group(3))
            return
        if self._epoch is None:
            return
        m = self.TRAIN_RE.search(line)
        if m and "train_loss" not in self._epoch:
            self._epoch["train_loss"] = float(m.group(1))
            return
        m = self.VAL_RE.search(line)
        if m:
            self._epoch.update(val_loss=float(m.group(1)), rrmse_t=float(m.group(2)),
                               rrmse_s=float(m.group(3)), acc=float(m.group(4)))
            self._metrics.append(self._epoch)
            self._epoch = None

    def _extra_status(self) -> dict:
        return {"metrics": list(self._metrics), "max_epochs": self._max_epochs}

    def metrics(self) -> list[dict]:
        with self._lock:
            return list(self._metrics)


class EvalJob(Job):
    """
    An ``eval.py`` subprocess, with the files it reports writing collected as
    they scroll past, so the page can show the figures without polling the
    filesystem or guessing their names.

    Only paths inside ``figures/`` are kept: that is the one directory the GUI
    serves from, so a path it cannot serve has no business in the list.
    """

    KIND = "eval"

    #   [Switch] Figure saved: figures/gui_eval_.../switch_response.png
    #   [Switch] Numbers saved: figures/gui_eval_.../switch_response.json
    SAVED_RE = re.compile(r"(?:Figure|Numbers) saved:\s*(\S.*?)\s*$")

    def _reset_state(self):
        self._artifacts: list[str] = []

    def _scan(self, line: str):
        m = self.SAVED_RE.search(line)
        if not m:
            return
        rel = figure_rel_path(m.group(1))
        if rel and rel not in self._artifacts:
            self._artifacts.append(rel)

    def _extra_status(self) -> dict:
        return {"artifacts": list(self._artifacts)}

    def artifacts(self) -> list[str]:
        with self._lock:
            return list(self._artifacts)


JOB = TrainingJob()
EVAL_JOB = EvalJob()


# ============================================================
# Evaluation commands
# ============================================================
#: Where eval.py is allowed to write, and the only tree the GUI serves images
#: from.  Everything the Evaluate tab produces lands in a timestamped folder
#: under here.
FIGURES_DIR = PROJECT_ROOT / "figures"

#: mode -> (needs a checkpoint, needs several runs, needs the dataset)
EVAL_MODES = {
    "switch":         dict(ckpt=True,  multi=False, data=True,
                           label="Artifact switch — error vs. distance to a switch"),
    "compare_switch": dict(ckpt=False, multi=True,  data=True,
                           label="Artifact switch — several models overlaid"),
    "snr":            dict(ckpt=True,  multi=False, data=True,
                           label="Per-SNR metrics (table only)"),
    "snr_plot":       dict(ckpt=True,  multi=False, data=True,
                           label="Per-SNR metrics + line charts"),
    "compare_snr":    dict(ckpt=False, multi=True,  data=True,
                           label="Per-SNR metrics — several models overlaid"),
    "waveform":       dict(ckpt=True,  multi=False, data=True,
                           label="Waveforms + PSD"),
    "attention":      dict(ckpt=True,  multi=False, data=True,
                           label="Attention heatmaps (attention variants only)"),
    "curves":         dict(ckpt=False, multi=False, data=False,
                           label="Training curves from history.json"),
    "compare_curves": dict(ckpt=False, multi=True,  data=False,
                           label="Training curves — several runs overlaid"),
    "all":            dict(ckpt=True,  multi=False, data=True,
                           label="Everything for one model"),
}


def figure_rel_path(path: str) -> str | None:
    """
    ``path`` as a slash-separated path relative to ``figures/``, or None if it
    does not point inside it.

    eval.py prints whatever ``--save_dir`` it was given, so the string can be
    absolute or relative to the project root; both are resolved the same way.
    """
    try:
        p = Path(path.strip())
        p = p if p.is_absolute() else (PROJECT_ROOT / p)
        return str(p.resolve().relative_to(FIGURES_DIR.resolve()))
    except (ValueError, OSError):
        return None


def figure_path(rel: str) -> Path:
    """
    Resolve a client-supplied path inside ``figures/``, or refuse it.

    The GUI binds to localhost with no authentication, so this is the one
    place a request names a file: it is resolved first and checked against the
    figures directory afterwards, which is what makes ``../`` and symlinks out
    of the tree fail rather than escape.
    """
    if not rel:
        raise ValueError("no figure named")
    target = (FIGURES_DIR / rel).resolve()
    root = FIGURES_DIR.resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"{rel} is outside figures/")
    if not target.is_file():
        raise FileNotFoundError(f"figures/{rel} does not exist")
    return target


def _run_dir(name: str) -> Path:
    """A runs/ subdirectory named by the client, validated."""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(f"invalid run name: {name!r}")
    d = RUNS_DIR / name
    if not d.is_dir():
        raise FileNotFoundError(f"no such run: runs/{name}")
    return d


def _flag(cmd: list[str], flag: str, value, *, skip_if=(None, "")):
    if value not in skip_if:
        cmd += [flag, str(value)]


def build_eval_command(spec: dict) -> tuple[list[str], str]:
    """
    ``python eval.py …`` for one Evaluate-tab submission, plus the save
    directory it will write into (relative to the project root).

    Every path the request names is validated here rather than at the shell:
    run names must be real directories under ``runs/`` and the output folder is
    pinned under ``figures/``.  ``extra_args`` is split with shlex and appended
    verbatim — the same escape hatch the Train tab has, and with the same
    caveat that it is passed through untouched.
    """
    mode = spec.get("mode") or "switch"
    if mode not in EVAL_MODES:
        raise ValueError(f"unknown mode: {mode!r} (expected one of {sorted(EVAL_MODES)})")
    needs = EVAL_MODES[mode]

    cmd = [sys.executable, "eval.py", "--mode", mode]

    if needs["multi"]:
        runs = spec.get("runs") or []
        if len(runs) < 2:
            raise ValueError(f"mode '{mode}' compares models — pick at least two runs")
        entries = []
        for r in runs:
            name = r["name"] if isinstance(r, dict) else str(r)
            _run_dir(name)
            label = (r.get("label") or "").strip() if isinstance(r, dict) else ""
            if ":" in label:
                raise ValueError(f"a label cannot contain ':' ({label!r})")
            entries.append(f"runs/{name}:{label}" if label else f"runs/{name}")
        cmd += ["--compare_runs", *entries]
    else:
        name = spec.get("run")
        if not name:
            raise ValueError(f"mode '{mode}' needs a run")
        run = _run_dir(name)
        if mode == "curves":
            if not (run / "history.json").is_file():
                raise FileNotFoundError(
                    f"runs/{name} has no history.json — that run never finished")
            cmd += ["--history", f"runs/{name}/history.json"]
        else:
            if not (run / "best.pt").is_file():
                raise FileNotFoundError(f"runs/{name} has no best.pt")
            cmd += ["--ckpt", f"runs/{name}/best.pt"]

    if needs["data"]:
        data_dir = (spec.get("data_dir") or "").strip()
        if not data_dir:
            raise ValueError(f"mode '{mode}' needs a data directory")
        cmd += ["--data_dir", data_dir, "--artifact", spec.get("artifact") or "EOG"]
        _flag(cmd, "--master_seed", spec.get("master_seed"))

    if mode in ("switch", "compare_switch", "all"):
        seeds = str(spec.get("probe_seeds") or "").replace(",", " ").split()
        if seeds:
            for seed in seeds:
                try:
                    int(seed)
                except ValueError:
                    raise ValueError(
                        f"probe seeds must be whole numbers, got {seed!r}") from None
            cmd += ["--probe_seeds", *seeds]
        _flag(cmd, "--switch_window", spec.get("switch_window"))
        _flag(cmd, "--switch_ramp", spec.get("switch_ramp"), skip_if=(None, "", 0, "0"))
        _flag(cmd, "--switch_snr", spec.get("switch_snr"))
        _flag(cmd, "--switch_epochs", spec.get("switch_epochs"))
        rng = spec.get("switch_range")
        if rng:
            cmd += ["--switch_range", str(rng[0]), str(rng[1])]
        if spec.get("switch_control") is False:
            cmd += ["--no-switch_control"]

    if mode in ("waveform", "attention", "all"):
        _flag(cmd, "--vis_snr", spec.get("vis_snr"))
        _flag(cmd, "--n_samples", spec.get("n_samples"))
        _flag(cmd, "--attn_sample", spec.get("attn_sample"))

    save_dir = (spec.get("save_dir") or "").strip()
    if not save_dir:
        save_dir = f"figures/gui_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    target = (PROJECT_ROOT / save_dir).resolve()
    root = FIGURES_DIR.resolve()
    if target != root and root not in target.parents:
        raise ValueError("save_dir must be inside figures/")
    cmd += ["--save_dir", save_dir]

    cmd += shlex.split(spec.get("extra_args") or "")
    return cmd, save_dir


def build_train_command(config_name: str, extra_args: str = "") -> list[str]:
    """`python -m trainer --config configs/<name> [extra flags]` with this interpreter."""
    config_io.config_path(config_name)                 # validates the name
    return [sys.executable, "-m", "trainer", "--config", f"configs/{config_name}",
            *shlex.split(extra_args or "")]


# ============================================================
# HTTP handler
# ============================================================
def meta() -> dict:
    from models.variants import VARIANT_REGISTRY, BIDIRECTIONAL_VARIANTS
    runs = config_io.list_runs()
    return {
        "kinds": config_io.runs_by_kind(runs),
        "times": config_io.runs_by_time(runs),
        "losses": config_io.runs_by_loss(runs),
        "optimizers": config_io.runs_by_optimizer(runs),
        "schema": config_io.schema_as_json(),
        "choices": {
            "variants": list(VARIANT_REGISTRY),
            "bidirectional_variants": BIDIRECTIONAL_VARIANTS,
            "artifacts": config_io.ARTIFACTS,
            "losses": config_io.LOSS_NAMES,
            "feature_losses": ["mse", "l1"],
            "tq_key_modes": ["prod", "mse"],
            "tq_train_modes": ["off", "ste"],
            "devices": _device_choices(),
        },
        "datalists": {"checkpoints": [r["checkpoint"] for r in runs]},
        "eval_modes": [{"mode": m, **spec} for m, spec in EVAL_MODES.items()],
        "eval_defaults": {
            "data_dir": config_io.FIELDS["data_dir"].default,
            "probe_seeds": " ".join(str(s) for s in _switch_probe_seeds()),
        },
        "configs": config_io.list_configs(),
        "runs": runs,
        "env": {"python": sys.executable, "project_root": str(PROJECT_ROOT),
                "device": _device_name()},
    }


def _switch_probe_seeds() -> list[int]:
    try:
        from data.Dataloader import SWITCH_PROBE_SEEDS
        return list(SWITCH_PROBE_SEEDS)
    except Exception:            # torch missing or broken: the GUI still works
        return [0, 1, 2, 3, 4]


def _device_name() -> str:
    try:
        from utils.utils import get_device
        return str(get_device())
    except Exception:            # torch missing or broken: the GUI still works
        return "unknown"


def _device_choices() -> list[str]:
    try:
        from utils.utils import DEVICE_CHOICES
        return list(DEVICE_CHOICES)
    except Exception:
        return ["auto", "cpu", "cuda", "mps"]


class Handler(BaseHTTPRequestHandler):
    server_version = "EEGTrainGUI/1.0"

    # ---- plumbing ----
    def log_message(self, fmt, *args):          # quieter console
        if self.server.verbose:                 # type: ignore[attr-defined]
            super().log_message(fmt, *args)

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message, status=400):
        self._json({"error": str(message)}, status)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        data = json.loads(raw or b"{}")
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _file(self, path: Path, content_type: str):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # ---- routes ----
    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path in ("/", "/index.html"):
                return self._file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            if url.path == "/api/meta":
                return self._json(meta())
            if url.path == "/api/config":
                return self._json(config_io.read_config(q.get("name", "")))
            if url.path == "/api/runs":
                runs = config_io.list_runs()
                return self._json({"runs": runs, "kinds": config_io.runs_by_kind(runs),
                                   "times": config_io.runs_by_time(runs),
                                   "losses": config_io.runs_by_loss(runs),
                                   "optimizers": config_io.runs_by_optimizer(runs)})
            if url.path == "/api/runs/kinds":
                return self._json({"kinds": config_io.runs_by_kind()})
            if url.path == "/api/runs/times":
                return self._json({"times": config_io.runs_by_time()})
            if url.path == "/api/runs/losses":
                return self._json({"losses": config_io.runs_by_loss()})
            if url.path == "/api/runs/optimizers":
                return self._json({"optimizers": config_io.runs_by_optimizer()})
            if url.path == "/api/runs/history":
                return self._json(config_io.read_run_history(q.get("name", "")))
            if url.path == "/api/params/run":
                return self._json(params.report_for_run(q.get("name", "")))
            if url.path == "/api/eval/status":
                return self._json(EVAL_JOB.status(int(q.get("since", 0))))
            if url.path == "/api/eval/figure":
                path = figure_path(q.get("path", ""))
                kinds = {".png": "image/png", ".svg": "image/svg+xml",
                         ".json": "application/json; charset=utf-8"}
                return self._file(path, kinds.get(path.suffix.lower(),
                                                  "application/octet-stream"))
            if url.path == "/api/train/status":
                return self._json(JOB.status(int(q.get("since", 0))))
            return self._error("not found", 404)
        except (ValueError, FileNotFoundError, KeyError) as e:
            return self._error(e, 400)

    def do_POST(self):
        url = urlparse(self.path)
        try:
            body = self._body()
            if url.path == "/api/preview":
                text = config_io.render_config(body["name"], body.get("values", {}))
                return self._json({"name": body["name"], "text": text})
            if url.path == "/api/config":
                text = config_io.save_config(body["name"], body.get("values", {}))
                return self._json({"name": body["name"], "text": text, "saved": True})
            if url.path == "/api/config/new":
                text = config_io.create_config(body["name"], body.get("template") or None)
                return self._json({"name": body["name"], "text": text,
                                   "configs": config_io.list_configs()})
            if url.path == "/api/params":
                return self._json(params.report(body.get("values", {})))
            if url.path == "/api/train":
                if JOB.running:
                    return self._error("a training run is already in progress", 409)
                cmd = build_train_command(body["name"], body.get("extra_args", ""))
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                JOB.start(cmd, PROJECT_ROOT, RUNS_DIR / f"gui_train_{stamp}.log")
                return self._json(JOB.status(0))
            if url.path == "/api/train/stop":
                return self._json({"stopped": JOB.stop(), **JOB.status(0)})
            if url.path == "/api/eval/preview":
                cmd, save_dir = build_eval_command(body)
                return self._json({"command": " ".join(shlex.quote(c) for c in cmd),
                                   "save_dir": save_dir})
            if url.path == "/api/eval":
                if EVAL_JOB.running:
                    return self._error("an evaluation is already in progress", 409)
                cmd, save_dir = build_eval_command(body)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                EVAL_JOB.start(cmd, PROJECT_ROOT, RUNS_DIR / f"gui_eval_{stamp}.log")
                return self._json({**EVAL_JOB.status(0), "save_dir": save_dir})
            if url.path == "/api/eval/stop":
                return self._json({"stopped": EVAL_JOB.stop(), **EVAL_JOB.status(0)})
            return self._error("not found", 404)
        except FileExistsError as e:
            return self._error(e, 409)
        except (ValueError, FileNotFoundError, KeyError, RuntimeError) as e:
            return self._error(e, 400)


class GUIServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, verbose: bool = False):
        super().__init__((host, port), Handler)
        self.verbose = verbose


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True,
          verbose: bool = False) -> None:
    server = GUIServer(host, port, verbose)
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"[gui] Training GUI at {url}   (project: {PROJECT_ROOT})")
    print("[gui] Ctrl-C to stop; a running training job is terminated with the server.")
    if open_browser:
        import webbrowser
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for job, what in ((JOB, "training"), (EVAL_JOB, "evaluation")):
            if job.running:
                print(f"[gui] stopping the {what} job ...")
                job.stop()
        server.server_close()
        time.sleep(0.1)
