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
    GET  /api/runs/losses        complete runs only, grouped by loss_fn x artifact
    GET  /api/runs/optimizers    complete runs only, grouped by optimizer recipe
    GET  /api/runs/history?name=X    one finished run's per-epoch curve, for the chart
    POST /api/params             {values} → parameter breakdown of that model
    GET  /api/params/run?name=X  the same breakdown for a finished run's checkpoint
    POST /api/train              {name, extra_args} → start `python -m trainer --config configs/X`
    POST /api/train/stop
    GET  /api/train/status?since=N   incremental log lines from index N, plus the
                                 per-epoch validation metrics parsed from them

Binds to 127.0.0.1 only; there is no authentication.  One training job at
a time; its output is kept in memory (last 5000 lines) and mirrored to
runs/gui_train_<timestamp>.log.
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
# Training job
# ============================================================
class TrainingJob:
    """
    Runs one trainer subprocess, buffers its output line by line, and reads the
    per-epoch validation metrics back out of that output so the page can plot
    them while the run is going.

    Parsing the log (rather than the checkpoints) keeps the trainer untouched
    and costs nothing: the lines are already streaming through here.
    """

    MAX_LINES = 5000

    #   [fit | Epoch 12/40]
    EPOCH_RE = re.compile(r"\[([^\]|]+?)\s*\|\s*Epoch\s+(\d+)\s*/\s*(\d+)\]")
    #     -> train_loss = 0.4234  (24.1s)        (KD adds [task=... output=...])
    TRAIN_RE = re.compile(r"train_loss\s*=\s*(-?[\d.]+(?:[eE][-+]?\d+)?)")
    #     -> val_loss   = 0.4232  RRMSE_t=0.760  RRMSE_s=0.700  ACC=0.659
    #   The arrow is required: KDTrainer prints a "Teacher reference: val_loss=..."
    #   summary in the same shape, and that is not an epoch of this run.
    VAL_RE = re.compile(
        r"→\s*val_loss\s*=\s*(-?[\d.]+(?:[eE][-+]?\d+)?)\s+"
        r"RRMSE_t=\s*(-?[\d.]+(?:[eE][-+]?\d+)?)\s+"
        r"RRMSE_s=\s*(-?[\d.]+(?:[eE][-+]?\d+)?)\s+"
        r"ACC=\s*(-?[\d.]+(?:[eE][-+]?\d+)?)")

    def __init__(self):
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._lines: collections.deque = collections.deque(maxlen=self.MAX_LINES)
        self._offset = 0                       # index of _lines[0] in the full stream
        self.command: list[str] = []
        self.started: str | None = None
        self.log_path: Path | None = None
        self.returncode: int | None = None
        self._metrics: list[dict] = []         # one record per finished epoch
        self._epoch: dict | None = None        # epoch in progress
        self._max_epochs: int | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, command: list[str], cwd: Path, log_path: Path | None = None):
        with self._lock:
            if self.running:
                raise RuntimeError("a training run is already in progress")
            self._lines.clear()
            self._offset = 0
            self._metrics = []
            self._epoch = None
            self._max_epochs = None
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

    def metrics(self) -> list[dict]:
        with self._lock:
            return list(self._metrics)

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
            metrics = list(self._metrics)
            max_epochs = self._max_epochs
        return {
            "metrics": metrics, "max_epochs": max_epochs,
            "running": self.running,
            "returncode": self.returncode,
            "command": " ".join(shlex.quote(c) for c in self.command),
            "started": self.started,
            "log_path": str(self.log_path) if self.log_path else None,
            "since": since, "next": nxt, "lines": lines,
        }


JOB = TrainingJob()


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
        "configs": config_io.list_configs(),
        "runs": runs,
        "env": {"python": sys.executable, "project_root": str(PROJECT_ROOT),
                "device": _device_name()},
    }


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
                                   "losses": config_io.runs_by_loss(runs),
                                   "optimizers": config_io.runs_by_optimizer(runs)})
            if url.path == "/api/runs/kinds":
                return self._json({"kinds": config_io.runs_by_kind()})
            if url.path == "/api/runs/losses":
                return self._json({"losses": config_io.runs_by_loss()})
            if url.path == "/api/runs/optimizers":
                return self._json({"optimizers": config_io.runs_by_optimizer()})
            if url.path == "/api/runs/history":
                return self._json(config_io.read_run_history(q.get("name", "")))
            if url.path == "/api/params/run":
                return self._json(params.report_for_run(q.get("name", "")))
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
        if JOB.running:
            print("[gui] stopping the training job ...")
            JOB.stop()
        server.server_close()
        time.sleep(0.1)
