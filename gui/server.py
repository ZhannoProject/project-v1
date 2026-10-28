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
    GET  /api/runs/deploys       one row per Pi/host measurement, newest first
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
    GET  /api/pi/meta            exports, test sets and latency traces on disk
    POST /api/pi/testdata        {model, n} → cut the test set matching that model's
                                 split, and the refreshed list
    GET  /api/eegio/meta         EEG-IO recordings on disk, and the runs worth
                                 comparing on them
    POST /api/eegio/evaluate     {runs, n_subjects, ...} → suppression, fidelity,
                                 residual detections and onset lead per subject,
                                 paired by subject between models
    GET  /api/kd/meta            distilled runs, their teachers and their controls
    GET  /api/kd/checks?student=X    the six KD invariants, with their evidence
    POST /api/kd/decompose       {teacher, students, ...} → ||S-G||^2 split into
                                 ||S-T||^2 + ||T-G||^2 + the cross term
    GET  /api/rf/meta            the receptive-field ladder and its defaults
    GET  /api/rf/results         one row per trained rung: reach vs metrics, over seeds
    POST /api/rf/preview         {variant, rungs, …} → the training commands
    POST /api/rf/run             the same, but starts them
    POST /api/rf/stop
    GET  /api/rf/status?since=N  incremental log lines for the sweep
    POST /api/pi/preview         {target, model, …} → the commands a run would use
    POST /api/pi/run             the same, but starts them
    POST /api/pi/stop
    GET  /api/pi/status?since=N  incremental log lines, the step the run is on,
                                 and the numbers parsed out of them so far
    GET  /api/pi/trace?name=X    one trace from deploy/artifacts/pi_latency/

Binds to 127.0.0.1 only; there is no authentication.  One training job, one
evaluation job and one deploy job at a time, each independent of the others;
their output is kept in memory (last 5000 lines) and mirrored to
runs/gui_{train,eval,deploy}_<timestamp>.log.
"""

from __future__ import annotations

import collections
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from gui import config_io, eegio_tab, kd, lookahead, params, pi, rfsweep
# The one place the "seed 42 keeps the bare name" rule is written down; the
# run titles have to derive the same path the trainer will.
from trainer.__main__ import DEFAULT_MASTER_SEED
from gui.config_io import PROJECT_ROOT, RUNS_DIR

STATIC_DIR = Path(__file__).resolve().parent / "static"


# ============================================================
# Job runner
# ============================================================
class Job:
    """
    Runs a subprocess — or a short sequence of them — buffers the output line
    by line, and serves that buffer back incrementally so the page can tail it.

    Subclasses give the lines meaning by overriding ``_scan`` (called with the
    lock held, once per line) and ``_extra_status``.  The GUI keeps one
    instance per kind of job — training, evaluation and deployment — so a long
    evaluation does not lock out the Train tab, and no job can clobber
    another's log.

    A sequence is one job, not three: the Deploy tab's "export, measure, then
    sweep the hop sizes" is a single thing the user started and a single log to
    read, and a step that fails stops the ones after it rather than leaving
    half a measurement to be puzzled over.
    """

    MAX_LINES = 5000
    KIND = "job"

    def __init__(self):
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._lines: collections.deque = collections.deque(maxlen=self.MAX_LINES)
        self._offset = 0                       # index of _lines[0] in the full stream
        self.command: list[str] = []
        # Identity, for the kinds of job there can be several of at once. A
        # lone job keeps the defaults and never has to be addressed by id.
        self.id: str = ""
        self.title: str = ""
        self.steps: list[dict] = []            # [{label, command, env, returncode}]
        self.step = 0                          # index of the step in progress
        self.started: str | None = None
        self.log_path: Path | None = None
        self.returncode: int | None = None
        self._active = False                   # true from start() until the last step ends
        self._cancelled = False
        # Bumped by every start, and again if a stop has to abandon a reader
        # that will not come back.  Output from an older generation is dropped
        # rather than appended to whatever is running now.
        self._gen = 0
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
        # Not "the subprocess is alive": between two steps of a sequence there
        # is no subprocess, and the job has not finished.
        return self._active

    def start(self, command: list[str], cwd: Path, log_path: Path | None = None):
        """One command, the ordinary case."""
        self.start_steps([{"label": self.KIND, "command": list(command)}], cwd, log_path)

    def start_steps(self, steps: list[dict], cwd: Path, log_path: Path | None = None):
        """
        Several commands, run in order as one job.

        Each step is ``{label, command[, env]}``; the first non-zero exit stops
        the rest.  ``env`` is merged into this process's environment for that
        step only — it is how the Deploy tab passes ``HOP=…`` to a shell script
        that reads its settings from the environment.
        """
        if not steps:
            raise ValueError("nothing to run")
        with self._lock:
            if self._active:
                raise RuntimeError(f"a {self.KIND} job is already in progress")
            self._lines.clear()
            self._offset = 0
            self._reset_state()
            self.steps = [{"label": s.get("label") or self.KIND, "command": list(s["command"]),
                           "env": dict(s.get("env") or {}), "returncode": None} for s in steps]
            self.step = 0
            self.command = list(self.steps[0]["command"])
            self.started = datetime.now().isoformat(timespec="seconds")
            self.returncode = None
            self._cancelled = False
            self._active = True
            self._gen += 1
            gen = self._gen
            self.log_path = log_path
            if log_path:
                log_path.parent.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self._run_steps, args=(cwd, gen), daemon=True).start()

    def _append(self, line: str):
        """Add one line to the buffer.  Caller holds the lock."""
        if len(self._lines) == self._lines.maxlen:
            self._offset += 1
        self._lines.append(line)

    def _emit(self, line: str, log, gen: int):
        with self._lock:
            if gen != self._gen:
                return          # a stop gave up on this step; its output is no longer ours
            self._append(line)
            self._scan(line)
        if log:
            log.write(line + "\n")
            log.flush()

    def _run_steps(self, cwd: Path, gen: int):
        log = open(self.log_path, "w") if self.log_path else None
        rc = 0
        try:
            for i, step in enumerate(self.steps):
                if self._cancelled:
                    self._emit(f"[gui] stopped before: {step['label']}", log, gen)
                    break
                self.step = i
                self.command = step["command"]
                if len(self.steps) > 1:
                    self._emit("", log, gen)
                    self._emit(f"=== step {i + 1}/{len(self.steps)}: {step['label']} ===", log, gen)
                rc = self._run_one(step, cwd, log, gen)
                if gen != self._gen:
                    return          # abandoned by stop(); the job has already moved on
                step["returncode"] = rc
                if rc != 0:
                    if len(self.steps) > 1 and i + 1 < len(self.steps):
                        self._emit(f"[gui] {step['label']} exited {rc} — "
                                   "the remaining steps were not run", log, gen)
                    break
        finally:
            with self._lock:
                if gen == self._gen:
                    self.returncode = rc
                    self._proc = None
                    self._active = False
            if log:
                log.write(f"\n[exit code {rc}]\n")
                log.close()

    def _spawn(self, step: dict, cwd: Path) -> subprocess.Popen | None:
        """
        Start one step's process, unless the job has already been cancelled.

        The check and the Popen are taken together under the lock because
        ``stop()`` can arrive between two steps — or a moment before the first
        process exists at all.  Without this it would find nothing to
        terminate and the step would start just afterwards anyway.

        Two details are what make a step stoppable.  ``start_new_session``
        gives it a session of its own, so ``stop()`` can signal everything it
        forked rather than only the process named here — a step is not always
        one process, and ``deploy/run_on_pi.sh`` forks ssh and scp.  And stdin
        is ``/dev/null``: a subprocess that unexpectedly asks a question (ssh
        wanting a password) would otherwise read the server's own stdin and
        wait there forever.
        """
        with self._lock:
            if self._cancelled:
                return None
            self._proc = subprocess.Popen(
                step["command"], cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1", **step.get("env", {})},
                start_new_session=(os.name == "posix"),
            )
            return self._proc

    def _run_one(self, step: dict, cwd: Path, log, gen: int) -> int:
        proc = self._spawn(step, cwd)
        if proc is None:
            self._emit(f"[gui] stopped before: {step['label']}", log, gen)
            return -1
        try:
            for line in proc.stdout:
                self._emit(line.rstrip("\n"), log, gen)
        except ValueError:                     # the pipe was closed under the reader
            pass
        finally:
            try:
                proc.stdout.close()
            except (OSError, ValueError):
                pass
        return proc.wait()

    # ---- stopping ----
    def _session_of(self, proc: subprocess.Popen) -> int | None:
        """
        The step's own process group, or None if it has not got one.

        The guard matters: if ``start_new_session`` did not take effect the
        process group is *this server's*, and signalling it would kill the GUI
        along with the job.
        """
        if os.name != "posix":
            return None
        try:
            pgid = os.getpgid(proc.pid)
        except OSError:
            return None
        return pgid if pgid == proc.pid else None

    def _signal(self, proc: subprocess.Popen, group: int | None, sig: int):
        try:
            if group is not None:
                os.killpg(group, sig)
            elif proc.poll() is None:
                proc.send_signal(sig)
        except (OSError, ValueError):          # already gone, or not ours to signal
            pass

    def _end_tree(self, proc: subprocess.Popen, grace: float):
        """
        End the step and everything it started.

        Signalling only the process named in the step is not enough: whatever
        it forked inherited this job's output pipe, and the pipe does not close
        while any of them holds it — so the reader below never sees the end of
        the stream and the job never comes to rest.  That is how a Pi run
        became impossible to stop: ``run_on_pi.sh`` forks ssh and scp, they
        outlive the SIGTERM sent to bash, and the job stayed "running" with
        nothing left to stop.
        """
        group = self._session_of(proc)
        self._signal(proc, group, signal.SIGTERM)
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
        # Whether or not the step itself went quietly, anything it left behind
        # is still holding the pipe.  One SIGKILL to the session settles both.
        self._signal(proc, group, signal.SIGKILL)
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass

    def stop(self, grace: float = 5.0):
        with self._lock:
            if not self._active:
                return False
            self._cancelled = True             # so a sequence stops here, not at the next step
            proc, gen = self._proc, self._gen
        if proc is not None:
            self._end_tree(proc, grace)
        # A process that escaped its session (anything that daemonises) could
        # still be holding the pipe.  Rather than leave a job that cannot be
        # stopped — the worst of the outcomes — give the reader a moment and
        # then let go of it: the generation counter makes an orphaned reader
        # harmless, since nothing it writes is accepted any more.
        for _ in range(20):
            if not self._active:
                return True
            time.sleep(0.05)
        with self._lock:
            if self._active and gen == self._gen:
                self._gen += 1
                self._proc = None
                self._active = False
                self.returncode = -int(signal.SIGKILL)
                self._append("[gui] stopped — something this step started is still holding "
                             "its output pipe open, so the job was let go of rather than "
                             "left running")
        return True

    def status(self, since: int = 0) -> dict:
        with self._lock:
            start = max(since - self._offset, 0)
            lines = list(self._lines)[start:]
            nxt = self._offset + len(self._lines)
            extra = self._extra_status()
            steps = [{"label": s["label"], "returncode": s["returncode"],
                      "command": " ".join(shlex.quote(c) for c in s["command"])}
                     for s in self.steps]
        return {
            **extra,
            "kind": self.KIND,
            "id": self.id, "title": self.title,
            "running": self.running,
            "returncode": self.returncode,
            "command": " ".join(shlex.quote(c) for c in self.command),
            "steps": steps, "step": self.step,
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


class TooManyJobs(RuntimeError):
    """
    The concurrent-run cap was reached.

    A conflict rather than a bad request: the call was well formed and would
    succeed once a run finishes, which is the distinction 409 carries and the
    page relies on to tell "wait" from "you asked for something impossible".
    """


class TrainingJobs:
    """
    Several training runs at once, each with its own log, buffer and curve.

    The tab used to hold one run because the server held one ``TrainingJob``,
    and a second **Train** was refused with a 409.  That is the wrong shape for
    what this project actually does: a seed repeat is three runs of the same
    config, and a lookahead ladder is one run per rung, so the common task is
    *n* runs that differ in one flag and are read side by side.  Waiting for
    each to finish before starting the next made that a sequence of sittings.

    What it does not do is make them free.  They share one machine, and on MPS
    one accelerator, so concurrent runs contend: the per-epoch wall times in
    each log stop being comparable with a run that had the box to itself, and
    on a laptop three at once is slower in total than three in a row.  The
    metrics are unaffected -- each run has its own seed, data and weights --
    which is why this is allowed at all. ``MAX_ACTIVE`` keeps it from becoming
    absurd, and the page says what concurrency costs rather than hiding it.

    A finished job is kept so its log and curve can still be read; the oldest
    are dropped once there are ``MAX_KEPT`` of them. Jobs still running are
    never dropped.
    """

    #: Concurrent runs. Four is already more than a laptop wants; the point of
    #: the cap is that an accidental loop cannot spawn fifty.
    MAX_ACTIVE = 4
    #: Finished runs kept for reading back, newest first.
    MAX_KEPT = 16

    def __init__(self, factory=TrainingJob):
        self._factory = factory
        self._lock = threading.Lock()
        self._jobs: "collections.OrderedDict[str, Job]" = collections.OrderedDict()

    # ---- identity ----
    def _new_id(self, stamp: str) -> str:
        """``gui_train_<stamp>``, with a suffix if that second is taken."""
        base = f"gui_train_{stamp}"
        if base not in self._jobs:
            return base
        n = 2
        while f"{base}_{n}" in self._jobs:
            n += 1
        return f"{base}_{n}"

    # ---- lifecycle ----
    def room(self) -> int:
        """
        How many more runs may start now.

        Asked before a batch rather than discovered during one: starting two
        of three seeds and refusing the third leaves a half-finished request
        that reads like a failure of the runs that did start.
        """
        with self._lock:
            return self.MAX_ACTIVE - len([j for j in self._jobs.values() if j.running])

    def start(self, command: list[str], cwd: Path, runs_dir: Path, title: str = "") -> Job:
        with self._lock:
            active = [j for j in self._jobs.values() if j.running]
            if len(active) >= self.MAX_ACTIVE:
                raise TooManyJobs(
                    f"{len(active)} training runs are already going, which is the limit "
                    f"({self.MAX_ACTIVE}). Stop one, or wait for it to finish — they share "
                    f"one machine and slow each other down.")
            job_id = self._new_id(datetime.now().strftime("%Y%m%d_%H%M%S"))
            job = self._factory()
            job.id = job_id
            # Two live runs deriving one directory is a real thing that
            # happens -- a preset run twice with nothing overridden -- and the
            # second will write over the first. The switcher cannot fix that,
            # but it must not show two identical pills while it happens.
            label = title or job_id
            taken = {j.title for j in self._jobs.values() if j.running}
            if label in taken:
                n = 2
                while f"{label} #{n}" in taken:
                    n += 1
                label = f"{label} #{n}"
            job.title = label
            self._jobs[job_id] = job
            self._prune()
        job.start(command, cwd, runs_dir / f"{job_id}.log")
        return job

    def _prune(self):
        """Drop the oldest finished jobs. Caller holds the lock."""
        done = [k for k, j in self._jobs.items() if not j.running]
        for key in done[: max(0, len(done) - self.MAX_KEPT)]:
            del self._jobs[key]

    def get(self, job_id: str | None) -> Job | None:
        """
        One job by id, or the one the page means when it does not say.

        Without an id this is the newest *running* job, falling back to the
        newest of any state — which is what a page that has just loaded, or an
        older client that never sent one, is asking for.
        """
        with self._lock:
            if job_id:
                return self._jobs.get(job_id)
            running = [j for j in self._jobs.values() if j.running]
            if running:
                return running[-1]
            return next(reversed(self._jobs.values()), None) if self._jobs else None

    def summaries(self) -> list[dict]:
        """
        One small record per job for the switcher: enough to label a pill and
        show how far along it is, without shipping anybody's log.
        """
        with self._lock:
            jobs = list(self._jobs.values())
        out = []
        for j in jobs:
            metrics = j.metrics() if hasattr(j, "metrics") else []
            best = min((m for m in metrics if m.get("val_loss") is not None),
                       key=lambda m: m["val_loss"], default=None)
            out.append({
                "id": j.id, "title": j.title, "running": j.running,
                "returncode": j.returncode, "started": j.started,
                "epoch": metrics[-1]["epoch"] if metrics else None,
                "max_epochs": j.status(0).get("max_epochs") if j.running or metrics else None,
                "best_val_loss": best["val_loss"] if best else None,
                "rrmse_t": metrics[-1].get("rrmse_t") if metrics else None,
            })
        out.reverse()                           # newest first, as the page lists them
        return out

    def stop(self, job_id: str | None = None) -> bool:
        job = self.get(job_id)
        return bool(job and job.stop())

    def stop_all(self) -> int:
        with self._lock:
            jobs = [j for j in self._jobs.values() if j.running]
        return sum(1 for j in jobs if j.stop())

    @property
    def running(self) -> bool:
        """True while any run is going — what the window-close guard asks."""
        with self._lock:
            return any(j.running for j in self._jobs.values())

    def reset(self):
        """Forget every job. For tests, and for nothing else."""
        self.stop_all()
        with self._lock:
            self._jobs.clear()


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


class DeployJob(Job):
    """
    The Deploy tab's job: ``deploy/raspi_infer.py`` on this machine, or
    ``deploy/run_on_pi.sh`` on the board, optionally preceded by an export.

    What it adds over a plain tail is the reading: the benchmark, the latency
    summaries, the duty-cycle rows and the test-set metrics are pulled out of
    the stream as they print, so a sweep that takes two minutes fills the tab
    in as it goes.  Those are a live preview — the same numbers land in
    ``deploy/artifacts/pi_latency/*.json``, which is what the page reads back
    once a trace exists.
    """

    KIND = "deploy"

    def _reset_state(self):
        self._scan_state = pi.LogScan()

    def _scan(self, line: str):
        self._scan_state.feed(line)

    def _extra_status(self) -> dict:
        return {"parsed": self._scan_state.state()}

    def parsed(self) -> dict:
        with self._lock:
            return self._scan_state.state()


TRAIN_JOBS = TrainingJobs()
EVAL_JOB = EvalJob()
DEPLOY_JOB = DeployJob()
#: The receptive-field sweep: a list of ordinary training runs, run in order.
#: Its own job rather than the training one, because a sweep is many runs and
#: Stop has to mean the sweep rather than whichever run is going.
SWEEP_JOB = DeployJob()


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


def _train_title(config_name: str, command: list[str]) -> str:
    """
    What to call a run on the switcher: the directory it will write.

    The save_dir is the run's identity everywhere else in this project -- it
    carries the variant, the artifact and the seed -- so it is the one label
    that tells apart runs of the same preset that differ in one flag, which is
    the case the switcher exists for.

    Naming it after the *preset* instead, which this did when the command
    carried no ``--save_dir``, labels every run from one file identically:
    six pills reading ``mla_rank_sweep_eog`` and no way to tell which is
    which. It also hid something worth seeing. A preset with no ``save_dir``
    has the trainer derive one from the variant, artifact and seed, so those
    six runs were not six experiments -- they were six runs deriving the same
    directory and overwriting each other. The derivation is repeated here,
    from the same three keys in the same order of precedence, so the label is
    the path the run will really write and a collision is visible before it
    happens rather than afterwards in the Runs tab.
    """
    def flag(name: str):
        return command[command.index(name) + 1] if name in command else None

    if flag("--save_dir"):
        return flag("--save_dir").rstrip("/").split("/")[-1]
    try:
        values = config_io.read_config(config_name).get("values") or {}
    except (ValueError, FileNotFoundError, OSError):
        values = {}
    if values.get("save_dir"):                 # the preset names one itself
        return str(values["save_dir"]).rstrip("/").split("/")[-1]

    variant = flag("--variant") or values.get("variant")
    artifact = flag("--artifact") or values.get("artifact")
    if not variant or not artifact:            # nothing to derive from
        return config_name[:-5] if config_name.endswith(".yaml") else config_name
    seed = flag("--master_seed")
    if seed is None:
        seed = values.get("master_seed")
    name = f"{variant}_{str(artifact).lower()}"
    if flag("--teacher_ckpt") or values.get("teacher_ckpt"):
        name += "_kd"                          # as trainer/__main__.py does
    # The default seed keeps the bare name, every other seed gets a suffix --
    # the rule the trainer and the Lookahead tab both follow.
    if seed is not None and int(seed) != DEFAULT_MASTER_SEED:
        name += f"_s{int(seed)}"
    return name


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
        # The other half of what this project did: the measurements made from
        # those runs. Cheap (a directory of small JSON files), so it rides
        # along with the training runs rather than needing its own refresh.
        "deploys": pi.list_deploy_runs(),
        "deploy_logs": pi.list_deploy_logs(),
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
        "pi": pi.meta(),
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
                                   "optimizers": config_io.runs_by_optimizer(runs),
                                   "deploys": pi.list_deploy_runs(),
                                   "deploy_logs": pi.list_deploy_logs()})
            if url.path == "/api/lookahead/meta":
                return self._json(lookahead.meta())
            if url.path == "/api/runs/kinds":
                return self._json({"kinds": config_io.runs_by_kind()})
            if url.path == "/api/runs/times":
                return self._json({"times": config_io.runs_by_time()})
            if url.path == "/api/runs/losses":
                return self._json({"losses": config_io.runs_by_loss()})
            if url.path == "/api/runs/optimizers":
                return self._json({"optimizers": config_io.runs_by_optimizer()})
            if url.path == "/api/runs/deploys":
                return self._json({"deploys": pi.list_deploy_runs(),
                                   "deploy_logs": pi.list_deploy_logs()})
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
                # One request serves the whole tab: the selected run's new
                # lines, and a small record per run for the switcher. Polling
                # each job separately would scale the request count with the
                # number of runs for no gain -- only one of them is on screen.
                job = TRAIN_JOBS.get(q.get("id"))
                status = job.status(int(q.get("since", 0))) if job else {
                    "running": False, "lines": [], "since": 0, "next": 0, "metrics": [],
                    "id": "", "title": "", "returncode": None, "command": "",
                    "steps": [], "step": 0, "started": None, "log_path": None,
                    "max_epochs": None, "kind": "train"}
                return self._json({**status, "jobs": TRAIN_JOBS.summaries(),
                                   "max_active": TrainingJobs.MAX_ACTIVE})
            if url.path == "/api/rf/meta":
                return self._json(rfsweep.meta())
            if url.path == "/api/rf/results":
                return self._json({"rows": rfsweep.results(
                    q.get("variant", "tcn"), q.get("artifact", "BOTH"),
                    int(q["seed"]) if q.get("seed") else None,
                    int(q.get("hidden_channels") or rfsweep.DEFAULT_HIDDEN))})
            if url.path == "/api/eegio/meta":
                return self._json(eegio_tab.meta(q.get("data_dir")))
            if url.path == "/api/kd/meta":
                return self._json(kd.meta())
            if url.path == "/api/kd/checks":
                if not q.get("student"):
                    return self._error("kd/checks needs a student run", 400)
                return self._json(kd.static_checks(q["student"], q.get("teacher") or None))
            if url.path == "/api/rf/status":
                return self._json(SWEEP_JOB.status(int(q.get("since", 0))))
            if url.path == "/api/pi/meta":
                return self._json(pi.meta())
            if url.path == "/api/pi/status":
                return self._json(DEPLOY_JOB.status(int(q.get("since", 0))))
            if url.path == "/api/pi/trace":
                return self._json(pi.read_trace(q.get("name", ""),
                                                int(q.get("max_points", 900))))
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
            if url.path == "/api/pi/testdata":
                # Cutting 100 epochs takes ~40 ms, so this answers inline
                # rather than going through the job machinery.
                out = pi.cut_testdata_for(body.get("model", ""),
                                          int(body.get("n") or 100),
                                          bool(body.get("overwrite")))
                out["testdata"] = pi.list_testdata()
                return self._json(out)
            if url.path == "/api/lookahead/measure":
                # Builds a model per rung and probes it; nothing is trained and
                # no checkpoint is read, so this is a plain request rather than
                # a job.
                return self._json(lookahead.measure(body))
            if url.path == "/api/lookahead/train":
                # Resolved here and handed to the ordinary training runner, so a
                # rung is launched by the same path as every other run rather
                # than by a second entry point that could drift from it.
                plan = lookahead.train_plan(body)
                runs = plan["runs"]
                if plan["taken"] and not body.get("overwrite"):
                    taken = ", ".join(plan["taken"])
                    return self._error(
                        f"{taken} already {'holds' if len(plan['taken']) == 1 else 'hold'} a run; "
                        f"the trainer's own guard would refuse it. Untick "
                        f"{'that seed' if len(plan['taken']) == 1 else 'those seeds'}, or tick "
                        f"overwrite.", 409)
                # Both checks run before anything starts, so the request is
                # either all launched or all refused.
                room = TRAIN_JOBS.room()
                if len(runs) > room:
                    return self._error(
                        f"{len(runs)} seeds is more than there is room for: "
                        f"{TRAIN_JOBS.MAX_ACTIVE - room} of {TRAIN_JOBS.MAX_ACTIVE} runs are "
                        f"already going. Start {room if room else 'fewer'} now, or wait — they "
                        f"share one machine and slow each other down.", 409)
                started = [TRAIN_JOBS.start(build_train_command(r["config"], r["extra"]),
                                            PROJECT_ROOT, RUNS_DIR,
                                            title=r["save_dir"].split("/")[-1])
                           for r in runs]
                # The first one's status, so the page opens on the run it will
                # watch; the rest are pills beside it.
                return self._json({**started[0].status(0), "planned": plan,
                                   "started": [j.id for j in started],
                                   "jobs": TRAIN_JOBS.summaries()})
            if url.path == "/api/lookahead/preview":
                return self._json(lookahead.train_plan(body))
            if url.path == "/api/train":
                cmd = build_train_command(body["name"], body.get("extra_args", ""))
                job = TRAIN_JOBS.start(cmd, PROJECT_ROOT, RUNS_DIR,
                                       title=_train_title(body["name"], cmd))
                return self._json({**job.status(0), "jobs": TRAIN_JOBS.summaries()})
            if url.path == "/api/train/stop":
                # No id stops the newest running run, which is what a page that
                # has only ever shown one means by "Stop".
                stopped = TRAIN_JOBS.stop(body.get("id"))
                job = TRAIN_JOBS.get(body.get("id"))
                return self._json({"stopped": stopped, **(job.status(0) if job else {}),
                                   "jobs": TRAIN_JOBS.summaries()})
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
            if url.path == "/api/eegio/evaluate":
                try:
                    return self._json(eegio_tab.evaluate(body))
                except (FileNotFoundError, ValueError) as e:
                    return self._error(str(e), 400)
            if url.path == "/api/kd/decompose":
                students = body.get("students") or []
                if not body.get("teacher") or not students:
                    return self._error("kd/decompose needs a teacher and at least one student", 400)
                return self._json(kd.decompose(
                    body["teacher"], students,
                    body.get("artifact") or "BOTH",
                    int(body.get("master_seed") or 42),
                    int(body.get("max_batches") or kd.DEFAULT_MAX_BATCHES)))
            if url.path == "/api/rf/preview":
                return self._json({"steps": pi.describe(rfsweep.build_steps(body))})
            if url.path == "/api/rf/run":
                if SWEEP_JOB.running:
                    return self._error("a sweep is already in progress", 409)
                steps = rfsweep.build_steps(body)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                SWEEP_JOB.start_steps(steps, PROJECT_ROOT, RUNS_DIR / f"gui_rfsweep_{stamp}.log")
                return self._json({**SWEEP_JOB.status(0), "planned": pi.describe(steps)})
            if url.path == "/api/rf/stop":
                return self._json({"stopped": SWEEP_JOB.stop(), **SWEEP_JOB.status(0)})
            if url.path == "/api/pi/preview":
                return self._json({"steps": pi.describe(pi.build_steps(body))})
            if url.path == "/api/pi/run":
                if DEPLOY_JOB.running:
                    return self._error("a deploy run is already in progress", 409)
                # The split check bites here, not on preview: this is the call
                # that goes on to produce a score.
                steps = pi.build_steps({**body, "_enforce_split": True})
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                DEPLOY_JOB.start_steps(steps, PROJECT_ROOT, RUNS_DIR / f"gui_deploy_{stamp}.log")
                return self._json({**DEPLOY_JOB.status(0), "planned": pi.describe(steps)})
            if url.path == "/api/pi/stop":
                return self._json({"stopped": DEPLOY_JOB.stop(), **DEPLOY_JOB.status(0)})
            return self._error("not found", 404)
        except (FileExistsError, TooManyJobs) as e:
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
        for job, what in ((JOB, "training"), (EVAL_JOB, "evaluation"), (DEPLOY_JOB, "deploy")):
            if job.running:
                print(f"[gui] stopping the {what} job ...")
                job.stop()
        server.server_close()
        time.sleep(0.1)
