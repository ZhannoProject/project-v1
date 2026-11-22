"""
The Deploy tab's half of the GUI: the runtime test of an exported model.

Three things live here, none of them HTTP:

``build_steps``   one submission of the tab -> the list of commands it runs.
                  A deploy run is a *sequence* -- optionally export the
                  checkpoint, then measure, then sweep the hop sizes -- because
                  ``raspi_infer.py`` returns early after ``--duty_cycle`` and
                  cannot both score the test set and sweep in one call, and
                  because an export has to finish before anything can be run.
``LogScan``       the numbers read back out of a run's output while it streams,
                  so the tab fills in live.  This is a *preview*: every figure
                  it parses is also written to a trace file, and the page
                  switches to the file as soon as one lands (see below).
``list_traces`` / ``read_trace``
                  the trace files themselves -- ``deploy/artifacts/pi_latency/
                  <stem>.json`` (the summary ``LatencyMonitor.save`` writes),
                  ``<stem>.csv`` (one row per hop) and ``<stem>_duty.json``
                  (one row per hop size).  These are the authoritative numbers
                  and they outlive the job, so a trace from last week reads
                  exactly like the one still running.

Nothing in this module measures anything itself: the measurement is
``deploy/raspi_infer.py``, on this machine or on the Pi via
``deploy/run_on_pi.sh``, and the tab is a front end for it in the same way the
Evaluate tab is a front end for ``eval.py``.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import re
import shlex
import shutil
import sys
from pathlib import Path
from typing import Optional

from deploy import preflight
from gui import config_io
from gui.config_io import PROJECT_ROOT, RUNS_DIR

#: The governor a reportable measurement is taken under.  Defined by the
#: checker, not here, so the form's default and the thing that judges it
#: cannot disagree.  (preflight is stdlib-only, like the rest of the GUI.)
WANTED_GOVERNOR = preflight.WANTED_GOVERNOR


def _bash_executable():
    if sys.platform != 'win32':
        return 'bash'
    for variable in ('ProgramW6432','ProgramFiles','ProgramFiles(x86)','LOCALAPPDATA'):
        base=os.environ.get(variable)
        if base:
            for suffix in ('Git/bin/bash.exe','Programs/Git/bin/bash.exe'):
                candidate=Path(base)/suffix
                if candidate.is_file():return str(candidate)
    found=shutil.which('bash')
    if found and 'system32' not in found.lower():return found
    raise ValueError('The Deploy Pi script requires Git for Windows (Git Bash). '
                     'Install it, or use EEG channels with native Windows OpenSSH.')

DEPLOY_DIR = PROJECT_ROOT / "deploy"
ARTIFACTS_DIR = DEPLOY_DIR / "artifacts"
TESTDATA_DIR = DEPLOY_DIR / "testdata"

#: Where ``--latency_out`` traces are collected.  ``run_on_pi.sh`` already
#: scp's the Pi's traces into this directory, so host runs write here too (with
#: a ``host_`` prefix, so a measurement of this laptop can never be mistaken
#: for one of the board).
TRACE_DIR = ARTIFACTS_DIR / "pi_latency"

RUNNER = DEPLOY_DIR / "run_on_pi.sh"

#: ``[user@]host`` for ssh and scp.  Deliberately narrow: this string is
#: handed to ``run_on_pi.sh`` as an argument (no shell is involved, so there is
#: nothing to inject), but a value starting with ``-`` would still arrive at
#: ssh as a flag, and a stray space would silently split into two arguments.
#:
#: ``:``, ``%`` and brackets are allowed because that is how a Pi on a direct
#: Ethernet link is reachable: mDNS gives it an IPv6 link-local address and no
#: IPv4 at all, and a link-local address needs the interface after a ``%``
#: (``changlong@fe80::8aa2:9eff:fe56:5033%en5``).  Without them the tab could
#: not drive that setup at all.
SSH_TARGET_RE = re.compile(r"^[A-Za-z0-9_.\[][A-Za-z0-9_.:%\[\]@-]*$")

#: An export stem: the name of a file in ``deploy/artifacts/``, never a path.
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

#: A CPU list for ``--cpus``: "0-1", "0,2", "0-3".  Narrow on purpose — it is
#: passed to a subprocess argument, and nothing else belongs in it.
CPU_LIST_RE = re.compile(r"^\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*$")

MAX_THREADS = 16
MAX_HOP = 4096
MAX_SWEEP = 12
MAX_DUTY_SECONDS = 600


# ============================================================
# What is on disk
# ============================================================
def _sidecar(path: Path) -> dict:
    """The ``.json`` written next to an export, or {} if it is missing."""
    try:
        data = json.loads(path.with_suffix(".json").read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def list_exports() -> list[dict]:
    """
    One row per ``deploy/artifacts/*.onnx``.

    The sidecar is what makes the row worth showing: the architecture the
    weights were inferred to have, and the verification the exporter refused
    to skip -- agreement with PyTorch and the causality check.  An export
    without one still lists (it can be run), with ``sidecar: false``.
    """
    rows = []
    if not ARTIFACTS_DIR.is_dir():
        return rows
    for onnx in sorted(ARTIFACTS_DIR.glob("*.onnx")):
        meta = _sidecar(onnx)
        model, pre = meta.get("model") or {}, meta.get("preprocessing") or {}
        ver, exp = meta.get("verification") or {}, meta.get("export") or {}
        rows.append({
            "name": onnx.stem,
            "path": f"deploy/artifacts/{onnx.name}",
            "size_kb": round(onnx.stat().st_size / 1024, 1),
            "sidecar": bool(meta),
            "variant": model.get("variant"),
            "params": model.get("n_params"),
            "hidden_channels": model.get("hidden_channels"),
            "n_attn_layers": model.get("n_attn_layers"),
            "kv_lora_rank": model.get("kv_lora_rank"),
            "n_heads": model.get("n_heads"),
            "checkpoint": model.get("checkpoint"),
            "window": pre.get("window_samples") or exp.get("seq_len"),
            "sample_rate_hz": pre.get("sample_rate_hz"),
            "artifact": pre.get("artifact_trained_on"),
            "causal": pre.get("causal"),
            "max_abs_diff_vs_torch": ver.get("max_abs_diff_vs_torch"),
            "causality_past_diff": ver.get("causality_past_diff"),
            # A step graph's sidecar records this instead of the two above.
            "max_abs_diff_vs_incremental": ver.get("max_abs_diff_vs_incremental"),
            "n_steps": ver.get("n_steps"),
            "lookahead_samples": pre.get("lookahead_samples") or 0,
            "future_wait_ms": pre.get("future_wait_ms"),
            "observed_future_reach": ver.get("observed_future_reach"),
            "host_latency_ms_median": ver.get("host_latency_ms_median"),
            "test_metrics": ver.get("test_metrics"),
        })
    return rows


def list_models() -> list[dict]:
    """
    Every model the tab can run: the exports, and the trained runs they come
    from, merged on the name they share.

    The two are the same thing at different stages — an export is named after
    its run — but only two of this project's twenty-nine runs have ever been
    exported, and a list of ``deploy/artifacts/*.onnx`` therefore offered
    almost nothing.  A run that has not been exported yet is still a model you
    can deploy; it just needs the export step first, which is what ``needs``
    says here.

      exported   there is a .onnx (and a sidecar) to run
      stale      the run's best.pt is newer than the export beside it
      causal     the exporter refuses a *_bi twin: it needs the future
    """
    # Step graphs belong to their checkpoint row, not a second model named *_step.
    exports = {e["name"]: e for e in list_exports() if not e["name"].endswith("_step")}
    try:
        runs = {r["name"]: r for r in config_io.list_runs()}
    except Exception:                          # torch missing: exports still list
        runs = {}
    rows = []
    for name in sorted(set(exports) | set(runs)):
        e, r = exports.get(name), runs.get(name)
        row = {"name": name, "exported": e is not None, "run": r is not None,
               "variant": (e or {}).get("variant") or (r or {}).get("variant"),
               # The run's own record wins: a sidecar's artifact used to come from
               # the exporter's --artifact flag, so an old export can still claim
               # the wrong one. The run reads it from the training config.
               "artifact": (r or {}).get("artifact") or (e or {}).get("artifact"),
               "params": (e or {}).get("params") or (r or {}).get("params"),
               "bidirectional": bool((r or {}).get("bidirectional")),
               # A bounded-lookahead rung reads L samples that have not arrived
               # yet, so it runs online with each output emitted L samples late.
               # The exporter verifies that contract and the windowed replay
               # honours it; only the one-sample step graph is causal-only.
               "lookahead": (r or {}).get("lookahead") or (e or {}).get("lookahead_samples") or None,
               # The split this model was fitted on, for the test-set pairing
               # check. None for runs trained before the seed was recorded.
               "master_seed": (r or {}).get("master_seed"),
               # The rate the model runs at; a test set must match it.
               "train_fs": (r or {}).get("train_fs") or (e or {}).get("sample_rate_hz"),
               # The one-sample graph is a separate artifact from the windowed
               # one, and the Pi needs it to stream: the page has to be able
               # to say whether this model has one yet.
               "step_exported": (ARTIFACTS_DIR / f"{name}_step.onnx").is_file(),
               "stale": False}
        if e and r:
            ckpt = RUNS_DIR / name / "best.pt"
            onnx = ARTIFACTS_DIR / f"{name}.onnx"
            try:
                row["stale"] = ckpt.stat().st_mtime > onnx.stat().st_mtime + 1
            except OSError:
                pass
        row["needs"] = ("none" if row["exported"] and not row["stale"]
                        else "re-export" if row["exported"] else "export")
        # The gate is "can the exporter take this": false only for an
        # unbounded teacher. A bounded rung exports with its L declared.
        row["exportable"] = not row["bidirectional"]
        row["deployable"] = row["exported"] or (row["run"] and row["exportable"])
        rows.append(row)
    return rows


def list_testdata() -> list[dict]:
    """
    One row per ``deploy/testdata/*.npz`` -- the scored sets, and the signal
    the duty-cycle sweep is replayed from.

    Each row carries the split it was cut from, so the page can refuse a pair
    before a Pi run rather than after one.  ``raspi_infer`` checks the same
    thing again at scoring time; this is the earlier, friendlier half of a
    check that has to exist on the board regardless.
    """
    if not TESTDATA_DIR.is_dir():
        return []
    rows = []
    for p in sorted(TESTDATA_DIR.glob("*.npz")):
        row = {"name": p.name, "size_kb": round(p.stat().st_size / 1024, 1)}
        try:
            from deploy import raspi_infer          # numpy + onnxruntime; imported lazily
            row.update(raspi_infer.testdata_provenance(p))
        except Exception:
            # numpy missing, or a file that is not a readable npz: the tab still
            # lists it and the scorer will say what is wrong with it.
            row.update({"source": "unknown", "artifact": None,
                        "master_seed": None, "split": None})
        rows.append(row)
    return rows


def cut_testdata_for(model_name: str, n: int = 100, overwrite: bool = False) -> dict:
    """
    Cut the test set that matches a model's split, from the model.

    The pairing rules make picking a set a small research question -- which
    seed was this run trained at, which artifact, does a matching file exist
    yet -- and the answer is already recorded in the run.  So rather than
    report a mismatch and leave the operator to translate it into a
    ``make_testdata`` command, this reads the artifact and ``master_seed`` off
    the model and cuts the right file.

    A run whose seed was never recorded is refused rather than cut at the
    default: guessing 42 would produce a file that *looks* matched, which is
    the failure this whole check exists to prevent.

    Returns ``{"name", "created", "artifact", "master_seed", "n"}``.  An
    existing matching file is returned with ``created: False`` instead of
    being rewritten -- it is already the right epochs, and rewriting it would
    churn a file that recorded measurements may refer to.
    """
    model = next((m for m in list_models() if m["name"] == model_name), None)
    if model is None:
        raise ValueError(f"no model called {model_name!r}")

    seed, artifact = model.get("master_seed"), model.get("artifact")
    if seed is None:
        raise ValueError(
            f"{model_name} does not record the seed it was trained at (the run predates "
            f"master_seed being saved), so the matching split cannot be worked out. Cut one "
            f"by hand with python -m deploy.make_testdata --artifact <EOG|EMG|BOTH> "
            f"--master_seed <seed> once you know it.")
    if artifact not in ("EOG", "EMG", "BOTH"):
        raise ValueError(f"{model_name} does not record which artifact it was trained on")
    if not 10 <= int(n) <= 2000:
        raise ValueError(f"epochs must be between 10 and 2000, got {n}")

    from deploy.make_testdata import make_testdata

    n = int(n)
    # The sample rate is part of the protocol too: a 250 Hz model refuses a
    # 256 Hz set on the board. Cutting from the checkpoint restores the whole
    # recorded protocol (rate, sampling, resampling), and the name carries the
    # rate the same way make_testdata writes it.
    ckpt = RUNS_DIR / model_name / "best.pt"
    fs = model.get("train_fs")
    stem = f"{artifact}_test{n}" + ("" if int(seed) == 42 else f"_s{int(seed)}")
    if fs and float(fs) != 256.0:
        stem += f"_{float(fs):g}hz"
    target = TESTDATA_DIR / f"{stem}.npz"
    if target.is_file() and not overwrite:
        return {"name": target.name, "created": False, "artifact": artifact,
                "master_seed": int(seed), "n": n}

    data_dir = PROJECT_ROOT / "data" / "EEGdenoisenet" / "data"
    if not (data_dir / "EEG_all_epochs.npy").is_file():
        raise FileNotFoundError(
            f"the EEGdenoiseNet .npy files are not in {data_dir} — see the Data section of "
            f"the README; a test set is cut from them, not shipped with the code")

    # make_testdata prints a summary line; the caller wants the result, and the
    # page has nowhere to show a stdout line.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        if ckpt.is_file():
            path = make_testdata(str(data_dir), artifact, n, out_dir=TESTDATA_DIR,
                                 master_seed=int(seed), checkpoint=str(ckpt))
        else:
            path = make_testdata(str(data_dir), artifact, n, out_dir=TESTDATA_DIR,
                                 master_seed=int(seed),
                                 target_fs=float(fs) if fs else None)
    return {"name": Path(path).name, "created": True, "artifact": artifact,
            "master_seed": int(seed), "n": n, "log": buf.getvalue().strip()}


def split_mismatch(model_name: str, testdata_name: str) -> Optional[dict]:
    """
    Why this model and this test set should not be scored together, if so.

    "Test split" is only meaningful relative to a seed: EEGDenoiseDataset
    permutes the epoch pool with ``master_seed`` before slicing it, so the
    seed-42 test epochs are 69% of them in the seed-67 *training* split.
    Scoring across that line returns an inflated number with nothing visibly
    wrong, which is why it is caught here and again on the board.

    Returns ``None`` when the pair is fine, else ``{"level", "text"}`` --
    ``error`` for a seed mismatch (the score would be meaningless), ``warn``
    for a pair that cannot be verified or is a deliberate cross-artifact probe.
    """
    sets = {t["name"]: t for t in list_testdata()}
    t = sets.get(testdata_name)
    if t is None:
        return None
    model = next((m for m in list_models() if m["name"] == model_name), None)
    if model is None:
        return None

    seed, artifact = model.get("master_seed"), model.get("artifact")
    if t.get("split") not in (None, "test"):
        return {"level": "error",
                "text": f"{testdata_name} is the '{t['split']}' split, not the test split"}
    if seed is None or t.get("master_seed") is None:
        why = ("this run predates the seed being recorded"
               if seed is None else f"{testdata_name} carries no recorded provenance")
        return {"level": "warn",
                "text": f"cannot verify the split — {why}. If the seeds differ these epochs "
                        f"are {model_name}'s training data and the score means nothing."}
    if int(seed) != int(t["master_seed"]):
        return {"level": "error",
                "text": f"{model_name} was trained at master_seed {seed} but {testdata_name} was "
                        f"cut from the master_seed {t['master_seed']} split — most of these epochs "
                        f"are in that model's training set. Cut a matching set: "
                        f"python -m deploy.make_testdata --artifact {t.get('artifact') or artifact} "
                        f"--master_seed {seed}"}
    fs_model, fs_set = model.get("train_fs"), t.get("fs")
    if fs_model and fs_set and abs(float(fs_model) - float(fs_set)) > 1e-6:
        return {"level": "error",
                "text": f"{model_name} runs at {float(fs_model):g} Hz but {testdata_name} holds "
                        f"{float(fs_set):g} Hz epochs — the board refuses this pair. Cut a matching set: "
                        f"python -m deploy.make_testdata --ckpt runs/{model_name}/best.pt"}
    if t.get("artifact") and artifact and t["artifact"] != artifact:
        return {"level": "warn",
                "text": f"{model_name} was trained on {artifact} but {testdata_name} is "
                        f"{t['artifact']} — a cross-artifact probe, not a like-for-like score"}
    if t.get("source") == "filename":
        return {"level": "warn",
                "text": f"{testdata_name} has no recorded provenance; its split was read off the "
                        f"filename. Re-cut it to stamp the provenance in."}
    return None


def trace_kind(name: str) -> str:
    """``duty`` for a hop sweep, ``latency`` for a per-hop trace."""
    return "duty" if name.endswith("_duty.json") else "latency"


def list_traces() -> list[dict]:
    """
    Every trace in ``deploy/artifacts/pi_latency/``, newest first, with just
    enough of its contents to choose between them: the hop it measured and the
    headline number, so the list reads as a history of measurements rather
    than a directory listing.
    """
    rows = []
    if not TRACE_DIR.is_dir():
        return rows
    for js in TRACE_DIR.glob("*.json"):
        try:
            data = json.loads(js.read_text())
        except (OSError, ValueError):
            continue
        kind = trace_kind(js.name)
        row = {"name": js.name, "kind": kind, "mtime": js.stat().st_mtime,
               "csv": (js.with_suffix(".csv")).is_file(),
               "host": js.name.startswith("host_")}
        if kind == "duty":
            sweep = data if isinstance(data, list) else (data or {}).get("rows") or []
            ok = [r for r in sweep if r.get("verdict") == "OK"]
            row.update(hops=[r.get("hop") for r in sweep],
                       best_hop=min((r["hop"] for r in ok), default=None),
                       n_rows=len(sweep),
                       environment=None if isinstance(data, list) else (data or {}).get("environment"))
        elif isinstance(data, dict):
            row.update(hop=data.get("hop"), n_hops=data.get("n_hops"),
                       sample_rate_hz=data.get("sample_rate_hz"),
                       median_ms=(data.get("total_ms") or {}).get("median"),
                       deadline_misses=data.get("deadline_misses"),
                       realtime=("e2e_latency_ms" in data),
                       environment=data.get("environment"),
                       scores=data.get("scores"),
                       inference=data.get("inference"))
        # Peak resident memory of the process that took the trace; absent (None)
        # on traces written before raspi_infer.py recorded it.
        if isinstance(data, dict):
            row["peak_rss_mib"] = (data.get("memory") or {}).get("peak_rss_mib")
        # The verdict the dropdown shows: a trace from before this was recorded
        # says "unknown", which is not the same as "unlocked" and must not read
        # as it.
        row["verdict"] = ((row.get("environment") or {}).get("verdict")
                          if row.get("environment") else None)
        rows.append(row)
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows


#: ``{host|latency}_<model>_hop<N>[_duty|_realtime].json``.  The model name is
#: a run directory name and carries underscores and digits of its own
#: (``tcn_mla_both_s67``), so the match is anchored on the ``_hop<N>`` at the
#: end rather than on the first underscore.
_TRACE_NAME = re.compile(
    r"^(?P<where>host|latency|incremental)_(?P<model>.+)_hop(?P<hop>\d+)"
    r"(?P<extra>_duty|_realtime)?\.json$")

#: Sampling rate the epochs were cut at, for turning a hop into a deadline.
TRACE_FS = 256.0


def _scores_from_logs() -> dict:
    """
    ``{trace name: scores}`` recovered from the Deploy tab's job logs.

    Accuracy used to be printed and dropped: ``evaluate_testdata`` returned
    it, ``main`` ignored the return, and only the timings reached a file.  The
    trace now carries it, but the measurements already on disk predate that,
    and their numbers are not lost -- they are in the job log that produced
    them, which also names the traces that run wrote.  That is the join.

    Newest log wins, because a trace is overwritten by whichever run wrote it
    last and the newest log is the one that describes the file now on disk.
    Scraped text, so it is marked ``log`` rather than ``trace`` and the page
    says which.
    """
    out: dict[str, dict] = {}
    for log in list_deploy_logs():                 # newest first
        path = RUNS_DIR / log["name"]
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        scan = LogScan()
        for line in text.splitlines():
            scan.feed(line)
        state = scan.state()
        test = state.get("test") or {}
        rows = {r["name"]: r for r in (test.get("rows") or [])}
        if not rows:
            continue
        scores = {
            "testdata": test.get("file"),
            "n_epochs": test.get("n_epochs"),
            "modes": {name: {k: r[k] for k in ("rrmse_t", "rrmse_s", "acc") if r.get(k) is not None}
                      for name, r in rows.items()},
            "source": "log",
            "from_log": log["name"],
        }
        for name in state.get("traces") or []:
            out.setdefault(name, scores)           # first = newest
    return out


def list_deploy_runs() -> list[dict]:
    """
    Deploy runs for the Runs tab: one row per measurement, newest first.

    The Runs tab is this project's history of what was actually done, and
    until now it held only half of it -- every training run, and none of the
    measurements made from them.  A deploy run is the other half: a model, a
    machine, a hop, and the latency it achieved there.

    A measurement is keyed by ``(where, model, hop)`` and gathers the three
    traces a single run can write -- the per-hop latency trace, the duty-cycle
    sweep beside it, and the real-time replay -- because they are one run's
    output, not three runs.

    Every row carries the environment verdict, which is the point: a latency
    figure is a statement about a model *and* about the machine, and a row
    that cannot say which is marked ``unknown`` rather than passing silently.
    ``host`` runs are this laptop and are never the measurement -- macOS has
    no governor and no affinity control -- so they are labelled, not hidden.
    """
    groups: dict[tuple, dict] = {}
    for t in list_traces():
        m = _TRACE_NAME.match(t["name"])
        if not m:
            continue
        # The filename prefix names the method for an incremental trace and
        # the machine for the other two, which is not enough on its own: a
        # step graph runs on the board as well as here. The environment block
        # the run recorded is authoritative, because it read the machine it
        # was actually on; the prefix is only the fallback for traces written
        # before that existed.
        prefix = m.group("where")
        board = ((t.get("environment") or {}).get("board") or "")
        if board:
            where = "pi" if "raspberry" in board.lower() else "host"
        else:
            where = "pi" if prefix == "latency" else "host"
        hop = int(m.group("hop"))
        key = (where, m.group("model"), hop)
        row = groups.setdefault(key, {
            "model": m.group("model"), "where": where, "hop": hop,
            # Set below from the trace's own sample rate (the model's, for
            # traces too old to record it): hop 4 is 16 ms at 250 Hz, not the
            # 15.62 ms a fixed 256 Hz made of it.
            "budget_ms": None, "sample_rate_hz": None,
            "mtime": 0.0, "traces": [], "duty": None, "realtime": False,
            "median_ms": None, "n_hops": None, "deadline_misses": None,
            "verdict": None, "csv": False, "scores": None,
            # Filled from the trace's own inference block; "windowed" is the
            # fallback for traces written before it was recorded, which is
            # what every one of them was.
            "mode": "incremental" if prefix == "incremental" else "windowed",
            "inference": None, "peak_rss_mib": None,
        })
        row["mtime"] = max(row["mtime"], t["mtime"])
        row["sample_rate_hz"] = row["sample_rate_hz"] or t.get("sample_rate_hz")
        row["traces"].append(t["name"])
        if t.get("scores") and not row.get("scores"):
            row["scores"] = {**t["scores"], "source": "trace"}
        if t.get("inference") and not row.get("inference"):
            row["inference"] = t["inference"]
            row["mode"] = t["inference"].get("mode", row["mode"])
        row["csv"] = row["csv"] or bool(t.get("csv"))
        # One process takes all of a run's traces, so the largest is the run's peak.
        if t.get("peak_rss_mib") is not None:
            row["peak_rss_mib"] = max(row["peak_rss_mib"] or 0, t["peak_rss_mib"])
        # A verdict from any of a run's traces describes the same run; prefer
        # one that exists over the None of an older sibling.
        row["verdict"] = row["verdict"] or t.get("verdict")

        if m.group("extra") == "_duty":
            if prefix == "incremental":
                # A step graph has no hop to sweep, so a duty trace under its
                # stem is not part of this measurement: it is a leftover from
                # before the combination was refused, or from a run that used
                # the same stem earlier. Merging it put "duty sweep · safe
                # from hop 16" on rows whose run never swept anything.
                continue
            row["duty"] = {"best_hop": t.get("best_hop"), "n_rows": t.get("n_rows"),
                           "hops": t.get("hops")}
        elif m.group("extra") == "_realtime":
            row["realtime"] = True
        else:
            row.update(median_ms=t.get("median_ms"), n_hops=t.get("n_hops"),
                       deadline_misses=t.get("deadline_misses"))
            # The per-hop trace is the measurement; its duty and realtime
            # siblings are extras taken during the same run. ``traces`` is in
            # whatever order the directory listing gave, so name the one a
            # reader should be shown rather than leaving it to position.
            row["primary"] = t["name"]

    # Accuracy for the rows whose traces predate it being saved.
    from_logs = _scores_from_logs()
    for row in groups.values():
        if row.get("scores"):
            continue
        for name in row["traces"]:
            if name in from_logs:
                row["scores"] = from_logs[name]
                break

    # What the model is, so the row reads as a measurement of something rather
    # than of a filename.
    models = {x["name"]: x for x in list_models()}
    for row in groups.values():
        mod = models.get(row["model"]) or {}
        row.setdefault("primary", row["traces"][0] if row["traces"] else None)
        row.update(variant=mod.get("variant"), artifact=mod.get("artifact"),
                   params=mod.get("params"), master_seed=mod.get("master_seed"),
                   known=bool(mod))
        fs = float(row["sample_rate_hz"] or mod.get("train_fs") or TRACE_FS)
        row["sample_rate_hz"] = fs
        # 2dp, not 1: hop 8 is 31.25 ms at 256 Hz, and 31.2 would be wrong.
        row["budget_ms"] = round(row["hop"] / fs * 1000, 2)
        if row["median_ms"] is not None and row["budget_ms"]:
            row["duty_cycle"] = round(row["median_ms"] / row["budget_ms"], 4)
            row["realtime_ok"] = row["median_ms"] < row["budget_ms"]
        else:
            row["duty_cycle"] = None
            row["realtime_ok"] = None

    return sorted(groups.values(), key=lambda r: r["mtime"], reverse=True)


def list_deploy_logs() -> list[dict]:
    """
    The Deploy tab's job logs, newest first.

    A run that failed before it measured anything leaves one of these and no
    trace, so counting them is how the Runs tab can say that a deploy was
    attempted more often than it succeeded.
    """
    if not RUNS_DIR.is_dir():
        return []
    out = []
    for f in sorted(RUNS_DIR.glob("gui_deploy_*.log")):
        try:
            st = f.stat()
        except OSError:
            continue
        out.append({"name": f.name, "mtime": st.st_mtime,
                    "size_kb": round(st.st_size / 1024, 1)})
    return sorted(out, key=lambda r: r["mtime"], reverse=True)


def trace_path(name: str) -> Path:
    """
    Resolve a client-supplied trace name inside ``pi_latency/``, or refuse it.

    Same rule as the Evaluate tab's figures: resolve first, check afterwards,
    which is what makes ``../`` and a symlink out of the tree fail rather than
    escape.  The GUI has no authentication, so this is the only place a
    request gets to name a file.
    """
    if not name:
        raise ValueError("no trace named")
    target = (TRACE_DIR / name).resolve()
    root = TRACE_DIR.resolve()
    if root not in target.parents:
        raise ValueError(f"{name} is outside deploy/artifacts/pi_latency/")
    if not target.is_file():
        raise FileNotFoundError(f"no such trace: {name}")
    return target


def _series(csv_path: Path, max_points: int) -> dict:
    """
    The per-hop columns of a trace's CSV, bucketed down to ``max_points``.

    A 20-minute run at hop 8 is 38k hops; sending them all would be a megabyte
    of JSON to draw 900 pixels.  Each bucket reports both its median and its
    **maximum**, because the whole reason to look at this curve is the
    outliers -- a thermal-throttling step or a single 16 ms stall -- and a
    plain decimation is exactly what hides them.
    """
    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {"points": [], "n_hops": 0, "bucket": 1}
    cols = [c for c in ("total_ms", "infer_ms", "e2e_latency_ms", "temp_c") if c in rows[0]]

    def num(row, key):
        try:
            v = float(row[key])
        except (TypeError, ValueError):
            return None
        return None if v != v else v                      # drop NaN (no thermal zone)

    bucket = max(1, -(-len(rows) // max(1, max_points)))  # ceil
    points = []
    for start in range(0, len(rows), bucket):
        chunk = rows[start:start + bucket]
        p = {"hop": int(float(chunk[-1]["hop"])), "n": len(chunk)}
        for c in cols:
            vals = sorted(v for v in (num(r, c) for r in chunk) if v is not None)
            if not vals:
                continue
            p[c] = round(vals[len(vals) // 2], 4)
            if c != "temp_c":
                p[c + "_max"] = round(vals[-1], 4)
        points.append(p)
    return {"points": points, "n_hops": len(rows), "bucket": bucket, "columns": cols}


def read_trace(name: str, max_points: int = 900) -> dict:
    """One trace, ready to draw: the summary it holds plus its per-hop series."""
    path = trace_path(name)
    data = json.loads(path.read_text())
    kind = trace_kind(path.name)
    out = {"name": path.name, "kind": kind, "mtime": path.stat().st_mtime,
           "path": f"deploy/artifacts/pi_latency/{path.name}"}
    if kind == "duty":
        # Sweeps written before the environment was recorded are a bare list;
        # since then they are {"rows": [...], "environment": {...}}.
        if isinstance(data, list):
            out["rows"] = data
        elif isinstance(data, dict) and isinstance(data.get("rows"), list):
            out["rows"] = data["rows"]
            out["environment"] = data.get("environment")
            out["memory"] = data.get("memory")
        else:
            raise ValueError(f"{name} does not hold duty-cycle rows")
        return out
    if not isinstance(data, dict):
        raise ValueError(f"{name} does not hold a latency summary")
    out["summary"] = data
    out["environment"] = data.get("environment")
    csv_path = path.with_suffix(".csv")
    if csv_path.is_file():
        out["series"] = _series(csv_path, max_points)
    return out


def host_environment() -> dict | None:
    """
    The state of the machine the GUI itself runs on, read on every refresh.

    Cheap (a handful of sysfs reads) and worth having before anything starts:
    it is what lets the tab say "this host cannot be locked" instead of
    offering controls that will not work.
    """
    try:
        return preflight.check()
    except Exception:                      # never let a broken read hide the tab
        return None


def meta() -> dict:
    """Everything the tab needs to draw itself before anything has been run."""
    exports = list_exports()
    models = list_models()
    ready = [m for m in models if m["exported"]]
    return {
        "exports": exports,
        # Every run, not only the ones already exported: the tab's model field
        # offers all of them and arranges the export when one is missing.
        "models": models,
        "testdata": list_testdata(),
        "traces": list_traces(),
        "runner": RUNNER.is_file(),
        "defaults": {
            "model": ready[0]["name"] if ready else None,
            "testdata": "EOG_test100.npz",
            "threads": 2, "hop": 32, "latency_every": 400,
            "duty_hops": "64,32,16,8,4", "duty_seconds": 20, "realtime_s": 30,
            "remote_dir": "eeg",
            # The top cores of a 4-core Pi 5, not the bottom ones: CPU 0 takes
            # the device interrupts and the timer tick, and a forward pass
            # sharing it gets a tail the model did not cause.  Neither control
            # exists on a macOS host, which is what the check then says.
            "cpus": "2-3", "governor": WANTED_GOVERNOR, "require_locked": True,
            # onnxruntime's own default; kept so the figures already measured
            # stay comparable, and recorded either way.
            "spin": True,
        },
        # What this machine can actually hold still, so the form can say so
        # before anything is run.
        "host_environment": host_environment(),
        "export_hint": "python -m deploy.export_onnx --ckpt runs/<run>/best.pt",
    }


# ============================================================
# Commands
# ============================================================
def _int(spec: dict, key: str, default: int, lo: int, hi: int) -> int:
    v = spec.get(key)
    if v in (None, ""):
        return default
    try:
        v = int(v)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a whole number, got {spec[key]!r}") from None
    if not lo <= v <= hi:
        raise ValueError(f"{key} must be between {lo} and {hi}, got {v}")
    return v


def _float(spec: dict, key: str, default: float, lo: float, hi: float) -> float:
    v = spec.get(key)
    if v in (None, ""):
        return default
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number, got {spec[key]!r}") from None
    if not lo <= v <= hi:
        raise ValueError(f"{key} must be between {lo} and {hi}, got {v}")
    return v


def _model_name(spec: dict) -> str:
    name = (spec.get("model") or "").strip()
    if not name:
        raise ValueError("pick an exported model")
    if not NAME_RE.match(name) or name.endswith(".onnx"):
        raise ValueError(f"not an export name: {name!r}")
    return name


def _testdata_name(spec: dict) -> str:
    name = (spec.get("testdata") or "").strip()
    if not name:
        raise ValueError("pick a test set — it is the signal the run is measured on")
    if not NAME_RE.match(name):
        raise ValueError(f"not a test-set name: {name!r}")
    if not (TESTDATA_DIR / name).is_file():
        raise FileNotFoundError(
            f"deploy/testdata/{name} does not exist — cut one with "
            "python -m deploy.make_testdata --artifact EOG --n 100")

    # Checked here as well as in the page, because the page can be stale and
    # the API can be called directly. raspi_infer refuses the same pair again
    # on the board; this just saves a round trip to the Pi to find out.
    #
    # Only on the path that actually produces a number. A preview is read-only,
    # and refusing it would replace the command text with an error the banner
    # above it is already showing -- and hide the command from someone who
    # means to copy it and add --allow-split-mismatch.
    if spec.get("_enforce_split") and not spec.get("allow_split_mismatch"):
        bad = split_mismatch((spec.get("model") or "").strip(), name)
        if bad and bad["level"] == "error":
            raise ValueError(bad["text"])
    return name


def _cpu_list(spec: dict, threads: int) -> str:
    """
    The cores to pin to.  Empty means "do not pin", which is a choice the form
    can make but not one a reportable measurement can.
    """
    raw = str(spec.get("cpus") or "").replace(" ", "")
    if not raw:
        return ""
    if not CPU_LIST_RE.match(raw):
        raise ValueError(f"not a CPU list: {raw!r} — expected something like 0-1 or 0,2")
    count = 0
    for part in raw.split(","):
        lo, _, hi = part.partition("-")
        lo, hi = int(lo), int(hi or lo)
        if hi < lo:
            raise ValueError(f"reversed CPU range: {part!r}")
        count += hi - lo + 1
    if count < threads:
        raise ValueError(f"{count} CPU(s) pinned for {threads} threads — "
                         "they would contend for the same core")
    return raw


def _hop_list(spec: dict) -> str:
    raw = str(spec.get("duty_hops") or "").replace(" ", "")
    hops = [h for h in raw.split(",") if h]
    if not hops:
        raise ValueError("the duty-cycle sweep needs at least one hop size")
    if len(hops) > MAX_SWEEP:
        raise ValueError(f"at most {MAX_SWEEP} hop sizes in one sweep, got {len(hops)}")
    out = []
    for h in hops:
        try:
            v = int(h)
        except ValueError:
            raise ValueError(f"hop sizes must be whole numbers, got {h!r}") from None
        if not 1 <= v <= MAX_HOP:
            raise ValueError(f"hop size out of range: {v}")
        out.append(v)
    return ",".join(str(v) for v in out)


def _export_step(spec: dict) -> dict:
    """
    ``python -m deploy.export_onnx`` for the run the tab picked.

    The artifact is named after its run, which is the exporter's own default,
    so the model the rest of the steps then test is the one just written —
    and so one name identifies the thing at both stages.
    """
    run = (spec.get("export_run") or spec.get("model") or "").strip()
    if not run:
        raise ValueError("pick a model")
    if not NAME_RE.match(run):
        raise ValueError(f"invalid run name: {run!r}")
    ckpt = RUNS_DIR / run / "best.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"runs/{run} has no best.pt — nothing to export")
    # Said here rather than two minutes into the export: a model with an
    # unbounded future is refused by the exporter. A bounded rung is not.
    row = next((m for m in list_models() if m["name"] == run), None)
    blocked = row if row and not row.get("exportable", not row["bidirectional"]) else None
    if row and row.get("lookahead") and spec.get("incremental"):
        fs = row.get("train_fs") or 256.0
        raise ValueError(
            f"{run} reads {row['lookahead']} samples ahead ({1000 * row['lookahead'] / fs:.1f} ms "
            f"at {fs:g} Hz); the one-sample step graph is causal-only. Untick incremental "
            f"inference: the windowed path emits each output once its future samples arrive")
    if blocked:
        raise ValueError(f"{run} is non-causal (its attention or its convolutions see "
                         "ahead) — the ONNX exporter refuses it, and it could not be "
                         "run online anyway")
    name = run
    cmd = [sys.executable, "-m", "deploy.export_onnx",
           "--ckpt", f"runs/{run}/best.pt", "--name", name]
    heads = spec.get("export_n_heads")
    if heads not in (None, ""):
        cmd += ["--n_heads", str(_int(spec, "export_n_heads", 4, 1, 64))]
    data_dir = (spec.get("export_data_dir") or "").strip()
    if data_dir:
        cmd += ["--data_dir", data_dir, "--artifact", spec.get("export_artifact") or "EOG"]
        if spec.get("export_max_batches") not in (None, ""):
            cmd += ["--max_batches", str(_int(spec, "export_max_batches", 4, 1, 10_000))]
    return {"label": f"export runs/{run}/best.pt → deploy/artifacts/{name}.onnx",
            "command": cmd, "name": name}


def build_steps(spec: dict) -> list[dict]:
    """
    One Deploy-tab submission -> the commands it will run, in order.

    Every path and number the request names is validated here rather than at
    the shell, and the result is what both the preview and the job use, so the
    grey line under the form cannot drift from what actually runs.
    """
    target = spec.get("target") or "host"
    if target not in ("host", "pi"):
        raise ValueError(f"unknown target: {target!r} (expected 'host' or 'pi')")

    threads = _int(spec, "threads", 2, 1, MAX_THREADS)
    hop = _int(spec, "hop", 32, 1, MAX_HOP)
    every = _int(spec, "latency_every", 400, 0, 1_000_000)
    measure, duty = bool(spec.get("measure")), bool(spec.get("duty"))
    incremental = bool(spec.get("incremental"))
    if duty and incremental:
        raise ValueError(
            "the duty-cycle sweep and incremental inference measure different things and "
            "cannot be combined: the sweep asks which hop size keeps up, and a step graph "
            "has no hop — it emits on every sample, so its duty cycle is the single number "
            "the latency run already reports. Sweep the windowed graph, or measure the step "
            "graph on its own.")
    if incremental:
        row = next((m for m in list_models()
                    if m["name"] == (spec.get("model") or "").strip()), None)
        if row and row.get("lookahead"):
            raise ValueError(
                f"{row['name']} reads {row['lookahead']} samples ahead; the one-sample step graph "
                f"is causal-only. Untick incremental inference: the windowed path emits each "
                f"output once its future samples have arrived")
    check_only = bool(spec.get("check_only"))
    extra = shlex.split(spec.get("extra_args") or "")

    # The locked state the measurement is taken under.  OMP goes in the
    # environment rather than on the command line because the kernels read it
    # at import, long before any flag is parsed.
    cpus = _cpu_list(spec, threads)
    require_locked = bool(spec.get("require_locked"))
    # OMP governs numpy here, not onnxruntime (its wheels carry no OpenMP);
    # onnxruntime's own count is the --threads flag, and its pool's spin policy
    # is --no-spin.  Both end up recorded in the trace.
    lock_env = {"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
    spin = spec.get("spin", True)
    lock_args = ((["--cpus", cpus] if cpus else [])
                 + (["--require-locked"] if require_locked else [])
                 + ([] if spin else ["--no-spin"]))

    # Reading the state of this machine needs no model and no test set: it is
    # the thing you do *before* committing to a run, so it asks for nothing.
    if check_only and target == "host":
        return [{"label": "read this machine's state — nothing is measured",
                 "command": [sys.executable, "-m", "deploy.preflight",
                             "--threads", str(threads), *lock_args],
                 "env": lock_env,
                 "note": "CPU governor, affinity, thread count, throttling, load"}]

    steps = []
    if spec.get("export"):
        step = _export_step(spec)
        name = step.pop("name")
        steps.append(step)
        if incremental:
            # A separate artifact and a separate command: the step graph is a
            # different signature, not a flag on the windowed one, and the Pi
            # needs it specifically. Exported after, so a failure here leaves
            # the windowed export standing.
            step_export = _export_step(spec)
            step_export.pop("name", None)
            step_export["command"] = step_export["command"] + ["--incremental"]
            step_export["label"] = "export the one-sample step graph"
            step_export["note"] = ("state in, state out; verified against "
                                   "models/incremental.py before it is written")
            steps.append(step_export)
    else:
        name = _model_name(spec)
        # The model is a trained run that has never been exported.  Say which
        # of the two it is, rather than reporting a missing file.
        if not (ARTIFACTS_DIR / f"{name}.onnx").is_file():
            known = {m["name"]: m for m in list_models()}.get(name)
            if known and known["run"]:
                raise ValueError(
                    f"{name} is a trained run with no ONNX export yet — tick "
                    "'export' to write one first")

    if target == "pi":
        if not RUNNER.is_file():
            raise FileNotFoundError("deploy/run_on_pi.sh is missing")
        if extra:
            raise ValueError("run_on_pi.sh takes no extra flags — the Pi run is "
                             "tuned with hop, threads and the sweep settings")
        if (incremental and not spec.get("export")
                and not (ARTIFACTS_DIR / f"{name}_step.onnx").is_file()):
            raise FileNotFoundError(
                f"deploy/artifacts/{name}_step.onnx does not exist. The Pi runs the step "
                f"graph, not models/incremental.py (it has no torch), so the graph has to be "
                f"exported first: python -m deploy.export_onnx --ckpt runs/{name}/best.pt "
                f"--name {name} --incremental")
        ssh = (spec.get("ssh") or "").strip()
        if not ssh:
            raise ValueError("the Pi's address is needed, e.g. pi@raspberrypi.local")
        if not SSH_TARGET_RE.match(ssh) or ssh.count("@") > 1:
            raise ValueError(f"not a [user@]host address: {ssh!r}")
        remote = (spec.get("remote_dir") or "eeg").strip()
        if not NAME_RE.match(remote):
            raise ValueError(f"not a directory name: {remote!r}")
        testdata = _testdata_name(spec)
        realtime = _int(spec, "realtime_s", 0, 0, 3600) if spec.get("realtime") else 0
        governor = (spec.get("governor") or "").strip()
        if governor and not NAME_RE.match(governor):
            raise ValueError(f"not a governor name: {governor!r}")
        env = {"INCREMENTAL": "1" if incremental else "",
               "REMOTE_DIR": remote, "TESTDATA": testdata, "HOP": str(hop),
               "THREADS": str(threads), "LATENCY_EVERY": str(every),
               "REALTIME_S": str(realtime),
               "DUTY_HOPS": _hop_list(spec) if duty else "",
               "DUTY_S": str(int(_float(spec, "duty_seconds", 20, 1, MAX_DUTY_SECONDS))),
               "CPUS": cpus, "GOVERNOR": governor,
               "REQUIRE_LOCKED": "1" if require_locked else "",
               "CHECK_ONLY": "1" if check_only else "",
               "SPIN": "1" if spin else "0"}
        steps.append({
            "label": (f"copy to {ssh} and read the board's state there" if check_only
                      else f"copy to {ssh} and run the test there"),
            "command": [_bash_executable(), RUNNER.relative_to(PROJECT_ROOT).as_posix(), ssh, name],
            "env": env,
            # run_on_pi.sh always benchmarks and scores the test set; the sweep
            # and the paced replay are the parts it switches on.
            "note": "environment check only, nothing is measured" if check_only else
                    (("one sample at a time, state in the graph — "
                      if incremental else "")
                     + "benchmark + test set + per-hop latency"
                     + (", duty-cycle sweep" if duty else "")
                     + (f", {realtime} s paced replay" if realtime else "")),
        })
        return steps

    # --- this machine ---
    if not (measure or duty or incremental or steps):
        raise ValueError("nothing to run — tick a measurement")

    if incremental:
        # On this machine either path is available, and they compute the same
        # function: models/incremental.py is the reference the step graph was
        # verified against. The torch one is used because it also reports the
        # cache breakdown; the Pi runs the graph, which is the point of it.
        ckpt = RUNS_DIR / name / "best.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"runs/{name}/best.pt does not exist — incremental inference runs the trained "
                f"checkpoint in torch, not the exported graph")
        steps.append({
            "label": "incremental inference on this machine (PyTorch, one sample at a time)",
            "command": [sys.executable, "-m", "deploy.bench_incremental",
                        "--ckpt", f"runs/{name}/best.pt",
                        "--hops", str(hop), "--threads", str(threads), "--trace",
                        *(["--no-preflight"] if not require_locked else [])],
            "env": lock_env,
            "note": "per-sample cost and the real KV cache, against the windowed path at "
                    f"hop {hop}; PyTorch, so it cannot run on the Pi",
        })

    if not (measure or duty):
        return steps                                   # export / incremental on their own
    testdata = _testdata_name(spec)
    if not spec.get("export") and not (ARTIFACTS_DIR / f"{name}.onnx").is_file():
        raise FileNotFoundError(
            f"deploy/artifacts/{name}.onnx does not exist — export it first: "
            f"python -m deploy.export_onnx --ckpt runs/{name}/best.pt")
    stem = f"deploy/artifacts/pi_latency/host_{name}_hop{hop}"
    base = [sys.executable, "deploy/raspi_infer.py", f"deploy/artifacts/{name}.onnx",
            "--threads", str(threads), "--hop", str(hop), *lock_args]
    if measure:
        steps.append({
            "label": "benchmark + test set + per-hop latency on this machine",
            "command": base + ["--benchmark", "--testdata", f"deploy/testdata/{testdata}",
                               "--latency", "--latency_every", str(every),
                               "--latency_out", stem, *extra],
            "env": lock_env,
        })
    if duty:
        # A separate call on purpose: raspi_infer returns straight after the
        # sweep, so a single invocation could not also score the test set.
        steps.append({
            "label": "duty-cycle sweep (paced replay per hop size)",
            "command": base + ["--testdata", f"deploy/testdata/{testdata}",
                               "--duty_cycle", _hop_list(spec),
                               "--duty_seconds", f"{_float(spec, 'duty_seconds', 20, 1, MAX_DUTY_SECONDS):g}",
                               "--latency_out", stem, *extra],
            "env": lock_env,
        })
    return steps


def describe(steps: list[dict]) -> list[dict]:
    """``build_steps`` output as the page shows it: one printable line each."""
    out = []
    for s in steps:
        env = " ".join(f"{k}={shlex.quote(v)}" for k, v in (s.get("env") or {}).items() if v != "")
        cmd = " ".join(shlex.quote(c) for c in s["command"])
        out.append({"label": s["label"], "note": s.get("note", ""),
                    "command": f"{env} {cmd}".strip()})
    return out


# ============================================================
# Reading the log as it streams
# ============================================================
_N = r"[-+]?\d+(?:\.\d+)?"
_NN = rf"(?:{_N}|None|nan)"


def _num(text: str | None):
    """A parsed field, or None for the ``None``/``n/a`` a summary can print."""
    if text is None or text in ("None", "nan", "n/a"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


class LogScan:
    """
    The numbers a deploy run prints, read back out of its output line by line.

    Everything here is also written to a trace file, and the page prefers the
    file the moment one appears -- so this is what makes the tab fill in while
    a two-minute sweep is still going, not the source of record.  The two are
    kept interchangeable by parsing the duty rows into the same field names
    ``save_duty_rows`` writes, so one renderer draws either.
    """

    #   "[raspi] m.onnx: window=512 fs=256 Hz  providers=[...]  intra_op=2 (1
    #    worker + the caller)  spinning=on".  The tail is matched separately
    #    because it has changed: it used to read "threads=2", before the
    #    difference between onnxruntime's intra-op count and OMP's mattered.
    #    Logs already in runs/ still parse.
    MODEL = re.compile(rf"^\[raspi\] (\S+): window=(\d+) fs=({_N}) Hz\s+"
                       rf"providers=\[([^\]]*)\]\s*(.*)$")
    MODEL_THREADS = re.compile(r"(?:intra_op|threads)=(\d+)")
    MODEL_SPIN = re.compile(r"spinning=(on|off)")
    BENCH = re.compile(rf"^\[bench\] median ({_N}) ms \| p95 ({_N}) ms \| max ({_N}) ms per "
                       rf"(\d+)-sample window;\s+hop (\d+) = ({_N}) ms budget "
                       rf"-> ({_NN})x real-time headroom")
    # A step graph advances one sample, so raspi_infer times one step instead.
    BENCH_STEP = re.compile(rf"^\[bench\] median ({_N}) ms \| p95 ({_N}) ms \| max ({_N}) ms "
                            rf"per sample;\s+one sample = ({_N}) ms at ({_N}) Hz "
                            rf"-> ({_NN})x real-time headroom")
    BOARD = re.compile(r"^-->\s+(.*\|\s*Python\s.*)$")
    THERMAL = re.compile(rf"^-->\s+(before|after):\s+({_NN})°C(.*)$")

    TEST = re.compile(rf"^\[test\] (\S+): (\d+) epochs x (\d+) samples, "
                      rf"SNR\s*({_N})\.\.({_N}) dB")
    #   "       epoch mode                  0.4601   0.4643  0.8826"
    TEST_ROW = re.compile(rf"^\s{{4,}}([A-Za-z][^|]*?)\s{{2,}}({_N})\s+({_N})\s+({_N})\s*$")
    #   "                                 -7.0  10   0.6534   0.7141  0.7798"
    SNR_ROW = re.compile(rf"^\s{{10,}}([-+]\d+(?:\.\d+)?)\s+(\d+)\s+({_N})\s+({_N})\s+({_N})\s*$")

    LAT_HEAD = re.compile(rf"^\[([^\]]+)\] (\d+) hops of (\d+) samples \(({_N}) ms budget each\), "
                          rf"({_N}) s wall, first window ({_NN}) ms")
    LAT_HOP = re.compile(rf"per hop\s*: median ({_N}) \| p95 ({_N}) \| p99 ({_N}) \| max ({_N}) ms"
                         rf"\s+\(jitter p95-median ({_N}) ms\)")
    LAT_OF = re.compile(rf"of which\s*: onnxruntime ({_N}) ms median / ({_N}) p95;"
                        rf"\s+pre/post-processing ({_N}) ms median")
    LAT_RT = re.compile(rf"real time\s*: (\d+) deadline misses \(({_N})%\), ({_NN})x headroom at p95, "
                        rf"({_NN}) hops/s, duty cycle ({_N})% mean / ({_N})% p99")
    LAT_E2E = re.compile(rf"end-to-end:.*?median ({_N}) \| p95 ({_N}) \| max ({_N}) ms"
                         rf"\s+\(drift over run ({_N}) ms\)")
    LAT_TEMP = re.compile(rf"SoC temp\s*: ({_N}) -> ({_N}) °C \(peak ({_N})\)")
    #   "           memory    : peak 112.4 MiB resident (61.0 MiB above the 51.4 MiB before the session)"
    LAT_MEM = re.compile(rf"memory\s*: peak ({_N}) MiB resident"
                         rf"(?: \(({_N}) MiB above the ({_N}) MiB before the session\))?")
    #   "[latency/testdata] hop    800 | last 2.86 ms (run 2.83) | ... | misses 0/800 of 125.0 ms budget"
    PROGRESS = re.compile(rf"^\[([^\]]+)\] hop\s+(\d+) \| last\s+({_N}) ms \(run\s+({_N})\)"
                          rf".*?misses (\d+)/(\d+) of ({_N}) ms budget(?:\s*\|\s*({_N})°C)?")

    #   "... wrote deploy/artifacts/pi_latency/host_tcn_mla_eog_hop32_duty.json and ..."
    TRACE_PATH = re.compile(r"pi_latency/([A-Za-z0-9][\w.-]*\.json)\b")

    DUTY_HEAD = re.compile(rf"^\[duty\] ({_N}) s paced replay per hop at ({_N}) Hz, window (\d+), "
                           rf"(\d+) onnxruntime threads")
    DUTY_MEM = re.compile(rf"^\[duty\] peak process memory ({_N}) MiB resident")
    DUTY_VERDICT = re.compile(r"^\[duty\] (smallest hop that is comfortably real-time:.*"
                              r"|no hop in the sweep is comfortably real-time.*)$")
    VERDICTS = ("OK", "MARGINAL", "FAIL")

    def __init__(self):
        self.reset()

    #: deploy/preflight.py prints its verdict twice: a human block, and one
    #: line of JSON for exactly this.  A measurement's conditions are not
    #: something to recover with a regex over a table.
    PREFLIGHT = "[preflight-json] "

    def reset(self):
        self.environment: dict | None = None
        self.model: dict | None = None
        self.board: str | None = None
        self.thermal: dict = {}
        self.bench: dict | None = None
        self.test: dict | None = None
        self.latency: list[dict] = []
        self.progress: dict | None = None
        self.duty: dict = {"rows": [], "verdict": None, "setup": None}
        self.traces: list[str] = []

    # ---- one line ----
    def feed(self, line: str) -> None:
        if self._structural(line):
            return
        if self._latency(line):
            return
        if self._duty(line):
            return
        self._test(line)
        self._trace(line)

    def _structural(self, line: str) -> bool:
        if line.startswith(self.PREFLIGHT):
            try:
                data = json.loads(line[len(self.PREFLIGHT):])
            except ValueError:
                return True
            if isinstance(data, dict):
                self.environment = data
            return True
        m = self.MODEL.match(line)
        if m:
            tail = m.group(5) or ""
            threads = self.MODEL_THREADS.search(tail)
            spin = self.MODEL_SPIN.search(tail)
            self.model = {"onnx": m.group(1), "window": int(m.group(2)),
                          "sample_rate_hz": _num(m.group(3)),
                          "providers": m.group(4).replace("'", "").strip(),
                          "threads": int(threads.group(1)) if threads else None,
                          "spinning": None if not spin else spin.group(1) == "on"}
            return True
        m = self.BENCH.match(line)
        if m:
            self.bench = {"median_ms": _num(m.group(1)), "p95_ms": _num(m.group(2)),
                          "max_ms": _num(m.group(3)), "window": int(m.group(4)),
                          "hop": int(m.group(5)), "hop_budget_ms": _num(m.group(6)),
                          "realtime_headroom_x": _num(m.group(7))}
            return True
        m = self.BENCH_STEP.match(line)
        if m:
            self.bench = {"median_ms": _num(m.group(1)), "p95_ms": _num(m.group(2)),
                          "max_ms": _num(m.group(3)), "per_sample": True, "window": 1,
                          "hop": 1, "hop_budget_ms": _num(m.group(4)),
                          "realtime_headroom_x": _num(m.group(6))}
            return True
        m = self.THERMAL.match(line)
        if m:
            self.thermal[m.group(1)] = {"temp_c": _num(m.group(2)), "detail": m.group(3).strip()}
            return True
        m = self.BOARD.match(line)
        if m:
            self.board = m.group(1).strip()
            return True
        return False

    def _latency(self, line: str) -> bool:
        m = self.LAT_HEAD.match(line)
        if m:
            self.latency.append({
                "label": m.group(1), "n_hops": int(m.group(2)), "hop": int(m.group(3)),
                "hop_budget_ms": _num(m.group(4)), "wall_s": _num(m.group(5)),
                "first_window_ms": _num(m.group(6)),
                "total_ms": {}, "infer_ms": {}, "overhead_ms": {},
            })
            return True
        # The rolling line comes first and repeats throughout the run, long
        # before the summary that creates the record below.
        m = self.PROGRESS.match(line)
        if m:
            self.progress = {"label": m.group(1), "hops": int(m.group(2)),
                             "last_ms": _num(m.group(3)), "infer_ms": _num(m.group(4)),
                             "deadline_misses": int(m.group(5)),
                             "hop_budget_ms": _num(m.group(7)), "temp_c": _num(m.group(8))}
            return True
        if not self.latency:
            return False
        cur = self.latency[-1]
        m = self.LAT_HOP.search(line)
        if m:
            cur["total_ms"] = {"median": _num(m.group(1)), "p95": _num(m.group(2)),
                               "p99": _num(m.group(3)), "max": _num(m.group(4))}
            cur["jitter_ms"] = _num(m.group(5))
            return True
        m = self.LAT_OF.search(line)
        if m:
            cur["infer_ms"] = {"median": _num(m.group(1)), "p95": _num(m.group(2))}
            cur["overhead_ms"] = {"median": _num(m.group(3))}
            return True
        m = self.LAT_RT.search(line)
        if m:
            cur.update(deadline_misses=int(m.group(1)),
                       miss_rate=round((_num(m.group(2)) or 0) / 100, 6),
                       realtime_headroom_x=_num(m.group(3)),
                       throughput_hops_per_s=_num(m.group(4)),
                       duty_cycle={"mean": round((_num(m.group(5)) or 0) / 100, 4),
                                   "p99": round((_num(m.group(6)) or 0) / 100, 4)})
            return True
        m = self.LAT_E2E.search(line)
        if m:
            cur["e2e_latency_ms"] = {"median": _num(m.group(1)), "p95": _num(m.group(2)),
                                     "max": _num(m.group(3))}
            cur["e2e_drift_ms"] = _num(m.group(4))
            return True
        m = self.LAT_TEMP.search(line)
        if m:
            cur["temp_c"] = {"start": _num(m.group(1)), "end": _num(m.group(2)),
                             "max": _num(m.group(3))}
            return True
        m = self.LAT_MEM.search(line)
        if m:
            cur["memory"] = {"peak_rss_mib": _num(m.group(1)), "added_by_session_mib": _num(m.group(2)),
                             "rss_before_session_mib": _num(m.group(3))}
            return True
        return False

    def _duty(self, line: str) -> bool:
        m = self.DUTY_HEAD.match(line)
        if m:
            self.duty["setup"] = {"seconds": _num(m.group(1)), "sample_rate_hz": _num(m.group(2)),
                                  "window": int(m.group(3)), "threads": int(m.group(4))}
            self.duty["rows"] = []
            return True
        m = self.DUTY_VERDICT.match(line)
        if m:
            self.duty["verdict"] = m.group(1).strip()
            return True
        m = self.DUTY_MEM.match(line)
        if m:
            self.duty["memory"] = {"peak_rss_mib": _num(m.group(1))}
            return True
        row = self._duty_row(line)
        if row:
            self.duty["rows"] = [r for r in self.duty["rows"] if r["hop"] != row["hop"]] + [row]
            return True
        return False

    @classmethod
    def _duty_row(cls, line: str) -> dict | None:
        """
        One row of the sweep table, parsed by shape rather than by width.

        ``duty_cycle_check`` prints it as five ``|``-separated groups of
        numbers and a verdict; checking the group sizes and the verdict word is
        what keeps this from matching the two header lines above it (or
        anything else with a pipe in it).
        """
        parts = [p.split() for p in line.split("|")]
        if len(parts) != 6 or [len(p) for p in parts] != [3, 3, 2, 3, 1, 1]:
            return None
        if parts[5][0] not in cls.VERDICTS:
            return None
        try:
            hop, budget, per_s = int(parts[0][0]), float(parts[0][1]), float(parts[0][2])
            med, p99, mx = (float(x) for x in parts[1])
            duty_mean, duty_p99 = (round(float(x) / 100, 4) for x in parts[2])
            e2e_med, e2e_max, drift = (float(x) for x in parts[3])
            misses = int(parts[4][0])
        except ValueError:
            return None
        return {"hop": hop, "hop_budget_ms": budget, "hops_per_s": per_s,
                "compute_median_ms": med, "compute_p99_ms": p99, "compute_max_ms": mx,
                "duty_mean": duty_mean, "duty_p99": duty_p99,
                "e2e_median_ms": e2e_med, "e2e_max_ms": e2e_max, "e2e_drift_ms": drift,
                "deadline_misses": misses, "verdict": parts[5][0],
                "total_latency_median_ms": round(budget + e2e_med, 2)}

    def _test(self, line: str) -> None:
        m = self.TEST.match(line)
        if m:
            self.test = {"file": m.group(1), "n_epochs": int(m.group(2)),
                         "window": int(m.group(3)), "snr_db": [_num(m.group(4)), _num(m.group(5))],
                         "rows": [], "per_snr": []}
            return
        if self.test is None:
            return
        m = self.TEST_ROW.match(line)
        if m:
            name = m.group(1).strip()
            row = {"name": name, "rrmse_t": _num(m.group(2)),
                   "rrmse_s": _num(m.group(3)), "acc": _num(m.group(4))}
            self.test["rows"] = [r for r in self.test["rows"] if r["name"] != name] + [row]
            return
        m = self.SNR_ROW.match(line)
        if m:
            snr = _num(m.group(1))
            row = {"snr_db": snr, "n": int(m.group(2)), "rrmse_t": _num(m.group(3)),
                   "rrmse_s": _num(m.group(4)), "acc": _num(m.group(5))}
            self.test["per_snr"] = [r for r in self.test["per_snr"]
                                    if r["snr_db"] != snr] + [row]

    def _trace(self, line: str) -> None:
        """
        Collect the trace files a run says it wrote.

        Matched on the directory name rather than by splitting the line into
        whitespace-separated paths, because this project's own directory has
        spaces in it and ``run_on_pi.sh`` announces the copies it brings back
        with absolute paths.  A Pi-side path (``/home/pi/eeg/...``) carries no
        ``pi_latency/`` and is ignored -- it names a file on the board, which
        is not one this GUI can read.  Existence is the last check: what is
        offered has to be openable.
        """
        for name in self.TRACE_PATH.findall(line):
            if name not in self.traces and (TRACE_DIR / name).is_file():
                self.traces.append(name)

    # ---- what the page gets ----
    def state(self) -> dict:
        return {"environment": self.environment,
                "model": self.model, "board": self.board, "thermal": self.thermal,
                "bench": self.bench, "test": self.test, "latency": list(self.latency),
                "progress": self.progress,
                "duty": {"rows": sorted(self.duty["rows"], key=lambda r: -r["hop"]),
                         "verdict": self.duty["verdict"], "setup": self.duty["setup"],
                         "memory": self.duty.get("memory")},
                "traces": list(self.traces)}
