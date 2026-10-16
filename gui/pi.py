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

import csv
import json
import re
import shlex
import sys
from pathlib import Path

from deploy import preflight
from gui import config_io
from gui.config_io import PROJECT_ROOT, RUNS_DIR

#: The governor a reportable measurement is taken under.  Defined by the
#: checker, not here, so the form's default and the thing that judges it
#: cannot disagree.  (preflight is stdlib-only, like the rest of the GUI.)
WANTED_GOVERNOR = preflight.WANTED_GOVERNOR

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
    exports = {e["name"]: e for e in list_exports()}
    try:
        runs = {r["name"]: r for r in config_io.list_runs()}
    except Exception:                          # torch missing: exports still list
        runs = {}
    rows = []
    for name in sorted(set(exports) | set(runs)):
        e, r = exports.get(name), runs.get(name)
        row = {"name": name, "exported": e is not None, "run": r is not None,
               "variant": (e or {}).get("variant") or (r or {}).get("variant"),
               "artifact": (e or {}).get("artifact") or (r or {}).get("artifact"),
               "params": (e or {}).get("params") or (r or {}).get("params"),
               "bidirectional": bool((r or {}).get("bidirectional")),
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
        row["deployable"] = row["exported"] or (row["run"] and not row["bidirectional"])
        rows.append(row)
    return rows


def list_testdata() -> list[dict]:
    """One row per ``deploy/testdata/*.npz`` -- the scored sets, and the signal
    the duty-cycle sweep is replayed from."""
    if not TESTDATA_DIR.is_dir():
        return []
    return [{"name": p.name, "size_kb": round(p.stat().st_size / 1024, 1)}
            for p in sorted(TESTDATA_DIR.glob("*.npz"))]


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
                       median_ms=(data.get("total_ms") or {}).get("median"),
                       deadline_misses=data.get("deadline_misses"),
                       realtime=("e2e_latency_ms" in data),
                       environment=data.get("environment"))
        # The verdict the dropdown shows: a trace from before this was recorded
        # says "unknown", which is not the same as "unlocked" and must not read
        # as it.
        row["verdict"] = ((row.get("environment") or {}).get("verdict")
                          if row.get("environment") else None)
        rows.append(row)
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return rows


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
    # Said here rather than two minutes into the export: a *_bi twin needs the
    # future, and the exporter refuses it for that reason.
    if any(m["name"] == run and m["bidirectional"] for m in list_models()):
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
        env = {"REMOTE_DIR": remote, "TESTDATA": testdata, "HOP": str(hop),
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
            "command": ["bash", str(RUNNER.relative_to(PROJECT_ROOT)), ssh, name],
            "env": env,
            # run_on_pi.sh always benchmarks and scores the test set; the sweep
            # and the paced replay are the parts it switches on.
            "note": "environment check only, nothing is measured" if check_only else
                    ("benchmark + test set + per-hop latency"
                     + (", duty-cycle sweep" if duty else "")
                     + (f", {realtime} s paced replay" if realtime else "")),
        })
        return steps

    # --- this machine ---
    if not (measure or duty or steps):
        raise ValueError("nothing to run — tick a measurement")
    if not (measure or duty):
        return steps                                   # export on its own
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
    #   "[latency/testdata] hop    800 | last 2.86 ms (run 2.83) | ... | misses 0/800 of 125.0 ms budget"
    PROGRESS = re.compile(rf"^\[([^\]]+)\] hop\s+(\d+) \| last\s+({_N}) ms \(run\s+({_N})\)"
                          rf".*?misses (\d+)/(\d+) of ({_N}) ms budget(?:\s*\|\s*({_N})°C)?")

    #   "... wrote deploy/artifacts/pi_latency/host_tcn_mla_eog_hop32_duty.json and ..."
    TRACE_PATH = re.compile(r"pi_latency/([A-Za-z0-9][\w.-]*\.json)\b")

    DUTY_HEAD = re.compile(rf"^\[duty\] ({_N}) s paced replay per hop at ({_N}) Hz, window (\d+), "
                           rf"(\d+) onnxruntime threads")
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
                         "verdict": self.duty["verdict"], "setup": self.duty["setup"]},
                "traces": list(self.traces)}
