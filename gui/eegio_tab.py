"""
Server side of the **EEG-IO test** tab: the project's models against real
recordings.

A thin layer over ``eegio_eval.py``.  It exists so the page can ask three
things without knowing anything about filters or blink detection: what is on
disk, which runs are worth comparing, and what happens when they are.

Comparisons report progress as each model and recording is processed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Where the recordings might be.  The download unpacks as
#: ``EEG-EyeBlinks/`` holding EEG-IO beside EEG-VV and EEG-VR, so that layout
#: is searched as it arrives rather than asking anyone to rename a directory
#: to match a constant in here.  First hit wins, in this order.
DATA_DIR_CANDIDATES = (
    PROJECT_ROOT / "data" / "EEG-IO",
    PROJECT_ROOT / "data" / "EEG-EyeBlinks" / "EEG-IO",
    PROJECT_ROOT / "data" / "EEGIO",
)


def find_data_dir() -> Path:
    """The first candidate that actually holds recordings, else the first."""
    for d in DATA_DIR_CANDIDATES:
        if d.is_dir() and any(f.name.endswith("_data.csv") for f in d.iterdir()):
            return d
    return DATA_DIR_CANDIDATES[0]


#: Kept for callers that want a path without searching.
DEFAULT_DATA_DIR = DATA_DIR_CANDIDATES[0]

#: Where a finished comparison is kept.  The Deploy tab's reason for saving
#: its traces applies here word for word: a measurement is a statement about
#: a model, a dataset and the settings it was taken under, and a number on a
#: screen that vanishes when the page reloads cannot be compared with next
#: week's.  Gitignored -- these reference recordings that are not
#: redistributable, and they are reproducible from the runs.
RESULTS_DIR = PROJECT_ROOT / "eegio_tests"

#: Kept at most this many, oldest discarded.  A comparison is a few hundred
#: kB of per-subject rows, so this is months of ordinary use.
MAX_RESULTS = 200

#: Use the full EEG-IO cohort by default.
DEFAULT_N_SUBJECTS = 20

#: The comparison this tab was built for: the causal model, the rung
#: licensed 16 samples (62.5 ms) ahead, and the unbounded twin that cannot be
#: deployed at all. The first is the baseline the others are tested against.
SUGGESTED = ("tcn_se_both", "tcn_se_la16_both", "tcn_se_bi_both")


def _finite(obj):
    """
    Replace every NaN and infinity with ``null`` on the way out.

    ``json.dumps`` writes them as bare ``NaN`` / ``Infinity``, which is not
    JSON: ``JSON.parse`` rejects the whole document, and a page that catches
    that failure sees an empty object rather than an error.  The metrics here
    produce NaN routinely and legitimately -- a subject with no usable blink
    has no suppression to report, a one-subject summary has no standard
    deviation -- so this is the normal path, not an error path.
    """
    import math

    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def available_runs() -> List[dict]:
    """
    Finished runs that can be evaluated, with what each one may read ahead
    and what rate it was trained at.

    The badges are the whole point of offering a list rather than a free text
    box.  A ``_bi`` twin can be run over a recording offline and is a
    legitimate control here, while on the Deploy tab it is refused.  And a
    model trained at 250 Hz is a different model from one trained at 256 --
    run at the wrong one it sees time dilated by 2.4%.  Which one a row is
    has to be visible before it is picked.
    """
    from gui import config_io

    out = []
    for r in config_io.list_runs():
        if r["status"] != "complete":
            continue
        out.append({
            "name": r["name"], "variant": r["variant"], "artifact": r["artifact"],
            "master_seed": r["master_seed"], "hidden_channels": r["hidden_channels"],
            "rrmse_t": r["rrmse_t"],
            "causal": r["causal"], "lookahead": r["lookahead"],
            "bidirectional": r["bidirectional"],
            "tcn_norm": r.get("tcn_norm"),
            "training_protocol": r.get("training_protocol"),
            #: The rate it trained at, and whether that is its own word for
            #: it or this module's inference from an older run's silence.
            "train_fs": r["train_fs"],
            "train_fs_recorded": r["train_fs_recorded"],
            "suggested": r["name"] in SUGGESTED,
        })
    out.sort(key=lambda r: (not r["suggested"], r["name"]))
    return out


def dataset_status(data_dir: Optional[str] = None) -> dict:
    """What is on disk, and what to do when nothing is."""
    import eegio_eval as E
    from data.EEGIO import DATASET_URL, list_subjects

    d = Path(data_dir) if data_dir else find_data_dir()
    try:
        subs = list_subjects(str(d))
        return {"present": True, "dir": str(d), "subjects": subs, "n": len(subs),
                "url": DATASET_URL, "fs": E.EEGIO_FS,
                "searched": [str(c) for c in DATA_DIR_CANDIDATES]}
    except FileNotFoundError as e:
        return {"present": False, "dir": str(d), "subjects": [], "n": 0,
                "url": DATASET_URL, "message": str(e),
                "searched": [str(c) for c in DATA_DIR_CANDIDATES]}


# ============================================================
# Keeping the comparisons
# ============================================================
def _result_path(name: str) -> Path:
    """
    Resolve a client-supplied result name inside ``eegio_tests/``, or refuse.

    Resolve first and check afterwards, which is what makes ``../`` and a
    symlink out of the tree fail rather than escape -- the same rule the
    Deploy tab's traces and the Evaluate tab's figures use.  The GUI has no
    authentication, so this is the only place a request gets to name a file.
    """
    if not name:
        raise ValueError("no result named")
    target = (RESULTS_DIR / name).resolve()
    if RESULTS_DIR.resolve() not in target.parents:
        raise ValueError(f"{name} is outside eegio_tests/")
    if not target.is_file():
        raise FileNotFoundError(f"no such result: {name}")
    return target


def save_result(out: dict) -> str:
    """Write a finished comparison and return the file name."""
    from datetime import datetime
    from uuid import uuid4
    from experiment.storage import atomic_json

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"eegio_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{uuid4().hex[:8]}.json"
    # Named before it is written, so the file knows what it is called. A copy
    # that does not is a copy the page cannot select in its own history list.
    out["name"] = name
    atomic_json(RESULTS_DIR / name, out)

    # Oldest first, so the survivors are the newest MAX_RESULTS.
    kept = sorted(RESULTS_DIR.glob("eegio_*.json"))
    for stale in kept[:-MAX_RESULTS]:
        stale.unlink(missing_ok=True)
    return name


def _describe(out: dict, name: str) -> dict:
    """One row for the history list: enough to choose between two of them."""
    results = out.get("results") or []
    saved = out.get("saved") or {}
    target = saved.get("target", "host")
    return {
        "name": name,
        "saved_at": saved.get("at"),
        "runs": [r.get("run") for r in results],
        "tcn_norms": {r.get("run"): (r.get("arch") or {}).get("tcn_norm") for r in results},
        "baseline": out.get("baseline"),
        "target": target,
        **({"replay_mode": saved.get("replay_mode", "paced"),
            "incremental": bool(saved.get("incremental"))} if target == "pi5" else {}),
        **({"lookahead_comparison": saved["lookahead_comparison"]} if "lookahead_comparison" in saved else {}),
        "offline_window": saved.get("offline_window"),
        "non_causal_test_radius": saved.get("non_causal_test_radius"),
        "metric_protocol": out.get("metric_protocol", "legacy"),
        "n_subjects": len(out.get("subjects") or []),
        "target_fs": out.get("target_fs"),
        "hop": out.get("hop"),
        "zero_phase": saved.get("zero_phase"),
        "rescale_input": bool(saved.get("rescale_input")),
        "offline_zero_phase": bool(saved.get("offline_zero_phase")),
        "zero_phase_runs": saved.get("zero_phase_runs") or [],
        "include_soft": saved.get("include_soft"),
        "n_mismatched": len(out.get("rate_mismatch") or []),
        # The headline of each, so the list can be read without opening a row.
        "summary": {r.get("run"): (r.get("summary") or {}) for r in results},
        **({"lookahead_summary": out["lookahead_summary"]} if "lookahead_summary" in out else {}),
    }


def _lookahead_seed_summary(results, comparison):
    """Summarize participant medians within seed, then mean/sample SD across seeds."""
    import statistics
    import math

    by_name = {r["run"]: r for r in results}
    groups = {}
    for member in comparison["members"]:
        result = by_name[member["run"]]
        group = groups.setdefault(member["lookahead"], [])
        group.append((member, result.get("summary") or {}))
    out = []
    for budget, rows in sorted(groups.items(), key=lambda item: (item[0] is None, item[0] or 0)):
        metrics = {}
        for key in sorted({key for _, summary in rows for key in summary}):
            values = [(m["master_seed"], s[key]["median"]) for m, s in rows
                      if isinstance(s.get(key), dict) and s[key].get("median") is not None]
            values = [(seed, value) for seed, value in values if isinstance(value, (int, float))
                      and math.isfinite(value)]
            if values:
                numbers = [v for _, v in values]
                metrics[key] = {"n": len(numbers), "mean": statistics.mean(numbers),
                                "sd": statistics.stdev(numbers) if len(numbers) > 1 else None,
                                "seeds": [seed for seed, _ in values], "values": numbers}
        out.append({"lookahead": budget, "seeds": [m["master_seed"] for m, _ in rows],
                    "runs": [m["run"] for m, _ in rows], "metrics": metrics})
    return out


def list_results() -> List[dict]:
    """Every saved comparison, newest first."""
    import json

    out = []
    if not RESULTS_DIR.is_dir():
        return out
    for f in sorted(RESULTS_DIR.glob("eegio_*.json"), reverse=True):
        try:
            out.append(_describe(json.loads(f.read_text()), f.name))
        except FileNotFoundError:
            continue  # Pruned while reading the history.
        except ValueError:
            # A half-written file is not worth failing the whole list over.
            out.append({"name": f.name, "saved_at": None, "runs": [],
                        "error": "could not be parsed"})
    return out


def read_result(name: str) -> dict:
    import json

    return json.loads(_result_path(name).read_text())


def meta(data_dir: Optional[str] = None) -> dict:
    import eegio_eval as E

    return {
        "dataset": dataset_status(data_dir),
        "runs": available_runs(),
        "history": list_results(),
        "suggested": list(SUGGESTED),
        "pi_replay_modes": ["paced", "fast"],
        "metrics": list(E.METRIC_PATHS),
        "metric_protocol": E.METRIC_PROTOCOL,
        "legacy_metrics": list(E.LEGACY_METRIC_PATHS),
        "defaults": {
            "n_subjects": DEFAULT_N_SUBJECTS, "hop": E.EvalConfig.hop, "channel": 0,
            "band_hz": list(E.TRAIN_BAND_HZ), "notch_hz": E.POWERLINE_HZ,
            "target_fs": E.PROJECT_FS, "detect_k": 4.0,
        },
        "rates": [{"value": E.PROJECT_FS, "label": f"{E.PROJECT_FS:g} Hz (resampled)",
                   "note": "resamples 250 -> 256, exactly 128/125"},
                  {"value": E.EEGIO_FS, "label": f"{E.EEGIO_FS:g} Hz (native)",
                   "note": "uses the recordings at their original sampling rate"}],
        "band_note": (f"{E.TRAIN_BAND_HZ[0]:g}–{E.TRAIN_BAND_HZ[1]:g} Hz with a "
                      f"{E.POWERLINE_HZ:g} Hz notch, read off EEGdenoiseNet's own "
                      f"preprocessing script, applied with sosfilt so the filter is "
                      f"causal too."),
    }


def _flag(spec: dict, key: str) -> bool:
    value = spec.get(key, False)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a boolean")
    return value


def _offline_reference_runs(comparison: dict, runs: List[str]) -> List[str]:
    """The comparison's offline (unbounded-future) runs, in evaluation order."""
    if "members" in comparison:
        offline = {m["run"] for m in comparison["members"] if m["lookahead"] is None}
    elif "rungs" in comparison:
        offline = {run for run, budget in comparison["rungs"].items() if budget is None}
    else:
        offline = set(runs[1:2]) if comparison.get("lookahead", 0) is None else set()
    return [run for run in runs if run in offline]


def evaluate(spec: dict, *, on_progress: Optional[Callable[[dict], None]] = None,
             evaluate_model: Optional[Callable] = None, target: str = "host",
             lookahead_comparison: Optional[dict] = None) -> dict:
    """
    Run the comparison the page asked for.

    ``spec``: ``{runs, data_dir, n_subjects, subjects, hop, channel,
    zero_phase, no_notch, include_soft}``.  The first run is the baseline the
    others are paired against, which is why the page keeps their order.
    """
    import eegio_eval as E

    offline = spec.get("offline", False)
    if type(offline) is not bool:
        raise ValueError("offline must be a boolean")
    if spec.get("non_causal_test"):
        raise ValueError("The short-window mode has been retired. Refresh the EEG-IO test page.")
    if offline and target != "host":
        raise ValueError("Offline comparison requires local inference")
    evaluate_model = evaluate_model or E.evaluate_subjects
    runs = [r for r in (spec.get("runs") or []) if r]
    if not runs:
        raise ValueError("pick at least one run")
    data_dir = str(spec.get("data_dir") or find_data_dir())

    status = dataset_status(data_dir)
    if not status["present"]:
        raise FileNotFoundError(status["message"])

    subjects = spec.get("subjects") or status["subjects"]
    n = int(spec.get("n_subjects") or DEFAULT_N_SUBJECTS)
    if n > 0:
        subjects = subjects[:n]

    cfg = E.EvalConfig(
        channel=int(spec.get("channel") or 0),
        hop=int(spec.get("hop") or E.EvalConfig.hop),
        zero_phase=bool(spec.get("zero_phase")),
        rescale_input=_flag(spec, "rescale_input"),
        notch_hz=None if spec.get("no_notch") else E.POWERLINE_HZ,
        include_soft=bool(spec.get("include_soft")),
        detect_k=float(spec.get("detect_k") or 4.0),
        target_fs=float(spec.get("target_fs") or E.PROJECT_FS),
        # The Lookahead comparison always asks for it; a Pi replay asks for it
        # in its spec. The first run sets the per-subject threshold.
        adjusted_blink_count=lookahead_comparison is not None
        or spec.get("adjusted_blink_count") is True,
    )
    checkpoints = {}
    if offline:
        cfg, checkpoints = E.prepare_offline(runs, cfg)
    # Optional: the offline reference reads zero-phase input, the filter its
    # training data had, while every causal rung keeps the causal filter.
    zero_phase_runs = []
    if _flag(spec, "offline_zero_phase"):
        if lookahead_comparison is None or not offline:
            raise ValueError("offline_zero_phase applies to Lookahead comparisons only")
        if cfg.zero_phase:
            raise ValueError("Every model is already zero-phase filtered")
        zero_phase_runs = _offline_reference_runs(lookahead_comparison, runs)
        if not zero_phase_runs:
            raise ValueError("This comparison has no offline reference to filter differently")
    from dataclasses import replace as _replace
    zero_phase_cfg = _replace(cfg, zero_phase=True)

    by_name = {r["name"]: r for r in available_runs()}
    mismatched = [
        {"run": r, "train_fs": by_name[r]["train_fs"],
         "recorded": by_name[r]["train_fs_recorded"]}
        for r in runs
        if r in by_name and by_name[r]["train_fs"] != cfg.target_fs
    ]

    subject_count = len(subjects)
    total = len(runs) * subject_count

    def report(update):
        if on_progress:
            on_progress({**update, "model": run, "model_index": index + 1,
                         "n_models": len(runs), "n_subjects": subject_count, "total": total,
                         "completed": index * subject_count + update.get("completed_subjects", 0)})

    results = []
    reference_thresholds_uv = None
    batch = bool(lookahead_comparison and lookahead_comparison.get("schema_version") == 2)
    baseline_by_run = lookahead_comparison.get("baseline_by_run", {}) if batch else {}
    threshold_by_baseline = {}
    for index, run in enumerate(runs):
        report({"phase": "loading_model"})
        if batch:
            baseline = baseline_by_run[run]
            if baseline != run and baseline not in threshold_by_baseline:
                raise ValueError("Evaluate each seed's causal baseline before its lookahead rungs")
            reference_thresholds_uv = threshold_by_baseline.get(baseline)
        results.append(evaluate_model(data_dir, run, subjects,
                                      zero_phase_cfg if run in zero_phase_runs else cfg,
                                      on_progress=report if on_progress else None,
                                      **({"checkpoint": checkpoints[run]} if offline else {}),
                                      **({"reference_thresholds_uv": reference_thresholds_uv}
                                         if reference_thresholds_uv is not None else {})))
        if cfg.adjusted_blink_count and (index == 0 or batch and baseline_by_run[run] == run):
            reference_thresholds_uv = {
                row["subject"]: row["metric_support"]["adjusted_blink_count"]["threshold_uv"]
                for row in results[-1]["rows"]
            }
            if batch:
                threshold_by_baseline[run] = reference_thresholds_uv
    if on_progress:
        on_progress({"phase": "saving", "completed": total, "total": total})
    from datetime import datetime

    hop = None if offline else cfg.hop
    if batch:
        by_name = {r["run"]: r for r in results}
        tests = []
        for seed, baseline in lookahead_comparison["baselines"].items():
            members = [by_name[name] for name in runs if name != baseline and baseline_by_run[name] == baseline]
            tests.extend({**test, "master_seed": int(seed)} for test in E.compare([by_name[baseline], *members]))
    else:
        tests = E.compare(results, (*E.ALL_METRIC_PATHS, E.ADJUSTED_BLINK_COUNT)
                          if cfg.adjusted_blink_count else tuple(E.ALL_METRIC_PATHS))
    out = _finite({
        # What produced this, so a file found next week can be read. The
        # settings matter as much as the numbers: the same models at another
        # hop or another rate are a different measurement.
        "saved": {
            "at": datetime.now().isoformat(timespec="seconds"),
            "target": target,
            **({"replay_mode": spec.get("replay_mode", "paced"),
                "incremental": bool(spec.get("incremental"))} if target == "pi5" else {}),
            **({"lookahead_comparison": lookahead_comparison} if lookahead_comparison is not None else {}),
            **({"blink_count_adjustment": {
                "reference_run": runs[0], "source": "output",
                **({"reference_by_run": baseline_by_run} if batch else {}),
                "mean_absolute_amplitude_multiplier": E.BLINK_THRESHOLD_MULTIPLIER,
            }} if cfg.adjusted_blink_count else {}),
            **({"offline_window": results[0]["config"]["offline_window"]} if offline else {}),
            "data_dir": data_dir, "n_subjects": subject_count,
            "channel": cfg.channel, "hop": hop, "target_fs": cfg.target_fs,
            "zero_phase": cfg.zero_phase, "notch_hz": cfg.notch_hz,
            "rescale_input": cfg.rescale_input,
            **({"offline_zero_phase": True, "zero_phase_runs": zero_phase_runs} if zero_phase_runs else {}),
            "include_soft": cfg.include_soft, "detect_k": cfg.detect_k,
            "band_hz": list(cfg.band),
            # Each model with the rate it trained at, which the run rows do
            # not carry and which decides whether a number is comparable.
            "models": [{"run": r, "train_fs": by_name.get(r, {}).get("train_fs"),
                        "train_fs_recorded": by_name.get(r, {}).get("train_fs_recorded")}
                       for r in runs],
        },
        # Runs whose training rate is not the rate they were just evaluated
        # at. Not refused -- comparing the two is a legitimate thing to want
        # -- but it changes what every number below means, so it is said.
        "rate_mismatch": mismatched,
        "results": results,
        "tests": tests,
        **({"lookahead_summary": _lookahead_seed_summary(results, lookahead_comparison)} if batch else {}),
        "subjects": subjects,
        "baseline": runs[0],
        "hop": hop,
        "target_fs": cfg.target_fs,
        "native_fs": E.EEGIO_FS,
        "causality": results[0]["causality"],
        "metrics": list(E.METRIC_PATHS),
        "metric_protocol": E.METRIC_PROTOCOL,
        "legacy_metrics": list(E.LEGACY_METRIC_PATHS),
    })
    # Saved before it is returned, so a result exists on disk even if the
    # page that asked for it is closed before the answer arrives.
    save_result(out)
    return out
