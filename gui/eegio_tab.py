"""
Server side of the **EEG-IO test** tab: the project's models against real
recordings.

A thin layer over ``eegio_eval.py``.  It exists so the page can ask three
things without knowing anything about filters or blink detection: what is on
disk, which runs are worth comparing, and what happens when they are.

The work is synchronous, like the KD audit tab's and unlike the Train and
Deploy tabs': a comparison is a few forward passes per subject, not a
training run, and ``n_subjects`` caps it so a click cannot turn into a
quarter of an hour.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

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

#: Subjects per comparison unless the page says otherwise.  Twenty subjects
#: times three models times a forward pass over 100 s is minutes, and the
#: page should come back before someone wonders whether it has hung.
DEFAULT_N_SUBJECTS = 4

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


def _runs_dir() -> Path:
    from gui import config_io
    return config_io.RUNS_DIR


def _trained_at(name: str) -> tuple:
    """
    The rate a run was trained at, and whether it said so itself.

    ``target_fs`` has only been recorded since EEGdenoiseNet became
    trainable at another rate; before that there was one rate and nothing to
    record.  So a run with nothing recorded was trained at 256 Hz -- but it
    is *inferred*, and the two are shown differently, because a directory
    name no longer implies a rate: ``runs/tcn_mla_both_s67`` held a 256 Hz
    run and now holds a 250 Hz one.
    """
    import json

    h = _runs_dir() / name / "history.json"
    if not h.is_file():
        return (None, False)
    try:
        fs = (json.loads(h.read_text()).get("config") or {}).get("target_fs")
    except ValueError:
        return (None, False)
    return (float(fs), True) if fs else (None, False)


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
    import eegio_eval as E
    from gui import config_io

    out = []
    for r in config_io.list_runs():
        if r["status"] != "complete":
            continue
        fs, recorded = _trained_at(r["name"])
        out.append({
            "name": r["name"], "variant": r["variant"], "artifact": r["artifact"],
            "master_seed": r["master_seed"], "hidden_channels": r["hidden_channels"],
            "rrmse_t": r["rrmse_t"],
            "causal": r["causal"], "lookahead": r["lookahead"],
            "bidirectional": r["bidirectional"],
            #: The rate it trained at, and whether that is its own word for
            #: it or this module's inference from an older run's silence.
            "train_fs": fs if recorded else E.PROJECT_FS,
            "train_fs_recorded": recorded,
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
    import json
    from datetime import datetime

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"eegio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    # Named before it is written, so the file knows what it is called. A copy
    # that does not is a copy the page cannot select in its own history list.
    out["name"] = name
    (RESULTS_DIR / name).write_text(json.dumps(out, indent=2))

    # Oldest first, so the survivors are the newest MAX_RESULTS.
    kept = sorted(RESULTS_DIR.glob("eegio_*.json"))
    for stale in kept[:-MAX_RESULTS]:
        stale.unlink()
    return name


def _describe(out: dict, name: str) -> dict:
    """One row for the history list: enough to choose between two of them."""
    results = out.get("results") or []
    return {
        "name": name,
        "saved_at": (out.get("saved") or {}).get("at"),
        "runs": [r.get("run") for r in results],
        "baseline": out.get("baseline"),
        "n_subjects": len(out.get("subjects") or []),
        "target_fs": out.get("target_fs"),
        "hop": out.get("hop"),
        "zero_phase": (out.get("saved") or {}).get("zero_phase"),
        "include_soft": (out.get("saved") or {}).get("include_soft"),
        "n_mismatched": len(out.get("rate_mismatch") or []),
        # The headline of each, so the list can be read without opening a row.
        "summary": {r.get("run"): (r.get("summary") or {}) for r in results},
    }


def list_results() -> List[dict]:
    """Every saved comparison, newest first."""
    import json

    out = []
    if not RESULTS_DIR.is_dir():
        return out
    for f in sorted(RESULTS_DIR.glob("eegio_*.json"), reverse=True):
        try:
            out.append(_describe(json.loads(f.read_text()), f.name))
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
        "metrics": list(E.METRIC_PATHS),
        "defaults": {
            "n_subjects": DEFAULT_N_SUBJECTS, "hop": 32, "channel": 0,
            "band_hz": list(E.TRAIN_BAND_HZ), "notch_hz": E.POWERLINE_HZ,
            "target_fs": E.PROJECT_FS, "detect_k": 4.0,
        },
        "rates": [{"value": E.PROJECT_FS, "label": f"{E.PROJECT_FS:g} Hz (training rate)",
                   "note": "resamples 250 -> 256, exactly 128/125"},
                  {"value": E.EEGIO_FS, "label": f"{E.EEGIO_FS:g} Hz (native)",
                   "note": "no resampling; the model sees time dilated by 2.4%"}],
        "band_note": (f"{E.TRAIN_BAND_HZ[0]:g}–{E.TRAIN_BAND_HZ[1]:g} Hz with a "
                      f"{E.POWERLINE_HZ:g} Hz notch, read off EEGdenoiseNet's own "
                      f"preprocessing script, applied with sosfilt so the filter is "
                      f"causal too."),
    }


def evaluate(spec: dict) -> dict:
    """
    Run the comparison the page asked for.

    ``spec``: ``{runs, data_dir, n_subjects, subjects, hop, channel,
    zero_phase, no_notch, include_soft}``.  The first run is the baseline the
    others are paired against, which is why the page keeps their order.
    """
    import eegio_eval as E

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
        hop=int(spec.get("hop") or 32),
        zero_phase=bool(spec.get("zero_phase")),
        notch_hz=None if spec.get("no_notch") else E.POWERLINE_HZ,
        include_soft=bool(spec.get("include_soft")),
        detect_k=float(spec.get("detect_k") or 4.0),
        target_fs=float(spec.get("target_fs") or E.PROJECT_FS),
    )

    by_name = {r["name"]: r for r in available_runs()}
    mismatched = [
        {"run": r, "train_fs": by_name[r]["train_fs"],
         "recorded": by_name[r]["train_fs_recorded"]}
        for r in runs
        if r in by_name and by_name[r]["train_fs"] != cfg.target_fs
    ]

    results = [E.evaluate_subjects(data_dir, r, subjects, cfg) for r in runs]
    from datetime import datetime

    out = _finite({
        # What produced this, so a file found next week can be read. The
        # settings matter as much as the numbers: the same models at another
        # hop or another rate are a different measurement.
        "saved": {
            "at": datetime.now().isoformat(timespec="seconds"),
            "data_dir": data_dir, "n_subjects": n,
            "channel": cfg.channel, "hop": cfg.hop, "target_fs": cfg.target_fs,
            "zero_phase": cfg.zero_phase, "notch_hz": cfg.notch_hz,
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
        "tests": E.compare(results),
        "subjects": subjects,
        "baseline": runs[0],
        "hop": cfg.hop,
        "target_fs": cfg.target_fs,
        "native_fs": E.EEGIO_FS,
        "causality": results[0]["causality"],
        "metrics": list(E.METRIC_PATHS),
    })
    # Saved before it is returned, so a result exists on disk even if the
    # page that asked for it is closed before the answer arrives.
    save_result(out)
    return out
