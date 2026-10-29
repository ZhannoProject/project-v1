"""
Evaluating a denoiser on real recordings, where there is no clean signal.

Everything else in this project is scored against a known target: an
EEGdenoiseNet epoch is a clean signal plus an artifact we mixed ourselves, so
RRMSE is computable.  EEG-IO is the other case -- real frontal EEG, really
contaminated, with the blink instants annotated by hand and no reference at
all (see ``data/EEGIO.py``).  None of the project's metrics apply.

What is available instead is a *contrast*: the annotated blink intervals
against the background between them.  A denoiser that is working removes
energy inside the intervals and leaves the background alone, and those are
two separate measurements that have to be reported together -- a model that
scores well on the first alone has simply learned to attenuate everything.

So four numbers, all of them relative to the **preprocessed signal that was
fed to the model**, never to the raw recording:

    suppression S      how much blink-interval power the model removed
    fidelity D_b       how much of the background survived it
    residual detect.   blinks the detector still finds in the output
    onset leakage      how far before a blink the model's correction starts

The fourth is the one this project cares about most.  A strictly causal model
cannot begin correcting before the artifact begins; a model licensed to read
16 samples ahead can, by up to 62.5 ms.  Measured on real blinks, that is the
lookahead ladder's claim meeting a recording it was not trained on.

**Preprocessing is causal and that is not free.** EEGdenoiseNet's own
pipeline bandpasses 1-80 Hz and notches 60 Hz with a zero-phase filter
(``code/preprocessing/EEG/IC_label_artifact_removal.m``), which no online
system can run: zero-phase filtering is two passes, one of them backwards.
Everything here uses ``scipy.signal.sosfilt`` -- one pass, forwards -- so the
band matches training and the phase does not.  That mismatch is a cost of
being deployable and it is measurable, which is why the zero-phase filter is
kept as a control rather than deleted.

    python eegio_eval.py --data_dir data/EEG-IO --runs runs/tcn_se_both runs/tcn_se_la16_both

Nothing here trains anything.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.EEGIO import (                                           # noqa: E402
    EEGIO_FS, PROJECT_FS, Recording, list_subjects, read_recording, resample_to,
)

# ============================================================
# Constants, taken from the training data rather than chosen
# ============================================================
#: The band EEGdenoiseNet was filtered to before it was resampled, read off
#: its own preprocessing script: ``filter_data(EEG_data, fs, 1, 80, order)``.
#: A model trained on that data has never seen anything outside it, so a real
#: recording has to be brought into it before the comparison means anything.
TRAIN_BAND_HZ: Tuple[float, float] = (1.0, 80.0)

#: ``powerline_freq = 60`` in the same script. EEG-IO was recorded in the
#: United States, so 60 Hz is right for both.
POWERLINE_HZ = 60.0

#: SNR range the models were trained over, from configs/*.yaml. The risk
#: check asks whether real blinks land inside it.
TRAIN_SNR_DB: Tuple[float, float] = (-7.0, 2.0)

#: Blink energy is low-frequency; the detector works on a smoothed copy.
BLINK_LOWPASS_HZ = 10.0

#: A detection must last at least this long to be a blink rather than a spike.
MIN_BLINK_S = 0.06

#: Candidate detections closer than this are one event.
MERGE_GAP_S = 0.10


# ============================================================
# 1. Causal preprocessing
# ============================================================
def design_filters(fs: float, band: Tuple[float, float] = TRAIN_BAND_HZ,
                   notch_hz: Optional[float] = POWERLINE_HZ,
                   order: int = 4, notch_q: float = 30.0) -> List[np.ndarray]:
    """
    Second-order sections for highpass, notch and lowpass, in that order.

    Returned as a list rather than one stacked array so a caller can drop a
    stage and say which one it dropped -- the controls do exactly that.

    The lowpass is skipped when the band's top is at or above Nyquist: at
    250 Hz, 80 Hz is comfortably inside it, but a caller running at a lower
    rate should get a filter that is simply absent rather than one silently
    clipped to something else.
    """
    from scipy.signal import butter, iirnotch, tf2sos

    nyq = fs / 2.0
    stages: List[np.ndarray] = []
    lo, hi = band
    if lo and lo > 0:
        stages.append(butter(order, lo / nyq, btype="highpass", output="sos"))
    if notch_hz and 0 < notch_hz < nyq:
        b, a = iirnotch(notch_hz / nyq, notch_q)
        stages.append(tf2sos(b, a))
    if hi and hi < nyq:
        stages.append(butter(order, hi / nyq, btype="lowpass", output="sos"))
    return stages


def preprocess(x: np.ndarray, fs: float, band: Tuple[float, float] = TRAIN_BAND_HZ,
               notch_hz: Optional[float] = POWERLINE_HZ, order: int = 4,
               zero_phase: bool = False) -> np.ndarray:
    """
    Bring a raw recording into the band the model was trained on.

    ``sosfilt`` -- one pass, forwards -- so the result at sample *t* depends
    only on samples up to *t*, which is the whole point: a non-causal filter
    in front of a causal model makes the pipeline non-causal and the model's
    guarantee worthless.

    ``zero_phase=True`` switches to ``sosfiltfilt``, which runs the filter
    forwards and then backwards.  It is what EEGdenoiseNet's own
    preprocessing did and it cannot run online.  It is here as the control
    that prices the difference, and ``preprocess_is_causal`` below is the
    check that the default really is one-directional.
    """
    from scipy.signal import sosfilt, sosfiltfilt

    y = np.asarray(x, dtype=np.float64)
    run = sosfiltfilt if zero_phase else sosfilt
    for sos in design_filters(fs, band, notch_hz, order):
        y = run(sos, y, axis=-1)
    return y.astype(np.float32)


def preprocess_is_causal(fs: float, n: int = 512, **kw) -> dict:
    """
    Perturb one sample of a flat input and look backwards, as everything else
    in this project does to a model.  A filter is the easiest place for the
    future to leak in and the hardest place to notice it.
    """
    t0 = n // 2
    a = np.zeros(n, dtype=np.float32)
    b = a.copy()
    b[t0] = 1.0
    ya, yb = preprocess(a, fs, **kw), preprocess(b, fs, **kw)
    return {"past_diff": float(np.abs(ya[:t0] - yb[:t0]).max()),
            "future_diff": float(np.abs(ya[t0:] - yb[t0:]).max()),
            "at_sample": t0}


# ============================================================
# 2. Blink intervals
# ============================================================
def _smooth(x: np.ndarray, fs: float, cutoff: float = BLINK_LOWPASS_HZ) -> np.ndarray:
    """
    Low-passed copy for the detector, zero-phase on purpose.

    The detector is analysis, not deployment: it marks where blinks *are* so
    the metrics can be computed over them, and a one-directional smoother
    would shift every interval later by its own group delay and bias the
    onset-leakage measurement by more than the effect being measured.
    """
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, min(cutoff / (fs / 2.0), 0.99), btype="lowpass", output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64)).astype(np.float32)


def detect_blinks(x: np.ndarray, fs: float, k: float = 4.0,
                  min_s: float = MIN_BLINK_S, merge_s: float = MERGE_GAP_S,
                  exclude: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
    """
    Candidate blink intervals, from the signal alone.

    A blink is a large low-frequency excursion, so: smooth, measure the
    background spread robustly (median absolute deviation, which a handful of
    blinks cannot inflate the way a standard deviation can), and take the
    runs where the excursion exceeds ``k`` times it.  Short runs are dropped
    and near neighbours merged, because one blink produces one event and the
    threshold crossing flickers around its edges.

    This is written out here rather than ported: the published Blink
    algorithm has its own detector, and the point of ``keep_annotated``
    below is that whichever detector is used, only the intervals a human
    also marked are kept -- so a false positive costs nothing and the
    detector does not have to be the published one to be fair.
    """
    s = _smooth(x, fs)
    base = np.median(s)
    dev = np.abs(s - base)
    mad = float(np.median(dev)) or float(np.mean(dev)) or 1.0
    hot = dev > k * 1.4826 * mad
    if exclude is not None:
        hot &= ~exclude

    runs: List[Tuple[int, int]] = []
    edges = np.flatnonzero(np.diff(hot.astype(np.int8)))
    bounds = ([0] if hot[0] else []) + list(edges + 1) + ([len(hot)] if hot[-1] else [])
    for a, b in zip(bounds[::2], bounds[1::2]):
        runs.append((int(a), int(b)))

    merged: List[Tuple[int, int]] = []
    gap = int(merge_s * fs)
    for a, b in runs:
        if merged and a - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(a, b) for a, b in merged if (b - a) >= int(min_s * fs)]


def keep_annotated(intervals: Sequence[Tuple[int, int]],
                   annotation_samples: Sequence[int]) -> List[Tuple[int, int]]:
    """
    Only the intervals a human also marked.

    This is what makes the detector's own errors irrelevant: a false positive
    has no annotation inside it and is dropped, and a blink the detector
    missed contributes nothing rather than contributing a wrong interval.
    What survives is a set of intervals that are blinks by both accounts.
    """
    ann = np.asarray(sorted(annotation_samples), dtype=np.int64)
    if ann.size == 0:
        return []
    out = []
    for a, b in intervals:
        if np.any((ann >= a) & (ann < b)):
            out.append((a, b))
    return out


def background_mask(n: int, intervals: Sequence[Tuple[int, int]], fs: float,
                    guard_s: float = 0.25,
                    exclude: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Everything that is not a blink, with a guard band either side.

    The guard matters: the fidelity measurement is about what the model does
    to signal it should not touch, and a sample 50 ms after a blink is still
    inside the model's reaction to it.  Corrupt stretches are excluded too --
    the annotators marked them unusable, and unusable cuts both ways.
    """
    mask = np.ones(n, dtype=bool)
    guard = int(guard_s * fs)
    for a, b in intervals:
        mask[max(a - guard, 0):min(b + guard, n)] = False
    if exclude is not None:
        mask &= ~exclude
    return mask


# ============================================================
# 3. Metrics
# ============================================================
def _power(x: np.ndarray, where: np.ndarray) -> float:
    sel = np.asarray(x, dtype=np.float64)[where]
    return float((sel ** 2).mean()) if sel.size else float("nan")


def _db(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den <= 0 or num <= 0:
        return float("nan")
    return float(10.0 * np.log10(num / den))


def _interval_mask(n: int, intervals: Sequence[Tuple[int, int]]) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for a, b in intervals:
        m[a:b] = True
    return m


def suppression_db(x: np.ndarray, y: np.ndarray,
                   intervals: Sequence[Tuple[int, int]]) -> float:
    """
    Blink-interval power removed, in dB. Positive means the model took energy
    out of the blinks; higher is more suppression. On its own it is not a
    good number -- a model that attenuated everything would score well -- so
    it is only ever read beside ``fidelity_db``.
    """
    m = _interval_mask(len(x), intervals)
    return _db(_power(x, m), _power(y, m))


def fidelity_db(x: np.ndarray, y: np.ndarray, background: np.ndarray) -> float:
    """
    Background signal-to-distortion, in dB: the power of the input the model
    should have left alone, over the power of what it changed. Higher is
    more faithful. This is the number that stops ``suppression_db`` from
    being gamed by attenuating the whole recording.
    """
    return _db(_power(x, background), _power(np.asarray(y) - np.asarray(x), background))


def residual_detectable(y: np.ndarray, fs: float,
                        kept: Sequence[Tuple[int, int]], **detect_kw) -> dict:
    """
    How many of the blinks are still findable in the output.

    The same detector, at the same threshold, run on what the model produced:
    a blink counts as still detectable if a detection overlaps the interval
    it occupied in the input. Lower is better, and unlike the two power
    measurements it is a statement about shape rather than energy.
    """
    if not len(kept):
        return {"n": 0, "still_detected": 0, "rate": float("nan")}
    found = detect_blinks(y, fs, **detect_kw)
    hit = sum(1 for a, b in kept if any(not (d < a or c >= b) for c, d in found))
    return {"n": len(kept), "still_detected": int(hit), "rate": hit / len(kept)}


def _onset_at_fraction(sig: np.ndarray, a: int, b: int, span: int,
                       frac: float) -> Optional[int]:
    """
    When ``sig`` first reaches ``frac`` of its own peak over this blink.

    Scale-free on purpose.  The obvious alternative -- first crossing of
    k times the background spread -- is not comparable between the two
    quantities this is used on: outside the blinks a well-behaved model's
    correction is almost exactly zero, so its background spread is tiny and
    it crosses immediately, while the signal's spread is the ambient EEG and
    it crosses late.  That asymmetry alone produced tens of samples of
    apparent lead for a model that anticipates nothing.  A fraction of each
    quantity's own peak has no such bias.
    """
    lo, hi = max(a - span, 0), min(b, len(sig))
    seg = sig[lo:hi]
    if seg.size == 0:
        return None
    peak = float(seg.max())
    if peak <= 0:
        return None
    hot = np.flatnonzero(seg >= frac * peak)
    return int(lo + hot[0]) if hot.size else None


def onset_leakage(x: np.ndarray, y: np.ndarray, fs: float,
                  kept: Sequence[Tuple[int, int]], background: np.ndarray,
                  frac: float = 0.1, search_s: float = 0.5) -> dict:
    """
    How far the model's correction starts before the signal itself moves.

    Two onsets, defined the same way so they can be subtracted:

        t_x   when |x - baseline| first reaches ``frac`` of its blink peak
        t_d   when |y - x|        first reaches ``frac`` of its blink peak

    and the lead is ``t_x - t_d``, in samples.  Positive means the model was
    already correcting before the artifact was visibly under way.

    Defining both as a fraction of their own peak is what makes the
    subtraction mean something: the two quantities have very different
    background levels, and any threshold set from that background measures
    the difference in levels rather than the difference in timing.

    This is the lookahead ladder's claim meeting a recording it was not
    trained on.  A strictly causal model should measure about zero; one
    licensed 16 samples ahead can lead by up to 16 (62.5 ms at 256 Hz).
    ``background`` is still taken, and still used, to centre the signal.

    **Only meaningful at ``hop=1``.**  The windowed runner hands every model
    ``hop - 1`` samples of future from its own buffer (see ``stream_apply``),
    which at any practical hop is larger than the licence being measured and
    hides it completely.  The first run of this on real recordings was at
    hop 64 and found no difference at all between a causal model and one
    licensed 62.5 ms ahead -- not because there is none, but because the
    buffer had already given both of them more than that.
    """
    x = np.asarray(x, dtype=np.float64)
    base = float(np.median(x[background])) if background.any() else float(np.median(x))
    dev = np.abs(x - base)
    d = np.abs(np.asarray(y, dtype=np.float64) - x)

    empty = {"n": len(kept), "n_measurable": 0, "median": float("nan"),
             "p90": float("nan"), "max": float("nan"),
             "median_ms": float("nan"), "p90_ms": float("nan"), "frac": frac}
    if not len(kept):
        return {**empty, "n": 0}

    span = int(search_s * fs)
    leads: List[int] = []
    for a, b in kept:
        t_x = _onset_at_fraction(dev, a, b, span, frac)
        t_d = _onset_at_fraction(d, a, b, span, frac)
        if t_x is None or t_d is None:
            continue
        leads.append(int(t_x - t_d))
    if not leads:
        return empty
    arr = np.asarray(leads, dtype=float)
    return {"n": len(kept), "n_measurable": len(leads), "frac": frac,
            "median": float(np.median(arr)), "p90": float(np.percentile(arr, 90)),
            "max": float(arr.max()),
            "median_ms": float(np.median(arr) / fs * 1e3),
            "p90_ms": float(np.percentile(arr, 90) / fs * 1e3)}


# ============================================================
# 4. The risk check
# ============================================================
def blink_snr_estimate(x: np.ndarray, kept: Sequence[Tuple[int, int]],
                       background: np.ndarray) -> dict:
    """
    Are real blinks as strong as the ones the models were trained on?

    There is no clean signal, so the estimate is the obvious one: treat the
    background as the brain signal, and the *excess* power inside a blink
    interval over that background as the artifact.

        SNR ~ 10 log10( P_background / (P_blink - P_background) )

    It assumes the brain signal continues through the blink at the same power
    and that the two add incoherently, both of which are approximations and
    both of which bias the estimate the same way (towards too high an SNR, by
    attributing some artifact power to the signal).  It is a range check, not
    a measurement: the question is whether real blinks land anywhere near the
    [-7, 2] dB the models saw, and an estimate good to a few dB answers it.
    """
    pb = _power(x, background)
    out = {"train_range_db": list(TRAIN_SNR_DB), "n": len(kept),
           "background_power": pb, "per_blink_db": []}
    for a, b in kept:
        m = np.zeros(len(x), dtype=bool)
        m[a:b] = True
        excess = _power(x, m) - pb
        out["per_blink_db"].append(_db(pb, excess) if excess > 0 else float("nan"))
    vals = np.asarray([v for v in out["per_blink_db"] if np.isfinite(v)])
    if vals.size:
        out.update(median_db=float(np.median(vals)),
                   p10_db=float(np.percentile(vals, 10)),
                   p90_db=float(np.percentile(vals, 90)),
                   inside_train_range=float(np.mean(
                       (vals >= TRAIN_SNR_DB[0]) & (vals <= TRAIN_SNR_DB[1]))))
    else:
        out.update(median_db=float("nan"), p10_db=float("nan"),
                   p90_db=float("nan"), inside_train_range=float("nan"))
    return out


# ============================================================
# 5. Running a model over a recording
# ============================================================
def stream_apply(model, x: np.ndarray, win: int = 512, hop: int = 32) -> np.ndarray:
    """
    Apply a windowed model along a recording, keeping only what a streaming
    deployment would have emitted.

    Each window is normalised by its own standard deviation and rescaled
    afterwards, exactly as ``data/Dataloader.py`` does and as the ONNX
    sidecar's preprocessing rule states; only the last ``hop`` outputs of
    each window are kept, because those are the samples a live loop would
    have had full context for.  The first window fills the head, which has
    the warm-up every causal model has at the start of a recording.

    **``hop`` is buffering latency, and it is also lookahead.**  The output
    at sample *t* comes from the window that ends at the next multiple of
    ``hop`` at or after *t*, so it may depend on inputs up to ``hop - 1``
    samples past *t*.  In a deployment that is exactly the trade the Deploy
    tab prices -- hop 32 is 125 ms of buffering -- and it is not a bug.  For
    ``onset_leakage`` it is fatal: at hop 64 every model, causal or not,
    gets up to 63 samples of effective lookahead from the buffer alone,
    which is four times the 16 samples that distinguish the models being
    compared.  Measure onset lead at ``hop=1``; the other three metrics are
    power ratios over whole intervals and are not sensitive to it.
    """
    import torch

    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    out = np.zeros(n, dtype=np.float32)
    if n < win:
        pad = np.zeros(win, dtype=np.float32)
        pad[-n:] = x
        with torch.no_grad():
            sigma = max(float(pad.std()), 1e-8)
            y = model(torch.from_numpy(pad / sigma)[None, None]).numpy()[0, 0] * sigma
        return y[-n:]

    with torch.no_grad():
        first = x[:win]
        sigma = max(float(first.std()), 1e-8)
        out[:win] = model(torch.from_numpy(first / sigma)[None, None]).numpy()[0, 0] * sigma
        for end in range(win + hop, n + hop, hop):
            end = min(end, n)
            chunk = x[end - win:end]
            sigma = max(float(chunk.std()), 1e-8)
            y = model(torch.from_numpy(chunk / sigma)[None, None]).numpy()[0, 0] * sigma
            keep = end - max(end - hop, win)
            if keep > 0:
                out[end - keep:end] = y[-keep:]
    return out


def load_run(run: str):
    """Build a model from a run directory's own checkpoint."""
    import torch

    from models.variants import build_model_from_config

    d = PROJECT_ROOT / "runs" / run if not os.path.isabs(run) else Path(run)
    ck = torch.load(d / "best.pt", map_location="cpu", weights_only=False)
    model = build_model_from_config(dict(ck["model_config"]))
    model.load_state_dict(ck["model_state"])
    return model.eval(), dict(ck["model_config"])


# ============================================================
# 6. One recording, one model
# ============================================================
@dataclass
class EvalConfig:
    band: Tuple[float, float] = TRAIN_BAND_HZ
    notch_hz: Optional[float] = POWERLINE_HZ
    zero_phase: bool = False
    target_fs: Optional[float] = PROJECT_FS
    channel: int = 0
    win: int = 512
    hop: int = 32
    detect_k: float = 4.0
    guard_s: float = 0.25
    include_soft: bool = False
    drop_corrupt: bool = True


def evaluate_recording(rec: Recording, apply_fn: Callable[[np.ndarray], np.ndarray],
                       cfg: EvalConfig = EvalConfig()) -> dict:
    """
    The four metrics and the risk check, for one subject and one model.

    Order matters and is the PI's: resample, preprocess causally, detect,
    keep only the annotated detections, drop the corrupt stretches, and
    measure everything against the preprocessed signal -- never the raw one.
    """
    fs = float(cfg.target_fs or rec.fs)
    raw = rec.signal[cfg.channel]
    sig = resample_to(raw[None, :], rec.fs, fs)[0] if fs != rec.fs else raw

    x = preprocess(sig, fs, cfg.band, cfg.notch_hz, zero_phase=cfg.zero_phase)
    n = len(x)

    corrupt = rec.corrupt_mask()
    if fs != rec.fs and corrupt.size:
        idx = np.minimum((np.arange(n) * rec.fs / fs).astype(np.int64), corrupt.size - 1)
        corrupt = corrupt[idx]
    elif corrupt.size != n:
        corrupt = np.zeros(n, dtype=bool)
    if not cfg.drop_corrupt:
        corrupt = np.zeros(n, dtype=bool)

    ann = np.round(rec.blink_times(cfg.include_soft) * fs).astype(np.int64)
    ann = ann[(ann >= 0) & (ann < n)]
    ann = ann[~corrupt[ann]] if corrupt.any() else ann

    found = detect_blinks(x, fs, k=cfg.detect_k, exclude=corrupt)
    kept = keep_annotated(found, ann)
    bg = background_mask(n, kept, fs, cfg.guard_s, exclude=corrupt)

    y = apply_fn(x)
    return {
        "subject": rec.subject, "fs": fs, "n_samples": n,
        "n_annotated": int(len(ann)), "n_detected": len(found), "n_kept": len(kept),
        "corrupt_fraction": float(corrupt.mean()) if corrupt.size else 0.0,
        "background_fraction": float(bg.mean()),
        "suppression_db": suppression_db(x, y, kept),
        "fidelity_db": fidelity_db(x, y, bg),
        "residual": residual_detectable(y, fs, kept, k=cfg.detect_k),
        "onset_leakage": onset_leakage(x, y, fs, kept, bg),
        "risk": blink_snr_estimate(x, kept, bg),
    }


def evaluate_subjects(data_dir: str, run: str, subjects: Optional[Sequence[str]] = None,
                      cfg: EvalConfig = EvalConfig(),
                      apply_fn: Optional[Callable] = None) -> dict:
    """One row per subject for one model, plus what it was run with."""
    subs = list(subjects) if subjects is not None else list_subjects(data_dir)
    arch = {}
    if apply_fn is None:
        model, arch = load_run(run)
        apply_fn = lambda x: stream_apply(model, x, cfg.win, cfg.hop)   # noqa: E731

    rows = [evaluate_recording(read_recording(data_dir, s), apply_fn, cfg) for s in subs]
    return {
        "run": run, "arch": arch, "subjects": subs, "rows": rows,
        "config": {"band_hz": list(cfg.band), "notch_hz": cfg.notch_hz,
                   "zero_phase": cfg.zero_phase, "target_fs": cfg.target_fs,
                   "win": cfg.win, "hop": cfg.hop, "detect_k": cfg.detect_k,
                   "guard_s": cfg.guard_s, "include_soft": cfg.include_soft,
                   "drop_corrupt": cfg.drop_corrupt},
        "causality": preprocess_is_causal(float(cfg.target_fs or EEGIO_FS),
                                          band=cfg.band, notch_hz=cfg.notch_hz,
                                          zero_phase=cfg.zero_phase),
        "summary": summarise(rows),
    }


METRIC_PATHS = {
    "suppression_db": ("suppression_db",),
    "fidelity_db": ("fidelity_db",),
    "residual_rate": ("residual", "rate"),
    "onset_leakage_ms": ("onset_leakage", "median_ms"),
}


def pluck(row: dict, metric: str) -> float:
    v = row
    for key in METRIC_PATHS[metric]:
        v = v.get(key, {}) if isinstance(v, dict) else float("nan")
    return float(v) if isinstance(v, (int, float)) else float("nan")


def summarise(rows: Sequence[dict]) -> dict:
    out = {}
    for metric in METRIC_PATHS:
        vals = np.asarray([pluck(r, metric) for r in rows], dtype=float)
        vals = vals[np.isfinite(vals)]
        out[metric] = {"n": int(vals.size),
                       "median": float(np.median(vals)) if vals.size else float("nan"),
                       "mean": float(vals.mean()) if vals.size else float("nan"),
                       "sd": float(vals.std(ddof=1)) if vals.size > 1 else float("nan")}
    return out


# ============================================================
# 7. Comparing two models, paired by subject
# ============================================================
def paired_wilcoxon(a: dict, b: dict, metric: str) -> dict:
    """
    Wilcoxon signed-rank over the subjects both models were scored on.

    Paired by subject because subjects differ enormously -- electrode
    placement, how hard someone blinks -- and that variance would swamp the
    difference between two models if the two groups were compared unpaired.
    It is the non-parametric test because 20 subjects is not enough to argue
    the differences are normal.
    """
    from scipy.stats import wilcoxon

    av = {r["subject"]: pluck(r, metric) for r in a["rows"]}
    bv = {r["subject"]: pluck(r, metric) for r in b["rows"]}
    shared = [s for s in av if s in bv and np.isfinite(av[s]) and np.isfinite(bv[s])]
    diffs = [bv[s] - av[s] for s in shared]
    out = {"metric": metric, "a": a["run"], "b": b["run"], "n_pairs": len(shared),
           "median_diff": float(np.median(diffs)) if diffs else float("nan"),
           "b_better_on": int(sum(d > 0 for d in diffs)),
           "subjects": shared}
    if len(shared) < 3 or not np.any(np.asarray(diffs) != 0):
        out.update(statistic=float("nan"), p_value=float("nan"),
                   note="too few paired subjects, or no difference at all, to test")
        return out
    stat, p = wilcoxon([av[s] for s in shared], [bv[s] for s in shared])
    out.update(statistic=float(stat), p_value=float(p))
    return out


def compare(results: Sequence[dict], metrics: Sequence[str] = tuple(METRIC_PATHS)) -> List[dict]:
    """Every result against the first, on every metric."""
    if len(results) < 2:
        return []
    base = results[0]
    return [paired_wilcoxon(base, other, m) for other in results[1:] for m in metrics]


# ============================================================
# CLI
# ============================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate denoisers on real EEG-IO recordings")
    from gui.eegio_tab import find_data_dir
    ap.add_argument("--data_dir", default=str(find_data_dir()),
                    help="EEG-IO recordings; data/EEG-IO and "
                         "data/EEG-EyeBlinks/EEG-IO are both found automatically")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run directories; the first is the baseline the rest are tested against")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--hop", type=int, default=32)
    ap.add_argument("--zero_phase", action="store_true",
                    help="control: the non-causal filter EEGdenoiseNet itself used")
    ap.add_argument("--no_notch", action="store_true")
    ap.add_argument("--include_soft", action="store_true")
    ap.add_argument("--json", default=None, help="write the full result here")
    a = ap.parse_args(argv)

    cfg = EvalConfig(channel=a.channel, hop=a.hop, zero_phase=a.zero_phase,
                     notch_hz=None if a.no_notch else POWERLINE_HZ,
                     include_soft=a.include_soft)
    results = [evaluate_subjects(a.data_dir, Path(r).name, a.subjects, cfg) for r in a.runs]

    c = results[0]["causality"]
    print(f"\n  preprocessing: {'zero-phase (NON-CAUSAL control)' if cfg.zero_phase else 'causal, sosfilt'}"
          f"  past_diff={c['past_diff']:.2e}")
    print(f"  {'run':<26}{'supp dB':>9}{'fidel dB':>10}{'resid':>8}{'lead ms':>9}{'subj':>6}")
    for r in results:
        s = r["summary"]
        print(f"  {r['run']:<26}{s['suppression_db']['median']:>9.2f}"
              f"{s['fidelity_db']['median']:>10.2f}{s['residual_rate']['median']:>8.2f}"
              f"{s['onset_leakage_ms']['median']:>9.1f}{len(r['rows']):>6}")

    for t in compare(results):
        if t["n_pairs"]:
            print(f"  {t['b']} vs {t['a']} · {t['metric']:<18} "
                  f"median Δ {t['median_diff']:+.3f}  p={t['p_value']:.4f}  n={t['n_pairs']}")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"results": results, "tests": compare(results)}, indent=2))
        print(f"\n  wrote {a.json}")
    return results


if __name__ == "__main__":
    main()
