"""
EEG-IO: real frontal EEG with hand-annotated eye blinks.

The counterpart to ``data/Dataloader.py``, and the opposite kind of dataset.
EEGdenoiseNet gives a *synthetic* mixture: a clean EEG epoch plus an artifact
epoch at an SNR we choose, so the clean signal is known exactly and RRMSE is
computable.  EEG-IO gives *real recordings* -- frontal EEG that was actually
contaminated while someone blinked -- with the blink instants annotated by
hand from a video feed, and **no clean reference at all**.

That difference decides what it can be used for, so it is worth being blunt
about it here rather than discovering it downstream:

* It **cannot** train or score a denoiser the way EEGdenoiseNet does.  There
  is no target, so there is no RRMSE_t, no RRMSE_f and no CC.  Nothing in
  this module returns a ``clean`` key, precisely so that a model-scoring loop
  written against ``EEGDenoiseDataset`` fails loudly instead of silently
  scoring against something invented.
* It **can** say what a model does to a real blink.  Every window carries a
  mask of the annotated blink instants inside it, so a denoiser's output can
  be compared at blink times against the same recording away from them --
  which is a measurement about artifact suppression that needs no target.
* It **can** supply real non-stationarity.  The artifact-switch probe in
  ``Dataloader.py`` splices two synthetic regimes together at a drawn index;
  here the onsets are real events with real timestamps, and the project's
  bracketing result (a sample inside a blink is resolved by seeing both of
  its edges) makes a prediction about them that can be checked.

Format, from the reference implementation of the Blink algorithm that was
released with the dataset (``meagmohit/BLINK``, ``blink.py``):

    data/EEG-IO/
    ├── S00_data.csv     ';'-delimited, one header row, columns
    │                      (time_seconds, ch1, ch2)
    ├── S00_labels.csv   ','-delimited, no header, three kinds of row:
    │                      "corrupt",<n>   -- the next <n> rows are intervals
    │                      <start>,<end>   -- a corrupt interval; end -1
    │                                         means "to the end of the file"
    │                      <time>,<code>   -- a blink, code 2 = soft blink
    └── ...

Twenty subjects, frontal electrodes (Fp1, Fp2), OpenBCI with a BIOPAC
Cap100C, about 25 blinks per subject. Sessions run 77-103 s at 250 Hz (S06
is the short one); the dataset page's "75 to 100 seconds" is approximate, and
the lengths are not all equal, so nothing here may assume a fixed one.
``fs`` is 250 Hz -- taken from the reference implementation's own ``fs =
250.0``, not from the dataset page, which does not state it.  (The "500" in
the paper's dataset table is the blink count across the 20 subjects, not a
rate.)

The data is not in this repository and is not redistributable: it is obtained
through the registration form on the dataset page, which
``require_dataset()`` prints when the directory is missing.

One thing this module cannot verify, flagged where it is used: what the blink
codes below 2 mean, since the reference implementation only ever separates
``< 2`` from ``== 2``. The channel order is no longer among them -- a real
``_data.csv`` names its columns ``Time (s);FP1;FP2;…`` in its header row.
"""

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

#: Sampling rate of every EEG-IO recording, in Hz.
#:
#: Confirmed three ways, because everything downstream is placed by it and
#: the timestamp column looks wrong at first glance.  Each ``_data.csv``
#: carries a ``Sampling Rate`` column reading 250; its timestamps span
#: ``n/250`` seconds (250.001 Hz measured across subjects); and converting
#: the hand annotations to samples at 250 Hz lands them on a large
#: low-frequency excursion 32% of the time against 8% at 256 Hz, which is
#: chance by construction of that test.
#:
#: The trap is the *step* between timestamps, which is 1/256 for 90% of
#: samples and so reads as a 256 Hz clock if only the median is taken.  The
#: real resolution is 1/1024 s: a 250 Hz sample is 4.096 of those ticks, so
#: the column emits 4 ticks 90.4% of the time and 5 ticks 9.6% of the time,
#: averaging 4.0960 -- which is 1024/250 to seven figures.  1/256 is simply
#: what 4 ticks happen to equal.  The acquisition is 250 Hz; the data has
#: not been resampled to anything.
EEGIO_FS = 250.0

#: The two frontal channels, in the column order the data file stores them.
#: Confirmed from the files themselves: the header row of a real EEG-IO
#: ``_data.csv`` reads ``Time (s);FP1;FP2;Channel 3;…``, so column 1 is Fp1
#: and column 2 is Fp2. (The later columns are unused electrodes parked at a
#: constant, and the last carries the sampling rate on the first row only.)
EEGIO_CHANNELS: Tuple[str, ...] = ("Fp1", "Fp2")

#: Label code the reference implementation treats as a *soft* blink.  It
#: separates ``< 2`` from ``== 2`` and nothing finer, so the meaning of 0
#: versus 1 is not documented and is not interpreted here either.
SOFT_BLINK_CODE = 2

#: Where to get the data.  Printed by require_dataset() rather than buried in
#: a traceback, because the download is behind a form and cannot be scripted.
DATASET_URL = "https://gnan.ece.gatech.edu/eeg-eyeblinks"

#: The rate this project's models were trained at.  EEG-IO is 250 Hz, so a
#: window has to be resampled before a checkpoint can be run on it; 250 -> 256
#: is the exact rational ratio 128/125.
PROJECT_FS = 256.0


# ============================================================
# Reading one recording
# ============================================================
@dataclass
class Recording:
    """One subject's session: the signal, the blinks, the unusable stretches."""

    subject: str
    signal: np.ndarray          # (n_channels, n_samples), float32
    #: Seconds, on the ``fs`` clock: ``arange(n) / fs``. Regular by
    #: construction, unlike the file's own timestamp column, which is
    #: quantised to 1/1024 and steps unevenly (see EEGIO_FS). Everything here
    #: -- blink_samples, corrupt_mask, duration_s -- is on this axis, so they
    #: cannot disagree with each other.
    t: np.ndarray
    fs: float
    blinks: np.ndarray          # (n_blinks, 2) -> time_s, type_code
    corrupt: np.ndarray         # (n_intervals, 2) -> start_s, end_s
    channels: Tuple[str, ...]
    #: The timestamp column exactly as the file stores it, for provenance.
    #: Kept so its jitter can be inspected rather than taken on trust;
    #: nothing in this module computes with it.
    t_recorded: Optional[np.ndarray] = None

    @property
    def n_samples(self) -> int:
        return self.signal.shape[1]

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.fs

    def blink_times(self, include_soft: bool = False) -> np.ndarray:
        """
        Annotated blink instants, in seconds.

        ``include_soft`` is the reference implementation's ``flag_soft``: a
        soft blink is annotated but is a weaker event, and whether it counts
        as ground truth is the caller's decision, not this module's.
        """
        if len(self.blinks) == 0:
            return np.zeros(0, dtype=np.float64)
        keep = (self.blinks[:, 1] < SOFT_BLINK_CODE)
        if include_soft:
            keep |= (self.blinks[:, 1] == SOFT_BLINK_CODE)
        return self.blinks[keep, 0]

    def blink_samples(self, include_soft: bool = False) -> np.ndarray:
        """The same instants as sample indices into ``signal``."""
        times = self.blink_times(include_soft)
        idx = np.round(times * self.fs).astype(np.int64)
        return idx[(idx >= 0) & (idx < self.n_samples)]

    def corrupt_mask(self) -> np.ndarray:
        """
        ``(n_samples,)`` boolean, True where the annotator marked the
        recording unusable.  An interval ending at -1 in the file has already
        been resolved to the end of the recording by ``read_recording``.
        """
        mask = np.zeros(self.n_samples, dtype=bool)
        for start, end in self.corrupt:
            a = max(int(np.floor(start * self.fs)), 0)
            b = min(int(np.ceil(end * self.fs)) + 1, self.n_samples)
            if b > a:
                mask[a:b] = True
        return mask


def list_subjects(data_dir: str) -> List[str]:
    """
    Every subject with a ``*_data.csv`` in ``data_dir``, sorted.

    The stem is whatever precedes ``_data``; the dataset ships ``S00``,
    ``S01``, ... but nothing here depends on that spelling.
    """
    require_dataset(data_dir)
    stems = [f[: -len("_data.csv")] for f in os.listdir(data_dir)
             if f.endswith("_data.csv")]
    return sorted(stems)


def require_dataset(data_dir: str) -> None:
    """Fail early, and say how to fix it: the download is behind a form."""
    if os.path.isdir(data_dir) and any(f.endswith("_data.csv")
                                       for f in os.listdir(data_dir)):
        return
    raise FileNotFoundError(
        f"No EEG-IO recordings in {data_dir!r}.\n"
        f"  EEG-IO is not redistributable and is not in this repository.\n"
        f"  Request it from {DATASET_URL} and unpack it so that the directory\n"
        f"  holds <subject>_data.csv / <subject>_labels.csv pairs.\n"
        f"  To exercise the pipeline without it, see write_synthetic_recording()."
    )


def _read_labels(path: str, t_last: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse ``<subject>_labels.csv`` into (blinks, corrupt intervals).

    The layout is the reference implementation's: an optional ``corrupt,<n>``
    row announces that the next ``<n>`` rows are intervals rather than
    blinks, a bare ``blinks`` row separates the two blocks, and an interval
    end of -1 means "to the end of the recording", which is resolved here so
    that no consumer has to know the convention.

    ``blinks`` is matched by name rather than left to fall through the
    too-few-fields branch below.  Every real file writes it on a line of its
    own, so either works today; one written ``blinks,25`` would be read as a
    blink at time "blinks" and crash, which is a silly way to lose a dataset.
    """
    blinks: List[Tuple[float, int]] = []
    corrupt: List[Tuple[float, float]] = []
    pending = 0
    with open(path, newline="") as fh:
        for row in csv.reader(fh, delimiter=","):
            row = [c.strip() for c in row if c.strip() != ""]
            if not row or row[0].lower() == "blinks":
                continue
            if len(row) < 2:
                continue
            if row[0].lower() == "corrupt":
                pending = int(float(row[1]))
                continue
            if pending > 0:
                end = t_last if float(row[1]) == -1 else float(row[1])
                corrupt.append((float(row[0]), end))
                pending -= 1
                continue
            blinks.append((float(row[0]), int(float(row[1]))))

    b = np.asarray(blinks, dtype=np.float64).reshape(-1, 2)
    c = np.asarray(corrupt, dtype=np.float64).reshape(-1, 2)
    return b[np.argsort(b[:, 0])] if len(b) else b, c


def read_recording(data_dir: str, subject: str) -> Recording:
    """
    One subject's data file and its label file, as a ``Recording``.

    The data file is ';'-delimited with one header row and its first column
    is the timestamp in seconds, which is why only columns 0-2 are read: a
    row may carry further columns this dataset does not use.
    """
    sig_path = os.path.join(data_dir, f"{subject}_data.csv")
    lab_path = os.path.join(data_dir, f"{subject}_labels.csv")
    for p in (sig_path, lab_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{p} is missing; EEG-IO ships the pair together")

    raw = np.loadtxt(sig_path, delimiter=";", skiprows=1, usecols=(0, 1, 2))
    raw = np.atleast_2d(raw)
    if raw.shape[1] != 3:
        raise ValueError(f"{sig_path}: expected 3 columns (time, ch1, ch2), "
                         f"got {raw.shape[1]}")
    if raw.shape[0] == 0:
        raise ValueError(f"{sig_path}: no samples")
    if not np.isfinite(raw[:, 1:]).all():
        n_bad = int((~np.isfinite(raw[:, 1:])).sum())
        raise ValueError(f"{sig_path}: {n_bad} non-finite sample(s) in the EEG columns")

    signal = np.ascontiguousarray(raw[:, 1:].T, dtype=np.float32)   # (2, N)
    n = signal.shape[1]
    # The canonical axis is the rate's, not the file's: the recorded
    # timestamps are quantised to 1/1024 and so step unevenly (see EEGIO_FS),
    # and an axis that jitters is a poor thing to place annotations on even
    # when its average is right.  The two agree to under a millisecond over
    # a 100 s recording, so this is tidiness rather than a correction.
    t = np.arange(n, dtype=np.float64) / EEGIO_FS
    blinks, corrupt = _read_labels(lab_path, float(n / EEGIO_FS))

    return Recording(subject=subject, signal=signal, t=t, fs=EEGIO_FS,
                     blinks=blinks, corrupt=corrupt, channels=EEGIO_CHANNELS,
                     t_recorded=raw[:, 0].astype(np.float64))


# ============================================================
# Rate conversion
# ============================================================
def resample_to(signal: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """
    Resample ``(n_channels, n_samples)`` from ``fs_in`` to ``fs_out``.

    EEG-IO is 250 Hz and every model in this project was trained at 256, so a
    checkpoint cannot be run on a raw EEG-IO window at all.  The ratio is
    exactly 128/125, so this is polyphase resampling with no approximation in
    the rate itself; ``scipy`` is already a dependency of this project.
    """
    if fs_in == fs_out:
        return np.ascontiguousarray(signal, dtype=np.float32)
    from fractions import Fraction

    from scipy.signal import resample_poly

    ratio = Fraction(fs_out).limit_denominator(10_000) / Fraction(fs_in).limit_denominator(10_000)
    out = resample_poly(np.asarray(signal, dtype=np.float64),
                        ratio.numerator, ratio.denominator, axis=-1)
    return np.ascontiguousarray(out, dtype=np.float32)


# ============================================================
# Subject-level splits
# ============================================================
def split_subjects(subjects: Sequence[str],
                   ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                   seed: int = 42) -> Dict[str, List[str]]:
    """
    Partition subjects into train/val/test.

    By **subject**, not by window, and that is the whole point: two windows
    from one recording share a session, an electrode placement and a person,
    so splitting windows would leak all three and report a model as
    generalising when it had merely memorised a recording.  This is the same
    guarantee ``Dataloader._split_indices`` gives at the epoch level, one
    level coarser because here the unit of independence is coarser.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1.0"
    subs = sorted(subjects)
    perm = np.random.default_rng(seed).permutation(len(subs))
    n_tr = int(len(subs) * ratios[0])
    n_va = int(len(subs) * ratios[1])
    take = lambda sl: sorted(subs[i] for i in perm[sl])      # noqa: E731
    return {"train": take(slice(0, n_tr)),
            "val": take(slice(n_tr, n_tr + n_va)),
            "test": take(slice(n_tr + n_va, None))}


# ============================================================
# Windowed dataset
# ============================================================
class EEGIOWindows(Dataset):
    """
    Sliding windows over EEG-IO recordings, with their blink annotations.

    Emits the same tensor contract as ``EEGDenoiseDataset`` for the input
    side -- ``(n_channels, win)`` normalised by the window's own sigma -- so a
    checkpoint trained on EEGdenoiseNet can be run on it directly.  It
    deliberately emits **no** ``clean`` key: there is no clean reference for a
    real recording, and a loop that averages RRMSE should fail on a missing
    key rather than quietly score against a substitute.

    Parameters
    ----------
    data_dir : str
        Directory holding ``<subject>_data.csv`` / ``<subject>_labels.csv``.
    subjects : sequence of str, optional
        Which recordings to use.  Default: all of them.  Pass one arm of
        ``split_subjects`` to keep subjects out of each other's splits.
    channels : sequence of str or int
        Channel names (``"Fp1"``, ``"Fp2"``) or column indices.  Default is
        the first channel alone, which is what the reference implementation
        detects blinks on and what this project's single-channel models
        expect.
    win, hop : int
        Window length and stride, in samples **at the output rate**.  The
        defaults are the 512-sample, 2-second epoch the project trains on,
        with 50 % overlap.
    target_fs : float or None
        Resample each recording to this rate before windowing.  Defaults to
        256 Hz, the rate the models were trained at; ``None`` keeps the
        native 250 Hz, which is right for analysis that does not involve a
        checkpoint.
    include_soft : bool
        Count soft blinks (code 2) as blinks in the mask and the counts.
    drop_corrupt : bool
        Skip windows that overlap an annotator-marked corrupt interval.  On
        by default: those stretches are marked precisely because the signal
        there is not usable evidence.
    blink_halfwidth_s : float
        An annotation is an *instant*, while a blink is an event of some
        length, so the mask marks this much either side of each instant.  The
        default 0.2 s spans the 200-400 ms onset-peak-offset morphology this
        project's EOG results describe.
    normalise : bool
        Divide each window by its own standard deviation, as
        ``EEGDenoiseDataset`` does, and return the divisor as ``sigma``.
    """

    def __init__(
        self,
        data_dir: str,
        subjects: Optional[Sequence[str]] = None,
        channels: Sequence = ("Fp1",),
        win: int = 512,
        hop: int = 256,
        target_fs: Optional[float] = PROJECT_FS,
        include_soft: bool = False,
        drop_corrupt: bool = True,
        blink_halfwidth_s: float = 0.2,
        normalise: bool = True,
    ):
        assert win > 0 and hop > 0, "win and hop must be positive"
        assert blink_halfwidth_s >= 0, "blink_halfwidth_s must be >= 0"

        self.data_dir = data_dir
        self.win, self.hop = int(win), int(hop)
        self.include_soft = bool(include_soft)
        self.drop_corrupt = bool(drop_corrupt)
        self.normalise = bool(normalise)
        self.fs = float(target_fs) if target_fs else EEGIO_FS
        self.blink_halfwidth = int(round(blink_halfwidth_s * self.fs))

        subs = list(subjects) if subjects is not None else list_subjects(data_dir)
        if not subs:
            raise ValueError("no subjects selected")
        self.subjects = subs
        self.channel_names = tuple(
            EEGIO_CHANNELS[c] if isinstance(c, int) else str(c) for c in channels)
        cols = [c if isinstance(c, int) else EEGIO_CHANNELS.index(c) for c in channels]

        # One pass over the recordings: resample, build the blink and corrupt
        # masks at the output rate, and record which window starts survive.
        # Windows are indexed rather than materialised, so a long recording
        # costs one copy of itself and not one copy per window.
        self._signals: List[np.ndarray] = []
        self._blink_masks: List[np.ndarray] = []
        self._index: List[Tuple[int, int]] = []       # (recording, start sample)
        self.dropped_corrupt = 0

        for r_i, sub in enumerate(subs):
            rec = read_recording(data_dir, sub)
            sig = resample_to(rec.signal[cols], rec.fs, self.fs)
            n = sig.shape[1]

            blink = np.zeros(n, dtype=bool)
            for s in np.round(rec.blink_times(self.include_soft) * self.fs).astype(np.int64):
                a, b = max(s - self.blink_halfwidth, 0), min(s + self.blink_halfwidth + 1, n)
                if b > a:
                    blink[a:b] = True

            corrupt = rec.corrupt_mask()
            if self.fs != rec.fs and len(corrupt):
                # A boolean mask must not be filtered; resample it by index.
                src = np.minimum((np.arange(n) * rec.fs / self.fs).astype(np.int64),
                                 len(corrupt) - 1)
                corrupt = corrupt[src]
            elif not len(corrupt):
                corrupt = np.zeros(n, dtype=bool)

            self._signals.append(sig)
            self._blink_masks.append(blink)
            for start in range(0, max(n - self.win + 1, 0), self.hop):
                if self.drop_corrupt and corrupt[start:start + self.win].any():
                    self.dropped_corrupt += 1
                    continue
                self._index.append((r_i, start))

        print(f"[EEG-IO] {len(subs)} subject(s) | {len(self._index)} windows of "
              f"{self.win} @ {self.fs:g} Hz | channels={list(self.channel_names)} | "
              f"soft blinks {'in' if self.include_soft else 'out'}"
              + (f" | {self.dropped_corrupt} windows dropped as corrupt"
                 if self.dropped_corrupt else ""))

    # -------- Dataset interface --------
    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        r_i, start = self._index[idx]
        stop = start + self.win
        x = self._signals[r_i][:, start:stop]
        blink = self._blink_masks[r_i][start:stop]

        sigma = 1.0
        if self.normalise:
            sigma = max(float(x.std()), 1e-8)
            x = x / sigma

        return {
            # Same shape and scale contract as EEGDenoiseDataset's "noisy".
            # There is no "clean": see this module's docstring.
            "noisy": torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
            "sigma": torch.tensor(float(sigma), dtype=torch.float32),
            "blink_mask": torch.from_numpy(blink.copy()),
            "n_blinks": torch.tensor(int(_count_runs(blink)), dtype=torch.long),
            "subject": self.subjects[r_i],
            "start_sample": torch.tensor(int(start), dtype=torch.long),
        }

    # -------- description --------
    def summary(self) -> dict:
        """Counts a caller would otherwise recompute to describe the split."""
        blinks = sum(int(_count_runs(m)) for m in self._blink_masks)
        samples = sum(int(s.shape[1]) for s in self._signals)
        return {
            "subjects": list(self.subjects),
            "n_windows": len(self._index),
            "n_blink_events": blinks,
            "seconds": samples / self.fs,
            "fs": self.fs,
            "channels": list(self.channel_names),
            "windows_dropped_corrupt": self.dropped_corrupt,
        }


def _count_runs(mask: np.ndarray) -> int:
    """Contiguous True runs in a boolean mask, i.e. distinct blink events."""
    if mask.size == 0:
        return 0
    return int(np.count_nonzero(np.diff(mask.astype(np.int8), prepend=0) == 1))


def build_eegio_loader(
    data_dir: str,
    split: Optional[str] = None,
    batch_size: int = 32,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    shuffle: bool = False,
    **kwargs,
) -> DataLoader:
    """
    A DataLoader over one subject-level split (or all subjects if ``split``
    is None).  Keyword arguments go straight to ``EEGIOWindows``.

    Not shuffled by default: the usual use is evaluation, where window order
    is how a response is read back against the recording.
    """
    subs = kwargs.pop("subjects", None)
    if subs is None and split is not None:
        subs = split_subjects(list_subjects(data_dir), split_ratios, seed)[split]
    ds = EEGIOWindows(data_dir, subjects=subs, **kwargs)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=_collate)


def _collate(items: List[dict]) -> dict:
    """Default collation, except that ``subject`` stays a list of strings."""
    out = {k: torch.stack([it[k] for it in items]) for k in items[0]
           if k != "subject"}
    out["subject"] = [it["subject"] for it in items]
    return out


# ============================================================
# A stand-in, for tests and for trying the pipeline without the download
# ============================================================
def write_synthetic_recording(
    data_dir: str,
    subject: str = "S00",
    duration_s: float = 20.0,
    n_blinks: int = 6,
    n_soft: int = 2,
    corrupt: Sequence[Tuple[float, float]] = (),
    seed: int = 0,
    amplitude: float = 30.0,
    drift: float = 1.0,
) -> Recording:
    """
    Write a fake subject in EEG-IO's exact on-disk format and return it.

    This is **not** EEG-IO and must never be reported as it: the signal is
    pink-ish noise with a blink-shaped bump at each annotated instant.  It
    exists so that the reader, the windowing, the masks and the corrupt-interval
    handling can be tested, and so that someone without the download can see
    the pipeline run end to end.

    ``amplitude`` and ``drift`` set how far the bumps stand out of the
    background.  The defaults give a drift-dominated recording, which is what
    a reader test wants -- nothing about parsing a CSV should depend on the
    signal being clean.  Anything that detects blinks wants the opposite, and
    ``drift=0`` with a large amplitude gives a recording where the bumps are
    the only large excursions.
    """
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    n = int(duration_s * EEGIO_FS)
    t = np.arange(n) / EEGIO_FS

    sig = rng.normal(0.0, 2.0, size=(2, n))
    if drift:
        sig = sig + drift * np.cumsum(rng.normal(0.0, 1.0, size=(2, n)), axis=1)

    edge = 1.5
    times = np.sort(rng.uniform(edge, max(duration_s - edge, edge + 1e-3),
                                size=n_blinks + n_soft))
    codes = np.array([0] * n_blinks + [SOFT_BLINK_CODE] * n_soft)
    rng.shuffle(codes)
    for ts, code in zip(times, codes):
        c = int(round(ts * EEGIO_FS))
        half = int(0.15 * EEGIO_FS)
        a, b = max(c - half, 0), min(c + half, n)
        scale = amplitude if code != SOFT_BLINK_CODE else amplitude / 3.0
        bump = np.hanning(b - a) * scale
        sig[:, a:b] += bump

    with open(os.path.join(data_dir, f"{subject}_data.csv"), "w") as fh:
        fh.write("time;ch1;ch2\n")
        for i in range(n):
            fh.write(f"{t[i]:.6f};{sig[0, i]:.6f};{sig[1, i]:.6f}\n")

    with open(os.path.join(data_dir, f"{subject}_labels.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        if corrupt:
            w.writerow(["corrupt", len(corrupt)])
            for a, b in corrupt:
                w.writerow([f"{a:.6f}", f"{b:.6f}"])
        for ts, code in zip(times, codes):
            w.writerow([f"{ts:.6f}", int(code)])

    return read_recording(data_dir, subject)


# ============================================================
# Entry point: describe what is on disk
# ============================================================
if __name__ == "__main__":
    import argparse

    default_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "EEG-IO",
    )
    ap = argparse.ArgumentParser(description="Describe the EEG-IO recordings on disk")
    ap.add_argument("--data_dir", default=default_dir)
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--target_fs", type=float, default=PROJECT_FS)
    ap.add_argument("--include_soft", action="store_true")
    args = ap.parse_args()

    subs = list_subjects(args.data_dir)
    print(f"\n{len(subs)} subject(s) in {args.data_dir}\n")
    print(f"  {'subject':<10}{'seconds':>9}{'blinks':>8}{'soft':>6}{'corrupt':>9}")
    for s in subs:
        rec = read_recording(args.data_dir, s)
        hard = int((rec.blinks[:, 1] < SOFT_BLINK_CODE).sum()) if len(rec.blinks) else 0
        soft = int((rec.blinks[:, 1] == SOFT_BLINK_CODE).sum()) if len(rec.blinks) else 0
        corrupt_s = float(np.sum(rec.corrupt[:, 1] - rec.corrupt[:, 0])) if len(rec.corrupt) else 0.0
        print(f"  {s:<10}{rec.duration_s:>9.1f}{hard:>8}{soft:>6}{corrupt_s:>8.1f}s")

    ds = EEGIOWindows(args.data_dir, win=args.win, hop=args.hop,
                      target_fs=args.target_fs, include_soft=args.include_soft)
    print("\n", ds.summary())
