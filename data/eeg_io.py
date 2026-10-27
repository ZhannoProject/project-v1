"""
EEG-IO: reader and epoch cutter for a dataset of *real* eye-blink artifacts.

    python -m data.eeg_io --data_dir data/EEG-IO                 # inspect
    python -m data.eeg_io --data_dir data/EEG-IO --out deploy/testdata/eeg_io.npz

Where EEGdenoiseNet gives us a clean epoch and an artifact epoch to mix at a
known SNR, EEG-IO gives us the opposite: genuine frontal EEG with genuine
blinks in it, annotated by hand from a video feed, and **no clean reference
whatsoever**.  Nothing here produces a ``clean`` array, because there is no
honest way to produce one -- see "What this cannot measure" below.

Dataset
-------
20 subjects, OpenBCI headset, electrodes Fp1 and Fp2, 250 Hz, ~500 blinks.
Subjects were asked to blink when an external stimulus was given.  From
Agarwal & Sivakumar, "BLINK: A Fully Automated Unsupervised Algorithm for
Eye-Blink Detection in EEG Signals", 57th Allerton Conference, IEEE, 2019.
The data is request-gated at http://gnan.ece.gatech.edu/eeg-eyeblinks/ and is
not redistributed here; ``data/EEG-IO/`` is where this module expects it.

File format (as documented in meagmohit/BLINK's data/EEG-IO/README.md)
----------------------------------------------------------------------
``S<ID>_data.csv``   ``;``-delimited, one header row.  Column 0 is time in
                     seconds, columns 1 and 2 are Fp1 and Fp2 in microvolts.
                     Later columns, if any, are ignored.
``S<ID>_labels.csv`` ``,``-delimited, no header:
                       * ``corrupt,<n>``  -- n corrupt intervals follow
                       * n rows of ``<start>,<end>`` in seconds; an end of
                         ``-1`` means "to the end of the recording"
                       * ``blinks``       -- section marker
                       * rows of ``<blink_time>,<code>``, where blink_time is
                         the *middle* of the blink (the EEG minimum) and code
                         is 0 normal, 1 stimulated, 2 soft.

Sample rate
-----------
EEG-IO is 250 Hz; this project's models are trained on EEGdenoiseNet at
256 Hz, and a 512-sample window therefore means 2.000 s to the model but
2.048 s of EEG-IO.  ``resample_to_project_rate`` closes that 2.4 % gap
exactly (128/125 polyphase) and is on by default, so a blink occupies the
number of samples the model was trained to expect.  ``--native_rate`` keeps
250 Hz if you want to measure what the mismatch costs.

What this cannot measure
------------------------
RRMSE_t, RRMSE_s and ACC (metrics.py) are all defined against a clean
reference x, so none of them can be computed on EEG-IO.  What the blink
annotations *do* support is a reference-free pair of measurements -- how much
a denoiser suppresses the signal inside annotated blink windows, against how
little it changes blink-free windows from the same recordings.  Either number
alone is meaningless (a filter that outputs zero wins the first outright), so
``cut_epochs`` returns both populations, tagged.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NATIVE_RATE_HZ = 250.0          # EEG-IO, as recorded
PROJECT_RATE_HZ = 256.0         # EEGdenoiseNet, what the models were trained on
WINDOW = 512                    # the models' input length, in samples
CHANNEL_NAMES = ("Fp1", "Fp2")  # data.csv columns 1 and 2

# labels.csv blink codes, verbatim from the dataset README
BLINK_CODES = {0: "normal", 1: "stimulated", 2: "soft"}
SOFT_BLINK_CODE = 2

_SUBJECT_RE = re.compile(r"^(?P<subject>S\d+)_data\.csv$", re.IGNORECASE)


# ============================================================
# Labels
# ============================================================
@dataclass
class Labels:
    """Parsed ``S<ID>_labels.csv``."""
    corrupt: List[Tuple[float, float]] = field(default_factory=list)
    blink_times: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    blink_codes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def blinks(self, include_soft: bool = True) -> np.ndarray:
        """Blink midpoints in seconds; ``include_soft=False`` drops code 2."""
        if include_soft:
            return self.blink_times
        return self.blink_times[self.blink_codes != SOFT_BLINK_CODE]


def read_labels(path: str | Path, duration_s: Optional[float] = None) -> Labels:
    """
    Parse a labels file.

    ``duration_s`` resolves a corrupt interval whose end is ``-1`` ("to the
    end of the recording").  Without it such an interval is left as ``inf``,
    which still excludes the right samples once compared against a time axis.
    """
    path = Path(path)
    open_end = float("inf") if duration_s is None else float(duration_s)

    corrupt: List[Tuple[float, float]] = []
    times: List[float] = []
    codes: List[int] = []

    # The file is a three-part sequence, not a table: a corrupt header, that
    # many interval rows, a 'blinks' marker, then blink rows.  Tracking the
    # expected number of intervals is what keeps an interval row from being
    # read as a blink and vice versa.
    expected_intervals = 0
    seen_corrupt_header = False
    seen_blinks_marker = False

    with open(path, newline="") as fh:
        for lineno, row in enumerate(csv.reader(fh), start=1):
            row = [cell.strip() for cell in row if cell.strip() != ""]
            if not row:
                continue
            tag = row[0].lower()

            if tag == "corrupt":
                if seen_corrupt_header:
                    raise ValueError(f"{path}:{lineno}: a second 'corrupt' header")
                if len(row) < 2:
                    raise ValueError(f"{path}:{lineno}: 'corrupt' without a count")
                expected_intervals = int(float(row[1]))
                if expected_intervals < 0:
                    raise ValueError(f"{path}:{lineno}: negative corrupt count {expected_intervals}")
                seen_corrupt_header = True
            elif expected_intervals > 0:
                if tag == "blinks":
                    raise ValueError(
                        f"{path}:{lineno}: 'blinks' marker reached with {expected_intervals} "
                        f"corrupt interval(s) still promised by the header"
                    )
                if len(row) < 2:
                    raise ValueError(f"{path}:{lineno}: corrupt interval needs start,end")
                start, end = float(row[0]), float(row[1])
                corrupt.append((start, open_end if end == -1 else end))
                expected_intervals -= 1
            elif tag == "blinks":
                if seen_blinks_marker:
                    raise ValueError(f"{path}:{lineno}: a second 'blinks' marker")
                seen_blinks_marker = True
            else:
                if len(row) < 2:
                    raise ValueError(f"{path}:{lineno}: blink row needs time,code")
                code = int(float(row[1]))
                if code not in BLINK_CODES:
                    raise ValueError(
                        f"{path}:{lineno}: blink code {code} is not one of "
                        f"{sorted(BLINK_CODES)} ({', '.join(BLINK_CODES.values())})"
                    )
                times.append(float(row[0]))
                codes.append(code)

    if expected_intervals:
        raise ValueError(f"{path}: 'corrupt' promised {expected_intervals} more interval(s)")

    order = np.argsort(times) if times else np.zeros(0, dtype=np.int64)
    return Labels(
        corrupt=corrupt,
        blink_times=np.asarray(times, dtype=np.float64)[order],
        blink_codes=np.asarray(codes, dtype=np.int64)[order],
    )


# ============================================================
# Signal
# ============================================================
def read_signal(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read ``S<ID>_data.csv`` → ``(time_s (T,), eeg (2, T))`` in microvolts.

    Columns beyond the first three are ignored, as BLINK's own reader does.
    """
    raw = np.loadtxt(path, delimiter=";", skiprows=1, usecols=(0, 1, 2), dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] < 2:
        raise ValueError(f"{path}: expected at least 2 rows of time;Fp1;Fp2, got {raw.shape}")
    return raw[:, 0], raw[:, 1:3].T


@dataclass
class Recording:
    """One subject's EEG-IO recording, with its annotations."""
    subject: str
    fs: float
    eeg: np.ndarray            # (2, T) microvolts, rows are CHANNEL_NAMES
    labels: Labels
    resampled_from: Optional[float] = None

    @property
    def duration_s(self) -> float:
        return self.eeg.shape[1] / self.fs

    def channel(self, name: str) -> np.ndarray:
        return self.eeg[CHANNEL_NAMES.index(name)]

    def corrupt_mask(self) -> np.ndarray:
        """(T,) bool — True where a sample falls inside an annotated corrupt interval."""
        t = np.arange(self.eeg.shape[1]) / self.fs
        mask = np.zeros(t.shape, dtype=bool)
        for start, end in self.labels.corrupt:
            mask |= (t >= start) & (t <= end)
        return mask


def resample_to_project_rate(eeg: np.ndarray, fs_in: float = NATIVE_RATE_HZ,
                             fs_out: float = PROJECT_RATE_HZ) -> np.ndarray:
    """
    Polyphase-resample ``(C, T)`` from EEG-IO's 250 Hz to the project's 256 Hz.

    256/250 reduces to 128/125, so this is an exact rational resample with no
    accumulating phase error -- annotation times stay valid in seconds and need
    only be divided by the new rate to become sample indices.
    """
    from fractions import Fraction

    from scipy.signal import resample_poly

    ratio = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(eeg, ratio.numerator, ratio.denominator, axis=-1).astype(np.float64)


# ============================================================
# Loading
# ============================================================
def list_subjects(data_dir: str | Path) -> List[str]:
    """Subject ids with both a data and a labels file, naturally sorted."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"{data_dir} is not a directory. EEG-IO is request-gated at "
            f"http://gnan.ece.gatech.edu/eeg-eyeblinks/ and is not shipped with this repo; "
            f"unpack it so that {data_dir}/S01_data.csv exists."
        )
    found = []
    for path in data_dir.iterdir():
        m = _SUBJECT_RE.match(path.name)
        if m and (data_dir / f"{m.group('subject')}_labels.csv").is_file():
            found.append(m.group("subject"))
    return sorted(found, key=lambda s: (len(s), s))


def load_subject(data_dir: str | Path, subject: str, *, resample: bool = True) -> Recording:
    """Read one subject, resampled to the project's 256 Hz unless told otherwise."""
    data_dir = Path(data_dir)
    time_s, eeg = read_signal(data_dir / f"{subject}_data.csv")

    # The time column is the authority on the rate; a file that is not 250 Hz
    # is a different dataset and silently resampling it would hide that.
    steps = np.diff(time_s)
    fs = 1.0 / float(np.median(steps)) if len(steps) else NATIVE_RATE_HZ
    if not np.isclose(fs, NATIVE_RATE_HZ, rtol=0.02):
        raise ValueError(
            f"{subject}: time column implies {fs:.2f} Hz, expected {NATIVE_RATE_HZ} Hz for EEG-IO"
        )

    labels = read_labels(data_dir / f"{subject}_labels.csv", duration_s=eeg.shape[1] / NATIVE_RATE_HZ)

    if resample:
        return Recording(subject=subject, fs=PROJECT_RATE_HZ,
                         eeg=resample_to_project_rate(eeg, NATIVE_RATE_HZ, PROJECT_RATE_HZ),
                         labels=labels, resampled_from=NATIVE_RATE_HZ)
    return Recording(subject=subject, fs=NATIVE_RATE_HZ, eeg=eeg, labels=labels)


def load_dataset(data_dir: str | Path, *, subjects: Optional[Sequence[str]] = None,
                 resample: bool = True) -> List[Recording]:
    """Every subject in ``data_dir`` (or just ``subjects``), in subject order."""
    wanted = list(subjects) if subjects is not None else list_subjects(data_dir)
    if not wanted:
        raise FileNotFoundError(f"no S<ID>_data.csv / S<ID>_labels.csv pairs found in {data_dir}")
    return [load_subject(data_dir, s, resample=resample) for s in wanted]


# ============================================================
# Epoch cutting
# ============================================================
@dataclass
class Epochs:
    """
    Model-ready windows cut from EEG-IO, with no clean counterpart.

    ``x``        (N, WINDOW) float32, raw microvolts
    ``is_blink`` (N,) bool     — True for a window centred on an annotated blink
    ``n_blinks`` (N,) int      — annotated blinks whose midpoint falls inside
    ``subject``  (N,) str      — which recording it came from
    ``channel``  (N,) str      — Fp1 or Fp2
    ``start``    (N,) int      — first sample index within that recording
    """
    x: np.ndarray
    is_blink: np.ndarray
    n_blinks: np.ndarray
    subject: np.ndarray
    channel: np.ndarray
    start: np.ndarray
    fs: float

    def __len__(self) -> int:
        return len(self.x)

    def blink(self) -> "Epochs":
        return self[self.is_blink]

    def clean(self) -> "Epochs":
        return self[~self.is_blink]

    def __getitem__(self, sel) -> "Epochs":
        return Epochs(x=self.x[sel], is_blink=self.is_blink[sel], n_blinks=self.n_blinks[sel],
                      subject=self.subject[sel], channel=self.channel[sel],
                      start=self.start[sel], fs=self.fs)


def cut_epochs(recordings: Sequence[Recording], *, window: int = WINDOW,
               channels: Sequence[str] = CHANNEL_NAMES, include_soft: bool = True,
               guard_s: float = 1.0, max_clean_per_channel: Optional[int] = None) -> Epochs:
    """
    Cut two populations of windows per channel:

    * **blink** windows, each centred on an annotated blink midpoint, and
    * **blink-free** windows, which contain no annotated blink and keep
      ``guard_s`` seconds of clearance from the nearest one.

    Both exclude any window overlapping an annotated corrupt interval, and any
    window that would run off either end of the recording.  The two together
    are what makes a reference-free comparison meaningful: suppression inside
    blink windows is only a denoising result if blink-free windows survive.

    The model is single-channel, so Fp1 and Fp2 become separate windows rather
    than two channels of one window.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    for name in channels:
        if name not in CHANNEL_NAMES:
            raise ValueError(f"unknown channel {name!r}; EEG-IO has {CHANNEL_NAMES}")

    xs, is_blink, n_blinks, subjects, chans, starts = [], [], [], [], [], []
    fs_seen = {rec.fs for rec in recordings}
    if len(fs_seen) > 1:
        raise ValueError(f"recordings disagree on sample rate: {sorted(fs_seen)}")
    fs = fs_seen.pop() if fs_seen else PROJECT_RATE_HZ
    half = window // 2
    guard = int(round(guard_s * fs))

    for rec in recordings:
        n = rec.eeg.shape[1]
        corrupt = rec.corrupt_mask()
        blink_t = rec.labels.blinks(include_soft=include_soft)
        blink_idx = np.round(blink_t * rec.fs).astype(np.int64)
        blink_idx = blink_idx[(blink_idx >= 0) & (blink_idx < n)]

        def usable(start: int) -> bool:
            return 0 <= start and start + window <= n and not corrupt[start:start + window].any()

        def blinks_inside(start: int) -> int:
            return int(np.count_nonzero((blink_idx >= start) & (blink_idx < start + window)))

        for name in channels:
            signal = rec.channel(name)

            # --- blink windows: one per annotation, centred on the minimum ---
            for centre in blink_idx:
                start = int(centre) - half
                if not usable(start):
                    continue
                xs.append(signal[start:start + window])
                is_blink.append(True)
                n_blinks.append(blinks_inside(start))
                subjects.append(rec.subject)
                chans.append(name)
                starts.append(start)

            # --- blink-free windows: tiled, guarded, non-overlapping ---
            taken = 0
            for start in range(0, n - window + 1, window):
                if max_clean_per_channel is not None and taken >= max_clean_per_channel:
                    break
                if not usable(start):
                    continue
                # Guard band on both sides, so a blink just outside the window
                # cannot have smeared into it.
                if blink_idx.size:
                    near = (blink_idx >= start - guard) & (blink_idx < start + window + guard)
                    if near.any():
                        continue
                xs.append(signal[start:start + window])
                is_blink.append(False)
                n_blinks.append(0)
                subjects.append(rec.subject)
                chans.append(name)
                starts.append(start)
                taken += 1

    if not xs:
        raise ValueError(
            "no usable windows: every candidate was corrupt, too close to a blink, or "
            "ran off the end of its recording"
        )

    return Epochs(
        x=np.stack(xs).astype(np.float32),
        is_blink=np.asarray(is_blink, dtype=bool),
        n_blinks=np.asarray(n_blinks, dtype=np.int64),
        subject=np.asarray(subjects),
        channel=np.asarray(chans),
        start=np.asarray(starts, dtype=np.int64),
        fs=fs,
    )


# ============================================================
# Export
# ============================================================
def save_epochs(epochs: Epochs, path: str | Path) -> Path:
    """
    Write an ``.npz`` of raw-microvolt windows: ``x``, ``is_blink``, ``n_blinks``,
    ``subject``, ``channel``, ``start``, ``fs``.

    Deliberately no ``clean`` array -- EEG-IO has no ground truth, and a
    zero-filled or copied one would be read as a reference by anything that
    found it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, x=epochs.x, is_blink=epochs.is_blink, n_blinks=epochs.n_blinks,
        subject=epochs.subject.astype("U8"), channel=epochs.channel.astype("U8"),
        start=epochs.start, fs=np.asarray(epochs.fs, dtype=np.float64),
    )
    return path


def summarise(recordings: Sequence[Recording], epochs: Optional[Epochs] = None) -> str:
    lines = []
    rate = recordings[0].fs if recordings else PROJECT_RATE_HZ
    resampled = any(rec.resampled_from for rec in recordings)
    total_blinks = sum(len(rec.labels.blink_times) for rec in recordings)
    by_code = {name: 0 for name in BLINK_CODES.values()}
    for rec in recordings:
        for code in rec.labels.blink_codes:
            by_code[BLINK_CODES[int(code)]] += 1
    corrupt_s = sum(min(end, rec.duration_s) - start
                    for rec in recordings for start, end in rec.labels.corrupt)
    total_s = sum(rec.duration_s for rec in recordings)

    lines.append(f"[EEG-IO] {len(recordings)} subject(s), {total_s:.0f} s at {rate:.0f} Hz"
                 + (f" (resampled from {recordings[0].resampled_from:.0f} Hz)" if resampled else ""))
    lines.append(f"[EEG-IO] {total_blinks} annotated blinks: "
                 + ", ".join(f"{n} {name}" for name, n in by_code.items()))
    lines.append(f"[EEG-IO] {corrupt_s:.0f} s marked corrupt ({100 * corrupt_s / max(total_s, 1e-9):.1f}%)")
    if epochs is not None:
        n_b, n_c = int(epochs.is_blink.sum()), int((~epochs.is_blink).sum())
        lines.append(f"[EEG-IO] cut {len(epochs)} windows of {epochs.x.shape[1]} samples: "
                     f"{n_b} blink-centred, {n_c} blink-free")
        lines.append(f"[EEG-IO] no clean reference exists for these windows -- "
                     f"RRMSE_t / RRMSE_s / ACC are not computable on EEG-IO")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="data/EEG-IO",
                   help="directory holding S<ID>_data.csv / S<ID>_labels.csv")
    p.add_argument("--out", default=None, help="write the cut windows to this .npz")
    p.add_argument("--subjects", nargs="+", default=None, help="subset, e.g. S01 S02")
    p.add_argument("--native_rate", action="store_true",
                   help="keep EEG-IO's 250 Hz instead of resampling to the project's 256 Hz")
    p.add_argument("--window", type=int, default=WINDOW)
    p.add_argument("--no_soft", action="store_true", help="ignore code-2 (soft) blinks")
    p.add_argument("--guard_s", type=float, default=1.0,
                   help="clearance a blink-free window keeps from the nearest blink")
    p.add_argument("--max_clean_per_channel", type=int, default=None,
                   help="cap blink-free windows per subject-channel (they outnumber blinks)")
    a = p.parse_args(argv)

    recordings = load_dataset(a.data_dir, subjects=a.subjects, resample=not a.native_rate)
    epochs = cut_epochs(recordings, window=a.window, include_soft=not a.no_soft,
                        guard_s=a.guard_s, max_clean_per_channel=a.max_clean_per_channel)
    print(summarise(recordings, epochs))
    if a.out:
        path = save_epochs(epochs, a.out)
        print(f"[EEG-IO] wrote {path}  ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
