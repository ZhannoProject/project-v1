"""
EEG Denoising Dataset with leak-free train/val/test splits.

Designed for the EEGdenoiseNet benchmark. Guarantees that no clean EEG
epoch or artifact epoch is shared across splits, which is required for
honest evaluation of denoising models.

Two sampling modes live here, ``SAMPLING_MODES`` below.  ``"legacy"`` is the
default and is what every run in ``runs/`` used; ``"balanced"`` fixes the two
quirks that audit found in it.  The default is the older behaviour on purpose:
these choices decide what the validation set *is*, so changing them silently
would make every metric already on disk describe a dataset that no longer
exists.
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


#: How a split draws its SNRs and how much of each pool it uses.
#:
#: ``"legacy"`` is what every run in ``runs/`` was trained and validated
#: under, and it is the default for exactly that reason: changing the
#: validation distribution would silently invalidate every number already on
#: disk.  It has two known quirks, both measured and both documented at their
#: sites: in BOTH mode the two artifacts land on the *same* grid point for
#: val/test (see ``_pick_snr``), and the clean pool is truncated to the size
#: of the artifact pool even for training, where the pairing is random anyway
#: (see ``__init__``).
#:
#: ``"balanced"`` fixes both and makes the deterministic grid divide the split
#: exactly.  It is a different dataset, so a run trained under it is not
#: comparable with one trained under ``"legacy"``; the mode is recorded on the
#: dataset so a caller can say which it used.
SAMPLING_MODES = ("legacy", "balanced")

#: The rate EEGdenoiseNet's epochs arrive at: 512 samples over 2 seconds.
#: Written down rather than inferred, because it is the denominator of every
#: "ms" in this project and a run that trains at another rate has to record
#: which one it used.
NATIVE_FS = 256.0


def resample_epochs(epochs: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """
    Resample ``(n_epochs, n_samples)`` from one rate to another.

    Done once, on the raw pools, before anything is split or mixed -- so the
    SNR calibration still happens at the rate the model will see, and the
    splits are untouched (they are index-level and the index set does not
    change). 256 -> 250 is exactly 125/128, so a 512-sample epoch becomes
    500 samples and no sample lands between two others.
    """
    if fs_in == fs_out:
        return np.ascontiguousarray(epochs, dtype=np.float32)
    from fractions import Fraction

    from scipy.signal import resample_poly

    ratio = (Fraction(fs_out).limit_denominator(10_000)
             / Fraction(fs_in).limit_denominator(10_000))
    out = resample_poly(np.asarray(epochs, dtype=np.float64),
                        ratio.numerator, ratio.denominator, axis=-1)
    return np.ascontiguousarray(out, dtype=np.float32)


#: An epoch whose RMS is below this is treated as degenerate.  Such an epoch
#: cannot be mixed at any SNR -- ``artifact_gain`` would return 0 and the
#: sample would silently come out as all zeros -- so it is refused at load
#: time rather than trained on.  No epoch in EEGdenoiseNet is anywhere near
#: it (the minima are 63.2, 18.8 and 564.1 for EEG, EOG and EMG).
MIN_EPOCH_RMS = 1e-6


def _load_epochs(data_dir: str, name: str, seq_len: Optional[int] = None) -> np.ndarray:
    """
    Load one ``<name>_all_epochs.npy`` and check it is what the mixer assumes.

    ``allow_pickle`` is off: these are plain float arrays, and leaving it on
    lets a crafted .npy execute code on load.

    The three checks are here because each failure is otherwise reported far
    away from its cause: a 1-D or 3-D array fails inside ``__getitem__``, an
    epoch length that differs from the clean signal's fails as a broadcast
    error one sample at a time, and a degenerate epoch fails as nothing at
    all -- it trains, on a silently all-zero sample.
    """
    path = os.path.join(data_dir, f"{name}_all_epochs.npy")
    arr = np.load(path, allow_pickle=False).astype(np.float32)

    if arr.ndim != 2:
        raise ValueError(f"{path}: expected (n_epochs, n_samples), got shape {arr.shape}")
    if seq_len is not None and arr.shape[1] != seq_len:
        raise ValueError(
            f"{path}: epoch length {arr.shape[1]} does not match the clean "
            f"signal's {seq_len}; they are added sample by sample"
        )

    rms = np.sqrt((arr.astype(np.float64) ** 2).mean(axis=1))
    bad = int((rms < MIN_EPOCH_RMS).sum())
    if bad:
        raise ValueError(
            f"{path}: {bad} epoch(s) have an RMS below {MIN_EPOCH_RMS:g} and "
            f"cannot be mixed at any SNR (first at index {int(np.argmin(rms))})"
        )
    return arr


# ============================================================
# Signal mixing utility
# ============================================================
def artifact_gain(clean: np.ndarray, artifact: np.ndarray, snr_db: float) -> float:
    """
    The scalar lambda that puts `artifact` at `snr_db` against `clean`.

        lambda = RMS(clean) / (RMS(artifact) * 10^(SNR_dB/20))

    Split out of scale_artifact because the switch probe measures the gain on
    one *segment* of the epoch and then applies it across the whole epoch, so
    the crossfade at the switch point has two full-length signals to blend.
    """
    rms_clean = float(np.sqrt(np.mean(clean ** 2)))
    rms_art = float(np.sqrt(np.mean(artifact ** 2)))

    if rms_art < 1e-8:
        return 0.0

    return rms_clean / (rms_art * 10 ** (snr_db / 20))


def scale_artifact(clean: np.ndarray, artifact: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Scale an artifact so that mixing it with `clean` achieves the target SNR.

    Returns the scaled artifact (NOT the mixed signal).  This lets callers
    stack multiple scaled artifacts before adding them to the clean signal:

        noisy = clean + scale_artifact(clean, eog, snr1) + scale_artifact(clean, emg, snr2)

    Formula: scaled = lambda * artifact
             where lambda = RMS(clean) / (RMS(artifact) * 10^(SNR_dB/20))
    """
    return artifact_gain(clean, artifact, snr_db) * artifact


def mix_at_snr(clean: np.ndarray, artifact: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Mix a clean EEG epoch with a single artifact at a target SNR (in dB).

    Convenience wrapper around scale_artifact for the single-artifact case.
    Returns a fresh array (the input `clean` is never modified in place).
    """
    return clean + scale_artifact(clean, artifact, snr_db)


def realised_snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    """
    The SNR a mixture actually came out at, measured off the two signals.

    The inverse of `scale_artifact`, and it exists because the two are not
    always the same number.  Ask for one artifact at s dB and you get s dB.
    Ask for *two*, each calibrated to s dB against the same clean signal, and
    each contributes the same noise power, so the mixture holds twice as much
    and lands at s - 10*log10(2), about 3 dB below what was requested.
    """
    rms_clean = float(np.sqrt(np.mean(np.asarray(clean, dtype=np.float64) ** 2)))
    rms_noise = float(np.sqrt(np.mean(np.asarray(noise, dtype=np.float64) ** 2)))
    if rms_noise < 1e-12 or rms_clean < 1e-12:
        return float("inf") if rms_noise < 1e-12 else float("-inf")
    return 20.0 * np.log10(rms_clean / rms_noise)


# ============================================================
# Dataset
# ============================================================
class EEGDenoiseDataset(Dataset):
    """
    EEGdenoiseNet dataset with strict epoch-level train/val/test splits.

    Key guarantees
    --------------
    - Given the same `master_seed`, the three splits receive disjoint
      sets of clean EEG epoch indices AND disjoint sets of artifact
      epoch indices. No leakage is possible across splits.
    - Clean and artifact use *different* permutation seeds, so the
      pairing is not biased by index correlation.
    - Training set uses random SNR sampling (data augmentation).
      Val/test use a deterministic SNR grid, so val_loss is comparable
      across epochs and runs.

    What it does NOT guarantee under the default ``sampling="legacy"``, both
    measured rather than assumed:

    - In BOTH mode the two artifacts of a val/test sample are mixed at the
      *same* SNR, while training draws the two independently.  See
      ``_pick_snr``; ``sampling="balanced"`` offsets the second artifact.
    - ``snr_db`` in BOTH mode is the mean of the two requested SNRs, which
      is about 3 dB above the mixture's realised SNR.  ``snr_mixture_db``
      carries the realised figure.
    """

    # Per-split offset for deriving sub-seeds from `master_seed`.
    # This keeps the three splits' RNG streams independent.
    SPLIT_OFFSET = {"train": 0, "val": 1, "test": 2}

    #: Levels in the deterministic val/test SNR grid.
    SNR_GRID_LEVELS = 10

    def __init__(
        self,
        data_dir: str,
        artifact_type: str = "EMG",
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        snr_range_db: Tuple[float, float] = (-7.0, 2.0),
        fixed_snr_db: Optional[float] = None,
        master_seed: int = 42,
        sampling: str = "legacy",
        target_fs: Optional[float] = None,
    ):
        assert split in self.SPLIT_OFFSET, f"split must be one of {list(self.SPLIT_OFFSET)}"
        assert artifact_type in ("EOG", "EMG", "BOTH"), \
            "artifact_type must be 'EOG', 'EMG', or 'BOTH'"
        assert abs(sum(split_ratios) - 1.0) < 1e-6, "split_ratios must sum to 1.0"
        assert sampling in SAMPLING_MODES, f"sampling must be one of {list(SAMPLING_MODES)}"

        self.split = split
        self.artifact_type = artifact_type
        self.snr_range_db = snr_range_db
        self.fixed_snr_db = fixed_snr_db
        self.sampling = sampling
        self.master_seed = int(master_seed)
        # None means "leave it alone", which is what every run on disk did
        # and what keeps them reproducible; a number resamples the pools.
        self.fs = float(target_fs) if target_fs else NATIVE_FS
        self.target_fs = target_fs

        # ---- 1. Load raw numpy arrays ----
        clean_all = _load_epochs(data_dir, "EEG")
        seq_len = clean_all.shape[1]
        rs = (lambda a: resample_epochs(a, NATIVE_FS, self.fs)) if target_fs else (lambda a: a)
        clean_all = rs(clean_all)

        # Build disjoint clean indices for this split
        clean_idx = self._split_indices(len(clean_all), split_ratios, split, master_seed)

        pair_rng = np.random.default_rng(master_seed + self.SPLIT_OFFSET[split])

        if artifact_type == "BOTH":
            # ---- BOTH mode: load EOG and EMG, pair each independently ----
            eog_all = rs(_load_epochs(data_dir, "EOG", seq_len))
            emg_all = rs(_load_epochs(data_dir, "EMG", seq_len))

            eog_idx = self._split_indices(len(eog_all), split_ratios, split, master_seed + 99)
            emg_idx = self._split_indices(len(emg_all), split_ratios, split, master_seed + 199)

            n = self._pair_count(len(clean_idx), len(eog_idx), len(emg_idx))
            n_clean = self._clean_count(len(clean_idx), n)
            self.clean = clean_all[pair_rng.choice(clean_idx, n_clean, replace=False)]
            self.artifact_eog = eog_all[pair_rng.choice(eog_idx, n, replace=False)]
            self.artifact_emg = emg_all[pair_rng.choice(emg_idx, n, replace=False)]
            self.artifact = None   # not used in BOTH mode

            print(f"[Dataset] {split:>5s} | BOTH (EOG+EMG) | pairs={n_clean}"
                  f"{'' if n_clean == n else f' (artifacts resampled from {n})'}"
                  f"{'' if sampling == 'legacy' else ' | sampling=balanced'}"
                  f"{'' if not target_fs else f' | {self.fs:g} Hz, {clean_all.shape[1]} samples'}")
        else:
            # ---- Single artifact mode (EOG or EMG) ----
            art_all = rs(_load_epochs(data_dir, artifact_type, seq_len))

            art_idx = self._split_indices(len(art_all), split_ratios, split, master_seed + 99)

            n = self._pair_count(len(clean_idx), len(art_idx))
            n_clean = self._clean_count(len(clean_idx), n)
            self.clean = clean_all[pair_rng.choice(clean_idx, n_clean, replace=False)]
            self.artifact = art_all[pair_rng.choice(art_idx, n, replace=False)]
            self.artifact_eog = None  # not used in single mode
            self.artifact_emg = None

            print(f"[Dataset] {split:>5s} | {artifact_type} | pairs={n_clean}"
                  f"{'' if n_clean == n else f' (artifacts resampled from {n})'}"
                  f"{'' if sampling == 'legacy' else ' | sampling=balanced'}"
                  f"{'' if not target_fs else f' | {self.fs:g} Hz, {clean_all.shape[1]} samples'}")

        # ---- Per-sample RNG (only used for random SNR in training) ----
        # Kept as an attribute, which is why ``worker_init_fn`` below exists:
        # a DataLoader worker is a fork, so every worker would otherwise hold
        # a copy of this generator at the same state and draw the same
        # artifacts and SNRs as all the others.
        self._sample_seed = master_seed + self.SPLIT_OFFSET[split] + 7777
        self.sample_rng = np.random.default_rng(self._sample_seed)

        # ---- Deterministic SNR grid for val/test ----
        self.snr_grid = np.linspace(snr_range_db[0], snr_range_db[1], 10)

    # -------- helpers --------
    @staticmethod
    def _split_indices(n_total, ratios, split, seed):
        """
        Permute [0, n_total) once with a fixed seed, then slice into
        train/val/test segments. Returns the index segment for `split`.
        """
        perm = np.random.default_rng(seed).permutation(n_total)
        n_tr = int(n_total * ratios[0])
        n_va = int(n_total * ratios[1])
        return {
            "train": perm[:n_tr],
            "val":   perm[n_tr:n_tr + n_va],
            "test":  perm[n_tr + n_va:],
        }[split]

    def _pair_count(self, *pool_sizes: int) -> int:
        """
        How many 1:1 (clean, artifact) pairs this split holds.

        ``"balanced"`` additionally trims val/test to a whole number of SNR
        grid cycles.  Without that, the deterministic grid does not divide the
        split -- EMG val is 677 epochs over 10 levels -- so the lowest levels
        appear once more than the highest and the split-averaged metric is
        tilted very slightly towards low SNR.
        """
        n = min(pool_sizes)
        if self.sampling == "balanced" and self.split != "train":
            n -= n % self.SNR_GRID_LEVELS
        return n

    def _clean_count(self, clean_pool: int, n_pairs: int) -> int:
        """
        How many clean epochs this split actually uses.

        Legacy uses ``n_pairs`` everywhere, which throws clean epochs away
        whenever the artifact pool is the smaller one: EOG and BOTH training
        drop 779 of 3159 clean epochs for no reason the training loop needs,
        since it already draws its artifact for each sample at random and with
        replacement.  ``"balanced"`` keeps the whole clean pool for training
        and leaves val/test at 1:1, where the pairing has to be fixed.
        """
        if self.sampling == "balanced" and self.split == "train":
            return clean_pool
        return n_pairs

    def reseed_for_worker(self, worker_id: int) -> None:
        """
        Give this copy of the dataset its own augmentation stream.

        A DataLoader worker is a fork, so without this every worker inherits
        ``sample_rng`` at the same state and draws the same artifact indices
        and the same SNRs as every other worker -- the augmentation would
        repeat once per worker instead of being independent.  Wired up by
        ``worker_init_fn`` below; harmless at ``num_workers=0``, which never
        calls it and therefore keeps the single-process stream exactly as it
        has always been.
        """
        self.sample_rng = np.random.default_rng(
            [self._sample_seed, 0x5EED, int(worker_id)]
        )

    def _pick_snr(self, idx: int, second: bool = False) -> float:
        """
        SNR selection priority:
          1. If `fixed_snr_db` was given, always use it (per-SNR evaluation).
          2. Training split -> uniform random from snr_range_db.
          3. Val/test split -> deterministic SNR from the precomputed grid,
             so every sample gets a reproducible SNR across runs.

        ``second`` marks the second of the two artifacts in BOTH mode.  Under
        ``"legacy"`` it changes nothing, and that is the quirk: the grid value
        depends on ``idx`` alone, so both artifacts of a val/test sample are
        mixed at *exactly* the same SNR, while training draws the two
        independently (they differ by 3.0 dB on average and by up to 8.4).
        Val and test therefore cover only the slice of the training
        distribution where neither artifact dominates.  Under ``"balanced"``
        the second artifact is offset half a grid away, so the pair spans the
        range and each artifact's marginal stays uniform over the grid.
        """
        if self.fixed_snr_db is not None:
            return float(self.fixed_snr_db)
        if self.split == "train":
            return float(self.sample_rng.uniform(*self.snr_range_db))
        n_levels = len(self.snr_grid)
        if second and self.sampling == "balanced":
            idx += n_levels // 2
        return float(self.snr_grid[idx % n_levels])

    # -------- Dataset interface --------
    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        clean = self.clean[idx]

        if self.artifact_type == "BOTH":
            # ---- BOTH mode: mix two artifact types simultaneously ----
            # Pick artifacts (random for train, fixed for val/test)
            if self.split == "train":
                eog = self.artifact_eog[self.sample_rng.integers(0, len(self.artifact_eog))]
                emg = self.artifact_emg[self.sample_rng.integers(0, len(self.artifact_emg))]
            else:
                eog = self.artifact_eog[idx]
                emg = self.artifact_emg[idx]

            # One SNR per artifact.  Independent on train; under "legacy"
            # these are the *same* number on val/test -- see _pick_snr.
            snr_eog = self._pick_snr(idx)
            snr_emg = self._pick_snr(idx, second=True)

            # Both contributions are calibrated against the original clean signal
            noisy = clean + scale_artifact(clean, eog, snr_eog) \
                         + scale_artifact(clean, emg, snr_emg)

            # The mean of the two per-artifact SNRs, which is NOT the SNR of
            # the mixture: two artifacts each calibrated to s dB against the
            # same clean signal put in twice the noise power, so the mixture
            # sits about 3 dB below s.  Kept as the reported value because
            # every per-SNR table and test set on disk is keyed on it;
            # ``snr_mixture_db`` below is the realised figure.
            snr_db = (snr_eog + snr_emg) / 2.0
        else:
            # ---- Single artifact mode ----
            if self.split == "train":
                art_idx = self.sample_rng.integers(0, len(self.artifact))
                artifact = self.artifact[art_idx]
            else:
                artifact = self.artifact[idx]

            snr_db = self._pick_snr(idx)
            noisy = mix_at_snr(clean, artifact, snr_db)

        # Normalize both signals by sigma of the noisy input.
        # This keeps the model's input scale stable across samples.
        sigma = max(noisy.std(), 1e-8)

        return {
            "noisy":  torch.tensor(noisy / sigma).unsqueeze(0),   # (1, T)
            "clean":  torch.tensor(clean / sigma).unsqueeze(0),   # (1, T)
            "sigma":  torch.tensor(sigma, dtype=torch.float32),
            # What the mix was *asked* for, per artifact.  In BOTH mode the
            # mean of the two, which is not the mixture's SNR (see above).
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
            # What the mix actually realised, measured off the signals
            # themselves.  Equal to snr_db for a single artifact; about 3 dB
            # below it for BOTH.  This is the one to label an axis with.
            "snr_mixture_db": torch.tensor(
                realised_snr_db(clean, noisy - clean), dtype=torch.float32),
        }


# ============================================================
# Non-stationary artifact-switch probe
# ============================================================
#: The series of probe seeds the switch experiment is reported over.
#: Each one is a fresh draw of switch points, SNRs, directions and pairings
#: on top of the *same* leak-free split, so the spread across them is the
#: uncertainty of the probe rather than of the model.
SWITCH_PROBE_SEEDS = (0, 1, 2, 3, 4)


class ArtifactSwitchDataset(Dataset):
    """
    Non-stationary probe: one artifact type before a switch point, the other
    after it.

    Each epoch takes a clean EEG epoch from the requested split, adds EOG to
    one side of a switch point and EMG to the other, and records where the
    switch was.  Nothing about the probe is constant enough to be memorised:

    - the switch index is drawn per sample inside ``switch_range`` (never a
      fixed position),
    - the two artifact types get independently drawn SNRs,
    - exactly half the epochs run EOG -> EMG and half run EMG -> EOG.

    This is an *evaluation* probe, not a training set.  It exists to measure
    how fast a model re-adapts when the artifact statistics change under it,
    so feed it models trained with ``artifact_type="BOTH"``.  It draws from
    the same leak-free pools as ``EEGDenoiseDataset(artifact_type="BOTH")``
    at the same ``master_seed``, so a model's test split is its test split
    here too.

    ``probe_seed`` is what makes a *series* of these.  ``master_seed`` fixes
    which epochs exist in the split; ``probe_seed`` fixes everything drawn on
    top of them.  Averaging over ``SWITCH_PROBE_SEEDS`` turns one realisation
    of the switch positions into an error bar.

    Every draw is made once in ``__init__`` and stored, so ``__getitem__`` is
    a pure function of the index: shuffling, re-batching or scoring a subset
    cannot change what a given sample contains.  ``n_epochs`` cuts the probe
    after those draws are made, so a short probe is the first n epochs of the
    full one and the two can be compared.

    Parameters
    ----------
    switch_range : (float, float)
        Switch position as a fraction of the epoch.  The default (0.25, 0.75)
        leaves at least a quarter of an epoch — 128 samples, 0.5 s at 256 Hz —
        on each side, which is the widest window the response curve can be
        measured over.
    ramp_samples : int
        Length of a raised-cosine crossfade centred on the switch.  The
        default 0 is a hard switch, which is what the experiment is about; a
        short ramp is the control for "is the effect just the discontinuity?".
    fixed_snr_db : float or None
        Overrides both SNR draws with one value, for a curve measured at a
        single noise level.  ``None`` (the default) samples the two artifact
        types independently, which is the intended design.
    order : {"both", "eog_first", "emg_first"}
        ``"both"`` balances the two directions exactly; the other two pin it,
        for the per-direction breakdown.
    control : bool
        ``True`` builds the **matched control**: the splice still happens at
        the same index, but what gets spliced in is a *different epoch of the
        same artifact type*, at the same SNR.  Every other draw is identical —
        clean epochs, first artifact, switch indices, SNRs, directions — and
        the control even borrows the switching epoch's sigma, so the two
        probes are bit-identical on ``[0, k)``.

        Two things make the control worth running.  It holds the *type* change
        as the only difference, so the response curve stops answering "does a
        splice hurt?" and starts answering "does a change of artifact
        *statistics* hurt?" — the thing the experiment is about.  And it
        cancels the within-epoch warm-up trend both probes share: a causal
        model has no left context at the start of an epoch, so its error falls
        through the first few hundred ms regardless of any artifact, and
        because the switch index varies that trend would otherwise tilt the
        whole response curve.

        For a causal model the difference between the two probes is exactly
        zero before the switch — the input is the same signal there.  For a
        non-causal one it is not, and how far it departs from zero is how far
        the switch leaks backwards.

        That exactness needs ``ramp_samples == 0``.  A crossfade starts before
        the switch index by design, so the two probes begin to differ there
        too and the pre-switch difference stops being a causality test (it
        stays a valid control for everything else).
    """

    SPLIT_OFFSET = EEGDenoiseDataset.SPLIT_OFFSET

    def __init__(
        self,
        data_dir: str,
        split: str = "test",
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        snr_range_db: Tuple[float, float] = (-7.0, 2.0),
        fixed_snr_db: Optional[float] = None,
        master_seed: int = 42,
        probe_seed: int = 0,
        switch_range: Tuple[float, float] = (0.25, 0.75),
        ramp_samples: int = 0,
        order: str = "both",
        n_epochs: Optional[int] = None,
        control: bool = False,
    ):
        assert split in self.SPLIT_OFFSET, f"split must be one of {list(self.SPLIT_OFFSET)}"
        assert order in ("both", "eog_first", "emg_first"), \
            "order must be 'both', 'eog_first' or 'emg_first'"
        assert 0.0 < switch_range[0] < switch_range[1] < 1.0, \
            "switch_range must be an increasing pair inside (0, 1)"
        assert ramp_samples >= 0, "ramp_samples must be >= 0"
        assert abs(sum(split_ratios) - 1.0) < 1e-6, "split_ratios must sum to 1.0"

        self.split = split
        self.probe_seed = int(probe_seed)
        self.ramp_samples = int(ramp_samples)
        self.order = order
        self.control = bool(control)

        # ---- 1. Same leak-free pools as EEGDenoiseDataset(BOTH) ----
        clean_all = _load_epochs(data_dir, "EEG")
        eog_all = _load_epochs(data_dir, "EOG", clean_all.shape[1])
        emg_all = _load_epochs(data_dir, "EMG", clean_all.shape[1])

        split_fn = EEGDenoiseDataset._split_indices
        clean_idx = split_fn(len(clean_all), split_ratios, split, master_seed)
        eog_idx = split_fn(len(eog_all), split_ratios, split, master_seed + 99)
        emg_idx = split_fn(len(emg_all), split_ratios, split, master_seed + 199)

        # ---- 2. One RNG for everything this probe draws ----
        # Seeding on the pair keeps the probe series independent of the split
        # seed: probe_seed 1 is not master_seed 43's anything.
        rng = np.random.default_rng([master_seed, 20260926, self.probe_seed])

        # Every draw below is made at the *full* split size and only then cut
        # down to ``n_epochs``, so a short probe is the first n epochs of the
        # long one rather than an unrelated draw.  It used to shrink ``n``
        # first, which changed how many numbers each ``choice`` consumed and
        # so changed everything downstream of it: a 100-epoch probe and the
        # full 510-epoch probe at the same seed shared no clean epoch and no
        # switch index.  Direction balance is restored afterwards, at the cut
        # size, because it is the last thing drawn (see step 5).
        n_full = min(len(clean_idx), len(eog_idx), len(emg_idx))
        n = n_full if n_epochs is None else min(n_full, int(n_epochs))

        eog_pick = rng.choice(eog_idx, n_full, replace=False)
        emg_pick = rng.choice(emg_idx, n_full, replace=False)
        self.clean = clean_all[rng.choice(clean_idx, n_full, replace=False)]
        self.eog = eog_all[eog_pick]
        self.emg = emg_all[emg_pick]

        # A second, independent pairing of each artifact pool.  The matched
        # control splices one of these in at the switch point instead of the
        # other artifact type, so the splice is held constant and only the
        # type changes.  Where the two pairings collide, take the neighbour:
        # the picks are without replacement, so it is guaranteed different,
        # and a control that spliced the very same epoch back in would have no
        # discontinuity to control for.
        roll = (np.arange(n_full) + 1) % n_full
        eog_alt = rng.choice(eog_idx, n_full, replace=False)
        emg_alt = rng.choice(emg_idx, n_full, replace=False)
        self.eog_alt = eog_all[np.where(eog_alt == eog_pick, eog_pick[roll], eog_alt)]
        self.emg_alt = emg_all[np.where(emg_alt == emg_pick, emg_pick[roll], emg_alt)]

        T = self.clean.shape[1]
        self.seq_len = T

        # ---- 3. Switch points, kept clear of the epoch edges ----
        lo = int(round(switch_range[0] * T))
        hi = int(round(switch_range[1] * T))
        margin = self.ramp_samples // 2 + 1
        lo, hi = max(lo, margin), min(hi, T - margin)
        if lo > hi:
            raise ValueError(
                f"switch_range {switch_range} leaves no room for a "
                f"{self.ramp_samples}-sample ramp in a {T}-sample epoch"
            )
        self.switch_idx = rng.integers(lo, hi + 1, size=n_full)
        self.switch_bounds = (lo, hi)

        # ---- 4. Independent SNR per artifact type ----
        if fixed_snr_db is not None:
            self.snr_eog = np.full(n_full, float(fixed_snr_db))
            self.snr_emg = np.full(n_full, float(fixed_snr_db))
        else:
            self.snr_eog = rng.uniform(snr_range_db[0], snr_range_db[1], size=n_full)
            self.snr_emg = rng.uniform(snr_range_db[0], snr_range_db[1], size=n_full)

        # ---- 4b. Cut to n_epochs, now that every draw above has been made ----
        if n < n_full:
            self.clean, self.eog, self.emg = self.clean[:n], self.eog[:n], self.emg[:n]
            self.eog_alt, self.emg_alt = self.eog_alt[:n], self.emg_alt[:n]
            self.switch_idx = self.switch_idx[:n]
            self.snr_eog, self.snr_emg = self.snr_eog[:n], self.snr_emg[:n]

        # ---- 5. Direction, balanced exactly rather than in expectation ----
        # Drawn at the cut size, and last, so that a truncated probe is still
        # exactly half of each direction -- slicing a balanced array of n_full
        # would only be balanced on average.  Nothing is drawn after this, so
        # taking a different count of random numbers here changes nothing else.
        if order == "both":
            flags = np.zeros(n, dtype=bool)
            flags[: n // 2] = True
            rng.shuffle(flags)
        else:
            flags = np.full(n, order == "eog_first", dtype=bool)
        self.eog_first = flags

        kind = "CONTROL (same type)" if self.control else "SWITCH (EOG<->EMG)"
        print(f"[Probe] {split:>5s} | {kind} | pairs={n} | "
              f"seed={master_seed}+{self.probe_seed} | "
              f"switch in [{lo}, {hi}] | ramp={self.ramp_samples}")

    # -------- helpers --------
    def _switch_weight(self, k: int) -> np.ndarray:
        """
        Weight of the *second* artifact: 0 before the switch, 1 after it.

        A hard step when ``ramp_samples == 0``; otherwise a raised cosine of
        that width centred on ``k``.
        """
        T = self.seq_len
        w = np.zeros(T, dtype=np.float32)
        r = self.ramp_samples
        if r <= 0:
            w[k:] = 1.0
            return w

        a = max(k - r // 2, 0)
        b = min(a + r, T)
        if b > a:
            ramp = 0.5 - 0.5 * np.cos(
                np.linspace(0.0, np.pi, b - a, dtype=np.float32)
            )
            w[a:b] = ramp
        w[b:] = 1.0
        return w

    # -------- Dataset interface --------
    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        clean = self.clean[idx]
        k = int(self.switch_idx[idx])
        eog_first = bool(self.eog_first[idx])

        if eog_first:
            first, second, same_type = self.eog[idx], self.emg[idx], self.eog_alt[idx]
            snr_first, snr_second = float(self.snr_eog[idx]), float(self.snr_emg[idx])
        else:
            first, second, same_type = self.emg[idx], self.eog[idx], self.emg_alt[idx]
            snr_first, snr_second = float(self.snr_emg[idx]), float(self.snr_eog[idx])

        # Each artifact is calibrated against the clean signal of the segment
        # it actually occupies.  Calibrating against the whole epoch instead
        # would make the realised per-segment SNR depend on where the switch
        # happened to land, which is exactly the variable being controlled.
        gain_first = artifact_gain(clean[:k], first[:k], snr_first)
        gain_second = artifact_gain(clean[k:], second[k:], snr_second)

        w = self._switch_weight(k)
        artifact_switch = (1.0 - w) * (gain_first * first) + w * (gain_second * second)

        if self.control:
            # Matched control: splice a different epoch of the *same* type in,
            # at the same SNR over the same segment, so the only thing the
            # probe removes is the change of artifact type.
            gain_same = artifact_gain(clean[k:], same_type[k:], snr_second)
            artifact = (1.0 - w) * (gain_first * first) + w * (gain_same * same_type)
        else:
            artifact = artifact_switch

        noisy = (clean + artifact).astype(np.float32)

        # Same per-epoch normalisation as EEGDenoiseDataset, so a model sees
        # the input scale it was trained on -- but the control borrows the
        # switching epoch's sigma rather than deriving its own.  That is what
        # makes it exact: the two probes are then bit-identical on [0, k), so a
        # causal model's residual before the switch is identical too and their
        # difference there is exactly zero.  A non-causal model's is not, and
        # how far it departs from zero is how far the switch leaks backwards.
        sigma_src = (clean + artifact_switch) if self.control else noisy
        sigma = max(float(np.asarray(sigma_src).std()), 1e-8)

        return {
            "noisy":      torch.tensor(noisy / sigma).unsqueeze(0),   # (1, T)
            "clean":      torch.tensor(clean / sigma).unsqueeze(0),   # (1, T)
            "sigma":      torch.tensor(sigma, dtype=torch.float32),
            "snr_db":     torch.tensor((snr_first + snr_second) / 2.0,
                                       dtype=torch.float32),
            "snr_eog":    torch.tensor(float(self.snr_eog[idx]), dtype=torch.float32),
            "snr_emg":    torch.tensor(float(self.snr_emg[idx]), dtype=torch.float32),
            "switch_idx": torch.tensor(k, dtype=torch.long),
            "eog_first":  torch.tensor(eog_first),
        }


def build_switch_probes(
    data_dir: str,
    probe_seeds=SWITCH_PROBE_SEEDS,
    **kwargs,
):
    """
    The series of switch probes, one per seed.

    Yields ``(probe_seed, ArtifactSwitchDataset)``.  Every keyword goes
    straight to ``ArtifactSwitchDataset``, so one call covers
    "five seeds, hard switch" as well as "five seeds, 16-sample ramp,
    EOG first, at -3 dB".
    """
    for seed in probe_seeds:
        yield int(seed), ArtifactSwitchDataset(data_dir, probe_seed=int(seed), **kwargs)


# ============================================================
# DataLoader builder
# ============================================================
def worker_init_fn(worker_id: int) -> None:
    """
    Give each DataLoader worker its own augmentation stream.

    A worker is a fork of the parent, so it inherits ``sample_rng`` at
    whatever state the parent held: with four workers and no hook, all four
    draw the same artifact indices and the same SNRs, and the training set's
    augmentation silently repeats four times over instead of being
    independent.  Pass this as ``worker_init_fn`` to any loader built over
    ``EEGDenoiseDataset`` with ``num_workers > 0``.
    """
    info = torch.utils.data.get_worker_info()
    ds = getattr(info, "dataset", None)
    if hasattr(ds, "reseed_for_worker"):
        ds.reseed_for_worker(worker_id)


def build_dataloaders(
    data_dir: str,
    artifact_type: str = "EMG",
    batch_size: int = 32,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    snr_range_db: Tuple[float, float] = (-7.0, 2.0),
    master_seed: int = 42,
    sampling: str = "legacy",
    num_workers: int = 0,
    target_fs: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, and test DataLoaders that share the same master seed
    (and therefore enjoy guaranteed zero leakage at the epoch level).

    The training shuffle draws from its own ``torch.Generator`` seeded with
    ``master_seed``, not from torch's global RNG: batch order is then a
    function of the seed alone and is identical across model variants,
    regardless of how many random numbers their construction consumed.

    ``num_workers`` defaults to 0, which is what every run on disk used and
    the only setting that is bit-reproducible: with workers the augmentation
    stream is split per worker (see ``worker_init_fn``), so the same seed
    gives the same *distribution* but not the same samples in the same order.
    ``sampling`` is passed through to the dataset; see ``SAMPLING_MODES``.
    ``target_fs`` resamples the epochs before anything else happens -- 250
    gives 500-sample epochs instead of 512 -- and ``None``, the default,
    leaves them at the 256 Hz they arrive at, which is what every run on
    disk used.
    """
    # Enable pinned memory for faster host-to-device transfer when on GPU/MPS
    pin = torch.cuda.is_available() or torch.backends.mps.is_available()

    def make(split: str) -> DataLoader:
        ds = EEGDenoiseDataset(
            data_dir=data_dir,
            artifact_type=artifact_type,
            split=split,
            split_ratios=split_ratios,
            snr_range_db=snr_range_db,
            master_seed=master_seed,
            sampling=sampling,
            target_fs=target_fs,
        )
        train = split == "train"
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=train,               # only training set is shuffled
            generator=torch.Generator().manual_seed(master_seed) if train else None,
            pin_memory=pin,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn if num_workers else None,
        )

    train_loader = make("train")
    val_loader = make("val")
    test_loader = make("test")

    print(
        f"[Loaders] train={len(train_loader.dataset)} | "
        f"val={len(val_loader.dataset)} | test={len(test_loader.dataset)} | "
        f"batch={batch_size}"
    )
    return train_loader, val_loader, test_loader


# ============================================================
# Entry point: quick smoke test
# ============================================================
if __name__ == "__main__":
    import argparse

    # Relative to the project root, which is this file's parent's parent, so
    # the smoke test runs from anywhere.  It used to hard-code an absolute
    # path to a directory that no longer exists, which made `python
    # data/Dataloader.py` fail before it reached any of the code below.
    default_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "EEGdenoisenet", "data",
    )
    ap = argparse.ArgumentParser(description="Dataloader smoke test")
    ap.add_argument("--data_dir", default=default_dir)
    ap.add_argument("--artifact", default="EMG", choices=["EOG", "EMG", "BOTH"])
    ap.add_argument("--sampling", default="legacy", choices=list(SAMPLING_MODES))
    args = ap.parse_args()

    # Build the three loaders
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        artifact_type=args.artifact,
        batch_size=32,
        sampling=args.sampling,
    )

    # Inspect a single training batch to confirm shapes and ranges
    batch = next(iter(train_loader))
    print("\n[Smoke test] First training batch:")
    for k, v in batch.items():
        print(f"  {k:15s} shape={tuple(v.shape)}  min={v.min():.2f}  max={v.max():.2f}")