"""
EEG Denoising Dataset with leak-free train/val/test splits.

Designed for the EEGdenoiseNet benchmark. Guarantees that no clean EEG
epoch or artifact epoch is shared across splits, which is required for
honest evaluation of denoising models.
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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
    """

    # Per-split offset for deriving sub-seeds from `master_seed`.
    # This keeps the three splits' RNG streams independent.
    SPLIT_OFFSET = {"train": 0, "val": 1, "test": 2}

    def __init__(
        self,
        data_dir: str,
        artifact_type: str = "EMG",
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        snr_range_db: Tuple[float, float] = (-7.0, 2.0),
        fixed_snr_db: Optional[float] = None,
        master_seed: int = 42,
    ):
        assert split in self.SPLIT_OFFSET, f"split must be one of {list(self.SPLIT_OFFSET)}"
        assert artifact_type in ("EOG", "EMG", "BOTH"), \
            "artifact_type must be 'EOG', 'EMG', or 'BOTH'"
        assert abs(sum(split_ratios) - 1.0) < 1e-6, "split_ratios must sum to 1.0"

        self.split = split
        self.artifact_type = artifact_type
        self.snr_range_db = snr_range_db
        self.fixed_snr_db = fixed_snr_db

        # ---- 1. Load raw numpy arrays ----
        clean_all = np.load(
            os.path.join(data_dir, "EEG_all_epochs.npy"),
            allow_pickle=True,
        ).astype(np.float32)

        # Build disjoint clean indices for this split
        clean_idx = self._split_indices(len(clean_all), split_ratios, split, master_seed)

        pair_rng = np.random.default_rng(master_seed + self.SPLIT_OFFSET[split])

        if artifact_type == "BOTH":
            # ---- BOTH mode: load EOG and EMG, pair each independently ----
            eog_all = np.load(
                os.path.join(data_dir, "EOG_all_epochs.npy"),
                allow_pickle=True,
            ).astype(np.float32)
            emg_all = np.load(
                os.path.join(data_dir, "EMG_all_epochs.npy"),
                allow_pickle=True,
            ).astype(np.float32)

            eog_idx = self._split_indices(len(eog_all), split_ratios, split, master_seed + 99)
            emg_idx = self._split_indices(len(emg_all), split_ratios, split, master_seed + 199)

            n = min(len(clean_idx), len(eog_idx), len(emg_idx))
            self.clean = clean_all[pair_rng.choice(clean_idx, n, replace=False)]
            self.artifact_eog = eog_all[pair_rng.choice(eog_idx, n, replace=False)]
            self.artifact_emg = emg_all[pair_rng.choice(emg_idx, n, replace=False)]
            self.artifact = None   # not used in BOTH mode

            print(f"[Dataset] {split:>5s} | BOTH (EOG+EMG) | pairs={n}")
        else:
            # ---- Single artifact mode (EOG or EMG) ----
            art_all = np.load(
                os.path.join(data_dir, f"{artifact_type}_all_epochs.npy"),
                allow_pickle=True,
            ).astype(np.float32)

            art_idx = self._split_indices(len(art_all), split_ratios, split, master_seed + 99)

            n = min(len(clean_idx), len(art_idx))
            self.clean = clean_all[pair_rng.choice(clean_idx, n, replace=False)]
            self.artifact = art_all[pair_rng.choice(art_idx, n, replace=False)]
            self.artifact_eog = None  # not used in single mode
            self.artifact_emg = None

            print(f"[Dataset] {split:>5s} | {artifact_type} | pairs={n}")

        # ---- Per-sample RNG (only used for random SNR in training) ----
        self.sample_rng = np.random.default_rng(
            master_seed + self.SPLIT_OFFSET[split] + 7777
        )

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

    def _pick_snr(self, idx: int) -> float:
        """
        SNR selection priority:
          1. If `fixed_snr_db` was given, always use it (per-SNR evaluation).
          2. Training split -> uniform random from snr_range_db.
          3. Val/test split -> deterministic SNR from the precomputed grid,
             so every sample gets a reproducible SNR across runs.
        """
        if self.fixed_snr_db is not None:
            return float(self.fixed_snr_db)
        if self.split == "train":
            return float(self.sample_rng.uniform(*self.snr_range_db))
        return float(self.snr_grid[idx % len(self.snr_grid)])

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

            # Each artifact gets its own independent SNR
            snr_eog = self._pick_snr(idx)
            snr_emg = self._pick_snr(idx)

            # Both contributions are calibrated against the original clean signal
            noisy = clean + scale_artifact(clean, eog, snr_eog) \
                         + scale_artifact(clean, emg, snr_emg)

            # Report the average SNR for logging purposes
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
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
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
    cannot change what a given sample contains.

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
        clean_all = np.load(os.path.join(data_dir, "EEG_all_epochs.npy"),
                            allow_pickle=True).astype(np.float32)
        eog_all = np.load(os.path.join(data_dir, "EOG_all_epochs.npy"),
                          allow_pickle=True).astype(np.float32)
        emg_all = np.load(os.path.join(data_dir, "EMG_all_epochs.npy"),
                          allow_pickle=True).astype(np.float32)

        split_fn = EEGDenoiseDataset._split_indices
        clean_idx = split_fn(len(clean_all), split_ratios, split, master_seed)
        eog_idx = split_fn(len(eog_all), split_ratios, split, master_seed + 99)
        emg_idx = split_fn(len(emg_all), split_ratios, split, master_seed + 199)

        # ---- 2. One RNG for everything this probe draws ----
        # Seeding on the pair keeps the probe series independent of the split
        # seed: probe_seed 1 is not master_seed 43's anything.
        rng = np.random.default_rng([master_seed, 20260926, self.probe_seed])

        n = min(len(clean_idx), len(eog_idx), len(emg_idx))
        if n_epochs is not None:
            n = min(n, int(n_epochs))

        eog_pick = rng.choice(eog_idx, n, replace=False)
        emg_pick = rng.choice(emg_idx, n, replace=False)
        self.clean = clean_all[rng.choice(clean_idx, n, replace=False)]
        self.eog = eog_all[eog_pick]
        self.emg = emg_all[emg_pick]

        # A second, independent pairing of each artifact pool.  The matched
        # control splices one of these in at the switch point instead of the
        # other artifact type, so the splice is held constant and only the
        # type changes.  Where the two pairings collide, take the neighbour:
        # the picks are without replacement, so it is guaranteed different,
        # and a control that spliced the very same epoch back in would have no
        # discontinuity to control for.
        roll = (np.arange(n) + 1) % n
        eog_alt = rng.choice(eog_idx, n, replace=False)
        emg_alt = rng.choice(emg_idx, n, replace=False)
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
        self.switch_idx = rng.integers(lo, hi + 1, size=n)
        self.switch_bounds = (lo, hi)

        # ---- 4. Independent SNR per artifact type ----
        if fixed_snr_db is not None:
            self.snr_eog = np.full(n, float(fixed_snr_db))
            self.snr_emg = np.full(n, float(fixed_snr_db))
        else:
            self.snr_eog = rng.uniform(snr_range_db[0], snr_range_db[1], size=n)
            self.snr_emg = rng.uniform(snr_range_db[0], snr_range_db[1], size=n)

        # ---- 5. Direction, balanced exactly rather than in expectation ----
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
def build_dataloaders(
    data_dir: str,
    artifact_type: str = "EMG",
    batch_size: int = 32,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    snr_range_db: Tuple[float, float] = (-7.0, 2.0),
    master_seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, and test DataLoaders that share the same master seed
    (and therefore enjoy guaranteed zero leakage at the epoch level).

    The training shuffle draws from its own ``torch.Generator`` seeded with
    ``master_seed``, not from torch's global RNG: batch order is then a
    function of the seed alone and is identical across model variants,
    regardless of how many random numbers their construction consumed.
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
        )
        train = split == "train"
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=train,               # only training set is shuffled
            generator=torch.Generator().manual_seed(master_seed) if train else None,
            pin_memory=pin,
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
    DATA_DIR = "/Users/zhanlungo/Desktop/Mphil Conference project/project v1/data/EEGdenoisenet/data/"

    # Build the three loaders
    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=DATA_DIR,
        artifact_type="EMG",
        batch_size=32,
    )

    # Inspect a single training batch to confirm shapes and ranges
    batch = next(iter(train_loader))
    print("\n[Smoke test] First training batch:")
    for k, v in batch.items():
        print(f"  {k:8s} shape={tuple(v.shape)}  min={v.min():.2f}  max={v.max():.2f}")