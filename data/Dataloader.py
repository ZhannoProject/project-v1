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
def scale_artifact(clean: np.ndarray, artifact: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Scale an artifact so that mixing it with `clean` achieves the target SNR.

    Returns the scaled artifact (NOT the mixed signal).  This lets callers
    stack multiple scaled artifacts before adding them to the clean signal:

        noisy = clean + scale_artifact(clean, eog, snr1) + scale_artifact(clean, emg, snr2)

    Formula: scaled = lambda * artifact
             where lambda = RMS(clean) / (RMS(artifact) * 10^(SNR_dB/20))
    """
    rms_clean = np.sqrt(np.mean(clean ** 2))
    rms_art = np.sqrt(np.mean(artifact ** 2))

    if rms_art < 1e-8:
        return np.zeros_like(artifact)

    lam = rms_clean / (rms_art * 10 ** (snr_db / 20))
    return lam * artifact


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
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),  # only training set is shuffled
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