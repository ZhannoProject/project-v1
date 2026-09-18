"""
Hardware and reproducibility utilities shared across the trainer package.
"""

import os
import random

import numpy as np
import torch


DEVICE_CHOICES = ["auto", "cpu", "cuda", "mps"]


def get_device():
    """Selects the best available hardware device (CUDA, Apple MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(name: str = "auto") -> torch.device:
    """
    ``auto`` → get_device(); otherwise the named backend, checked for
    availability.  ``cpu`` is the choice for bit-exact reproducibility:
    the MPS backend is not deterministic even when seeded (two identical
    runs drift at the 1e-4 level within one epoch), and CUDA is only
    deterministic with the kernels seed_everything() switches on.
    """
    if name == "auto":
        return get_device()
    if name not in DEVICE_CHOICES:
        raise ValueError(f"device must be one of {DEVICE_CHOICES}, got {name!r}")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device=mps requested but MPS is not available")
    return torch.device(name)


def seed_everything(seed: int) -> int:
    """
    Seed every RNG a training run draws from, so the same config gives the
    same weights, dropout masks and batch order on the same hardware.

    Covers Python's ``random``, NumPy's legacy global RNG, and torch on
    CPU / every CUDA device.  cuDNN is switched to its deterministic kernels
    (``benchmark`` off, since autotuning picks kernels non-deterministically)
    and ``torch.use_deterministic_algorithms`` is enabled in warn-only mode,
    with the cuBLAS workspace pinned as that requires.  ``PYTHONHASHSEED``
    is set for any subprocess the run spawns.

    The data pipeline has its own seeds (``master_seed`` → numpy
    ``default_rng`` in data/Dataloader.py and a ``torch.Generator`` for the
    training DataLoader), so split, SNR and batch order do not depend on how
    much of the global stream model construction consumed.

    Bit-exact results across different GPUs / CPUs / torch versions are not
    guaranteed by any seeding; this removes the run-to-run variance on CPU
    and CUDA.  Apple MPS stays non-deterministic regardless (see
    resolve_device): use ``--device cpu`` when two runs must match exactly.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Must be set before the first cuBLAS call for deterministic GEMMs.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)          # also seeds every CUDA device
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return seed
