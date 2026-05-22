"""
Hardware utilities shared across the trainer package.
"""

import torch


def get_device():
    """Selects the best available hardware device (CUDA, Apple MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
