"""
trainer package — modular training pipeline for EEG denoising models.

Modules:
    training.py   — Trainer class (train / val loops, fit, staged_fit)
    distillation.py — KDTrainer + DistillationLoss (bidirectional teacher → causal student)
    losses.py     — loss function registry (MSE, L1, STFT, L2 combined, L1 combined)
    optimizer.py  — param-group construction, freeze / unfreeze helpers
    checkpoint.py — model checkpointing and history I/O
    logger.py     — TensorBoard and console logging
    utils.py      — device selection
    __main__.py   — CLI entry point (``python -m trainer``)
"""

from trainer.training import Trainer
from trainer.distillation import KDTrainer, DistillationLoss, load_teacher
from trainer.losses import build_loss
from utils.utils import get_device
