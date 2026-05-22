"""
Loss functions for EEG causal denoising models.

Available losses:
    mse         — Mean Squared Error / L2 loss (nn.MSELoss).
    l1          — Mean Absolute Error / L1 loss (nn.L1Loss).
    stft        — Pure multi-resolution STFT loss (spectral convergence +
                  log-magnitude, averaged over several FFT resolutions).
    l2_combined — L2 + 0.05 * STFT  (MSE-based combined).
    l1_combined — L1 + 0.05 * STFT  (MAE-based combined, default).

Use ``build_loss(name)`` as the single entry point.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ====================================================================
# Single-resolution STFT loss
# ====================================================================

class SingleResolutionSTFTLoss(nn.Module):
    """
    STFT loss at one (fft_size, hop_length, win_length) setting.

    Two complementary terms:
      * **Spectral convergence** — Frobenius-norm of the magnitude error
        normalised by the Frobenius-norm of the target magnitude.  Captures
        large-scale spectral shape.
      * **Log-magnitude** — L1 distance between log-compressed STFT
        magnitudes.  Sensitive to fine spectral detail and low-energy
        components that the Frobenius norm under-weights.
    """

    def __init__(self, fft_size: int, hop_length: int, win_length: int):
        super().__init__()
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.win_length = win_length
        # Register a Hann window as a buffer (moves with .to(device))
        self.register_buffer(
            "window", torch.hann_window(win_length, periodic=True)
        )

    def _stft_mag(self, x: torch.Tensor) -> torch.Tensor:
        """Compute STFT magnitude for a (B, T) signal."""
        # Suppress the PyTorch internal deprecation warning about output
        # tensor resizing inside _VF.stft (fixed in newer PyTorch versions).
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*output with one or more elements was resized.*",
                category=UserWarning,
            )
            spec = torch.stft(
                x,
                n_fft=self.fft_size,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                return_complex=True,
            )
        return spec.abs()  # (B, n_fft//2+1, frames)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Parameters
        ----------
        pred, target : (B, T) or (B, 1, T)
            Predicted and clean waveforms.

        Returns
        -------
        sc_loss : scalar — spectral convergence loss
        mag_loss : scalar — log-magnitude loss
        """
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        pred_mag = self._stft_mag(pred)
        target_mag = self._stft_mag(target)

        # Spectral convergence: ||target_mag - pred_mag||_F / ||target_mag||_F
        sc_loss = torch.norm(target_mag - pred_mag, p="fro") / (
            torch.norm(target_mag, p="fro") + 1e-8
        )

        # Log-magnitude: L1 on log-compressed magnitudes
        mag_loss = F.l1_loss(
            torch.log(pred_mag + 1e-8), torch.log(target_mag + 1e-8)
        )

        return sc_loss, mag_loss


# ====================================================================
# Multi-Resolution STFT Loss
# ====================================================================

class MultiResolutionSTFTLoss(nn.Module):
    """
    Pure multi-resolution STFT loss (spectral only — no time-domain term).

    Averages (spectral-convergence + log-magnitude) losses over several
    (fft_size, hop, win) settings.  Smaller FFT windows capture transient /
    high-frequency detail; larger windows capture low-frequency structure —
    together they provide broad spectral supervision.

        L_stft = mean_over_resolutions(sc + mag)

    Parameters
    ----------
    fft_sizes : list[int]
        FFT sizes for each resolution (e.g. [64, 128, 256]).
    hop_lengths : list[int]
        Hop lengths matching each FFT size.
    win_lengths : list[int]
        Window lengths matching each FFT size.
    """

    def __init__(
        self,
        fft_sizes: List[int] = None,
        hop_lengths: List[int] = None,
        win_lengths: List[int] = None,
    ):
        super().__init__()

        # Defaults tuned for 512-sample EEG segments at ~256 Hz
        if fft_sizes is None:
            fft_sizes = [64, 128, 256]
        if hop_lengths is None:
            hop_lengths = [16, 32, 64]
        if win_lengths is None:
            win_lengths = [64, 128, 256]

        assert len(fft_sizes) == len(hop_lengths) == len(win_lengths), (
            "fft_sizes, hop_lengths, and win_lengths must have the same length"
        )

        self.stft_losses = nn.ModuleList([
            SingleResolutionSTFTLoss(n, h, w)
            for n, h, w in zip(fft_sizes, hop_lengths, win_lengths)
        ])

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Parameters
        ----------
        pred, target : (B, 1, T) or (B, T)

        Returns
        -------
        loss : scalar — pure spectral loss (averaged over resolutions)
        """
        sc_total = 0.0
        mag_total = 0.0
        for stft_fn in self.stft_losses:
            sc, mag = stft_fn(pred, target)
            sc_total += sc
            mag_total += mag

        n_res = len(self.stft_losses)
        return (sc_total + mag_total) / n_res


# ====================================================================
# L2 Combined Loss: MSE + 0.05 * STFT
# ====================================================================

class L2CombinedLoss(nn.Module):
    """
    L2 (MSE) combined with multi-resolution STFT loss.

        L = w_mse * MSE(pred, target) + w_stft * STFT(pred, target)

    Default: L = 1.0 * MSE + 0.05 * STFT

    Parameters
    ----------
    weight_mse : float
        Weight for the MSE (L2) term.
    weight_stft : float
        Weight for the STFT term.
    """

    def __init__(self, weight_mse: float = 1.0, weight_stft: float = 0.05):
        super().__init__()
        self.weight_mse = weight_mse
        self.weight_stft = weight_stft
        self.mse = nn.MSELoss()
        self.stft = MultiResolutionSTFTLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Parameters
        ----------
        pred, target : (B, 1, T) or (B, T)

        Returns
        -------
        loss : scalar — weighted sum of MSE and STFT losses
        """
        mse_loss = self.mse(pred, target)
        stft_loss = self.stft(pred, target)
        return self.weight_mse * mse_loss + self.weight_stft * stft_loss


# ====================================================================
# L1 Combined Loss: L1 (MAE) + 0.05 * STFT
# ====================================================================

class L1CombinedLoss(nn.Module):
    """
    L1 (MAE) combined with multi-resolution STFT loss.

        L = w_l1 * L1(pred, target) + w_stft * STFT(pred, target)

    Default: L = 1.0 * L1 + 0.05 * STFT

    L1 is more robust to outlier artifacts than MSE because it penalises
    large errors linearly rather than quadratically, preventing the
    optimizer from over-focusing on a few high-amplitude spikes at the
    expense of overall waveform fidelity.

    Parameters
    ----------
    weight_l1 : float
        Weight for the L1 (MAE) term.
    weight_stft : float
        Weight for the STFT term.
    """

    def __init__(self, weight_l1: float = 1.0, weight_stft: float = 0.05):
        super().__init__()
        self.weight_l1 = weight_l1
        self.weight_stft = weight_stft
        self.l1 = nn.L1Loss()
        self.stft = MultiResolutionSTFTLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Parameters
        ----------
        pred, target : (B, 1, T) or (B, T)

        Returns
        -------
        loss : scalar — weighted sum of L1 and STFT losses
        """
        l1_loss = self.l1(pred, target)
        stft_loss = self.stft(pred, target)
        return self.weight_l1 * l1_loss + self.weight_stft * stft_loss


# ====================================================================
# Factory
# ====================================================================

LOSS_REGISTRY = {
    "mse":         "Mean Squared Error / L2 (time-domain only)",
    "l1":          "Mean Absolute Error / L1 (time-domain only)",
    "stft":        "Multi-Resolution STFT (spectral only)",
    "l2_combined": "L2 + 0.05 * STFT",
    "l1_combined": "L1 + 0.05 * STFT (default)",
}


def build_loss(name: str = "l1_combined") -> nn.Module:
    """
    Build a loss function by name.

    Parameters
    ----------
    name : str
        ``"mse"``         — standard MSE / L2 loss.
        ``"l1"``          — standard MAE / L1 loss.
        ``"stft"``        — pure multi-resolution STFT loss.
        ``"l2_combined"`` — L2 + 0.05 * STFT.
        ``"l1_combined"`` — L1 + 0.05 * STFT (default).

    Returns
    -------
    nn.Module with a standard ``(pred, target) -> scalar`` forward().
    """
    name = name.lower().strip()
    if name == "mse":
        return nn.MSELoss()
    elif name == "l1":
        return nn.L1Loss()
    elif name == "stft":
        return MultiResolutionSTFTLoss()
    elif name == "l2_combined":
        return L2CombinedLoss(weight_mse=1.0, weight_stft=0.05)
    elif name == "l1_combined":
        return L1CombinedLoss(weight_l1=1.0, weight_stft=0.05)
    else:
        raise ValueError(
            f"Unknown loss function '{name}'. "
            f"Available: {list(LOSS_REGISTRY.keys())}"
        )
