"""
EEGDenoiseNet standard evaluation metrics from the paper
- RRMSE_temporal : Relative Root Mean Square Error in the time domain
- RRMSE_spectral : Relative Root Mean Square Error in the frequency domain
- CC            : Average Correlation Coefficient
"""
import warnings

import torch


def rms(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Root Mean Square — Calculated along the specified dimension"""
    return torch.sqrt(torch.mean(x ** 2, dim=dim) + 1e-12)


def rrmse_temporal(pred: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """
    Equation 8 in the paper: RRMSE_t = RMS(f(y) - x) / RMS(x)

    pred, clean: (B, 1, T) or (B, T)
    Returns: (B,) RRMSE for each sample
    """
    if pred.dim() == 3:
        pred = pred.squeeze(1)
        clean = clean.squeeze(1)
    
    return rms(pred - clean, dim=-1) / rms(clean, dim=-1)


def rrmse_spectral(pred: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """
    Equation 9 in the paper: RRMSE_f = RMS(PSD(f(y)) - PSD(x)) / RMS(PSD(x))

    Using FFT amplitude spectrum instead of PSD (PSD is a normalized version of |FFT|^2,
    which is equivalent when used as a relative error for the numerator and denominator)
    """
    if pred.dim() == 3:
        pred = pred.squeeze(1)
        clean = clean.squeeze(1)
    
    # Real FFT → power spectrum: |FFT|^2
    # Suppress the PyTorch internal deprecation warning about output tensor
    # resizing inside rfft (fixed in newer PyTorch versions).
    n = pred.shape[-1]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*output with one or more elements was resized.*",
            category=UserWarning,
        )
        pred_psd = torch.fft.rfft(pred, n=n, dim=-1).abs().pow(2)
        clean_psd = torch.fft.rfft(clean, n=n, dim=-1).abs().pow(2)
    
    return rms(pred_psd - clean_psd, dim=-1) / rms(clean_psd, dim=-1)


def acc(pred: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """
    Equation 10 in the paper: CC = Cov(f(y), x) / sqrt(Var(f(y)) * Var(x))
    
    This is the Pearson correlation coefficient.
    """
    if pred.dim() == 3:
        pred = pred.squeeze(1)
        clean = clean.squeeze(1)

    pred_mean  = pred.mean(dim=-1, keepdim=True)
    clean_mean = clean.mean(dim=-1, keepdim=True)
    
    pred_c  = pred - pred_mean
    clean_c = clean - clean_mean
    
    cov = (pred_c * clean_c).mean(dim=-1)
    var_pred  = (pred_c ** 2).mean(dim=-1)
    var_clean = (clean_c ** 2).mean(dim=-1)
    
    return cov / (torch.sqrt(var_pred * var_clean) + 1e-12)


def compute_all_metrics(pred: torch.Tensor, clean: torch.Tensor) -> dict:
    """Calculate three metrics at once, returning the batch average"""
    return {
        "rrmse_t": rrmse_temporal(pred, clean).mean().item(),
        "rrmse_s": rrmse_spectral(pred, clean).mean().item(),
        "acc":     acc(pred, clean).mean().item(),
    }


# ── Self-check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # Perfect prediction: rrmse=0, acc=1
    x = torch.randn(8, 1, 512)
    print("[Perfect Prediction]")
    print(compute_all_metrics(x, x))
    
    # Completely random prediction: rrmse approaches sqrt(2), acc approaches 0
    print("\n[Random Prediction]")
    print(compute_all_metrics(torch.randn(8, 1, 512), x))
    
    # Expected values in paper (complex CNN on EOG): rrmse_t≈0.34, acc≈0.92
    # Add a small amount of noise to see if it falls in a reasonable range
    pred = x + 0.3 * torch.randn_like(x)
    print("\n[Add 30% Noise]")
    print(compute_all_metrics(pred, x))