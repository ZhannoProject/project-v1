"""Rational FIR conversion with explicit, uncompensated causal filter delay."""

from fractions import Fraction
import numpy as np


def ratio(fs_in, fs_out):
    if not np.isfinite(fs_in) or not np.isfinite(fs_out) or min(fs_in, fs_out) <= 0:
        raise ValueError("Sampling rates must be finite and positive")
    value = Fraction(str(fs_out)) / Fraction(str(fs_in))
    value = value.limit_denominator(10000)
    return value.numerator, value.denominator


def resampling_info(fs_in, fs_out):
    up, down = ratio(fs_in, fs_out)
    taps = 1 if up == down else 20 * max(up, down) + 1
    return dict(method="causal_fir", up=up, down=down, taps=taps,
                delay_ms=1000 * (taps - 1) / (2 * up * fs_in),
                warmup_samples=int(np.ceil((taps - 1) / down)))


def causal_resample(x, fs_in, fs_out):
    """y[j] uses x[:floor(j*down/up)+1]. Never trim away the FIR delay."""
    from scipy.signal import firwin, upfirdn
    up, down = ratio(fs_in, fs_out)
    x = np.asarray(x)
    if up == down:
        return np.ascontiguousarray(x, dtype=np.float32)
    taps = firwin(20 * max(up, down) + 1, 1 / max(up, down), window=("kaiser", 5.)) * up
    n = (x.shape[-1] * up + down - 1) // down
    return np.ascontiguousarray(upfirdn(taps, x, up, down, axis=-1)[..., :n], dtype=np.float32)
