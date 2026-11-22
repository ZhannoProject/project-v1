"""Stateful forward-only filtering; usable on the Pi without PyTorch."""

import numpy as np


class CausalFilter:
    def __init__(self, fs, band=(1., 80.), notch_hz=60., order=4):
        from scipy.signal import butter, iirnotch, tf2sos
        self.stages = []
        low, high = band
        if low:
            self.stages.append(butter(order, low, btype="highpass", fs=fs, output="sos"))
        if notch_hz and notch_hz < fs/2:
            self.stages.append(tf2sos(*iirnotch(notch_hz, 30., fs=fs)))
        if high and high < fs/2:
            self.stages.append(butter(order, high, btype="lowpass", fs=fs, output="sos"))
        self.state = [np.zeros((len(sos), 2)) for sos in self.stages]

    def process(self, chunk):
        from scipy.signal import sosfilt
        y = np.asarray(chunk, dtype=np.float64)
        if y.ndim != 1 or not np.isfinite(y).all():
            raise ValueError("Filter input must be a finite 1-D signal")
        for i, sos in enumerate(self.stages):
            y, self.state[i] = sosfilt(sos, y, zi=self.state[i])
        return y.astype(np.float32)
