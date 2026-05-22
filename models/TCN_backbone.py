"""
Causal Temporal Convolutional Network (TCN) building blocks.

This file provides three levels of abstraction:

  CausalConv1d        — single causal convolution (left-padding only)
  ResidualBlock       — two CausalConv1d + PReLU + BN + Dropout + skip
  CausalTCNBackbone   — stack of ResidualBlocks with configurable dilations

CausalTCNBackbone replaces the old EEGCausualAllCNN. The key change is
using a `dilations` list instead of `num_layers`, so the same class can
build both a full TCN and a split TCN (pt1 / pt2) for the sandwich
architecture.

============================================================================
All modules operate in (B, C, T) — channel-first, matching nn.Conv1d.
============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ============================================================
# Atomic component: single causal convolution
# ============================================================
class CausalConv1d(nn.Module):
    """
    1D Causal Convolution layer.
    It pads the input sequence on the left side only, ensuring that the model
    cannot "look into the future" when predicting the current time step.

    Input  : (B, C_in, T)
    Output : (B, C_out, T)
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        # Calculate padding needed to maintain sequence length while remaining causal
        self.pad_left = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,         # Padding is handled manually in the forward pass
            dilation=dilation,
            **kwargs,
        )

    def forward(self, x):
        # Pad the temporal dimension strictly on the left
        x = F.pad(x, (self.pad_left, 0))
        return self.conv(x)


# ============================================================
# Residual block: two causal convolutions + skip connection
# ============================================================
class ResidualBlock(nn.Module):
    """
    A Temporal Convolutional Network (TCN) Residual Block containing two Causal Conv
    layers, activation functions, batch normalization, and dropout.

        x → CausalConv → PReLU → BN → Dropout
          → CausalConv → PReLU → BN → Dropout → (+shortcut) → out

    If in_channels != out_channels, a 1×1 conv aligns the skip path.

    Input  : (B, C_in, T)
    Output : (B, C_out, T)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        # Main convolutional pathway
        self.conv_block = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout),
        )

        # Shortcut connection for the residual addition
        if in_channels != out_channels:
            # Match channel dimensions using a 1x1 convolution if they differ
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        # Element-wise addition of the input (shortcut) and the block's output
        return self.conv_block(x) + self.shortcut(x)


# ============================================================
# Configurable TCN backbone
# ============================================================
class CausalTCNBackbone(nn.Module):
    """
    The main Causal TCN backbone for EEG Denoising.
    Stacks multiple ResidualBlocks with caller-specified dilation rates
    to grow the receptive field.

    Using a `dilations` list instead of `num_layers` gives full control:
      - Full TCN    : dilations=[1, 2, 4, 8, 16]
      - TCN pt1     : dilations=[1, 2, 4]          (small RF, tokenise)
      - TCN pt2     : dilations=[8, 16, 32]         (large RF, reconstruct)

    The first block handles in_channels → hidden_channels.
    All subsequent blocks are hidden_channels → hidden_channels.
    No output head here — that belongs in the variant model.

    Input  : (B, in_channels, T)
    Output : (B, hidden_channels, T)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        dilations: List[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Build the layer stack with the specified dilation schedule
        layers = []
        for i, dilation_rate in enumerate(dilations):
            # First block: in_channels → hidden_channels
            # All subsequent blocks: hidden_channels → hidden_channels
            c_in = in_channels if i == 0 else hidden_channels
            layers.append(
                ResidualBlock(c_in, hidden_channels, kernel_size, dilation=dilation_rate, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

        # Store for external inspection (receptive field calculation, etc.)
        self.dilations = dilations
        self.hidden_channels = hidden_channels

    def forward(self, x):
        return self.network(x)


# ============================================================
# Self-test
# ============================================================
if __name__ == "__main__":
    batch_size = 4
    seq_length = 512

    # Full backbone (equivalent to old EEGCausualAllCNN without the head)
    full = CausalTCNBackbone(in_channels=1, hidden_channels=64, dilations=[1, 2, 4, 8, 16])
    noisy_EEG = torch.randn(batch_size, 1, seq_length)
    y = full(noisy_EEG)
    print(f"Full TCN : {noisy_EEG.shape} → {y.shape}")

    # Split backbone pt1 (small dilations, tokenise)
    pt1 = CausalTCNBackbone(in_channels=1, hidden_channels=64, dilations=[1, 2, 4])
    y1 = pt1(noisy_EEG)
    print(f"TCN pt1  : {noisy_EEG.shape} → {y1.shape}")

    # Split backbone pt2 (large dilations, reconstruct)
    pt2 = CausalTCNBackbone(in_channels=64, hidden_channels=64, dilations=[8, 16, 32])
    y2 = pt2(y1)
    print(f"TCN pt2  : {y1.shape} → {y2.shape}")

    # Parameter counts
    for name, model in [("Full", full), ("pt1", pt1), ("pt2", pt2)]:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:>4s} params: {n_params:,}")