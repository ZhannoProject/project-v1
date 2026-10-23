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

from models.lookahead import allocate


# ============================================================
# Atomic component: single causal convolution
# ============================================================
class CausalConv1d(nn.Module):
    """
    1D Causal Convolution layer.
    It pads the input sequence on the left side only, ensuring that the model
    cannot "look into the future" when predicting the current time step.

    ``causal=False`` moves the same amount of padding to both sides, so
    position t sees a window centred on itself instead of ending at it.  The
    kernel, the channel counts and the output length are untouched — only
    where the padding goes changes — so a causal layer and its non-causal
    twin have identical weights and interchangeable state dicts.  The name
    stays ``CausalConv1d`` for the same reason ``CausalMHA`` does: the
    causal form is the deployable one, and the switch exists to build the
    offline teacher (see the ``*_bi`` variants in models/variants.py).

    Input  : (B, C_in, T)
    Output : (B, C_out, T)
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, causal=True,
                 lookahead: int | None = None, **kwargs):
        super().__init__()
        # Padding that keeps the sequence length; all of it on the left keeps
        # the layer causal, split evenly it does not.  An odd kernel (this
        # project uses 3) splits exactly; an even one puts the extra tap ahead.
        #
        # ``lookahead`` names the right-hand padding outright, in samples, for
        # the bounded settings between those two: it is how many future taps
        # this layer may use, and the rest of the padding stays on the left so
        # the output length never changes.  ``None`` is the legacy pair of
        # settings and is left exactly as it was.  See models/lookahead.py for
        # who decides the number -- a layer is told its share, it does not
        # work out its own.
        total = (kernel_size - 1) * dilation
        if lookahead is None:
            self.pad_right = 0 if causal else total - total // 2
        else:
            self.pad_right = min(int(lookahead), total)
        self.pad_left = total - self.pad_right
        self.lookahead = self.pad_right
        # Derived, not stored as given: a layer is causal exactly when nothing
        # is padded on its right.  models/incremental.py and step_graph.py
        # refuse a layer whose ``causal`` is False, so a lookahead layer is
        # refused by the streaming paths for the true reason -- it needs
        # samples that have not arrived -- without either of them being taught
        # about lookahead.
        self.causal = self.pad_right == 0
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,         # Padding is handled manually in the forward pass
            dilation=dilation,
            **kwargs,
        )

    def forward(self, x):
        # Pad the temporal dimension: left only when causal, both sides otherwise
        x = F.pad(x, (self.pad_left, self.pad_right))
        return self.conv(x)


# ============================================================
# Squeeze-and-Excitation, without looking ahead
# ============================================================
class CausalSE(nn.Module):
    """
    Channel attention (Hu et al., 2018) in a form a streaming model can use.

    Standard SE squeezes a (B, C, T) feature map to (B, C) by averaging over
    the *whole* sequence, then rescales every channel by a gate computed from
    it.  That is not usable here: the gate applied at t = 0 would depend on
    sample 511, so the model would be reading the future through the channel
    weights while every convolution in it stayed scrupulously causal.  The
    leak would not show up in the loss -- it would show up as a model that
    cannot be deployed, found out much later.

    So the squeeze is a **running** average instead: position t is scaled by a
    gate computed from [0..t] only.

        s_t = (1/(t+1)) * sum_{i<=t} x_i        causal squeeze
        g_t = sigmoid(W2 @ relu(W1 @ s_t))      excitation, per timestep
        y_t = g_t * x_t                         scale

    Within one 512-sample epoch -- which is how these models are trained and
    scored -- this is the exact causal counterpart of SE's global pool. It
    also has a one-line recurrent form (keep the running sum and the count),
    which is what lets it stream a sample at a time later.

    Cost is two 1x1 convolutions on a C/r bottleneck, so a few thousand
    parameters and no temporal arithmetic at all. That is the point of trying
    it: attention costs this project roughly 5x the baseline's compute, and
    this is the cheapest thing that could plausibly do a similar job.

    ``causal=False`` restores the standard global pool, for the non-deployable
    ``*_bi`` teachers. Only the pooling changes -- the parameters are the
    same either way, so a teacher and its causal twin still share a state
    dict and can be distilled into one another.

    ``lookahead=L`` is the bounded setting in between: position t is scaled by
    a gate computed from ``[0 .. t + L]``, so the gate may peek L samples past
    the sample it is scaling and no further. L = 0 is the running mean above,
    and the global pool is the limit as L reaches the end of the epoch -- one
    continuous knob whose two ends are the two settings that already existed.
    """

    def __init__(self, channels: int, reduction: int = 8, causal: bool = True,
                 lookahead: int | None = None):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.lookahead = None if lookahead is None else max(0, int(lookahead))
        # The unbounded pool is the one setting that is not a lookahead at all,
        # so it is flagged rather than inferred at every forward pass.
        self.global_pool = lookahead is None and not causal
        # As in CausalConv1d: causal means this gate uses nothing ahead of the
        # position it scales, which a nonzero lookahead is precisely not.
        self.causal = causal and not self.lookahead
        self.reduction = reduction
        # 1x1 convolutions rather than Linear: they apply per timestep without
        # any reshaping, which keeps the whole module in (B, C, T).
        self.fc1 = nn.Conv1d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden, channels, kernel_size=1)
        self.act = nn.ReLU()

    def squeeze(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) -> (B, C, T): the context each position is allowed."""
        if self.global_pool:
            return x.mean(dim=2, keepdim=True).expand_as(x)
        T = x.shape[2]
        counts = torch.arange(1, T + 1, device=x.device, dtype=x.dtype)
        running = x.cumsum(dim=2) / counts.view(1, 1, -1)
        if not self.lookahead:
            return running
        # The gate for position t is the running mean at t + L: the same
        # quantity, read L samples later.  Clamped at the end of the epoch,
        # where there is no more future to read -- the same truncation the
        # convolutions' padding makes at that edge, rather than a wrap or a
        # different rule for the tail.
        idx = (torch.arange(T, device=x.device) + self.lookahead).clamp(max=T - 1)
        return running.index_select(2, idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.fc2(self.act(self.fc1(self.squeeze(x)))))


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
    ``causal`` is passed straight to both convolutions; the 1×1 shortcut
    has no temporal extent and is unaffected.

    Input  : (B, C_in, T)
    Output : (B, C_out, T)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1,
                 causal=True, se_reduction: int | None = None,
                 conv_lookahead: tuple | None = None, se_lookahead: int | None = None):
        super().__init__()
        # The block is told its sites' shares rather than working them out: the
        # budget is global and only CausalTCNBackbone sees the whole stack, so
        # splitting it here as well would round twice and the total would stop
        # matching what was asked for.
        la1, la2 = conv_lookahead if conv_lookahead is not None else (None, None)
        # Main convolutional pathway
        self.conv_block = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation, causal=causal,
                         lookahead=la1),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation, causal=causal,
                         lookahead=la2),
            nn.PReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout),
        )

        # Squeeze-and-Excitation on the block's output, before the residual
        # add -- SE-ResNet's placement. ``None`` leaves the block exactly as it
        # was, so a model built without it is unchanged down to the parameter
        # ordering and its checkpoints still load.
        self.se = (CausalSE(out_channels, se_reduction, causal=causal, lookahead=se_lookahead)
                   if se_reduction else None)

        # Shortcut connection for the residual addition
        if in_channels != out_channels:
            # Match channel dimensions using a 1x1 convolution if they differ
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv_block(x)
        if self.se is not None:
            out = self.se(out)
        # Element-wise addition of the input (shortcut) and the block's output
        return out + self.shortcut(x)


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

    ``causal=False`` centres every convolution's padding instead of putting
    it all on the left, which makes the whole stack non-causal: with
    dilations [1, 2, 4, 8, 16] and kernel 3 each position then sees 31
    samples ahead as well as behind.  Weights and output shape are
    unchanged, so the two forms share a state dict.  Used for the offline
    ``tcn_bi`` teacher; the deployable models leave it at True.

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
        causal: bool = True,
        se_reduction: int | None = None,
        lookahead: int | None = None,
    ):
        super().__init__()

        # One budget, spent once, here: this is the only object that sees every
        # site in the stack, and the realised forward reach is the sum over all
        # of them (see models/lookahead.py).  Sites are listed in the order the
        # blocks are built so a share can be handed to the layer it belongs to.
        per_conv = [(kernel_size - 1) * d for d in dilations]
        sites = [p for d in per_conv for p in (d, d)]          # two convs per block
        if se_reduction:
            sites += per_conv                                  # one SE gate per block
        shares = allocate(sites, lookahead)
        n_conv = 2 * len(dilations)
        conv_shares = None if shares is None else shares[:n_conv]
        se_shares = (shares[n_conv:] if shares is not None and se_reduction
                     else [None] * len(dilations))

        # Build the layer stack with the specified dilation schedule
        layers = []
        for i, dilation_rate in enumerate(dilations):
            # First block: in_channels → hidden_channels
            # All subsequent blocks: hidden_channels → hidden_channels
            c_in = in_channels if i == 0 else hidden_channels
            layers.append(
                ResidualBlock(c_in, hidden_channels, kernel_size, dilation=dilation_rate,
                              dropout=dropout, causal=causal, se_reduction=se_reduction,
                              conv_lookahead=(None if conv_shares is None
                                              else (conv_shares[2 * i], conv_shares[2 * i + 1])),
                              se_lookahead=se_shares[i])
            )
        self.network = nn.Sequential(*layers)

        # Store for external inspection (receptive field calculation, etc.)
        self.dilations = dilations
        self.hidden_channels = hidden_channels
        self.lookahead = lookahead
        #: What the budget actually bought, summed over every site.  Equal to
        #: ``lookahead`` unless it was clamped at capacity, and equal to the
        #: forward receptive field that models/lookahead.py measures.
        self.granted_lookahead = None if shares is None else sum(shares)
        self.causal = causal and not self.granted_lookahead

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