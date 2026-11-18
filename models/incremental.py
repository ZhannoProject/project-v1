"""
Incremental (streaming) inference for the causal denoising variants.

Why this file exists
====================
``deploy/raspi_infer.py`` streams by sliding a 512-sample window forward by
``hop`` samples and keeping the newest ``hop`` outputs of each call.  That is
correct, and it is also wasteful: every hop recomputes all 512 positions to
keep 32 of them, so 94% of the arithmetic is thrown away.  Worse for this
project, it makes MLA's central claim unmeasurable.  MLA compresses the KV
*cache*; a path with no cache cannot show a cache win, which is why
``export_onnx.py`` has to say "the graph has no KV cache: every call
recomputes".

This module is the path with the cache.  It advances the model one sample at a
time, keeping exactly the state a causal model needs:

  * every ``CausalConv1d`` keeps a ring buffer of its ``(k-1)*dilation`` past
    inputs — its receptive field, nothing more,
  * ``CausalMHA`` keeps K and V for the last ``max_seq_len`` steps,
    ``2 * d_model`` floats per step,
  * ``CausalMLA`` keeps only the compressed latent ``c_kv``,
    ``kv_lora_rank`` floats per step — the 4x compression the architecture
    was chosen for, now an actual measurement rather than an estimate.

Equivalence, and the one place it is not exact
==============================================
Against a **single forward pass over the whole signal** the stream is exact to
floating-point tolerance for every variant: ~3e-7 on a 300-sample signal.  That
is the property that matters for correctness, and it is what pins the caches,
the ring buffers and the warmup behaviour.

Against the **sliding-window path** in ``deploy/raspi_infer.py`` the two agree
exactly only for the TCN baseline (~7e-8).  For the attention variants they
differ by ~1e-5 relative to signal RMS, and the reason is worth stating because
it is a property of the architecture rather than a bug in either path:

  * The TCN receptive field is 253 samples (sandwich) or 225 (parallel), both
    shorter than the 512-sample attention window.  So at the window's *last*
    position the convolutional taps are all real samples in both paths, and
    the baseline — which is only convolutions — agrees to float noise.
  * But attention at that last position attends over all 512 cached positions,
    and in the windowed path the earliest ~252 of them were computed by a TCN
    that was reading the window's left zero-padding.  The stream has real
    history there instead.  The attention inputs therefore genuinely differ on
    the oldest positions of the cache, and ~1e-5 of that survives into the
    output.

Neither is "wrong".  The windowed path reproduces the training-time
computation exactly — training epochs are 512 samples with a zero-padded left
edge, so that is the function the weights were fitted to.  The stream gives the
oldest cached positions their fully resolved features, which is better-informed
and very slightly off-distribution.  The difference is bounded, does not grow
with ``t``, and is five orders of magnitude below an RRMSE of ~0.65, so it
cannot move any reported metric.  ``tests/test_incremental.py`` asserts both
tolerances separately, so a real regression cannot hide inside the looser one.

Before the cache fills, the stream matches a forward pass over ``[0, t]`` with
the same left zero-padding, exactly.

Two properties of the architecture make it this simple
=====================================================
**There is no positional encoding anywhere in the decoder.**  Attention is
therefore permutation-equivariant over the cached past: softmax and the
weighted sum do not care what order the cache is in.  The KV ring buffer can
wrap around and overwrite its oldest slot in place, with no reindexing and no
rolling — the thing that usually makes a ring-buffer KV cache fiddly does not
arise.

**The non-attention layers are pointwise in time.**  ``PReLU``, ``LayerNorm``,
the FFN, ``BatchNorm1d`` in eval mode (running statistics, so an affine map),
``Dropout`` in eval mode (identity) and every ``kernel_size=1`` convolution
(``squeeze``, ``expand``, the heads) carry no temporal state at all.  Only
dilated convolutions and attention need caching.

The steppers hold references to the trained modules and read their weight
tensors directly.  Nothing is copied and no layer is reimplemented, so the
streaming path cannot drift from the batch path as the models change.

Usage
=====
    from models.incremental import IncrementalModel

    stream = IncrementalModel(model)          # model must be causal + eval()
    y = stream.run(signal)                    # 1-D in, 1-D out
    print(stream.cache_report())              # state bytes, split by kind

    stream.reset()
    for x_t in signal:                        # or one sample at a time
        y_t = stream.step(x_t)

``IncrementalModel`` refuses bidirectional models (``tcn_*_bi``,
``tcn_*_full_bi``): they attend to future samples and have no streaming form.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.AttentionBlocks.attention import (
    BottleneckSkip,
    CausalMHA,
    CausalMLA,
    CausalTransformerBlock,
    CausalTransformerDecoder,
)
from models.TCN_backbone import CausalConv1d, CausalTCNBackbone, ResidualBlock
from models.causal_config import ChannelNorm

#: Bytes per cached float.  The caches are float32 like the weights; this is
#: the unit every number in ``cache_report()`` is quoted in.
BYTES_FP32 = 4


# ============================================================
# Convolution: ring buffer over the receptive field
# ============================================================
class ConvStepper:
    """
    One ``CausalConv1d``, advanced a sample at a time.

    A causal convolution with kernel ``k`` and dilation ``d`` reads taps at
    offsets ``0, -d, ..., -(k-1)*d``, so it needs the last ``(k-1)*d`` inputs
    and nothing else.  The buffer holds ``(k-1)*d + 1`` columns (the taps plus
    the current sample) and a write pointer walks it; no column is ever moved.

    Starting the buffer at zero is exactly the left zero-padding
    ``CausalConv1d.forward`` applies with ``F.pad``, which is what makes the
    first ``(k-1)*d`` steps agree with a windowed forward pass rather than
    merely resemble one.

    ``kernel_size=1`` needs no history at all and takes the stateless path.
    """

    def __init__(self, conv: CausalConv1d):
        if not conv.causal:
            raise ValueError(
                "ConvStepper needs a causal convolution; this one centres its padding "
                "and reads samples ahead of the current one."
            )
        self.conv = conv.conv                      # the bare nn.Conv1d
        self.kernel_size = self.conv.kernel_size[0]
        self.dilation = self.conv.dilation[0]
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels

        # Weight flattened once, to turn k gathered taps into a single matmul.
        # nn.Conv1d stores (C_out, C_in, k) with k contiguous, so the flattened
        # column index is c_in * k + tap, which is the layout the gather below
        # produces.
        self._w2 = self.conv.weight.reshape(self.out_channels, self.in_channels * self.kernel_size)
        self._bias = self.conv.bias

        self.span = (self.kernel_size - 1) * self.dilation   # samples of history needed
        self.length = self.span + 1                          # ring columns
        self.stateless = self.span == 0

        # Tap index table: row p gives, for pointer p, the column holding each
        # of the k taps.  Precomputed so the hot path has no modular arithmetic.
        if not self.stateless:
            offsets = torch.tensor(
                [-(self.kernel_size - 1 - i) * self.dilation for i in range(self.kernel_size)],
                dtype=torch.long,
            )
            table = (torch.arange(self.length).unsqueeze(1) + offsets.unsqueeze(0)) % self.length
            self._tap_idx = table                            # (length, k)

        self._buf: Optional[torch.Tensor] = None
        self._p = 0

    # -- state ------------------------------------------------
    def reset(self, batch_size: int, device, dtype):
        if self.stateless:
            return
        self._buf = torch.zeros(batch_size, self.in_channels, self.length,
                                device=device, dtype=dtype)
        # First write lands on column 0.
        self._p = self.length - 1

    def state_bytes(self) -> int:
        return 0 if self._buf is None else self._buf.numel() * BYTES_FP32

    # -- step -------------------------------------------------
    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        """``x_t``: (B, C_in) -> (B, C_out)."""
        if self.stateless:
            return F.linear(x_t, self._w2, self._bias)

        self._p = (self._p + 1) % self.length
        self._buf[:, :, self._p] = x_t

        taps = self._buf.index_select(2, self._tap_idx[self._p])   # (B, C_in, k)
        return F.linear(taps.reshape(taps.shape[0], -1), self._w2, self._bias)


# ============================================================
# Pointwise and composite TCN pieces
# ============================================================
def _is_pointwise(module: nn.Module) -> bool:
    """
    True for layers that map sample t to sample t with no temporal state.

    ``BatchNorm1d`` qualifies **only in eval mode**, where it applies the
    recorded running statistics as a fixed affine map; in training mode it
    would normalise over a batch of one sample.  ``IncrementalModel`` forces
    eval, and this check backs that up rather than trusting it.
    """
    if isinstance(module, nn.BatchNorm1d):
        if module.training:
            raise RuntimeError(
                "BatchNorm1d is in training mode; streaming would normalise each sample "
                "against itself. Call model.eval() first."
            )
        return True
    return isinstance(module, (nn.PReLU, nn.ReLU, nn.GELU, nn.Identity, nn.Dropout,
                               nn.LayerNorm, ChannelNorm, nn.Sigmoid, nn.Tanh))


def _step_pointwise(module: nn.Module, x_t: torch.Tensor) -> torch.Tensor:
    """
    Apply a pointwise layer to one sample, shaped the way that layer expects.

    The channel-first layers (``BatchNorm1d``, ``PReLU`` with per-channel
    weights) want (B, C, L); the channel-last ones want (B, C).  One unsqueeze
    either way keeps the trained module itself in the loop, so there is no
    second implementation of its arithmetic to keep in sync.
    """
    if isinstance(module, (nn.BatchNorm1d, nn.PReLU)):
        return module(x_t.unsqueeze(-1)).squeeze(-1)
    return module(x_t)


class SequentialStepper:
    """
    An ``nn.Sequential`` of convolutions and pointwise layers, stepped in order.

    Walking the container rather than hard-coding its contents means the
    ``head`` stacks (``CausalConv1d`` -> ``PReLU`` -> ``CausalConv1d``) and the
    two-convolution bodies inside ``ResidualBlock`` are handled by the same
    code, and a layer added to either keeps working or fails loudly.
    """

    def __init__(self, seq: nn.Module):
        self.items: List[tuple] = []
        for child in (seq if isinstance(seq, nn.Sequential) else [seq]):
            if isinstance(child, CausalConv1d):
                self.items.append(("conv", ConvStepper(child)))
            elif isinstance(child, nn.Conv1d) and child.kernel_size[0] == 1:
                # The 1x1 residual shortcut: no temporal extent, so no state.
                self.items.append(("pointwise1x1", child))
            elif _is_pointwise(child):
                self.items.append(("pointwise", child))
            else:
                raise TypeError(
                    f"SequentialStepper cannot step {type(child).__name__}; it is neither a "
                    f"causal convolution nor pointwise in time."
                )

    def reset(self, batch_size, device, dtype):
        for kind, item in self.items:
            if kind == "conv":
                item.reset(batch_size, device, dtype)

    def state_bytes(self) -> int:
        return sum(item.state_bytes() for kind, item in self.items if kind == "conv")

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        for kind, item in self.items:
            if kind == "conv":
                x_t = item.step(x_t)
            elif kind == "pointwise1x1":
                x_t = F.linear(x_t, item.weight.squeeze(-1), item.bias)
            else:
                x_t = _step_pointwise(item, x_t)
        return x_t


class SEStepper:
    """
    ``CausalSE`` advanced one sample at a time.

    The squeeze is a running mean, so streaming it needs exactly two numbers:
    the sum so far and how many samples went into it. That is the whole state
    -- no window, no ring buffer -- which is why this gate costs so much less
    to stream than attention does.

    A non-causal SE pools over the whole epoch and has no streaming form at
    all; it is refused rather than approximated by a running mean, which
    would be a different model wearing the same weights.
    """

    def __init__(self, se):
        if not se.causal:
            raise ValueError(
                "this SE block pools over the whole sequence (causal=False); its gate at t "
                "depends on samples after t, so it cannot be streamed. Stream the causal twin.")
        self.se = se
        self.channels = se.fc1.in_channels
        self._sum = None
        self._n = 0

    def reset(self, batch_size, device, dtype):
        self._sum = (None if self.se.pool_window else
                     torch.zeros(batch_size, self.channels, device=device, dtype=dtype))
        self._n = 0
        self._buffer = (torch.zeros(batch_size, self.channels, self.se.pool_window,
                                   device=device, dtype=dtype) if self.se.pool_window else None)

    def state_bytes(self) -> int:
        if self.se.pool_window:
            return 0 if getattr(self, "_buffer", None) is None else self._buffer.numel() * BYTES_FP32
        return 0 if self._sum is None else self._sum.numel() * BYTES_FP32

    def step(self, x_t):
        """``x_t``: (B, C) -> (B, C), scaled by the gate this sample earns."""
        if self.se.pool_window:
            self._buffer = torch.cat((self._buffer[..., 1:], x_t.unsqueeze(-1)), dim=-1)
            s = self._buffer.mean(-1)
        else:
            self._sum = self._sum + x_t
            self._n += 1
            s = self._sum / self._n
        h = F.relu(F.linear(s, self.se.fc1.weight.squeeze(-1), self.se.fc1.bias))
        g = torch.sigmoid(F.linear(h, self.se.fc2.weight.squeeze(-1), self.se.fc2.bias))
        return x_t * g


class ResidualBlockStepper:
    """``ResidualBlock``: the two-convolution body plus its skip, both stepped."""

    def __init__(self, block: ResidualBlock):
        self.body = SequentialStepper(block.conv_block)
        # Optional, and skipping it silently would be the worst outcome: the
        # stream would return plausible numbers from a model that is not the
        # one on disk.
        self.se = SEStepper(block.se) if getattr(block, "se", None) is not None else None
        shortcut = block.shortcut
        if isinstance(shortcut, nn.Identity):
            self.shortcut = None
        elif isinstance(shortcut, nn.Conv1d) and shortcut.kernel_size[0] == 1:
            self.shortcut = shortcut
        else:
            raise TypeError(f"Unsupported ResidualBlock shortcut: {type(shortcut).__name__}")

    def reset(self, batch_size, device, dtype):
        self.body.reset(batch_size, device, dtype)
        if self.se is not None:
            self.se.reset(batch_size, device, dtype)

    def state_bytes(self) -> int:
        return self.body.state_bytes() + (self.se.state_bytes() if self.se else 0)

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        skip = x_t if self.shortcut is None else F.linear(
            x_t, self.shortcut.weight.squeeze(-1), self.shortcut.bias)
        out = self.body.step(x_t)
        if self.se is not None:
            out = self.se.step(out)
        return out + skip


class TCNStepper:
    """``CausalTCNBackbone``: its residual blocks in sequence."""

    def __init__(self, backbone: CausalTCNBackbone):
        if not backbone.causal:
            raise ValueError(
                "This TCN backbone was built with causal=False (centred padding); it reads "
                "samples ahead of the current one and has no streaming form."
            )
        self.blocks = [ResidualBlockStepper(b) for b in backbone.network]

    @property
    def receptive_field(self) -> int:
        """Samples of history the stack depends on, counting the current sample."""
        return 1 + sum(s.span for blk in self.blocks
                       for kind, s in blk.body.items if kind == "conv")

    def reset(self, batch_size, device, dtype):
        for b in self.blocks:
            b.reset(batch_size, device, dtype)

    def state_bytes(self) -> int:
        return sum(b.state_bytes() for b in self.blocks)

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x_t = b.step(x_t)
        return x_t


# ============================================================
# Attention: the caches that are the point of the exercise
# ============================================================
class _RingCache:
    """
    Fixed-capacity cache over the trained attention window.

    Capacity is ``max_seq_len``, read off the model's own causal mask, so the
    stream offers each position the same amount of context the model was
    trained with and never more.

    Once full it overwrites its oldest slot in place.  That is sound here
    *because the decoder has no positional encoding*: the scores and the
    weighted sum are symmetric in the cached positions, so the physical order
    of the slots is not observable in the output.  A model with positional
    embeddings would need the slots kept in order, or the positions re-encoded
    on every wrap.

    Before the cache fills, the written slots are a contiguous prefix, so the
    valid region is a plain slice and the warmup needs no mask.
    """

    def __init__(self, capacity: int, *, trailing: tuple, heads: Optional[int] = None):
        self.capacity = capacity
        self.trailing = trailing            # per-slot shape, e.g. (head_dim,) or (rank,)
        self.heads = heads
        self.buf: Optional[torch.Tensor] = None
        self.n_valid = 0
        self._p = 0

    def reset(self, batch_size, device, dtype):
        shape = (batch_size,)
        if self.heads is not None:
            shape += (self.heads,)
        shape += (self.capacity,) + self.trailing
        self.buf = torch.zeros(*shape, device=device, dtype=dtype)
        self.n_valid = 0
        self._p = 0

    def state_bytes(self) -> int:
        return 0 if self.buf is None else self.buf.numel() * BYTES_FP32

    def append(self, value: torch.Tensor) -> torch.Tensor:
        """
        Write one step's value and return the valid region.

        ``value`` is (B, *trailing) or (B, heads, *trailing).  The returned view
        has the cached axis in position -2 of the trailing block, which is what
        the attention matmuls below expect.
        """
        if self.heads is None:
            self.buf[:, self._p] = value
        else:
            self.buf[:, :, self._p] = value

        self._p = (self._p + 1) % self.capacity
        self.n_valid = min(self.n_valid + 1, self.capacity)

        if self.n_valid == self.capacity:
            return self.buf
        # Warmup: writes so far occupy slots [0, n_valid) in order.
        return self.buf[:, :self.n_valid] if self.heads is None \
            else self.buf[:, :, :self.n_valid]


class MHAStepper:
    """
    ``CausalMHA``, advanced one step, caching K and V.

    Per cached step this holds ``2 * d_model`` floats: the full keys and the
    full values, which is what standard attention has to keep.  This is the
    number MLA is claimed to beat, so it is measured the same way
    (``cache_bytes``) rather than argued about.

    Work per step is O(window * d_model) for the scores and the weighted sum,
    against O(window^2 * d_model) for recomputing the whole window.
    """
    kind = "mha"

    def __init__(self, attn: CausalMHA):
        if not attn.causal:
            raise ValueError("CausalMHA was built with causal=False; it attends to the future.")
        if attn.turboquant is not None:
            raise ValueError(
                "This attention block has TurboQuant attached. TurboQuant fake-quantises K and V "
                "inside the batch forward pass; streaming them would quantise a growing cache "
                "each step, which is a different computation. Export the full-precision model."
            )
        self.attn = attn
        self.n_heads = attn.n_heads
        self.head_dim = attn.head_dim
        self.d_model = attn.d_model
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.window = getattr(attn, "attention_window", None) or attn.causal_mask.shape[-1]

        self.k_cache = _RingCache(self.window, trailing=(self.head_dim,), heads=self.n_heads)
        self.v_cache = _RingCache(self.window, trailing=(self.head_dim,), heads=self.n_heads)

    def reset(self, batch_size, device, dtype):
        self.k_cache.reset(batch_size, device, dtype)
        self.v_cache.reset(batch_size, device, dtype)

    def cache_bytes(self) -> int:
        return self.k_cache.state_bytes() + self.v_cache.state_bytes()

    def floats_per_step(self) -> int:
        """Cached floats each new sample adds: K and V in full."""
        return 2 * self.d_model

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        """``x_t``: (B, d_model) -> (B, d_model)."""
        B = x_t.shape[0]
        q, k, v = F.linear(x_t, self.attn.qkv_proj.weight,
                           self.attn.qkv_proj.bias).split(self.d_model, dim=-1)

        q = q.view(B, self.n_heads, 1, self.head_dim)
        k = k.view(B, self.n_heads, self.head_dim)
        v = v.view(B, self.n_heads, self.head_dim)

        k_all = self.k_cache.append(k)                 # (B, H, n, head_dim)
        v_all = self.v_cache.append(v)

        att = (q @ k_all.transpose(-2, -1)) * self.scale    # (B, H, 1, n)
        att = F.softmax(att, dim=-1)
        y = (att @ v_all).reshape(B, self.d_model)          # (B, d_model)
        return F.linear(y, self.attn.out_proj.weight, self.attn.out_proj.bias)


class MLAStepper:
    """
    ``CausalMLA``, advanced one step, caching only the compressed latent.

    This is the module the architecture was chosen for.  Per cached step it
    holds ``kv_lora_rank`` floats — the post-LayerNorm latent ``c_kv`` — rather
    than the ``2 * d_model`` floats MHA must keep.  At the project's defaults
    (``d_model`` 32 after the channel squeeze, rank 16) that is 16 floats
    against 64: **4x less cache per step**, which is where the "4x KV cache
    compression" in the docstrings comes from and what this path finally
    measures instead of estimating.

    Two ways to spend the compression
    ---------------------------------
    ``absorb=False`` decompresses the cache back into K and V every step
    (``k_up_proj`` / ``v_up_proj`` over the whole window).  The memory is
    small, but the arithmetic is MHA's plus the up-projections.

    ``absorb=True`` (the default) never materialises K or V at all, using
    DeepSeek-V2's projection absorption.  For head ``h``, write the key
    up-projection's rows for that head as ``W_k[h]`` (head_dim x rank):

        q_h . k_j,h = q_h^T (W_k[h] c_j + b_k[h])
                    = (W_k[h]^T q_h) . c_j  +  (q_h . b_k[h])

    The first term is a rank-dimensional query against the cached latent; the
    second does not depend on ``j``, so it is a constant added to every score
    in that row and **softmax removes it exactly**.  The key up-projection's
    bias therefore never has to be applied at all.  On the value side, because
    the attention weights sum to one:

        sum_j a_j v_j,h = W_v[h] (sum_j a_j c_j) + b_v[h]

    so the weighted sum is taken in the rank-dimensional latent space and
    up-projected once, for the single output step, instead of once per cached
    position.  Both rewrites are exact, not approximations;
    ``tests/test_incremental.py`` checks the two modes against each other and
    against the batch forward pass.

    The result is that MLA's low rank cuts the per-step *work* as well as the
    cache: the scores and the weighted sum run over ``rank`` instead of
    ``head_dim``, and the up-projections leave the per-step path entirely.
    """
    kind = "mla"

    def __init__(self, attn: CausalMLA, absorb: bool = True):
        if not attn.causal:
            raise ValueError("CausalMLA was built with causal=False; it attends to the future.")
        if attn.turboquant is not None:
            raise ValueError(
                "This attention block has TurboQuant attached. TurboQuant fake-quantises K and V "
                "inside the batch forward pass; streaming them would quantise a growing cache "
                "each step, which is a different computation. Export the full-precision model."
            )
        self.attn = attn
        self.n_heads = attn.n_heads
        self.head_dim = attn.head_dim
        self.d_model = attn.d_model
        self.rank = attn.kv_lora_rank
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.window = getattr(attn, "attention_window", None) or attn.causal_mask.shape[-1]
        self.absorb = absorb

        #: The cache: one rank-dimensional latent per step, no heads axis.
        self.c_cache = _RingCache(self.window, trailing=(self.rank,))

        if absorb:
            # Per-head slices of the up-projections, folded once at construction.
            # k_absorb[h] = W_k[h]^T  (rank x head_dim), applied to q_h.
            w_k = attn.k_up_proj.weight.view(self.n_heads, self.head_dim, self.rank)
            self._k_absorb = w_k.transpose(1, 2).contiguous()        # (H, rank, head_dim)
            self._w_v = attn.v_up_proj.weight.view(self.n_heads, self.head_dim, self.rank)
            self._b_v = attn.v_up_proj.bias.view(self.n_heads, self.head_dim)

    def reset(self, batch_size, device, dtype):
        self.c_cache.reset(batch_size, device, dtype)

    def cache_bytes(self) -> int:
        return self.c_cache.state_bytes()

    def floats_per_step(self) -> int:
        """Cached floats each new sample adds: the latent alone."""
        return self.rank

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        """``x_t``: (B, d_model) -> (B, d_model)."""
        B = x_t.shape[0]
        q = F.linear(x_t, self.attn.q_proj.weight, self.attn.q_proj.bias)

        c = F.linear(x_t, self.attn.kv_down_proj.weight, self.attn.kv_down_proj.bias)
        c = self.attn.kv_norm(c)                              # (B, rank)
        c_all = self.c_cache.append(c)                        # (B, n, rank)

        if not self.absorb:
            # Decompress the cache into K and V, then plain attention.
            k = F.linear(c_all, self.attn.k_up_proj.weight, self.attn.k_up_proj.bias)
            v = F.linear(c_all, self.attn.v_up_proj.weight, self.attn.v_up_proj.bias)
            n = c_all.shape[1]
            k = k.view(B, n, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, n, self.n_heads, self.head_dim).transpose(1, 2)
            q = q.view(B, self.n_heads, 1, self.head_dim)
            att = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
            y = (att @ v).reshape(B, self.d_model)
            return F.linear(y, self.attn.out_proj.weight, self.attn.out_proj.bias)

        # Absorbed path: K and V are never built.
        q = q.view(B, self.n_heads, self.head_dim)
        # q_latent[b,h,:] = W_k[h]^T q[b,h,:]   -> (B, H, rank)
        q_latent = torch.einsum("bhd,hrd->bhr", q, self._k_absorb)

        # Scores against the shared latent cache; the key bias drops out of softmax.
        att = torch.einsum("bhr,bnr->bhn", q_latent, c_all) * self.scale
        att = F.softmax(att, dim=-1)                          # (B, H, n)

        # Weighted sum in latent space, then one up-projection per head.
        z = torch.einsum("bhn,bnr->bhr", att, c_all)          # (B, H, rank)
        y = torch.einsum("bhr,hdr->bhd", z, self._w_v) + self._b_v
        y = y.reshape(B, self.d_model)
        return F.linear(y, self.attn.out_proj.weight, self.attn.out_proj.bias)


def _attention_stepper(attn: nn.Module, absorb: bool):
    if isinstance(attn, CausalMLA):
        return MLAStepper(attn, absorb=absorb)
    if isinstance(attn, CausalMHA):
        return MHAStepper(attn)
    raise TypeError(f"No streaming form for attention module {type(attn).__name__}")


class TransformerBlockStepper:
    """
    ``CausalTransformerBlock``: attention is cached, everything else is pointwise.

    ``ln_1``, ``ln_2``, the FFN and the optional ``BottleneckSkip`` all map
    sample t to sample t, so the block's only state is its attention cache.
    """

    def __init__(self, block: CausalTransformerBlock, absorb: bool):
        self.block = block
        self.attn = _attention_stepper(block.attn, absorb)
        skip = block.attn_skip
        if not isinstance(skip, (nn.Identity, BottleneckSkip)):
            raise TypeError(f"Unsupported attention skip: {type(skip).__name__}")

    def reset(self, batch_size, device, dtype):
        self.attn.reset(batch_size, device, dtype)

    def cache_bytes(self) -> int:
        return self.attn.cache_bytes()

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        x_t = self.block.attn_skip(x_t) + self.attn.step(self.block.ln_1(x_t))
        return x_t + self.block.ffn(self.block.ln_2(x_t))


class DecoderStepper:
    """
    ``CausalTransformerDecoder``: its blocks in sequence, then the final norm.

    The batch forward pass permutes (B, C, T) -> (B, T, C) on the way in and
    back on the way out.  A single step has no time axis to permute, so the
    stepper works in (B, C) throughout and the boundary disappears.
    """

    def __init__(self, decoder: CausalTransformerDecoder, absorb: bool):
        if not decoder.causal:
            raise ValueError(
                "This decoder was built with causal=False (a bidirectional teacher); it attends "
                "to future samples and cannot be streamed."
            )
        self.decoder = decoder
        self.blocks = [TransformerBlockStepper(b, absorb) for b in decoder.blocks]

    @property
    def window(self) -> int:
        return self.blocks[0].attn.window if self.blocks else 0

    def reset(self, batch_size, device, dtype):
        for b in self.blocks:
            b.reset(batch_size, device, dtype)

    def cache_bytes(self) -> int:
        return sum(b.cache_bytes() for b in self.blocks)

    def floats_per_step(self) -> int:
        return sum(b.attn.floats_per_step() for b in self.blocks)

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x_t = b.step(x_t)
        return self.decoder.ln_f(x_t)


# ============================================================
# The model, one sample at a time
# ============================================================
class IncrementalModel:
    """
    Streaming wrapper around a trained causal variant.

    Detects the topology from the modules the model actually has rather than
    from its class name, so the rank aliases (``tcn_mla_r8`` and friends) and
    the distilled students work without being enumerated:

      * ``backbone`` + ``head``                      -> the TCN baseline
      * ``squeeze`` + ``decoder`` + ``expand``       -> the sandwich
      * ``decoder`` + ``tcn_pt2``, no ``squeeze``    -> the parallel branches

    Bidirectional variants are refused: ``causal=False`` means attention reads
    ahead, ``causal_tcn=False`` means the convolutions do, and neither has a
    streaming form.  The refusal is deliberate — a ``tcn_*_bi`` teacher that
    silently streamed would produce plausible numbers from an impossible model.

    Parameters
    ----------
    model : nn.Module
        A built variant with weights loaded.  Forced to ``eval()``: dropout
        must be identity and BatchNorm must use its running statistics for a
        single sample to mean anything.
    absorb : bool
        For MLA, use DeepSeek-V2 projection absorption so K and V are never
        materialised (default).  ``False`` decompresses the latent cache each
        step instead; the two agree to floating-point tolerance and differ
        only in cost.

    Notes
    -----
    The stream works in the model's own input space.  The per-epoch
    normalisation that ``data/Dataloader.py`` applies (divide by the standard
    deviation of the noisy input) is the caller's business, exactly as it is
    for the windowed path in ``deploy/raspi_infer.py`` — a live stream has no
    future samples to take a standard deviation over, so it has to carry a
    running or a calibrated sigma.
    """

    def __init__(self, model: nn.Module, absorb: bool = True):
        if getattr(model, "causal", True) is False:
            raise ValueError(
                "This model has causal=False: its attention attends to future samples. "
                "Bidirectional teachers cannot be streamed — stream the distilled causal "
                "student instead."
            )
        if getattr(model, "causal_tcn", True) is False:
            raise ValueError(
                "This model has causal_tcn=False: its convolutions centre their padding and "
                "read ahead of the current sample. It cannot be streamed."
            )

        self.model = model.eval()
        self.absorb = absorb
        self._device = next(model.parameters()).device
        self._dtype = next(model.parameters()).dtype

        has = lambda name: isinstance(getattr(model, name, None), nn.Module)

        if has("backbone"):
            self.topology = "tcn"
            self.tcn_pt1 = TCNStepper(model.backbone)
            self.decoder = None
            self.tcn_pt2 = None
            self.squeeze = self.expand = None
            self.head = SequentialStepper(model.head)
        elif has("decoder") and has("squeeze") and has("expand"):
            self.topology = "sandwich"
            self.tcn_pt1 = TCNStepper(model.tcn_pt1)
            self.squeeze = ConvStepper(model.squeeze)
            self.decoder = DecoderStepper(model.decoder, absorb)
            self.expand = ConvStepper(model.expand)
            self.tcn_pt2 = TCNStepper(model.tcn_pt2)
            self.head = SequentialStepper(model.head)
        elif has("decoder") and has("tcn_pt2"):
            self.topology = "parallel"
            self.tcn_pt1 = TCNStepper(model.tcn_pt1)
            self.squeeze = self.expand = None
            self.decoder = DecoderStepper(model.decoder, absorb)
            self.tcn_pt2 = TCNStepper(model.tcn_pt2)
            self.head = SequentialStepper(model.head)
        else:
            raise TypeError(
                f"{type(model).__name__} has no recognised streaming topology. Expected a "
                f"'backbone' (TCN baseline), a 'squeeze'/'decoder'/'expand' sandwich, or a "
                f"'decoder' + 'tcn_pt2' parallel pair."
            )

        self._parts = [p for p in (self.tcn_pt1, self.squeeze, self.decoder,
                                   self.expand, self.tcn_pt2, self.head) if p is not None]
        self.n_seen = 0
        self.reset()

    # -- properties -------------------------------------------
    @property
    def attention_window(self) -> int:
        """Cached attention positions, i.e. the window the model was trained with."""
        return 0 if self.decoder is None else self.decoder.window

    @property
    def receptive_field(self) -> int:
        """
        Samples of convolutional history the output depends on.

        The parallel topology runs ``tcn_pt2`` on ``tcn_pt1``'s output rather
        than after it, so the two stages overlap instead of adding up.
        """
        rf1 = self.tcn_pt1.receptive_field
        if self.tcn_pt2 is None:
            return rf1
        rf2 = self.tcn_pt2.receptive_field
        return max(rf1, rf2) if self.topology == "parallel" else rf1 + rf2 - 1

    @property
    def warmup(self) -> int:
        """
        Steps before an output has the full context the batch path would give it.

        Outputs before this still match a windowed forward pass over the
        samples seen so far — they are not wrong, they are just working from a
        shorter history, exactly as the first positions of any window do.
        """
        from experiment.methodology import context
        try:
            past, _ = context(self.model)
            return past + 1
        except ValueError:  # legacy cumulative SE / whole-epoch attention
            return max(self.receptive_field, self.attention_window)

    # -- state ------------------------------------------------
    def reset(self, batch_size: int = 1):
        """Clear every buffer. Two streams from a reset see identical state."""
        for p in self._parts:
            p.reset(batch_size, self._device, self._dtype)
        self.n_seen = 0
        self._batch_size = batch_size

    def cache_report(self) -> dict:
        """
        Where the streaming state lives, in bytes, at the current batch size.

        ``attn_floats_per_step`` is the architectural number the Letter wants:
        how many floats one more sample of context costs, summed over the
        decoder's layers.  It is ``2 * d_model`` per layer for MHA and
        ``kv_lora_rank`` per layer for MLA, and it is read off the modules
        rather than recomputed from the config.
        """
        conv_bytes = sum(p.state_bytes() for p in self._parts
                         if hasattr(p, "state_bytes"))
        attn_bytes = 0 if self.decoder is None else self.decoder.cache_bytes()
        return {
            "topology": self.topology,
            "attention": None if self.decoder is None else self.decoder.blocks[0].attn.kind,
            "absorb": self.absorb if self.decoder is not None else None,
            "batch_size": self._batch_size,
            "conv_state_bytes": conv_bytes,
            "attn_cache_bytes": attn_bytes,
            "total_state_bytes": conv_bytes + attn_bytes,
            "attn_floats_per_step": 0 if self.decoder is None else self.decoder.floats_per_step(),
            "attention_window": self.attention_window,
            "receptive_field": self.receptive_field,
            "warmup": self.warmup,
        }

    # -- stepping ---------------------------------------------
    @torch.no_grad()
    def step(self, x_t) -> torch.Tensor:
        """
        Advance by one sample.

        ``x_t`` is a scalar, or a tensor broadcastable to (B, 1) — one new
        sample per stream.  Returns (B,): the denoised sample at this instant.
        """
        x = torch.as_tensor(x_t, device=self._device, dtype=self._dtype).reshape(-1, 1)
        if x.shape[0] != self._batch_size:
            raise ValueError(
                f"step() got {x.shape[0]} streams but the state was reset for "
                f"{self._batch_size}; call reset(batch_size) first."
            )

        h = self.tcn_pt1.step(x)                       # (B, 1) -> (B, H)

        if self.topology == "tcn":
            out = self.head.step(h)
        elif self.topology == "sandwich":
            a = self.decoder.step(self.squeeze.step(h))
            out = self.head.step(self.tcn_pt2.step(self.expand.step(a)))
        else:                                          # parallel
            branch_a = self.decoder.step(h)
            branch_b = self.tcn_pt2.step(h)
            out = self.head.step(torch.cat([branch_a, branch_b], dim=1))

        self.n_seen += 1
        return out.squeeze(1)

    @torch.no_grad()
    def run(self, signal) -> torch.Tensor:
        """
        Stream a whole signal from a fresh state.

        ``signal`` is (T,) for one stream or (B, T) for several in parallel.
        Returns the same shape.  Equivalent to ``reset()`` then ``step()`` per
        sample; provided because that is what every caller writes.
        """
        x = torch.as_tensor(signal, device=self._device, dtype=self._dtype)
        squeeze_out = x.dim() == 1
        if squeeze_out:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValueError(f"run() expects (T,) or (B, T); got shape {tuple(x.shape)}")

        self.reset(batch_size=x.shape[0])
        out = torch.empty_like(x)
        for t in range(x.shape[1]):
            out[:, t] = self.step(x[:, t])
        return out.squeeze(0) if squeeze_out else out


__all__ = [
    "BYTES_FP32",
    "SEStepper",
    "ConvStepper",
    "DecoderStepper",
    "IncrementalModel",
    "MHAStepper",
    "MLAStepper",
    "ResidualBlockStepper",
    "SequentialStepper",
    "TCNStepper",
    "TransformerBlockStepper",
]
