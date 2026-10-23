"""
Model variants for the EEG causal denoising ablation study.

Five groups of variants, one file:
  - TCNOnly     : pure causal TCN (baseline)
  - TCNWithMHA  : TCN_pt1 + Causal MHA + TCN_pt2 (sandwich)
  - TCNWithMLA  : TCN_pt1 + Causal MLA + TCN_pt2 (sandwich)
  - tcn_mla_rN  : MLA with explicit kv_lora_rank=N (rank sweep ablation)
  - tcn_bi      : the baseline with non-causal convolutions (offline control)
  - tcn_*_full_bi : bidirectional attention AND non-causal convolutions
  - tcn_*_bi    : the same models with BIDIRECTIONAL attention — non-causal
                  teachers for knowledge distillation (see below)

All variants share the same building blocks (CausalTCNBackbone, CausalConv1d)
and produce the same tensor shape: (B, 1, T) → (B, 1, T).

============================================================================
Assembly philosophy
============================================================================

Building a deep learning model is like assembling a pipeline of LEGO blocks.
Each block has a contract: "give me shape X, I'll give you shape Y."

The assembly flow for the sandwich architecture:

  Step 1 — Define the interfaces
  ────────────────────────────────
  Every module declares its input/output shapes in the docstring.
  Before writing a single line of forward(), draw the shape chain:

      (B, 1, T)  ──►  TCN_pt1  ──►  (B, H, T)
      (B, H, T)  ──►  Decoder  ──►  (B, H, T)     ← attention lives here
      (B, H, T)  ──►  TCN_pt2  ──►  (B, H, T)
      (B, H, T)  ──►  Head     ──►  (B, 1, T)

  If every arrow's output matches the next arrow's input, the model works.
  If any shapes mismatch, you'll get a RuntimeError immediately.

  Step 2 — Pick the components
  ────────────────────────────────
  Each box in the chain is a class from tcn.py or attention.py:

      TCN_pt1 = CausalTCNBackbone(in=1, hidden=H, dilations=[1,2,4])
      Decoder = CausalTransformerDecoder(d_model=H, attention_type="mha")
      TCN_pt2 = CausalTCNBackbone(in=H, hidden=H, dilations=[8,16,32])
      Head    = CausalConv1d(in=H, out=1, kernel_size=1)

  Note how H (hidden_channels / d_model) is the glue — it must be
  the same number everywhere in the middle of the chain.

  Step 3 — Wire them in forward()
  ────────────────────────────────
  forward() is just calling the components in order:

      def forward(self, x):         # x: (B, 1, T)
          x = self.tcn_pt1(x)       # → (B, H, T)
          x = self.decoder(x)       # → (B, H, T)   ← permute is inside
          x = self.tcn_pt2(x)       # → (B, H, T)
          x = self.head(x)          # → (B, 1, T)
          return x

  That's the entire model. No magic, no hidden state.

  Step 4 — Ablation = swap one component
  ────────────────────────────────
  TCNOnly   : remove the Decoder entirely (or equivalently, set n_layers=0)
  TCNWithMHA: Decoder uses attention_type="mha"
  TCNWithMLA: Decoder uses attention_type="mla"

  The rest (TCN_pt1, TCN_pt2, Head, loss function, training loop) is
  IDENTICAL across all three. This is what makes it a fair comparison.

============================================================================
Dilation assignment between pt1 and pt2
============================================================================

  The total set of dilations [1, 2, 4, 8, 16, 32] is SPLIT, not duplicated:

    TCN_pt1: [1, 2, 4]      — small receptive field, acts as a tokeniser
                               Extracts local patterns (spikes, slopes, bursts)
                               before attention sees them.

    TCN_pt2: [8, 16, 32]    — large receptive field, acts as a reconstructor
                               After attention mixes global context, pt2
                               renders the output back into fine-grained
                               temporal detail.

  Why not [1,2,4] + [1,2,4]?  Because then pt2 only has a small RF and
  can't leverage the global information that attention just provided.

  Why not give all dilations to one side?  Because the split lets each
  side specialise: pt1 for local feature extraction, pt2 for global-aware
  reconstruction. This is the same logic as a U-Net encoder/decoder.

  The baseline TCNOnly uses [1, 2, 4, 8, 16] all in one backbone.
  Adding the [32] in pt2 gives the sandwich models a slightly larger
  total RF, which is intentional — the attention needs room to be useful.

============================================================================
Bidirectional teachers and knowledge distillation
============================================================================

  Every attention variant accepts ``causal``.  ``causal=True`` (default)
  is the deployable model: attention position t sees [0..t].  With
  ``causal=False`` the attention mask is dropped and each position sees
  the whole 512-sample window.  The TCN stages stay causal, but because
  the attention output already carries future information the model as
  a whole is non-causal.

  The registry exposes the bidirectional twins as ``tcn_mha_bi``,
  ``tcn_mla_bi``, ``tcn_mha_par_bi``, ``tcn_mla_par_bi``.  They are
  trained exactly like the causal models and serve as TEACHERS:

      teacher  = tcn_mla_bi   (offline, sees the future)   ─┐ frozen
      student  = tcn_mla      (online, past-only)           ─┘ distilled

  Student and teacher share every tensor shape, so the student can be
  initialised from the teacher, and the post-attention features
  (``decoder`` output, (B, H', T)) can be matched directly.  The training
  side lives in trainer/distillation.py (DistillationLoss, KDTrainer).

  TCNOnly has no attention to unmask, so its twin ``tcn_bi`` drops
  causality in the convolutions instead: every position sees the whole
  window, the TCN stages included.  That makes it the control for the
  question the ``_bi`` teachers cannot answer on their own — how much of
  their advantage is seeing the future at all, rather than attending to
  it.  It is also the teacher a distilled ``tcn`` student needs.

============================================================================
"""

from functools import partial
from typing import Sequence

import inspect
import torch
import torch.nn as nn
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from models.TCN_backbone import CausalTCNBackbone, CausalConv1d
from models.AttentionBlocks.attention import CausalTransformerDecoder, init_transformer_weights
from models.lookahead import LOOKAHEAD_LADDER
from models.receptive_field import DEFAULT_DILATIONS, RF_LADDER, field

#: Default squeeze ratio for the SE variants: C -> C/8 -> C.
SE_REDUCTION = 8


# ============================================================
# Variant 1: Pure TCN (baseline)
# ============================================================
class TCNOnly(nn.Module):
    """
    Baseline: pure causal TCN, no attention.

    Architecture:
        input (B, 1, T)
          → CausalTCNBackbone [1, 2, 4, 8, 16]     → (B, H, T)
          → 1×1 CausalConv head                      → (B, 1, T)

    This is functionally equivalent to the original EEGCausualAllCNN,
    but uses the refactored CausalTCNBackbone with explicit dilations.

    ``causal=False`` centres the padding in every convolution instead of
    putting it all on the left (registry: ``tcn_bi``).  Unlike the
    attention twins — where only attention stops being causal and the TCN
    stages still cannot see ahead — this makes the model non-causal
    throughout: it is the offline counterpart of the baseline, and the
    control that separates "seeing the future helps" from "attention
    helps".  Same weights and shapes as ``tcn``, so it distils into it.
    """

    def __init__(self, hidden_channels: int = 64, dropout: float = 0.1, causal: bool = True,
                 se_reduction: int | None = None, lookahead: int | None = None,
                 dilations: Sequence[int] | None = DEFAULT_DILATIONS):
        super().__init__()
        if causal and lookahead:
            raise ValueError(
                f"lookahead={lookahead} contradicts causal=True: a causal model reads nothing "
                f"ahead of the sample it emits. Pass causal=False with the lookahead, which is "
                f"what the tcn_se_la{{N}} variants pin, or drop the lookahead."
            )
        self.causal = causal
        self.lookahead = lookahead
        # How far back the stack reaches is the whole subject of the receptive
        # field sweep, so it is an argument rather than a constant. The default
        # is what every existing run and checkpoint used.
        # Spelled as the signature default, not resolved from None here, so
        # that resolved_arch() reads the real schedule off the signature and
        # every run records the reach it was trained at. ``None`` still means
        # the default, for a config that says so explicitly.
        self.dilations = list(dilations or DEFAULT_DILATIONS)

        self.backbone = CausalTCNBackbone(
            in_channels=1,
            hidden_channels=hidden_channels,
            dilations=self.dilations,
            dropout=dropout,
            causal=causal,
            se_reduction=se_reduction,
            lookahead=lookahead,
        )

        # 1×1 conv maps hidden_channels back to 1 channel (the denoised signal).
        # Kernel 1 has no temporal extent, so causality does not apply here.
        self.head = nn.Sequential(
            CausalConv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.PReLU(),
            CausalConv1d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x):
        # (B, 1, T) → backbone → (B, H, T) → head → (B, 1, T)
        x = self.backbone(x)
        x = self.head(x)
        return x


class TCNWithSE(TCNOnly):
    """
    The baseline with a causal Squeeze-and-Excitation gate on every block.

    A subclass rather than ``partial(TCNOnly, se_reduction=8)`` because a
    partial's keywords are *pinned*: ``build_model`` refuses to let a caller
    override them, which is what ``tcn_mla_r8`` wants (its name states the
    rank) and the opposite of what this wants. Here 8 is a default and the
    config may say otherwise; ``tcn_se_r{N}`` exists for the sweep, where the
    name does state it and pinning is right again.

    The weights are identical to ``TCNOnly(se_reduction=...)``, so this is a
    name for a configuration, not a different model.
    """

    def __init__(self, hidden_channels: int = 64, dropout: float = 0.1, causal: bool = True,
                 se_reduction: int | None = SE_REDUCTION, lookahead: int | None = None,
                 dilations: Sequence[int] | None = DEFAULT_DILATIONS):
        # Spelled out rather than *args/**kwargs: resolved_arch() reads this
        # signature to record what a run trained, and a catch-all would hide
        # every inherited parameter from it -- the run would record its name
        # and nothing else.
        super().__init__(hidden_channels=hidden_channels, dropout=dropout, causal=causal,
                         se_reduction=se_reduction if se_reduction else SE_REDUCTION,
                         lookahead=lookahead, dilations=dilations)


# ============================================================
# Variant 2: TCN + Causal MHA (sandwich)
# ============================================================
class TCNWithMHA(nn.Module):
    """
    Sandwich architecture: TCN_pt1 → Causal MHA → TCN_pt2 → Head.

    Architecture:
        input (B, 1, T)
          → TCN_pt1  [1, 2, 4]              → (B, H, T)   local features
          → CausalTransformerDecoder (MHA)   → (B, H, T)   global context
          → TCN_pt2  [8, 16, 32]             → (B, H, T)   reconstruction
          → 1×1 CausalConv head              → (B, 1, T)   output

    Topological collapse and the optional BottleneckSkip
    ─────────────────────────────────────────────────────
    Inside each transformer block the identity skip ``x + Attn(LN(x))``
    lets the optimizer keep x unchanged and drive Attn → 0, so the whole
    sandwich can collapse into a pure TCN (observed on EMG, see the paper).

    ``skip_bottleneck_ratio`` (default None = identity skip, what every
    reference checkpoint uses) replaces that skip with a low-rank
    BottleneckSkip that intentionally discards high-frequency detail,
    forcing the attention branch to carry fine-grained temporal
    information.  It is an experiment switch, not a validated improvement.

    ``bottleneck_ratio`` is a different knob: the CHANNEL squeeze between
    the TCN (H channels) and the attention decoder (H' = H × ratio).

    The (B,C,T) ↔ (B,T,C) permutation for attention is handled
    INSIDE CausalTransformerDecoder — this forward() stays in (B,C,T).

    ``causal=False`` unmasks attention and builds the bidirectional teacher
    (registry: ``tcn_mha_bi``); the TCN stages stay causal, so the model looks
    ahead only through attention.  ``causal_tcn=False`` centres the padding in
    those stages as well (registry: ``tcn_mha_full_bi``), which makes nothing in
    the model causal.  Both keep the weights and shapes of the causal twin.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        n_heads: int = 4,
        n_attn_layers: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        bottleneck_ratio: float = 0.5,
        skip_bottleneck_ratio: float | None = None,
        causal: bool = True,
        causal_tcn: bool = True,
    ):
        super().__init__()
        self.causal = causal
        self.causal_tcn = causal_tcn

        # Attention operates in a compressed channel space:
        #   H channels  →  squeeze to H' = H * ratio  →  attention  →  expand back to H
        attn_dim = int(hidden_channels * bottleneck_ratio)

        # --- Stage 1: local feature extraction ---
        self.tcn_pt1 = CausalTCNBackbone(
            in_channels=1,
            hidden_channels=hidden_channels,
            dilations=[1, 2, 4],
            dropout=dropout,
            causal=causal_tcn,
        )

        # --- Channel bottleneck: compress before attention ---
        self.squeeze = CausalConv1d(hidden_channels, attn_dim, kernel_size=1)

        # --- Stage 2: global context mixing via MHA ---
        self.decoder = CausalTransformerDecoder(
            d_model=attn_dim,
            n_heads=n_heads,
            n_layers=n_attn_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
            attention_type="mha",
            skip_bottleneck_ratio=skip_bottleneck_ratio,
            causal=causal,
        )

        # --- Channel bottleneck: expand after attention ---
        self.expand = CausalConv1d(attn_dim, hidden_channels, kernel_size=1)

        # --- Stage 3: global-aware reconstruction ---
        self.tcn_pt2 = CausalTCNBackbone(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            dilations=[8, 16, 32],
            dropout=dropout,
            causal=causal_tcn,
        )

        # --- Output head ---
        self.head = CausalConv1d(hidden_channels, 1, kernel_size=1)

        # Apply minGPT-style init to the transformer decoder
        init_transformer_weights(self.decoder, n_layers=n_attn_layers)

    def forward(self, x):
        x = self.tcn_pt1(x)        # (B, 1, T)  → (B, H, T)
        x = self.squeeze(x)        # (B, H, T)  → (B, H', T)
        x = self.decoder(x)        # (B, H', T) → (B, H', T)
        x = self.expand(x)         # (B, H', T) → (B, H, T)
        x = self.tcn_pt2(x)        # (B, H, T)  → (B, H, T)
        x = self.head(x)           # (B, H, T)  → (B, 1, T)
        return x


# ============================================================
# Variant 3: TCN + Causal MLA (sandwich)
# ============================================================
class TCNWithMLA(nn.Module):
    """
    Sandwich architecture: TCN_pt1 → Causal MLA → TCN_pt2 → Head.

    Identical to TCNWithMHA except attention_type="mla".
    MLA compresses KV through a low-rank bottleneck, which:
      - During training: acts as implicit regularisation
      - During inference: reduces KV cache size (useful for BCI streaming)

    Architecture:
        input (B, 1, T)
          → TCN_pt1  [1, 2, 4]              → (B, H, T)   local features
          → CausalTransformerDecoder (MLA)   → (B, H, T)   global context
          → TCN_pt2  [8, 16, 32]             → (B, H, T)   reconstruction
          → 1×1 CausalConv head              → (B, 1, T)   output

    ``causal=False`` unmasks attention and builds the bidirectional teacher
    (registry: ``tcn_mla_bi``); the TCN stages stay causal, so the model looks
    ahead only through attention.  ``causal_tcn=False`` centres the padding in
    those stages as well (registry: ``tcn_mla_full_bi``), which makes nothing in
    the model causal.  Both keep the weights and shapes of the causal twin.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        n_heads: int = 4,
        n_attn_layers: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        kv_lora_rank: int | None = None,    # default: hidden_channels // 4
        bottleneck_ratio: float = 0.5,
        skip_bottleneck_ratio: float | None = None,
        causal: bool = True,
        causal_tcn: bool = True,
    ):
        super().__init__()
        self.causal = causal
        self.causal_tcn = causal_tcn

        attn_dim = int(hidden_channels * bottleneck_ratio)

        # --- Stage 1: local feature extraction (same as MHA variant) ---
        self.tcn_pt1 = CausalTCNBackbone(
            in_channels=1,
            hidden_channels=hidden_channels,
            dilations=[1, 2, 4],
            dropout=dropout,
            causal=causal_tcn,
        )

        # --- Channel bottleneck: compress before attention ---
        self.squeeze = CausalConv1d(hidden_channels, attn_dim, kernel_size=1)

        # --- Stage 2: global context mixing via MLA ---
        self.decoder = CausalTransformerDecoder(
            d_model=attn_dim,
            n_heads=n_heads,
            n_layers=n_attn_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
            attention_type="mla",
            kv_lora_rank=kv_lora_rank,
            skip_bottleneck_ratio=skip_bottleneck_ratio,
            causal=causal,
        )

        # --- Channel bottleneck: expand after attention ---
        self.expand = CausalConv1d(attn_dim, hidden_channels, kernel_size=1)

        # --- Stage 3: global-aware reconstruction (same as MHA variant) ---
        self.tcn_pt2 = CausalTCNBackbone(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            dilations=[8, 16, 32],
            dropout=dropout,
            causal=causal_tcn,
        )

        # --- Output head ---
        self.head = CausalConv1d(hidden_channels, 1, kernel_size=1)

        # Apply minGPT-style init to the transformer decoder
        init_transformer_weights(self.decoder, n_layers=n_attn_layers)

    def forward(self, x):
        x = self.tcn_pt1(x)        # (B, 1, T)  → (B, H, T)
        x = self.squeeze(x)        # (B, H, T)  → (B, H', T)
        x = self.decoder(x)        # (B, H', T) → (B, H', T)
        x = self.expand(x)         # (B, H', T) → (B, H, T)
        x = self.tcn_pt2(x)        # (B, H, T)  → (B, H, T)
        x = self.head(x)           # (B, H, T)  → (B, 1, T)
        return x


# ============================================================
# Variant 4: TCN + Causal MHA (parallel branches)
# ============================================================
class TCNParallelMHA(nn.Module):
    """
    Parallel architecture: attention and TCN_pt2 run side-by-side,
    then concatenate before the output head.

    Architecture:
        input (B, 1, T)
          → TCN_pt1  [1, 2, 4]                     → x_1 (B, H, T)
          → ┌─ Branch A: CausalTransformerDecoder (MHA) → (B, H, T)  global
            └─ Branch B: TCN_pt2  [8, 16, 32]           → (B, H, T)  local
          → torch.cat(A, B, dim=1)                  → (B, 2H, T)
          → 1×1 CausalConv head                     → (B, 1, T)

    Why parallel instead of sequential?
    ────────────────────────────────────
    In the sandwich (sequential) design, the identity skip inside each
    transformer block lets the optimizer bypass attention entirely — the
    model can collapse into a pure TCN.

    The parallel topology eliminates this risk structurally: the head
    receives a concatenation of BOTH branches, so attention must produce
    useful features or the head loses half its input channels.  There is
    no residual path that lets TCN_pt2 compensate for a dead attention.
    The per-block identity skip inside the decoder still exists;
    ``skip_bottleneck_ratio`` swaps it for a BottleneckSkip here too.

    ``causal=False`` unmasks attention and builds the bidirectional teacher
    (registry: ``tcn_mha_par_bi``); the TCN stages stay causal, so the model looks
    ahead only through attention.  ``causal_tcn=False`` centres the padding in
    those stages as well (registry: ``tcn_mha_par_full_bi``), which makes nothing in
    the model causal.  Both keep the weights and shapes of the causal twin.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        n_heads: int = 4,
        n_attn_layers: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        skip_bottleneck_ratio: float | None = None,
        causal: bool = True,
        causal_tcn: bool = True,
    ):
        super().__init__()
        self.causal = causal
        self.causal_tcn = causal_tcn

        # --- Stage 1: shared feature extraction ---
        self.tcn_pt1 = CausalTCNBackbone(
            in_channels=1,
            hidden_channels=hidden_channels,
            dilations=[1, 2, 4],
            dropout=dropout,
            causal=causal_tcn,
        )

        # --- Branch A: global context via MHA ---
        self.decoder = CausalTransformerDecoder(
            d_model=hidden_channels,
            n_heads=n_heads,
            n_layers=n_attn_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
            attention_type="mha",
            skip_bottleneck_ratio=skip_bottleneck_ratio,
            causal=causal,
        )

        # --- Branch B: local detail via TCN ---
        self.tcn_pt2 = CausalTCNBackbone(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            dilations=[8, 16, 32],
            dropout=dropout,
            causal=causal_tcn,
        )

        # --- Output head: fuses both branches (2H → 1) ---
        self.head = CausalConv1d(hidden_channels * 2, 1, kernel_size=1)

        init_transformer_weights(self.decoder, n_layers=n_attn_layers)

    def forward(self, x):
        x = self.tcn_pt1(x)                    # (B, 1, T)  → (B, H, T)
        branch_a = self.decoder(x)             # (B, H, T)  → (B, H, T)  global
        branch_b = self.tcn_pt2(x)             # (B, H, T)  → (B, H, T)  local
        x = torch.cat([branch_a, branch_b], dim=1)  # → (B, 2H, T)
        x = self.head(x)                       # (B, 2H, T) → (B, 1, T)
        return x


# ============================================================
# Variant 5: TCN + Causal MLA (parallel branches)
# ============================================================
class TCNParallelMLA(nn.Module):
    """
    Parallel architecture: MLA attention and TCN_pt2 run side-by-side.

    Identical to TCNParallelMHA except attention_type="mla".
    The low-rank KV bottleneck inside MLA provides additional compression
    on top of the parallel branch structure.

    Architecture:
        input (B, 1, T)
          → TCN_pt1  [1, 2, 4]                     → x_1 (B, H, T)
          → ┌─ Branch A: CausalTransformerDecoder (MLA) → (B, H, T)  global
            └─ Branch B: TCN_pt2  [8, 16, 32]           → (B, H, T)  local
          → torch.cat(A, B, dim=1)                  → (B, 2H, T)
          → 1×1 CausalConv head                     → (B, 1, T)

    ``causal=False`` unmasks attention and builds the bidirectional teacher
    (registry: ``tcn_mla_par_bi``); the TCN stages stay causal, so the model looks
    ahead only through attention.  ``causal_tcn=False`` centres the padding in
    those stages as well (registry: ``tcn_mla_par_full_bi``), which makes nothing in
    the model causal.  Both keep the weights and shapes of the causal twin.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        n_heads: int = 4,
        n_attn_layers: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        kv_lora_rank: int | None = None,
        skip_bottleneck_ratio: float | None = None,
        causal: bool = True,
        causal_tcn: bool = True,
    ):
        super().__init__()
        self.causal = causal
        self.causal_tcn = causal_tcn

        # --- Stage 1: shared feature extraction ---
        self.tcn_pt1 = CausalTCNBackbone(
            in_channels=1,
            hidden_channels=hidden_channels,
            dilations=[1, 2, 4],
            dropout=dropout,
            causal=causal_tcn,
        )

        # --- Branch A: global context via MLA ---
        self.decoder = CausalTransformerDecoder(
            d_model=hidden_channels,
            n_heads=n_heads,
            n_layers=n_attn_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
            attention_type="mla",
            kv_lora_rank=kv_lora_rank,
            skip_bottleneck_ratio=skip_bottleneck_ratio,
            causal=causal,
        )

        # --- Branch B: local detail via TCN ---
        self.tcn_pt2 = CausalTCNBackbone(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            dilations=[8, 16, 32],
            dropout=dropout,
            causal=causal_tcn,
        )

        # --- Output head: fuses both branches (2H → 1) ---
        self.head = CausalConv1d(hidden_channels * 2, 1, kernel_size=1)

        init_transformer_weights(self.decoder, n_layers=n_attn_layers)

    def forward(self, x):
        x = self.tcn_pt1(x)                    # (B, 1, T)  → (B, H, T)
        branch_a = self.decoder(x)             # (B, H, T)  → (B, H, T)  global
        branch_b = self.tcn_pt2(x)             # (B, H, T)  → (B, H, T)  local
        x = torch.cat([branch_a, branch_b], dim=1)  # → (B, 2H, T)
        x = self.head(x)                       # (B, 2H, T) → (B, 1, T)
        return x


# ============================================================
# Registry: one-line model construction
# ============================================================
VARIANT_REGISTRY = {
    "tcn":          TCNOnly,
    "tcn_mha":      TCNWithMHA,
    "tcn_mla":      TCNWithMLA,
    "tcn_mha_par":  TCNParallelMHA,
    "tcn_mla_par":  TCNParallelMLA,
}

# Squeeze-and-Excitation baseline: the same TCN with a causal SE gate on every
# residual block.  It is in the study as the cheap end of "attention": SE
# reweights channels from a running summary and does no temporal mixing at
# all, for about 5k parameters and no measurable compute, where the MHA/MLA
# decoder costs roughly 5x the baseline's inference time on the Pi.  If it
# closes the same gap, the expensive machinery is not what was buying it.
VARIANT_REGISTRY["tcn_se"] = TCNWithSE

# The sweep, where the name fixes the squeeze ratio the way tcn_mla_r{N} fixes
# the KV rank: tcn_se_r2 .. tcn_se_r16, so a run's name cannot misdescribe it.
SE_REDUCTION_SWEEP = [2, 4, 8, 16]
for _r in SE_REDUCTION_SWEEP:
    VARIANT_REGISTRY[f"tcn_se_r{_r}"] = partial(TCNWithSE, se_reduction=_r)

# MLA rank sweep: tcn_mla_r4, tcn_mla_r8, tcn_mla_r16, tcn_mla_r32
# Each fixes kv_lora_rank so the only variable across the sweep is the
# compression ratio of the KV bottleneck.
MLA_RANK_SWEEP = [4, 8, 16, 32]
for _rank in MLA_RANK_SWEEP:
    VARIANT_REGISTRY[f"tcn_mla_r{_rank}"] = partial(TCNWithMLA, kv_lora_rank=_rank)

# Non-causal twins: tcn_bi, tcn_mha_bi, tcn_mla_bi, tcn_mha_par_bi, tcn_mla_par_bi
# Same weights and shapes as the causal model; what is dropped differs by
# family.  For the attention variants it is the causal mask, and the TCN
# stages stay causal.  For tcn_bi there is no attention to unmask, so the
# convolutions themselves become non-causal and the model sees ahead
# everywhere.  All of them are non-deployable teachers for distillation.
BIDIRECTIONAL_SUFFIX = "_bi"
BIDIRECTIONAL_VARIANTS = []
for _base in ["tcn", "tcn_se", "tcn_mha", "tcn_mla", "tcn_mha_par", "tcn_mla_par"]:
    _name = _base + BIDIRECTIONAL_SUFFIX
    VARIANT_REGISTRY[_name] = partial(VARIANT_REGISTRY[_base], causal=False)
    BIDIRECTIONAL_VARIANTS.append(_name)

# Fully non-causal twins: tcn_mha_full_bi, tcn_mla_full_bi, tcn_mha_par_full_bi,
# tcn_mla_par_full_bi.  ``X_bi`` unmasks attention and leaves the convolutions
# causal; ``X_full_bi`` also centres their padding, so nothing in the model is
# causal any more.  Between them they separate "sees the future" from "attends
# to it": X_bi is the first without the second, X_full_bi is both, and tcn_bi is
# the first without any attention at all.
#
# tcn has no attention, so tcn_bi is already fully non-causal and there is no
# tcn_full_bi — the two axes collapse into one when there is no mask to drop.
# Bounded lookahead: tcn_se_la1 .. tcn_se_la186.  Between tcn_se (reads
# nothing ahead) and tcn_se_bi (reads the whole epoch) there is a continuum
# nothing in this project could express, and it is the one an online system
# actually sits on: L samples of lookahead is L/fs of added output latency,
# 3.9 ms a sample at 256 Hz, and the question is what each millisecond buys.
#
# The name pins the budget for the same reason tcn_mla_r{N} pins the rank --
# a run called tcn_se_la32 cannot turn out to hold anything else -- and it
# pins causal=False with it, because a model that reads ahead is not causal
# and every tool here decides what it may do from that flag: the ONNX
# exporter refuses it, and the streaming paths in incremental.py and
# step_graph.py refuse it, all correctly.  Deploying one needs an output
# delay of L samples, which no path here implements yet.
#
# See models/lookahead.py for how a budget is spread over the stack.
LOOKAHEAD_SUFFIX = "_la"
LOOKAHEAD_VARIANTS = []
for _la in LOOKAHEAD_LADDER:
    _name = f"tcn_se{LOOKAHEAD_SUFFIX}{_la}"
    VARIANT_REGISTRY[_name] = partial(TCNWithSE, causal=False, lookahead=_la)
    LOOKAHEAD_VARIANTS.append(_name)

# Receptive-field sweep: tcn_rf5 .. tcn_rf509, and the SE family alongside.
# The name states the reach in samples, so a run cannot be mislabelled about
# the one axis the sweep is measuring -- the same reason tcn_mla_r{N} states
# its rank. Registered for the plain baseline first: it isolates reach from
# the channel gate, which is the comparison the sweep exists to make.
RF_SUFFIX = "_rf"
RECEPTIVE_FIELD_VARIANTS = []
for _base, _cls in (("tcn", TCNOnly), ("tcn_se", TCNWithSE)):
    for _dil in RF_LADDER:
        _name = f"{_base}{RF_SUFFIX}{field(_dil)}"
        VARIANT_REGISTRY[_name] = partial(_cls, dilations=list(_dil))
        RECEPTIVE_FIELD_VARIANTS.append(_name)

FULLY_BIDIRECTIONAL_SUFFIX = "_full" + BIDIRECTIONAL_SUFFIX
FULLY_BIDIRECTIONAL_VARIANTS = []
for _base in ["tcn_mha", "tcn_mla", "tcn_mha_par", "tcn_mla_par"]:
    _name = _base + FULLY_BIDIRECTIONAL_SUFFIX
    VARIANT_REGISTRY[_name] = partial(VARIANT_REGISTRY[_base], causal=False, causal_tcn=False)
    BIDIRECTIONAL_VARIANTS.append(_name)
    FULLY_BIDIRECTIONAL_VARIANTS.append(_name)


def variant_is_causal(variant: str) -> bool:
    """
    True for every deployable (past-only) variant, False for the ``_bi``
    teachers and the bounded-lookahead ``_la{N}`` rungs.

    Read off what the name pins rather than from a list of names, so a new
    family of non-causal variants cannot be added without this answering for
    it. A lookahead variant pins ``causal=False`` precisely so that it lands
    here: it reads fewer future samples than a ``_bi`` teacher, but "fewer"
    is not "none", and everything downstream needs the same answer it gives
    for a teacher.
    """
    if variant.endswith(BIDIRECTIONAL_SUFFIX):
        return False
    if variant not in VARIANT_REGISTRY:
        # A legacy checkpoint records no variant at all, and
        # load_checkpoint_model asks this of the empty string.  The suffix test
        # this replaced answered True there, and the answer has to stay True:
        # those checkpoints are the project's own causal runs, and reading them
        # as non-causal would have the exporter refuse every one of them.
        return True
    return bool(pinned_kwargs(variant).get("causal", True))


def pinned_kwargs(variant: str) -> dict:
    """
    Constructor arguments that the variant NAME itself fixes.

    ``tcn_mla_r8`` pins ``kv_lora_rank=8``; every ``*_bi`` twin pins
    ``causal=False``; the five base variants pin nothing.  These are the
    registry's ``functools.partial`` keywords, and they are authoritative:
    a caller must not be able to override them, or a checkpoint saved as
    ``tcn_mla_r8`` could silently hold a rank-16 model.
    """
    entry = VARIANT_REGISTRY[variant]
    return dict(entry.keywords) if isinstance(entry, partial) else {}


def base_variant(variant: str) -> str:
    """
    The architecture behind a variant name: ``tcn_mla_r8`` and ``tcn_mla_bi``
    are both ``tcn_mla``; a base name maps to itself.  Aliases share the
    base's class and state-dict layout, so anything that identifies a model
    from its weights (deploy/export_onnx.py) can only ever recover the base.
    """
    if variant not in VARIANT_REGISTRY:
        raise ValueError(f"Unknown variant '{variant}'. Available: {list(VARIANT_REGISTRY.keys())}")

    # Strip the non-causal suffix first, longest first so ``_full_bi`` is not
    # read as ``_bi``.  This has to come before the class lookup because
    # functools.partial *flattens*: ``tcn_se_bi`` is one partial over TCNOnly,
    # indistinguishable by class from ``tcn``, and the lookup below would hand
    # back the wrong twin -- a teacher paired with a student of a different
    # shape, which only shows up as a shape error much later.
    for suffix in (FULLY_BIDIRECTIONAL_SUFFIX, BIDIRECTIONAL_SUFFIX):
        if variant.endswith(suffix):
            stripped = variant[: -len(suffix)]
            if stripped in VARIANT_REGISTRY:
                return stripped

    entry = VARIANT_REGISTRY[variant]
    cls = entry.func if isinstance(entry, partial) else entry   # nested partials flatten
    return next(name for name, e in VARIANT_REGISTRY.items() if e is cls)


def build_model(variant: str, *, turboquant=False, turboquant_bits=3,
                turboquant_key_mode="prod", turboquant_train_mode="off",
                turboquant_seed=42, **kwargs) -> nn.Module:
    """
    Factory function for model construction.

    Usage:
        model = build_model("tcn",         hidden_channels=64)
        model = build_model("tcn_mha",     hidden_channels=64, n_heads=4)
        model = build_model("tcn_mla",     hidden_channels=64, kv_lora_rank=16)
        model = build_model("tcn_mla_r8",  hidden_channels=64)   # rank sweep
        model = build_model("tcn_mla",     skip_bottleneck_ratio=0.5)  # BottleneckSkip on

    The Trainer and eval scripts only need this function.
    They never import TCNOnly / TCNWithMHA / TCNWithMLA directly.

    Anything the variant name pins (``kv_lora_rank`` for ``tcn_mla_r{N}``,
    ``causal`` for ``*_bi``) cannot be overridden: passing a different value
    raises rather than silently building a model that its name misdescribes.
    """
    if variant not in VARIANT_REGISTRY:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Available: {list(VARIANT_REGISTRY.keys())}"
        )
    # Causality is part of the variant name so checkpoints are self-describing;
    # refuse a `causal` kwarg that contradicts it rather than silently obeying it.
    if "causal" in kwargs and kwargs["causal"] != variant_is_causal(variant):
        # Suggest the twin, but only when it is a name the registry actually has:
        # tcn_mla_full_bi's causal counterpart is tcn_mla, not "tcn_mla_full".
        want = base_variant(variant) if kwargs["causal"] else variant + BIDIRECTIONAL_SUFFIX
        hint = f"; use '{want}' instead" if want in VARIANT_REGISTRY else ""
        raise ValueError(
            f"causal={kwargs['causal']} contradicts variant '{variant}'{hint}"
        )
    # Same rule for the rank aliases.  functools.partial lets a call-time kwarg
    # override its keywords, so without this check build_model("tcn_mla_r8",
    # kv_lora_rank=16) would quietly return a rank-16 model.
    pinned = pinned_kwargs(variant)
    for key, value in pinned.items():
        if key in kwargs and kwargs[key] != value:
            raise ValueError(
                f"{key}={kwargs[key]} contradicts variant '{variant}' (which fixes {key}={value}); "
                f"drop {key} or use the generic variant with {key}={kwargs[key]}"
            )
    from models.AttentionBlocks.turboquant import attach_turboquant
    model = VARIANT_REGISTRY[variant](**kwargs)
    attach_turboquant(model, turboquant, turboquant_bits, turboquant_key_mode,
                     turboquant_train_mode, turboquant_seed)
    # Record the pinned values too, so the checkpoint states its rank / causality
    # explicitly instead of leaving eval, export and the GUI to decode the name.
    model.model_config = dict(variant=variant, **{**pinned, **kwargs}, turboquant=turboquant,
                              turboquant_bits=turboquant_bits,
                              turboquant_key_mode=turboquant_key_mode,
                              turboquant_train_mode=turboquant_train_mode,
                              turboquant_seed=turboquant_seed)
    return model


# ============================================================
# Self-test: verify all three variants
# ============================================================
if __name__ == "__main__":
    B, T = 4, 512
    x = torch.randn(B, 1, T)

    print("=" * 60)
    print("Model variant comparison")
    print("=" * 60)

    results = {}

    all_variants = (
        ["tcn", "tcn_mha", "tcn_mla", "tcn_mha_par", "tcn_mla_par"]
        + [f"tcn_mla_r{r}" for r in MLA_RANK_SWEEP]
        + BIDIRECTIONAL_VARIANTS
    )

    for name in all_variants:
        model = build_model(name, hidden_channels=64)

        # Forward pass
        model.eval()
        with torch.no_grad():
            y = model(x)

        # Parameter count
        n_params = sum(p.numel() for p in model.parameters())

        # Shape check
        assert y.shape == x.shape, f"{name}: output shape {y.shape} != input shape {x.shape}"

        results[name] = n_params
        print(f"\n[{name.upper()}]")
        print(f"  Input  : {x.shape}")
        print(f"  Output : {y.shape}")
        print(f"  Params : {n_params:,}")

    # ---- Causality check ----
    for variant_name in ["tcn_mha", "tcn_mha_par"]:
        print(f"\n{'='*60}")
        print(f"Causality check ({variant_name})")
        print("=" * 60)

        model = build_model(variant_name, hidden_channels=64)
        model.eval()

        x1 = torch.randn(B, 1, T)
        x2 = x1.clone()
        x2[:, :, 256] += 10.0          # perturb future time step

        with torch.no_grad():
            y1 = model(x1)
            y2 = model(x2)

        past_diff = (y1[:, :, :256] - y2[:, :, :256]).abs().max().item()
        future_diff = (y1[:, :, 256:] - y2[:, :, 256:]).abs().max().item()

        print(f"  past_diff   = {past_diff:.2e}  (should be ~0)")
        print(f"  future_diff = {future_diff:.2e}  (should be >0)")
        assert past_diff < 1e-5, f"CAUSALITY VIOLATED in {variant_name}!"
        print("  Causality verified.")

    # ---- Bidirectional teachers must NOT be causal ----
    for variant_name in BIDIRECTIONAL_VARIANTS:
        model = build_model(variant_name, hidden_channels=64).eval()
        x1 = torch.randn(B, 1, T)
        x2 = x1.clone()
        x2[:, :, 256] += 10.0
        with torch.no_grad():
            past_diff = (model(x1)[:, :, :256] - model(x2)[:, :, :256]).abs().max().item()
        assert past_diff > 0, f"{variant_name} should see the future"
        print(f"  {variant_name:<16s} past_diff = {past_diff:.2e}  (bidirectional, as intended)")

    # ---- Summary table ----
    print(f"\n{'='*60}")
    print(f"{'Variant':<16} {'Params':>10} {'vs TCN':>10}")
    print(f"{'-'*16} {'-'*10} {'-'*10}")
    base = results["tcn"]
    for name, n in results.items():
        diff = f"+{n - base:,}" if n > base else f"{n - base:,}"
        print(f"{name:<16} {n:>10,} {diff:>10}")


def resolved_arch(model) -> dict:
    """
    Every architecture argument a model was built with, defaults filled in.

    ``model.model_config`` records only what was *passed*, so a run that took
    a default does not record it: ``build_model("tcn")`` stores no
    ``hidden_channels`` at all.  That is enough to rebuild the model today and
    not enough to say what was trained, which is the question asked of a run
    months later — and if a default ever moves, every run that relied on it
    becomes ambiguous in hindsight.

    This resolves the constructor's signature against what was passed, so the
    answer is complete and does not depend on the defaults staying put.  A
    rank alias's pinned value wins, because that is what the name means.

    Used by the trainer to write the architecture into ``history.json``.  Runs
    from before that carry none, which is how the 2-layer MHA / 4-layer MLA
    pair in this project went unnoticed: both configs said 2, the MLA runs
    were trained at 4, and nothing in the run recorded either.
    """
    cfg = dict(getattr(model, "model_config", {}) or {})
    variant = cfg.get("variant")
    if not variant or variant not in VARIANT_REGISTRY:
        return cfg

    out = {"variant": variant}
    try:
        signature = inspect.signature(VARIANT_REGISTRY[variant])
    except (TypeError, ValueError):            # pragma: no cover - defensive
        return cfg
    for name, param in signature.parameters.items():
        if param.default is inspect.Parameter.empty or name == "self":
            continue
        out[name] = cfg.get(name, param.default)

    # What the variant's own name fixes beats anything passed alongside it.
    out.update(pinned_kwargs(variant))
    # Not constructor arguments, but part of what produced the weights.
    for key in ("turboquant", "turboquant_bits", "turboquant_key_mode",
                "turboquant_train_mode", "turboquant_seed"):
        if key in cfg:
            out[key] = cfg[key]
    return out


def build_model_from_config(config):
    """
    Forward supported architecture and quantization options from a run config.

    Run configs are layered (trainer defaults < YAML < CLI < checkpoint
    ``model_config``) and routinely carry a generic ``kv_lora_rank`` meant for
    ``tcn_mla``.  For a rank alias such as ``tcn_mla_r8`` the name wins: the
    config value is dropped (with a note if it differed) instead of being
    forwarded, which is what used to turn every sweep run into rank 16.
    """
    import inspect
    variant = config.get("variant", "tcn_mla")
    signature = inspect.signature(VARIANT_REGISTRY[variant])
    kwargs = {key: value for key, value in config.items()
              if key in signature.parameters and value is not None}
    for key, value in pinned_kwargs(variant).items():
        if key in kwargs and kwargs[key] != value:
            print(f"[build_model] variant '{variant}' fixes {key}={value}; "
                  f"ignoring {key}={kwargs[key]} from the config")
        kwargs.pop(key, None)
    for key in ("turboquant", "turboquant_bits", "turboquant_key_mode",
                "turboquant_train_mode", "turboquant_seed"):
        if key in config:
            kwargs[key] = config[key]
    return build_model(variant, **kwargs)
