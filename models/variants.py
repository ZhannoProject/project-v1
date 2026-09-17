"""
Model variants for the EEG causal denoising ablation study.

Five groups of variants, one file:
  - TCNOnly     : pure causal TCN (baseline)
  - TCNWithMHA  : TCN_pt1 + Causal MHA + TCN_pt2 (sandwich)
  - TCNWithMLA  : TCN_pt1 + Causal MLA + TCN_pt2 (sandwich)
  - tcn_mla_rN  : MLA with explicit kv_lora_rank=N (rank sweep ablation)
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

  TCNOnly has no attention, hence no bidirectional twin.

============================================================================
"""

from functools import partial

import torch
import torch.nn as nn
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from models.TCN_backbone import CausalTCNBackbone, CausalConv1d
from models.AttentionBlocks.attention import CausalTransformerDecoder, init_transformer_weights

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
    """

    def __init__(self, hidden_channels: int = 64, dropout: float = 0.1):
        super().__init__()

        self.backbone = CausalTCNBackbone(
            in_channels=1,
            hidden_channels=hidden_channels,
            dilations=[1, 2, 4, 8, 16],
            dropout=dropout,
        )

        # 1×1 conv maps hidden_channels back to 1 channel (the denoised signal)
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

    Topological collapse prevention
    ────────────────────────────────
    Without intervention, the attention decoder can be bypassed entirely:
    inside each transformer block the identity skip ``x + Attn(LN(x))``
    lets the optimizer keep x unchanged and drive Attn → 0, so the whole
    sandwich collapses into a pure TCN.

    Setting ``bottleneck_ratio`` (e.g. 0.5) replaces the identity skip
    with a low-rank bottleneck that intentionally discards high-frequency
    detail, forcing the attention branch to become the primary source of
    fine-grained temporal information.

    The (B,C,T) ↔ (B,T,C) permutation for attention is handled
    INSIDE CausalTransformerDecoder — this forward() stays in (B,C,T).

    ``causal=False`` builds the bidirectional teacher (registry: ``tcn_mha_bi``).
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        n_heads: int = 4,
        n_attn_layers: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        bottleneck_ratio: float = 0.5,
        causal: bool = True,
    ):
        super().__init__()
        self.causal = causal

        # Attention operates in a compressed channel space:
        #   H channels  →  squeeze to H' = H * ratio  →  attention  →  expand back to H
        attn_dim = int(hidden_channels * bottleneck_ratio)

        # --- Stage 1: local feature extraction ---
        self.tcn_pt1 = CausalTCNBackbone(
            in_channels=1,
            hidden_channels=hidden_channels,
            dilations=[1, 2, 4],
            dropout=dropout,
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

    ``causal=False`` builds the bidirectional teacher (registry: ``tcn_mla_bi``).
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
        causal: bool = True,
    ):
        super().__init__()
        self.causal = causal

        attn_dim = int(hidden_channels * bottleneck_ratio)

        # --- Stage 1: local feature extraction (same as MHA variant) ---
        self.tcn_pt1 = CausalTCNBackbone(
            in_channels=1,
            hidden_channels=hidden_channels,
            dilations=[1, 2, 4],
            dropout=dropout,
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

    ``causal=False`` builds the bidirectional teacher (registry: ``tcn_mha_par_bi``).
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        n_heads: int = 4,
        n_attn_layers: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        causal: bool = True,
    ):
        super().__init__()
        self.causal = causal

        # --- Stage 1: shared feature extraction ---
        self.tcn_pt1 = CausalTCNBackbone(
            in_channels=1,
            hidden_channels=hidden_channels,
            dilations=[1, 2, 4],
            dropout=dropout,
        )

        # --- Branch A: global context via MHA ---
        self.decoder = CausalTransformerDecoder(
            d_model=hidden_channels,
            n_heads=n_heads,
            n_layers=n_attn_layers,
            max_seq_len=max_seq_len,
            dropout=dropout,
            attention_type="mha",
            causal=causal,
        )

        # --- Branch B: local detail via TCN ---
        self.tcn_pt2 = CausalTCNBackbone(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            dilations=[8, 16, 32],
            dropout=dropout,
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

    ``causal=False`` builds the bidirectional teacher (registry: ``tcn_mla_par_bi``).
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        n_heads: int = 4,
        n_attn_layers: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        kv_lora_rank: int | None = None,
        causal: bool = True,
    ):
        super().__init__()
        self.causal = causal

        # --- Stage 1: shared feature extraction ---
        self.tcn_pt1 = CausalTCNBackbone(
            in_channels=1,
            hidden_channels=hidden_channels,
            dilations=[1, 2, 4],
            dropout=dropout,
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
            causal=causal,
        )

        # --- Branch B: local detail via TCN ---
        self.tcn_pt2 = CausalTCNBackbone(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            dilations=[8, 16, 32],
            dropout=dropout,
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

# MLA rank sweep: tcn_mla_r4, tcn_mla_r8, tcn_mla_r16, tcn_mla_r32
# Each fixes kv_lora_rank so the only variable across the sweep is the
# compression ratio of the KV bottleneck.
MLA_RANK_SWEEP = [4, 8, 16, 32]
for _rank in MLA_RANK_SWEEP:
    VARIANT_REGISTRY[f"tcn_mla_r{_rank}"] = partial(TCNWithMLA, kv_lora_rank=_rank)

# Bidirectional twins: tcn_mha_bi, tcn_mla_bi, tcn_mha_par_bi, tcn_mla_par_bi
# Same weights and shapes as the causal model, attention mask removed.
# Non-deployable; they are the teachers for knowledge distillation.
BIDIRECTIONAL_SUFFIX = "_bi"
BIDIRECTIONAL_VARIANTS = []
for _base in ["tcn_mha", "tcn_mla", "tcn_mha_par", "tcn_mla_par"]:
    _name = _base + BIDIRECTIONAL_SUFFIX
    VARIANT_REGISTRY[_name] = partial(VARIANT_REGISTRY[_base], causal=False)
    BIDIRECTIONAL_VARIANTS.append(_name)


def variant_is_causal(variant: str) -> bool:
    """True for every deployable (past-only) variant, False for the ``_bi`` teachers."""
    return not variant.endswith(BIDIRECTIONAL_SUFFIX)


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

    The Trainer and eval scripts only need this function.
    They never import TCNOnly / TCNWithMHA / TCNWithMLA directly.
    """
    if variant not in VARIANT_REGISTRY:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Available: {list(VARIANT_REGISTRY.keys())}"
        )
    # Causality is part of the variant name so checkpoints are self-describing;
    # refuse a `causal` kwarg that contradicts it rather than silently obeying it.
    if "causal" in kwargs and kwargs["causal"] != variant_is_causal(variant):
        want = variant + BIDIRECTIONAL_SUFFIX if kwargs["causal"] is False \
            else variant[: -len(BIDIRECTIONAL_SUFFIX)]
        raise ValueError(
            f"causal={kwargs['causal']} contradicts variant '{variant}'; use '{want}' instead"
        )
    from models.AttentionBlocks.turboquant import attach_turboquant
    model = VARIANT_REGISTRY[variant](**kwargs)
    attach_turboquant(model, turboquant, turboquant_bits, turboquant_key_mode,
                     turboquant_train_mode, turboquant_seed)
    model.model_config = dict(variant=variant, **kwargs, turboquant=turboquant,
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


def build_model_from_config(config):
    """Forward supported architecture and quantization options from a run config."""
    import inspect
    variant = config.get("variant", "tcn_mla")
    signature = inspect.signature(VARIANT_REGISTRY[variant])
    kwargs = {key: value for key, value in config.items()
              if key in signature.parameters and value is not None}
    for key in ("turboquant", "turboquant_bits", "turboquant_key_mode",
                "turboquant_train_mode", "turboquant_seed"):
        if key in config:
            kwargs[key] = config[key]
    return build_model(variant, **kwargs)
