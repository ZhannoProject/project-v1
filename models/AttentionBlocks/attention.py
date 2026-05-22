"""
Causal Attention modules for 1D temporal signal denoising.

Adapted from Karpathy's minGPT (https://github.com/karpathy/minGPT).
All language-model specifics removed; re-targeted at continuous 1D signals.

Contains two attention variants for ablation:
  - CausalMHA : standard Multi-Head Attention (from minGPT)
  - CausalMLA : Multi-head Latent Attention (from DeepSeek-V2)
                 KV pairs go through a low-rank bottleneck before attention.

Both share the same CausalTransformerBlock and CausalTransformerDecoder
via dependency injection — the Block doesn't know which attention it holds.

============================================================================
Tensor shape conventions
============================================================================
  TCN world  :  (B, C, T)   — channel-first, matches nn.Conv1d
  Attn world :  (B, T, C)   — channel-last, matches nn.Linear / minGPT

  Permutation happens ONCE inside CausalTransformerDecoder.forward().
  Everything below (CausalMHA, CausalMLA, CausalTransformerBlock)
  operates in (B, T, C).
============================================================================

============================================================================
MHA vs MLA : what differs, what stays the same
============================================================================

  MHA (standard):
      Q = W_q @ x               →  (B, T, d_model)
      K = W_k @ x               →  (B, T, d_model)
      V = W_v @ x               →  (B, T, d_model)

  MLA (low-rank KV):
      Q     = W_q @ x           →  (B, T, d_model)       ← same as MHA
      c_kv  = W_down @ x        →  (B, T, kv_lora_rank)  ← compress
      c_kv  = LayerNorm(c_kv)                             ← stabilise latent
      K     = W_up_k @ c_kv     →  (B, T, d_model)       ← decompress K
      V     = W_up_v @ c_kv     →  (B, T, d_model)       ← decompress V

  After Q, K, V are produced, the attention computation is IDENTICAL:
      att = softmax(causal_mask(Q @ K^T / sqrt(d_k))) @ V

  Ablation isolates exactly one variable: how K and V are produced.
============================================================================
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================
# Causal Multi-Head Attention (from minGPT)
# ============================================================
class CausalMHA(nn.Module):
    """
    Standard multi-head self-attention with a causal mask.

    Adapted from minGPT CausalSelfAttention:
      - config object  →  plain constructor args
      - n_embd         →  d_model
      - block_size     →  max_seq_len
      - Fused QKV projection kept for efficiency

    Input  : (B, T, d_model)
    Output : (B, T, d_model)

    Attention weight capture
    ────────────────────────
    Call ``model.decoder.blocks[i].attn.capture_weights(True)`` before a
    forward pass.  After the pass, ``model.decoder.blocks[i].attn.last_attn_weights``
    holds the (B, n_heads, T, T) softmax attention matrix (detached, on CPU).
    Call ``capture_weights(False)`` to stop capturing and free memory.
    """
    causal_mask : torch.Tensor
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Fused QKV projection: one matmul produces Q, K, V together
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Regularisation
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: position i can only attend to positions [0..i]
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, max_seq_len, max_seq_len))

        # Attention weight capture (disabled by default to save memory)
        self._capture = False
        self.last_attn_weights = None

    def capture_weights(self, enable: bool = True):
        """Enable / disable attention weight capture for visualisation."""
        self._capture = enable
        if not enable:
            self.last_attn_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Fused QKV: (B, T, d_model) → (B, T, 3*d_model) → 3 × (B, T, d_model)
        q, k, v = self.qkv_proj(x).split(self.d_model, dim=2)

        # Reshape: (B, T, d_model) → (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        # Optionally store attention weights for heatmap visualisation
        if self._capture:
            self.last_attn_weights = att.detach().cpu()

        att = self.attn_dropout(att)

        # Weighted sum → concatenate heads → output projection
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.out_proj(y))
        return y


# ============================================================
# Causal Multi-head Latent Attention (from DeepSeek-V2)
# ============================================================
class CausalMLA(nn.Module):
    """
    Multi-head Latent Attention: KV pairs are produced through a
    low-rank bottleneck, reducing the KV cache size during inference.

    Architecture (DeepSeek-V2 style, simplified for EEG):
      Q path  :  x  →  W_q              →  Q        (standard, same as MHA)
      KV path :  x  →  W_down  →  c_kv  →  LayerNorm
                       (compress)           →  W_up_k  →  K     (decompress)
                                            →  W_up_v  →  V     (decompress)

    The low-rank bottleneck (d_model → kv_lora_rank → d_model) means:
      - Training  : similar to MHA, but the information bottleneck acts
                    as implicit regularisation on the KV representations
      - Inference : only c_kv (size kv_lora_rank) needs caching per step,
                    instead of full K and V (size 2 * d_model). For BCI
                    streaming on embedded hardware, this is a real win.

    Default kv_lora_rank = d_model // 4 gives 4x KV cache compression.

    Input  : (B, T, d_model)
    Output : (B, T, d_model)

    Supports the same ``capture_weights()`` API as CausalMHA for
    attention heatmap visualisation.
    """
    causal_mask : torch.Tensor
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        kv_lora_rank: int | None = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Default: 4× compression ratio
        if kv_lora_rank is None:
            kv_lora_rank = d_model // 4
        self.kv_lora_rank = kv_lora_rank

        # ---- Q path (standard, same as MHA) ----
        self.q_proj = nn.Linear(d_model, d_model)

        # ---- KV path (low-rank compression) ----
        # Step 1: compress input to a low-dimensional latent
        self.kv_down_proj = nn.Linear(d_model, kv_lora_rank)
        # Step 2: normalise the latent (stabilises training, from DeepSeek-V2)
        self.kv_norm = nn.LayerNorm(kv_lora_rank)
        # Step 3: decompress latent into K and V separately
        self.k_up_proj = nn.Linear(kv_lora_rank, d_model)
        self.v_up_proj = nn.Linear(kv_lora_rank, d_model)

        # ---- Output projection ----
        self.out_proj = nn.Linear(d_model, d_model)

        # ---- Regularisation ----
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # ---- Causal mask (identical to MHA) ----
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, max_seq_len, max_seq_len))

        # Attention weight capture (disabled by default to save memory)
        self._capture = False
        self.last_attn_weights = None

    def capture_weights(self, enable: bool = True):
        """Enable / disable attention weight capture for visualisation."""
        self._capture = enable
        if not enable:
            self.last_attn_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # ---- Q: standard projection (same as MHA) ----
        q = self.q_proj(x)                          # (B, T, d_model)

        # ---- KV: compress → latent → decompress ----
        c_kv = self.kv_down_proj(x)                 # (B, T, kv_lora_rank)
        c_kv = self.kv_norm(c_kv)                   # stabilise the bottleneck
        k = self.k_up_proj(c_kv)                    # (B, T, d_model)
        v = self.v_up_proj(c_kv)                    # (B, T, d_model)

        # ---- From here on, IDENTICAL to MHA ----

        # Reshape: (B, T, d_model) → (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        # Optionally store attention weights for heatmap visualisation
        if self._capture:
            self.last_attn_weights = att.detach().cpu()

        att = self.attn_dropout(att)

        # Weighted sum → concatenate heads → output projection
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.out_proj(y))
        return y


# ============================================================
# Bottleneck Skip Connection
# ============================================================
class BottleneckSkip(nn.Module):
    """
    Bottleneck skip connection: compresses and restores the residual
    stream through a low-dimensional bottleneck.

    Standard residual (identity skip):
        out = x + Attn(LN(x))
        └─ x passes through untouched, preserving all detail

    Bottleneck residual:
        out = BottleneckSkip(x) + Attn(LN(x))
        └─ x is squeezed to fewer channels then restored, intentionally
           losing high-frequency detail. The network must rely on the
           attention branch to recover the missing information.

    Architecture:
        x (B, T, d_model)
          → Linear down  (d_model → bottleneck_dim)
          → GELU
          → Linear up    (bottleneck_dim → d_model)
          → out (B, T, d_model)

    Only applied to the attention residual path, NOT the FFN path.
    """

    def __init__(self, d_model: int, bottleneck_ratio: float = 0.5):
        super().__init__()
        bottleneck_dim = max(1, int(d_model * bottleneck_ratio))
        self.net = nn.Sequential(
            nn.Linear(d_model, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Causal Transformer Block (injection-based)
# (adapted from minGPT Block)
#
# The Block does NOT know which attention variant it holds.
# It only calls self.attn(x) and expects (B,T,C) → (B,T,C).
# ============================================================
class CausalTransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer block with injectable attention.

    Structure (default, bottleneck_ratio=None — same as minGPT):
        x  →  LN  →  Attention  →  (+x)  →  LN  →  FFN  →  (+x)  →  out

    Structure (with bottleneck skip):
        x  →  LN  →  Attention  ──┐
        x  →  BottleneckSkip  ────┘(+)  →  LN  →  FFN  →  (+x)  →  out

    The bottleneck deliberately discards high-frequency detail in the
    skip path, forcing the attention branch to become the primary source
    of fine-grained temporal information.

    The `attention` argument can be CausalMHA, CausalMLA, or any
    future variant — as long as its forward is (B,T,C) → (B,T,C).

    Input  : (B, T, d_model)
    Output : (B, T, d_model)
    """

    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        dropout: float = 0.1,
        bottleneck_ratio: float | None = None,
    ):
        super().__init__()

        # Pre-norm layers
        self.ln_1 = nn.LayerNorm(d_model)
        self.ln_2 = nn.LayerNorm(d_model)

        # Injected attention (could be MHA, MLA, or anything else)
        self.attn = attention

        # Attention skip path: identity (standard) or bottleneck
        if bottleneck_ratio is not None:
            self.attn_skip = BottleneckSkip(d_model, bottleneck_ratio)
        else:
            self.attn_skip = nn.Identity()

        # Feed-forward network (4× expansion, GELU, same as minGPT)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn_skip(x) + self.attn(self.ln_1(x))  # attention + residual
        x = x + self.ffn(self.ln_2(x))                    # FFN + residual (always identity)
        return x


# ============================================================
# Causal Transformer Decoder (stack of blocks)
# Handles (B,C,T) ↔ (B,T,C) boundary with the TCN world
# ============================================================
class CausalTransformerDecoder(nn.Module):
    """
    Stack of CausalTransformerBlocks with a final LayerNorm.

    Uses a causal (autoregressive) attention mask — each position can
    only attend to itself and past positions, making this a decoder in
    the transformer taxonomy (like GPT), not an encoder (like BERT).

    Sits between TCN_pt1 and TCN_pt2 in the sandwich architecture.
    The `attention_type` parameter is the ONLY switch between
    MHA and MLA — everything else (block structure, FFN, norms) is shared.

    Input  : (B, C, T)   — from TCN_pt1 output (channel-first)
    Output : (B, C, T)   — ready for TCN_pt2 input (channel-first)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        attention_type: str = "mha",        # "mha" or "mla"
        kv_lora_rank: int | None = None,    # only used when attention_type="mla"
        bottleneck_ratio: float | None = None,  # if set, compress the attention skip path
    ):
        super().__init__()
        self.attention_type = attention_type

        def _build_attention() -> nn.Module:
            """Factory: create one attention instance based on attention_type."""
            if attention_type == "mha":
                return CausalMHA(d_model, n_heads, max_seq_len, dropout)
            elif attention_type == "mla":
                return CausalMLA(d_model, n_heads, max_seq_len, dropout, kv_lora_rank)
            else:
                raise ValueError(
                    f"Unknown attention_type '{attention_type}'. Use 'mha' or 'mla'."
                )

        # Each block gets its own attention instance (independent weights)
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(
                attention=_build_attention(),
                d_model=d_model,
                dropout=dropout,
                bottleneck_ratio=bottleneck_ratio,
            )
            for _ in range(n_layers)
        ])

        # Final layer norm (same as minGPT's ln_f)
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, T) → (B, T, C) : enter attention world
        x = x.permute(0, 2, 1)

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        # (B, T, C) → (B, C, T) : return to TCN world
        x = x.permute(0, 2, 1)
        return x


# ============================================================
# Weight initialisation (adapted from minGPT)
# ============================================================
def init_transformer_weights(module: nn.Module, n_layers: int = 2):
    """
    Apply minGPT-style weight init to all transformer layers.

    - Linear layers   : N(0, 0.02)
    - Residual-path projections (out_proj, ffn second linear) :
      N(0, 0.02 / sqrt(2 * n_layers))  — GPT-2 scaled init
    - LayerNorm       : bias=0, weight=1
    """
    for name, param in module.named_parameters():
        if param.dim() < 2:
            continue

        # Residual-path projections get scaled init for stable deep residual streams
        is_residual_proj = ("out_proj.weight" in name) or ("ffn.2.weight" in name)
        std = 0.02 / math.sqrt(2 * n_layers) if is_residual_proj else 0.02
        nn.init.normal_(param, mean=0.0, std=std)

    for m in module.modules():
        if isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)


# ============================================================
# Self-test: verify both variants
# ============================================================
if __name__ == "__main__":

    B, C, T = 4, 64, 512       # batch, channels (=d_model), time steps

    def test_variant(attention_type: str, **kwargs):
        """Run shape check, causality check, and parameter count."""
        print(f"\n{'='*50}")
        print(f"Testing: {attention_type.upper()}")
        print(f"{'='*50}")

        # Build decoder
        decoder = CausalTransformerDecoder(
            d_model=C,
            n_heads=4,
            n_layers=2,
            max_seq_len=T,
            dropout=0.1,
            attention_type=attention_type,
            **kwargs,
        )
        init_transformer_weights(decoder, n_layers=2)

        # ---- Shape check ----
        x = torch.randn(B, C, T)
        y = decoder(x)
        print(f"Input  : {x.shape}")
        print(f"Output : {y.shape}")
        assert x.shape == y.shape, "Shape mismatch!"
        print("Shape check passed.")

        # ---- Causality check ----
        # Perturb time step 256 and verify past outputs are unchanged
        x1 = x.clone()
        x2 = x.clone()
        x2[:, :, 256] += 10.0

        decoder.eval()
        with torch.no_grad():
            y1 = decoder(x1)
            y2 = decoder(x2)

        past_diff = (y1[:, :, :256] - y2[:, :, :256]).abs().max().item()
        future_diff = (y1[:, :, 256:] - y2[:, :, 256:]).abs().max().item()

        print(f"Causality: past_diff={past_diff:.2e} (should be ~0), "
              f"future_diff={future_diff:.2e} (should be >0)")
        assert past_diff < 1e-5, "CAUSALITY VIOLATED!"
        print("Causality check passed.")

        # ---- Parameter count ----
        n_params = sum(p.numel() for p in decoder.parameters())
        print(f"Parameters: {n_params:,}")

        return n_params

    # Test both variants
    mha_params = test_variant("mha")
    mla_params = test_variant("mla", kv_lora_rank=16)

    # ---- Comparison summary ----
    print(f"\n{'='*50}")
    print(f"Parameter comparison (attention decoder only)")
    print(f"{'='*50}")
    print(f"MHA : {mha_params:>8,}")
    print(f"MLA : {mla_params:>8,}  ({mla_params/mha_params:.1%} of MHA)")
    print(f"Diff: {mha_params - mla_params:>+8,}")
