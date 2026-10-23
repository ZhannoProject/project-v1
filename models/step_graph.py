"""
One incremental step as a static graph, so it can leave PyTorch.

Why this exists
===============
``models/incremental.py`` advances a model one sample at a time, but it is
PyTorch, and the Pi gets ``numpy`` + ``onnxruntime`` only -- ``run_on_pi.sh``
copies six files and torch is not among them.  So the streaming path could be
measured on the host and nowhere else, and MLA's KV cache stayed a thing this
project could describe but not demonstrate on the hardware it was chosen for.

The obstacle is that ``IncrementalModel`` keeps its state in Python objects: a
ring buffer with a write pointer that moves per call.  A static graph has no
way to carry a pointer between calls.  This module expresses the same
computation with the state **passed in and handed back**, which is something
ONNX can represent:

    (x_t, state...) -> (y_t, new_state...)

The ring buffers become shift-and-append.  A ring avoids copying by moving an
index instead; a shift copies, which costs nothing here (the buffers are a few
hundred floats) and is pure tensor algebra, so it exports.

What it is checked against
==========================
``IncrementalModel`` is already pinned to the batch model from both ends, so
it is used as the oracle: ``tests/test_step_graph.py`` asserts this module
agrees with it sample for sample, and separately that the exported ONNX graph
agrees with this module.  Nothing here is believed because it looks right.

Warmup
======
A ring buffer can slice itself to the part that has been written.  A static
graph cannot, so every call attends over the whole window and a **validity
mask** rides along as state: one more column marked valid per step, with the
rest driven to zero weight through the softmax.  That makes the first
``window`` outputs exactly equal to the ring-buffer path's rather than
approximately so, which is what lets the two be compared at all.

Usage
=====
    graph = StepGraph(model)                 # causal + eval
    state = graph.initial_state()
    y, state = graph(x_t, *state)            # one sample, new state back

    graph.state_spec                         # names/shapes, for the runtime
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.AttentionBlocks.attention import (
    BottleneckSkip, CausalMHA, CausalMLA, CausalTransformerBlock,
    CausalTransformerDecoder,
)
from models.TCN_backbone import CausalConv1d, CausalTCNBackbone, ResidualBlock

#: Added to the scores of positions the stream has not reached yet.  Large
#: enough that ``exp`` underflows to exactly zero in float32, small enough not
#: to produce an inf when something downstream sums it.
MASK_FILL = -1e9


# ============================================================
# Convolution
# ============================================================
class ConvStep(nn.Module):
    """
    One ``CausalConv1d``, with its history passed in and handed back.

    The history is the ``(k-1)*dilation`` inputs before the current one.
    Concatenating it with the new sample gives exactly the window the
    convolution's own dilated kernel reads, so the step is a native ``Conv``
    over a short input rather than a hand-rolled gather -- one ONNX node,
    using the runtime's own kernel.
    """

    def __init__(self, conv: CausalConv1d):
        super().__init__()
        if not conv.causal:
            raise ValueError("ConvStep needs a causal convolution")
        self.conv = conv.conv
        self.dilation = self.conv.dilation[0]
        self.in_channels = self.conv.in_channels
        self.span = (self.conv.kernel_size[0] - 1) * self.dilation

    @property
    def state_shapes(self) -> List[Tuple[str, tuple]]:
        return [] if self.span == 0 else [("conv", (self.in_channels, self.span))]

    def forward(self, x_t, state):
        """``x_t``: (B, C_in, 1). Returns (B, C_out, 1) and the new history."""
        if self.span == 0:
            return F.conv1d(x_t, self.conv.weight, self.conv.bias), []
        full = torch.cat([state[0], x_t], dim=2)            # (B, C_in, span+1)
        y = F.conv1d(full, self.conv.weight, self.conv.bias, dilation=self.dilation)
        return y, [full[:, :, 1:]]


def _pointwise(module: nn.Module, x_t):
    """A layer that maps sample t to sample t, applied in (B, C, 1)."""
    if isinstance(module, nn.BatchNorm1d):
        if module.training:
            raise RuntimeError("BatchNorm1d is in training mode; call model.eval() first")
        return module(x_t)
    if isinstance(module, nn.PReLU):
        return module(x_t)
    if isinstance(module, (nn.Dropout, nn.Identity)):
        return x_t
    if isinstance(module, (nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh)):
        return module(x_t)
    if isinstance(module, nn.LayerNorm):                     # channel-last
        return module(x_t.transpose(1, 2)).transpose(1, 2)
    raise TypeError(f"not pointwise in time: {type(module).__name__}")


class SequentialStep(nn.Module):
    """An ``nn.Sequential`` of convolutions and pointwise layers."""

    def __init__(self, seq: nn.Module):
        super().__init__()
        self.steps = nn.ModuleList()
        self.kinds: List[str] = []
        for child in (seq if isinstance(seq, nn.Sequential) else [seq]):
            if isinstance(child, CausalConv1d):
                self.steps.append(ConvStep(child))
                self.kinds.append("conv")
            elif isinstance(child, nn.Conv1d) and child.kernel_size[0] == 1:
                self.steps.append(child)
                self.kinds.append("conv1x1")
            else:
                _pointwise(child, torch.zeros(1, 1, 1)) if False else None
                self.steps.append(child)
                self.kinds.append("pointwise")

    @property
    def state_shapes(self):
        out = []
        for kind, step in zip(self.kinds, self.steps):
            if kind == "conv":
                out += step.state_shapes
        return out

    def forward(self, x_t, state):
        out, i = x_t, 0
        new = []
        for kind, step in zip(self.kinds, self.steps):
            if kind == "conv":
                n = len(step.state_shapes)
                out, s = step(out, state[i:i + n])
                new += s
                i += n
            elif kind == "conv1x1":
                out = F.conv1d(out, step.weight, step.bias)
            else:
                out = _pointwise(step, out)
        return out, new


class SEStep(nn.Module):
    """
    ``CausalSE`` as a step: the running sum and the count come in and go out.

    Two state tensors and no window, which is the whole reason this gate is
    cheap to stream where attention is not. The count rides as a tensor
    rather than a Python integer for the same reason the attention mask does:
    a static graph cannot carry a counter between calls.
    """

    def __init__(self, se):
        super().__init__()
        if not se.causal:
            raise ValueError(
                "this SE block pools over the whole sequence (causal=False); its gate at t "
                "depends on samples after t and it has no streaming form")
        self.se = se
        self.channels = se.fc1.in_channels

    @property
    def state_shapes(self):
        return [("se_sum", (self.channels, 1)), ("se_count", (1, 1))]

    def forward(self, x_t, state):
        """``x_t``: (B, C, 1) -> (B, C, 1), with the running summary advanced."""
        run_sum, count = state
        run_sum = run_sum + x_t
        count = count + 1.0
        g = torch.sigmoid(self.se.fc2(self.se.act(self.se.fc1(run_sum / count))))
        return x_t * g, [run_sum, count]


class ResidualBlockStep(nn.Module):
    def __init__(self, block: ResidualBlock):
        super().__init__()
        self.body = SequentialStep(block.conv_block)
        # Optional. Leaving it out silently would make the graph a different
        # model from the checkpoint it was exported from.
        self.se = SEStep(block.se) if getattr(block, "se", None) is not None else None
        self.shortcut = None if isinstance(block.shortcut, nn.Identity) else block.shortcut

    @property
    def state_shapes(self):
        return self.body.state_shapes + (self.se.state_shapes if self.se is not None else [])

    def forward(self, x_t, state):
        skip = x_t if self.shortcut is None else F.conv1d(
            x_t, self.shortcut.weight, self.shortcut.bias)
        n = len(self.body.state_shapes)
        out, new = self.body(x_t, state[:n])
        if self.se is not None:
            out, se_state = self.se(out, state[n:])
            new = new + se_state
        return out + skip, new


class TCNStep(nn.Module):
    def __init__(self, backbone: CausalTCNBackbone):
        super().__init__()
        if not backbone.causal:
            raise ValueError("this TCN backbone is not causal")
        self.blocks = nn.ModuleList([ResidualBlockStep(b) for b in backbone.network])

    @property
    def state_shapes(self):
        return [s for b in self.blocks for s in b.state_shapes]

    def forward(self, x_t, state):
        out, i, new = x_t, 0, []
        for b in self.blocks:
            n = len(b.state_shapes)
            out, s = b(out, state[i:i + n])
            new += s
            i += n
        return out, new


# ============================================================
# Attention: the cache as a graph tensor, in and out
# ============================================================
class MHAStep(nn.Module):
    """
    ``CausalMHA`` as a step: K and V come in, the shifted caches go out.

    ``2 * d_model`` floats per cached position, which is the number MLA is
    claimed to beat and the reason this path can finally weigh it on the
    board rather than on the host.
    """
    kind = "mha"

    def __init__(self, attn: CausalMHA):
        super().__init__()
        if not attn.causal:
            raise ValueError("CausalMHA is bidirectional; it cannot be streamed")
        if attn.turboquant is not None:
            raise ValueError("TurboQuant quantises K and V inside the batch pass; "
                             "export the full-precision model")
        self.attn = attn
        self.h, self.hd = attn.n_heads, attn.head_dim
        self.d = attn.d_model
        self.window = attn.causal_mask.shape[-1]
        self.scale = self.hd ** -0.5

    @property
    def state_shapes(self):
        return [("kv_k", (self.h, self.window, self.hd)),
                ("kv_v", (self.h, self.window, self.hd))]

    def forward(self, x_t, state, mask):
        """``x_t``: (B, 1, d_model). ``mask``: (B, 1, 1, window), 0 or MASK_FILL."""
        k_cache, v_cache = state
        B = x_t.shape[0]
        q, k, v = F.linear(x_t, self.attn.qkv_proj.weight,
                           self.attn.qkv_proj.bias).split(self.d, dim=2)
        q = q.view(B, 1, self.h, self.hd).permute(0, 2, 1, 3)        # (B,H,1,hd)
        k = k.view(B, 1, self.h, self.hd).permute(0, 2, 1, 3)
        v = v.view(B, 1, self.h, self.hd).permute(0, 2, 1, 3)

        k_new = torch.cat([k_cache[:, :, 1:, :], k], dim=2)
        v_new = torch.cat([v_cache[:, :, 1:, :], v], dim=2)

        att = (q @ k_new.transpose(-2, -1)) * self.scale + mask      # (B,H,1,window)
        att = torch.softmax(att, dim=-1)
        y = (att @ v_new).permute(0, 2, 1, 3).reshape(B, 1, self.d)
        y = F.linear(y, self.attn.out_proj.weight, self.attn.out_proj.bias)
        return y, [k_new, v_new]


class MLAStep(nn.Module):
    """
    ``CausalMLA`` as a step, caching the latent and nothing else.

    ``kv_lora_rank`` floats per cached position against MHA's
    ``2 * d_model``.  K and V are never built: DeepSeek-V2's absorption folds
    the key up-projection into the query and the value up-projection past the
    weighted sum, exactly as in ``models/incremental.py``.  The key
    up-projection's bias is dropped because it is constant across the cached
    positions and softmax removes it.

    This is the whole point of the exercise. Here the cache is a tensor the
    graph declares, so its size is something onnxruntime allocates on the
    board rather than something this project computes about itself.
    """
    kind = "mla"

    def __init__(self, attn: CausalMLA):
        super().__init__()
        if not attn.causal:
            raise ValueError("CausalMLA is bidirectional; it cannot be streamed")
        if attn.turboquant is not None:
            raise ValueError("TurboQuant quantises K and V inside the batch pass; "
                             "export the full-precision model")
        self.attn = attn
        self.h, self.hd = attn.n_heads, attn.head_dim
        self.d, self.r = attn.d_model, attn.kv_lora_rank
        self.window = attn.causal_mask.shape[-1]
        self.scale = self.hd ** -0.5
        # Folded once, at construction: (H, r, hd) and (H, hd, r).
        # Detached because these are constants derived from the weights, and
        # a tensor that still carries requires_grad cannot be inlined into a
        # TorchScript graph -- the export fails on it rather than on anything
        # about the arithmetic.
        with torch.no_grad():
            w_k = attn.k_up_proj.weight.detach().view(self.h, self.hd, self.r)
            self.register_buffer("k_absorb", w_k.transpose(1, 2).contiguous(), persistent=False)
            self.register_buffer("w_v",
                                 attn.v_up_proj.weight.detach()
                                 .view(self.h, self.hd, self.r).contiguous(), persistent=False)
            self.register_buffer("b_v",
                                 attn.v_up_proj.bias.detach()
                                 .view(self.h, self.hd).contiguous(), persistent=False)

    @property
    def state_shapes(self):
        return [("kv_c", (self.window, self.r))]

    def forward(self, x_t, state, mask):
        c_cache, = state
        B = x_t.shape[0]
        q = F.linear(x_t, self.attn.q_proj.weight, self.attn.q_proj.bias)
        c = self.attn.kv_norm(F.linear(x_t, self.attn.kv_down_proj.weight,
                                       self.attn.kv_down_proj.bias))       # (B,1,r)
        c_new = torch.cat([c_cache[:, 1:, :], c], dim=1)                   # (B,window,r)

        q = q.view(B, self.h, self.hd)
        q_latent = torch.einsum("bhd,hrd->bhr", q, self.k_absorb)          # (B,H,r)
        att = torch.einsum("bhr,bnr->bhn", q_latent, c_new) * self.scale
        att = att + mask.view(B, 1, self.window)
        att = torch.softmax(att, dim=-1)
        z = torch.einsum("bhn,bnr->bhr", att, c_new)                       # (B,H,r)
        y = torch.einsum("bhr,hdr->bhd", z, self.w_v) + self.b_v
        y = y.reshape(B, 1, self.d)
        return F.linear(y, self.attn.out_proj.weight, self.attn.out_proj.bias), [c_new]


class BlockStep(nn.Module):
    """``CausalTransformerBlock``: attention is stateful, the rest is pointwise."""

    def __init__(self, block: CausalTransformerBlock):
        super().__init__()
        self.block = block
        if isinstance(block.attn, CausalMLA):
            self.attn = MLAStep(block.attn)
        elif isinstance(block.attn, CausalMHA):
            self.attn = MHAStep(block.attn)
        else:
            raise TypeError(f"no step form for {type(block.attn).__name__}")
        if not isinstance(block.attn_skip, (nn.Identity, BottleneckSkip)):
            raise TypeError(f"unsupported attention skip: {type(block.attn_skip).__name__}")

    @property
    def state_shapes(self):
        return self.attn.state_shapes

    def forward(self, x_t, state, mask):
        """``x_t``: (B, 1, d_model), channel-last like the batch block."""
        a, new = self.attn(self.block.ln_1(x_t), state, mask)
        x_t = self.block.attn_skip(x_t) + a
        return x_t + self.block.ffn(self.block.ln_2(x_t)), new


class DecoderStep(nn.Module):
    """
    The decoder, plus the one validity mask its blocks share.

    All layers see the same timeline, so the mask is decoder-level state
    rather than one per block: it is shifted once and handed to each.
    """

    def __init__(self, decoder: CausalTransformerDecoder):
        super().__init__()
        if not decoder.causal:
            raise ValueError("this decoder is bidirectional; it cannot be streamed")
        self.decoder = decoder
        self.blocks = nn.ModuleList([BlockStep(b) for b in decoder.blocks])
        self.window = self.blocks[0].attn.window

    @property
    def state_shapes(self):
        return ([("attn_mask", (1, self.window))]
                + [s for b in self.blocks for s in b.state_shapes])

    def forward(self, x_t, state):
        """``x_t``: (B, C, 1) from the TCN world; returns the same."""
        mask_state = state[0]
        # One more position becomes valid with this sample.
        mask_new = torch.cat([mask_state[:, :, 1:], torch.zeros_like(mask_state[:, :, :1])], dim=2)
        mask = mask_new.view(-1, 1, 1, self.window)

        h = x_t.transpose(1, 2)                                  # (B,1,C)
        i, new = 1, [mask_new]
        for b in self.blocks:
            n = len(b.state_shapes)
            h, s = b(h, state[i:i + n], mask)
            new += s
            i += n
        return self.decoder.ln_f(h).transpose(1, 2), new


# ============================================================
# The model, as one exportable step
# ============================================================
class StepGraph(nn.Module):
    """
    A whole variant as ``(x_t, state...) -> (y_t, new_state...)``.

    The topology is detected the same way ``IncrementalModel`` detects it --
    from the modules the model has rather than its class name -- so the rank
    aliases and the distilled students work without being listed.

    The state is a flat list of tensors because that is what an ONNX graph
    signature is.  ``state_spec`` names them in order and gives their shapes
    without a batch axis, which is what the runtime needs to allocate them.

    Bidirectional and TurboQuant models are refused, for the same reasons the
    rest of the deployment path refuses them.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        if getattr(model, "causal", True) is False:
            raise ValueError("this model's attention is bidirectional; it cannot be streamed")
        if getattr(model, "causal_tcn", True) is False:
            raise ValueError("this model's convolutions are non-causal; it cannot be streamed")
        self.model = model.eval()

        has = lambda n: isinstance(getattr(model, n, None), nn.Module)
        if has("backbone"):
            self.topology = "tcn"
            self.tcn_pt1, self.decoder = TCNStep(model.backbone), None
            self.squeeze = self.expand = self.tcn_pt2 = None
            self.head = SequentialStep(model.head)
        elif has("decoder") and has("squeeze") and has("expand"):
            self.topology = "sandwich"
            self.tcn_pt1 = TCNStep(model.tcn_pt1)
            self.squeeze, self.expand = ConvStep(model.squeeze), ConvStep(model.expand)
            self.decoder = DecoderStep(model.decoder)
            self.tcn_pt2 = TCNStep(model.tcn_pt2)
            self.head = SequentialStep(model.head)
        elif has("decoder") and has("tcn_pt2"):
            self.topology = "parallel"
            self.tcn_pt1 = TCNStep(model.tcn_pt1)
            self.squeeze = self.expand = None
            self.decoder = DecoderStep(model.decoder)
            self.tcn_pt2 = TCNStep(model.tcn_pt2)
            self.head = SequentialStep(model.head)
        else:
            raise TypeError(f"{type(model).__name__} has no recognised streaming topology")

        self._parts = [p for p in (self.tcn_pt1, self.squeeze, self.decoder,
                                   self.expand, self.tcn_pt2, self.head) if p is not None]

    # -- the signature --------------------------------------
    @property
    def state_spec(self) -> List[dict]:
        """
        One entry per state tensor, in the order ``forward`` takes them.

        Names are made unique by position, because several convolutions
        contribute a tensor called ``conv`` and an ONNX input needs its own
        name.
        """
        spec, seen = [], {}
        for part in self._parts:
            for base, shape in getattr(part, "state_shapes", []):
                seen[base] = seen.get(base, 0) + 1
                spec.append({"name": f"{base}_{seen[base] - 1}", "shape": list(shape)})
        return spec

    @property
    def window(self) -> int:
        """
        How much history an output depends on.

        The attention window where there is attention; otherwise the
        convolutions' receptive field, which is the honest answer for the
        baseline rather than the zero "no decoder" would give.
        """
        if self.decoder is not None:
            return self.decoder.window
        return 1 + sum(s.span for part in self._parts
                       for s in self._conv_steps(part))

    @staticmethod
    def _conv_steps(part):
        """Every ConvStep inside a part, however it is nested."""
        return [m for m in part.modules() if isinstance(m, ConvStep)] \
            if isinstance(part, nn.Module) else []

    def initial_state(self, batch: int = 1) -> List[torch.Tensor]:
        """
        Zeroed buffers, and a mask that marks every position invalid.

        Zero is the left padding ``CausalConv1d`` applies, so the first steps
        reproduce a window that starts at t=0 rather than merely resembling
        one; the mask starts fully closed so the attention sees only what the
        stream has actually fed it.
        """
        out = []
        for s in self.state_spec:
            shape = (batch, *s["shape"])
            if s["name"].startswith("attn_mask"):
                out.append(torch.full(shape, MASK_FILL))
            else:
                out.append(torch.zeros(shape))
        return out

    # -- the step -------------------------------------------
    def forward(self, x_t, *state):
        """``x_t``: (B, 1, 1). Returns ``(y_t, *new_state)``."""
        state = list(state)
        new: List[torch.Tensor] = []

        def run(part, value):
            nonlocal state, new
            n = len(getattr(part, "state_shapes", []))
            taken, state = state[:n], state[n:]
            out, produced = part(value, taken)
            new += produced
            return out

        h = run(self.tcn_pt1, x_t)
        if self.topology == "tcn":
            y = run(self.head, h)
        elif self.topology == "sandwich":
            a = run(self.decoder, run(self.squeeze, h))
            y = run(self.head, run(self.tcn_pt2, run(self.expand, a)))
        else:
            branch_a = run(self.decoder, h)
            branch_b = run(self.tcn_pt2, h)
            y = run(self.head, torch.cat([branch_a, branch_b], dim=1))
        return (y, *new)


__all__ = ["MASK_FILL", "BlockStep", "ConvStep", "DecoderStep", "MHAStep", "MLAStep",
           "ResidualBlockStep", "SequentialStep", "StepGraph", "TCNStep"]
