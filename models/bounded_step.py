"""Stateful bounded-lookahead TCN, emitting target t at acquisition t + L.

Each layer advances only after its upstream layer has produced a real sample.
Residual branches are delayed to the same target index. Cumulative SE starts
at recording index zero, just as a single whole-recording forward pass does;
it does NOT reproduce SE reset independently in overlapping sliding windows.
The final L targets are unavailable until more real input arrives. No padding
or fabricated future observations are emitted as valid streaming estimates.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.TCN_backbone import CausalConv1d
from models.step_graph import _pointwise


def _advance(old, new, acquired, offset):
    return torch.where(acquired > offset, new, old)


class _Conv(nn.Module):
    def __init__(self, conv, offset):
        super().__init__()
        self.conv, self.offset = conv.conv, offset
        self.delay = conv.pad_right
        self.span = (self.conv.kernel_size[0] - 1) * self.conv.dilation[0]
        self.shapes = [("conv", (self.conv.in_channels, self.span))] if self.span else []

    def forward(self, x, acquired, state):
        if not self.span:
            return self.conv(x), []
        full = torch.cat((state[0], x), -1)
        y = F.conv1d(full, self.conv.weight, self.conv.bias, dilation=self.conv.dilation)
        return y, [_advance(state[0], full[..., 1:], acquired, self.offset)]


class _Sequence(nn.Module):
    def __init__(self, seq, offset):
        super().__init__()
        self.layers, self.kinds, self.shapes = nn.ModuleList(), [], []
        for layer in (seq if isinstance(seq, nn.Sequential) else [seq]):
            if isinstance(layer, CausalConv1d):
                layer = _Conv(layer, offset)
                offset += layer.delay
                self.shapes += layer.shapes
                self.kinds.append("conv")
            else:
                self.kinds.append("pointwise")
            self.layers.append(layer)
        self.end_offset = offset

    def forward(self, x, acquired, state):
        new, i = [], 0
        for kind, layer in zip(self.kinds, self.layers):
            if kind == "conv":
                n = len(layer.shapes)
                x, s = layer(x, acquired, state[i:i+n])
                i += n
                new += s
            elif isinstance(layer, nn.Conv1d) and layer.kernel_size == (1,):
                x = layer(x)
            else:
                x = _pointwise(layer, x)
        return x, new


class _SE(nn.Module):
    def __init__(self, se, offset):
        super().__init__()
        if se.global_pool or se.pool_window is not None:
            raise ValueError("Bounded streaming requires cumulative SE, without global/finite pooling")
        self.se, self.offset = se, offset
        self.delay = int(se.lookahead or 0)
        self.shapes = [("se_sum", (se.fc1.in_channels, 1))]
        if self.delay:
            self.shapes.append(("se_delay", (se.fc1.in_channels, self.delay)))

    def forward(self, x, acquired, state):
        total = _advance(state[0], state[0] + x, acquired, self.offset)
        count = torch.clamp(acquired - self.offset, min=1.)
        gate = torch.sigmoid(self.se.fc2(self.se.act(self.se.fc1(total / count))))
        new = [total]
        if self.delay:
            delayed = state[1][..., :1]
            new.append(_advance(state[1], torch.cat((state[1][..., 1:], x), -1),
                                acquired, self.offset))
            x = delayed
        return x * gate, new


class _Block(nn.Module):
    def __init__(self, block, offset):
        super().__init__()
        self.offset = offset
        self.body = _Sequence(block.conv_block, offset)
        se = getattr(block, "se", None)
        self.se = _SE(se, self.body.end_offset) if se is not None else None
        self.end_offset = self.body.end_offset + (self.se.delay if self.se else 0)
        self.delay = self.end_offset - offset
        self.shortcut = block.shortcut
        channels = block.conv_block[0].conv.in_channels
        self.shapes = self.body.shapes + (self.se.shapes if self.se else [])
        if self.delay:
            self.shapes += [("residual_delay", (channels, self.delay))]

    def forward(self, x, acquired, state):
        n = len(self.body.shapes)
        out, new = self.body(x, acquired, state[:n])
        if self.se is not None:
            m = len(self.se.shapes)
            out, s = self.se(out, acquired, state[n:n+m])
            new += s
        skip = x
        if self.delay:
            skip = state[-1][..., :1]
            new.append(_advance(state[-1], torch.cat((state[-1][..., 1:], x), -1),
                                acquired, self.offset))
        return out + self.shortcut(skip), new


class BoundedStepGraph(nn.Module):
    """Static ONNX-compatible step graph for the architectural TCN ladder.

    x[u] produces y[u-L]. The first L returned values are invalid zeros.
    Reset states only at an independent recording boundary, not per hop.
    """
    topology = "tcn_bounded"

    def __init__(self, model):
        super().__init__()
        from models.lookahead import architectural
        horizon = architectural(model)
        if horizon is None or horizon <= 0 or not hasattr(model, "backbone"):
            raise ValueError("Expected a finite-lookahead TCN backbone")
        self.blocks = nn.ModuleList()
        offset = 0
        for block in model.backbone.network:
            part = _Block(block, offset)
            self.blocks.append(part)
            offset = part.end_offset
        self.head = _Sequence(model.head, offset)
        self.lookahead = self.head.end_offset
        if self.lookahead != horizon:
            raise ValueError("Layer delays do not match the declared model horizon")
        self.window = 1 + sum((m.conv.kernel_size[0]-1)*m.conv.dilation[0]
                              for m in model.modules() if isinstance(m, CausalConv1d))
        self.eval()

    @property
    def state_spec(self):
        shapes = [("acquired", (1, 1))]
        shapes += [s for b in self.blocks for s in b.shapes] + self.head.shapes
        return [{"name": f"{name}_{i}", "shape": list(shape)}
                for i, (name, shape) in enumerate(shapes)]

    def initial_state(self, batch=1):
        return [torch.zeros(batch, *s["shape"]) for s in self.state_spec]

    def forward(self, x_t, *state):
        acquired = state[0] + 1.
        new, i = [acquired], 1
        x = x_t
        for part in [*self.blocks, self.head]:
            n = len(part.shapes)
            x, s = part(x, acquired, state[i:i+n])
            i += n
            new += s
        return (torch.where(acquired > self.lookahead, x, torch.zeros_like(x)), *new)
