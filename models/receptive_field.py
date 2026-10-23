"""
Receptive field: how far back a causal model is allowed to look.

``models/lookahead.py`` asks what *future* context is worth, by buying it a
sample at a time and charging latency for it. This is the mirror question,
and the cheaper one, because past context costs no latency at all:

    how much of the past does the model actually need?

Why it is worth a sweep
-----------------------
The attention variants are justified by long-range context: a 512-sample
window mixed globally, against a TCN that reaches back 125 samples (0.49 s at
256 Hz). On this project's own Pi measurements that costs roughly 5x the
baseline's compute, and the accuracy it buys is about 1% of CC -- within the
spread of a single seed.

But "attention helps" and "more context helps" are different claims, and the
study cannot separate them while the TCN's reach is fixed. A dilated stack
can be made to reach as far as the window for a few thousand parameters and
no attention at all. So:

* if error keeps falling as the field grows, the benefit was context, and the
  cheap way to buy it is another dilation level;
* if it flattens at 125, the model already had all the context it could use,
  and whatever attention contributes is not reach.

Either answer is worth having, and neither is available from the runs that
exist.

The ladder
----------
One dilation level per rung, doubling, so the field roughly doubles with it::

    [1]                      ->    5 samples    20 ms
    [1, 2]                   ->   13             51 ms
    [1, 2, 4]                ->   29            113 ms
    [1, 2, 4, 8]             ->   61            238 ms
    [1, 2, 4, 8, 16]         ->  125            488 ms   <- the project default
    [1, 2, 4, 8, 16, 32]     ->  253            988 ms
    [1, 2, 4, 8, 16, 32, 64] ->  509           1988 ms   ~ the whole 512 window

The top rung reaches essentially the attention window, which is the point:
it is the TCN given the same span attention has, without the attention.

Parameters do not confound it the way they would for a width sweep -- a rung
adds one residual block, about 25k parameters at the default width -- but
they are not constant either, and ``rung_table()`` reports them so the
comparison is read with that in view.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch.nn as nn

from models.TCN_backbone import CausalConv1d

#: Sampling rate the epochs were cut at, for turning samples into milliseconds.
SAMPLE_RATE_HZ = 256.0

#: The dilation schedules the sweep is reported over, shortest reach first.
#: Each rung adds one level and doubles the dilation, so the field doubles too.
RF_LADDER: tuple = (
    (1,),
    (1, 2),
    (1, 2, 4),
    (1, 2, 4, 8),
    (1, 2, 4, 8, 16),
    (1, 2, 4, 8, 16, 32),
    (1, 2, 4, 8, 16, 32, 64),
)

#: What ``tcn`` and ``tcn_se`` use when nothing says otherwise.
DEFAULT_DILATIONS: tuple = (1, 2, 4, 8, 16)


def field(dilations: Sequence[int], kernel_size: int = 3, *, se: bool = False) -> int:
    """
    Samples of history an output depends on, counting the sample itself.

    Two convolutions per residual block, each reaching back ``(k-1) * d``, and
    they compose by addition down the stack. An SE gate adds nothing here: its
    squeeze is a running mean over everything already seen, so it widens no
    further than the convolutions beneath it have already reached.

    Verified against the perturbation probe in ``measured_field`` and against
    ``IncrementalModel.receptive_field``, which arrives at the same number by
    counting the ring buffers it has to allocate.
    """
    return 1 + sum(2 * (kernel_size - 1) * d for d in dilations)


def ms(samples: Optional[int], fs: float = SAMPLE_RATE_HZ) -> Optional[float]:
    """Samples as milliseconds, for reading a rung as a span of signal."""
    return None if samples is None else round(1000.0 * samples / fs, 1)


def architectural(model: nn.Module) -> int:
    """
    The field the built model actually has, summed off its own convolutions.

    Read from the modules rather than from a config, so it answers for a
    checkpoint that has been loaded as well as for one just constructed -- the
    same reason ``lookahead.architectural`` works that way.
    """
    span = 0
    for m in model.modules():
        if isinstance(m, CausalConv1d):
            span += (m.conv.kernel_size[0] - 1) * m.conv.dilation[0]
    return 1 + span


def dilations_of(model: nn.Module) -> List[int]:
    """The schedule a model was built with, in order, one entry per block."""
    out = []
    for name, m in model.named_modules():
        if name.endswith("backbone") or name.endswith("tcn_pt1") or name.endswith("tcn_pt2"):
            out.extend(getattr(m, "dilations", []))
    return out


def rung_table(hidden_channels: int = 64, variant: str = "tcn") -> List[dict]:
    """
    One row per rung: the schedule, its reach, and what it costs.

    Built by constructing each model, so the parameter counts are the real
    ones rather than a formula that could drift from the constructors.
    """
    from models.variants import build_model

    rows = []
    for dil in RF_LADDER:
        model = build_model(variant, hidden_channels=hidden_channels, dilations=list(dil))
        rf = architectural(model)
        rows.append({
            "dilations": list(dil),
            "levels": len(dil),
            "receptive_field": rf,
            "ms": ms(rf),
            "params": sum(p.numel() for p in model.parameters()),
            "variant": f"{variant}_rf{rf}",
            "is_default": tuple(dil) == DEFAULT_DILATIONS,
        })
    return rows


def measured_field(model: nn.Module, seq_len: int = 1024) -> int:
    """
    The field by experiment: perturb one sample and see how far it reaches.

    The architectural sum is arithmetic over the modules; this is the model
    answering for itself. They are asserted equal in the tests, because a
    schedule that does not produce the reach it claims would make every point
    on the sweep's x-axis wrong.

    Probed in float64, which is not fussiness. The path that reaches furthest
    is a single chain of kernel edges, attenuated once per block, so at the
    far end of a 253-sample field the output moves by about 6e-9 while the
    output itself is of order 0.1 -- below the float32 spacing at that
    magnitude (3e-8).  The difference then rounds to exactly zero and the
    probe reports one sample short of the truth, at some inits and not
    others.  Nothing is dead: the dependence is real and PReLU keeps a slope
    of 0.25 on its negative side, so no unit can zero it (the same reason the
    lookahead probe's ``licensed`` and ``measured`` agree on the convolution
    stack).  In float64 that tail sits eight orders above the rounding floor.

    Raises if a perturbation shows up *before* the sample it was applied to,
    which would mean the stack reads ahead.  That direction needs no care
    about precision: for a causal model the difference in the past is
    structurally, exactly zero.
    """
    import copy

    import torch

    probe = copy.deepcopy(model).double().eval()   # a copy: .double() is in place
    hit = seq_len // 2
    x1 = torch.zeros(1, 1, seq_len, dtype=torch.float64)
    x2 = x1.clone()
    x2[0, 0, hit] = 1.0
    with torch.no_grad():
        d = (probe(x1) - probe(x2))[0, 0].abs()
    touched = (d > 0).nonzero()
    if not len(touched):
        return 0
    first = int(touched.min())
    if first < hit:
        raise ValueError(
            f"perturbing sample {hit} changed output {hit - first} samples earlier: "
            f"this model reads ahead, so it has no causal receptive field"
        )
    return int(touched.max()) - hit + 1


__all__ = ["DEFAULT_DILATIONS", "RF_LADDER", "SAMPLE_RATE_HZ", "architectural",
           "dilations_of", "field", "measured_field", "ms", "rung_table"]
