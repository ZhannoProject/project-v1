"""
Lookahead: causality as a dial rather than a switch.

This project has always had two causality settings and nothing in between.
``tcn_se`` is strictly causal and deployable; ``tcn_se_bi`` sees the whole
512-sample epoch and cannot be. The gap between them is real and large --
RRMSE_t ~0.708 against ~0.595, the "causality cost" -- but two points do not
say *where* in between the cost is paid, and that is the question an online
BCI actually asks. Future context is not free or forbidden; it is buyable, at
a price fixed by physics:

    L samples of lookahead  =  L / fs seconds of added output latency
                            =  3.9 ms per sample at 256 Hz

So the useful form of the question is a curve: how much does the error fall
per millisecond of latency conceded, and where does it stop falling. A system
designer with a 50 ms budget can read the answer off it; "causal or not"
cannot be read off anything.

What a lookahead budget means here
----------------------------------
``lookahead=L`` means the output at sample t may depend on the input up to and
including sample ``t + L``, and on nothing after it. That is a statement about
the model's **total forward receptive field**, not about any one layer, and it
is chosen that way because it is the only reading that converts directly into
latency.

Every site in the stack that can reach forward does so additively:

* each dilated convolution may move up to ``(k-1) * dilation`` taps of its
  padding from the left to the right edge;
* each Squeeze-and-Excitation gate may average over ``[0 .. t + L_se]``
  instead of ``[0 .. t]``.

For a feed-forward stack these compose by addition -- a block whose output at
t reads its input to ``t + a``, feeding one that reads to ``t + b``, together
read to ``t + a + b`` -- so a budget is *allocated* across the sites and the
licensed total is exactly the sum of what each was given. ``allocate()``
spends the budget proportionally to what each site can take.

Two numbers, and they are not the same number
---------------------------------------------
``architectural()`` sums what the built model is *licensed* to read. That is
the one that fixes latency: an output that may depend on ``t + L`` cannot be
emitted until sample ``t + L`` has arrived, whether or not the weights happen
to use it. It is exact, and it is a property of the graph.

``forward_receptive_field()`` measures how far the model's weights *actually*
reach, by perturbing one input sample and finding how far back the output
moves. For the convolution stack the two agree exactly -- 62 samples on
``tcn_bi``, the number the switch probe independently recovers as 242 ms at
256 Hz -- because nothing on that path can zero a dependence: PReLU has a
nonzero slope either side and BatchNorm is affine in eval mode.

For the SE gates they do not always agree, and the reason is worth stating
rather than discovering later. The gate is ``sigmoid(W2 @ relu(W1 @ s))``, and
**ReLU can be exactly dead**: where ``W1 @ s`` is negative the gate stops
depending on ``s`` at all, so a gate licensed to peek L ahead contributes
nothing there. At the narrow end this is not a tail risk -- ``channels //
reduction`` is 1 for a 8-channel block at reduction 8, a single hidden unit,
dead for about half of random inits -- and even at the trained width of 8
hidden units a particular draw and a particular input leave some positions
dead. So the measured reach is ``<= architectural``, sometimes by a lot:
186 licensed measures ~150 on a fresh init.

Neither number is the "real" one. The licensed figure is what you must pay
for in latency; the measured figure is how much of that licence a trained
model exercises, which is itself a result worth having -- a model given 186
samples of future that only ever uses 150 of them is telling you something
about where the useful context is. The Lookahead tab shows both, and
tests/test_lookahead.py asserts the ordering between them plus exact equality
on the convolution-only path.

The endpoints are the models that already exist:

    lookahead=0     ==  tcn_se      (causal, cumulative-mean SE)
    lookahead=None  ==  tcn_se_bi   (centred padding, global-mean SE)

``L=0`` is bit-identical to the model built without this module, and ``None``
leaves the legacy non-causal path untouched, so nothing already trained or
measured moves.

Why the top of the ladder is 186 and not 512
--------------------------------------------
Capacity is finite. A convolution can shift its padding right but cannot
invent taps: with dilations [1, 2, 4, 8, 16] and kernel 3 the conv stack holds
124 samples of forward reach (62 when centred, which is what ``tcn_bi`` has
and what the perturbation probe measures on it). Adding the five SE gates at
the same per-block capacity gives 186. Beyond that there is no bounded setting
left and the only remaining rung is the unbounded one, ``tcn_se_bi``, whose
global pool reaches the whole epoch. A request above capacity is clamped, and
the clamp is reported rather than hidden -- `granted` is what the model got.
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn


#: The rungs the Lookahead tab offers by default, in samples. Powers of two up
#: to the capacity of the SE baseline's stack, plus the two anchors that are
#: worth naming: 62 is the forward reach of ``tcn_bi`` (centred convolutions),
#: so a model at 62 sees exactly as far ahead as the project's existing
#: non-causal baseline while keeping a bounded horizon; 186 is full capacity.
#: 0 and "unbounded" are not in here because they are not lookahead variants
#: at all -- they are ``tcn_se`` and ``tcn_se_bi``, already trained.
LOOKAHEAD_LADDER = (1, 2, 4, 8, 16, 32, 62, 124, 186)

#: Sample rate the ladder is quoted in milliseconds against (EEGdenoiseNet).
SAMPLE_RATE_HZ = 256.0


def ms(samples: Optional[int], fs: float = SAMPLE_RATE_HZ) -> Optional[float]:
    """Latency a lookahead of ``samples`` costs, in milliseconds."""
    return None if samples is None else 1000.0 * samples / fs


def allocate(caps: Sequence[int], budget: Optional[int]) -> Optional[List[int]]:
    """
    Spend ``budget`` samples of lookahead across sites of capacity ``caps``.

    Returns one integer per site, summing to ``min(budget, sum(caps))``, with
    no site over its own capacity. ``None`` passes straight through: it is the
    legacy "centred / global" setting, which is not a budget and is not
    allocated.

    Proportional to capacity, with the remainder going to the sites whose
    exact share was cut by most. Proportional rather than greedy-from-the-top
    because the alternative buries a second variable in the knob: filling the
    earliest layers first would make a small budget a *shallow* lookahead and
    a large one a deep one, and the curve would then be measuring the shape of
    the allocation as much as its extent. Spreading it keeps the shape fixed
    and the extent the only thing that moves.
    """
    if budget is None:
        return None
    if not isinstance(budget, int) or isinstance(budget, bool):
        raise TypeError(f"lookahead must be an int or None, not {type(budget).__name__}")
    if budget < 0:
        raise ValueError(f"lookahead must be >= 0, not {budget}")

    caps = list(caps)
    total = sum(caps)
    if total == 0 or budget == 0:
        return [0] * len(caps)

    budget = min(budget, total)
    exact = [budget * c / total for c in caps]
    out = [int(e) for e in exact]
    # Hand out what rounding down left over, largest shortfall first; ties go
    # to the larger site. A site already at its cap has zero shortfall and so
    # sorts last, which is what keeps every entry within capacity.
    order = sorted(range(len(caps)), key=lambda i: (-(exact[i] - out[i]), -caps[i]))
    for i in order[: budget - sum(out)]:
        out[i] += 1

    assert sum(out) == budget, (out, budget)
    assert all(0 <= o <= c for o, c in zip(out, caps)), (out, caps)
    return out


def capacity(dilations: Sequence[int], kernel_size: int = 3, *, se: bool = False) -> int:
    """
    The largest bounded lookahead a stack of these dilations can hold.

    Two convolutions per residual block, each able to move ``(k-1) * d`` taps
    to the right, plus -- when the blocks carry SE gates -- one gate per block
    at the same per-block capacity.
    """
    per_conv = [(kernel_size - 1) * d for d in dilations]
    return sum(2 * p for p in per_conv) + (sum(per_conv) if se else 0)


def architectural(model: nn.Module) -> Optional[int]:
    """
    The lookahead the built model is *licensed* to use, in samples.

    Summed off the modules themselves rather than read from a config, so it
    answers for a checkpoint that has been loaded as well as for a model that
    has just been configured. ``None`` means unbounded: a global-pool SE gate
    or an unmasked attention block reaches the whole window, and no finite
    number describes it.

    This is the figure that sets the output delay a deployment must carry.
    """
    # Imported here, not at module scope: TCN_backbone imports allocate() from
    # this module, so a top-level import the other way is a cycle.
    from models.TCN_backbone import CausalConv1d, CausalSE
    from models.AttentionBlocks.attention import CausalMHA, CausalMLA

    total = 0
    for m in model.modules():
        if isinstance(m, CausalSE):
            if m.global_pool:
                return None
            total += m.lookahead or 0
        elif isinstance(m, CausalConv1d):
            total += m.pad_right
        elif isinstance(m, (CausalMHA, CausalMLA)) and not m.causal:
            # Attention with its mask dropped reaches the whole window; no
            # finite sum describes it.
            return None
    # Matched by class, not by a ``causal`` flag: the model, its backbone and
    # its blocks all carry one too, and all of them read False as soon as any
    # site below them looks ahead -- so a flag test would see the container
    # first and call every bounded model unbounded.
    return total


def backward_taps(model: nn.Module) -> int:
    """
    How much of the past the convolution stack still reaches, in taps.

    The counterpart to ``architectural``, and the quantity the ladder was
    missing. A convolution of kernel k and dilation d has ``(k-1)*d`` taps to
    place, and places each one either side of t, never both: ``pad_left +
    pad_right`` is fixed by the kernel. **A lookahead budget is therefore not
    extra context -- it is the same window slid forward, and every tap handed
    to the future is taken from the past.**

    That is why the ladder's price curve turns around at the top instead of
    flattening. At the full budget for this stack (186) every ``pad_left`` is
    0: the convolutions see nothing at all before t, and the model is the
    anti-causal *mirror* of the causal baseline rather than a better-informed
    version of it. Measured against the trained runs, the error tracks how far
    the window sits from centred, not how much lookahead was bought::

        model          back  ahead   RRMSE_t
        tcn_se          124      0    0.7104     <- all past
        tcn_se_la62      82     42    0.6046
        tcn_se_la124     41     83    0.6025
        tcn_se_la186      0    124    0.6673     <- all future, the mirror
        tcn_se_bi        62     62    0.5950     <- centred, and the best

    The SE gates are left out of the count on purpose: their squeeze is a
    running mean from 0, so they keep reaching the whole past at every rung
    (that is why the top rung still beats the causal baseline -- it has the
    past's *statistics* without any of its detail). This number is about the
    convolution window, which is what the budget moves.
    """
    from models.TCN_backbone import CausalConv1d

    return sum(m.pad_left for m in model.modules() if isinstance(m, CausalConv1d))


@torch.no_grad()
def forward_receptive_field(model: nn.Module, *, seq_len: int = 512,
                            probe_at: Optional[int] = None, tol: float = 0.0,
                            device=None, seed: int = 0) -> Optional[int]:
    """
    How far ahead the model actually looks, measured rather than computed.

    Perturbs one input sample and asks how far *back* the output moves: if
    changing sample p changes the output at p - d, then the output at p - d
    depended on a sample d ahead of it. The largest such d is the forward
    receptive field. This is the same probe that measures 62 samples on
    ``tcn_bi`` -- the number the switch probe independently recovers as 242 ms
    at 256 Hz -- so it is the project's established instrument, not a new one.

    Returns the distance in samples, 0 for a strictly causal model, or
    ``None`` when the output moves at every offset measured: the reach then
    exceeds what this window can show (a global-pool SE reads the whole
    epoch), which is a different answer from a large finite number and is
    reported as one.

    ``tol`` defaults to 0 -- any difference at all counts -- because beyond
    the reach there is nothing to threshold: those outputs are computed from
    inputs that did not change, so they are bit-identical and the difference
    is exactly zero. That is what makes this a proof of a boundary rather
    than an estimate of one, and it is why a nonzero ``tol`` reads *short*:
    at 1e-6 a long chain of small sensitivities falls under the threshold and
    186 licensed samples measure 128 instead of ~150.

    Runs in ``eval()`` on CPU by default. Both matter: in training mode
    BatchNorm's statistics are taken over the batch *and* time, which couples
    every position to every other and would report an unbounded reach for a
    strictly causal model; and CPU keeps the untouched region bit-identical,
    so the measurement is a step rather than a threshold on noise.

    See the module docstring for why this can come back below
    ``architectural()``: a dead ReLU in an SE gate drops a licensed
    dependence to exactly zero.
    """
    was_training = model.training
    model.eval()
    try:
        device = torch.device(device) if device is not None else torch.device("cpu")
        model = model.to(device)
        p = seq_len // 2 if probe_at is None else probe_at
        if not 0 < p < seq_len:
            raise ValueError(f"probe_at must be inside the window, got {p} for seq_len {seq_len}")

        g = torch.Generator(device="cpu").manual_seed(seed)
        x = torch.randn(1, 1, seq_len, generator=g).to(device)
        bumped = x.clone()
        # Large enough that the change cannot be lost in fp32 rounding where
        # the model's gain is small, which would read as a shorter reach.
        bumped[0, 0, p] += 10.0

        base = model(x)
        # Reduced over every axis but time, not read off one channel: a change
        # can surface in any of them, and the earliest offset at which *any*
        # output moved is the one that bounds the reach.
        delta = (base - model(bumped)).abs()
        moved = delta.amax(dim=tuple(range(delta.dim() - 1))) if delta.dim() > 1 else delta
        changed = (moved > tol).nonzero().flatten()
        if changed.numel() == 0:
            return 0                      # nothing moved anywhere: constant output
        first = int(changed[0].item())
        if first == 0:
            return None                   # moved as far back as the window allows
        return max(0, p - first)
    finally:
        model.train(was_training)
