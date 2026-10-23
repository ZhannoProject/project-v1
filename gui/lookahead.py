"""
The Lookahead tab: what a millisecond of future context is worth.

This project has measured the cost of causality exactly twice -- once as
``tcn_se`` (0.708 RRMSE_t) and once as ``tcn_se_bi`` (0.595) -- and called the
0.11 between them "the causality cost". That is a true number and a useless
one for building anything, because it prices an infinite amount of future
context against none at all, and no online system is offered that choice. It
is offered a latency budget: 30 ms, 50 ms, 125 ms. The question it has to ask
is how much of that 0.11 it can buy for its budget.

So this tab turns the two points into a curve. A rung of the ladder is
``tcn_se_la{N}``, a model licensed to read N samples past the one it emits,
which costs exactly N/fs of output delay -- 3.9 ms a sample at 256 Hz. The x
axis is milliseconds of latency, the same axis the Deploy tab's hop budgets
live on, so the two halves of the question (what does the model cost to run,
what does it cost to wait for) are finally in the same units.

What this module does, and does not do
--------------------------------------
Measuring a rung needs no training: the licensed reach is a property of the
graph and the realised reach is a perturbation probe, so the ladder's whole x
axis is available in under a second, before any run exists. That is the part
that is instant, and it is deliberately separated from the part that is not
-- a rung's *accuracy* needs a 200-epoch run, and the tab reports which rungs
have one rather than pretending the curve is complete.

The two numbers per rung are both shown, and they are different things:
``licensed`` is what the graph may read, hence the latency that must be paid;
``measured`` is what the weights actually reach. See models/lookahead.py for
why the second can fall short of the first (a dead ReLU in an SE gate), and
why neither is the one true answer.
"""

import re
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui.config_io import list_runs
from models.lookahead import (LOOKAHEAD_LADDER, SAMPLE_RATE_HZ, architectural, backward_taps,
                              capacity, forward_receptive_field, ms)
from models.variants import (LOOKAHEAD_VARIANTS, VARIANT_REGISTRY, build_model, pinned_kwargs,
                             variant_is_causal)

#: The loss that config trains with. A run on another one cannot join a rung's
#: average: its val loss is not even on the same scale.
LADDER_LOSS = "mse"

#: The config the ladder is trained from. Every rung overrides only --variant
#: and --save_dir on top of it, so a rung differs from the causal baseline in
#: the one thing the experiment is about.
LADDER_CONFIG = "tcn_se_both.yaml"

#: The causal and unbounded ends of the ladder. They are ordinary variants with
#: runs already trained, not lookahead rungs, which is why they are named here
#: rather than derived: the ladder is bounded by two things that existed first.
CAUSAL_VARIANT = "tcn_se"
UNBOUNDED_VARIANT = "tcn_se_bi"

#: Width every rung is built and measured at: the one the SE repeats used.
DEFAULT_HIDDEN = 64

#: Measuring the whole ladder is two forward passes per rung, but a request
#: should still not be able to ask for thousands.
MAX_BUDGETS = 32


def _rung_of(row: dict):
    """
    Which rung of the ladder a finished run sits on.

    0 for a causal run, an integer for a bounded one, ``None`` for an
    unbounded (``*_bi``) one. Read from the variant the run recorded, not from
    its directory name: a name is a convention and the registry is not.
    """
    if row.get("lookahead"):
        return row["lookahead"]                # list_runs already resolved it
    variant = row.get("variant") or ""
    if variant in LOOKAHEAD_VARIANTS:
        return pinned_kwargs(variant)["lookahead"]
    if variant in VARIANT_REGISTRY:
        return 0 if variant_is_causal(variant) else None
    return None if row.get("bidirectional") else 0


def _is_se_family(row: dict) -> bool:
    """A run of the SE baseline, any rung of it, or its unbounded twin."""
    variant = row.get("variant") or ""
    return variant == CAUSAL_VARIANT or variant == UNBOUNDED_VARIANT \
        or variant in LOOKAHEAD_VARIANTS


def ladder_runs(artifact: str = "BOTH") -> list[dict]:
    """
    Every finished SE-family run, tagged with the rung it sits on.

    Reuses the Runs tab's own row builder, so a run appears here with the same
    params, seed, status and metrics it shows there and the two cannot drift.
    Interrupted and stale runs are kept but marked: a stale row's metrics
    describe weights that no longer exist, so the page can refuse to plot it
    rather than put a point on the curve that nothing stands behind.
    """
    rows = []
    for row in list_runs():
        if not _is_se_family(row):
            continue
        if artifact and row.get("artifact") and row["artifact"] != artifact:
            continue
        rung = _rung_of(row)
        rows.append({
            "name": row["name"], "variant": row.get("variant"),
            "artifact": row.get("artifact"), "status": row.get("status"),
            "lookahead": rung, "latency_ms": ms(rung),
            "unbounded": rung is None,
            "params": row.get("params"), "hidden_channels": row.get("hidden_channels"),
            "master_seed": row.get("master_seed"), "epochs": row.get("epochs"),
            # Carried because a rung's membership turns on it: a distilled run
            # is not a seed repeat of the run it was distilled into.
            "kd": row.get("kd"),
            "val_loss": row.get("val_loss"), "loss_fn": row.get("loss_fn"),
            "rrmse_t": row.get("rrmse_t"), "rrmse_s": row.get("rrmse_s"),
            "acc": row.get("acc"),
        })
    rows.sort(key=lambda r: (r["lookahead"] is None, r["lookahead"] or 0, r["name"]))
    return rows


def _seed_stats(values: list[float]) -> dict:
    """Mean and spread over the seed repeats of one rung."""
    clean = [v for v in values if isinstance(v, (int, float))]
    if not clean:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    n = len(clean)
    mean = sum(clean) / n
    sd = None
    if n > 1:
        sd = (sum((v - mean) ** 2 for v in clean) / (n - 1)) ** 0.5
    return {"n": n, "mean": mean, "sd": sd, "min": min(clean), "max": max(clean)}


#: Weight seed every measured model is built at. The realised reach depends on
#: the draw -- a dead ReLU in an SE gate drops a licensed dependence to exactly
#: zero -- so an unseeded build makes the ``measured`` column flicker between
#: page loads, which reads as instrument noise rather than the property it is.
#: Pinned so the same rung measures the same number twice.
PROBE_SEED = 0


def _fresh(factory, *args, **kwargs):
    """A model at a fixed weight draw, in eval mode, for measuring."""
    import torch

    torch.manual_seed(PROBE_SEED)
    return factory(*args, **kwargs).eval()


def share_label(budget: Optional[int], total: int) -> str:
    """
    A budget as a share of the licence, which is what the number is.

    ``tcn_se_la186`` reads as 186 of something, and the obvious something --
    the convolution window -- is 124 taps, so the name invites an arithmetic
    that does not work out. The denominator is capacity: 124 taps the
    convolutions can move plus 62 the SE gates can peek, which the gates have
    and the window does not. As a share it comes out exactly, and the top
    three rungs are the thirds: 62, 124 and 186 are 1/3, 2/3 and all of it.

    Simple fractions are named as fractions and the rest as a percentage,
    because "3/124 of the licence" is not an improvement on "la4". The full
    budget is "1/1" rather than "all", so that it reads as the last of the
    thirds in a column that already holds 1/3 and 2/3.

    One short token, because it goes in two places -- the rung picker and the
    rung column -- and they have to agree.
    """
    from fractions import Fraction

    if not budget or not total:
        return ""
    share = Fraction(int(budget), int(total))
    if share >= 1:
        return "1/1"
    return (f"{share.numerator}/{share.denominator}" if share.denominator <= 4
            else f"{100.0 * share:.0f}%" if share >= Fraction(1, 50)
            else f"{100.0 * share:.1f}%")


def why_excluded(run: dict, hidden_channels: int, loss_fn: str = LADDER_LOSS) -> Optional[str]:
    """
    Why a run on the right rung still cannot join its average, or None.

    A point on either ladder means "the same model, trained the same way,
    allowed to see further" -- further ahead here, further back on the
    receptive-field sweep -- so both tabs ask this one question. Everything
    below is a way that quietly stops being true, and all three are on disk:

    * a width that differs -- ``runs/tcn_se_both_s67`` is 48 channels beside
      two 64-channel companions, which once made a three-seed error bar out of
      two seeds and a width experiment, invisible for a fortnight;
    * a distilled run -- ``runs/tcn_se_both_kd_s67`` had a bidirectional
      teacher. That is a different training signal and an ablation row of its
      own, not a seed repeat, and it sat in the causal anchor this whole
      ladder is priced against: it moved the anchor from 0.7104 to 0.7118 and
      widened its spread from 0.0020 to 0.0033, so every Delta in the table
      was measured from a point that was part KD;
    * a different loss -- not comparable at all on val loss, since the numbers
      are not on one scale.

    The reason is returned rather than a boolean so a tab can name what it
    left out. A run that exists and is excluded is something to show.
    """
    if run.get("hidden_channels") not in (None, hidden_channels):
        return f"{run.get('hidden_channels')} channels, not {hidden_channels}"
    if run.get("kd"):
        return "distilled from a teacher, not trained like the rest of the ladder"
    if run.get("loss_fn") not in (None, loss_fn):
        return f"trained with {run.get('loss_fn')}, not {loss_fn}"
    return None


def _runs_on(budget, hidden: int, by_rung: dict) -> dict:
    """
    The runs that may be averaged on one rung, and the ones that may not.

    A rung's error bar is only an error bar over seeds if the runs on it are
    the same model trained the same way. They are not automatically, and
    ``why_excluded`` above lists the three ways they stop being. The rest are
    returned alongside under ``mismatched`` rather than dropped -- a run that
    exists and is excluded is something the reader should see, not something
    to quietly lose.

    A stale or interrupted run is excluded from both: its metrics describe
    weights that are gone, so there is nothing to put on a curve.
    """
    complete = [r for r in by_rung.get(budget, []) if r["status"] == "complete"]
    why = {r["name"]: why_excluded(r, hidden) for r in complete}
    mine = [r for r in complete if not why[r["name"]]]
    others = [r for r in complete if why[r["name"]]]
    return {
        "runs": mine,
        "n_runs": len(mine),
        "mismatched": [dict(r, why=why[r["name"]]) for r in others],
        "seeds": sorted({r["master_seed"] for r in mine if r["master_seed"] is not None}),
        "rrmse_t": _seed_stats([r["rrmse_t"] for r in mine]),
        "rrmse_s": _seed_stats([r["rrmse_s"] for r in mine]),
        "acc": _seed_stats([r["acc"] for r in mine]),
    }


def measure(spec: dict | None = None) -> dict:
    """
    The ladder's x axis, measured on freshly built models.

    One row per rung: what it is licensed to read, what that costs in latency,
    what its weights actually reach, and which trained runs sit on it. No
    checkpoint is loaded and nothing is trained -- the reach is a property of
    the architecture, so this answers before any run of the ladder exists.

    ``budgets`` overrides the default ladder, for a rung that has no registry
    entry: measuring one needs no name, because nothing is saved that a name
    could then misdescribe. Training one does, which is why the rows say
    whether a rung is trainable.
    """
    spec = spec or {}
    # Missing means "the default"; 0 means someone typed 0, and quietly
    # turning that into 64 would report a measurement of a model nobody asked
    # for under a width they did not choose.
    raw_width = spec.get("hidden_channels")
    hidden = DEFAULT_HIDDEN if raw_width in (None, "") else int(raw_width)
    if not 1 <= hidden <= 1024:
        raise ValueError(f"hidden_channels out of range: {hidden}")
    artifact = spec.get("artifact") or "BOTH"

    budgets = spec.get("budgets")
    if budgets is None:
        budgets = list(LOOKAHEAD_LADDER)
    else:
        if len(budgets) > MAX_BUDGETS:
            raise ValueError(f"at most {MAX_BUDGETS} budgets at a time, got {len(budgets)}")
        seen, cleaned = set(), []
        for b in budgets:
            if not isinstance(b, int) or isinstance(b, bool):
                raise ValueError(f"budget must be a whole number of samples, got {b!r}")
            if not 0 <= b <= capacity([1, 2, 4, 8, 16], se=True):
                raise ValueError(f"budget out of range: {b}")
            if b not in seen:
                seen.add(b)
                cleaned.append(b)
        budgets = cleaned

    runs = ladder_runs(artifact)
    by_rung: dict = {}
    for r in runs:
        by_rung.setdefault(r["lookahead"], []).append(r)

    conv = capacity([1, 2, 4, 8, 16])

    def row_for(variant: str, budget, *, trainable: bool) -> dict:
        model = _fresh(build_model, variant, hidden_channels=hidden)
        licensed = architectural(model)
        measured = forward_receptive_field(model)
        past = backward_taps(model)
        return {
            "variant": variant,
            "lookahead": budget,
            "unbounded": budget is None,
            "licensed": licensed,
            "latency_ms": ms(licensed),
            "measured": measured,
            # What the budget cost: the convolution taps left behind t. Shown
            # beside what it bought, because they come out of the same fixed
            # window and a rung with none left is not on the same curve.
            "past": past,
            "ahead": conv - past,
            "window": conv,
            "share": share_label(budget, capacity([1, 2, 4, 8, 16], se=True)),
            "one_sided": past == 0,
            "exercised": (None if not licensed or measured is None
                          else round(100.0 * measured / licensed, 1)),
            "params": sum(p.numel() for p in model.parameters()),
            "trainable": trainable,
            **_runs_on(budget, hidden, by_rung),
        }

    rows = [row_for(CAUSAL_VARIANT, 0, trainable=True)]
    for budget in budgets:
        if budget == 0:
            continue                                  # that is the causal row above
        name = f"tcn_se_la{budget}"
        if name in VARIANT_REGISTRY:
            rows.append(row_for(name, budget, trainable=True))
        else:
            rows.append(_measured_only(budget, hidden, by_rung))
    rows.append(row_for(UNBOUNDED_VARIANT, None, trainable=True))

    return {
        "rows": rows,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "hidden_channels": hidden,
        "artifact": artifact,
        "ladder": list(LOOKAHEAD_LADDER),
        "capacity": {"conv": capacity([1, 2, 4, 8, 16]),
                     "conv_and_se": capacity([1, 2, 4, 8, 16], se=True)},
        "config": LADDER_CONFIG,
        "runs": runs,
    }


def _measured_only(budget: int, hidden: int, by_rung: dict) -> dict:
    """
    A rung with no registry entry: measurable, not trainable.

    Built directly rather than through the registry, because there is no
    variant name for it -- and that is exactly why it cannot be trained here.
    A checkpoint needs a name, and a name that did not state this budget would
    be a run that misdescribes itself.
    """
    from models.variants import TCNWithSE

    model = _fresh(TCNWithSE, hidden_channels=hidden, causal=False, lookahead=budget)
    licensed = architectural(model)
    measured = forward_receptive_field(model)
    past = backward_taps(model)
    return {
        "variant": None, "lookahead": budget, "unbounded": False,
        "licensed": licensed, "latency_ms": ms(licensed), "measured": measured,
        "past": past, "ahead": capacity([1, 2, 4, 8, 16]) - past,
        "window": capacity([1, 2, 4, 8, 16]),
        "share": share_label(budget, capacity([1, 2, 4, 8, 16], se=True)),
        "one_sided": past == 0,
        "exercised": (None if not licensed or measured is None
                      else round(100.0 * measured / licensed, 1)),
        "params": sum(p.numel() for p in model.parameters()),
        "trainable": False,
        "note": (f"no tcn_se_la{budget} in the registry: measurable, but a run needs a "
                 f"name that states its budget. Add {budget} to LOOKAHEAD_LADDER in "
                 f"models/lookahead.py to train it."),
        **_runs_on(budget, hidden, by_rung),
    }


def train_spec(spec: dict) -> dict:
    """
    What training one rung would run, and where it would land.

    Returns the config and the extra flags for the Train tab's own runner, so
    a rung is launched by exactly the path every other run here takes --
    ``python -m trainer --config ... --variant ... --save_dir ...`` -- rather
    than by a second training entry point that could drift from it.

    The directory carries the rung and the seed, because both are part of what
    the run is: ``runs/tcn_se_la32_both_s67``. The trainer's own overwrite
    guard then does the rest of the work.
    """
    budget = spec.get("lookahead")
    if budget is None:
        raise ValueError("pick a rung first")
    if not isinstance(budget, int) or isinstance(budget, bool):
        raise ValueError(f"lookahead must be a whole number of samples, got {budget!r}")

    seed = int(spec.get("master_seed") or 42)
    if not 0 <= seed < 2 ** 31:
        raise ValueError(f"master_seed out of range: {seed}")
    artifact = (spec.get("artifact") or "BOTH").upper()
    if artifact not in ("EOG", "EMG", "BOTH"):
        raise ValueError(f"unknown artifact {artifact!r}")

    if budget == 0:
        variant = CAUSAL_VARIANT
    else:
        variant = f"tcn_se_la{budget}"
        if variant not in VARIANT_REGISTRY:
            raise ValueError(
                f"there is no {variant}: a trained run needs a variant name that states its "
                f"budget, or the checkpoint misdescribes itself. Add {budget} to "
                f"LOOKAHEAD_LADDER in models/lookahead.py first.")

    base = f"runs/{variant}_{artifact.lower()}"
    save_dir = base if seed == 42 else f"{base}_s{seed}"
    extra = [f"--variant {variant}", f"--artifact {artifact}",
             f"--master_seed {seed}", f"--save_dir {save_dir}"]
    if spec.get("max_epochs"):
        extra.append(f"--max_epochs {int(spec['max_epochs'])}")
    # Passed through to the trainer's own guard rather than worked around
    # here: the directory is cleared by the thing that writes it.
    if spec.get("overwrite"):
        extra.append("--overwrite")
    return {"config": LADDER_CONFIG, "extra": " ".join(extra),
            "variant": variant, "save_dir": save_dir, "master_seed": seed,
            "exists": (PROJECT_ROOT / save_dir / "best.pt").is_file()}


def train_plan(spec: dict) -> dict:
    """
    The same rung at every seed asked for: what would run, and what already has.

    A rung trained once is a point with no spread -- the chart draws it hollow
    to say the error bar there is unknown, not zero -- so the useful unit of
    work on this ladder is a rung at three seeds, not a rung. Asking for them
    one at a time made that three sittings with a person waiting between each,
    which is exactly what the Train tab stopped requiring when it learned to
    hold several runs.

    Every seed goes through ``train_spec``, so a rung launched as one of three
    is the same command as a rung launched alone, and the caller is told which
    directories are already taken rather than finding out from the trainer's
    guard one run at a time.
    """
    raw = spec.get("master_seeds")
    if raw is None:
        raw = spec.get("master_seed", 42)
    if isinstance(raw, (int, str)) or raw is None:
        raw = [raw]

    seeds: List[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"master_seed must be a whole number, got {value!r}")
        if value not in seeds:
            seeds.append(value)
    if not seeds:
        raise ValueError("pick at least one seed")

    runs = []
    for seed in sorted(seeds):
        run = train_spec({**spec, "master_seed": seed, "overwrite": False})
        # --overwrite only on the directories that have something in them: a
        # blanket flag would also switch off the guard for the seeds that did
        # not need it, which is where a mistyped seed quietly lands.
        if spec.get("overwrite") and run["exists"]:
            run = train_spec({**spec, "master_seed": seed, "overwrite": True})
        runs.append(run)
    return {
        "config": runs[0]["config"],
        "runs": runs,
        "seeds": [r["master_seed"] for r in runs],
        "lookahead": spec.get("lookahead"),
        # Split out so the page can say which seeds it would actually train
        # and leave the rest alone, instead of refusing the whole request
        # because one of them has been done before.
        "taken": [r["save_dir"] for r in runs if r["exists"]],
        "fresh": [r["master_seed"] for r in runs if not r["exists"]],
    }


def meta() -> dict:
    """Everything the tab needs before it asks for a measurement."""
    return {
        "ladder": list(LOOKAHEAD_LADDER),
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "capacity": {"conv": capacity([1, 2, 4, 8, 16]),
                     "conv_and_se": capacity([1, 2, 4, 8, 16], se=True)},
        "variants": {"causal": CAUSAL_VARIANT, "unbounded": UNBOUNDED_VARIANT,
                     "rungs": list(LOOKAHEAD_VARIANTS)},
        "config": LADDER_CONFIG,
        "hidden_channels": DEFAULT_HIDDEN,
        "artifacts": ["BOTH", "EOG", "EMG"],
        "seeds": [42, 54, 67],
        # So the picker can say what 186 is 186 *of*, where the question
        # actually comes up.
        "shares": {str(n): share_label(n, capacity([1, 2, 4, 8, 16], se=True))
                   for n in LOOKAHEAD_LADDER},
    }
