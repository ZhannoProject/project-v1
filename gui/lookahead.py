"""
The Lookahead tab: what a millisecond of future context is worth.

This project has measured the cost of causality exactly twice -- once as
``tcn_se`` (0.708 RRMSE_t) and once as ``tcn_se_bi`` (0.595) -- and called the
0.11 between them "the causality cost". That is a true number and a useless
one for building anything, because it prices an infinite amount of future
context against none at all, and no online system is offered that choice. It
is offered a latency budget: 30 ms, 50 ms, 125 ms. The question it has to ask
is how much of that 0.11 it can buy for its budget.

So this tab turns the two points into a curve. A rung of either ladder is
``tcn_se_la{N}`` or ``tcn_la{N}``, a model licensed to read N samples past the one it emits,
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

import hashlib
import json
import re
import sys
from pathlib import Path
from functools import partial
from types import SimpleNamespace
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gui.config_io import _checkpoint_info, list_runs
from models.lookahead import (LOOKAHEAD_LADDER, TCN_LOOKAHEAD_LADDER, SAMPLE_RATE_HZ, architectural, backward_taps,
                              capacity, forward_receptive_field, ms)
from models.receptive_field import DEFAULT_DILATIONS
from models.variants import (LOOKAHEAD_VARIANTS, TCN_LOOKAHEAD_VARIANTS, TCNOnly, TCNWithSE,
                             VARIANT_REGISTRY, build_model, pinned_kwargs, variant_is_causal)

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
LADDER_SEEDS = (42, 54, 67)

#: Measuring the whole ladder is two forward passes per rung, but a request
#: should still not be able to ask for thousands.
MAX_BUDGETS = 32
PLAIN_TCN_RECIPE = (SAMPLE_RATE_HZ, "fixed", tuple(DEFAULT_DILATIONS))


def _plain_tcn(row):
    variant = row.get("variant")
    factory = VARIANT_REGISTRY.get(variant)
    cls = factory.func if isinstance(factory, partial) else factory
    return (isinstance(cls, type) and issubclass(cls, TCNOnly)
            and not issubclass(cls, TCNWithSE)
            and not row.get("se_reduction") and not pinned_kwargs(variant).get("se_reduction"))


def _tcn_recipe(row):
    pins = pinned_kwargs(row["variant"])
    return (row.get("train_fs") or SAMPLE_RATE_HZ, row.get("input_normalization") or "epoch",
            tuple(row.get("dilations") or pins.get("dilations") or DEFAULT_DILATIONS))


def _tcn_family_id(recipe):
    if recipe == PLAIN_TCN_RECIPE:
        return "tcn"
    rate, normalization, dilations = recipe
    return f"tcn/{rate:g}/{normalization}/" + "-".join(map(str, dilations))


def _family(spec, records=None):
    family_id = spec.get("variant") or CAUSAL_VARIANT
    if family_id == CAUSAL_VARIANT:
        return _family_meta(CAUSAL_VARIANT)
    if family_id == "tcn" and records is None:
        return _family_meta("tcn")
    if isinstance(family_id, str):
        rows = list_runs() if records is None else records
        profiles = {**_se_families(rows), **_tcn_families(rows)}
        if family_id in profiles:
            return profiles[family_id]
    raise ValueError(f"Unknown lookahead architecture {family_id!r}")


def _hidden_channels(spec):
    raw = spec.get("hidden_channels")
    if raw in (None, ""):
        return DEFAULT_HIDDEN
    try:
        hidden = int(raw)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("hidden_channels must be a whole number") from None
    if isinstance(raw, bool) or (isinstance(raw, float) and raw != hidden):
        raise ValueError("hidden_channels must be a whole number")
    if not 1 <= hidden <= 1024:
        raise ValueError(f"hidden_channels out of range: {hidden}")
    return hidden


def _family_meta(variant, recipe=PLAIN_TCN_RECIPE, recorded_budgets=()):
    with_se = variant == CAUSAL_VARIANT
    rate, normalization, dilations = recipe
    conv = capacity(dilations)
    total = capacity(dilations, se=with_se)
    standard = dilations == tuple(DEFAULT_DILATIONS)
    can_train = standard and normalization in ("fixed", "epoch")
    train_budgets = LOOKAHEAD_LADDER if with_se else TCN_LOOKAHEAD_LADDER if can_train else ()
    if with_se:
        ladder = train_budgets
        label = "TCN + SE · all saved configurations"
    else:
        ladder = sorted({*train_budgets, conv // 2, conv, *recorded_budgets})
        label = f"TCN · ChannelNorm · {rate:g} Hz · {normalization} input scale"
        if not standard:
            label += f" · dilations {', '.join(map(str, dilations))}"
    return {
        "id": variant if with_se else _tcn_family_id(recipe),
        "label": label,
        "variant": variant,
        "sample_rate_hz": rate,
        "input_normalization": normalization,
        "dilations": list(dilations),
        "ladder": list(ladder),
        "train_budgets": [0, *train_budgets] if can_train else [],
        "capacity": {"conv": conv, "conv_and_se": total},
        "variants": {"causal": variant, "unbounded": UNBOUNDED_VARIANT if with_se else None,
                     "rungs": list(LOOKAHEAD_VARIANTS if with_se else TCN_LOOKAHEAD_VARIANTS)},
        "config": LADDER_CONFIG if with_se else "tcn_both.yaml",
        "shares": {str(n): share_label(n, total) for n in ladder},
    }


def _tcn_families(records):
    profiles = {PLAIN_TCN_RECIPE: set()}
    for row in records:
        if row.get("status") == "complete" and row.get("tcn_norm") == "channel" and _plain_tcn(row):
            recipe = _tcn_recipe(row)
            budgets = profiles.setdefault(recipe, set())
            rung = _rung_of(row)
            if rung and rung <= capacity(recipe[2]):
                budgets.add(rung)
    return {_tcn_family_id(recipe): _family_meta("tcn", recipe, budgets)
            for recipe, budgets in sorted(profiles.items())}


def _se_recipe(row):
    rate, normalization, dilations = _tcn_recipe(row)
    pins = pinned_kwargs(row["variant"])
    reduction = row.get("se_reduction") or pins.get("se_reduction") or 8
    return rate, normalization, dilations, row.get("tcn_norm"), reduction, row.get("se_window")


def _se_families(records):
    profiles = {(SAMPLE_RATE_HZ, "fixed", tuple(DEFAULT_DILATIONS), "channel", 8, None)}
    profiles.update(_se_recipe(row) for row in records
                    if row.get("status") == "complete" and _is_se_family(row)
                    and row.get("tcn_norm") in ("channel", "batch", "none"))
    families = {CAUSAL_VARIANT: _family_meta(CAUSAL_VARIANT)}
    for recipe in sorted(profiles, key=str):
        rate, normalization, dilations, norm, reduction, window = recipe
        family = _family_meta(CAUSAL_VARIANT, (rate, normalization, dilations))
        norm_label = {"channel": "ChannelNorm", "batch": "BatchNorm", "none": "No normalization"}[norm]
        family["id"] = f"tcn_se/{norm}/{rate:g}/{normalization}/r{reduction}/w{window or 0}/" + "-".join(map(str, dilations))
        family["label"] = f"TCN + SE · {norm_label} · {rate:g} Hz · {normalization} input scale"
        if reduction != 8:
            family["label"] += f" · SE reduction {reduction}"
        if dilations != tuple(DEFAULT_DILATIONS):
            family["label"] += " · dilations " + ", ".join(map(str, dilations))
        family.update(tcn_norm=norm, se_reduction=reduction, se_window=window)
        if norm == "batch" or window:
            family["train_budgets"] = []
        if window:
            family["label"] += f" · SE window {window}"
            family["ladder"] = []
            family["variants"] = {**family["variants"], "unbounded": None}
        elif dilations != tuple(DEFAULT_DILATIONS):
            family["ladder"] = [family["capacity"]["conv_and_se"]]
        families[family["id"]] = family
    return families


def _matches_family(run, family):
    if family["variant"] == "tcn":
        return _matches_tcn_family(run, family)
    if "tcn_norm" not in family:
        return True
    return (run["train_fs"] == family["sample_rate_hz"]
            and run["input_normalization"] == family["input_normalization"]
            and run["tcn_norm"] == family["tcn_norm"]
            and (run.get("se_reduction") or 8) == family["se_reduction"]
            and run.get("se_window") == family["se_window"])


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
    if variant in VARIANT_REGISTRY:
        pinned_budget = pinned_kwargs(variant).get("lookahead")
        if pinned_budget:
            return pinned_budget
        return 0 if variant_is_causal(variant) else None
    return None if row.get("bidirectional") else 0


def _is_se_family(row: dict) -> bool:
    """A run of the SE baseline, any rung of it, or its unbounded twin."""
    factory = VARIANT_REGISTRY.get(row.get("variant"))
    cls = factory.func if isinstance(factory, partial) else factory
    return isinstance(cls, type) and issubclass(cls, TCNWithSE)


def ladder_runs(artifact: str = "BOTH", family: str | dict = CAUSAL_VARIANT, *,
                records: list[dict] | None = None) -> list[dict]:
    """
    Runs for one family, tagged with the rung they sit on.

    Reuses the Runs tab's own row builder, so a run appears here with the same
    params, seed, status and metrics it shows there and the two cannot drift.
    Interrupted and stale runs are kept but marked: a stale row's metrics
    describe weights that no longer exist, so the page can refuse to plot it
    rather than put a point on the curve that nothing stands behind.
    """
    records = list_runs() if records is None else records
    if isinstance(family, str):
        family = _family({"variant": family}, records)
    rows = []
    plain = family["variant"] == "tcn"
    for row in records:
        belongs = _plain_tcn(row) if plain else _is_se_family(row)
        if not belongs:
            continue
        if artifact and row.get("artifact") and row["artifact"] != artifact:
            continue
        if family["variant"] == CAUSAL_VARIANT and "tcn_norm" in family:
            if _tcn_recipe(row)[2] != tuple(family["dilations"]):
                continue
        rung = _rung_of(row)
        if plain:
            if _tcn_recipe(row)[2] != tuple(family["dilations"]):
                continue
            if rung is None:
                # Centred convolution padding has a finite future budget.
                rung = family["capacity"]["conv"] // 2
        rows.append({
            "name": row["name"], "variant": row.get("variant"),
            "artifact": row.get("artifact"), "status": row.get("status"),
            "lookahead": rung, "latency_ms": ms(rung, row.get("train_fs") or SAMPLE_RATE_HZ),
            "unbounded": rung is None,
            "params": row.get("params"), "hidden_channels": row.get("hidden_channels"),
            "tcn_norm": row.get("tcn_norm"), "train_fs": row.get("train_fs"),
            "input_normalization": row.get("input_normalization"),
            "se_reduction": row.get("se_reduction") or (
                pinned_kwargs(row["variant"]).get("se_reduction") if not plain else None),
            "se_window": row.get("se_window"),
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


def _matches_tcn_family(run, family):
    return (run["tcn_norm"] == "channel"
            and run["train_fs"] == family["sample_rate_hz"]
            and run["input_normalization"] == family["input_normalization"])


def _comparison_recipe(run, family, info):
    from data.protocol import evaluation_data

    config = info.get("run_config") or {}
    arch = info.get("model_config") or {}
    causal = run["lookahead"] == 0
    if (not arch or config.get("output_delay") or config.get("teacher_ckpt")
            or arch.get("causal", causal) != causal):
        return None
    try:
        data = evaluation_data(SimpleNamespace(run_config=config),
                               seed=run["master_seed"], artifact=run["artifact"])
    except (TypeError, ValueError):
        return None
    data["target_fs"] = run["train_fs"]
    return {
        "data": data,
        "training": {key: config.get(key) for key in ("protocol", "batch_size", "data_dir")},
        "optimizer": info.get("optimizer"),
        "model": {key: arch.get(key) for key in ("dropout", "se_reduction", "se_window")},
        "dilations": tuple(arch.get("dilations") or family["dilations"]),
        **{key: run[key] for key in ("tcn_norm", "input_normalization", "params", "epochs", "loss_fn")},
    }


def _comparison_runs(runs, family, hidden):
    baselines, rungs = [], []
    for run in runs:
        seed = run["master_seed"]
        if (run["status"] != "complete" or run["hidden_channels"] != hidden or run["kd"]
                or not _matches_family(run, family)
                or type(seed) is not int or not run["artifact"]):
            continue
        info = _checkpoint_info(PROJECT_ROOT / "runs" / run["name"] / "best.pt")
        recipe = _comparison_recipe(run, family, info)
        if recipe is not None:
            (baselines if run["lookahead"] == 0 else rungs).append((run, recipe))
    return baselines, rungs


def _eegio_pairs(runs, family, hidden):
    """Find a completed causal run and a bounded rung with the same recipe."""
    baselines, rungs = _comparison_runs(runs, family, hidden)
    return [
        {"runs": [baseline["name"], rung["name"]], "variant": family["id"],
         "lookahead": rung["lookahead"], "master_seed": rung["master_seed"],
         "hidden_channels": hidden, "artifact": rung["artifact"],
         "target_fs": rung["train_fs"], "latency_ms": ms(rung["lookahead"], rung["train_fs"]),
         "tcn_norm": rung["tcn_norm"], "input_normalization": rung["input_normalization"]}
        for rung, recipe in rungs
        for baseline, baseline_recipe in baselines if recipe == baseline_recipe
    ]


def _eegio_groups(runs, family, hidden):
    """One matched recipe across seeds, with one checkpoint per seed/rung."""
    baselines, rungs = _comparison_runs(runs, family, hidden)
    grouped = {}
    for run, recipe in baselines:
        key = json.dumps(_without_seed(recipe), sort_keys=True)
        grouped.setdefault(key, []).append((run, recipe))
    groups = []
    for key, references in grouped.items():
        members, duplicates, baseline_map = [], [], {}
        for baseline, recipe in sorted(references, key=lambda item: (item[0]["master_seed"], item[0]["name"])):
            seed = baseline["master_seed"]
            if str(seed) in baseline_map:
                duplicates.append(baseline["name"])
                continue
            baseline_map[str(seed)] = baseline["name"]
            candidates = [baseline, *(run for run, actual in rungs
                                      if run["master_seed"] == seed and actual == recipe)]
            seen = set()
            for run in sorted(candidates, key=lambda r: (r["lookahead"] is None, r["lookahead"] or 0, r["name"])):
                budget = run["lookahead"]
                if budget in seen:
                    duplicates.append(run["name"])
                    continue
                seen.add(budget)
                members.append({"run": run["name"], "master_seed": seed, "lookahead": budget,
                                "baseline": baseline["name"]})
        # All baselines precede the rungs so each seed's count reference is ready.
        members.sort(key=lambda m: (m["lookahead"] != 0, m["master_seed"],
                                    m["lookahead"] is None, m["lookahead"] or 0))
        template = references[0][0]
        recipe = references[0][1]
        optimizer = recipe.get("optimizer") or {}
        optimizer_name = (optimizer.get("label") or optimizer.get("name", "")) if isinstance(optimizer, dict) else str(optimizer)
        label = f"{template['epochs']} epochs · {template['loss_fn'] or LADDER_LOSS}"
        if recipe["training"].get("batch_size"):
            label += f" · batch {recipe['training']['batch_size']}"
        if optimizer_name:
            label += f" · {optimizer_name}"
        budgets = sorted({m["lookahead"] for m in members}, key=lambda b: (b is None, b or 0))
        groups.append({"id": hashlib.sha256(key.encode()).hexdigest()[:16], "label": label,
                       "baseline": members[0]["run"], "baselines": baseline_map,
                       "runs": [m["run"] for m in members], "members": members,
                       "seeds": sorted({m["master_seed"] for m in members}),
                       "rungs": budgets, "duplicates": duplicates,
                       "coverage": [{"lookahead": b, "seeds": [m["master_seed"] for m in members if m["lookahead"] == b]}
                                    for b in budgets],
                       "target_fs": template["train_fs"], "tcn_norm": template["tcn_norm"],
                       "input_normalization": template["input_normalization"],
                       "variant": family["id"], "hidden_channels": hidden, "artifact": template["artifact"],
                       "settings": _without_seed(recipe)})
    # Name otherwise identical choices by their actual data/model settings.
    labels = [g["label"] for g in groups]
    for group in groups:
        if labels.count(group["label"]) > 1:
            data = group["settings"]["data"]
            group["label"] += (f" · SNR {data['snr_range_db']} · split {data['split_ratios']}"
                               f" · dropout {group['settings']['model'].get('dropout')}")
    return sorted(groups, key=lambda g: (-len(g["runs"]), -len(g["seeds"]), g["id"]))


def _rescale_flag(spec: dict) -> bool:
    value = spec.get("rescale_input", False)
    if type(value) is not bool:
        raise ValueError("rescale_input must be a boolean")
    return value


def _offline_zero_phase_flag(spec: dict, zero_phase: bool) -> bool:
    """Zero-phase filtering for the offline reference only; causal rungs keep the causal filter."""
    value = spec.get("offline_zero_phase", False)
    if type(value) is not bool:
        raise ValueError("offline_zero_phase must be a boolean")
    if value and zero_phase:
        raise ValueError("Every model is already zero-phase filtered; choose the causal filter "
                         "to give only the offline reference zero-phase input")
    return value


def eegio_spec(spec: dict) -> tuple[dict, dict]:
    """Resolve a baseline and matching rungs for one shared EEG-IO evaluation."""
    from gui.eegio_tab import DEFAULT_N_SUBJECTS

    records = list_runs()
    family, hidden = _family(spec, records), _hidden_channels(spec)
    runs = spec.get("runs")
    if runs is None:
        groups = _eegio_groups(ladder_runs(spec.get("artifact") or "BOTH", family, records=records), family, hidden)
        selected = next((g for g in groups if g["id"] == spec.get("comparison_id")), None)
        if selected is None:
            raise ValueError("Choose a model type and a matching training setup. Refresh Lookahead after training.")
        if not any(m["lookahead"] != 0 for m in selected["members"]):
            raise ValueError("This model type has no completed matching lookahead rungs yet.")
        n = spec.get("n_subjects", DEFAULT_N_SUBJECTS)
        zero_phase = spec.get("zero_phase", False)
        if type(n) is not int or not 1 <= n <= DEFAULT_N_SUBJECTS:
            raise ValueError(f"Choose between 1 and {DEFAULT_N_SUBJECTS} subjects")
        if type(zero_phase) is not bool:
            raise ValueError("zero_phase must be a boolean")
        comparison = {**selected, "schema_version": 2, "scope": "all_seeds_all_rungs",
                      "master_seed": None, "lookahead": None, "latency_ms": None,
                      "rungs": {m["run"]: m["lookahead"] for m in selected["members"]},
                      "baseline_by_run": {m["run"]: m["baseline"] for m in selected["members"]}}
        return {"runs": selected["runs"], "offline": True, "target_fs": selected["target_fs"],
                "n_subjects": n, "zero_phase": zero_phase,
                "rescale_input": _rescale_flag(spec),
                "offline_zero_phase": _offline_zero_phase_flag(spec, zero_phase)}, comparison
    if (not isinstance(runs, list) or len(runs) < 2
            or any(not isinstance(run, str) for run in runs) or len(set(runs)) != len(runs)):
        raise ValueError("Choose a causal baseline and at least one distinct lookahead rung")
    if family["id"] == "tcn":
        baseline = next((row for row in records if row["name"] == runs[0]), None)
        if baseline and baseline.get("tcn_norm") == "channel" and _plain_tcn(baseline):
            family = _family({"variant": _tcn_family_id(_tcn_recipe(baseline))}, records)
    runs_on_ladder = ladder_runs(spec.get("artifact") or "BOTH", family, records=records)
    pairs = _eegio_pairs(runs_on_ladder, family, hidden)
    matched = {p["runs"][1]: p for p in pairs if p["runs"][0] == runs[0]}
    if any(run not in matched for run in runs[1:]):
        raise ValueError("Choose completed lookahead runs with a matching causal baseline. Refresh Lookahead after training.")
    selected = [matched[run] for run in runs[1:]]
    budgets = [p["lookahead"] for p in selected]
    if len(set(budgets)) != len(budgets):
        raise ValueError("Choose one checkpoint per lookahead rung")
    comparison = selected[0]
    if len(selected) > 1:
        comparison = {**comparison, "runs": runs, "lookahead": None, "latency_ms": None,
                      "rungs": dict(zip(runs, [0, *budgets]))}
    n = spec.get("n_subjects", DEFAULT_N_SUBJECTS)
    if type(n) is not int or not 1 <= n <= DEFAULT_N_SUBJECTS:
        raise ValueError(f"Choose between 1 and {DEFAULT_N_SUBJECTS} subjects")
    zero_phase = spec.get("zero_phase", False)
    if type(zero_phase) is not bool:
        raise ValueError("zero_phase must be a boolean")
    return {"runs": runs, "offline": True, "target_fs": comparison["target_fs"],
            "n_subjects": n, "zero_phase": zero_phase,
            "rescale_input": _rescale_flag(spec),
            "offline_zero_phase": _offline_zero_phase_flag(spec, zero_phase)}, comparison


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


def _runs_on(budget, hidden: int, by_rung: dict, *, tcn_family=None) -> dict:
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
    if tcn_family:
        seeds_seen = {}
        for run in complete:
            name = run["name"]
            if why[name]:
                continue
            required_norm = tcn_family.get("tcn_norm", "channel")
            if run["tcn_norm"] != required_norm:
                label = "batchnorm" if run["tcn_norm"] == "batch" else run["tcn_norm"] or "unknown normalization"
                required = {"channel": "ChannelNorm", "batch": "BatchNorm", "none": "no normalization"}[required_norm]
                why[name] = f"{label}; this ladder uses {required}"
            elif not _matches_family(run, tcn_family):
                why[name] = (f"this ladder uses {tcn_family['sample_rate_hz']:g} Hz training "
                             f"and {tcn_family['input_normalization']} input normalization"
                             + (" with the selected SE settings" if tcn_family["variant"] == CAUSAL_VARIANT else ""))
            elif run["master_seed"] is not None:
                seed = run["master_seed"]
                if seed in seeds_seen:
                    why[name] = f"seed {seed} is already represented by {seeds_seen[seed]}"
                else:
                    seeds_seen[seed] = name
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
    records = list_runs()
    family = _family(spec, records)
    variant = family["variant"]
    plain = variant == "tcn"
    conv = family["capacity"]["conv"]
    total = family["capacity"]["conv_and_se"]
    # Missing means "the default"; 0 means someone typed 0, and quietly
    # turning that into 64 would report a measurement of a model nobody asked
    # for under a width they did not choose.
    hidden = _hidden_channels(spec)
    artifact = spec.get("artifact") or "BOTH"

    budgets = spec.get("budgets")
    if budgets is None:
        budgets = family["ladder"]
    else:
        if len(budgets) > MAX_BUDGETS:
            raise ValueError(f"at most {MAX_BUDGETS} budgets at a time, got {len(budgets)}")
        seen, cleaned = set(), []
        for b in budgets:
            if not isinstance(b, int) or isinstance(b, bool):
                raise ValueError(f"budget must be a whole number of samples, got {b!r}")
            if not 0 <= b <= total:
                raise ValueError(f"budget out of range: {b}")
            if b not in seen:
                seen.add(b)
                cleaned.append(b)
        budgets = cleaned

    runs = ladder_runs(artifact, family, records=records)
    by_rung: dict = {}
    for r in runs:
        by_rung.setdefault(r["lookahead"], []).append(r)

    def row_for(name, budget) -> dict:
        trainable = name in VARIANT_REGISTRY and budget in family["train_budgets"]
        registered = name in VARIANT_REGISTRY and family["dilations"] == list(DEFAULT_DILATIONS)
        if name == UNBOUNDED_VARIANT and registered and family.get("tcn_norm") != "batch":
            trainable = True
        kwargs = {"hidden_channels": hidden}
        if "tcn_norm" in family:
            kwargs.update(tcn_norm=family["tcn_norm"], se_reduction=family["se_reduction"],
                          se_window=family["se_window"])
        if registered:
            model = _fresh(build_model, name, **kwargs)
        else:
            from models.causal_config import configure_causal_blocks

            factory = TCNOnly if plain else TCNWithSE
            model = _fresh(factory, hidden_channels=hidden, causal=budget == 0,
                           lookahead=budget, dilations=family["dilations"],
                           **({"se_reduction": family["se_reduction"]} if "tcn_norm" in family else {}))
            if "tcn_norm" in family:
                configure_causal_blocks(model, family["tcn_norm"], family["se_window"])
            name = next((r["variant"] for r in by_rung.get(budget, [])
                         if r["status"] == "complete" and _matches_family(r, family)), None)
        licensed = architectural(model)
        measured = forward_receptive_field(model)
        past = backward_taps(model)
        return {
            "variant": name,
            "lookahead": budget,
            "unbounded": budget is None,
            "licensed": licensed,
            "latency_ms": ms(licensed, family["sample_rate_hz"]),
            "measured": measured,
            # What the budget cost: the convolution taps left behind t. Shown
            # beside what it bought, because they come out of the same fixed
            # window and a rung with none left is not on the same curve.
            "past": past,
            "ahead": conv - past,
            "window": conv,
            "share": share_label(budget, total),
            "one_sided": past == 0,
            "exercised": (None if not licensed or measured is None
                          else round(100.0 * measured / licensed, 1)),
            "params": sum(p.numel() for p in model.parameters()),
            "trainable": trainable,
            **({"note": "This architecture has no training template."}
               if not family["train_budgets"] else {"note": f"Add {budget} to "
                        + ("LOOKAHEAD_LADDER" if variant == CAUSAL_VARIANT else "TCN_LOOKAHEAD_LADDER")
                        + f" to register {variant}_la{budget} for training."} if not trainable else {}),
            **_runs_on(budget, hidden, by_rung,
                       tcn_family=family if plain or "tcn_norm" in family else None),
        }

    rows = [row_for(variant, 0)]
    for budget in budgets:
        if budget == 0:
            continue                                  # that is the causal row above
        rows.append(row_for(f"{variant}_la{budget}", budget))
    if family["variants"]["unbounded"]:
        rows.append(row_for(UNBOUNDED_VARIANT, None))

    return {
        "rows": rows,
        "hidden_channels": hidden,
        "artifact": artifact,
        **family,
        "runs": runs,
        "eegio_pairs": _eegio_pairs(runs, family, hidden),
        "eegio_groups": _eegio_groups(runs, family, hidden),
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

    family = _family(spec)
    architecture = family["variant"]
    hidden = _hidden_channels(spec)
    if budget == 0:
        variant = architecture
    else:
        variant = f"{architecture}_la{budget}"
    if budget not in family["train_budgets"] or variant not in VARIANT_REGISTRY:
        raise ValueError(f"there is no registered training rung {variant} for {family['label']}")

    base = f"runs/{variant}_{artifact.lower()}"
    explicit_se = architecture == CAUSAL_VARIANT and "tcn_norm" in family
    if architecture == "tcn" or explicit_se:
        rate = family["sample_rate_hz"]
        normalization = family["input_normalization"]
        norm = family.get("tcn_norm", "channel")
        scale_suffix = "" if normalization == "fixed" else f"_{normalization}"
        norm_suffix = {"channel": "channelnorm", "none": "nonorm", "batch": "batchnorm"}[norm]
        se_suffix = f"_r{family['se_reduction']}" if explicit_se and family["se_reduction"] != 8 else ""
        save_dir = f"{base}_{norm_suffix}_{rate:g}hz{scale_suffix}{se_suffix}_s{seed}"
    else:
        save_dir = base if seed == 42 else f"{base}_s{seed}"
    if hidden != DEFAULT_HIDDEN:
        save_dir += f"_h{hidden}"
    extra = [f"--variant {variant}", f"--artifact {artifact}",
             f"--master_seed {seed}", f"--save_dir {save_dir}"]
    if architecture == "tcn" or explicit_se:
        extra.extend((f"--tcn_norm {norm}", f"--normalization {normalization}", f"--target_fs {rate:g}"))
        if explicit_se:
            extra.append(f"--se_reduction {family['se_reduction']}")
    if hidden != DEFAULT_HIDDEN:
        extra.append(f"--hidden_channels {hidden}")
    if spec.get("max_epochs"):
        extra.append(f"--max_epochs {int(spec['max_epochs'])}")
    # Passed through to the trainer's own guard rather than worked around
    # here: the directory is cleared by the thing that writes it.
    if spec.get("overwrite"):
        extra.append("--overwrite")
    return {"config": family["config"], "extra": " ".join(extra),
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


def _without_seed(recipe):
    data = {key: value for key, value in recipe["data"].items()
            if key not in ("master_seed", "normalization_scale")}
    return {**recipe, "data": data}


def _batch_config(row, seed_recipe, family, variant, seed):
    from experiment.config import defaults, resolve_config

    info = _checkpoint_info(PROJECT_ROOT / "runs" / row["name"] / "best.pt")
    recorded = {**(info.get("run_config") or {}), **info["model_config"]}
    expected = seed_recipe["data"]
    cfg = {key: recorded[key] for key in defaults() if key in recorded
           and key not in ("save_dir", "causal", "lookahead")}
    cfg.update(variant=variant, master_seed=seed, artifact=row["artifact"],
               target_fs=row["train_fs"], tcn_norm=row["tcn_norm"],
               normalization=expected["normalization"],
               normalization_scale=expected["normalization_scale"],
               loss_fn=row["loss_fn"] or LADDER_LOSS, max_epochs=row["epochs"],
               overwrite=False)
    if cfg.get("se_window") and variant != family["variant"]:
        raise ValueError("This baseline uses finite SE pooling, which the architectural lookahead rungs do not support.")
    cfg = resolve_config(cfg)
    tag = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    stem = f"{variant}_{row['artifact'].lower()}_{row['train_fs']:g}hz_s{seed}_{tag}"
    path = Path("runs") / stem
    retry = 1
    while (PROJECT_ROOT / path).exists():
        retry += 1
        path = Path("runs") / f"{stem}_retry{retry}"
    cfg["save_dir"] = path.as_posix()
    return cfg


def missing_train_plan(spec: dict) -> dict:
    """Fill the selected recipe's comparison grid while retaining completed matches."""
    records = list_runs()
    family = _family(spec, records)
    hidden = _hidden_channels(spec)
    runs = ladder_runs(spec.get("artifact") or "BOTH", family, records=records)
    baselines, rungs = _comparison_runs(runs, family, hidden)
    baseline = spec.get("baseline")
    selected = next((item for item in baselines if item[0]["name"] == baseline), None)
    if selected is None:
        raise ValueError("Choose a completed causal baseline with a recorded training recipe.")
    template, recipe = selected
    architecture = family["variant"]
    budgets = {budget: f"{architecture}_la{budget}" if budget else architecture
               for budget in family["train_budgets"]}
    if not budgets:
        raise ValueError("This architecture has no Lookahead training template.")
    if family["variants"]["unbounded"]:
        budgets[None] = family["variants"]["unbounded"]

    recipe_family = _without_seed(recipe)
    matched = [(run, actual) for run, actual in baselines + rungs
               if _without_seed(actual) == recipe_family]
    pending = []
    for seed in LADDER_SEEDS:
        candidates = [(run, actual) for run, actual in matched if run["master_seed"] == seed]
        reference = selected if seed == template["master_seed"] else next(iter(candidates), None)
        if reference:
            seed_template, seed_recipe = reference
            completed = {run["lookahead"] for run, actual in candidates if actual == seed_recipe}
        else:
            seed_template = template
            seed_recipe = {**recipe, "data": {**recipe["data"], "normalization_scale": None}}
            completed = set()
        for budget, variant in budgets.items():
            if budget in completed:
                continue
            cfg = _batch_config(seed_template, seed_recipe, family, variant, seed)
            pending.append({"lookahead": budget, "master_seed": seed,
                            "save_dir": cfg["save_dir"], "config": cfg})
    return {"baseline": baseline, "seeds": list(LADDER_SEEDS), "runs": pending,
            "completed": len(budgets) * len(LADDER_SEEDS) - len(pending),
            "queue_key": json.dumps(recipe_family, sort_keys=True),
            "sample_rate_hz": template["train_fs"], "tcn_norm": template["tcn_norm"]}


def meta() -> dict:
    """Everything the tab needs before it asks for a measurement."""
    records = list_runs()
    se = _family_meta(CAUSAL_VARIANT)
    families = {**_se_families(records), **_tcn_families(records)}
    artifacts = ["BOTH", "EOG", "EMG"]
    baselines = {}
    for family in families.values():
        for run in ladder_runs("", family, records=records):
            width = run["hidden_channels"]
            if (run["status"] != "complete" or run["lookahead"] != 0 or run["kd"]
                    or run["artifact"] not in artifacts or type(width) is not int or not 1 <= width <= 1024
                    or not _matches_family(run, family)):
                continue
            baselines[run["name"]] = {"variant": family["id"], "artifact": run["artifact"],
                                      "hidden_channels": width}
    return {
        **se,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "families": families,
        "baselines": baselines,
        "hidden_channels": DEFAULT_HIDDEN,
        "artifacts": artifacts,
        "seeds": list(LADDER_SEEDS),
    }
