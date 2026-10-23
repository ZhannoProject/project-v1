"""
The receptive-field sweep tab: one TCN per rung of the ladder.

``models/receptive_field.py`` says what each rung is; this says how to train
the set of them and what came back. It is deliberately thin -- a sweep is a
list of ordinary training runs that differ in one argument, so it builds the
same commands the Train tab builds and reads the same ``runs/`` directories
the Runs tab reads. Nothing here knows how to train anything.

The one thing it does add is the x-axis. A point on this curve is only
meaningful next to its reach, so every row carries the receptive field read
off the trained model rather than parsed from its name -- a run called
``tcn_rf253`` whose weights say otherwise is reported as the weights say.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

from gui import config_io
from gui.config_io import PROJECT_ROOT, RUNS_DIR

#: Shared with the Train tab so a sweep run is an ordinary run.
TRAINER = [sys.executable, "-m", "trainer"]

MAX_EPOCHS = 2000

#: The width the sweep is read at. A rung's runs have to be the same model
#: apart from their reach, so a run of another width is reported beside the
#: curve rather than averaged into it.
DEFAULT_HIDDEN = 64

#: The recipe each family's ladder is trained from, one per family, so a rung
#: differs from the 125-sample runs already on disk in its reach and nothing
#: else. Not optional: the trainer's own fallbacks are l1_combined at lr 5e-3
#: for 150 epochs, where every baseline here is mse at 5e-4 for 200, so a rung
#: launched without a config would differ from the point it is being compared
#: against in the loss, the learning rate and the length at once -- and would
#: look merely worse.
LADDER_CONFIGS = {"tcn": "configs/tcn_both.yaml", "tcn_se": "configs/tcn_se_both.yaml"}

#: The loss the ladder is trained with, from the configs above. A run trained
#: with another one cannot go on the curve: its val loss is not even on the
#: same scale, and the project has a run of exactly that kind on the 125 rung.
LADDER_LOSS = "mse"

#: The reach of a run that recorded no dilation schedule. Written out rather
#: than read from ``DEFAULT_DILATIONS`` on purpose: this is a historical fact
#: about runs already on disk, and moving the default must not move them.
LEGACY_FIELD = 125

#: Variants whose schedule was fixed before the sweep existed, so a run of one
#: that recorded no architecture can still be placed. Only the two causal
#: family heads: the ``_bi`` twins are non-causal teachers and have no reach to
#: report, and a name carrying a reach it did not record is not evidence.
LEGACY_VARIANTS = frozenset({"tcn", "tcn_se"})


def ladder(variant: str = "tcn", hidden_channels: int = 64) -> List[dict]:
    """Every rung: its schedule, reach, span in ms, and parameter cost."""
    from models.receptive_field import rung_table
    return rung_table(hidden_channels=hidden_channels, variant=variant)


def run_name(variant: str, artifact: str, seed: int, rf: int) -> str:
    """
    Where a rung's run lands.

    Named for the reach, not the rung index, so the directory says what the
    point on the curve is. The seed suffix follows the trainer's own rule:
    the default seed keeps the bare name.
    """
    from trainer.__main__ import DEFAULT_MASTER_SEED
    base = f"{variant}_rf{rf}_{artifact.lower()}"
    return base if seed == DEFAULT_MASTER_SEED else f"{base}_s{seed}"


def build_steps(spec: dict) -> List[dict]:
    """
    One training command per selected rung, in order of increasing reach.

    Shortest first on purpose: they are the quickest to train, so a mistake in
    the shared settings shows up in a couple of minutes rather than after the
    longest run of the set.

    Each command is the family's own recipe with four things overridden -- the
    variant, the artifact, the seed and the output directory -- so the only
    thing that varies down the ladder is the one the sweep is about.
    """
    variant = (spec.get("variant") or "tcn").strip()
    if variant not in ("tcn", "tcn_se"):
        raise ValueError(f"the sweep is for the TCN families; {variant!r} is not one of them")

    artifact = (spec.get("artifact") or "BOTH").strip().upper()
    if artifact not in config_io.ARTIFACTS:
        raise ValueError(f"unknown artifact {artifact!r}")

    try:
        seed = int(spec.get("master_seed", 42))
        epochs = int(spec.get("max_epochs", 200))
    except (TypeError, ValueError):
        raise ValueError("master_seed and max_epochs must be whole numbers") from None
    if not 1 <= epochs <= MAX_EPOCHS:
        raise ValueError(f"max_epochs must be between 1 and {MAX_EPOCHS}")

    rows = {r["receptive_field"]: r for r in ladder(variant)}
    wanted = spec.get("rungs")
    if wanted is None:
        chosen = sorted(rows)
    else:
        chosen = sorted({int(r) for r in wanted})
        unknown = [r for r in chosen if r not in rows]
        if unknown:
            raise ValueError(f"not rungs of this ladder: {unknown}; "
                             f"the ladder is {sorted(rows)}")
    if not chosen:
        raise ValueError("pick at least one rung")

    extra = (spec.get("extra_args") or "").strip()
    steps = []
    for rf in chosen:
        name = run_name(variant, artifact, seed, rf)
        cmd = [*TRAINER, "--config", LADDER_CONFIGS[variant],
               "--variant", f"{variant}_rf{rf}",
               "--artifact", artifact, "--master_seed", str(seed),
               "--max_epochs", str(epochs), "--save_dir", f"runs/{name}"]
        if spec.get("overwrite"):
            cmd.append("--overwrite")
        if extra:
            import shlex
            cmd += shlex.split(extra)
        steps.append({
            "label": f"receptive field {rf} samples ({rows[rf]['ms']:.0f} ms) → runs/{name}",
            "command": cmd,
            "note": f"dilations {rows[rf]['dilations']} · {rows[rf]['params']:,} params",
        })
    return steps


def _run_rf(run_dir: Path, variant: Optional[str] = None) -> Optional[int]:
    """
    The reach of a finished run, from what it recorded about itself.

    The architecture the trainer wrote into history.json comes first: a run
    can be named anything, and the point's position on the x-axis has to come
    from the model that produced it.

    A run that recorded no schedule is placed only when its variant has never
    had more than one.  That is what puts the project's existing ``tcn`` and
    ``tcn_se`` runs on the 125 rung instead of leaving the sweep to re-measure
    a point it already has: the dilations were a constant until this sweep
    made them an argument, so those runs were all trained at that reach, and
    they are the comparison the sweep exists to extend.  Anything else -- a
    variant whose name states a reach but whose history is silent, or a
    variant off these two families -- is left off the curve.
    """
    from models.receptive_field import field

    hist = run_dir / "history.json"
    if not hist.is_file():
        return None
    try:
        cfg = (json.loads(hist.read_text()) or {}).get("config") or {}
    except (OSError, ValueError):
        return None
    dil = (cfg.get("arch") or {}).get("dilations")
    if dil:
        return field(dil)
    name = variant or (cfg.get("arch") or {}).get("variant") or cfg.get("variant")
    if name in LEGACY_VARIANTS:
        return LEGACY_FIELD
    return None


def results(variant: str = "tcn", artifact: str = "BOTH", seed: Optional[int] = None,
            hidden_channels: int = DEFAULT_HIDDEN) -> List[dict]:
    """
    The curve so far: one row per *rung*, averaged over the seeds on it.

    Reads ``runs/`` directly rather than keeping its own record, so a run
    trained from the Train tab or the command line counts the same as one the
    sweep launched. Which is exactly why a rung can arrive with five runs on
    it -- the 125 rung has every ``tcn`` and ``tcn_se`` run the project has
    ever done -- and why they cannot simply be plotted or the last one kept:

    * a run joins the average only if its width matches, because
      ``runs/tcn_se_both_s67`` holds a 48-channel model beside two 64-channel
      ones, and averaging the three once made a three-seed error bar out of
      two seeds and a width experiment. Excluded runs come back under
      ``mismatched`` with the reason, since a run that exists and is left out
      is something to show, not to lose;
    * a stale or interrupted run is in neither: its metrics describe weights
      that are gone.

    The spread is over seeds at one reach, which is the only quantity that
    makes a difference between two rungs readable.
    """
    from gui.lookahead import _seed_stats, why_excluded   # one definition, both tabs
    from models.receptive_field import ms

    by_rung: dict = {}
    for r in config_io.list_runs():
        rf = _run_rf(RUNS_DIR / r["name"], r.get("variant"))
        if rf is None or r["val_loss"] is None or r.get("status") != "complete":
            continue
        if r.get("variant") not in (variant, f"{variant}_rf{rf}"):
            continue
        if artifact and r.get("artifact") != artifact:
            continue
        if seed is not None and r.get("master_seed") not in (None, seed):
            continue
        by_rung.setdefault(rf, []).append(r)

    rows = []
    for rf, runs in sorted(by_rung.items()):
        excluded = {r["name"]: why_excluded(r, hidden_channels, LADDER_LOSS) for r in runs}
        mine = [r for r in runs if not excluded[r["name"]]]
        others = [r for r in runs if excluded[r["name"]]]
        if not mine:
            continue
        widths = {r.get("params") for r in mine}
        rows.append({
            "receptive_field": rf, "ms": ms(rf),
            # Only when the runs agree: a number that is true of some of them
            # would be read as the rung's cost.
            "params": widths.pop() if len(widths) == 1 else None,
            "n_runs": len(mine),
            "seeds": sorted({r["master_seed"] for r in mine if r["master_seed"] is not None}),
            "runs": [r["name"] for r in mine],
            "mismatched": [{"name": r["name"], "why": excluded[r["name"]]} for r in others],
            "val_loss": _seed_stats([r["val_loss"] for r in mine]),
            "rrmse_t": _seed_stats([r.get("rrmse_t") for r in mine]),
            "rrmse_s": _seed_stats([r.get("rrmse_s") for r in mine]),
            "acc": _seed_stats([r.get("acc") for r in mine]),
        })
    return rows


def meta() -> dict:
    """Everything the tab needs to draw itself before anything is trained."""
    return {
        "ladder": {v: ladder(v) for v in ("tcn", "tcn_se")},
        "artifacts": list(config_io.ARTIFACTS),
        "defaults": {"variant": "tcn", "artifact": "BOTH",
                     "master_seed": 42, "max_epochs": 200},
    }
