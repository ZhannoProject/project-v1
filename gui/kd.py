"""
Knowledge distillation, audited: is the KD wired correctly, and if it is,
why did it not help?

This project has one distilled run, ``runs/tcn_se_both_kd_s67``, and it is
**worse** than the plain student trained at the same seed and width
(RRMSE_t 0.7161 against 0.7110).  A negative result like that has two
explanations and they need separating before either is reported: the KD was
broken, or the KD worked and distillation from this teacher genuinely costs
accuracy.  The Evaluate tab cannot tell them apart, because both look like a
student with a worse number.

So this tab asks two questions, in order.

**Was it wired correctly?**  Six checks, each a property that has to hold or
the run means nothing: the teacher loads under ``strict=True``, it is frozen
and in eval, it is the non-causal twin it was supposed to be, it reproduces
the metrics its own run recorded, the student and the teacher were scored on
the same split, and the student's ``best.pt`` was selected against the clean
signal rather than against the teacher.  They are cheap, and five of them
are properties of ``trainer/distillation.py`` rather than of any particular
run -- which is the point: a check that passes trivially today is a check
that fails loudly if someone edits that file.

**If it was, where did the accuracy go?**  Expand the student's error about
the teacher:

    ||S-G||^2  =  ||S-T||^2  +  ||T-G||^2  +  2<S-T, T-G>

``||T-G||^2`` is the teacher's own error and is the same constant for every
student, so a difference between two students is entirely in the other two.
``||S-T||^2`` is what KD optimises -- if it fell, the distillation did
something.  The cross term is what decides whether that was worth having: it
is negative when a student's deviation from the teacher *cancels* part of
the teacher's error, and KD, by pulling the student onto the teacher, gives
that cancellation up.

On this project's runs the second term is the whole story, which is what
makes the negative result a finding rather than a missing one.  The teacher
is non-causal, so ``T-G`` is largely the part of its behaviour that depends
on future context -- the part a causal student cannot reproduce at all.
Pulling the student toward it transfers that error in as noise.

Nothing here trains anything; it reads ``runs/`` and runs a few forward
passes over the validation split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Batches of the validation split the decomposition runs over.  The whole
#: split is ~16 batches of 32, so this is "all of it" with a ceiling that
#: keeps the tab responsive on a laptop; the figure is reported with the row.
DEFAULT_MAX_BATCHES = 16

#: Checks that are properties of trainer/distillation.py rather than of a
#: run.  Listed here so the tab can show them before a run is picked, and so
#: that adding one is a one-line change in a place a reader will find.
STATIC_CHECK_NAMES = (
    "teacher loads strict",
    "teacher frozen and in eval",
    "teacher is the non-causal twin",
    "teacher reproduces its recorded metrics",
    "same split as the student",
    "student selected against the clean signal",
)


# ============================================================
# What is on disk
# ============================================================
def _runs_dir() -> Path:
    from gui import config_io
    return config_io.RUNS_DIR


def kd_runs() -> List[dict]:
    """
    Every distilled run, with the teacher it names and the plain run that is
    its control.

    The control is the run of the same variant, artifact, width and seed
    trained *without* a teacher: without it a KD number has nothing to be
    better or worse than, and finding it by hand is where the project has
    already gone wrong once (``runs/tcn_se_both_kd_s67`` was read as a seed
    repeat of the causal rung and moved the ladder's anchor).
    """
    from gui import config_io

    rows = config_io.list_runs()
    by_name = {r["name"]: r for r in rows}
    out = []
    for r in rows:
        if not r.get("kd"):
            continue
        teacher = _recorded_teacher(r["name"])
        control = _control_for(r, rows)
        out.append({
            "name": r["name"], "variant": r["variant"], "artifact": r["artifact"],
            "master_seed": r["master_seed"], "hidden_channels": r["hidden_channels"],
            "epochs": r["epochs"], "status": r["status"],
            "rrmse_t": r["rrmse_t"], "rrmse_s": r["rrmse_s"], "acc": r["acc"],
            "teacher": teacher,
            "teacher_run": _run_name_of(teacher, by_name),
            "control": control["name"] if control else None,
            "control_rrmse_t": control["rrmse_t"] if control else None,
            "recipe": _recorded_recipe(r["name"]),
            "target_fs": (_history(r["name"]).get("config") or {}).get("target_fs"),
        })
    return out


def _history(name: str) -> dict:
    import json
    p = _runs_dir() / name / "history.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except ValueError:
        return {}


def _recorded_teacher(name: str) -> Optional[str]:
    cfg = (_history(name).get("config") or {})
    t = cfg.get("teacher_ckpt")
    if not t:
        t = ((_history(name).get("teacher") or {}).get("checkpoint"))
    return t


def _recorded_recipe(name: str) -> dict:
    """
    The KD hyperparameters the run wrote down.

    Runs trained before ``trainer/training.py`` recorded them have every
    value ``None``, and that is shown rather than filled in with the
    defaults: a run whose alpha is unknown cannot be reproduced, and putting
    0.5 there would claim it can.
    """
    cfg = (_history(name).get("config") or {})
    keys = ("kd_alpha", "kd_feature_weight", "kd_feature_layers",
            "kd_output_loss", "kd_feature_loss", "init_from_teacher")
    recipe = {k: cfg.get(k) for k in keys}
    recipe["recorded"] = any(v is not None for v in recipe.values())
    return recipe


def _run_name_of(ckpt: Optional[str], by_name: dict) -> Optional[str]:
    if not ckpt:
        return None
    name = Path(str(ckpt)).parent.name
    return name if name in by_name else None


def _control_for(kd_row: dict, rows: Sequence[dict]) -> Optional[dict]:
    """The same model, same seed, same width, trained without a teacher."""
    for r in rows:
        if r["name"] == kd_row["name"] or r.get("kd"):
            continue
        if (r["variant"] == kd_row["variant"] and r["artifact"] == kd_row["artifact"]
                and r["master_seed"] == kd_row["master_seed"]
                and r["hidden_channels"] == kd_row["hidden_channels"]
                and r["status"] == "complete"):
            return r
    return None


# ============================================================
# Static checks
# ============================================================
def _load(run: str):
    """Build a model from a run's own checkpoint, without the export guards."""
    import torch
    from models.variants import build_model_from_config

    ck = torch.load(_runs_dir() / run / "best.pt", map_location="cpu", weights_only=False)
    model = build_model_from_config(dict(ck["model_config"]))
    model.load_state_dict(ck["model_state"])
    return model.eval(), ck


def static_checks(student: str, teacher_run: Optional[str] = None) -> dict:
    """
    The six invariants, each with the evidence for its verdict.

    Returns ``{"checks": [{name, ok, detail}], "ok": bool, ...}``.  A check
    that cannot be run (no teacher recorded, dataset absent) comes back
    ``ok: None`` and says why, which is a different thing from failing and is
    shown as a different thing.
    """
    from gui import config_io
    from models.variants import variant_is_causal

    hist = _history(student)
    cfg = hist.get("config") or {}
    teacher_ckpt = _recorded_teacher(student)
    if teacher_run is None:
        teacher_run = _run_name_of(teacher_ckpt,
                                   {r["name"]: r for r in config_io.list_runs()})
    checks: List[dict] = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": ok, "detail": detail})

    # 1. strict load -- the teacher's weights must fit the architecture the
    #    checkpoint claims, with nothing missing and nothing left over.
    teacher = None
    if not teacher_ckpt:
        add(STATIC_CHECK_NAMES[0], None, f"{student} records no teacher_ckpt")
    else:
        path = PROJECT_ROOT / str(teacher_ckpt).lstrip("./")
        try:
            from trainer.distillation import load_teacher
            teacher = load_teacher(path)
            add(STATIC_CHECK_NAMES[0], True, f"strict=True against {teacher_ckpt}")
        except Exception as e:                                   # noqa: BLE001
            add(STATIC_CHECK_NAMES[0], False, f"{type(e).__name__}: {e}")

    # 2. frozen -- load_teacher calls eval() and clears requires_grad; if a
    #    later edit stopped doing either, the teacher would drift during
    #    training and the run would be distilling from a moving target.
    if teacher is None:
        add(STATIC_CHECK_NAMES[1], None, "teacher not loaded")
    else:
        grads = [n for n, p in teacher.named_parameters() if p.requires_grad]
        ok = not grads and not teacher.training
        add(STATIC_CHECK_NAMES[1], ok,
            "eval() and no parameter requires grad" if ok else
            f"training={teacher.training}, {len(grads)} parameter(s) still require grad")

    # 3. the intended teacher is a non-causal twin: distilling from a causal
    #    model of the same shape transfers nothing the student could not
    #    already learn, which would make a null result uninformative.
    if teacher is None:
        add(STATIC_CHECK_NAMES[2], None, "teacher not loaded")
    else:
        v = teacher.model_config.get("variant", "?")
        causal = variant_is_causal(v)
        add(STATIC_CHECK_NAMES[2], not causal,
            f"teacher variant {v} is non-causal" if not causal else
            f"teacher variant {v} is causal; it has no future context to transfer")

    # 4. reproduce the teacher -- the decisive one. The number the KD run
    #    wrote down for its teacher has to come back when the teacher is
    #    rebuilt and re-scored here, or the thing that trained the student
    #    was not the model this tab is looking at.
    #
    #    Over the WHOLE validation split, with no batch cap. The recorded
    #    figure is a whole-split average, so scoring a prefix and comparing
    #    the two measures sampling rather than agreement: three batches of
    #    the 16 read 0.6018 against a recorded 0.5928, which looks exactly
    #    like a mismatch and is not one.
    recorded = (hist.get("teacher") or {})
    if teacher is None or not recorded:
        add(STATIC_CHECK_NAMES[3], None,
            "the run recorded no teacher metrics" if teacher is not None else "teacher not loaded")
    else:
        try:
            got = _score(teacher, cfg.get("artifact") or "BOTH",
                         int(cfg.get("master_seed") or 42), max_batches=None,
                         target_fs=cfg.get("target_fs"))
            diffs = {k: abs(got[k] - recorded[k]) for k in ("rrmse_t", "acc") if k in recorded}
            ok = all(v < 1e-3 for v in diffs.values()) if diffs else None
            add(STATIC_CHECK_NAMES[3], ok,
                f"recorded RRMSE_t {recorded.get('rrmse_t', float('nan')):.4f}, "
                f"recomputed {got['rrmse_t']:.4f} on the whole validation split "
                f"(max |diff| {max(diffs.values()):.2e})" if diffs else "nothing to compare")
        except Exception as e:                                   # noqa: BLE001
            add(STATIC_CHECK_NAMES[3], None, f"{type(e).__name__}: {e}")

    # 5. one split -- master_seed picks the split, so a student and a teacher
    #    at different seeds were scored on different epochs and nothing below
    #    is comparable.
    t_seed = (_history(teacher_run).get("config") or {}).get("master_seed") if teacher_run else None
    s_seed = cfg.get("master_seed")
    if t_seed is None or s_seed is None:
        add(STATIC_CHECK_NAMES[4], None,
            f"seed unrecorded (student {s_seed}, teacher {t_seed})")
    else:
        add(STATIC_CHECK_NAMES[4], t_seed == s_seed,
            f"student seed {s_seed}, teacher seed {t_seed}")

    # 6. selection -- KDTrainer validates against the clean signal, so
    #    best.pt is the best denoiser and not the best teacher impersonator.
    vl = hist.get("val_loss") or []
    if not vl:
        add(STATIC_CHECK_NAMES[5], None, "no history")
    else:
        import torch as _t
        best_epoch = min(range(len(vl)), key=vl.__getitem__) + 1
        saved = _t.load(_runs_dir() / student / "best.pt",
                        map_location="cpu", weights_only=False).get("epoch")
        ok = saved in (None, best_epoch)
        add(STATIC_CHECK_NAMES[5], ok,
            f"best.pt is epoch {saved}, argmin of val_loss (against clean) is {best_epoch}")

    verdicts = [c["ok"] for c in checks]
    return {
        "student": student, "teacher_run": teacher_run, "teacher_ckpt": teacher_ckpt,
        "recipe": _recorded_recipe(student),
        "checks": checks,
        "ok": all(v for v in verdicts if v is not None) and any(verdicts),
        "n_failed": sum(1 for v in verdicts if v is False),
        "n_skipped": sum(1 for v in verdicts if v is None),
    }


# ============================================================
# The decomposition
# ============================================================
def _loader(artifact: str, master_seed: int, batch_size: int = 32,
            target_fs: Optional[float] = None):
    """
    The validation split a run was scored on.

    ``target_fs`` is not optional in spirit: a run trained at 250 Hz was
    validated on 500-sample epochs, and re-scoring its teacher on the 512-
    sample ones would compare two different datasets. That is exactly what
    the "teacher reproduces its recorded metrics" check would then report --
    a mismatch, correctly, for a reason that has nothing to do with the
    teacher. The rate comes from the run's own history.
    """
    from data.Dataloader import build_dataloaders
    _, val, _ = build_dataloaders(str(PROJECT_ROOT / "data" / "EEGdenoisenet" / "data"),
                                  artifact_type=artifact, batch_size=batch_size,
                                  master_seed=master_seed, target_fs=target_fs)
    return val


def _score(model, artifact: str, master_seed: int,
           max_batches: Optional[int] = None,
           target_fs: Optional[float] = None) -> dict:
    """Metrics over the validation split; ``max_batches=None`` is all of it."""
    import torch

    import metrics

    agg, n = {"rrmse_t": 0.0, "rrmse_s": 0.0, "acc": 0.0}, 0
    with torch.no_grad():
        for i, b in enumerate(_loader(artifact, master_seed, target_fs=target_fs)):
            if max_batches is not None and i >= max_batches:
                break
            m = metrics.compute_all_metrics(model(b["noisy"]), b["clean"])
            bs = b["noisy"].size(0)
            for k in agg:
                agg[k] += float(m[k]) * bs
            n += bs
    return {k: v / max(n, 1) for k, v in agg.items()}


def decompose(teacher_run: str, students: Sequence[str], artifact: str = "BOTH",
              master_seed: int = 42, max_batches: int = DEFAULT_MAX_BATCHES,
              target_fs: Optional[float] = None) -> dict:
    """
    ``||S-G||^2 = ||S-T||^2 + ||T-G||^2 + 2<S-T, T-G>`` for each student.

    One pass over the validation split with every model in memory at once,
    so the three terms are measured on exactly the same batches -- summing
    them has to reproduce the direct ``||S-G||^2``, and the row reports that
    residual so a reader can see it does.
    """
    import torch

    teacher, _ = _load(teacher_run)
    models = {s: _load(s)[0] for s in students}

    tot = 0
    tg = 0.0
    acc = {s: {"st": 0.0, "cross": 0.0, "sg": 0.0} for s in students}
    with torch.no_grad():
        for i, b in enumerate(_loader(artifact, master_seed, target_fs=target_fs)):
            if i >= max_batches:
                break
            x, g = b["noisy"], b["clean"]
            T = teacher(x)
            tg += float(((T - g) ** 2).sum())
            tot += g.numel()
            for s, m in models.items():
                y = m(x)
                acc[s]["st"] += float(((y - T) ** 2).sum())
                acc[s]["cross"] += float((2 * (y - T) * (T - g)).sum())
                acc[s]["sg"] += float(((y - g) ** 2).sum())

    tot = max(tot, 1)
    rows = []
    for s, a in acc.items():
        st, cross, sg = a["st"] / tot, a["cross"] / tot, a["sg"] / tot
        rows.append({
            "run": s, "kd": bool((_history(s).get("config") or {}).get("teacher_ckpt")),
            "s_minus_t": st, "t_minus_g": tg / tot, "cross": cross,
            "s_minus_g": sg, "sum": st + tg / tot + cross,
            "residual": abs(sg - (st + tg / tot + cross)),
        })
    rows.sort(key=lambda r: r["kd"])          # control first, KD beneath it

    out = {"teacher": teacher_run, "artifact": artifact, "master_seed": master_seed,
           "max_batches": max_batches, "target_fs": target_fs,
           "t_minus_g": tg / tot, "rows": rows}
    if len(rows) == 2:
        out["delta"] = _read_off(rows[0], rows[1])
    return out


def _read_off(control: dict, kd: dict) -> dict:
    """
    The two numbers the tab exists to put side by side, and what they mean.

    ``d_s_minus_t`` negative is the only evidence that the distillation did
    anything at all; ``d_cross`` positive is the price it charged for it.
    """
    d_st = kd["s_minus_t"] - control["s_minus_t"]
    d_cross = kd["cross"] - control["cross"]
    d_sg = kd["s_minus_g"] - control["s_minus_g"]
    if d_st < 0 and d_sg > 0:
        verdict = ("KD worked and cost accuracy: the student moved onto the teacher "
                   "and gave up the error cancellation that paid for its old score")
    elif d_st >= 0:
        verdict = ("the student did not move toward the teacher -- check the static "
                   "checks above before reading anything into the score")
    else:
        verdict = "KD moved the student toward the teacher and improved it"
    return {"d_s_minus_t": d_st, "d_cross": d_cross, "d_s_minus_g": d_sg,
            "control": control["run"], "kd": kd["run"], "verdict": verdict}


# ============================================================
# What the tab draws itself from
# ============================================================
def meta() -> dict:
    runs = kd_runs()
    return {
        "kd_runs": runs,
        "check_names": list(STATIC_CHECK_NAMES),
        "default_max_batches": DEFAULT_MAX_BATCHES,
        "identity": "||S-G||^2 = ||S-T||^2 + ||T-G||^2 + 2<S-T, T-G>",
        "note": ("Nothing here trains. The checks and the decomposition both read "
                 "runs/ and run a few forward passes over the validation split."),
    }
