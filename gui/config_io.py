"""
Schema of the training YAML files + a comment-preserving writer.

The presets in configs/ are documentation as much as configuration: every
key carries a comment.  PyYAML's dumper throws all of that away, so the
GUI never re-serialises a file.  Instead it rewrites individual
``key: value`` lines in place, keeping comments, blank lines, section
headers and the original number formatting of anything that did not
change.  Keys the file does not have yet are appended under their
schema section header.

``SCHEMA`` is the single source of truth for what the form shows: field
type, default (mirroring trainer/__main__.py), choices, help text and a
``show_if`` tag the page uses to hide fields that do not apply to the
selected variant.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
RUNS_DIR = PROJECT_ROOT / "runs"

LOSS_NAMES = ["l1_combined", "l2_combined", "mse", "l1", "stft"]
ARTIFACTS = ["EOG", "EMG", "BOTH"]


# ============================================================
# Schema
# ============================================================
@dataclass
class Field:
    key: str
    type: str                          # int | float | str | bool | select | range2 | list_str
    default: Any = None
    help: str = ""
    choices: Optional[str] = None      # name of a choice list served by /api/meta
    nullable: bool = False             # empty input → null (else empty is an error)
    show_if: Optional[str] = None      # attention | mla | sandwich | staged | turboquant | kd
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    datalist: Optional[str] = None     # name of a suggestion list served by /api/meta


@dataclass
class Section:
    title: str
    fields: List[Field] = field(default_factory=list)


SCHEMA: List[Section] = [
    Section("Model", [
        Field("variant", "select", "tcn_mla", choices="variants",
              help="Registry name. *_bi = bidirectional teacher (not deployable)."),
        Field("hidden_channels", "int", 64, min=8, step=8, help="TCN width H"),
        Field("n_heads", "int", 4, min=1, show_if="attention",
              help="Attention heads; attn_dim must be divisible by it"),
        Field("n_attn_layers", "int", 2, min=1, show_if="attention",
              help="Transformer depth (presets: 2 / 10 / 20 / 50)"),
        Field("max_seq_len", "int", 512, min=16, show_if="attention",
              help="Causal-mask size; must be ≥ the 512-sample epoch length"),
        Field("kv_lora_rank", "int", 16, min=1, show_if="mla",
              help="MLA latent KV rank (d_model // 4 = 16 by default)"),
        Field("bottleneck_ratio", "float", 0.5, min=0.05, max=1.0, step=0.05, show_if="sandwich",
              help="Channel squeeze before attention: attn_dim = H × ratio"),
        Field("skip_bottleneck_ratio", "float", None, nullable=True, min=0.05, max=1.0, step=0.05,
              show_if="attention",
              help="BottleneckSkip on each attention residual (anti-collapse); empty = identity skip"),
        Field("dropout", "float", 0.1, min=0.0, max=0.9, step=0.05,
              help="Dropout in TCN blocks and attention"),
    ]),
    Section("Data", [
        Field("artifact", "select", "EOG", choices="artifacts"),
        Field("data_dir", "str", "./data/EEGdenoisenet/data/",
              help="Folder with EEG/EOG/EMG_all_epochs.npy"),
        Field("batch_size", "int", 32, min=1),
        Field("snr_range_db", "range2", [-7.0, 2.0],
              help="Train: uniform in [lo, hi]; val/test: 10-point grid over it"),
        Field("master_seed", "int", 42, min=0,
              help="Seed for splits, SNR, weight init, dropout and batch order"),
        Field("device", "select", "auto", choices="devices",
              help="auto = cuda > mps > cpu; cpu for bit-exact repeats (MPS is not deterministic)"),
    ]),
    Section("Training", [
        Field("learning_rate", "float", 5e-4, min=0.0, step=1e-4, help="Base LR (TCN)"),
        Field("attn_lr_scale", "float", 3.0, min=0.0, step=0.5, show_if="attention",
              help="Decoder LR = base × scale"),
        Field("weight_decay", "float", 0.0, min=0.0, step=1e-4),
        Field("max_epochs", "int", 100, min=1),
        Field("log_every", "int", 20, min=1, help="Print every N steps"),
    ]),
    Section("Scheduler", [
        Field("sched_warmup_epochs", "int", 5, min=0, help="Linear warmup before cosine decay"),
        Field("eta_min", "float", 1e-6, min=0.0, step=1e-6, help="Cosine floor"),
    ]),
    Section("Staged training", [
        Field("staged", "bool", False, help="Phase 1: TCN only (decoder frozen) → Phase 2: all"),
        Field("warmup_epochs", "int", 20, min=1, show_if="staged", help="Phase 1 length"),
        Field("warmup_lr", "float", 1e-3, min=0.0, step=1e-4, show_if="staged"),
        Field("finetune_lr", "float", 5e-4, min=0.0, step=1e-4, show_if="staged"),
    ]),
    Section("Loss", [
        Field("loss_fn", "select", "l1_combined", choices="losses",
              help="l1_combined = L1 + 0.05 × multi-res STFT"),
    ]),
    Section("Output", [
        Field("save_dir", "str", None, nullable=True,
              help="Checkpoint folder; empty → runs/<variant>_<artifact>[_kd][_s<seed>] "
                   "— the seed suffix appears for any seed but the default 42, so two "
                   "seeds of one model keep their own results"),
        Field("overwrite", "bool", False,
              help="Replace the run already in that folder. Without it, training refuses "
                   "to start on top of a run from a different (or unrecorded) master_seed"),
        Field("tensorboard", "bool", False),
    ]),
    Section("TurboQuant", [
        Field("turboquant", "bool", False, show_if="attention",
              help="Fake K/V quantisation experiment (no runtime benefit)"),
        Field("turboquant_bits", "int", 3, min=2, max=8, show_if="turboquant"),
        Field("turboquant_key_mode", "select", "prod", choices="tq_key_modes", show_if="turboquant"),
        Field("turboquant_train_mode", "select", "off", choices="tq_train_modes", show_if="turboquant"),
        Field("turboquant_seed", "int", 42, min=0, show_if="turboquant"),
    ]),
    Section("Knowledge distillation", [
        Field("teacher_ckpt", "str", None, nullable=True, datalist="checkpoints",
              help="Teacher best.pt (normally a *_bi run). Setting it enables KD."),
        Field("teacher_config", "str", None, nullable=True, show_if="kd",
              help="Only for legacy teacher checkpoints without model_config"),
        Field("kd_alpha", "float", 0.5, min=0.0, max=1.0, step=0.05, show_if="kd",
              help="Weight of loss(student, teacher); ground truth gets 1 − α"),
        Field("kd_feature_weight", "float", 0.0, min=0.0, step=0.05, show_if="kd",
              help="Weight of MSE(student features, teacher features); 0 = off"),
        Field("kd_feature_layers", "list_str", ["decoder"], show_if="kd",
              help="Submodules to match, comma-separated; student:teacher pairs for different depths "
                   "(e.g. decoder, decoder.blocks.1:decoder.blocks.9)"),
        Field("kd_output_loss", "select", None, choices="losses", nullable=True, show_if="kd",
              help="Loss vs teacher output; empty → same as loss_fn"),
        Field("kd_feature_loss", "select", "mse", choices="feature_losses", show_if="kd"),
        Field("init_from_teacher", "bool", False, show_if="kd",
              help="Copy the teacher's weights into the student first (leading slice if the student is smaller)"),
    ]),
]

FIELDS: Dict[str, Field] = {f.key: f for s in SCHEMA for f in s.fields}
SECTION_OF: Dict[str, str] = {f.key: s.title for s in SCHEMA for f in s.fields}
NULL_OK = {k for k, f in FIELDS.items() if f.nullable}


def schema_as_json() -> list:
    return [{"title": s.title, "fields": [asdict(f) for f in s.fields]} for s in SCHEMA]


# ============================================================
# Reading
# ============================================================
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*\.ya?ml$")


def config_path(name: str) -> Path:
    """Resolve a bare file name inside configs/, refusing anything else."""
    if not _NAME_RE.match(name):
        raise ValueError(f"invalid config name {name!r} (letters, digits, _ . - and .yaml only)")
    p = (CONFIG_DIR / name).resolve()
    if p.parent != CONFIG_DIR.resolve():
        raise ValueError("config must live directly in configs/")
    return p


def list_configs() -> List[str]:
    return sorted(p.name for p in CONFIG_DIR.glob("*.yaml"))


def read_config(name: str) -> dict:
    """
    Returns {name, text, values, extra, present}:
      values  – schema keys found in the file (raw parsed values)
      extra   – non-schema keys, kept verbatim by the writer
      present – every top-level key in file order
    """
    path = config_path(name)
    text = path.read_text()
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{name} is not a mapping")
    return {
        "name": name,
        "text": text,
        "values": {k: v for k, v in data.items() if k in FIELDS},
        "extra": {k: v for k, v in data.items() if k not in FIELDS},
        "present": list(data.keys()),
    }


# ============================================================
# Value coercion / validation (client values → typed Python)
# ============================================================
def coerce(key: str, value: Any) -> Any:
    """Turn a form value into the typed Python value the YAML should hold."""
    f = FIELDS[key]
    empty = value is None or value == "" or value == []
    if empty:
        if f.nullable:
            return None
        raise ValueError(f"{key}: a value is required")

    if f.type == "int":
        v = int(value) if not isinstance(value, bool) else int(value)
        if isinstance(value, float) and value != v:
            raise ValueError(f"{key}: must be an integer")
    elif f.type == "float":
        v = float(value)
    elif f.type == "bool":
        v = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
    elif f.type == "select":
        v = str(value)
    elif f.type == "str":
        v = str(value)
    elif f.type == "range2":
        seq = value if isinstance(value, (list, tuple)) else [x for x in re.split(r"[,\s]+", str(value)) if x]
        if len(seq) != 2:
            raise ValueError(f"{key}: expected two numbers [low, high]")
        v = [float(seq[0]), float(seq[1])]
        if v[0] > v[1]:
            raise ValueError(f"{key}: low must be ≤ high")
    elif f.type == "list_str":
        seq = value if isinstance(value, (list, tuple)) else [x.strip() for x in str(value).split(",")]
        v = [str(x) for x in seq if str(x).strip()]
        if not v:
            raise ValueError(f"{key}: at least one entry")
    else:  # pragma: no cover
        raise ValueError(f"{key}: unknown field type {f.type}")

    if f.type in ("int", "float"):
        if f.min is not None and v < f.min:
            raise ValueError(f"{key}: must be ≥ {f.min}")
        if f.max is not None and v > f.max:
            raise ValueError(f"{key}: must be ≤ {f.max}")
    return v


def coerce_all(values: Dict[str, Any]) -> Dict[str, Any]:
    out, errors = {}, []
    for k, v in values.items():
        if k not in FIELDS:
            errors.append(f"{k}: not a known setting")
            continue
        try:
            out[k] = coerce(k, v)
        except (ValueError, TypeError) as e:
            errors.append(str(e))
    if errors:
        raise ValueError("; ".join(errors))
    return out


# ============================================================
# Writing (comment-preserving)
# ============================================================
def yaml_scalar(v: Any) -> str:
    """Render one value the way the presets are written."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        # PyYAML's float form always round-trips (it writes 1.0e-06, not 1e-06,
        # which PyYAML itself would read back as a string).
        return yaml.safe_dump(v).split("\n")[0]
    if isinstance(v, str):
        return json.dumps(v)                     # double-quoted, escaped
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(yaml_scalar(x) for x in v) + "]"
    raise TypeError(f"cannot render {type(v).__name__} as a YAML scalar")


_KEY_LINE = re.compile(r"^(?P<key>[A-Za-z_][A-Za-z0-9_]*):(?P<sep>[ \t]*)(?P<rest>.*)$")


def _split_comment(rest: str):
    """Split 'value   # comment' → ('value', '   # comment'), honouring quotes."""
    quote = None
    for i, ch in enumerate(rest):
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "#" and (i == 0 or rest[i - 1] in " \t"):
            j = i
            while j > 0 and rest[j - 1] in " \t":
                j -= 1
            return rest[:j], rest[j:]
    return rest.rstrip(), rest[len(rest.rstrip()):]


def _same(existing_text: str, new_value: Any) -> bool:
    try:
        old = yaml.safe_load(existing_text) if existing_text.strip() else None
    except yaml.YAMLError:
        return False
    if isinstance(old, (list, tuple)) and isinstance(new_value, (list, tuple)):
        return list(old) == list(new_value)
    return old == new_value and type(old) is type(new_value)


def update_yaml_text(text: str, values: Dict[str, Any]) -> str:
    """
    Rewrite top-level ``key: value`` lines for the keys in ``values``.

    Lines whose parsed value already equals the new value are left
    byte-for-byte unchanged; everything else in the file (comments, blank
    lines, unknown keys) is untouched.  Keys not found are appended,
    grouped under their schema section.
    """
    pending = dict(values)
    out = []
    for line in text.splitlines():
        m = _KEY_LINE.match(line)
        if m and m.group("key") in pending:
            key = m.group("key")
            new_value = pending.pop(key)
            value_text, comment = _split_comment(m.group("rest"))
            if not _same(value_text, new_value):
                sep = m.group("sep") or " "
                line = f"{key}:{sep}{yaml_scalar(new_value)}{comment}"
        out.append(line)

    if pending:
        by_section: Dict[str, List[str]] = {}
        for key, v in pending.items():
            by_section.setdefault(SECTION_OF.get(key, "Other"), []).append(f"{key}: {yaml_scalar(v)}")
        for title, lines in by_section.items():
            if out and out[-1].strip():
                out.append("")
            out.append(f"# --- {title} (added by the training GUI) ---")
            out.extend(lines)

    return "\n".join(out).rstrip("\n") + "\n"


def render_config(name: str, values: Dict[str, Any]) -> str:
    """The text that save_config would write, without writing it."""
    current = read_config(name)
    return update_yaml_text(current["text"], coerce_all(values))


def save_config(name: str, values: Dict[str, Any]) -> str:
    new_text = render_config(name, values)
    yaml.safe_load(new_text)                       # must still parse
    config_path(name).write_text(new_text)
    return new_text


def create_config(name: str, template: Optional[str] = None) -> str:
    """Create configs/<name> as a copy of ``template`` (or a minimal file)."""
    path = config_path(name)
    if path.exists():
        raise FileExistsError(f"{name} already exists")
    if template:
        text = config_path(template).read_text()
        header = f"# Created by the training GUI from {template}\n"
        text = header + text
    else:
        text = "# Created by the training GUI\n\n"
        text += "".join(f"{f.key}: {yaml_scalar(f.default)}\n"
                        for f in (FIELDS["variant"], FIELDS["artifact"], FIELDS["data_dir"],
                                  FIELDS["batch_size"], FIELDS["learning_rate"],
                                  FIELDS["max_epochs"], FIELDS["loss_fn"]))
    path.write_text(text)
    return text


# ============================================================
# Runs (for the teacher picker and the runs table)
# ============================================================
_CKPT_CACHE: Dict[str, tuple] = {}          # path -> (mtime, size, info); torch.load is ~12 ms


def _fmt_lr(v: Optional[float]) -> str:
    return "?" if v is None else f"{v:.2g}"


def optimizer_info(ck: dict) -> dict:
    """
    What optimizer and schedule produced a checkpoint, read from the state it saves.

    Nothing has to be recomputed: ``optimizer_state`` and ``scheduler_state`` are
    written by every checkpoint, so this works on runs of any age.

      family     Adam and AdamW differ only in ``decoupled_weight_decay`` (torch 2.x
                 implements AdamW as Adam with that flag on); SGD by ``momentum``.
      scheduler  ReduceLROnPlateau keeps ``factor``/``patience``; the project's
                 warmup->cosine chain is a SequentialLR holding a LinearLR and a
                 CosineAnnealingLR; an empty state means no scheduler at all.
      lr         ``initial_lr`` is the configured value and is exact.  Without it
                 (no LR scheduler ever attached) only the *decayed* rate at save
                 time survives, so ``lr_exact`` is False and the value is not the
                 one that was configured.
      attn_lr_scale  decoder LR / TCN LR.  Both groups decay together, so this
                 ratio is exact even when the rates themselves are not.
    """
    groups = (ck.get("optimizer_state") or {}).get("param_groups") or []
    if not groups:
        return {}
    g0 = groups[0]
    if "betas" in g0:
        dwd = g0.get("decoupled_weight_decay")
        family = "Adam/AdamW" if dwd is None else ("AdamW" if dwd else "Adam")
    elif "momentum" in g0:
        family = "SGD"
    else:
        family = "?"

    sch = ck.get("scheduler_state") or {}
    if "factor" in sch and "patience" in sch:
        scheduler = "plateau"
    elif "_schedulers" in sch:
        inner = sch["_schedulers"]
        scheduler = "cosine+warmup" if any("T_max" in x for x in inner) else "chained"
    elif "T_max" in sch:
        scheduler = "cosine"
    elif not sch:
        scheduler = "none"
    else:
        scheduler = "?"

    initial = [g.get("initial_lr") for g in groups]
    lr_exact = all(v is not None for v in initial)
    lrs = initial if lr_exact else [g.get("lr") for g in groups]
    names = [g.get("name") for g in groups]
    lr = lrs[0] if lrs else None
    scale = None
    if len(groups) == 2 and lrs[0]:
        # the group order is (tcn, decoder) as build_param_groups creates them
        i_dec = names.index("decoder") if "decoder" in names else 1
        i_tcn = 1 - i_dec
        lr = lrs[i_tcn]
        if lrs[i_tcn]:
            scale = round(lrs[i_dec] / lrs[i_tcn], 4)

    label = f"{family} · lr {_fmt_lr(lr)}{'' if lr_exact else ' (decayed)'}"
    if scale is not None:
        label += f" · attn ×{scale:g}"
    label += f" · {scheduler}"
    # runs whose configured LR was never recorded group together rather than
    # splitting by however far each had decayed when it was saved
    key = (f"{family} · lr {_fmt_lr(lr) if lr_exact else '?'}"
           + (f" · attn ×{scale:g}" if scale is not None else "")
           + f" · {scheduler}")
    return {"family": family, "scheduler": scheduler, "lr": lr, "lr_exact": lr_exact,
            "attn_lr_scale": scale, "weight_decay": g0.get("weight_decay"),
            "betas": list(g0["betas"]) if g0.get("betas") else None,
            "groups": [n for n in names if n], "label": label, "key": key}


def _checkpoint_info(path: Path) -> dict:
    """
    ``epoch`` and ``model_config`` of one checkpoint, cached on (mtime, size).

    Returns ``{}`` when torch is missing or the file cannot be read — the GUI
    stays usable without torch, it just cannot show the architecture.
    """
    try:
        st = path.stat()
    except OSError:
        return {}
    key = str(path)
    hit = _CKPT_CACHE.get(key)
    if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
        return hit[2]
    info: dict = {}
    try:
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        info = {"epoch": ck.get("epoch"), "saved_at": ck.get("saved_at"),
                "model_config": dict(ck.get("model_config") or {}),
                "run_config": dict(ck.get("run_config") or {}),
                "params": sum(v.numel() for k, v in (ck.get("model_state") or {}).items()
                              if not k.endswith("causal_mask")),
                "optimizer": optimizer_info(ck)}
    except Exception:                        # torch missing, half-written file, foreign format
        info = {}
    _CKPT_CACHE[key] = (st.st_mtime, st.st_size, info)
    return info


def _run_time(run_dir: Path, best_info: dict) -> dict:
    """
    When the run last wrote, as ``(finished_at, finished_at_source)``.

    ``recorded``  the checkpoint says so itself (``saved_at``, since Sep 2026)
    ``file``      the mtime of last.pt, which is the same answer until the
                  directory is copied, restored from a backup or synced — then
                  it is the time of the copy, not of the run

    Sorting by it is still the right default for older runs; the view marks the
    ones it had to guess so a suspicious ordering can be explained.
    """
    last = run_dir / "last.pt"
    newest = last if last.is_file() else run_dir / "best.pt"
    recorded = (_checkpoint_info(newest) or {}).get("saved_at") or best_info.get("saved_at")
    if recorded:
        return {"finished_at": recorded, "finished_at_source": "recorded"}
    try:
        stamp = datetime.fromtimestamp(newest.stat().st_mtime)
    except OSError:
        return {"finished_at": None, "finished_at_source": "unknown"}
    return {"finished_at": stamp.isoformat(timespec="seconds"), "finished_at_source": "file"}


def _artifact_from_name(name: str) -> Optional[str]:
    """``tcn_mla_bi_both`` -> ``BOTH``.  The artifact is a data setting, so it is
    not in ``model_config``; the run directory name is the only record of it."""
    low = name.lower()
    for token in ("both", "eog", "emg"):
        if low.endswith("_" + token) or ("_" + token + "_") in low:
            return token.upper()
    return None


def _variant_from_name(name: str, artifact: Optional[str]) -> Optional[str]:
    """
    Fall back to the run directory name for checkpoints saved before
    ``model_config`` existed: runs are created as ``<variant>_<artifact>[_kd]``.
    """
    stem = name[:-3] if name.endswith("_kd") else name
    if artifact and stem.lower().endswith("_" + artifact.lower()):
        stem = stem[: -(len(artifact) + 1)]
    return stem or None


def _loss_of(run_dir: Path, history: Optional[dict]) -> tuple[Optional[str], str]:
    """
    ``(loss_fn, source)`` for one run.

    ``recorded``   the run saved it itself (history.json["config"], since Sep 2026)
    ``recovered``  ``python -m trainer.identify_loss`` reproduced it from the weights
    ``unknown``    neither — its ``val_loss`` cannot be compared with anything

    Both sources are plain JSON, so the GUI still reads them without torch.
    """
    cfg = (history or {}).get("config") or {}
    if cfg.get("loss_fn"):
        return cfg["loss_fn"], "recorded"
    side = run_dir / "loss_fn.json"
    if side.is_file():
        try:
            rec = json.loads(side.read_text())
        except ValueError:
            return None, "unknown"
        if rec.get("loss_fn") and rec.get("status") in ("recovered", "recorded"):
            # The verdict was reached by running that checkpoint; if best.pt has
            # been rewritten since (the run was retrained), it no longer applies.
            try:
                st = (run_dir / "best.pt").stat()
            except OSError:
                return None, "unknown"
            stamped = (rec.get("checkpoint_mtime"), rec.get("checkpoint_size"))
            if stamped != (None, None) and stamped != (st.st_mtime, st.st_size):
                return None, "unknown"
            return rec["loss_fn"], "recovered"
    return None, "unknown"


def list_runs() -> List[dict]:
    """
    One row per runs/<dir> that holds a best.pt.

    Besides the best-epoch metrics from history.json, each row carries what the
    checkpoint says about itself (``variant``, depth, rank, parameter count) and
    a ``status``:

      complete     history.json exists and last.pt sits at its final epoch
      stale        history.json exists but last.pt is at a different epoch — the
                   directory was re-used, so the metrics shown describe weights
                   that have been overwritten
      interrupted  no history.json: the run never reached _finish()

    ``kind`` groups runs that are the same experiment (variant x artifact).
    """
    rows = []
    if not RUNS_DIR.is_dir():
        return rows
    for d in sorted(RUNS_DIR.iterdir()):
        best = d / "best.pt"
        if not d.is_dir() or not best.is_file():
            continue
        info = _checkpoint_info(best)
        mc = info.get("model_config") or {}
        artifact = _artifact_from_name(d.name)
        variant = mc.get("variant") or _variant_from_name(d.name, artifact)
        row = {"name": d.name, "checkpoint": f"./runs/{d.name}/best.pt",
               "bidirectional": bool(mc.get("causal") is False) or d.name.endswith("_bi")
                                or "_bi_" in d.name,
               "epochs": None, "best_epoch": None, "val_loss": None,
               "rrmse_t": None, "rrmse_s": None, "acc": None, "kd": False,
               "variant": variant, "artifact": artifact,
               "kind": f"{variant} · {artifact}" if variant and artifact else (variant or d.name),
               "hidden_channels": mc.get("hidden_channels"),
               "n_attn_layers": mc.get("n_attn_layers"),
               "kv_lora_rank": mc.get("kv_lora_rank"),
               "params": info.get("params"),
               "master_seed": (info.get("run_config") or {}).get("master_seed"),
               "last_epoch": (_checkpoint_info(d / "last.pt") or info).get("epoch"),
               **_run_time(d, info),
               "loss_fn": None, "loss_source": "unknown",
               "optimizer": (info.get("optimizer") or {}).get("label"),
               "optimizer_key": (info.get("optimizer") or {}).get("key"),
               "optimizer_detail": info.get("optimizer") or {},
               "status": "interrupted"}
        hist = d / "history.json"
        h = None
        if hist.is_file():
            try:
                h = json.loads(hist.read_text())
                vl = h.get("val_loss") or []
                if vl:
                    i = min(range(len(vl)), key=vl.__getitem__)
                    row.update(epochs=len(vl), best_epoch=i + 1, val_loss=vl[i],
                               rrmse_t=h["rrmse_t"][i], rrmse_s=h["rrmse_s"][i], acc=h["acc"][i],
                               kd="kd_output" in h)
                    # Same value as the checkpoint's run_config, readable
                    # without torch; only fall back when it is missing.
                    if (h.get("config") or {}).get("master_seed") is not None:
                        row["master_seed"] = h["config"]["master_seed"]
                    # last.pt should sit at the final epoch of that history; if it
                    # does not, the directory was re-used by a later, shorter run.
                    last = row["last_epoch"]
                    row["status"] = "complete" if last in (None, len(vl)) else "stale"
            except (ValueError, KeyError, IndexError):
                h = None
        row["loss_fn"], row["loss_source"] = _loss_of(d, h)
        rows.append(row)
    return rows


def read_run_history(name: str) -> dict:
    """
    The per-epoch history of one finished run, in the same record shape the
    training stream produces, so the page can plot a past run with the same
    code that draws the live one.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name or ""):
        raise ValueError(f"invalid run name {name!r}")
    d = (RUNS_DIR / name).resolve()
    if d.parent != RUNS_DIR.resolve() or not d.is_dir():
        raise ValueError("run must be a directory directly inside runs/")
    hist = d / "history.json"
    if not hist.is_file():
        raise FileNotFoundError(f"{name} has no history.json (the run did not finish)")
    h = json.loads(hist.read_text())
    vl = h.get("val_loss") or []
    records = []
    for i in range(len(vl)):
        rec = {"epoch": i + 1, "phase": "history", "val_loss": vl[i]}
        for key in ("train_loss", "rrmse_t", "rrmse_s", "acc"):
            seq = h.get(key) or []
            if i < len(seq):
                rec[key] = seq[i]
        records.append(rec)
    return {"name": name, "max_epochs": len(records), "metrics": records}


def runs_by_loss(runs: Optional[List[dict]] = None) -> List[dict]:
    """
    Complete runs grouped by ``loss_fn`` x ``artifact``, best val_loss first.

    ``val_loss`` is whatever ``loss_fn`` the run minimised, measured on whichever
    artifact it trained against, so it is only comparable inside one of these
    groups — across groups the same number means different things (an MSE 0.13
    and an L1+STFT 0.44 can be the same model).  Within a group the runs are
    ranked by val_loss, which is the one view where that ranking is meaningful.

    Runs whose loss is unknown are grouped separately rather than mixed in;
    ``python -m trainer.identify_loss`` recovers it for those that can be.
    """
    groups: Dict[str, dict] = {}
    for r in (list_runs() if runs is None else runs):
        if r["status"] != "complete":
            continue
        loss = r.get("loss_fn") or "unknown"
        artifact = r.get("artifact") or "?"
        key = f"{loss} · {artifact}"
        g = groups.setdefault(key, {"key": key, "loss_fn": r.get("loss_fn"),
                                    "artifact": r.get("artifact"), "runs": [],
                                    "comparable": bool(r.get("loss_fn"))})
        g["runs"].append(r)
    for g in groups.values():
        g["runs"].sort(key=lambda r: (r["val_loss"] is None, r["val_loss"]))
        g["n"] = len(g["runs"])
        g["sources"] = sorted({r.get("loss_source", "unknown") for r in g["runs"]})
    # known losses first, then by loss name, then artifact
    return sorted(groups.values(),
                  key=lambda g: (not g["comparable"], g["loss_fn"] or "~", g["artifact"] or "~"))


def runs_by_optimizer(runs: Optional[List[dict]] = None) -> List[dict]:
    """
    Complete runs grouped by the optimizer recipe their checkpoint records:
    family, configured LR, decoder-to-TCN LR ratio and schedule.

    Two runs in different groups were optimised differently, so a difference
    between them is not attributable to the thing an ablation means to vary.
    """
    groups: Dict[str, dict] = {}
    for r in (list_runs() if runs is None else runs):
        if r["status"] != "complete":
            continue
        key = r.get("optimizer_key") or "unknown"
        g = groups.setdefault(key, {"key": key, "label": r.get("optimizer") or "unknown",
                                    "detail": r.get("optimizer_detail") or {}, "runs": []})
        g["runs"].append(r)
    for g in groups.values():
        g["runs"].sort(key=lambda r: (r["rrmse_t"] is None, r["rrmse_t"]))
        g["n"] = len(g["runs"])
        g["known"] = g["key"] != "unknown"
    return sorted(groups.values(), key=lambda g: (not g["known"], g["key"]))


def runs_by_time(runs: Optional[List[dict]] = None) -> List[dict]:
    """
    Every run, newest first, grouped by the day it last wrote.

    The other views answer "which of these is best"; this one answers "what did
    I just run, and did it survive".  That is why it keeps the interrupted and
    stale runs the result views drop — a run that died is exactly what you come
    here looking for.

    Only the calendar date is grouped on.  Turning it into "Today" belongs to
    the page, whose clock is the one the reader is actually looking at.
    """
    groups: Dict[str, dict] = {}
    undated = []
    for r in (list_runs() if runs is None else runs):
        stamp = r.get("finished_at")
        if not stamp:
            undated.append(r)
            continue
        groups.setdefault(stamp[:10], {"date": stamp[:10], "runs": []})["runs"].append(r)
    out = []
    for g in sorted(groups.values(), key=lambda g: g["date"], reverse=True):
        g["runs"].sort(key=lambda r: r["finished_at"], reverse=True)
        g["n"] = len(g["runs"])
        # A day is only as trustworthy as its weakest timestamp; say so once per
        # heading rather than on every row.
        g["sources"] = sorted({r.get("finished_at_source") or "unknown" for r in g["runs"]})
        out.append(g)
    if undated:
        out.append({"date": None, "runs": undated, "n": len(undated), "sources": ["unknown"]})
    return out


def runs_by_kind(runs: Optional[List[dict]] = None) -> List[dict]:
    """
    Complete runs only, grouped by kind (variant x artifact), best first.

    This is the list to read results off: it leaves out interrupted runs and
    runs whose history no longer matches the weights beside it.
    """
    groups: Dict[str, dict] = {}
    for r in (list_runs() if runs is None else runs):
        if r["status"] != "complete":
            continue
        g = groups.setdefault(r["kind"], {"kind": r["kind"], "variant": r.get("variant"),
                                          "artifact": r.get("artifact"), "runs": []})
        g["runs"].append(r)
    for g in groups.values():
        g["runs"].sort(key=lambda r: (r["rrmse_t"] is None, r["rrmse_t"]))
        g["n"] = len(g["runs"])
    return sorted(groups.values(), key=lambda g: (g["artifact"] or "~", g["kind"]))
