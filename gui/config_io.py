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
        Field("master_seed", "int", 42, min=0, help="Split / pairing / SNR seed"),
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
              help="Checkpoint folder; empty → runs/<variant>_<artifact>[_kd]"),
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
              help="Submodules to match, comma-separated (e.g. decoder, decoder.blocks.0)"),
        Field("kd_output_loss", "select", None, choices="losses", nullable=True, show_if="kd",
              help="Loss vs teacher output; empty → same as loss_fn"),
        Field("kd_feature_loss", "select", "mse", choices="feature_losses", show_if="kd"),
        Field("init_from_teacher", "bool", False, show_if="kd",
              help="Copy the teacher's weights into the student first"),
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
def list_runs() -> List[dict]:
    """One row per runs/<dir> that holds a best.pt; history summary if present."""
    rows = []
    if not RUNS_DIR.is_dir():
        return rows
    for d in sorted(RUNS_DIR.iterdir()):
        best = d / "best.pt"
        if not d.is_dir() or not best.is_file():
            continue
        row = {"name": d.name, "checkpoint": f"./runs/{d.name}/best.pt",
               "bidirectional": d.name.endswith("_bi") or "_bi_" in d.name,
               "epochs": None, "best_epoch": None, "val_loss": None,
               "rrmse_t": None, "rrmse_s": None, "acc": None, "kd": False}
        hist = d / "history.json"
        if hist.is_file():
            try:
                h = json.loads(hist.read_text())
                vl = h.get("val_loss") or []
                if vl:
                    i = min(range(len(vl)), key=vl.__getitem__)
                    row.update(epochs=len(vl), best_epoch=i + 1, val_loss=vl[i],
                               rrmse_t=h["rrmse_t"][i], rrmse_s=h["rrmse_s"][i], acc=h["acc"][i],
                               kd="kd_output" in h)
            except (ValueError, KeyError, IndexError):
                pass
        rows.append(row)
    return rows
