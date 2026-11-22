"""Data provenance shared by training, held-out evaluation and distillation."""

import math


DEFAULT_SPLIT = (0.7, 0.15, 0.15)


def normalization_config(config):
    """Missing metadata means the historical offline, per-epoch protocol."""
    mode = config.get("normalization", "epoch")
    scale = config.get("normalization_scale")
    if mode not in ("epoch", "fixed"):
        raise ValueError(f"Unknown normalization: {mode!r}")
    if mode == "fixed":
        if scale is None or not math.isfinite(float(scale)) or float(scale) <= 0:
            raise ValueError("fixed normalization requires a finite, positive normalization_scale")
        scale = float(scale)
    elif scale is not None:
        raise ValueError("normalization_scale is only supported with normalization='fixed'")
    return {"normalization": mode, "normalization_scale": scale}


def bind_run_config(model, checkpoint, fallback=None):
    """Attach provenance without mixing it into architecture constructor arguments."""
    cfg = dict(fallback or {})
    cfg.update((checkpoint.get("history") or {}).get("config") or {})
    cfg.update(checkpoint.get("run_config") or {})
    model.run_config = cfg
    model.checkpoint_loaded = True
    return model


def dataset_protocol(dataset):
    return {
        "master_seed": dataset.master_seed,
        "artifact": dataset.artifact_type,
        "split_ratios": list(dataset.split_ratios),
        "target_fs": dataset.target_fs,
        "snr_range_db": list(dataset.snr_range_db),
        "sampling": dataset.sampling,
        "normalization": dataset.normalization,
        "normalization_scale": dataset.normalization_scale,
        "resampling": getattr(dataset, "resampling", "offline"),
    }


def evaluation_data(model, seed=None, artifact=None):
    """Resolve the held-out protocol; explicit split overrides must agree."""
    cfg = dict(getattr(model, "run_config", {}) or {})
    recorded_seed = cfg.get("master_seed")
    if recorded_seed is not None:
        if seed is not None and int(seed) != int(recorded_seed):
            raise ValueError(f"master_seed={seed} differs from checkpoint master_seed={recorded_seed}; "
                             "this would evaluate on a different split, including training epochs")
        seed = int(recorded_seed)
    elif seed is None:
        if getattr(model, "checkpoint_loaded", False):
            raise ValueError("Checkpoint has no recorded master_seed; supply the original training "
                             "seed explicitly (or a legacy training config). Do not guess a test split.")
        seed = 42                 # untrained/hand-built models, not checkpoint evaluation
    recorded_artifact = cfg.get("artifact")
    if artifact is not None and recorded_artifact and artifact != recorded_artifact:
        raise ValueError(f"artifact={artifact} differs from checkpoint artifact={recorded_artifact}; "
                         "artifact pools use different split seeds")
    artifact = recorded_artifact or artifact or "EOG"
    return dict(master_seed=int(seed), artifact_type=artifact,
                split_ratios=tuple(cfg.get("split_ratios", DEFAULT_SPLIT)),
                snr_range_db=tuple(cfg.get("snr_range_db", (-7., 2.))),
                sampling=cfg.get("sampling", "legacy"), target_fs=cfg.get("target_fs"),
                resampling=cfg.get("resampling", "offline"),
                **normalization_config(cfg))


def validate_distillation_data(teacher, train_loader, val_loader, student_config=None):
    """Check before copying weights or running a teacher on student-held-out data."""
    cfg = dict(getattr(teacher, "run_config", {}) or {})
    train_ds = getattr(train_loader, "dataset", None)
    val_ds = getattr(val_loader, "dataset", None)
    real_data = hasattr(train_ds, "master_seed")
    # Hand-built teachers and synthetic loaders are useful in unit tests and research.
    if not real_data and not getattr(teacher, "checkpoint_loaded", False) and not cfg:
        return
    student = dataset_protocol(train_ds) if real_data else dict(student_config or {})
    if cfg.get("master_seed") is None or student.get("master_seed") is None:
        raise ValueError("Cannot verify teacher/student split: master_seed is missing. "
                         "For a legacy teacher, supply --teacher_config with its original data settings.")
    if not cfg.get("artifact") or not student.get("artifact"):
        raise ValueError("Cannot verify teacher/student artifact pools: artifact is missing")
    keys = ("master_seed", "artifact", "split_ratios", "target_fs", "sampling",
            "normalization", "normalization_scale", "resampling")
    def canonical(c):
        return dict(master_seed=int(c["master_seed"]), artifact=c["artifact"],
                    split_ratios=tuple(c.get("split_ratios", DEFAULT_SPLIT)),
                    target_fs=float(c.get("target_fs") or 256.),
                    sampling=c.get("sampling", "legacy"), resampling=c.get("resampling", "offline"),
                    **normalization_config(c))
    teacher_data, student_data = canonical(cfg), canonical(student)
    for key in keys:
        if teacher_data[key] != student_data[key]:
            raise ValueError(f"Teacher/student data mismatch: {key}: teacher={teacher_data[key]!r}, "
                             f"student={student_data[key]!r}. Use a teacher trained on the same "
                             "split and preprocessing protocol; otherwise held-out data may leak.")
    if real_data:
        if getattr(train_ds, "split", None) != "train" or getattr(val_ds, "split", None) != "val":
            raise ValueError("Distillation requires train and val datasets, respectively")
        actual_val = canonical(dataset_protocol(val_ds))
        if actual_val != student_data:
            raise ValueError("Student train/validation data protocols differ")
