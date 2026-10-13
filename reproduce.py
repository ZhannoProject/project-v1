#!/usr/bin/env python3
"""
Out-of-box reproducibility test for the EEG causal denoising project.

Runs entirely from scratch — no pre-trained checkpoints, no manual setup.
Designed to work both locally and inside GitHub Actions CI.

What it verifies
----------------
  1. All model variants build without errors
  2. Forward pass produces correct shapes for every variant
  3. Strict causality holds (perturbing the future doesn't change the past)
  4. Training loop runs end-to-end (tiny data, 3 epochs)
  5. Checkpoint saving / loading round-trips correctly
  6. Per-SNR evaluation pipeline runs and produces valid metrics
  7. Metrics are sane (perfect-prediction sanity check)
  8. Bidirectional teachers see the future, and knowledge distillation
     (teacher → causal student) trains end-to-end, including a smaller
     student with feature projectors
  9. The optional BottleneckSkip (skip_bottleneck_ratio) builds, replaces
     the identity skip, and keeps the model causal
 10. Seeding is deterministic: same master_seed → same weights, same
     batch order
 11. The artifact-switch probe is reproducible, realises the SNR it asks
     for in each segment, and its matched control differs from it only
     after the switch — so the response curve reads look-ahead back out
     (zero for a causal model, non-zero for one that peeks)

If real EEGdenoiseNet data is present, it uses that.
Otherwise, it synthesises small random .npy files so CI works without
downloading the ~53 MB dataset.

Usage
-----
    python reproduce.py              # run all checks
    python reproduce.py --quick      # skip training, eval-only smoke test
"""

import os
import sys
import json
import shutil
import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch

# ── Ensure project root is importable ────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.variants import (build_model, build_model_from_config, VARIANT_REGISTRY,
                             MLA_RANK_SWEEP, BIDIRECTIONAL_VARIANTS)
from models.TCN_backbone import CausalTCNBackbone, CausalConv1d
from models.AttentionBlocks.attention import (
    CausalMHA, CausalMLA, CausalTransformerBlock, CausalTransformerDecoder,
    BottleneckSkip, init_transformer_weights,
)
from metrics import compute_all_metrics, rrmse_temporal, rrmse_spectral, acc
from trainer.losses import build_loss
from data.Dataloader import build_dataloaders, EEGDenoiseDataset
from utils.utils import seed_everything

DEVICE = torch.device("cpu")  # CI has no GPU; force CPU for determinism
B, T = 4, 512
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
n_passed = 0
n_failed = 0


def report(name: str, ok: bool, detail: str = ""):
    global n_passed, n_failed
    status = PASS if ok else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")
    if ok:
        n_passed += 1
    else:
        n_failed += 1


# =====================================================================
# 1. Model construction & forward pass
# =====================================================================
def test_model_builds():
    print("\n" + "=" * 60)
    print("1. Model construction & forward pass")
    print("=" * 60)

    x = torch.randn(B, 1, T)

    all_names = (
        ["tcn", "tcn_mha", "tcn_mla", "tcn_mha_par", "tcn_mla_par"]
        + [f"tcn_mla_r{r}" for r in MLA_RANK_SWEEP]
        + BIDIRECTIONAL_VARIANTS
    )

    for name in all_names:
        try:
            model = build_model(name, hidden_channels=64)
            model.eval()
            with torch.no_grad():
                y = model(x)
            shape_ok = y.shape == x.shape
            n_params = sum(p.numel() for p in model.parameters())
            report(
                f"{name:<16s} build + forward",
                shape_ok,
                f"shape={tuple(y.shape)}, params={n_params:,}",
            )
        except Exception as e:
            report(f"{name:<16s} build + forward", False, str(e))

    # Rank sweep: the alias must win over the generic kv_lora_rank that every
    # run config carries (trainer default / YAML preset), or the four sweep
    # runs silently come out identical.  This is the path the trainer uses.
    for r in MLA_RANK_SWEEP:
        name = f"tcn_mla_r{r}"
        try:
            m = build_model_from_config({"variant": name, "hidden_channels": 64, "kv_lora_rank": 16})
            rank_built = m.decoder.blocks[0].attn.kv_down_proj.weight.shape[0]
            report(f"{name:<16s} rank via config", rank_built == r and m.model_config["kv_lora_rank"] == r,
                   f"config kv_lora_rank=16 → built rank {rank_built}, model_config={m.model_config['kv_lora_rank']}")
        except Exception as e:
            report(f"{name:<16s} rank via config", False, str(e))
    try:
        build_model("tcn_mla_r8", hidden_channels=64, kv_lora_rank=16)
        report("tcn_mla_r8 rejects kv_lora_rank=16", False, "no error raised")
    except ValueError as e:
        report("tcn_mla_r8 rejects kv_lora_rank=16", "contradicts" in str(e), str(e))

    # Optional BottleneckSkip: off by default, a real module when requested
    for name in ["tcn_mha", "tcn_mla", "tcn_mha_par", "tcn_mla_par"]:
        try:
            plain = build_model(name, hidden_channels=64)
            skip = build_model(name, hidden_channels=64, skip_bottleneck_ratio=0.5)
            off_ok = all(isinstance(b.attn_skip, torch.nn.Identity) for b in plain.decoder.blocks)
            on_ok = all(isinstance(b.attn_skip, BottleneckSkip) for b in skip.decoder.blocks)
            skip.eval()
            with torch.no_grad():
                y = skip(x)
            n_extra = sum(p.numel() for p in skip.parameters()) - sum(p.numel() for p in plain.parameters())
            report(f"{name:<16s} skip_bottleneck_ratio", off_ok and on_ok and y.shape == x.shape,
                   f"default=Identity, 0.5=BottleneckSkip (+{n_extra:,} params), "
                   f"model_config={skip.model_config.get('skip_bottleneck_ratio')}")
        except Exception as e:
            report(f"{name:<16s} skip_bottleneck_ratio", False, str(e))


# =====================================================================
# 2. Causality verification
# =====================================================================
def test_causality():
    print("\n" + "=" * 60)
    print("2. Causality verification")
    print("=" * 60)

    cases = [(name, {}) for name in ["tcn", "tcn_mha", "tcn_mla", "tcn_mha_par", "tcn_mla_par"]]
    cases += [(name, {"skip_bottleneck_ratio": 0.5}) for name in ["tcn_mha", "tcn_mla", "tcn_mla_par"]]
    for name, kw in cases:
        label = name + (" +skip" if kw else "")
        try:
            model = build_model(name, hidden_channels=64, **kw)
            model.eval()

            x1 = torch.randn(B, 1, T)
            x2 = x1.clone()
            perturb_t = T // 2
            x2[:, :, perturb_t] += 10.0

            with torch.no_grad():
                y1 = model(x1)
                y2 = model(x2)

            past_diff = (y1[:, :, :perturb_t] - y2[:, :, :perturb_t]).abs().max().item()
            future_diff = (y1[:, :, perturb_t:] - y2[:, :, perturb_t:]).abs().max().item()

            causal = past_diff < 1e-5 and future_diff > 1e-3
            report(
                f"{label:<16s} causality",
                causal,
                f"past_diff={past_diff:.2e}, future_diff={future_diff:.2e}",
            )
        except Exception as e:
            report(f"{label:<16s} causality", False, str(e))

    # Bidirectional teachers: the same perturbation MUST leak into the past.
    # Causal models give an exact 0 (masked logits → softmax weight 0).
    for name in BIDIRECTIONAL_VARIANTS:
        try:
            model = build_model(name, hidden_channels=64)
            model.eval()
            x1 = torch.randn(B, 1, T)
            x2 = x1.clone()
            x2[:, :, T // 2] += 10.0
            with torch.no_grad():
                past_diff = (model(x1)[:, :, :T // 2] - model(x2)[:, :, :T // 2]).abs().max().item()
            report(f"{name:<16s} sees the future", past_diff > 0, f"past_diff={past_diff:.2e}")
        except Exception as e:
            report(f"{name:<16s} sees the future", False, str(e))


# =====================================================================
# 2b. Seeding
# =====================================================================
def test_seeding():
    print("\n" + "=" * 60)
    print("2b. Seeding (same seed → same weights)")
    print("=" * 60)

    def build(seed):
        seed_everything(seed)
        m = build_model("tcn_mla", hidden_channels=32, kv_lora_rank=4)
        return [p.detach().clone() for p in m.parameters()]

    try:
        a, b, c = build(123), build(123), build(124)
        same = all(torch.equal(x, y) for x, y in zip(a, b))
        differ = any(not torch.equal(x, y) for x, y in zip(a, c))
        report("same master_seed → identical init", same)
        report("different master_seed → different init", differ)
    except Exception as e:
        report("seeded model init", False, str(e))

    try:
        seed_everything(7)
        r1 = torch.randn(8)
        seed_everything(7)
        r2 = torch.randn(8)
        report("torch RNG reseeded", torch.equal(r1, r2))
    except Exception as e:
        report("torch RNG reseeded", False, str(e))


# =====================================================================
# 3. Loss functions
# =====================================================================
def test_losses():
    print("\n" + "=" * 60)
    print("3. Loss functions")
    print("=" * 60)

    pred = torch.randn(B, 1, T)
    target = torch.randn(B, 1, T)

    for loss_name in ["mse", "l1", "stft", "l2_combined", "l1_combined"]:
        try:
            loss_fn = build_loss(loss_name)
            val = loss_fn(pred, target)
            ok = val.dim() == 0 and val.item() > 0 and torch.isfinite(val)
            report(f"{loss_name:<16s}", ok, f"loss={val.item():.4f}")
        except Exception as e:
            report(f"{loss_name:<16s}", False, str(e))


# =====================================================================
# 4. Metrics sanity
# =====================================================================
def test_metrics():
    print("\n" + "=" * 60)
    print("4. Metrics sanity")
    print("=" * 60)

    x = torch.randn(B, 1, T)

    # Perfect prediction
    m = compute_all_metrics(x, x)
    perfect_ok = m["rrmse_t"] < 1e-4 and m["rrmse_s"] < 1e-4 and m["acc"] > 0.999
    report("perfect prediction", perfect_ok,
           f"rrmse_t={m['rrmse_t']:.6f}, acc={m['acc']:.6f}")

    # Random prediction
    m2 = compute_all_metrics(torch.randn(B, 1, T), x)
    random_ok = m2["rrmse_t"] > 0.5 and m2["acc"] < 0.5
    report("random prediction", random_ok,
           f"rrmse_t={m2['rrmse_t']:.3f}, acc={m2['acc']:.3f}")


# =====================================================================
# 5. Data + training loop (end-to-end)
# =====================================================================
def _ensure_data(data_dir: str) -> str:
    """
    If real EEGdenoiseNet data exists, return its path.
    Otherwise, create synthetic .npy files for CI.
    """
    real_dir = os.path.join(PROJECT_ROOT, "data", "EEGdenoisenet", "data")
    if (
        os.path.isfile(os.path.join(real_dir, "EEG_all_epochs.npy"))
        and os.path.isfile(os.path.join(real_dir, "EOG_all_epochs.npy"))
        and os.path.isfile(os.path.join(real_dir, "EMG_all_epochs.npy"))
    ):
        print("  Using real EEGdenoiseNet data.")
        return real_dir

    # Synthesise small random data
    print("  Real data not found — synthesising random .npy files for CI.")
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(42)
    n_epochs, t_len = 200, 512
    np.save(os.path.join(data_dir, "EEG_all_epochs.npy"),
            rng.standard_normal((n_epochs, t_len)).astype(np.float32))
    np.save(os.path.join(data_dir, "EOG_all_epochs.npy"),
            rng.standard_normal((n_epochs, t_len)).astype(np.float32))
    np.save(os.path.join(data_dir, "EMG_all_epochs.npy"),
            rng.standard_normal((n_epochs, t_len)).astype(np.float32))
    return data_dir


def test_training_loop(tmpdir: str):
    print("\n" + "=" * 60)
    print("5. Training loop (end-to-end, 3 epochs)")
    print("=" * 60)

    data_dir = _ensure_data(os.path.join(tmpdir, "synth_data"))
    save_dir = os.path.join(tmpdir, "runs", "smoke_test")

    # Build data
    try:
        train_loader, val_loader, test_loader = build_dataloaders(
            data_dir=data_dir, artifact_type="EOG", batch_size=8,
        )
        report("data loading", True, f"train={len(train_loader.dataset)}")
    except Exception as e:
        report("data loading", False, str(e))
        return None

    # Batch order must depend on master_seed alone, not on the global RNG state
    try:
        torch.manual_seed(0)
        first_a = next(iter(build_dataloaders(data_dir=data_dir, artifact_type="EOG",
                                              batch_size=8, master_seed=42)[0]))["clean"]
        torch.manual_seed(999)
        first_b = next(iter(build_dataloaders(data_dir=data_dir, artifact_type="EOG",
                                              batch_size=8, master_seed=42)[0]))["clean"]
        first_c = next(iter(build_dataloaders(data_dir=data_dir, artifact_type="EOG",
                                              batch_size=8, master_seed=43)[0]))["clean"]
        report("batch order seeded by master_seed",
               torch.equal(first_a, first_b) and not torch.equal(first_a, first_c))
    except Exception as e:
        report("batch order seeded by master_seed", False, str(e))

    # Build model
    variant = "tcn_mla"
    model = build_model(variant, hidden_channels=32, bottleneck_ratio=0.5, kv_lora_rank=2)

    # Import trainer
    from trainer.training import Trainer

    try:
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=save_dir,
            learning_rate=1e-3,
            max_epochs=3,
            log_every=999,
            device=DEVICE,
            loss_fn="mse",
        )
        trainer.fit()
        report("training 3 epochs", True)
    except Exception as e:
        report("training 3 epochs", False, str(e))
        return None

    # Check checkpoint exists
    best_pt = os.path.join(save_dir, "best.pt")
    history_json = os.path.join(save_dir, "history.json")
    report("best.pt saved", os.path.isfile(best_pt))
    report("history.json saved", os.path.isfile(history_json))

    # Validate history
    try:
        with open(history_json) as f:
            history = json.load(f)
        has_keys = all(k in history for k in ["train_loss", "val_loss", "rrmse_t", "acc"])
        has_entries = len(history["train_loss"]) == 3
        report("history content", has_keys and has_entries,
               f"keys={list(history.keys())}, epochs={len(history['train_loss'])}")
    except Exception as e:
        report("history content", False, str(e))

    return save_dir, data_dir, variant


# =====================================================================
# 6. Checkpoint round-trip
# =====================================================================
def test_checkpoint_roundtrip(save_dir: str, variant: str):
    print("\n" + "=" * 60)
    print("6. Checkpoint round-trip")
    print("=" * 60)

    best_pt = os.path.join(save_dir, "best.pt")
    if not os.path.isfile(best_pt):
        report("checkpoint load", False, "best.pt not found")
        return

    try:
        model = build_model(variant, hidden_channels=32, bottleneck_ratio=0.5, kv_lora_rank=2)
        ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        x = torch.randn(1, 1, T)
        with torch.no_grad():
            y = model(x)

        shape_ok = y.shape == x.shape
        report("load + forward", shape_ok, f"epoch={ckpt['epoch']}")
    except Exception as e:
        report("load + forward", False, str(e))


# =====================================================================
# 7. Per-SNR evaluation pipeline
# =====================================================================
def test_eval_pipeline(data_dir: str, save_dir: str, variant: str):
    print("\n" + "=" * 60)
    print("7. Per-SNR evaluation pipeline")
    print("=" * 60)

    best_pt = os.path.join(save_dir, "best.pt")
    if not os.path.isfile(best_pt):
        report("eval pipeline", False, "best.pt not found")
        return

    try:
        model = build_model(variant, hidden_channels=32, bottleneck_ratio=0.5, kv_lora_rank=2)
        ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval().to(DEVICE)

        # Evaluate at 3 SNR levels (keep it fast)
        from eval import evaluate_per_snr
        results, avg = evaluate_per_snr(
            model, data_dir,
            artifact_type="EOG",
            snr_list=[-3, 0, 2],
            batch_size=8,
            device=DEVICE,
        )

        has_snrs = all(snr in results for snr in [-3, 0, 2])
        metrics_valid = all(
            0 < results[s]["rrmse_t"] < 10 and -1 <= results[s]["acc"] <= 1
            for s in results
        )
        report(
            "per-SNR eval",
            has_snrs and metrics_valid,
            f"avg_acc={avg['acc']:.3f}, avg_rrmse_t={avg['rrmse_t']:.3f}",
        )
    except Exception as e:
        report("per-SNR eval", False, str(e))


# =====================================================================
# 8. Knowledge distillation
# =====================================================================
def test_distillation():
    print("\n" + "=" * 60)
    print("8. Knowledge distillation (teacher → student)")
    print("=" * 60)

    from trainer.distillation import (
        DistillationLoss, FeatureTap, init_student_from_teacher,
    )

    kw = dict(hidden_channels=32, bottleneck_ratio=0.5, kv_lora_rank=4)
    teacher = build_model("tcn_mla_bi", **kw).eval()
    student = build_model("tcn_mla", **kw)

    # Identical state dicts → the student can be initialised from the teacher
    try:
        n_copied, n_skipped = init_student_from_teacher(student, teacher)
        same = all(torch.equal(a, b) for a, b in
                   zip(student.state_dict().values(), teacher.state_dict().values()))
        report("student init from teacher", n_skipped == 0 and same,
               f"{n_copied} tensors copied, {n_skipped} skipped")
    except Exception as e:
        report("student init from teacher", False, str(e))

    # Distillation loss: gradients reach the student only, feature shapes match
    try:
        x = torch.randn(B, 1, T)
        clean = torch.randn(B, 1, T)
        s_tap, t_tap = FeatureTap(student, ["decoder"]), FeatureTap(teacher, ["decoder"])
        with torch.no_grad():
            t_out = teacher(x)
        s_out = student(x)
        crit = DistillationLoss(build_loss("l1"), build_loss("l1"), alpha=0.5, feature_weight=0.1)
        total, parts = crit(s_out, t_out, clean, s_tap.features, t_tap.features)
        total.backward()
        student_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                               for p in student.parameters())
        teacher_no_grad = all(p.grad is None for p in teacher.parameters())
        feat_shape = tuple(s_tap.features["decoder"].shape)
        report("distillation loss + backward",
               torch.isfinite(total).item() and student_has_grad and teacher_no_grad,
               f"total={total.item():.4f} task={parts['task']:.4f} "
               f"output={parts['output']:.4f} feature={parts['feature']:.4f}, "
               f"decoder feature {feat_shape}")
        s_tap.remove(); t_tap.remove()
    except Exception as e:
        report("distillation loss + backward", False, str(e))

    # alpha=0, feature_weight=0 must reduce to the plain task loss
    try:
        crit0 = DistillationLoss(build_loss("mse"), build_loss("mse"), alpha=0.0, feature_weight=0.0)
        with torch.no_grad():
            total0, _ = crit0(s_out, t_out, clean)
            plain = build_loss("mse")(s_out, clean)
        report("alpha=0 reduces to task loss", torch.allclose(total0, plain),
               f"kd={total0.item():.6f} plain={plain.item():.6f}")
    except Exception as e:
        report("alpha=0 reduces to task loss", False, str(e))


def test_kd_training_loop(tmpdir: str, data_dir: str):
    print("\n" + "=" * 60)
    print("9. KD training loop (teacher → student, 2 epochs)")
    print("=" * 60)

    from trainer.training import Trainer
    from trainer.distillation import KDTrainer, load_teacher

    kw = dict(hidden_channels=32, bottleneck_ratio=0.5, kv_lora_rank=2)
    teacher_dir = os.path.join(tmpdir, "runs", "kd_teacher")
    student_dir = os.path.join(tmpdir, "runs", "kd_student")

    try:
        train_loader, val_loader, _ = build_dataloaders(
            data_dir=data_dir, artifact_type="EOG", batch_size=8,
        )
        # Teacher: 1 epoch is enough to produce a checkpoint with model_config
        Trainer(model=build_model("tcn_mla_bi", **kw), train_loader=train_loader,
                val_loader=val_loader, save_dir=teacher_dir, learning_rate=1e-3,
                max_epochs=1, log_every=999, device=DEVICE, loss_fn="mse").fit()
        teacher = load_teacher(os.path.join(teacher_dir, "best.pt"))
        report("teacher checkpoint reload", teacher.model_config["variant"] == "tcn_mla_bi"
               and not teacher.causal)
    except Exception as e:
        report("teacher checkpoint reload", False, str(e))
        return

    try:
        kd = KDTrainer(
            model=build_model("tcn_mla", **kw), teacher=teacher,
            train_loader=train_loader, val_loader=val_loader,
            kd_alpha=0.5, kd_feature_weight=0.1, init_from_teacher=True,
            save_dir=student_dir, learning_rate=1e-3, max_epochs=2,
            log_every=999, device=DEVICE, loss_fn="mse",
        )
        kd.fit()
        report("KD training 2 epochs", True)
    except Exception as e:
        report("KD training 2 epochs", False, str(e))
        return

    try:
        with open(os.path.join(student_dir, "history.json")) as f:
            history = json.load(f)
        kd_keys = ["kd_task", "kd_output", "kd_feature"]
        ok = (all(len(history[k]) == 2 for k in kd_keys)
              and "teacher" in history and "rrmse_t" in history["teacher"])
        report("KD history content", ok,
               f"kd_output={history['kd_output'][-1]:.4f}, "
               f"teacher acc={history['teacher']['acc']:.3f}")
    except Exception as e:
        report("KD history content", False, str(e))

    # The distilled student must still be strictly causal
    try:
        ckpt = torch.load(os.path.join(student_dir, "best.pt"), map_location="cpu", weights_only=False)
        student = build_model("tcn_mla", **kw)
        student.load_state_dict(ckpt["model_state"])
        student.eval()
        x1 = torch.randn(1, 1, T)
        x2 = x1.clone()
        x2[:, :, T // 2] += 10.0
        with torch.no_grad():
            past_diff = (student(x1)[:, :, :T // 2] - student(x2)[:, :, :T // 2]).abs().max().item()
        report("distilled student is causal", past_diff < 1e-5,
               f"past_diff={past_diff:.2e}, variant={ckpt['model_config']['variant']}")
    except Exception as e:
        report("distilled student is causal", False, str(e))

    # Optional compression: a narrower, shallower student with feature KD through projectors
    small_dir = os.path.join(tmpdir, "runs", "kd_student_small")
    try:
        small_kw = dict(hidden_channels=16, bottleneck_ratio=0.5, kv_lora_rank=2, n_attn_layers=1)
        kd = KDTrainer(
            model=build_model("tcn_mla", **small_kw), teacher=teacher,
            train_loader=train_loader, val_loader=val_loader,
            kd_alpha=0.5, kd_feature_weight=0.1,
            kd_feature_layers=["decoder", "decoder.blocks.0:decoder.blocks.1"],
            init_from_teacher=True,
            save_dir=small_dir, learning_rate=1e-3, max_epochs=1,
            log_every=999, device=DEVICE, loss_fn="mse",
        )
        kd.fit()
        n_s, n_t = kd.history["student"]["n_params"], kd.history["teacher"]["n_params"]
        ckpt = torch.load(os.path.join(small_dir, "best.pt"), map_location="cpu", weights_only=False)
        rebuilt = build_model("tcn_mla", **small_kw)
        rebuilt.load_state_dict(ckpt["model_state"], strict=True)     # no projector keys
        ok = n_s < n_t and len(kd.kd_projectors) == 2 and ckpt["model_config"]["hidden_channels"] == 16
        report("KD compressed student", ok,
               f"student {n_s:,} / teacher {n_t:,} params ({n_t / n_s:.1f}x smaller), "
               f"{len(kd.kd_projectors)} feature projectors, checkpoint clean")
    except Exception as e:
        report("KD compressed student", False, str(e))


# =====================================================================
# 10. Config loading
# =====================================================================
def test_configs():
    print("\n" + "=" * 60)
    print("10. Config file loading")
    print("=" * 60)

    from configs import load_config
    config_dir = PROJECT_ROOT / "configs"
    yamls = list(config_dir.glob("*.yaml"))

    if not yamls:
        report("config files found", False, "no .yaml files in configs/")
        return

    for ypath in yamls:
        try:
            cfg = load_config(str(ypath))
            has_variant = "variant" in cfg
            report(f"{ypath.name}", has_variant, f"variant={cfg.get('variant', '?')}")
        except Exception as e:
            report(f"{ypath.name}", False, str(e))


# =====================================================================
# 11. Non-stationary artifact-switch probe
# =====================================================================
def test_switch_probe(data_dir: str):
    print("\n" + "=" * 60)
    print("11. Artifact-switch probe + response curve")
    print("=" * 60)

    import eval as evaltools
    from data.Dataloader import ArtifactSwitchDataset, SWITCH_PROBE_SEEDS

    def probe(**kw):
        kw.setdefault("n_epochs", 16)
        return ArtifactSwitchDataset(data_dir, **kw)

    # --- reproducibility of the probe series ---
    try:
        a, b, c = probe(probe_seed=0), probe(probe_seed=0), probe(probe_seed=1)
        same = (np.array_equal(a.switch_idx, b.switch_idx)
                and torch.equal(a[1]["noisy"], b[1]["noisy"]))
        moved = not np.array_equal(a.switch_idx, c.switch_idx)
        report("probe: same probe_seed → identical epochs", same)
        report("probe: different probe_seed → different switch points", moved,
               f"seeds={list(SWITCH_PROBE_SEEDS)}")
    except Exception as e:
        report("probe: reproducibility", False, str(e))
        return

    # --- __getitem__ must not depend on access order ---
    try:
        first = a[2]["noisy"].clone()
        for j in (5, 0, 2, 9):
            a[j]
        report("probe: __getitem__ is a pure function of the index",
               torch.equal(first, a[2]["noisy"]))
    except Exception as e:
        report("probe: __getitem__ is a pure function of the index", False, str(e))

    # --- the design: randomised switch, balanced directions ---
    try:
        lo, hi = a.switch_bounds
        inside = bool((a.switch_idx >= lo).all() and (a.switch_idx <= hi).all())
        varying = len(np.unique(a.switch_idx)) > 1
        report("probe: switch point randomised inside its range", inside and varying,
               f"[{lo}, {hi}], {len(np.unique(a.switch_idx))} distinct")
        report("probe: directions exactly balanced",
               int(a.eog_first.sum()) == len(a) // 2)
        report("probe: the two SNRs are drawn independently",
               not np.allclose(a.snr_eog, a.snr_emg))
    except Exception as e:
        report("probe: design", False, str(e))

    # --- the SNR asked for is the SNR realised, per segment ---
    try:
        worst = 0.0
        for i in range(len(a)):
            k = int(a.switch_idx[i])
            s = a[i]
            clean = a.clean[i]
            art = s["noisy"].squeeze().numpy() * s["sigma"].item() - clean

            def snr(x, y):
                return 20 * np.log10(np.sqrt(np.mean(x ** 2)) / np.sqrt(np.mean(y ** 2)))

            want_pre = a.snr_eog[i] if a.eog_first[i] else a.snr_emg[i]
            want_post = a.snr_emg[i] if a.eog_first[i] else a.snr_eog[i]
            worst = max(worst, abs(snr(clean[:k], art[:k]) - want_pre),
                        abs(snr(clean[k:], art[k:]) - want_post))
        report("probe: each segment realises its own requested SNR", worst < 1e-2,
               f"max error {worst:.1e} dB")
    except Exception as e:
        report("probe: each segment realises its own requested SNR", False, str(e))

    # --- the matched control differs only after the switch ---
    try:
        ctrl = probe(probe_seed=0, control=True)
        pre_gap = post_gap = 0.0
        for i in range(len(a)):
            k = int(a.switch_idx[i])
            x = a[i]["noisy"].squeeze().numpy()
            y = ctrl[i]["noisy"].squeeze().numpy()
            pre_gap = max(pre_gap, np.abs(x[:k] - y[:k]).max())
            post_gap = max(post_gap, np.abs(x[k:] - y[k:]).max())
        report("probe: control is bit-identical before the switch", pre_gap == 0.0)
        report("probe: control splices something different after it", post_gap > 0.0)
    except Exception as e:
        report("probe: matched control", False, str(e))

    # --- the response curve reads causality back out ---
    class _Causal(torch.nn.Module):
        def forward(self, x):
            return 0.5 * (x + torch.nn.functional.pad(x, (1, 0))[..., :-1])

    class _Peeking(torch.nn.Module):
        def forward(self, x):
            return 0.5 * (x + torch.nn.functional.pad(x, (0, 1))[..., 1:])

    try:
        kw = dict(probe_seeds=(0, 1), n_epochs=16, batch_size=8,
                  device=DEVICE, verbose=False)
        causal = evaltools.evaluate_switch_response(_Causal(), data_dir, **kw)
        peeking = evaltools.evaluate_switch_response(_Peeking(), data_dir, **kw)

        off = causal["offsets"]
        report("switch curve: offsets symmetric around the switch",
               off[0] == -off[-1] and 0 in off and len(off) == len(causal["all"]["err"]),
               f"±{off[-1]} samples")
        report("switch curve: causal model leaks nothing before the switch",
               causal["all"]["summary"]["pre_leak"] == 0.0)
        report("switch curve: non-causal model does leak",
               peeking["all"]["summary"]["pre_leak"] > 0.0,
               f"pre_leak={peeking['all']['summary']['pre_leak']:.4f}")
        report("switch curve: per-direction breakdown present",
               all(d in causal for d in ("all", "eog_to_emg", "emg_to_eog")))
        keys = {"pre", "post_floor", "peak", "overshoot", "delta_floor",
                "transient_peak", "transient_area"}
        report("switch curve: summary carries the reported fields",
               keys <= set(causal["all"]["summary"]))
        json.dumps(evaltools._switch_result_to_json(causal))
        report("switch curve: result is JSON-serialisable", True)
    except Exception as e:
        report("switch curve", False, str(e))


# =====================================================================
# Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Reproducibility smoke test")
    parser.add_argument("--quick", action="store_true",
                        help="Skip training loop (fast, ~10s)")
    args = parser.parse_args()

    global n_passed, n_failed

    print("=" * 60)
    print("  EEG Causal Denoising — Reproducibility Test")
    print("=" * 60)
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  NumPy   : {np.__version__}")
    print(f"  Device  : {DEVICE}")

    # Always run these (fast, no data needed)
    test_model_builds()
    test_causality()
    test_seeding()
    test_losses()
    test_metrics()
    test_distillation()
    test_configs()

    if not args.quick:
        tmpdir = tempfile.mkdtemp(prefix="eeg_reproduce_")
        try:
            result = test_training_loop(tmpdir)
            if result is not None:
                save_dir, data_dir, variant = result
                test_checkpoint_roundtrip(save_dir, variant)
                test_eval_pipeline(data_dir, save_dir, variant)
                test_switch_probe(data_dir)
                test_kd_training_loop(tmpdir, data_dir)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        print("\n  [SKIP] Training / eval tests (--quick mode)")

    # Summary
    total = n_passed + n_failed
    print("\n" + "=" * 60)
    print(f"  RESULTS: {n_passed}/{total} passed, {n_failed} failed")
    print("=" * 60)

    if n_failed > 0:
        sys.exit(1)
    else:
        print("\n  All checks passed. The project is reproducible.")
        sys.exit(0)


if __name__ == "__main__":
    main()
