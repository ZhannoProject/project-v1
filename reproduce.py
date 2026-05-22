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

from models.variants import build_model, VARIANT_REGISTRY, MLA_RANK_SWEEP
from models.TCN_backbone import CausalTCNBackbone, CausalConv1d
from models.AttentionBlocks.attention import (
    CausalMHA, CausalMLA, CausalTransformerBlock, CausalTransformerDecoder,
    init_transformer_weights,
)
from metrics import compute_all_metrics, rrmse_temporal, rrmse_spectral, acc
from trainer.losses import build_loss
from data.Dataloader import build_dataloaders, EEGDenoiseDataset

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


# =====================================================================
# 2. Causality verification
# =====================================================================
def test_causality():
    print("\n" + "=" * 60)
    print("2. Causality verification")
    print("=" * 60)

    for name in ["tcn", "tcn_mha", "tcn_mla", "tcn_mha_par", "tcn_mla_par"]:
        try:
            model = build_model(name, hidden_channels=64)
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
                f"{name:<16s} causality",
                causal,
                f"past_diff={past_diff:.2e}, future_diff={future_diff:.2e}",
            )
        except Exception as e:
            report(f"{name:<16s} causality", False, str(e))


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
# 8. Config loading
# =====================================================================
def test_configs():
    print("\n" + "=" * 60)
    print("8. Config file loading")
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
    test_losses()
    test_metrics()
    test_configs()

    if not args.quick:
        tmpdir = tempfile.mkdtemp(prefix="eeg_reproduce_")
        try:
            result = test_training_loop(tmpdir)
            if result is not None:
                save_dir, data_dir, variant = result
                test_checkpoint_roundtrip(save_dir, variant)
                test_eval_pipeline(data_dir, save_dir, variant)
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
