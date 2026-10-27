"""
Unified evaluation script for EEG causal denoising models.

Seven tools in one CLI:
  1. Per-SNR evaluator              — quantitative metrics across noise levels
  2. Training curve plotter         — loss and metric curves from history.json
  3. Waveform visualizer            — time-domain + PSD comparison plots
  4. Attention heatmap plotter      — per-head, per-layer attention weight maps
  5. Per-SNR metric line chart      — RRMSE_t, RRMSE_s, ACC plotted over SNR
  6. Multi-model training comparison— overlay training curves from multiple runs
  7. Multi-model SNR comparison     — overlay per-SNR metrics from multiple models

Usage:
    # Run all single-model tools:
    python eval.py --ckpt runs/tcn_mha_eog/best.pt --data_dir ./data/EEGdenoisenet/data/ \\
                   --variant tcn_mha --artifact EOG

    # Training curves only (no checkpoint needed):
    python eval.py --history runs/tcn_mha_eog/history.json --mode curves

    # Compare multiple models (training curves):
    python eval.py --mode compare_curves \\
                   --compare_runs runs/tcn_eog runs/tcn_mha_eog runs/tcn_mla_eog

    # Compare multiple models (per-SNR metrics):
    python eval.py --mode compare_snr --data_dir ./data/EEGdenoisenet/data/ \\
                   --compare_runs runs/tcn_eog:tcn runs/tcn_mha_eog:tcn_mha runs/tcn_mla_eog:tcn_mla
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

from data.Dataloader import EEGDenoiseDataset
from torch.utils.data import DataLoader
from metrics import compute_all_metrics
from models.variants import build_model, build_model_from_config, VARIANT_REGISTRY
from utils.utils import get_device

DEFAULT_MASTER_SEED = 42


# ====================================================================
# 0. Which split belongs to this run
# ====================================================================
def resolve_master_seed(ckpt: Optional[dict] = None, history_path: Optional[str] = None,
                        override: Optional[int] = None, *, what: str = "evaluation") -> int:
    """
    The ``master_seed`` whose splits this checkpoint was trained against.

    ``master_seed`` *defines* the train/val/test partition (data/Dataloader.py),
    so evaluating a model on a different seed's "test" split is not a weaker
    measurement -- it is a contaminated one.  The splits of two seeds overlap
    almost completely: ~85% of one seed's test epochs were in the other seed's
    train or val set, so the number that comes out is partly train accuracy.

    Resolution order is explicit > recorded > the historical default, and the
    last case warns, because silently assuming 42 for a run trained at another
    seed is exactly the failure this function exists to prevent.
    """
    if override is not None:
        return int(override)

    for source, cfg in (("checkpoint run_config", (ckpt or {}).get("run_config")),
                        ("checkpoint model_config", (ckpt or {}).get("model_config"))):
        if isinstance(cfg, dict) and cfg.get("master_seed") is not None:
            seed = int(cfg["master_seed"])
            print(f"[Eval] master_seed = {seed} (from the {source})")
            return seed

    if history_path:
        try:
            with open(history_path) as fh:
                cfg = (json.load(fh) or {}).get("config") or {}
            if cfg.get("master_seed") is not None:
                seed = int(cfg["master_seed"])
                print(f"[Eval] master_seed = {seed} (from history.json)")
                return seed
        except (OSError, ValueError):
            pass

    print(f"[Eval] WARNING: this run records no master_seed, so {what} falls back to "
          f"{DEFAULT_MASTER_SEED}. If it was trained with --master_seed other than "
          f"{DEFAULT_MASTER_SEED}, these splits are NOT its held-out data -- around 85% of "
          f"them were in its training or validation set. Pass --master_seed to say which.")
    return DEFAULT_MASTER_SEED


# ====================================================================
# 1. Per-SNR Evaluator
# ====================================================================

@torch.no_grad()
def evaluate_per_snr(
    model,
    data_dir: str,
    artifact_type: str = "EOG",
    snr_list: list = None,
    batch_size: int = 64,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> tuple:
    """
    Evaluates the model across a predefined list of SNRs to assess
    performance under varying noise conditions.

    Returns
    -------
    results : dict[int, dict]   — per-SNR metric dicts
    avg     : dict              — global averages across all SNRs
    """
    if snr_list is None:
        snr_list = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]

    device = device or get_device()
    model.eval().to(device)
    results = {}

    print(f"\n{'='*60}")
    print(f"Evaluation graded by SNR ({artifact_type})")
    print(f"{'='*60}")
    print(f"{'SNR':>6} | {'RRMSE_t':>8} | {'RRMSE_s':>8} | {'ACC':>6}")
    print(f"{'-'*6:>6}-+-{'-'*8:>8}-+-{'-'*8:>8}-+-{'-'*6:>6}")

    for snr in snr_list:
        ds = EEGDenoiseDataset(
            data_dir=data_dir,
            artifact_type=artifact_type,
            split="test",
            fixed_snr_db=float(snr),
            master_seed=seed,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        agg = {"rrmse_t": 0.0, "rrmse_s": 0.0, "acc": 0.0}
        n_seen = 0

        for batch in loader:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            pred = model(noisy)

            metrics = compute_all_metrics(pred, clean)
            bs = noisy.size(0)
            for k, v in metrics.items():
                agg[k] += v * bs
            n_seen += bs

        for k in agg:
            agg[k] /= n_seen
        results[snr] = agg

        print(f"{snr:>6} | {agg['rrmse_t']:>8.3f} | "
              f"{agg['rrmse_s']:>8.3f} | {agg['acc']:>6.3f}")

    avg = {k: np.mean([r[k] for r in results.values()])
           for k in ["rrmse_t", "rrmse_s", "acc"]}
    print(f"{'-'*6:>6}-+-{'-'*8:>8}-+-{'-'*8:>8}-+-{'-'*6:>6}")
    print(f"{'AVG':>6} | {avg['rrmse_t']:>8.3f} | "
          f"{avg['rrmse_s']:>8.3f} | {avg['acc']:>6.3f}")

    return results, avg


# ====================================================================
# 2. Training Curve Plotter
# ====================================================================

def plot_training_curves(
    history_path: str,
    save_path: Optional[str] = None,
):
    """
    Load a ``history.json`` file and plot training / validation loss
    and the three evaluation metrics (RRMSE_t, RRMSE_s, ACC) over epochs.

    Layout: 2 x 2 grid
      [0,0] train_loss + val_loss     [0,1] RRMSE_temporal
      [1,0] RRMSE_spectral            [1,1] ACC

    Parameters
    ----------
    history_path : str
        Path to the history.json saved by the Trainer.
    save_path : str or None
        If provided, saves the figure to this path instead of displaying it.
    """
    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Training Curves — {Path(history_path).parent.name}", fontsize=14)

    # --- [0,0] Loss curves ---
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="train_loss", color="#2980b9", linewidth=1.2)
    ax.plot(epochs, history["val_loss"],   label="val_loss",   color="#e74c3c", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- [0,1] RRMSE temporal ---
    ax = axes[0, 1]
    ax.plot(epochs, history["rrmse_t"], color="#8e44ad", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RRMSE_t")
    ax.set_title("RRMSE Temporal (lower = better)")
    ax.grid(True, alpha=0.3)

    # --- [1,0] RRMSE spectral ---
    ax = axes[1, 0]
    ax.plot(epochs, history["rrmse_s"], color="#d35400", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RRMSE_s")
    ax.set_title("RRMSE Spectral (lower = better)")
    ax.grid(True, alpha=0.3)

    # --- [1,1] ACC ---
    ax = axes[1, 1]
    ax.plot(epochs, history["acc"], color="#27ae60", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ACC")
    ax.set_title("Correlation Coefficient (higher = better)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Curves] Figure saved: {save_path}")
    else:
        plt.show()


# ====================================================================
# 3. Waveform Visualizer
# ====================================================================

@torch.no_grad()
def visualize_samples(
    model,
    data_dir: str,
    artifact_type: str = "EOG",
    snr: float = -3.0,
    n_samples: int = 4,
    save_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    seed: int = DEFAULT_MASTER_SEED,
):
    """
    Plots noisy, clean (ground-truth), and denoised signals side by side
    in both the time domain and frequency domain (PSD via Welch's method).

    ``seed`` must be the run's own ``master_seed``: it selects the test split
    these waveforms are drawn from, and a mismatched seed plots epochs the
    model was trained on.
    """
    device = device or get_device()
    model.eval().to(device)
    fs = 256  # Sampling frequency

    ds = EEGDenoiseDataset(
        data_dir=data_dir,
        artifact_type=artifact_type,
        split="test",
        fixed_snr_db=float(snr),
        master_seed=seed,
    )

    # Spread the samples across whatever the split actually holds.  A fixed
    # stride of 100 overruns any split smaller than n_samples * 100 -- which
    # includes every split of a small artifact set, and the synthetic data
    # reproduce.py falls back to.
    if len(ds) == 0:
        print(f"[Waveform] the {artifact_type} test split is empty — skipping.")
        return
    n_samples = min(n_samples, len(ds))
    stride = max(1, len(ds) // n_samples)

    fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 6), squeeze=False)
    fig.suptitle(
        f"{artifact_type} Denoising Performance Comparison (SNR = {snr} dB)",
        fontsize=13,
    )

    for i in range(n_samples):
        sample = ds[(i * stride) % len(ds)]
        noisy = sample["noisy"].unsqueeze(0).to(device)
        clean = sample["clean"].squeeze().cpu().numpy()
        sigma = sample["sigma"].item()

        pred = model(noisy).squeeze().cpu().numpy()
        noisy_np = sample["noisy"].squeeze().cpu().numpy()

        # Restore original amplitude
        clean_real = clean * sigma
        noisy_real = noisy_np * sigma
        pred_real = pred * sigma

        t = np.arange(len(clean)) / fs

        # --- Row 1: Time domain ---
        ax = axes[0, i]
        ax.plot(t, noisy_real, color="#888", alpha=0.6, linewidth=0.7, label="noisy")
        ax.plot(t, clean_real, color="#16a085", linewidth=1.0, label="clean (GT)")
        ax.plot(t, pred_real, color="#c0392b", linewidth=1.0, label="denoised", alpha=0.85)
        ax.set_title(f"Sample {i+1}")
        ax.set_xlabel("Time (s)")
        if i == 0:
            ax.set_ylabel("Amplitude")
            ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

        # --- Row 2: Frequency domain (PSD) ---
        ax = axes[1, i]
        for sig, c, lab, alpha in [
            (noisy_real, "#888", "noisy", 0.6),
            (clean_real, "#16a085", "clean", 1.0),
            (pred_real, "#c0392b", "denoised", 0.85),
        ]:
            f_psd, pxx = welch(sig, fs=fs, nperseg=256)
            ax.semilogy(f_psd, pxx, color=c, alpha=alpha, linewidth=0.9, label=lab)
        ax.set_xlabel("Frequency (Hz)")
        if i == 0:
            ax.set_ylabel("PSD")
            ax.legend(fontsize=8)
        ax.set_xlim(0, 64)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Waveform] Figure saved: {save_path}")
    else:
        plt.show()


# ====================================================================
# 4. Attention Weight Extractor + Heatmap Plotter
# ====================================================================

def _get_attn_modules(model) -> List:
    """
    Walk the model tree and return every CausalMHA / CausalMLA module.
    Returns a list of (layer_idx, module) tuples.
    """
    from models.AttentionBlocks.attention import CausalMHA, CausalMLA

    attn_modules = []
    if not hasattr(model, "decoder"):
        return attn_modules

    for i, block in enumerate(model.decoder.blocks):
        if isinstance(block.attn, (CausalMHA, CausalMLA)):
            attn_modules.append((i, block.attn))
    return attn_modules


@torch.no_grad()
def plot_attention_heatmaps(
    model,
    data_dir: str,
    artifact_type: str = "EOG",
    snr: float = -3.0,
    sample_idx: int = 0,
    save_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    seed: int = DEFAULT_MASTER_SEED,
):
    """
    Extract attention weights from every layer / head and plot heatmaps.

    Layout: one row per transformer layer, one column per attention head.
    Each heatmap is a (T, T) causal attention matrix — position i's
    row shows how much it attends to positions [0..i].

    Only applicable to models with a ``decoder`` attribute (sandwich
    architectures).  TCN-only models are silently skipped.

    Parameters
    ----------
    model : nn.Module
        A trained model variant (e.g. TCNWithMHA, TCNWithMLA).
    data_dir : str
        Path to the EEGdenoisenet data directory.
    artifact_type : str
        Artifact type for test data.
    snr : float
        SNR level for the test sample.
    sample_idx : int
        Which test sample to use (default: 0).
    save_path : str or None
        If provided, saves the figure instead of displaying it.
    seed : int
        The run's own ``master_seed``, which selects the test split the sample
        is drawn from. A mismatched seed shows attention over an epoch the
        model was trained on.
    """
    attn_modules = _get_attn_modules(model)
    if not attn_modules:
        print("[Attention] No attention layers found — skipping heatmap.")
        return

    device = device or get_device()
    model.eval().to(device)

    # Enable weight capture on all attention layers
    for _, attn in attn_modules:
        attn.capture_weights(True)

    # Run a single forward pass to populate the attention weights
    ds = EEGDenoiseDataset(
        data_dir=data_dir,
        artifact_type=artifact_type,
        split="test",
        fixed_snr_db=float(snr),
        master_seed=seed,
    )
    sample = ds[sample_idx]
    noisy = sample["noisy"].unsqueeze(0).to(device)  # (1, 1, T)
    _ = model(noisy)

    # Collect weights: list of (layer_idx, weights (1, n_heads, T, T))
    all_weights = []
    for layer_idx, attn in attn_modules:
        w = attn.last_attn_weights  # (1, n_heads, T, T)
        if w is not None:
            all_weights.append((layer_idx, w[0]))  # drop batch dim → (n_heads, T, T)

    # Disable capture
    for _, attn in attn_modules:
        attn.capture_weights(False)

    if not all_weights:
        print("[Attention] No attention weights captured.")
        return

    n_layers = len(all_weights)
    n_heads = all_weights[0][1].shape[0]

    fig, axes = plt.subplots(
        n_layers, n_heads,
        figsize=(3.5 * n_heads, 3 * n_layers),
        squeeze=False,
    )

    attn_type = model.decoder.attention_type.upper()
    fig.suptitle(
        f"{attn_type} Attention Heatmaps (SNR={snr} dB, sample={sample_idx})",
        fontsize=13,
    )

    for row, (layer_idx, weights) in enumerate(all_weights):
        for head in range(n_heads):
            ax = axes[row, head]
            w = weights[head].numpy()  # (T, T)

            im = ax.imshow(w, aspect="auto", cmap="viridis", vmin=0)
            ax.set_title(f"Layer {layer_idx} Head {head}", fontsize=9)

            if head == 0:
                ax.set_ylabel("Query pos")
            if row == n_layers - 1:
                ax.set_xlabel("Key pos")

            # Thin tick labels for large T
            T = w.shape[0]
            tick_step = max(T // 4, 1)
            ticks = list(range(0, T, tick_step))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.tick_params(labelsize=7)

    fig.colorbar(im, ax=axes, shrink=0.6, label="Attention weight")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Attention] Figure saved: {save_path}")
    else:
        plt.show()


# ====================================================================
# 5. Per-SNR Metric Line Chart
# ====================================================================

def plot_snr_curves(
    results: Dict[int, dict],
    title: str = "Per-SNR Metrics",
    save_path: Optional[str] = None,
):
    """
    Plot RRMSE_t, RRMSE_s, and ACC as line charts with SNR (dB) on the x-axis.

    Parameters
    ----------
    results : dict[int, dict]
        Output from ``evaluate_per_snr()`` — keys are SNR values, values are
        metric dicts with keys "rrmse_t", "rrmse_s", "acc".
    title : str
        Figure title.
    save_path : str or None
        If provided, saves the figure to this path instead of displaying it.
    """
    snrs = sorted(results.keys())
    rrmse_t = [results[s]["rrmse_t"] for s in snrs]
    rrmse_s = [results[s]["rrmse_s"] for s in snrs]
    accs = [results[s]["acc"] for s in snrs]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(title, fontsize=13)

    # RRMSE temporal
    ax = axes[0]
    ax.plot(snrs, rrmse_t, marker="o", color="#8e44ad", linewidth=1.5)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("RRMSE_t")
    ax.set_title("RRMSE Temporal (↓ better)")
    ax.grid(True, alpha=0.3)

    # RRMSE spectral
    ax = axes[1]
    ax.plot(snrs, rrmse_s, marker="s", color="#d35400", linewidth=1.5)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("RRMSE_s")
    ax.set_title("RRMSE Spectral (↓ better)")
    ax.grid(True, alpha=0.3)

    # ACC
    ax = axes[2]
    ax.plot(snrs, accs, marker="^", color="#27ae60", linewidth=1.5)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("ACC")
    ax.set_title("Correlation Coefficient (↑ better)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SNR Curves] Figure saved: {save_path}")
    else:
        plt.show()


# ====================================================================
# 6. Multi-Model Training Comparison
# ====================================================================

def compare_training_curves(
    run_dirs: List[str],
    labels: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    """
    Overlay training curves from multiple runs on shared axes for ablation
    comparison.

    Parameters
    ----------
    run_dirs : list[str]
        Paths to run directories (each must contain ``history.json``).
    labels : list[str] or None
        Display names for each run. If None, uses the directory name.
    save_path : str or None
        If provided, saves the figure instead of displaying it.
    """
    if labels is None:
        labels = [Path(d).name for d in run_dirs]

    # Load all histories
    histories = []
    for d in run_dirs:
        hp = Path(d) / "history.json"
        if not hp.exists():
            print(f"[Warning] history.json not found in {d} — skipping")
            continue
        with open(hp, "r") as f:
            histories.append(json.load(f))

    if not histories:
        print("[Compare Curves] No valid history files found.")
        return

    # Color palette (up to 8 runs)
    colors = ["#2980b9", "#e74c3c", "#27ae60", "#8e44ad",
              "#d35400", "#1abc9c", "#f39c12", "#34495e"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Multi-Model Training Comparison", fontsize=14)

    for idx, (hist, label) in enumerate(zip(histories, labels)):
        c = colors[idx % len(colors)]
        epochs = np.arange(1, len(hist["train_loss"]) + 1)

        # [0,0] Loss
        axes[0, 0].plot(epochs, hist["val_loss"], label=label, color=c, linewidth=1.3)

        # [0,1] RRMSE_t
        axes[0, 1].plot(epochs, hist["rrmse_t"], label=label, color=c, linewidth=1.3)

        # [1,0] RRMSE_s
        axes[1, 0].plot(epochs, hist["rrmse_s"], label=label, color=c, linewidth=1.3)

        # [1,1] ACC
        axes[1, 1].plot(epochs, hist["acc"], label=label, color=c, linewidth=1.3)

    axes[0, 0].set_title("Validation Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("RRMSE Temporal (↓ better)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("RRMSE_t")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("RRMSE Spectral (↓ better)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("RRMSE_s")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("Correlation Coefficient (↑ better)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("ACC")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Compare Curves] Figure saved: {save_path}")
    else:
        plt.show()


# ====================================================================
# 7. Multi-Model SNR Comparison
# ====================================================================

@torch.no_grad()
def compare_snr_curves(
    run_dirs: List[str],
    data_dir: str,
    artifact_type: str = "EOG",
    labels: Optional[List[str]] = None,
    snr_list: list = None,
    batch_size: int = 64,
    save_path: Optional[str] = None,
    device: Optional[torch.device] = None,
):
    """
    Load multiple checkpoints, run ``evaluate_per_snr()`` on each, and overlay
    per-SNR metric curves on the same figure for model comparison.

    Parameters
    ----------
    run_dirs : list[str]
        Paths to run directories. Each must contain ``best.pt`` and a variant
        name derivable from the directory name (or from ``config.json``).
    data_dir : str
        Path to the EEGdenoisenet data directory.
    artifact_type : str
        "EOG" or "EMG".
    labels : list[str] or None
        Display names for each model. If None, uses the directory name.
    snr_list : list[int] or None
        SNR values to evaluate at (default: -7 to +2).
    save_path : str or None
        If provided, saves the figure instead of displaying it.
    """
    if snr_list is None:
        snr_list = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]

    if labels is None:
        labels = [Path(d).name for d in run_dirs]

    device = device or get_device()

    # Collect per-model results
    all_results = []
    seeds_used: List[tuple] = []
    for run_dir, label in zip(run_dirs, labels):
        ckpt_path = Path(run_dir) / "best.pt"
        if not ckpt_path.exists():
            print(f"[Warning] best.pt not found in {run_dir} — skipping")
            continue

        run_cfg = {}
        # Detect variant from directory name or config
        config_path = Path(run_dir) / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                run_cfg = json.load(f)
            variant = run_cfg.get("variant", "tcn")
            hidden_channels = run_cfg.get("hidden_channels", 64)
            kv_lora_rank = run_cfg.get("kv_lora_rank", None)
        else:
            # Infer variant from directory name
            dirname = Path(run_dir).name
            variant = "tcn"
            for v in sorted(VARIANT_REGISTRY.keys(), key=len, reverse=True):
                if v.replace("_", "") in dirname.replace("_", ""):
                    variant = v
                    break
            hidden_channels = 64
            kv_lora_rank = None

        # Prefer architecture metadata stored with the weights.
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_cfg = dict(variant=variant, hidden_channels=hidden_channels)
        if kv_lora_rank is not None:
            model_cfg["kv_lora_rank"] = kv_lora_rank
        model_cfg.update(run_cfg)
        model_cfg.update(ckpt.get("model_config", {}))
        model = build_model_from_config(model_cfg)
        model.load_state_dict(ckpt["model_state"])
        model.eval().to(device)

        print(f"\n[Compare SNR] Evaluating: {label} ({variant})")
        # Each run is scored on its *own* seed's test split: that is the only
        # split it was actually held out from.  Reusing one seed for all of them
        # would score every other run on epochs it trained on.
        seed = resolve_master_seed(ckpt, str(Path(run_dir) / "history.json"),
                                   what=f"{label}'s evaluation")
        seeds_used.append((label, seed))
        results, avg = evaluate_per_snr(
            model, data_dir,
            artifact_type=artifact_type,
            snr_list=snr_list,
            batch_size=batch_size,
            device=device,
            seed=seed,
        )
        all_results.append((label, results))

    if not all_results:
        print("[Compare SNR] No valid checkpoints found.")
        return

    # Runs at different seeds were held out from different epochs, so their
    # curves are each valid on their own but are not a head-to-head on shared
    # data.  Say so on the figure rather than letting it read as one.
    distinct = sorted({s for _, s in seeds_used})
    split_note = ""
    if len(distinct) > 1:
        print(f"[Compare SNR] WARNING: these runs use different master_seeds "
              f"({', '.join(f'{lbl}={s}' for lbl, s in seeds_used)}), so each curve comes "
              f"from a different test split. Each is a valid held-out score for its own "
              f"run, but differences between curves also carry the split change -- for a "
              f"head-to-head, compare runs that share a seed.")
        split_note = f"  [different test splits: seeds {', '.join(map(str, distinct))}]"

    # Plot
    colors = ["#2980b9", "#e74c3c", "#27ae60", "#8e44ad",
              "#d35400", "#1abc9c", "#f39c12", "#34495e"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f"Multi-Model Per-SNR Comparison ({artifact_type}){split_note}", fontsize=13)

    for idx, (label, results) in enumerate(all_results):
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        snrs = sorted(results.keys())

        rrmse_t = [results[s]["rrmse_t"] for s in snrs]
        rrmse_s = [results[s]["rrmse_s"] for s in snrs]
        accs = [results[s]["acc"] for s in snrs]

        axes[0].plot(snrs, rrmse_t, marker=m, color=c, linewidth=1.4, label=label)
        axes[1].plot(snrs, rrmse_s, marker=m, color=c, linewidth=1.4, label=label)
        axes[2].plot(snrs, accs, marker=m, color=c, linewidth=1.4, label=label)

    axes[0].set_title("RRMSE Temporal (↓ better)")
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("RRMSE_t")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("RRMSE Spectral (↓ better)")
    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("RRMSE_s")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Correlation Coefficient (↑ better)")
    axes[2].set_xlabel("SNR (dB)")
    axes[2].set_ylabel("ACC")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Compare SNR] Figure saved: {save_path}")
    else:
        plt.show()


# ====================================================================
# CLI
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation: metrics, curves, waveforms, attention heatmaps",
    )

    # --- Required for most modes ---
    parser.add_argument("--ckpt", default=None,
                        help="Path to model checkpoint (e.g. runs/exp1/best.pt)")
    parser.add_argument("--data_dir", default=None,
                        help="Path to EEGdenoisenet data directory")
    parser.add_argument("--variant", default=None, choices=list(VARIANT_REGISTRY.keys()),
                        help="Model variant (must match the checkpoint)")
    parser.add_argument("--artifact", default="EOG", choices=["EOG", "EMG"])
    parser.add_argument("--master_seed", type=int, default=None,
                        help="The run's master_seed, which defines its train/val/test split. "
                             "Default: read from the checkpoint, else 42 with a warning. "
                             "A wrong value evaluates on epochs the model was trained on.")

    # --- Mode selection ---
    parser.add_argument("--mode", default="all",
                        choices=["all", "snr", "curves", "waveform", "attention",
                                 "snr_plot", "compare_curves", "compare_snr"],
                        help="Which tool(s) to run (default: all)")

    # --- Training curve plotter ---
    parser.add_argument("--history", default=None,
                        help="Path to history.json (auto-detected from --ckpt if omitted)")

    # --- Waveform visualizer ---
    parser.add_argument("--vis_snr", type=float, default=-3.0,
                        help="SNR for waveform / attention visualisation")
    parser.add_argument("--n_samples", type=int, default=4,
                        help="Number of samples for waveform visualisation")

    # --- Attention heatmap ---
    parser.add_argument("--attn_sample", type=int, default=0,
                        help="Test sample index for attention heatmap")

    # --- Multi-model comparison ---
    parser.add_argument("--compare_runs", nargs="+", default=None,
                        help="Run directories for multi-model comparison. "
                             "Format: path[:label] (e.g. runs/tcn_eog:TCN)")

    # --- Output ---
    parser.add_argument("--save_dir", default=None,
                        help="Directory to save all figures (default: display)")

    parser.add_argument("--config", default=None,
                        help="YAML model config for old checkpoints or evaluation overrides")
    parser.add_argument("--turboquant", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--turboquant_bits", type=int, choices=range(2, 9), default=None)
    parser.add_argument("--turboquant_key_mode", choices=["prod", "mse"], default=None)
    parser.add_argument("--turboquant_seed", type=int, default=None)
    args = parser.parse_args()
    if args.mode == "compare_snr" and any(value is not None for value in
            (args.config, args.turboquant, args.turboquant_bits,
             args.turboquant_key_mode, args.turboquant_seed)):
        parser.error("compare_snr uses each checkpoint's saved model config; overrides apply to single-model modes")

    # ---- Resolve paths ----
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect history.json from checkpoint directory
    history_path = args.history
    if history_path is None and args.ckpt is not None:
        candidate = Path(args.ckpt).parent / "history.json"
        if candidate.exists():
            history_path = str(candidate)

    # ---- Load model (needed for single-model modes) ----
    model = None
    no_ckpt_modes = {"curves", "compare_curves", "compare_snr"}
    if args.mode not in no_ckpt_modes:
        if args.ckpt is None:
            parser.error(f"--ckpt is required for mode '{args.mode}'")
        if args.data_dir is None:
            parser.error(f"--data_dir is required for mode '{args.mode}'")

        ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model_cfg = {"variant": "tcn", "hidden_channels": 64}
        model_cfg.update(ckpt.get("model_config", {}))
        if args.config:
            from configs import load_config
            model_cfg.update(load_config(args.config))
        for key in ("variant", "turboquant", "turboquant_bits",
                    "turboquant_key_mode", "turboquant_seed"):
            value = getattr(args, key)
            if value is not None:
                model_cfg[key] = value
        model = build_model_from_config(model_cfg)
        model.load_state_dict(ckpt["model_state"])
        print(f"[Eval] Loaded checkpoint: {args.ckpt}  (epoch={ckpt['epoch']})")
        # The split this model's held-out data actually lives in.
        seed = resolve_master_seed(ckpt, history_path, args.master_seed)

    run_all = args.mode == "all"

    # ---- 1. Per-SNR evaluation ----
    if run_all or args.mode == "snr":
        evaluate_per_snr(
            model, args.data_dir,
            artifact_type=args.artifact,
            seed=seed,
        )

    # ---- 2. Training curves ----
    if run_all or args.mode == "curves":
        if history_path is None:
            if args.mode == "curves":
                parser.error("--history is required when --mode=curves and no --ckpt given")
            else:
                print("[Curves] No history.json found — skipping training curve plot.")
        else:
            plot_training_curves(
                history_path,
                save_path=str(save_dir / "training_curves.png") if save_dir else None,
            )

    # ---- 3. Waveform visualisation ----
    if run_all or args.mode == "waveform":
        visualize_samples(
            model, args.data_dir,
            artifact_type=args.artifact,
            snr=args.vis_snr,
            n_samples=args.n_samples,
            save_path=str(save_dir / "waveforms.png") if save_dir else None,
            seed=seed,
        )

    # ---- 4. Attention heatmaps ----
    if run_all or args.mode == "attention":
        plot_attention_heatmaps(
            model, args.data_dir,
            artifact_type=args.artifact,
            snr=args.vis_snr,
            sample_idx=args.attn_sample,
            save_path=str(save_dir / "attention_heatmaps.png") if save_dir else None,
            seed=seed,
        )

    # ---- 5. Per-SNR metric line chart ----
    if run_all or args.mode == "snr_plot":
        if model is None:
            parser.error("--ckpt is required for snr_plot mode")
        results, _ = evaluate_per_snr(
            model, args.data_dir,
            artifact_type=args.artifact,
            seed=seed,
        )
        variant_label = args.variant.upper().replace("_", " ")
        plot_snr_curves(
            results,
            title=f"Per-SNR Metrics — {variant_label} ({args.artifact})",
            save_path=str(save_dir / "snr_curves.png") if save_dir else None,
        )

    # ---- Parse compare_runs into dirs and labels ----
    compare_dirs, compare_labels = None, None
    if args.compare_runs:
        compare_dirs = []
        compare_labels = []
        for entry in args.compare_runs:
            if ":" in entry:
                path, label = entry.rsplit(":", 1)
            else:
                path, label = entry, Path(entry).name
            compare_dirs.append(path)
            compare_labels.append(label)

    # ---- 6. Multi-model training comparison ----
    if args.mode == "compare_curves":
        if not compare_dirs:
            parser.error("--compare_runs is required for compare_curves mode")
        compare_training_curves(
            compare_dirs,
            labels=compare_labels,
            save_path=str(save_dir / "compare_training.png") if save_dir else None,
        )

    # ---- 7. Multi-model SNR comparison ----
    if args.mode == "compare_snr":
        if not compare_dirs:
            parser.error("--compare_runs is required for compare_snr mode")
        if args.data_dir is None:
            parser.error("--data_dir is required for compare_snr mode")
        compare_snr_curves(
            compare_dirs,
            data_dir=args.data_dir,
            artifact_type=args.artifact,
            labels=compare_labels,
            save_path=str(save_dir / "compare_snr.png") if save_dir else None,
        )


if __name__ == "__main__":
    main()
