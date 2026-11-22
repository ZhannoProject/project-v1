"""
Unified evaluation script for EEG causal denoising models.

Eight tools in one CLI:
  1. Per-SNR evaluator              — quantitative metrics across noise levels
  2. Training curve plotter         — loss and metric curves from history.json
  3. Waveform visualizer            — time-domain + PSD comparison plots
  4. Attention heatmap plotter      — per-head, per-layer attention weight maps
  5. Per-SNR metric line chart      — RRMSE_t, RRMSE_f, CC plotted over SNR
  6. Multi-model training comparison— overlay training curves from multiple runs
  7. Multi-model SNR comparison     — overlay per-SNR metrics from multiple models
  8. Artifact-switch response       — error vs. distance to a non-stationary
                                      artifact switch, one curve per model

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

    # Artifact-switch response (non-stationary probe, models trained on BOTH):
    python eval.py --mode compare_switch --data_dir ./data/EEGdenoisenet/data/ \\
                   --compare_runs runs/tcn_both:TCN-only runs/tcn_mla_both:TCN+MLA \\
                   --save_dir ./figures/
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

from data.Dataloader import (EEGDenoiseDataset, ArtifactSwitchDataset,
                            build_switch_probes, SWITCH_PROBE_SEEDS)
from torch.utils.data import DataLoader
from metrics import compute_all_metrics
from models.variants import VARIANT_REGISTRY
from utils.utils import get_device
from data.protocol import evaluation_data
from experiment.methodology import aligned, target_interval, aligned_output


# ====================================================================
# 1. Per-SNR Evaluator
# ====================================================================

@torch.no_grad()
def evaluate_per_snr(
    model,
    data_dir: str,
    artifact_type: str = None,
    snr_list: list = None,
    batch_size: int = 64,
    seed: int = None,
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
    data_kwargs = evaluation_data(model, seed, artifact_type)
    artifact_type = data_kwargs["artifact_type"]

    print(f"\n{'='*60}")
    print(f"Evaluation graded by SNR ({artifact_type})")
    print(f"{'='*60}")
    print(f"{'SNR':>6} | {'RRMSE_t':>8} | {'RRMSE_f':>8} | {'CC':>6}")
    print(f"{'-'*6:>6}-+-{'-'*8:>8}-+-{'-'*8:>8}-+-{'-'*6:>6}")

    for snr in snr_list:
        ds = EEGDenoiseDataset(
            data_dir=data_dir,
            split="test",
            fixed_snr_db=float(snr),
            **data_kwargs,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        agg = {"rrmse_t": 0.0, "rrmse_s": 0.0, "acc": 0.0}
        n_seen = 0

        for batch in loader:
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            pred = model(noisy)
            pred, clean = aligned(pred, clean, model)
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
    and the three evaluation metrics (RRMSE_t, RRMSE_f, CC) over epochs.

    Layout: 2 x 2 grid
      [0,0] train_loss + val_loss     [0,1] RRMSE_temporal
      [1,0] RRMSE_spectral            [1,1] CC

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
    ax.set_ylabel("RRMSE_f")
    ax.set_title("RRMSE Spectral (lower = better)")
    ax.grid(True, alpha=0.3)

    # --- [1,1] CC ---
    ax = axes[1, 1]
    ax.plot(epochs, history["acc"], color="#27ae60", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CC")
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
    artifact_type: str = None,
    snr: float = -3.0,
    n_samples: int = 4,
    save_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    seed: int = None,
):
    """
    Plots noisy, clean (ground-truth), and denoised signals side by side
    in both the time domain and frequency domain (PSD via Welch's method).
    """
    device = device or get_device()
    model.eval().to(device)
    data_kwargs = evaluation_data(model, seed, artifact_type)
    artifact_type = data_kwargs["artifact_type"]

    ds = EEGDenoiseDataset(
        data_dir=data_dir,
        split="test",
        fixed_snr_db=float(snr),
        **data_kwargs,
    )
    fs = ds.fs

    fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 6))
    fig.suptitle(
        f"{artifact_type} Denoising Performance Comparison (SNR = {snr} dB)",
        fontsize=13,
    )

    for i in range(n_samples):
        sample = ds[i * 100]
        noisy = sample["noisy"].unsqueeze(0).to(device)
        clean = sample["clean"].squeeze().cpu().numpy()
        sigma = sample["sigma"].item()

        pred = model(noisy).squeeze().cpu().numpy()
        noisy_np = sample["noisy"].squeeze().cpu().numpy()
        start, stop = target_interval(len(clean), getattr(model, "run_config", {}))
        delay = getattr(model, "run_config", {}).get("output_delay", 0)
        pred = pred[start + delay:stop + delay]
        clean, noisy_np = clean[start:stop], noisy_np[start:stop]

        # Restore original amplitude
        clean_real = clean * sigma
        noisy_real = noisy_np * sigma
        pred_real = pred * sigma

        t = (np.arange(len(clean)) + start) / fs

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
    artifact_type: str = None,
    snr: float = -3.0,
    sample_idx: int = 0,
    save_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    seed: int = None,
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
    """
    attn_modules = _get_attn_modules(model)
    if not attn_modules:
        print("[Attention] No attention layers found — skipping heatmap.")
        return

    device = device or get_device()
    model.eval().to(device)

    data_kwargs = evaluation_data(model, seed, artifact_type)

    # Enable weight capture on all attention layers
    for _, attn in attn_modules:
        attn.capture_weights(True)

    # Run a single forward pass to populate the attention weights
    ds = EEGDenoiseDataset(
        data_dir=data_dir,
        split="test",
        fixed_snr_db=float(snr),
        **data_kwargs,
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
    Plot RRMSE_t, RRMSE_f, and CC as line charts with SNR (dB) on the x-axis.

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
    ax.set_ylabel("RRMSE_f")
    ax.set_title("RRMSE Spectral (↓ better)")
    ax.grid(True, alpha=0.3)

    # CC
    ax = axes[2]
    ax.plot(snrs, accs, marker="^", color="#27ae60", linewidth=1.5)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("CC")
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

        # [1,0] RRMSE_f
        axes[1, 0].plot(epochs, hist["rrmse_s"], label=label, color=c, linewidth=1.3)

        # [1,1] CC
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
    axes[1, 0].set_ylabel("RRMSE_f")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("Correlation Coefficient (↑ better)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("CC")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Compare Curves] Figure saved: {save_path}")
    else:
        plt.show()


# ====================================================================
# Shared: rebuild the model a run directory holds
# ====================================================================

def load_run_model(run_dir: str, device: Optional[torch.device] = None):
    """
    Rebuild the trained model stored in ``run_dir/best.pt``.

    The architecture comes from the checkpoint's own ``model_config`` when it
    has one; otherwise from ``config.json``, and failing that from the
    directory name.  Returns ``(model, variant)``, or ``(None, None)`` if the
    directory holds no checkpoint — the multi-model tools skip those rather
    than aborting the whole comparison.
    """
    ckpt_path = Path(run_dir) / "best.pt"
    if not ckpt_path.exists():
        print(f"[Warning] best.pt not found in {run_dir} — skipping")
        return None, None

    from experiment.checkpoints import load_model
    model, arch, _ = load_model(ckpt_path, device=device or get_device())
    return model, arch["variant"]


# ====================================================================
# 7. Multi-Model SNR Comparison
# ====================================================================

@torch.no_grad()
def compare_snr_curves(
    run_dirs: List[str],
    data_dir: str,
    artifact_type: str = None,
    labels: Optional[List[str]] = None,
    snr_list: list = None,
    batch_size: int = 64,
    master_seed: int = None,
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
    for run_dir, label in zip(run_dirs, labels):
        model, variant = load_run_model(run_dir, device=device)
        if model is None:
            continue

        print(f"\n[Compare SNR] Evaluating: {label} ({variant})")
        results, avg = evaluate_per_snr(
            model, data_dir,
            artifact_type=artifact_type,
            snr_list=snr_list,
            batch_size=batch_size,
            seed=master_seed,
            device=device,
        )
        all_results.append((label, results))

    if not all_results:
        print("[Compare SNR] No valid checkpoints found.")
        return

    # Plot
    colors = ["#2980b9", "#e74c3c", "#27ae60", "#8e44ad",
              "#d35400", "#1abc9c", "#f39c12", "#34495e"]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(f"Multi-Model Per-SNR Comparison ({artifact_type})", fontsize=13)

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
    axes[1].set_ylabel("RRMSE_f")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Correlation Coefficient (↑ better)")
    axes[2].set_xlabel("SNR (dB)")
    axes[2].set_ylabel("CC")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Compare SNR] Figure saved: {save_path}")
    else:
        plt.show()


# ====================================================================
# 8. Artifact-Switch Response  (non-stationary probe)
# ====================================================================

#: Directions the response is broken down by, and how they are labelled.
SWITCH_DIRECTIONS = {
    "all":        "both directions",
    "eog_to_emg": "EOG → EMG",
    "emg_to_eog": "EMG → EOG",
}

#: The same, short enough for a table column.
SWITCH_DIRECTION_SHORT = {"all": "both", "eog_to_emg": "EOG→EMG", "emg_to_eog": "EMG→EOG"}


def _switch_summary(offsets: np.ndarray, err: np.ndarray, fs: int,
                    recovery_tol: float, floor_frac: float = 0.25,
                    delta: Optional[np.ndarray] = None) -> dict:
    """
    Reduce one response curve to the numbers a table can carry.

    The raw curve mixes three things, and the summary separates them.

    1. **The splice.** Most of the spike at tau = 0 is simply that the signal
       was cut and something else glued on; it happens whether or not the
       artifact *type* changed.  The matched control splices a different epoch
       of the same type, so ``delta = switch − control`` has it removed.
    2. **The regime shift.** EOG and EMG are not equally hard, so the error
       settles at a different level once the new type takes over, and delta
       keeps that as a constant offset far from the switch.  That offset,
       ``delta_floor``, is reported on its own — it is a property of the
       artifacts, not of adaptation, and scoring the transient against zero
       would let an EMG→EOG switch book a negative adaptation cost purely
       because EOG is the easier artifact.
    3. **The transient** — the re-adaptation cost, which is what the
       experiment is about.  It is ``delta − delta_floor``, and the
       ``transient_*`` fields are all measured on it.

    ``pre_leak`` is the validity check, and doubles as a measurement: for a
    causal model the two probes are the same signal before the switch, so it
    is exactly 0.  For a non-causal model it is not, and ``pre_leak_far`` —
    the same thing restricted to the far half of the pre-switch window — says
    whether the look-ahead has a horizon (finite receptive field: leaks near
    the switch, zero far from it) or none (bidirectional attention over the
    whole epoch: leaks everywhere).  Both only read as look-ahead under a hard
    switch: with ``ramp_samples > 0`` the crossfade itself reaches back before
    the switch index, so a causal model registers a leak too.

    Without a control (``delta=None``) only the descriptive fields and a
    floor-referenced fallback are available.

    Fields
    ------
    pre, post_floor, regime_shift, peak, peak_ms, overshoot
        Read off the raw curve; ``overshoot`` is peak ÷ post_floor.
    recovery_ms, floor_transient_area
        Floor-referenced fallback, used when no control was run.
    delta_pre, pre_leak, pre_leak_far, delta_floor
        The signed and RMS pre-switch leakage, the same restricted to the far
        half of the pre-switch window, and the isolated regime offset.
    transient_peak, transient_peak_ms, transient_recovery_ms, transient_area
        The re-adaptation cost.  ``transient_recovery_ms`` is the time for the
        excess to decay to ``recovery_tol`` of its own peak, which needs no
        reference level at all; ``transient_area`` is in relative error × ms.
    """
    pre_mask = offsets < 0
    post_mask = offsets >= 0

    pre = float(np.sqrt(np.mean(err[pre_mask] ** 2)))
    post_err = err[post_mask]
    post_off = offsets[post_mask]

    n_floor = max(int(round(floor_frac * len(post_err))), 1)
    post_floor = float(np.sqrt(np.mean(post_err[-n_floor:] ** 2)))

    peak_i = int(np.argmax(post_err))
    threshold = post_floor * (1.0 + recovery_tol)
    recovered = np.where(post_err[peak_i:] <= threshold)[0]

    out = {
        "pre": pre,
        "post_floor": post_floor,
        "regime_shift": post_floor / pre if pre > 0 else float("nan"),
        "peak": float(post_err[peak_i]),
        "peak_ms": float(post_off[peak_i] / fs * 1000.0),
        "overshoot": float(post_err[peak_i]) / post_floor if post_floor > 0 else float("nan"),
        "recovery_ms": (float(post_off[peak_i + recovered[0]] / fs * 1000.0)
                        if len(recovered) else None),
        "recovery_tol": recovery_tol,
        "floor_frac": floor_frac,
        "floor_transient_area": float(np.sum(post_err - post_floor) / fs * 1000.0),
    }

    if delta is not None:
        post_delta = delta[post_mask]
        # The offset delta settles to is the regime shift, not adaptation.
        delta_floor = float(np.mean(post_delta[-n_floor:]))
        transient = post_delta - delta_floor

        t_peak_i = int(np.argmax(transient))
        t_peak = float(transient[t_peak_i])
        # Decay to recovery_tol of its own peak -- no reference level needed.
        t_recovered = np.where(transient[t_peak_i:] <= recovery_tol * t_peak)[0]
        pre_delta = delta[pre_mask]
        far = pre_delta[: max(len(pre_delta) // 2, 1)]     # the oldest half
        out.update({
            "delta_pre": float(np.mean(pre_delta)),
            "pre_leak": float(np.sqrt(np.mean(pre_delta ** 2))),
            "pre_leak_far": float(np.sqrt(np.mean(far ** 2))),
            "delta_floor": delta_floor,
            "transient_peak": t_peak,
            "transient_peak_ms": float(post_off[t_peak_i] / fs * 1000.0),
            "transient_recovery_ms": (
                float(post_off[t_peak_i + t_recovered[0]] / fs * 1000.0)
                if len(t_recovered) else None),
            "transient_area": float(np.sum(transient) / fs * 1000.0),
        })

    return out


@torch.no_grad()
def _response_curves(model, dataset, half_width: int, batch_size: int,
                     device: torch.device) -> Dict[str, np.ndarray]:
    """
    Pointwise relative error at every offset from each epoch's switch index.

        err(tau) = sqrt( mean_n [ (pred_n - clean_n)(k_n + tau) / RMS(clean_n) ]^2 )

    Dividing by each epoch's own clean RMS before pooling makes this a
    time-resolved RRMSE_t: average err(tau)^2 over all tau and the ordinary
    RRMSE_t comes back, so the curve and the headline table are the same
    quantity at two resolutions.

    Returns one curve per entry of SWITCH_DIRECTIONS.
    """
    off = np.arange(-half_width, half_width + 1)
    off_t = torch.as_tensor(off, dtype=torch.long, device=device)

    sq = {d: np.zeros(len(off), dtype=np.float64) for d in SWITCH_DIRECTIONS}
    cnt = {d: 0 for d in SWITCH_DIRECTIONS}

    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device).squeeze(1)              # (B, T)
        pred = aligned_output(model(noisy), model).squeeze(1)     # (B, T), target aligned

        rms_clean = torch.sqrt(torch.mean(clean ** 2, dim=-1) + 1e-12)
        resid = (pred - clean) / rms_clean.unsqueeze(1)

        k = batch["switch_idx"].to(device)                        # (B,)
        idx = k.unsqueeze(1) + off_t.unsqueeze(0)                 # (B, L)
        vals = resid.gather(1, idx) ** 2
        if not torch.isfinite(vals).all():
            raise ValueError("Switch probe reaches excluded warm-up/tail; narrow the probe interval")

        eog_first = batch["eog_first"].to(device)
        for name, mask in (("all", None),
                           ("eog_to_emg", eog_first),
                           ("emg_to_eog", ~eog_first)):
            sel = vals if mask is None else vals[mask]
            if sel.shape[0] == 0:
                continue
            # .cpu() first: MPS has no float64.
            sq[name] += sel.sum(0).cpu().double().numpy()
            cnt[name] += sel.shape[0]

    return {d: np.sqrt(sq[d] / cnt[d]) for d in SWITCH_DIRECTIONS if cnt[d]}


@torch.no_grad()
def evaluate_switch_response(
    model,
    data_dir: str,
    probe_seeds=SWITCH_PROBE_SEEDS,
    window: Optional[int] = None,
    batch_size: int = 64,
    master_seed: int = None,
    fs: int = None,
    recovery_tol: float = 0.10,
    control: bool = True,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    **probe_kwargs,
) -> dict:
    """
    Error as a function of distance to an artifact switch.

    Runs the model over the ``ArtifactSwitchDataset`` series, aligns every
    epoch on its own switch point, and reports the pointwise relative error
    (see ``_response_curves``).  ``tau = 0`` is the first sample carrying the
    second artifact; negative offsets are the settled regime the model has
    already adapted to, which is what makes the post-switch rise readable as a
    re-adaptation cost rather than as a property of the model.

    ``control=True`` (the default) runs a second probe with every draw
    identical except that the epoch spliced in at the switch point is a
    different epoch of the *same* artifact type, and reports the difference
    ``delta = switch − control``.  That removes the cost of the splice itself
    and the within-epoch warm-up trend both probes share, leaving the change
    of artifact statistics as the only difference.

    The curve is measured once per entry in ``probe_seeds``; the difference is
    taken **within** each seed and then averaged, so it is a paired
    comparison.  A bump has to survive every draw of the switch positions to
    appear in the figure.

    Extra keyword arguments go to ``ArtifactSwitchDataset`` (``ramp_samples``,
    ``fixed_snr_db``, ``switch_range``, ``n_epochs``, ...).

    Returns
    -------
    dict with ``offsets`` (samples), ``t_ms``, and for each direction in
    ``SWITCH_DIRECTIONS`` a block of ``err`` / ``err_std`` / ``control`` /
    ``delta`` / ``per_seed`` / ``summary``.
    """
    device = device or get_device()
    model.eval().to(device)

    curves = {d: [] for d in SWITCH_DIRECTIONS}
    ctrl_curves = {d: [] for d in SWITCH_DIRECTIONS}
    offsets = None
    n_epochs_total = 0

    data_kwargs = evaluation_data(model, master_seed, "BOTH")
    fs_actual = data_kwargs["target_fs"] or 256.
    if fs is not None and fs != fs_actual:
        raise ValueError(f"fs={fs} differs from the model data rate {fs_actual}")
    fs = fs_actual
    for key in ("master_seed", "split_ratios", "snr_range_db", "target_fs",
                "normalization", "normalization_scale", "resampling"):
        if key in probe_kwargs and probe_kwargs[key] != data_kwargs[key]:
            raise ValueError(f"Switch probe {key} differs from the checkpoint data protocol")
        probe_kwargs[key] = data_kwargs[key]
    for probe_seed, ds in build_switch_probes(data_dir, probe_seeds=probe_seeds,
                                              **probe_kwargs):
        lo, hi = ds.switch_bounds
        T = ds.seq_len
        # Widest window every epoch can supply on both sides of its switch.
        w_max = min(lo, T - 1 - hi)
        w = w_max if window is None else min(int(window), w_max)
        if window is not None and w < window and verbose:
            print(f"[Switch] window clipped to {w} samples "
                  f"({w / fs * 1000:.0f} ms) by switch_range")
        if offsets is None:
            offsets = np.arange(-w, w + 1)

        for d, c in _response_curves(model, ds, w, batch_size, device).items():
            curves[d].append(c)
        n_epochs_total += len(ds)

        if control:
            ds_c = ArtifactSwitchDataset(data_dir, probe_seed=probe_seed,
                                         control=True, **probe_kwargs)
            for d, c in _response_curves(model, ds_c, w, batch_size, device).items():
                ctrl_curves[d].append(c)

    if offsets is None:
        raise ValueError("probe_seeds was empty — nothing to evaluate")

    result = {
        "offsets": offsets,
        "t_ms": offsets / fs * 1000.0,
        "fs": fs,
        "probe_seeds": list(probe_seeds),
        "n_epochs": n_epochs_total,
        "control": control,
        "probe": dict(probe_kwargs),
    }
    for d in SWITCH_DIRECTIONS:
        if not curves[d]:
            continue
        stack = np.stack(curves[d])
        block = {
            "err": stack.mean(0),
            "err_std": stack.std(0),
            "per_seed": {int(s): c for s, c in zip(probe_seeds, stack)},
        }
        delta = None
        if control and ctrl_curves[d]:
            ctrl = np.stack(ctrl_curves[d])
            # Paired within seed, then averaged.
            per_seed_delta = stack - ctrl
            delta = per_seed_delta.mean(0)
            block.update({
                "control": ctrl.mean(0),
                "control_std": ctrl.std(0),
                "delta": delta,
                "delta_std": per_seed_delta.std(0),
                "per_seed_delta": {int(s): c for s, c in zip(probe_seeds, per_seed_delta)},
            })
        block["summary"] = _switch_summary(offsets, block["err"], fs,
                                           recovery_tol, delta=delta)
        result[d] = block

    if verbose:
        print_switch_summary({"model": result})
    return result


def print_switch_summary(results_by_label: Dict[str, dict]):
    """Table of the switch-response scalars, one row per model and direction."""
    has_delta = any("delta_pre" in r[d]["summary"]
                    for r in results_by_label.values()
                    for d in SWITCH_DIRECTIONS if d in r)
    width = 120
    print(f"\n{'=' * width}")
    print("Artifact-switch response (pointwise relative error, aligned on the switch)")
    print(f"{'=' * width}")
    tail = (f"{'leak':>5} | {'far':>5} | {'Δfloor':>6} | {'Tpeak':>6} | "
            f"{'Trec ms':>7} | {'Tarea':>6}"
            if has_delta else f"{'rec ms':>6} | {'area':>7}")
    print(f"{'Model':>16} | {'Dir':>7} | {'pre':>5} | {'floor':>5} | "
          f"{'peak':>5} | {'oversh':>6} | {tail}")
    print("-" * width)
    for label, res in results_by_label.items():
        for d in SWITCH_DIRECTIONS:
            if d not in res:
                continue
            m = res[d]["summary"]
            if has_delta and "delta_pre" in m:
                rec = ("—" if m["transient_recovery_ms"] is None
                       else f"{m['transient_recovery_ms']:.0f}")
                tail = (f"{m['pre_leak']:>5.3f} | {m['pre_leak_far']:>5.3f} | "
                        f"{m['delta_floor']:>6.3f} | {m['transient_peak']:>6.3f} | "
                        f"{rec:>7} | {m['transient_area']:>6.2f}")
            else:
                rec = "—" if m["recovery_ms"] is None else f"{m['recovery_ms']:.0f}"
                tail = f"{rec:>6} | {m['floor_transient_area']:>7.2f}"
            print(f"{label:>16} | {SWITCH_DIRECTION_SHORT[d]:>7} | "
                  f"{m['pre']:>5.3f} | {m['post_floor']:>5.3f} | "
                  f"{m['peak']:>5.3f} | {m['overshoot']:>6.2f} | {tail}")
    print("-" * width)
    print("Raw curve : pre / floor are the settled errors either side of the switch;")
    print("            oversh = peak ÷ floor.  Most of that peak is the splice itself.")
    if has_delta:
        print("Δ         : switch probe − same-type-splice control, paired per probe seed.")
        print("            leak   = RMS of Δ before the switch; 0.000 for a causal model (the")
        print("                     two probes are one signal there).  far = the same over the")
        print("                     oldest half: leak>0 with far=0 is a finite look-ahead")
        print("                     horizon, both >0 is look-ahead over the whole epoch.")
        print("            Δfloor = the regime shift alone: how much harder the new artifact")
        print("                     is at matched SNR, with the splice controlled for.")
        print("T…        : the transient, Δ minus Δfloor — the re-adaptation cost.")
        print("            Tpeak, Trec (decay to 10% of Tpeak), Tarea (rel. error × ms).")
    else:
        print("rec / area: time back to within 10% of floor, and the area above it.")


def plot_switch_response(
    results_by_label: Dict[str, dict],
    title: str = "Error vs. time from artifact switch",
    save_path: Optional[str] = None,
    show_band: bool = True,
):
    """
    The figure: pointwise relative error against time from the switch point,
    every model on the same axes, one column per direction.

    Top row is the raw curve, with the same-type-splice control dotted
    underneath it: the gap between them is what the *type* change costs, as
    opposed to the splice.  Bottom row is that gap, and it is the row to read
    a "who recovers faster" claim off — before the switch it is exactly zero
    for a causal model, and the level it settles back to is the regime shift.

    The shaded band is ± 1 SD across the probe seeds, so a difference between
    two models can be weighed against the probe's own variability instead of
    taken on trust.
    """
    colors = ["#2980b9", "#e74c3c", "#27ae60", "#8e44ad",
              "#d35400", "#1abc9c", "#f39c12", "#34495e"]

    panels = [d for d in SWITCH_DIRECTIONS
              if any(d in r for r in results_by_label.values())]
    has_delta = any("delta" in r[d] for r in results_by_label.values()
                    for d in panels if d in r)
    n_rows = 2 if has_delta else 1

    fig, axes = plt.subplots(n_rows, len(panels),
                             figsize=(5.2 * len(panels), 4.0 * n_rows),
                             squeeze=False, sharex=True)
    fig.suptitle(title, fontsize=13)

    def mark_switch(ax, label=True):
        ax.axvline(0.0, color="#555", linestyle="--", linewidth=1.0)
        if label:
            ax.annotate("switch", xy=(0, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(4, -12), textcoords="offset points",
                        fontsize=8, color="#555")
        ax.grid(True, alpha=0.3)

    for col, d in enumerate(panels):
        ax = axes[0][col]
        for i, (label, res) in enumerate(results_by_label.items()):
            if d not in res:
                continue
            c = colors[i % len(colors)]
            t, block = res["t_ms"], res[d]
            ax.plot(t, block["err"], color=c, linewidth=1.4, label=label)
            if show_band:
                sd = block["err_std"]
                ax.fill_between(t, block["err"] - sd, block["err"] + sd,
                                color=c, alpha=0.15, linewidth=0)
            if "control" in block:
                ax.plot(t, block["control"], color=c, linewidth=1.0,
                        linestyle=":", alpha=0.75,
                        label=f"{label} (same-type control)" if i == 0 else None)
        mark_switch(ax)
        ax.set_title(SWITCH_DIRECTIONS[d], fontsize=11)
        if not has_delta:
            ax.set_xlabel("Time from switch (ms)")

    axes[0][0].set_ylabel("Relative error (↓ better)")
    axes[0][0].legend(fontsize=8)

    if has_delta:
        for col, d in enumerate(panels):
            ax = axes[1][col]
            for i, (label, res) in enumerate(results_by_label.items()):
                if d not in res or "delta" not in res[d]:
                    continue
                block = res[d]
                c = colors[i % len(colors)]
                t, delta = res["t_ms"], block["delta"]
                ax.plot(t, delta, color=c, linewidth=1.4, label=label)
                if show_band:
                    sd = block["delta_std"]
                    ax.fill_between(t, delta - sd, delta + sd,
                                    color=c, alpha=0.15, linewidth=0)
            ax.axhline(0.0, color="#999", linewidth=0.8)
            mark_switch(ax, label=False)
            ax.set_xlabel("Time from switch (ms)")
        axes[1][0].set_ylabel("Excess error vs. same-type splice")
        axes[1][0].legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Switch] Figure saved: {save_path}")
    else:
        plt.show()


def _switch_result_to_json(res: dict) -> dict:
    """Make one evaluate_switch_response() result JSON-serialisable."""
    out = {
        "offsets": [int(v) for v in res["offsets"]],
        "t_ms": [float(v) for v in res["t_ms"]],
        "fs": res["fs"],
        "probe_seeds": res["probe_seeds"],
        "n_epochs": res["n_epochs"],
        "control": res["control"],
        "probe": {k: (list(v) if isinstance(v, tuple) else v)
                  for k, v in res["probe"].items()},
    }
    for d in SWITCH_DIRECTIONS:
        if d not in res:
            continue
        block = res[d]
        entry = {"summary": block["summary"]}
        for key in ("err", "err_std", "control", "control_std", "delta", "delta_std"):
            if key in block:
                entry[key] = [float(v) for v in block[key]]
        # Per-seed curves, so a difference between two models can be tested
        # seed by seed (paired) rather than eyeballed off the band.
        for key in ("per_seed", "per_seed_delta"):
            if key in block:
                entry[key] = {str(k): [float(v) for v in c]
                              for k, c in block[key].items()}
        out[d] = entry
    return out


@torch.no_grad()
def compare_switch_response(
    run_dirs: List[str],
    data_dir: str,
    labels: Optional[List[str]] = None,
    probe_seeds=SWITCH_PROBE_SEEDS,
    window: Optional[int] = None,
    batch_size: int = 64,
    master_seed: int = None,
    control: bool = True,
    save_path: Optional[str] = None,
    json_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    **probe_kwargs,
) -> Dict[str, dict]:
    """
    Run the switch probe on several checkpoints and overlay their response
    curves — the TCN-only vs. attention comparison the experiment exists for.

    Each checkpoint uses its own held-out split and recorded preprocessing.
    A supplied ``master_seed`` must match every checkpoint being compared.
    """
    if labels is None:
        labels = [Path(d).name for d in run_dirs]

    device = device or get_device()
    results: Dict[str, dict] = {}

    for run_dir, label in zip(run_dirs, labels):
        model, variant = load_run_model(run_dir, device=device)
        if model is None:
            continue
        print(f"\n[Switch] Evaluating: {label} ({variant})")
        results[label] = evaluate_switch_response(
            model, data_dir,
            probe_seeds=probe_seeds,
            window=window,
            batch_size=batch_size,
            master_seed=master_seed,
            control=control,
            device=device,
            verbose=False,
            **probe_kwargs,
        )

    if not results:
        print("[Switch] No valid checkpoints found.")
        return {}

    print_switch_summary(results)
    plot_switch_response(results, save_path=save_path)

    if json_path:
        with open(json_path, "w") as f:
            json.dump({k: _switch_result_to_json(v) for k, v in results.items()},
                      f, indent=2)
        print(f"[Switch] Numbers saved: {json_path}")

    return results


# ====================================================================
# CLI
# ====================================================================

#: Every --mode this script accepts.  Named here rather than inline in the
#: parser so other front ends (the GUI's Evaluate tab) can offer exactly the
#: modes that exist, and a test can prove the two lists have not drifted.
MODES = (
    "all", "snr", "snr_plot", "curves", "waveform", "attention",
    "compare_curves", "compare_snr", "switch", "compare_switch",
)

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
    parser.add_argument("--artifact", default=None, choices=["EOG", "EMG", "BOTH"],
                        help="Artifact type of the test set. BOTH mixes EOG and EMG "
                             "into every epoch, as the *_both runs were trained")

    # --- Mode selection ---
    parser.add_argument("--mode", default="all", choices=list(MODES),
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

    # --- Seeds ---
    parser.add_argument("--master_seed", type=int, default=None,
                        help="The run's data seed: it selects the train/val/test split. "
                             "Must match the --master_seed the checkpoint was trained "
                             "with, or the model is scored on the wrong split")

    # --- Artifact-switch probe ---
    parser.add_argument("--probe_seeds", type=int, nargs="+", default=list(SWITCH_PROBE_SEEDS),
                        help="Switch-probe seeds to average the response curve over. "
                             "These sit on top of --master_seed: the split stays put "
                             "and the switch points, SNRs and directions are redrawn")
    parser.add_argument("--switch_window", type=int, default=None,
                        help="Half-width of the response curve in samples "
                             "(default: the widest the switch range allows)")
    parser.add_argument("--switch_ramp", type=int, default=0,
                        help="Raised-cosine crossfade at the switch, in samples "
                             "(0 = hard switch; a control for the discontinuity)")
    parser.add_argument("--switch_snr", type=float, default=None,
                        help="Fix both artifact SNRs instead of sampling them "
                             "independently")
    parser.add_argument("--switch_range", type=float, nargs=2, default=[0.25, 0.75],
                        metavar=("LO", "HI"),
                        help="Switch position range as a fraction of the epoch")
    parser.add_argument("--switch_epochs", type=int, default=None,
                        help="Cap the number of probe epochs (default: the whole split)")
    parser.add_argument("--switch_control", action=argparse.BooleanOptionalAction, default=True,
                        help="Also run the stationary no-switch probe and report the "
                             "difference (removes the within-epoch warm-up trend)")

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
    no_ckpt_modes = {"curves", "compare_curves", "compare_snr", "compare_switch"}
    if args.mode not in no_ckpt_modes:
        if args.ckpt is None:
            parser.error(f"--ckpt is required for mode '{args.mode}'")
        if args.data_dir is None:
            parser.error(f"--data_dir is required for mode '{args.mode}'")

        from experiment.checkpoints import load_model
        overrides = {key: getattr(args, key) for key in (
            "variant", "turboquant", "turboquant_bits", "turboquant_key_mode", "turboquant_seed")}
        model, model_cfg, ckpt = load_model(args.ckpt, config=args.config, overrides=overrides)
        print(f"[Eval] Loaded checkpoint: {args.ckpt}  (epoch={ckpt['epoch']})")

    run_all = args.mode == "all"

    # ---- 1. Per-SNR evaluation ----
    if run_all or args.mode == "snr":
        evaluate_per_snr(
            model, args.data_dir,
            artifact_type=args.artifact,
            seed=args.master_seed,
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
            seed=args.master_seed,
        )

    # ---- 4. Attention heatmaps ----
    if run_all or args.mode == "attention":
        plot_attention_heatmaps(
            model, args.data_dir,
            artifact_type=args.artifact,
            snr=args.vis_snr,
            sample_idx=args.attn_sample,
            save_path=str(save_dir / "attention_heatmaps.png") if save_dir else None,
            seed=args.master_seed,
        )

    # ---- 5. Per-SNR metric line chart ----
    if run_all or args.mode == "snr_plot":
        if model is None:
            parser.error("--ckpt is required for snr_plot mode")
        results, _ = evaluate_per_snr(
            model, args.data_dir,
            artifact_type=args.artifact,
            seed=args.master_seed,
        )
        # --variant is optional; the checkpoint knows its own architecture.
        label = args.variant or model_cfg.get("variant") or Path(args.ckpt).parent.name
        variant_label = label.upper().replace("_", " ")
        plot_snr_curves(
            results,
            title=f"Per-SNR Metrics — {variant_label} ({args.artifact})",
            save_path=str(save_dir / "snr_curves.png") if save_dir else None,
        )

    # ---- 8a. Artifact-switch response (single model) ----
    switch_kwargs = dict(
        probe_seeds=args.probe_seeds,
        master_seed=args.master_seed,
        window=args.switch_window,
        control=args.switch_control,
        ramp_samples=args.switch_ramp,
        fixed_snr_db=args.switch_snr,
        switch_range=tuple(args.switch_range),
        n_epochs=args.switch_epochs,
    )
    switch_artifact = (evaluation_data(model, args.master_seed, args.artifact)["artifact_type"]
                       if run_all else None)
    if run_all and switch_artifact != "BOTH":
        print("[Switch] Skipping: the switch probe requires a BOTH-trained checkpoint")
    if args.mode == "switch" or (run_all and switch_artifact == "BOTH"):
        if args.data_dir is None:
            parser.error("--data_dir is required for switch mode")
        res = evaluate_switch_response(model, args.data_dir, **switch_kwargs)
        label = (args.variant or Path(args.ckpt).parent.name).upper().replace("_", " ")
        plot_switch_response(
            {label: res},
            save_path=str(save_dir / "switch_response.png") if save_dir else None,
        )
        if save_dir:
            with open(save_dir / "switch_response.json", "w") as f:
                json.dump({label: _switch_result_to_json(res)}, f, indent=2)

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
            master_seed=args.master_seed,
            save_path=str(save_dir / "compare_snr.png") if save_dir else None,
        )

    # ---- 8b. Artifact-switch response (multi-model) ----
    if args.mode == "compare_switch":
        if not compare_dirs:
            parser.error("--compare_runs is required for compare_switch mode")
        if args.data_dir is None:
            parser.error("--data_dir is required for compare_switch mode")
        compare_switch_response(
            compare_dirs,
            data_dir=args.data_dir,
            labels=compare_labels,
            save_path=str(save_dir / "switch_response.png") if save_dir else None,
            json_path=str(save_dir / "switch_response.json") if save_dir else None,
            **switch_kwargs,
        )


if __name__ == "__main__":
    main()
