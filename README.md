# Causal EEG Denoising with Multi-head Latent Attention

![Reproduce](https://github.com/ZhannoProject/project-v1/actions/workflows/reproduce.yml/badge.svg)

An ablation study comparing causal attention mechanisms for real-time EEG artifact removal, built on the [EEGdenoiseNet](https://github.com/ncclabsustech/EEGdenoiseNet) benchmark.

All models are strictly causal (no future information leakage), making them suitable for online brain-computer interface (BCI) streaming.

## Key idea

We take a Temporal Convolutional Network (TCN) baseline and insert a causal attention decoder in the middle, creating a **sandwich architecture**: local feature extraction (TCN) → global context mixing (attention) → reconstruction (TCN). The ablation isolates exactly one variable at a time — the attention mechanism — while keeping everything else identical.

The study also introduces **Multi-head Latent Attention (MLA)**, adapted from DeepSeek-V2, which compresses key-value pairs through a low-rank bottleneck. This reduces KV cache size during inference (relevant for embedded BCI hardware) and acts as implicit regularisation during training.

## Model variants

| Variant | Architecture | Attention | Params |
|---------|-------------|-----------|--------|
| `tcn` | Pure TCN baseline | None | 117k |
| `tcn_mha` | Sandwich: TCN → MHA → TCN | Multi-Head Attention | 168k |
| `tcn_mla` | Sandwich: TCN → MLA → TCN | Multi-head Latent Attention | 165k |
| `tcn_mha_par` | Parallel: TCN + MHA branches | Multi-Head Attention | 238k |
| `tcn_mla_par` | Parallel: TCN + MLA branches | Multi-head Latent Attention | 228k |
| `tcn_mla_r{N}` | MLA with fixed kv_lora_rank=N | MLA (rank sweep) | varies |

### Sandwich topology

```
(B,1,T) → TCN_pt1[1,2,4] → Squeeze → Attention Decoder → Expand → TCN_pt2[8,16,32] → Head → (B,1,T)
```

TCN_pt1 extracts local features with small dilations; the attention decoder mixes global context; TCN_pt2 reconstructs with large dilations. A channel bottleneck (`bottleneck_ratio`) compresses before attention and expands after.

### Parallel topology

```
(B,1,T) → TCN_pt1 → ┬─ Attention → (B,H,T) ─┐
                     └─ TCN_pt2   → (B,H,T) ─┘→ cat → Head → (B,1,T)
```

Both branches run side-by-side, then concatenate. This structurally prevents attention bypass — the head needs both branches.

### Anti-collapse mechanism

In the sandwich design, the standard residual connection `x + Attn(LN(x))` lets the optimizer bypass attention entirely. A **BottleneckSkip** replaces the identity skip with a lossy low-rank bottleneck, forcing the attention branch to carry fine-grained temporal information.

## Project structure

```
.
├── configs/                    # YAML experiment configs
│   ├── tcn_eog.yaml            #   TCN baseline
│   ├── tcn_mha_eog.yaml        #   TCN + MHA sandwich
│   ├── tcn_mla_eog.yaml        #   TCN + MLA sandwich
│   └── mla_rank_sweep_eog.yaml #   KV rank sweep
├── data/
│   ├── Dataloader.py           # Leak-free train/val/test splits + SNR mixing
│   └── EEGdenoisenet/data/     # .npy files (not in repo, see Data below)
├── models/
│   ├── TCN_backbone.py         # CausalConv1d → ResidualBlock → CausalTCNBackbone
│   ├── AttentionBlocks/
│   │   └── attention.py        # CausalMHA, CausalMLA, TransformerBlock/Decoder
│   └── variants.py             # 5 model variants + registry + build_model()
├── trainer/
│   ├── __main__.py             # CLI entry point: python -m trainer
│   ├── training.py             # Trainer class (fit / staged_fit)
│   ├── optimizer.py            # Per-component LR groups, freeze/unfreeze
│   ├── losses.py               # MSE, L1, STFT, L1+STFT, L2+STFT
│   ├── checkpoint.py           # Save/load checkpoints + history
│   └── logger.py               # TensorBoard integration
├── eval.py                     # 7-in-1 evaluation: metrics, curves, waveforms, heatmaps
├── metrics.py                  # RRMSE_temporal, RRMSE_spectral, ACC
├── reproduce.py                # Reproducibility smoke test (32 checks)
├── requirements.txt
└── .github/workflows/
    └── reproduce.yml           # CI: proves the code works from scratch
```

## Setup

```bash
# Clone
git clone https://github.com/ZhannoProject/project-v1.git
cd project-v1

# Create environment
python -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows

# Install dependencies (CPU-only torch is sufficient for training on this model size)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Data

This project uses the [EEGdenoiseNet](https://github.com/ncclabsustech/EEGdenoiseNet) benchmark dataset. Download the three `.npy` files and place them at:

```
data/EEGdenoisenet/data/
├── EEG_all_epochs.npy    # 4514 clean EEG epochs, each 512 samples
├── EOG_all_epochs.npy    # 3400 EOG artifact epochs
└── EMG_all_epochs.npy    # 5598 EMG artifact epochs
```

The data loader guarantees **zero leakage** between splits: clean EEG indices AND artifact indices are disjoint across train/val/test (epoch-level splitting with a fixed master seed).

## Training

```bash
# Train with a YAML config
python -m trainer --config configs/tcn_mla_eog.yaml

# Train with CLI overrides
python -m trainer --variant tcn_mha --artifact EOG --max_epochs 100

# Two-phase staged training (freeze TCN → unfreeze all)
python -m trainer --config configs/tcn_mla_eog.yaml --staged

# KV rank sweep
python -m trainer --config configs/mla_rank_sweep_eog.yaml --variant tcn_mla_r4
python -m trainer --config configs/mla_rank_sweep_eog.yaml --variant tcn_mla_r8
python -m trainer --config configs/mla_rank_sweep_eog.yaml --variant tcn_mla_r16
python -m trainer --config configs/mla_rank_sweep_eog.yaml --variant tcn_mla_r32
```

Config priority: **CLI flag > YAML file > hardcoded defaults**.

Checkpoints are saved to `runs/<variant>_<artifact>/` with `best.pt`, `last.pt`, and `history.json`.

## Evaluation

```bash
# Full evaluation (per-SNR metrics + training curves + waveforms + attention heatmaps)
python eval.py --ckpt runs/tcn_mla_eog/best.pt --data_dir ./data/EEGdenoisenet/data/ \
               --variant tcn_mla --artifact EOG

# Training curves only
python eval.py --history runs/tcn_mla_eog/history.json --mode curves

# Compare multiple models
python eval.py --mode compare_snr --data_dir ./data/EEGdenoisenet/data/ \
               --compare_runs runs/tcn_eog:TCN runs/tcn_mha_eog:MHA runs/tcn_mla_eog:MLA

# Save figures instead of displaying
python eval.py --ckpt runs/tcn_mla_eog/best.pt --data_dir ./data/EEGdenoisenet/data/ \
               --variant tcn_mla --artifact EOG --save_dir ./figures/
```

### Evaluation modes

| Mode | Description |
|------|-------------|
| `snr` | Per-SNR quantitative metrics (RRMSE_t, RRMSE_s, ACC) |
| `curves` | Training loss and metric curves from history.json |
| `waveform` | Time-domain + PSD comparison plots |
| `attention` | Per-head, per-layer attention weight heatmaps |
| `snr_plot` | Per-SNR metric line charts |
| `compare_curves` | Overlay training curves from multiple runs |
| `compare_snr` | Overlay per-SNR metrics from multiple models |
| `all` | Run all single-model tools (default) |

### Metrics

Following the EEGdenoiseNet benchmark:

- **RRMSE_temporal** — relative RMS error in the time domain (lower is better)
- **RRMSE_spectral** — relative RMS error in the frequency domain (lower is better)
- **ACC** — average Pearson correlation coefficient (higher is better)

## Reproducibility

The project includes an automated reproducibility test that verifies the entire pipeline from scratch:

```bash
# Full test (builds all models, checks causality, trains 3 epochs, evaluates)
python reproduce.py

# Quick test (skips training, ~10 seconds)
python reproduce.py --quick
```

This runs **32 automated checks** covering model construction, causality, losses, metrics, training, checkpointing, and evaluation. It synthesises random data if the real dataset is not present, so it works in CI without any external downloads.

The same script runs in GitHub Actions on every push — see the badge at the top.

## Ablation design

The study isolates one variable at a time while sharing everything else:

| Comparison | Variable | Controls |
|-----------|----------|----------|
| TCN vs TCN+Attention | Does global context help? | Same TCN backbone, loss, data |
| MHA vs MLA | Does KV compression hurt/help? | Same block structure, FFN, norms |
| Sandwich vs Parallel | Which topology prevents collapse? | Same components, different wiring |(not included in the paper yet, further experiment is expected)
| KV rank sweep (r4/r8/r16/r32) | How much compression is viable? | Same architecture, only rank varies |

## License

This project is part of an MPhil research study. Please contact the authors for usage terms.
