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
| `tcn_mha_bi`, `tcn_mla_bi`, `tcn_mha_par_bi`, `tcn_mla_par_bi` | Same as above, **bidirectional** attention (non-causal) | KD teachers only | same as causal twin |
| `tcn_bi` | Baseline with **non-causal convolutions** (centred padding) | None | 117k |
| `tcn_mha_full_bi`, `tcn_mla_full_bi`, `tcn_mha_par_full_bi`, `tcn_mla_par_full_bi` | Bidirectional attention **and** non-causal convolutions | KD teachers only | same as causal twin |

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

### Optional anti-collapse skip (`skip_bottleneck_ratio`)

In the sandwich design, the standard residual connection `x + Attn(LN(x))` lets the optimizer bypass attention entirely — the paper reports exactly this collapse on EMG. By default every model keeps that identity skip; the reference checkpoints and all reported results use it.

Setting `skip_bottleneck_ratio` (e.g. `0.5`) swaps the identity skip around each attention block for a **BottleneckSkip** — a lossy `Linear → GELU → Linear` low-rank bottleneck — so the attention branch has to carry the fine-grained temporal information. It works with every attention variant, is recorded in the checkpoint's `model_config`, exports to ONNX, and is inferred from the weights for legacy checkpoints. It is an experiment switch, not a validated improvement.

```bash
python -m trainer --config configs/tcn_mla_eog_bskip.yaml                   # ready-made preset
python -m trainer --config configs/tcn_mha_eog.yaml --skip_bottleneck_ratio 0.5 \
       --save_dir runs/tcn_mha_eog_bskip                                    # any preset
```

Not to be confused with `bottleneck_ratio`, which is the channel squeeze between the TCN (H) and the attention decoder (H' = H × ratio).

## Project structure

```
.
├── configs/                    # YAML experiment configs
│   ├── tcn_eog.yaml            #   TCN baseline
│   ├── tcn_mha_eog.yaml        #   TCN + MHA sandwich
│   ├── tcn_mla_eog.yaml        #   TCN + MLA sandwich
│   ├── mla_rank_sweep_eog.yaml #   KV rank sweep
│   ├── tcn_mla_eog_bskip.yaml  #   MLA sandwich with the optional BottleneckSkip
│   ├── tcn_{mha,mla}_bi_eog.yaml   # bidirectional teachers (non-causal)
│   ├── tcn_{mha,mla}_eog_kd.yaml   # causal students distilled from them (same size)
│   └── tcn_mla_eog_kd_small.yaml   # a 32-channel student distilled from the 64-channel teacher
├── data/
│   ├── Dataloader.py           # Leak-free train/val/test splits + SNR mixing (seeded)
│   └── EEGdenoisenet/data/     # .npy files (not in repo, see Data below)
├── models/
│   ├── TCN_backbone.py         # CausalConv1d → ResidualBlock → CausalTCNBackbone
│   ├── AttentionBlocks/
│   │   └── attention.py        # CausalMHA, CausalMLA, TransformerBlock/Decoder
│   └── variants.py             # 5 model variants + registry + build_model()
├── trainer/
│   ├── __main__.py             # CLI entry point: python -m trainer
│   ├── training.py             # Trainer class (fit / staged_fit)
│   ├── distillation.py         # KDTrainer + DistillationLoss: bidirectional teacher → causal student (optionally smaller)
│   ├── optimizer.py            # Per-component LR groups, freeze/unfreeze
│   ├── losses.py               # MSE, L1, STFT, L1+STFT, L2+STFT
│   ├── checkpoint.py           # Save/load checkpoints + history
│   └── logger.py               # TensorBoard integration
├── gui/
│   ├── __main__.py             # python -m gui → local web GUI for the YAML presets + launching runs
│   ├── config_io.py            # field schema + comment-preserving YAML writer
│   ├── server.py               # stdlib HTTP/JSON API, training-job runner
│   └── static/index.html       # the page
├── deploy/
│   ├── export_onnx.py          # checkpoint → verified ONNX (+ sidecar JSON) for Raspberry Pi 5
│   ├── run_on_pi.sh            # one command: copy to the Pi over SSH, set up, benchmark, score
│   ├── raspi_infer.py          # numpy + onnxruntime inference (epoch + causal streaming), --testdata scoring, per-hop latency monitor
│   ├── make_testdata.py        # cut a small raw-scale test set from the leak-free test split
│   ├── testdata/EOG_test100.npz#   100 EOG test epochs, 10 per SNR level, ready for the Pi
│   └── requirements-raspi.txt  # the only two packages the Pi needs
├── utils/utils.py              # get_device(), seed_everything()
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

### Seeding

`master_seed` (default 42, `--master_seed N` on the CLI) is the single seed for a run: the train/val/test split, artifact pairing and SNR draws (numpy), the training batch order (a dedicated `torch.Generator`, so it is identical across variants), and weight initialisation and dropout (`seed_everything` in `utils/utils.py`, which also switches cuDNN / `torch.use_deterministic_algorithms` to deterministic kernels).

```bash
python -m trainer --config configs/tcn_mla_eog.yaml --master_seed 1 --save_dir runs/tcn_mla_eog_s1
python -m trainer --config configs/tcn_mla_eog.yaml --device cpu     # bit-exact repeats
```

Two runs of the same config give bit-identical checkpoints on CPU (verified) and on CUDA with the deterministic kernels; **Apple MPS is not deterministic even when seeded** (two runs drift at the 1e-4 level within one epoch), so the trainer prints a reminder and `--device cpu` (or `device: cpu` in the YAML / GUI) is the option for exact repeats. Results across different hardware or torch versions can still differ at floating-point level. Checkpoints produced before seeding was added were not seeded, so their run-to-run variance is unknown — re-run a config with a few seeds before reading small differences as real.

### Training GUI

A local app that edits the presets in `configs/` and launches
`python -m trainer` from them. The page is served by the stdlib
`http.server` and shown in a native window (via `pywebview`, listed in
`requirements.txt`); without `pywebview` it opens in a browser tab instead.

```bash
python -m gui                     # native window (default when pywebview is installed)
python -m gui --browser           # serve http://127.0.0.1:8765/ and open a browser tab
python -m gui.make_app            # macOS: build a double-clickable "EEG Training GUI.app"
```

`make_app` writes a launcher bundle (with icon) into the project root that
runs *this* venv's Python with `-m gui --native`, so it always uses the
current code; the paths live in `Contents/Resources/launch.cfg`, so rebuild
(or edit that file) if you move the project or the venv. The launcher is
`gui/launcher.c` compiled with the system `cc` (Xcode Command Line Tools):
it has to be a real binary that keeps Python as a child process, because
macOS attributes Desktop/Documents folder permissions to the launched app —
a script that `exec`'d Python would hand that identity away and the venv
would be unreadable without any prompt. On first launch macOS asks for
access to the folder that holds the project — allow it (System Settings ›
Privacy & Security › Files and Folders if you need to change it later).
Failures show an alert pointing at `~/Library/Logs/eeg-training-gui.log`.
The bundle is git-ignored. Closing the window stops the server (and asks
first if a training run is still going).

- **Form ↔ YAML.** Every key of `trainer/__main__.py` is a field, grouped as in
  the preset files (Model, Data, Training, Scheduler, Staged, Loss, Output,
  TurboQuant, Knowledge distillation). Fields that do not apply to the selected
  variant are hidden (e.g. `kv_lora_rank` for MHA, attention settings for `tcn`);
  keys absent from the file are tagged `default` and only written if you change them.
- **Comments survive.** The GUI never re-serialises a file: it rewrites only the
  `key: value` lines that changed and appends new keys under their section
  header, so `5.0e-4`, inline comments and unknown keys such as `rank_sweep` are
  kept byte-for-byte. The YAML tab shows the exact text a save will write, with
  the changed lines highlighted.
- **Save / Save as… / New…** write to `configs/`; **Train** runs
  `python -m trainer --config configs/<file> [extra flags]` with the same
  interpreter and streams the log into the page (also mirrored to
  `runs/gui_train_<timestamp>.log`); **Stop** terminates it.
- **Live metric curve.** While a run is going, the Train tab plots RRMSE_t,
  RRMSE_s and ACC per epoch, with the current values, the best epoch so far and
  the epoch counter. The numbers are parsed out of the training log as it
  streams, so nothing in the trainer changes and the curve survives a page
  reload (the server keeps them for the running job). The y-axis is fitted to
  the bulk of the run — epoch 1 is often several times worse than the rest and
  would otherwise flatten everything; the caption says when early epochs run
  off the top.
- **Runs** has five views. *Complete by kind* groups finished runs by
  `variant × artifact` (the thing an ablation row compares), best RRMSE_t first,
  with the depth, KV rank and parameter count read from the checkpoint itself.
  *Complete by loss* groups them by `loss_fn × artifact` instead, ranked by
  val loss. This is the grouping to read val loss off: `val_loss` is whatever
  loss that run minimised on whichever artifact it trained against, so the same
  number means different things across groups — an MSE 0.13 and an L1+STFT 0.44
  can be the same model. Every row carries a badge with its loss, solid when the
  run recorded it and dashed when it was recovered from the weights (see below);
  runs whose loss is unknown are grouped separately rather than mixed in.

  *Complete by optimizer* groups them by the recipe their checkpoint records —
  optimizer family, configured LR, decoder-to-TCN LR ratio and schedule. Nothing
  has to be recomputed for this: `optimizer_state` and `scheduler_state` are
  saved by every checkpoint, so it works on runs of any age. Each row also shows
  an `Adam`/`AdamW` badge whose tooltip carries the full recipe.

  *By time* is the other kind of question: not which run is best but what you
  ran, and when. Every run, newest first, grouped by the day it last wrote, with
  the clock time on each row — and unlike the three views above it keeps the
  interrupted and stale runs, because a run that died is the main thing you come
  here looking for. Checkpoints written since September 2026 record their own
  `saved_at`; older runs fall back to the mtime of `last.pt`, which is the same
  answer until the directory is copied or restored. Those rows are marked with a
  `~` so a surprising order can be explained rather than puzzled over.

  *All runs* lists every `runs/*/best.pt` and marks the ones the grouped views
  leave out:
  - `partial` — no `history.json`: the run never reached the end.
  - `stale` — `history.json` and the `last.pt` beside it are from different
    runs, i.e. the directory was re-used by a later (often shorter) run. The
    metrics in that row describe weights that no longer exist, so the row is
    not comparable with anything.

  **curve** on any finished run plots its saved history in the same chart, and
  “Use as teacher” fills `teacher_ckpt` for a distillation run.

The server binds to 127.0.0.1 only and runs one training job at a time.
Validation: `python -m unittest discover -s tests -p 'test_gui.py'`.

### Which loss a run used

`val_loss` is only comparable across runs that share `loss_fn` **and**
`artifact` (and `snr_range_db` / `master_seed`). Runs record their `loss_fn` in
`history.json["config"]` and in every checkpoint, so the GUI can group by it.

Runs trained before that was recorded carry no such field. Recover it:

```bash
python -m trainer.identify_loss              # runs with no recorded loss
python -m trainer.identify_loss --dry-run    # report without writing
python -m trainer.identify_loss --run tcn_eog --all
```

It rebuilds each model from `best.pt`, evaluates it on that run's own validation
split and computes the val loss under every loss in the registry; exactly one
reproduces the recorded number. The result goes to `runs/<run>/loss_fn.json`,
which the GUI reads. Nothing is claimed unless it is safe to:

- the recomputed RRMSE_t must match the RRMSE_t in `history.json`, proving
  `best.pt` really is the model that history describes (a re-used directory
  fails here and is reported as `checkpoint-mismatch`);
- the winning candidate must be far closer than the runner-up, since `l1` and
  `l2_combined` can land within a percent of each other.

Validation: `python -m unittest discover -s tests -p 'test_identify_loss.py'`.

### Which optimizer a run used

Read straight out of the checkpoint, so no tool run is needed:

- **family** — `Adam` and `AdamW` are indistinguishable by their state *keys*;
  torch 2.x implements AdamW as Adam with `decoupled_weight_decay=True`, and
  that flag is what separates them. When it is absent (older torch) the family
  is reported as `Adam/AdamW` rather than guessed.
- **schedule** — `ReduceLROnPlateau` keeps `factor`/`patience`; this project's
  warmup→cosine chain is a `SequentialLR` holding a `LinearLR` and a
  `CosineAnnealingLR`; an empty state means no scheduler.
- **learning rate** — `initial_lr` is the configured value and is exact. Without
  it, only the *decayed* rate at save time survives; those runs are flagged
  `(decayed)` and grouped together rather than split by how far each had decayed.
- **attention LR scale** — decoder LR ÷ TCN LR. Both groups decay together, so
  this ratio is exact even when the rates themselves are not.

Runs trained from now on also record the recipe explicitly in
`history.json["config"]` (`optimizer`, `learning_rate`, `attn_lr_scale`,
`weight_decay`, `sched_warmup_epochs`, `eta_min`).

### Transformer depth choices

The MHA and MLA EOG presets offer 2, 10, 20, or 50 transformer layers.
The training entry point passes `n_attn_layers` to the model; the default is 2.

| Layers | MHA config | MLA config |
|--------|------------|------------|
| 2 | `configs/tcn_mha_eog.yaml` | `configs/tcn_mla_eog.yaml` |
| 10 | `configs/tcn_mha_eog_l10.yaml` | `configs/tcn_mla_eog_l10.yaml` |
| 20 | `configs/tcn_mha_eog_l20.yaml` | `configs/tcn_mla_eog_l20.yaml` |
| 50 | `configs/tcn_mha_eog_l50.yaml` | `configs/tcn_mla_eog_l50.yaml` |

```bash
python -m trainer --config configs/tcn_mla_eog_l20.yaml
python -m trainer --config configs/tcn_mha_eog_l50.yaml
# Custom depth (use a distinct output directory for each experiment):
python -m trainer --config configs/tcn_mla_eog.yaml --n_attn_layers 10 --save_dir runs/tcn_mla_eog_custom_l10
```

Each deeper preset has its own output directory and otherwise keeps the
corresponding 2-layer preset settings. These are experiment choices, not
validated improvements. Deeper models require more memory and training time;
reduce `--batch_size` if needed.

### Knowledge distillation: bidirectional teacher → causal student

Every attention variant has a **bidirectional twin** (`tcn_mla_bi`, `tcn_mha_bi`,
`tcn_mha_par_bi`, `tcn_mla_par_bi`): identical weights and shapes, but the
attention mask is dropped so each position sees the whole 512-sample window.
The TCN stages in those twins stay causal — only attention looks ahead.

Causality is really two switches, and the registry exposes every useful
setting of them:

| Variant | Attention | Convolutions | What it isolates |
|---------|-----------|--------------|------------------|
| `tcn_mla` | causal | causal | the deployable model |
| `tcn_mla_bi` | bidirectional | causal | attending to the future, and nothing else |
| `tcn_mla_full_bi` | bidirectional | non-causal | the full offline ceiling |
| `tcn_bi` | — | non-causal | seeing the future without attention |

`tcn` has no attention to unmask, so `tcn_bi` is already fully non-causal and
there is no `tcn_full_bi`. All of them keep the weights and shapes of their
causal twin, so any of them can teach it.
Such a model cannot run online — it needs future samples — but on the same
weight budget it is a stronger denoiser. It is used as a frozen **teacher**,
and the deployable causal model is trained as a **student** to reproduce, from
past-only context, what the teacher produces with full context:

```
noisy ──► teacher (frozen, causal=False) ──► teacher_out, teacher.decoder
  │                                              │            │
  └─────► student (trained, causal=True)  ──► student_out, student.decoder
                                                 │            │
L = (1 − α) · loss(student_out, clean)           │            │
  +      α  · loss(student_out, teacher_out) ◄───┘            │
  +      β  · MSE(student.decoder, teacher.decoder) ◄─────────┘
```

The output term uses the same waveform loss as training (`loss_fn`); the
feature term matches the post-attention `(B, H', T)` tensor — exactly the block
that differs between the two models — and works on any dotted submodule path.
Because the state dicts are identical, the student can also be initialised from
the teacher (`init_from_teacher`).

```bash
# 1. Train the teacher (non-causal; never deploy this one)
python -m trainer --config configs/tcn_mla_bi_eog.yaml         # → runs/tcn_mla_bi_eog/

# 2. Distil the causal student from it
python -m trainer --config configs/tcn_mla_eog_kd.yaml         # → runs/tcn_mla_eog_kd/

# Or add distillation to any preset from the CLI
python -m trainer --config configs/tcn_mha_eog.yaml \
       --teacher_ckpt runs/tcn_mha_bi_eog/best.pt --kd_alpha 0.5 \
       --kd_feature_weight 0.1 --init_from_teacher --save_dir runs/tcn_mha_eog_kd
```

| Config key | Default | Meaning |
|------------|---------|---------|
| `teacher_ckpt` | — | Teacher checkpoint; setting it switches to `KDTrainer` |
| `teacher_config` | — | YAML of the teacher, only for legacy checkpoints without `model_config` |
| `kd_alpha` | `0.5` | Weight of the teacher-output term (α); the ground-truth term gets 1 − α |
| `kd_feature_weight` | `0.0` | Weight of the feature term (β); `0` disables the feature taps |
| `kd_feature_layers` | `["decoder"]` | Submodules to match, e.g. `["decoder", "decoder.blocks.0"]` |
| `kd_output_loss` | `loss_fn` | Loss between student and teacher outputs (`mse`, `l1`, `stft`, `l1_combined`, `l2_combined`) |
| `kd_feature_loss` | `mse` | `mse` or `l1` between features |
| `init_from_teacher` | `false` | Copy the teacher's weights into the student before training |

`kd_alpha: 0` and `kd_feature_weight: 0` recover ordinary training. The KD
presets use α = 0.5, β = 0.1 and `init_from_teacher: true` as starting points —
they are experiment choices, not validated optima. `--staged` works with KD.

#### Compressing the student (optional)

By default the student is the teacher's causal twin — same size, only the
attention mask differs — so KD transfers *hindsight*, not capacity. The
student can also be **smaller**: its size comes from its own model keys
(`hidden_channels`, `n_attn_layers`, `kv_lora_rank`, `bottleneck_ratio`,
even the variant), so a KD preset simply stops mirroring the teacher.
`configs/tcn_mla_eog_kd_small.yaml` distils the 64-channel teacher into a
32-channel student (42.6k vs 166.6k params, 3.9× smaller):

```bash
python -m trainer --config configs/tcn_mla_bi_eog.yaml            # teacher, 64 ch
python -m trainer --config configs/tcn_mla_eog_kd_small.yaml      # student, 32 ch
# or shrink any KD run from the CLI
python -m trainer --config configs/tcn_mla_eog_kd.yaml --hidden_channels 32 \
       --kv_lora_rank 8 --save_dir runs/tcn_mla_eog_kd_32
```

What changes when the sizes differ:

| Term | Behaviour with a smaller student |
|------|----------------------------------|
| output (α) | Unchanged — both models emit `(B, 1, T)` |
| feature (β) | A learnable **projector** (one `Linear` along the channel axis, pointwise in time, hence causal) maps each tapped student feature to the teacher's width. It is trained with the student in its own optimizer group (also through `--staged`) and is **not** saved in the checkpoint. Depth mismatch: name `student:teacher` pairs, e.g. `kd_feature_layers: ["decoder", "decoder.blocks.1:decoder.blocks.9"]` |
| `init_from_teacher` | Same-name tensors of different shape receive the teacher's **leading slice** (first C_s of C_t channels, first n of N blocks); the rest keeps its random init. The log reports how many tensors were copied whole, sliced, or skipped |

The trainer prints the student/teacher parameter ratio, and `history.json`
records both under `student` and `teacher`. `best.pt` is an ordinary
student checkpoint (its `model_config` is the small architecture), so
`eval.py` and the ONNX exporter need nothing extra. The natural control is
the same small architecture trained without KD (`teacher_ckpt: null`).

The student is validated and checkpointed exactly like a normal run (against
the clean signal), so `runs/tcn_mla_eog_kd/best.pt` is a drop-in for `eval.py`
and `deploy/export_onnx.py`. Its `history.json` additionally records the
per-epoch `kd_task` / `kd_output` / `kd_feature` terms and the teacher's own
validation metrics under `teacher`. The ONNX exporter refuses `*_bi`
checkpoints, since they are not causal.

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

## Deployment: ONNX on Raspberry Pi 5

The trained sandwich model (by default the 2-layer `tcn_mla`) can be exported
to a single ONNX file and run on a Raspberry Pi 5 with nothing but
`numpy` + `onnxruntime` — no PyTorch on the device.

### 1. Export on the host

```bash
pip install onnx onnxruntime onnxscript          # export-time dependencies
python -m deploy.export_onnx --ckpt runs/tcn_mla_eog/best.pt \
       --data_dir data/EEGdenoisenet/data --artifact EOG
```

This writes `deploy/artifacts/tcn_mla_eog.onnx` (fp32, opset 18, ~1.2 MB,
dynamic batch axis, standard `ai.onnx` ops only) and a `tcn_mla_eog.json`
sidecar recording the architecture, the I/O contract, the preprocessing rule
and the verification results. The export is refused unless the graph:

- matches PyTorch to < 1e-4 on random windows (typically ~1e-6),
- is **strictly causal** (a perturbation at sample 256 changes nothing before it),
- and, when `--data_dir` is given, reproduces the test-split RRMSE_t / RRMSE_s / ACC
  of the PyTorch model (identical to 4 decimals for the reference checkpoint).

The architecture is inferred from the weights, not from the YAML, so older
checkpoints without `model_config` export correctly — e.g. the reference
`runs/tcn_mla_eog/best.pt` was trained with `kv_lora_rank=2`, not the 16 in
its preset. `n_heads` is the one setting the weights cannot reveal; pass
`--n_heads` if a run used something other than 4. Any variant exports the
same way (`--ckpt runs/tcn_mha_emg/best.pt`, …). TurboQuant is always
stripped: it is a fake-quantisation experiment with no runtime benefit.

Options: `--seq_len` bakes in a shorter window (≤ `max_seq_len`; attention is
O(T²)), `--static_batch` fixes batch = 1, `--max_batches N` makes the metric
check quick.

### 2. One-shot test over SSH

With the Pi reachable by SSH, a single command from this machine copies the
five files it needs, creates a venv there on first use, installs `numpy` +
`onnxruntime`, and runs the benchmark plus the bundled test set:

```bash
deploy/run_on_pi.sh pi@raspberrypi.local
```

`deploy/run_on_pi.sh pi@192.168.1.42 tcn_eog` tests another exported model;
`HOP=64 THREADS=3 deploy/run_on_pi.sh …` tunes the streaming hop and thread
count. Re-runs reuse the venv and take a few seconds. The script refuses to
start if the export or test set is missing and prints the command to create
it.

### 3. Manual steps on the Pi

```bash
# on the Pi (64-bit Raspberry Pi OS, Python 3.11+)
python -m venv ~/eeg-env && source ~/eeg-env/bin/activate
pip install -r requirements-raspi.txt        # numpy + onnxruntime (aarch64 wheels)
python raspi_infer.py tcn_mla_eog.onnx --benchmark
python raspi_infer.py tcn_mla_eog.onnx --input noisy.npy --output clean.npy          # (N, 512) epochs
python raspi_infer.py tcn_mla_eog.onnx --input rec.npy --output clean.npy --stream --hop 32
```

Only `raspi_infer.py`, `requirements-raspi.txt`, the `.onnx` and its `.json`
need to be copied (plus a test `.npz`, see below). From Python:

```python
from raspi_infer import EEGDenoiser, StreamingDenoiser
d = EEGDenoiser("tcn_mla_eog.onnx")            # 2 onnxruntime threads (measured best on a Pi 5)
clean = d.denoise_epochs(noisy)                 # (N, 512) raw µV in, raw µV out
clean = d.denoise_stream(recording, hop=32)     # any length, sample-aligned

s = StreamingDenoiser(d, hop=32)                # live acquisition loop
for chunk in acquire():                         # chunks of any size
    out = s.push(chunk)                         # denoised samples as they become available
```

Inputs are raw signal; each 512-sample window is divided by its own standard
deviation before the model and multiplied back afterwards, exactly as
`data/Dataloader.py` does in training. Streaming exploits causality: the model
sees the latest 512 samples and only the newest `hop` outputs are kept, so
every emitted sample has the same left-context it had in training. Latency is
one forward pass per hop (hop 32 at 256 Hz = 125 ms cadence); the first
output appears once a full window (2 s) has been acquired. The push-based
`StreamingDenoiser` is bit-identical to `denoise_stream`.

Measured on a Raspberry Pi 5 Model B (4 GB, Python 3.13, onnxruntime 1.30,
ondemand governor) for the shipped 2-layer MLA model, one 512-sample window:

| onnxruntime threads | median | p95 | max | headroom at hop 32 (125 ms) |
|---|---|---|---|---|
| 1 | 24.4 ms | 24.7 ms | 31.5 ms | 5.1× |
| **2 (default)** | **18.2 ms** | **18.9 ms** | **19.2 ms** | **6.6×** |
| 3 | 17.4 ms | 19.3 ms | 23.7 ms | 6.5× |
| 4 | 20.0 ms | 29.7 ms | 40.1 ms | 4.2× |

Two threads give the best tail latency; using all four cores contends with
the OS and doubles the p95. The test-set metrics on the Pi are identical to
the host (epoch mode RRMSE_t 0.4601 / RRMSE_s 0.4643 / ACC 0.8826). On an
Apple-silicon host the same window takes ~2 ms. `--benchmark` prints the
numbers for whatever board it runs on.

### Ready-made test data for the Pi

`deploy/testdata/EOG_test100.npz` (373 KB) holds 100 epochs cut from the
project's own leak-free **test** split at raw scale — `noisy`, `clean`,
`snr_db` — 10 epochs at each of the 10 SNR levels from −7 dB to +2 dB. Score
it on the Pi with the same command you would use on the host:

```bash
python raspi_infer.py tcn_mla_eog.onnx --testdata EOG_test100.npz
```

This prints RRMSE_t / RRMSE_s / ACC for epoch mode, streaming mode, and the
untouched noisy input as a baseline, plus a per-SNR table and timings. The
metrics are numpy replicas of `metrics.py` (they agree to 2e-7), so host and
Pi numbers are directly comparable. Reference for the shipped MLA model:
epoch mode RRMSE_t 0.460 / RRMSE_s 0.464 / ACC 0.883 versus 1.41 / 4.91 /
0.595 for the raw noisy input. The streaming row is a pessimistic bound —
the epochs are independent 2-s segments, so every window straddles two
unrelated signals — and is not what a continuous recording would give.

To cut a different subset (other artifact, more epochs) on the host:

```bash
python -m deploy.make_testdata --data_dir data/EEGdenoisenet/data --artifact EMG --n 200
```

### Per-hop latency monitor

`--benchmark` times one window in a tight loop. `--latency` instead watches
the streaming path hop by hop — what a live loop on the Pi actually
experiences — and works with `--testdata` and with `--stream`:

```bash
python raspi_infer.py tcn_mla_eog.onnx --testdata EOG_test100.npz --latency --latency_every 400
python raspi_infer.py tcn_mla_eog.onnx --input rec.npy --stream --realtime --latency --latency_out lat/rec
```

For every hop it records the onnxruntime call (`infer_ms`), the whole hop
including buffer roll, normalisation and rescaling (`total_ms`), the
completion time and — on Raspberry Pi OS — the SoC temperature. The summary
gives median / p95 / p99 / max, jitter (p95 − median), **deadline misses**
against the hop budget (`hop / fs`, 125 ms at hop 32), real-time headroom,
throughput and the temperature drift over the run:

```
[latency/testdata] 1584 hops of 32 samples (125.0 ms budget each), 4.5 s wall, first window 3.1 ms
           per hop   : median 2.821 | p95 2.973 | p99 3.565 | max 16.474 ms  (jitter p95-median 0.152 ms)
           of which  : onnxruntime 2.806 ms median / 2.936 p95;  pre/post-processing 0.015 ms median
           real time : 0 deadline misses (0.00%), 42.04x headroom at p95, 317.19 hops/s
```

`--latency_every N` prints a rolling-window line every N hops (useful for
spotting thermal throttling during a long run); `--latency_out STEM` writes
`STEM.json` (the summary) and `STEM.csv` (one row per hop: `hop, infer_ms,
total_ms, t_s, temp_c`) for plotting.

**`--realtime`** replays a `--stream` input at the sample rate through
`StreamingDenoiser`, pushing one hop of samples only once it "has arrived"
(so it takes as long as the recording). The monitor then also reports
**end-to-end latency** — from a hop's last sample being available to its
denoised samples being emitted, scheduler jitter included — and the CSV gains
an `e2e_latency_ms` column. Expect the paced per-hop cost to be higher than
the back-to-back number (the cores idle between hops, so caches are cold and
the clock has dropped): on an Apple-silicon host the same model measures
2.8 ms back-to-back but 12 ms paced, with 17 ms end-to-end. The paced figure
is the honest one for a live loop; both are far inside the 125 ms budget.

`deploy/run_on_pi.sh` runs `--latency` as part of the standard Pi test and
copies the trace back to `deploy/artifacts/pi_latency/<model>_hop<hop>.{json,csv}`
(`LATENCY_EVERY=200` changes the reporting cadence). From Python:

```python
from raspi_infer import EEGDenoiser, StreamingDenoiser, LatencyMonitor, stream_realtime
d = EEGDenoiser("tcn_mla_eog.onnx")
m = LatencyMonitor(hop=32, sample_rate=d.sample_rate, report_every=100)
s = StreamingDenoiser(d, hop=32, monitor=m)     # live acquisition loop
for chunk in acquire():
    out = s.push(chunk)
m.print_summary(); m.save("lat/session")        # dict + .json/.csv

clean = stream_realtime(d, recording, hop=32, monitor=m)   # paced replay of a recording
```

### Duty-cycle check: is real-time processing applicable?

Every hop costs one full-window forward pass (~16 ms on a Pi 5) regardless of
the hop size, so the **duty cycle** — compute per hop ÷ hop period
(`hop / fs`) — doubles each time the hop is halved. `--duty_cycle` sweeps hop
sizes with a paced replay and reports, per hop, the duty cycle, the end-to-end
latency and its drift over the run (a loop that cannot keep up emits later and
later), deadline misses and a verdict: **OK** (p99 duty ≤ 70 %, no misses,
no drift), **MARGINAL**, or **FAIL** (misses, or a backlog of more than one
period).

```bash
python raspi_infer.py tcn_mla_eog.onnx --testdata EOG_test100.npz --duty_cycle 64,32,16,8,4,2 --duty_seconds 20
DUTY_HOPS=64,32,16,8,4,2 DUTY_S=20 deploy/run_on_pi.sh pi@raspberrypi.local     # same, over SSH, results copied back
```

Measured on the Raspberry Pi 5 (Model B Rev 1.1, Python 3.13.5, onnxruntime
1.30, 2 threads, 20 s paced replay per hop, never throttled, 52 → 58 °C):

| hop | period | forwards/s | compute median / p99 | duty cycle mean / p99 | e2e latency median / max | drift over 20 s | misses | verdict |
|---|---|---|---|---|---|---|---|---|
| 64 | 250 ms | 4 | 16.1 / 20.9 ms | 6.8 % / 8.4 % | 16.2 / 21.0 ms | +0.0 ms | 0 | **OK** |
| 32 | 125 ms | 8 | 18.0 / 18.7 ms | 14.0 % / 14.9 % | 18.1 / 20.0 ms | +2.1 ms | 0 | **OK** |
| 16 | 62.5 ms | 16 | 16.1 / 16.5 ms | 25.7 % / 26.5 % | 16.1 / 20.6 ms | +0.1 ms | 0 | **OK** |
| 8 | 31.3 ms | 32 | 16.0 / 17.6 ms | 51.5 % / 56.3 % | 16.1 / 18.5 ms | +0.1 ms | 0 | **OK** |
| 4 | 15.6 ms | 64 | 16.1 / 17.7 ms | 103 % / 113 % | 368 / 759 ms | +565 ms | 1152 (all) | FAIL |
| 2 | 7.8 ms | 128 | 16.1 / 17.7 ms | 206 % / 227 % | 9.7 / 19.5 s | +14.6 s | 2304 (all) | FAIL |

So real-time processing is applicable down to **hop 8**: 31 ms of buffering
plus 16 ms of compute gives **≈ 47 ms total algorithmic latency at a 52 %
duty cycle**, with no drift over the run. Hop 4 needs 64 forward passes per
second and the board delivers ~62, so the loop falls behind at ~28 ms per
second of signal; hop 2 is twice as bad. The default hop 32 (≈ 143 ms total
latency, 14 % duty) leaves ample margin for an acquisition driver and a
downstream classifier on the same board. Pushing below hop 8 would need a
KV cache / incremental attention so that the per-hop cost scales with the hop
instead of the window (see below), not more threads: 3–4 threads do not make
one 512-sample forward faster than ~15 ms on this SoC.

### What the export does not do

- **No INT8.** Dynamic INT8 quantisation was tried and rejected: it derives
  activation scales per tensor at run time, so a future sample changes the
  scale applied to past samples and the graph stops being causal
  (past_diff = 0.32 on the reference checkpoint). It was also slower for
  these tiny matmuls and slightly less accurate (ACC 0.8754 vs 0.8772).
- **No KV cache.** Each call recomputes attention over the whole window. With
  T = 512 and d_model = 32 this is negligible; the KV-cache size advantage of
  MLA is therefore not exercised by this deployment path.

Validation: `python -m unittest discover -s tests -p 'test_export_onnx.py'`
covers architecture inference (including legacy checkpoints and `tcn_mla_r{N}` labels), export, the per-hop latency monitor,
agreement, causality, and the Pi-side epoch/streaming contract.

## Reproducibility

The project includes an automated reproducibility test that verifies the entire pipeline from scratch:

```bash
# Full test (builds all models, checks causality, trains 3 epochs, evaluates)
python reproduce.py

# Quick test (skips training, ~10 seconds)
python reproduce.py --quick
```

This runs **90 automated checks** covering model construction, causality (including that the `*_bi` teachers do see the future, and that the optional BottleneckSkip stays causal), seeding (same `master_seed` → same init and batch order), losses, metrics, training, checkpointing, evaluation, and knowledge distillation. It synthesises random data if the real dataset is not present, so it works in CI without any external downloads.

The same script runs in GitHub Actions on every push — see the badge at the top.

## Ablation design

The study isolates one variable at a time while sharing everything else:

| Comparison | Variable | Controls |
|-----------|----------|----------|
| TCN vs TCN+Attention | Does global context help? | Same TCN backbone, loss, data |
| MHA vs MLA | Does KV compression hurt/help? | Same block structure, FFN, norms |
| Sandwich vs Parallel | Which topology prevents collapse? | Same components, different wiring |(not included in the paper yet, further experiment is expected)
| KV rank sweep (r4/r8/r16/r32) | How much compression is viable? | Same architecture, only rank varies |
| Causal vs bidirectional (`X` vs `X_bi`) | What does causality cost? | Same weights and shapes, only the attention mask differs |
| `tcn_bi` vs `tcn_mla_bi` | Is the gain from seeing the future, or from attending to it? | Both non-causal; only one has attention |
| `X_bi` vs `X_full_bi` | How much is attention alone worth against the full offline ceiling? | Same weights; only the convolution padding differs |
| Student vs distilled student (`X` vs `X` + KD) | Can a non-causal teacher close that gap? | Same model, only the training signal differs |

## License

This project is part of an MPhil research study. Please contact the authors for usage terms.

## Optional TurboQuant

TurboQuant is an optional, parameter-free K/V quantization module for all four
attention architectures and MLA rank aliases. It is disabled by default and
can be combined with any transformer depth. TCN-only cannot enable it.

This independent reference implementation follows sections 3.1–3.2 of
[TurboQuant (Zandieh et al.)](https://arxiv.org/abs/2504.19874): random orthogonal
rotation, scalar Lloyd-Max quantization fitted to the exact sphere-coordinate
marginal, and optional QJL residual correction for keys. Values use the MSE
variant. In MLA it quantizes the reconstructed per-head K/V, not the latent.

| Config key | Default | Options |
|------------|---------|---------|
| `turboquant` | `false` | `true` / `false` |
| `turboquant_bits` | `3` | integers 2–8; excludes norms and byte padding |
| `turboquant_key_mode` | `prod` | `prod` (bits−1 scalar bits + 1 QJL bit), `mse` |
| `turboquant_train_mode` | `off` | `off` (evaluation only), `ste` (experimental straight-through gradients) |
| `turboquant_seed` | `42` | nonnegative integer below 2^63; fixed per-layer random transforms |

Ready-made presets: `configs/tcn_mha_eog_turboquant.yaml` and
`configs/tcn_mla_eog_turboquant.yaml`. Both start with 2 layers.

```bash
# Train normally; validation uses TurboQuant when train_mode is off.
python -m trainer --config configs/tcn_mla_eog_turboquant.yaml --n_attn_layers 20 --save_dir runs/tcn_mla_eog_l20_tq3
# Or enable on any existing attention preset:
python -m trainer --config configs/tcn_mha_eog_l10.yaml --turboquant --turboquant_bits 4 --save_dir runs/tcn_mha_eog_l10_tq4
# Evaluate a pre-existing full-precision checkpoint with quantization:
python eval.py --ckpt runs/tcn_mla_eog/best.pt --config configs/tcn_mla_eog.yaml --turboquant --data_dir data/EEGdenoisenet/data --mode snr
# New checkpoints restore architecture and TurboQuant settings automatically.
# --no-turboquant provides a full-precision comparison with identical weights.
```

Training now forwards supported model-constructor options (including heads,
dropout and sequence length) from YAML. New checkpoints contain `model_config`;
old checkpoints still load, with `--config` supplying non-default architecture.
Multi-run evaluation restores each checkpoint's settings and does not accept
single-model quantization overrides.

The codec exposes packed-byte `encode`/`decode` methods, but attention currently
decodes to floating-point K/V before its normal matrix operations. There is no
persistent streaming cache or fused low-bit attention kernel in this integration:
it is for quantization accuracy experiments, with no claimed runtime memory or
speed improvement. Norms (and QJL residual norms), padding and fixed rotation
buffers add overhead, especially at the small head dimensions used here.
Trainable parameter counts are unchanged. EEG accuracy must be measured; the
paper's LLM results do not establish denoising accuracy for these models.

Validation: `python -m unittest discover -s tests -p 'test_turboquant.py'`
checks packing, quantization error, QJL averaging, disabled compatibility,
10/20/50-layer causality, gradients, config wiring and checkpoint restoration.

`python -m unittest discover -s tests -p 'test_distillation.py'` covers the
bidirectional attention flag, teacher/student state-dict compatibility, the
distillation loss terms, teacher loading, the export refusal, a full
`KDTrainer` run, and the compressed-student path (sliced initialisation,
feature projectors, `student:teacher` layer pairs, clean checkpoints).
