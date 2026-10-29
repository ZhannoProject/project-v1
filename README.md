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
| `tcn_se` | Baseline + a causal Squeeze-and-Excitation gate on every block | Channel gate only, no temporal mixing | 123k |
| `tcn_se_la{N}` | `tcn_se` licensed to read N samples past the one it emits | Channel gate only | 123k |
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
│   ├── Dataloader.py           # Leak-free train/val/test splits + SNR mixing (seeded),
│   │                           #   and the non-stationary artifact-switch probe
│   ├── EEGIO.py                # the other kind of data: real recordings with
│   │                           #   hand-annotated blinks and no clean reference
│   └── EEGdenoisenet/data/     # .npy files (not in repo, see Data below)
├── models/
│   ├── TCN_backbone.py         # CausalConv1d → ResidualBlock → CausalTCNBackbone
│   ├── AttentionBlocks/
│   │   └── attention.py        # CausalMHA, CausalMLA, TransformerBlock/Decoder
│   ├── variants.py             # 5 model variants + registry + build_model()
│   ├── lookahead.py           # causality as a dial: budget allocation across the stack,
│   │                           #   and the probe that measures what a model really reaches
│   ├── receptive_field.py      # the mirror question: how far *back* the stack reaches,
│   │                           #   the dilation ladder, and the same probe pointed backwards
│   ├── incremental.py          # streaming inference: one sample at a time, with a real
│   │                           #   KV / latent cache (see Incremental inference below)
│   └── step_graph.py           # the same step as an exportable graph: state in, state out,
│                               #   so it runs on the Pi with onnxruntime and no torch
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
│   ├── server.py               # stdlib HTTP/JSON API, training / evaluation / deploy job runners
│   ├── pi.py                   # Deploy tab: step builder, log reader, latency + duty traces
│   ├── lookahead.py            # Lookahead tab: the ladder, measured per rung, joined to runs
│   ├── rfsweep.py              # Receptive field tab: one training run per rung of the ladder
│   └── static/index.html       # the page
├── deploy/
│   ├── preflight.py            # the state of the machine a timing run is taken on, judged
│   ├── bench_incremental.py    # per-sample cost and cache size: incremental vs windowed
│   ├── export_onnx.py          # checkpoint → verified ONNX (+ sidecar JSON) for Raspberry Pi 5
│   ├── run_on_pi.sh            # one command: copy to the Pi over SSH, set up, benchmark, score
│   ├── raspi_infer.py          # numpy + onnxruntime inference (epoch + causal streaming), --testdata scoring, per-hop latency monitor
│   ├── make_testdata.py        # cut a small raw-scale test set from the leak-free test split
│   ├── testdata/EOG_test100.npz#   100 EOG test epochs, 10 per SNR level, ready for the Pi
│   └── requirements-raspi.txt  # the only two packages the Pi needs
├── utils/utils.py              # get_device(), seed_everything()
├── eval.py                     # 8-in-1 evaluation: metrics, curves, waveforms,
│                               #   heatmaps, artifact-switch response
├── metrics.py                  # RRMSE_t, RRMSE_f, CC
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

The data loader guarantees **zero leakage** between splits: clean EEG indices AND artifact indices are disjoint across train/val/test (epoch-level splitting with a fixed master seed). That is asserted rather than assumed — the three index sets are checked disjoint and exhaustive for all three pools.

### What a BOTH sample's SNR means

`snr_db` is the SNR each artifact was **asked for**, and in BOTH mode it is the
mean of the two. It is not the SNR of the mixture: two artifacts each
calibrated to *s* dB against the same clean signal put in twice the noise
power, so the mixture lands at *s* − 10·log₁₀2, about 3 dB lower (measured:
−3.00 dB on the val split). Every sample now also carries
**`snr_mixture_db`**, the realised figure, measured off the signals
themselves — that is the one to label an axis with. `snr_db` is still what is
reported and what `deploy/make_testdata.py` writes into the Pi test sets,
because every per-SNR table on disk is keyed on it.

### Two sampling modes

`sampling="legacy"` (the default) is what every run in `runs/` was trained and
validated under. It has two quirks, both found by audit, both measured:

- **In BOTH mode, val and test give the two artifacts the same SNR.** The
  deterministic grid value depends on the sample index alone, and
  `__getitem__` asks for it twice, so `snr_eog == snr_emg` on 100 % of
  val/test samples — while training draws the two independently (they differ
  by 3.0 dB on average, up to 8.4). Val and test therefore cover only the
  slice of the training distribution where neither artifact dominates.
- **The clean pool is truncated to the artifact pool, including for
  training.** EOG and BOTH training use 2380 of 3159 clean epochs, although
  the training loop draws its artifact per sample at random and with
  replacement and so needs no 1:1 pairing.

`sampling="balanced"` fixes both — the second artifact is offset half a grid
away, the whole clean pool is used for training, and the grid is made to
divide val/test exactly (EMG val was 677 epochs over 10 levels, so the lowest
levels appeared once more than the highest).

The default is the older behaviour **on purpose**: these choices decide what
the validation set *is*, so flipping them would make every metric already in
`runs/` describe a dataset that no longer exists. `legacy` is verified
bit-identical to the pre-audit loader on all three artifact types and all
three splits. A run under `balanced` is not comparable with one under
`legacy`; the mode is recorded on the dataset.

```bash
python -m trainer --config configs/tcn_se_both.yaml      # legacy, as every run on disk
```

Neither mode changes the leak-free split, the seeding, or the artifact-switch
probe, whose draws are bit-identical either way.

## A second dataset: EEG-IO (real recordings, annotated blinks)

`data/EEGIO.py` reads [EEG-IO](https://gnan.ece.gatech.edu/eeg-eyeblinks), and
it is the opposite kind of dataset to everything above. EEGdenoiseNet is a
*synthetic mixture*: clean EEG plus an artifact at an SNR we choose, so the
target is known exactly and RRMSE is computable. EEG-IO is 20 subjects of
**real frontal EEG** (Fp1, Fp2; OpenBCI with a BIOPAC Cap100C; 250 Hz; one
session each of ~75–100 s, ~25 blinks per subject) with the blink instants
annotated by hand from a video feed — and **no clean reference at all**.

That decides what it is for:

- **It cannot train or score a denoiser the way EEGdenoiseNet does.** No
  target means no RRMSE_t, no RRMSE_f, no CC. `EEGIOWindows` returns no
  `clean` key, deliberately, so a scoring loop written against
  `EEGDenoiseDataset` fails loudly instead of quietly scoring against
  something invented.
- **It can say what a model does to a real blink.** Every window carries a
  mask of the annotated blink instants inside it, so a denoiser's output at
  blink times can be compared against the same recording away from them.
  That is a statement about artifact suppression that needs no target.
- **It can supply real non-stationarity.** The switch probe splices two
  synthetic regimes at a drawn index; here the onsets are real events with
  real timestamps, and the bracketing result above — a sample inside a blink
  is resolved by seeing both of its edges — makes a prediction about them.

```python
from data.EEGIO import EEGIOWindows, build_eegio_loader, split_subjects, list_subjects

ds = EEGIOWindows("data/EEG-IO", win=512, hop=256)      # 512 @ 256 Hz, Fp1
item = ds[0]            # noisy (1,512), sigma, blink_mask (512,), n_blinks, subject
loader = build_eegio_loader("data/EEG-IO", split="test", batch_size=32)
```

```bash
python data/EEGIO.py --data_dir data/EEG-IO      # per-subject seconds, blinks, corrupt time
```

Four things worth knowing before using it:

- **250 Hz, not 256.** Every model here was trained at 256 Hz, so a raw
  EEG-IO window is the wrong rate for a checkpoint. `target_fs` defaults to
  256 and resamples with `scipy.signal.resample_poly` at the exact rational
  ratio 128/125; `target_fs=None` keeps the native rate for analysis that
  involves no checkpoint.
- **Splits are by subject, not by window.** Two windows of one recording
  share a session, an electrode placement and a person, so splitting windows
  would leak all three. `split_subjects` is the same guarantee
  `Dataloader._split_indices` gives at the epoch level, one level coarser,
  because here the unit of independence is coarser.
- **Corrupt intervals are honoured.** The annotators marked stretches as
  unusable; windows overlapping one are dropped by default and the count is
  reported rather than silently absorbed.
- **Soft blinks are a choice, not a default.** The dataset's label code 2 is
  a soft blink, which the reference implementation gates behind its own
  `flag_soft`. `include_soft` does the same here, and is off by default.

The format is taken from the reference implementation released with the
dataset ([`meagmohit/BLINK`](https://github.com/meagmohit/BLINK),
`blink.py`), not guessed: `<subject>_data.csv` is `;`-delimited with one
header row and columns `(time_s, ch1, ch2)`; `<subject>_labels.csv` is
`,`-delimited, where a `corrupt,<n>` row announces `<n>` interval rows (an
end of `-1` meaning "to the end of the recording"), a bare `blinks` line
separates the two blocks, and every other row is `<time_s>,<code>`. The real
files settle the channel order themselves — their header reads
`Time (s);FP1;FP2;Channel 3;…`, so column 1 is Fp1 and column 2 is Fp2, and
the later columns are unused electrodes parked at a constant. One thing the
sources still do not settle is flagged in the code: what codes 0 and 1 mean,
since the reference only ever separates `< 2` from `== 2`.

Read against the real download, the loader finds **20 subjects, 102.3 s each,
517 blinks and 41 soft blinks** — about 25 per subject, as the dataset page
describes.

**The data is not in this repository and is not redistributable** — it comes
through a registration form on the dataset page, which the loader prints when
the directory is missing. The download unpacks as `EEG-EyeBlinks/` holding
`EEG-IO` beside `EEG-VV` and `EEG-VR`, so both `data/EEG-IO/` and
`data/EEG-EyeBlinks/EEG-IO/` are found automatically; the EEG-IO test tab has
a **Refresh** button for the case where it is unpacked while the page is
open. Only EEG-IO is read — `EEG-VV` and `EEG-VR` are a different file format
(comma-delimited, five header rows). `write_synthetic_recording()` writes a fake subject
in the same on-disk format so the pipeline can be exercised without it; it is
a fixture for tests and demos, never a result.

### Testing a model on it: the EEG-IO test tab

`eegio_eval.py` and the GUI's **EEG-IO test** tab run the project's models on
the real recordings. With no clean signal there is no RRMSE, so the
measurement is a contrast instead: the annotated blink intervals against the
background between them. Everything is relative to **the preprocessed signal
that went into the model**, never the raw recording.

**Preprocessing is causal, and that is not free.** EEGdenoiseNet's own
pipeline bandpasses 1–80 Hz and notches 60 Hz — read off
`data/EEGdenoisenet/code/preprocessing/EEG/IC_label_artifact_removal.m`,
not guessed — with a zero-phase filter, which is two passes, one of them
backwards. No online system can run that. So `eegio_eval.preprocess` uses
`scipy.signal.sosfilt`, one pass forwards: the band matches training and the
phase does not. The same perturb-and-look-back probe the models get is run on
the filter itself, and reports `past_diff = 0.0`. The zero-phase filter stays
available as a control that prices the difference.

Four numbers, and the first two only mean anything together:

| | what it is |
|---|---|
| **suppression S** (dB) | blink-interval power the model removed; higher is more |
| **fidelity D_b** (dB) | background power over what the model changed there; higher is more faithful |
| **residual detectability** | annotated blinks the detector still finds in the output; lower is better |
| **onset lead** (ms) | how far before the signal moves the model's correction starts |

A model that simply attenuated everything would score well on suppression
alone, which is why fidelity sits beside it — there is a test asserting
exactly that pairing still catches it.

**Blink intervals** come from a detector (smooth, then threshold on a robust
background spread) filtered down to **only the intervals that also contain a
human annotation**. That is what makes the detector's own errors irrelevant: a
false positive has no annotation in it and is dropped, and a missed blink
contributes nothing rather than a wrong interval. Corrupt stretches are
excluded from both.

**Onset lead** is the one this project cares about, and it took two attempts
to define. Measuring where `|y − x|` first crosses a multiple of its
background spread gives a *strictly causal* model tens of samples of apparent
lookahead — because outside the blinks a good model's correction is nearly
zero, so its background spread is tiny and it crosses immediately, while the
signal's spread is the ambient EEG and it crosses late. The asymmetry is in
the thresholds, not the timing. Both onsets are now defined as the point
where each quantity reaches 10 % of **its own** peak over that blink, which
is scale-free; a test plants a correction shifted by a known 8, 16 or 32
samples and the metric recovers it to within 2.

**Risk check**: whether real blinks are as strong as the training mixtures.
With no clean signal the estimate treats the background between blinks as the
brain signal and the excess power inside a blink as the artifact, so it
assumes the two add incoherently. It is a range check against the
`[-7, 2]` dB the models were trained over, not a measurement.

**Controls and the test.** The zero-phase filter, the bidirectional twin
(legitimate here, where it runs offline, and refused on the Deploy tab), and
the INT8 graph are all selectable. Models are compared **paired by subject**
with a Wilcoxon signed-rank test, because subjects differ far more than
models do — electrode placement, how hard someone blinks — and that spread
would swamp the comparison if the groups were pooled.

```bash
python eegio_eval.py --data_dir data/EEG-IO \
       --runs runs/tcn_se_both runs/tcn_se_la16_both runs/tcn_se_bi_both
```

The first run is the baseline the rest are paired against. The tab suggests
exactly that triple: the causal model, the rung licensed 62.5 ms of
lookahead, and the unbounded twin.

Not yet done: the 250 Hz retraining. Every model here was trained at 256 Hz
and is run on EEG-IO by resampling 250 → 256, which is exact in rate
(128/125) but is not the same as training at the recording's own rate.

Validation: `python -m unittest discover -s tests -p 'test_eegio_eval.py'`
covers the filter's causality and the zero-phase control's lack of it, the
band and notch, detection and annotation filtering, each metric against a
model whose answer is known in advance (identity, attenuator, a correction
shifted by a known amount), the risk check, the streaming wrapper, the
pairing, and that the JSON the tab returns has no bare `NaN` in it —
`json.dumps` writes those unquoted and `JSON.parse` rejects the whole
document.

Validation: `python -m unittest discover -s tests -p 'test_eegio.py'` covers
the reader against files in the dataset's own format, the `corrupt,<n>`
convention including the `-1` end, the subject-level split, the 250 → 256 Hz
conversion, the blink mask and soft-blink gating, window counts, the absent
`clean` key, the missing-data message, and that a project model runs on a
window unchanged.

Validation: `python -m unittest discover -s tests -p 'test_dataloader.py'`
covers the split guarantee, both modes, the reported-versus-realised SNR, the
per-worker augmentation streams, the probe's truncation, and the load-time
checks (shape, epoch length, degenerate epochs, and that `np.load` no longer
unpickles).

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

Checkpoints are saved to `runs/<variant>_<artifact>[_s<seed>]/` with `best.pt`, `last.pt`, and `history.json` — see [One seed, one directory](#one-seed-one-directory) for the seed suffix and the overwrite guard.

### Seeding

`master_seed` (default 42, `--master_seed N` on the CLI) is the single seed for a run: the train/val/test split, artifact pairing and SNR draws (numpy), the training batch order (a dedicated `torch.Generator`, so it is identical across variants), and weight initialisation and dropout (`seed_everything` in `utils/utils.py`, which also switches cuDNN / `torch.use_deterministic_algorithms` to deterministic kernels).

```bash
python -m trainer --config configs/tcn_mla_eog.yaml --master_seed 1   # → runs/tcn_mla_eog_s1
python -m trainer --config configs/tcn_mla_eog.yaml --device cpu      # bit-exact repeats
```

#### One seed, one directory

The seed is part of a run's identity, so it is part of its path. The default
`save_dir` is `runs/<variant>_<artifact>[_kd][_s<seed>]`, and the `_s<seed>`
suffix appears for every seed but the default 42 — so seed 42 keeps the bare
name every existing directory already has, and seeds 55 and 67 of the same
model get `runs/tcn_both_s55` and `runs/tcn_both_s67` instead of fighting over
`runs/tcn_both`. That is what makes a set of per-seed repeats an error bar
rather than a single surviving run.

**Training refuses to start on top of another run.** A directory that already
holds `best.pt` / `last.pt` / `history.json` is reused only when the run in it
recorded the *same* seed — the ordinary "train it again" loop, which still just
warns. Anything else stops before the dataset is even loaded:

```
[Trainer] refusing to start:
  runs/tcn_both already holds a run at master_seed=55 (best.pt, last.pt, history.json), and this run uses 67.
  Starting here would overwrite it, and the weights are not recoverable.
```

A run whose seed was never recorded (anything trained before the trainer wrote
its config down) counts as another run too: it cannot be shown to be the same
one, and the weights underneath it are no less real. `--overwrite` replaces
whatever is there on purpose, and is the only way past the check:

```bash
python -m trainer --config configs/tcn_eog.yaml --master_seed 55            # its own folder
python -m trainer --config configs/tcn_eog.yaml --save_dir runs/scratch --overwrite
```

This is not a hypothetical. The project's own `tcn_both` and `tcn_mla_both`
directories were overwritten twice — seed 42 replaced by seed 55 in September,
and seed 55 by seed 67 on 13 October — because the path carried no seed and
`save_checkpoint` writes in place. The per-epoch metrics survived in
`runs/gui_train_*.log`, which record the config and every validation line; the
weights did not. Validation: `python -m unittest discover -s tests -p 'test_run_isolation.py'`.

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

The page has ten tabs: **YAML** (the text a save would write), **Train**,
**Evaluate**, **Deploy**, **Lookahead**, **Receptive field**, **EEG-IO
test**, **KD audit**, **Params** and **Runs**. The Evaluate and Deploy tabs borrow the Runs table as a run picker,
so all three share one renderer.

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
  RRMSE_f and CC per epoch, with the current values, the best epoch so far and
  the epoch counter. The numbers are parsed out of the training log as it
  streams, so nothing in the trainer changes and the curve survives a page
  reload (the server keeps them for the running job). The y-axis is fitted to
  the bulk of the run — epoch 1 is often several times worse than the rest and
  would otherwise flatten everything; the caption says when early epochs run
  off the top.
- **Runs** has six views. *Complete by kind* groups finished runs by
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

  Every row also shows the **seed** the run used, which is what makes a set of
  rows an error bar rather than a list: runs of the same `variant × artifact`
  that differ only in `master_seed` are independent repeats, and *Complete by
  kind* already groups them together. Note that this project has one seed, not
  several — `master_seed` drives the data splits, the SNR pairing, weight init,
  dropout and batch order together, so varying it varies all of them at once and
  the spread you measure is total run-to-run variance. Runs from before the
  trainer recorded its config show `—`; their seed is genuinely unknown and
  filling in the default would put invented points in an error bar.

  Every row carries its **parameter count** — the same figure the Params tab
  calls the checkpoint count, read from the weights rather than recomputed, with
  the exact number on hover. The grouped views still print it in the group
  header, but only when every run in that group agrees; two runs of the same
  `variant × artifact` trained at different widths no longer both inherit the
  first one's count.

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

  **How far ahead a run reads** is a badge on every row, because it decides
  what the run can be used for and nothing else on the row says it:
  - no badge — causal. The deployable case.
  - `bi` (solid) — non-causal with no bounded horizon: its attention or its
    pooling reads the whole 512-sample epoch. A teacher.
  - `la{N}` `+{t} ms` (dashed) — a bounded-lookahead rung: it may read N
    samples past the one it emits, which is N/256 s of output delay. Not
    causal either, but the horizon is finite and stated, and the badge gives
    the latency it costs.

  The dashed/solid pair is the same distinction the loss badges draw: dashed
  is the qualified answer. Keeping the two apart is not cosmetic. A rung and a
  teacher both record `causal: false`, so reading that flag alone — which this
  tab did until the ladder existed — badges every rung as a non-deployable
  teacher, and `runs/tcn_se_la32_both` would claim to be something it is not.
  The Deploy tab refuses both, and now says which refusal it is: a teacher
  needs the whole epoch, a rung needs an output delay that no path here
  implements yet.

  **curve** on any finished run plots its saved history in the same chart, and
  “Use as teacher” fills `teacher_ckpt` for a distillation run.

#### Several runs at once

The Train tab holds more than one run. Each gets its own log, its own live
curve and its own read offset, and a row of pills above the chart switches
between them — a dot per run, green when it finished, red on a non-zero exit,
pulsing while it is going. One run shows no pills: there is nothing to choose.

This is not a convenience feature bolted on. The ordinary task in this project
is *n* runs that differ in one flag — three seeds of one config, or one run per
rung of the lookahead ladder — and the server used to refuse the second with a
409, which made that a sequence of sittings with a person waiting between them.

What it does not do is make them free. They share one machine, and on MPS one
accelerator, so concurrent runs contend: **the per-epoch wall times in each log
stop being comparable** with a run that had the box to itself, and three at
once on a laptop finish later than three in a row. The metrics are unaffected —
each run has its own seed, data, weights and output directory — which is why
this is allowed at all, and the tab says so in a line above the chart whenever
more than one is going rather than leaving it to be discovered. For anything
where the *timing* is the measurement, use the Deploy tab, which holds the
machine still on purpose.

Four at once is the cap, so a mistake cannot spawn fifty; the fifth is refused
with a 409 that names the limit. **Stop** means the run whose log you are
reading, not "the training job". A finished run stays in the list so its log
and curve can still be read, and the oldest are dropped once sixteen have
accumulated. Runs still going are never dropped.

The server binds to 127.0.0.1 only.
Validation: `python -m unittest discover -s tests -p 'test_gui.py'`.

#### Evaluate tab

The other half of the GUI: it runs `eval.py` on a finished run and shows what
comes back. Every `--mode` the script has is offered — per-SNR metrics, training
curves, waveforms, attention heatmaps, the artifact-switch probe, and the
multi-model comparisons — and the mode decides which controls are shown, so the
switch options appear for a switch run and nothing else.


  *Deploy runs* is the other half of what this project did. The five views
  above list what was trained; this one lists what was **measured** — one row
  per `(machine, model, hop)`, newest first, with the median hop time against
  its deadline (`hop / 256 Hz`), the duty cycle, the deadline misses, and the
  environment verdict the trace recorded. A single deploy run can write three
  traces — the per-hop latency, the duty-cycle sweep beside it and the
  real-time replay — and they are gathered into one row, because they are one
  run's output rather than three runs.

  The **state** column is the point: a latency figure is a statement about a
  model *and* about the machine it ran on, so a row says which of the two was
  held still. `locked` means every control held; a trace from before the
  environment was recorded reads `unknown`, which is deliberately not the same
  as `unlocked`. Runs on this machine are labelled *this machine* rather than
  hidden — macOS exposes neither a governor nor an affinity API, so a host run
  is a convenience and the board's run is the measurement. **Open** sends the
  row's primary trace to the Deploy tab's viewer.

  The row also carries what the model **scored** on its test set —
  **RRMSE_t**, **RRMSE_f** and **CC**, epoch mode — because a latency without
  an accuracy beside it says nothing about whether the speed was worth having.
  Hovering a value gives the streaming figure (a pessimistic bound: the probe
  concatenates independent epochs, so every window straddles two unrelated
  signals), the test set it was scored on, and where the number came from.

  Those scores used to be printed and dropped: `evaluate_testdata` computed
  them, returned them, and `main` ignored the return, so the trace saved the
  milliseconds and lost the thing they were spent on. The trace now records
  them beside the timings. Measurements taken before that are **recovered
  from the job log that produced them**, which also names the traces it wrote
  — that is the join — and are shown with a dashed underline, the same
  solid/dashed distinction the loss badges draw. A row whose numbers are in
  neither place shows `—` rather than a guess.

  On the Pi at hop 32 the table reads both costs straight off: the TCN
  baseline at 3.2 ms against a 125 ms deadline for RRMSE_t 0.713, the MLA
  sandwich at 15.0 ms for 0.705 — roughly 5x the compute for a 1% difference,
  on one seed.
- **The run is picked from the Runs table, not a dropdown.** Clicking the
  *run* field opens the Runs tab's own table in a dialog — the same groupings
  (by kind, loss, optimizer, time, or all), the same params, seed, epochs, val
  loss, RRMSE_t / RRMSE_f / CC and the same `bi` / `la{N}` / `stale` /
  `partial` badges —
  and clicking a row picks it. Both are rendered by one function, so the picker
  cannot drift from the tab. The two actions that would navigate away mid-choice
  (`curve`, "Use as teacher") are dropped in the picker.

  Nothing is selected until you select it. The plain dropdown this replaced
  defaulted to whatever sorted first alphabetically — `ctrl_mla_both_bi`, a
  *non-causal control* — so evaluating the wrong model took one click and
  nothing said so. The field now reads "Choose a run…" and the command preview
  says `mode 'X' needs a run` until one is picked. The choice is remembered
  between sessions.
- **Separate jobs, not one.** Training, evaluation and deployment are
  independent; starting an evaluation neither blocks a training run nor
  clobbers its log. Training is the one that can have several going at once
  (see [Several runs at once](#several-runs-at-once)); evaluation and
  deployment stay one at a time.
- **The command is the preview.** The grey line under the form is not built in
  the page — it is the exact argv the server would run, fetched on every
  keystroke. So is every refusal: pick one run for a comparison mode, or a run
  with no `history.json` for `curves`, and the reason appears there before
  anything starts.
- **Figures appear inline.** `eval.py` prints what it saves; the tab picks those
  paths out of the log and renders them, with an `open` link for a full-size
  view and the JSON listed beside the PNG.
- **Output is pinned to `figures/`.** Each run writes to
  `figures/gui_eval_<timestamp>/`, which is also the only directory the server
  will serve a file from. The log is mirrored to `runs/gui_eval_<timestamp>.log`.

**Two seeds, and the tab shows both.** `master_seed` is the run's data seed —
it picks the split, and it has to match what the checkpoint was trained with.
`probe_seeds` is the switch probe's own series, redrawn on top of that split.
The Model form's `master_seed` field is the *training* one; they are the same
number for the same run, which is why the tab asks for it separately rather
than guessing.

#### Deploy tab

The third half of the GUI: the runtime test. It is a front end for
`deploy/export_onnx.py`, `deploy/raspi_infer.py` and `deploy/run_on_pi.sh`, and
it answers the question the Evaluate tab cannot — not *how well* does this
model denoise, but *what does it cost to run, and is that inside a real-time
budget*.

- **Two targets.** *This machine* runs `raspi_infer.py` here (the same script
  the board runs, with onnxruntime and nothing else), and *Raspberry Pi over
  SSH* hands the whole thing to `run_on_pi.sh`: copy six files, make the venv
  on first use, measure there, copy the traces back. The host target is not a
  substitute for the board — an Apple-silicon host is ~6× faster per window,
  and it cannot be locked at all (see below) — but it is how you see the
  measurement work without unplugging anything, and the two are directly
  comparable because it is one script.
- **The conditions come first, and they stay attached.** **Check state** reads
  the machine without measuring anything: CPU governor, affinity, thread count,
  throttle flag and load, each with a verdict and the reason. The same check
  then runs before every measurement, and its result is written into the trace,
  so the panel above the charts always describes *the numbers on screen* rather
  than the machine as it happens to be now. Every chart title carries the
  verdict too — **locked**, **partly locked**, **not locked**, or *conditions
  not recorded* for a trace from before this existed, which is a different
  thing and is shown as one. The panel ends with the sentence to paste into a
  methods section. The controls are *pin to CPUs*, *governor* (Pi only, needs
  passwordless sudo there) and *only measure a locked machine*, which stops the
  run before it starts rather than producing a figure that cannot be
  reproduced. On a host with no governor and no affinity API the pin field says
  so instead of offering a setting that would do nothing. See
  [the state of the machine](#0-the-state-of-the-machine-and-why-it-is-held-still)
  for what each control is for.
- **A run is a sequence, not a command.** Up to three steps, each its own
  process, shown with a tick as it passes: export the checkpoint, benchmark +
  score the test set + measure the per-hop latency, sweep the hop sizes. The
  measurement and the sweep are separate invocations because `raspi_infer.py`
  returns straight after `--duty_cycle` and one call could not do both. A step
  that fails stops the ones after it, and the log is one log.
- **The model is a trained run, not a file.** The field offers every run in
  `runs/`, picked from the Runs table — the same picker the Evaluate tab uses,
  with the same val loss, RRMSE and CC in front of you, because choosing what
  to deploy is choosing among them. It used to list `deploy/artifacts/*.onnx`
  instead: two files against twenty-nine runs, so twenty-seven models could not
  be chosen at all. Each row says what picking it would mean — `onnx` for one
  that is ready, `export first` for a run that has never been exported, and
  `no export` for a `*_bi` twin, which needs the future and which the exporter
  refuses (said here rather than two minutes into the export). Picking a run
  with no export ticks *export first* for you, and the command preview then
  shows the extra step; an export older than the run it is named after says so
  and ticks it too. So train → export → measure is three clicks, and the
  artifact is named after its run — one name for the thing at both stages.
- **The panels say where their numbers came from.** While a run is going the
  only live source is its output, so the figures are parsed out of the log as
  it streams (the pill counts hops). The moment the run writes a trace the page
  loads *that* instead — `deploy/artifacts/pi_latency/<stem>.json` has every
  hop in it at full precision, and it is still there next week. The caption
  under each chart names the file, and the title names the machine: host traces
  are written as `host_<model>_hop<N>` precisely so a measurement of a laptop
  can never be read as one of the board.
- **Per-hop latency**, from the trace's CSV: the median per hop and, as a
  fainter line, the worst hop in each bucket — a 40k-hop run is bucketed down
  to something drawable, and plain decimation would drop exactly the stalls the
  curve is read for. onnxruntime's own time, the end-to-end latency of a paced
  run and the SoC temperature are drawn alongside when the trace has them. The
  hop budget is drawn as a line when it fits on the axis and named in the
  caption when it is far off the top, which on a Pi 5 at hop 32 it is.
- **Duty cycle**, from `<stem>_duty.json`: one bar per hop size, the mean with
  the p99 as a cap, coloured by the verdict, against the 70 % (marginal) and
  100 % (cannot keep up) lines, with the full table below and the conclusion
  spelled out — *real-time processing is applicable down to hop N: X ms
  buffering + Y ms compute = Z ms total latency*.
- **The test set, scored on the device**, in epoch and streaming mode next to
  the untouched noisy input, with the per-SNR table. These are numpy replicas
  of `metrics.py`, so the row the Pi prints and the row the host prints are the
  same measurement.
- **Old measurements are first-class.** The trace dropdown lists every saved
  trace, newest first, labelled by machine, kind and hop; opening the tab with
  nothing running shows the most recent one of each kind rather than a blank
  chart. The log is mirrored to `runs/gui_deploy_<timestamp>.log`.
- **Stop stops the whole thing.** A deploy step is not always one process —
  `run_on_pi.sh` forks `ssh` and `scp` — and those inherit the job's output
  pipe, which does not close while any of them holds it. So each step runs in
  a session of its own and **Stop** signals the session, not just the command
  it started: SIGTERM, five seconds, then SIGKILL to whatever is left. Stopping
  a Pi run closes the connection, which is what ends the remote command. Its
  stdin is `/dev/null`, so the Pi target needs key-based ssh — with no terminal
  to answer it, `run_on_pi.sh` asks ssh to fail (`BatchMode`) rather than hang
  on a prompt nobody can see. See [connecting to the Pi](#connecting-to-the-pi)
  for what each failure means.

The Pi's address is validated as `[user@]host` before it reaches `ssh`, and the
run is started with `subprocess` rather than a shell, so nothing in the form
reaches a command line. Training, evaluation and deployment are three
independent jobs: a sweep that takes two minutes does not block a training run.

Validation: `python -m unittest discover -s tests -p 'test_pi_tab.py'` covers
the command builders and their refusals, the log reader against the output of a
real run (and against what `LatencyMonitor.print_summary` prints, so the parser
cannot drift from the printer), the trace files and their sandbox, the
step sequence (order, a failing step, a stop between steps), the routes and the
page's own tables.

#### Lookahead tab

The causality cost, priced by the millisecond instead of quoted as a single
number.

This project has measured that cost exactly twice: `tcn_se` at RRMSE_t
0.7104 ± 0.0020 (3 seeds) and its unbounded twin `tcn_se_bi` at
0.5950 ± 0.0020. The 0.1154 between them is real, and as a design input it is
useless — it prices *all* the future against *none* of it, and no online
system is offered that choice. It is offered a latency budget. So the tab
turns the two points into a curve, on the axis the budget is denominated in:

```
L samples of lookahead  =  L / fs of output latency  =  3.9 ms per sample at 256 Hz
```

A rung is `tcn_se_la{N}` — the same model, licensed to read N samples past the
one it emits. The ladder is 1, 2, 4, 8, 16, 32, 62, 124, 186 samples (3.9 ms
to 727 ms), anchored by `tcn_se` at 0 and `tcn_se_bi` at unbounded. 62 is
worth naming: it is the forward reach of `tcn_bi`, so a model at 62 sees
exactly as far ahead as this project's existing non-causal baseline while
keeping a horizon that can be stated.

**The x axis is available before any rung is trained.** The reach is a
property of the graph, so the tab measures the whole ladder in under a second
on freshly built models and says which rungs have a run behind them. The
accuracy half is a 200-epoch run per rung; the tab builds and launches it
through the ordinary training runner, from `configs/tcn_se_both.yaml` — the SE
recipe the three seed repeats used — overriding only the variant, the seed and
the output directory.

**A rung is trained at every seed at once.** The seeds are tick boxes, not a
one-of-three menu, because a rung trained once is a point whose spread is
unknown — the chart draws it hollow to say so — and the useful unit of work on
this ladder is therefore a rung at three seeds, not a rung. Asking for them
one at a time made that three sittings with someone waiting between each,
which is exactly what the Train tab stopped requiring when it learned to hold
several runs. A seed that already has a run opens unticked and marked ✓, with
the directory on hover, so adding the two that are missing is one click; it is
unticked rather than disabled, because with **overwrite** — a box in the same
row — redoing it is a legitimate thing to ask for. That flag is put only on
the directories that have something in them: a blanket `--overwrite` would
also switch off the trainer's guard for the seeds that did not need it, which
is where a mistyped seed quietly lands on top of a finished run. The request
is all-or-none: if a ticked seed is taken and overwrite is off, or there is
not room under the four-run cap, nothing starts and the refusal names what is
in the way — starting two of three and refusing the third would read as a
failure of the two that ran.

**Two numbers per rung, and they are not the same number.**

| Column | What it is |
|---|---|
| `licensed` | What the graph may read. This is what sets the latency, whether or not the weights use it. Exact, and a property of the architecture |
| `measured` | How far the weights actually reach, by perturbing one input sample and finding how far back the output moves — the probe that measures 62 on `tcn_bi`, the number the switch probe independently recovers as 242 ms |

For the convolution stack they agree exactly: nothing on that path can zero a
dependence, since PReLU has a slope either side and BatchNorm is affine in
eval mode. For the SE gates they do not. The gate is
`sigmoid(W2 @ relu(W1 @ s))`, and **ReLU can be exactly dead** — where
`W1 @ s` is negative the gate stops depending on `s` at all, so a gate
licensed to peek L ahead contributes nothing there. A fresh 186-sample model
measures about 150. Neither number is the honest one on its own: you pay for
`licensed` in latency, and `measured` tells you how much of that licence the
weights bother to use, which is a result in itself.

A budget is spent across the stack proportionally to what each site can take
(two convolutions and one SE gate per block), so the sites' reaches add up to
the budget exactly. Capacity is 186 samples — 124 in the convolutions, 62 in
the gates — and a request above it is clamped, and says so.

**Capacity is not a target.** Spending all 124 convolution taps forward means
spending none backward: `pad_left + pad_right` is fixed per layer, so the
budget slides the window rather than widening it, and the full budget leaves
the model with no past at all. The `past` column in the tab is that number —
the taps still behind *t* — and a rung that reaches 0 is marked ✕ and left
off the price curve. See *What a millisecond is worth* below for what it does
to the measured curve.

**What the rung number is a number of.** `tcn_se_la186` reads as 186 of
something, and the obvious something — the convolution window — is 124 taps,
so the name invites an arithmetic that does not work out. The denominator is
the *licence*: 124 taps the convolutions can move plus 62 the SE gates can
peek, which the gates have and the window does not. Against that the ladder's
top three rungs are exactly the thirds — 62, 124 and 186 are 1/3, 2/3 and all
of it — which is why they are the rungs. The tab prints the share beside the
name in both places it is read: the rung picker
(`124 — tcn_se_la124 · 2/3 ahead`) and the rung column
(`tcn_se_la124  2/3`), with the window itself in its own column (`41 / 83`,
behind and ahead, always summing to 124). The top rung is written `1/1`
rather than "all" so that it reads as the last of the thirds. `tcn_se_bi` is
marked **centred** beside "unbounded" for the same reason: unbounded is its
SE gate, while its convolutions see 62 either side — the same 125-sample
window as every rung, in the middle.

The names themselves are left alone, for two reasons: the number *is* the
output latency in samples, which is the contract a deployment carries and the
axis the ladder is priced on; and 27 run directories and the architecture
recorded inside each of them already name the variant. A share would read
better and say less.

**A rung's name states its budget**, the way `tcn_mla_r{N}` states its rank:
there is no `--lookahead` flag, because a run called anything else could hold
any budget at all. An off-ladder budget is measurable in the tab (nothing is
saved, so no name can misdescribe it) and refused for training until it is
added to `LOOKAHEAD_LADDER` in `models/lookahead.py`.

#### What a millisecond is worth

Three seeds (42, 54, 67) per rung, 200 epochs, `configs/tcn_se_both.yaml`,
everything but `master_seed` identical. Δ is paired within each seed — the
seed fixes the split, so pairing removes split variance and leaves the thing
being measured:

The `window` column is the convolution stack's taps behind and ahead of the
sample it emits. It is in the table because the two numbers always sum to 124
— and that is the whole story of the top rung:

| rung | latency | window (back/ahead) | seeds | RRMSE_t | CC | Δ from causal, paired | of the gap |
|---|---|---|---|---|---|---|---|
| `tcn_se` | 0 ms | 124 / 0 | 3 | 0.7104 ± 0.0020 | 0.7074 ± 0.0029 | — | — |
| `tcn_se_la1` | 3.9 ms | 123 / 1 | 3 | 0.6885 ± 0.0017 | 0.7293 ± 0.0026 | 0.0219 ± 0.0003 | **19%** |
| `tcn_se_la2` | 7.8 ms | 122 / 2 | 3 | 0.6806 ± 0.0018 | 0.7364 ± 0.0026 | 0.0298 ± 0.0002 | 26% |
| `tcn_se_la4` | 15.6 ms | 121 / 3 | 3 | 0.6763 ± 0.0016 | 0.7401 ± 0.0023 | 0.0341 ± 0.0004 | 30% |
| `tcn_se_la8` | 31.2 ms | 118 / 6 | 3 | 0.6552 ± 0.0017 | 0.7572 ± 0.0024 | 0.0552 ± 0.0004 | 48% |
| `tcn_se_la16` | 62.5 ms | 113 / 11 | 3 | 0.6268 ± 0.0006 | 0.7796 ± 0.0009 | 0.0836 ± 0.0016 | **72%** |
| `tcn_se_la32` | 125 ms | 102 / 22 | 3 | 0.6131 ± 0.0016 | 0.7897 ± 0.0012 | 0.0973 ± 0.0026 | 84% |
| `tcn_se_la62` | 242 ms | 82 / 42 | 3 | 0.6046 ± 0.0012 | 0.7952 ± 0.0010 | 0.1058 ± 0.0015 | 92% |
| `tcn_se_la124` | 484 ms | 41 / 83 | 3 | 0.6025 ± 0.0015 | 0.7964 ± 0.0008 | 0.1079 ± 0.0025 | **94%** |
| `tcn_se_la186` | 727 ms | **0 / 124** | 3 | 0.6673 ± 0.0015 | 0.7418 ± 0.0015 | 0.0431 ± 0.0005 | 37% |
| `tcn_se_bi` | unbounded | 62 / 62 | 3 | 0.5950 ± 0.0020 | 0.7996 ± 0.0012 | 0.1154 ± 0.0028 | 100% |

**One sample of lookahead — a single sample period, 3.9 ms — recovers a fifth
of the causality cost.** The step is 0.0219 against a paired spread of 0.0003,
and it is in the same direction on 3 seeds of 3. Sixteen samples recover
nearly three quarters, and 124 samples recover 94%.

**Then it turns around.** The top rung, at 727 ms of latency, recovers 37% —
worse than 16 samples at 62 ms. That is not noise (±0.0005 paired, three
seeds of three) and not a broken run. A convolution of kernel *k* and dilation
*d* has `(k-1)·d` taps to place and puts each one on one side of *t* or the
other: `pad_left + pad_right` is fixed. **Lookahead does not widen the window,
it slides it.** Every tap handed to the future is taken from the past, and at
the full budget every `pad_left` is 0 — `tcn_se_la186` has no past at all. It
is the causal baseline's *mirror*, not a better-informed version of it, and
the table shows the two scoring alike (0.7104 one-sided backwards, 0.6673
one-sided forwards; the mirror does slightly better because the SE gate's
running mean still integrates the past, so it keeps the past's statistics
without any of its detail).

**What that rung is, exactly.** A causal layer computes
`w0·x[t-2d] + w1·x[t-d] + w2·x[t]`; move all the padding to the right and it
computes `w0·x[t] + w1·x[t+d] + w2·x[t+2d]`. That is the same filter applied to
the *reversed* signal with its kernel read backwards — and the kernel is
learned, so reading it backwards costs nothing. The tests check the identity
to float precision: an anti-causal TCN **is** this model class run on the
recording backwards. Not a degenerate model, just one that needs the whole
future, which rules out online use for the obvious reason rather than a
subtle one. (With SE it holds only approximately: the gate's squeeze is a
running mean from sample 0, so it keeps accumulating forwards however the
convolutions are padded. That is the one part that does not reverse, and it
is why the mirror rung is not simply the baseline's score reflected — it has
the past's statistics and the future's detail at once.)

So its score is a measurement about the **signal**, not about the model: it
asks whether this denoising problem is easier backwards than forwards. The
answer is that it is, by 0.0431 ± 0.0005 paired — small beside the 0.1154
gap, and far outside the seed spread. Whether that is the SE asymmetry above
or a genuine time-asymmetry in EOG/EMG envelopes (sharp onset, slow decay)
is separable by one experiment: a plain `tcn` at lookahead 124 has no SE, so
it is an *exact* time-reverse, and if it still beats causal `tcn` the
asymmetry is in the data. That variant is not registered yet.

Read down the `window` column and the result is simpler than the latency
axis suggests: **error tracks how far the window sits from centred.** The
ceiling makes the point — `tcn_se_bi` is called unbounded because its SE gate
pools globally, but its convolutions see 62 either side, the same 125 samples
as every rung, centred. It is the best model here because it is the balanced
one. So what the ladder prices is not "how much future" but "how much of the
window may sit ahead of now", and the answer is: up to about half, after which
you are paying latency to throw information away.

Put the three one-sided and two-sided results side by side and the headline
changes: 124 taps of past scores 0.7104, 124 taps of future scores 0.6673, and
**62 of each scores 0.5950**. Half as much context on each side beats twice as
much on one. What lookahead buys is not quantity, it is *bracketing* — and for
artifact removal that is what one would expect, since a sample in the middle
of a blink is resolved by seeing both of its edges, not by seeing further in
one direction. "One sample of lookahead recovers a fifth of the causality
cost" is better read as one sample of bracketing. The tab marks
`tcn_se_la186` ✕ and leaves it off the price curve for that reason — it is a
real measurement and a useless purchase.

The rungs are compared **paired within each seed** and against a baseline that
is the same model trained the same way. Both matter, and the second is not
pedantry — it has gone wrong twice:

* the first version of this table built its baseline path from the naming
  convention (`tcn_se_both` + `_s67`) and picked up `runs/tcn_se_both_s67`,
  which is the 48-channel run, not the 64-channel one;
* later `runs/tcn_se_both_kd_s67` — a *distilled* student, its own row in the
  ablation table — landed on the causal rung and was counted as a seed repeat,
  moving the anchor from 0.7104 to 0.7118 and widening its spread from 0.0020
  to 0.0033. Since every Δ in the table is measured from that anchor, the
  whole column was priced against a point that was part KD.

`gui/lookahead.py:why_excluded` now answers this once, for both this tab and
the receptive-field sweep, and names what it left out rather than dropping it:
another width, a distilled run, or another loss. The analysis has to apply the
same rule.

This is what the two-point framing hid. "Non-causal is 0.116 better" is true
and prices nothing a real system can buy; 19% for one sample period is a
design input.

Caveats that belong next to those numbers: one artifact mix (`BOTH`), one
width, one loss, and three seeds of a model whose seed drives the split, the
SNR pairing, the init and the batch order together — so the spread quoted is
total run-to-run variance, not an isolated effect. Every rung is trained at
all three seeds, so the curve is nine points and the ceiling.

**Trained models use the licence they are given.** Measured on the
checkpoints rather than on a fresh init: `tcn_se_la1` reaches exactly 1 sample
on all three seeds, and `tcn_se_la16` reaches 15, 16 and 15 of its 16 — 94% to
100%, against 88% for an untrained model of the same shape. Training moves the
SE gates out of the dead-ReLU zone described above, so the shortfall the tab
shows on fresh models is a property of initialisation, not a ceiling on what
the architecture can exercise.

**What this cannot do yet.** A rung is not causal, and every tool here reads
that correctly and refuses it: the ONNX exporter will not export one, and
`models/incremental.py` and `models/step_graph.py` will not stream one. That
is right as far as it goes — but a bounded-lookahead model *is* deployable,
by delaying the output L samples, and no path in this repo implements that
contract. So the ladder currently prices latency it cannot yet spend on the
board. The arithmetic is on the Deploy tab's axis deliberately, so the two
can be put together when it does.

Averages are per rung **and per width**: a run joins a rung's mean only if its
`hidden_channels` matches the width the rung was measured at, and excluded
runs are counted in the row rather than dropped. That rule is not theoretical
— `runs/tcn_se_both_s67` holds a 48-channel run beside two 64-channel ones,
and averaging the three made a "three-seed error bar" out of two seeds and a
width experiment.

Validation: `python -m unittest discover -s tests -p 'test_lookahead.py'`
covers the allocation, the two endpoints being bit-identical to the models
that already exist, licensed-vs-measured on every rung, the dead-ReLU
shortfall, the streaming refusals, and the tab's width filtering.

#### Receptive field tab

The mirror of the Lookahead tab, and the cheaper question. Lookahead asks what
*future* context is worth and charges latency for it. This asks how much of
the **past** the model actually needs — which costs no latency at all, only
parameters.

It exists because the study cannot currently separate two claims. The
attention variants are justified by long-range context: a 512-sample window
mixed globally, against a TCN that reaches back 125 samples (488 ms at
256 Hz). On this project's own Pi measurements that costs roughly 5x the
baseline's compute for about 1% of CC — inside the spread of a single seed.
But "attention helps" and "more context helps" are different claims, and every
run in the project was trained at the same reach, so nothing on disk tells
them apart. A dilated stack can be made to reach as far as the window for a
few thousand parameters and no attention at all:

```
[1]                      →    5 samples      20 ms
[1, 2]                   →   13              51 ms
[1, 2, 4]                →   29             113 ms
[1, 2, 4, 8]             →   61             238 ms
[1, 2, 4, 8, 16]         →  125             488 ms   ← the project default
[1, 2, 4, 8, 16, 32]     →  253             988 ms
[1, 2, 4, 8, 16, 32, 64] →  509            1988 ms   ≈ the whole 512 window
```

So if error keeps falling as the field grows, the benefit was context and the
cheap way to buy it is another dilation level; if it flattens at 125, the
model already had all the context it could use, and whatever attention
contributes is not reach. Either answer is worth having and neither is
available from the runs that exist.

**The x axis is exact before anything is trained**, as on the Lookahead tab,
and for the same reason: reach is a property of the graph. It is computed two
ways and the tests assert they agree at every rung —
`1 + Σ 2(k-1)d` over the schedule, and a perturbation probe that moves one
input sample and finds how far the output moves with it.

The probe runs in float64, which is not fussiness. The path that reaches
furthest is a single chain of kernel edges, attenuated once per block, so at
the far end of a 253-sample field the output moves by about 6e-9 while the
output itself is of order 0.1 — below the float32 spacing at that magnitude
(3e-8). In float32 the difference rounds to exactly zero and the probe reports
**252**, at some initialisations and not others. Nothing is dead there: PReLU
keeps a slope of 0.25 on its negative side, so no unit can zero the
dependence — the same reason `licensed` and `measured` agree on the
convolution stack in the Lookahead tab. It is pure rounding, and in float64
that tail sits eight orders above the floor.

**A rung's name states its reach**: `tcn_rf{N}` and `tcn_se_rf{N}`, with N the
field in samples, so `tcn_rf509` is the baseline given attention's span
without the attention. The tab launches one ordinary training run per rung,
shortest first, through the same runner the Train tab uses, from the family's
own recipe — `configs/tcn_both.yaml` or `configs/tcn_se_both.yaml` — with four
things overridden: the variant, the artifact, the seed and the output
directory.

Training from a config rather than from the trainer's own fallbacks is not
decoration. The fallbacks are `l1_combined` at lr 5e-3 for 150 epochs; every
baseline on disk is `mse` at 5e-4 for 200. A rung launched with no config
would differ from the 125-sample point it is plotted against in the loss, the
learning rate and the length at once, and all three would show up as nothing
but a worse number on the reach axis. `configs/tcn_both.yaml` was written for
this and records what `runs/tcn_both` was actually trained with, read back out
of its `history.json` — it is `tcn_se_both.yaml` with the gate taken off, and
the tests compare the two files key by key to keep it that way.

**A point is placed by what the run recorded, not by what it is called.** The
dilation schedule now goes into `history.json` with the rest of the
architecture (`### What shape a run actually trained`, below), and the curve
reads it from there — a directory named `tcn_rf253` whose weights say
otherwise is plotted where its weights say. Runs that predate the sweep
recorded no schedule, and they are placed at 125 anyway, because the dilations
were a constant until this tab made them an argument: that is what puts the
existing `tcn` and `tcn_se` results on the curve instead of making the sweep
re-measure a point it already has.

**A rung is a point, not a run.** Several runs can sit on one reach — the 125
rung arrives with every `tcn` or `tcn_se` run the project has ever done — so
the table and the curve show the mean over seeds with ±1 sd whiskers, and a
run joins that average only if it is the same model trained the same way.
Three things on disk already break that, and each is excluded *and named* —
the rung's `runs` cell says how many were left out and names them on hover,
because a run that exists and is excluded is something to show, not to lose.
The rule is `gui/lookahead.py:why_excluded`, shared with the Lookahead tab so
one run cannot be judged two ways:

| Excluded | Why | Example |
|---|---|---|
| another width | an error bar over widths is not an error bar over seeds | `runs/tcn_se_both_s67` is 48 channels beside its 64-channel companions |
| a distilled run | a bidirectional teacher is a different training signal, and its own row in the ablation table | `runs/tcn_se_both_kd_s67` |
| another loss | the val losses are not on one scale | anything trained `l1_combined` |

A stale or interrupted run is in neither column: its metrics describe weights
that are gone.

**Parameters are not held constant, and the table says so.** A rung adds one
residual block — 26,058 parameters at width 64, the same step at every rung —
so reach and depth move
together, and a difference between two rungs is "one more dilation level",
not "more context" in isolation. That is the honest reading of this ladder;
holding parameters fixed would mean widening the kernel instead, which buys
reach far more expensively and changes what each layer computes.

Validation: `python -m unittest discover -s tests -p 'test_receptive_field.py'`
covers the architectural sum against the probe at every rung, the streaming
wrapper's independent count, each variant reaching what its name claims, the
one-block cost per rung, the plan the tab builds (including that each command
carries its family's recipe and that the two recipes differ only in the gate),
and every placement and exclusion rule above.

### What shape a run actually trained

`history.json` records the architecture of the model that was trained, read
off the model itself rather than off the config meant to build it:

```json
"config": {
  "loss_fn": "l1_combined", "master_seed": 42,
  "n_params": 166605,
  "arch": {"variant": "tcn_mla", "hidden_channels": 64, "n_attn_layers": 2,
           "n_heads": 4, "kv_lora_rank": 16, "bottleneck_ratio": 0.5,
           "causal": true, "causal_tcn": true, ...}
}
```

This exists because the project was bitten by its absence. Every trained MLA
checkpoint here has **4 attention layers and 2 heads**; every MHA checkpoint
has **2 layers and 4 heads**; both YAML presets say 2 and 4. So the headline
MHA-vs-MLA comparison differs in depth, head count and parameter count
(167,533 against 191,085 — MLA is the *larger* model, not the smaller one),
and the README's "isolates exactly one variable at a time" does not hold for
it. Nothing recorded either number: the trainer filtered the architecture keys
out of its history, and a checkpoint's `model_config` stores only what was
explicitly passed, so a run that took a default recorded nothing at all.

`resolved_arch()` resolves the constructor's signature against what was
passed, so the record is complete and does not depend on the defaults staying
put. A rank alias's pinned value wins, and `causal` / `causal_tcn` are
included — a `*_bi` teacher and its causal student share a state dict, so the
weights cannot tell them apart and only the record can.

Runs trained before this carry no `arch`. The Runs tab falls back to the
checkpoint's `model_config` for those, and leaves the columns blank where
neither knows.

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
checkpoints, since they are not causal. It also records the recipe —
`kd_alpha`, `kd_feature_weight`, `kd_feature_layers`, `kd_output_loss`,
`kd_feature_loss`, `init_from_teacher` — which runs trained before 28 October
2026 did not: the history allow-list kept `teacher_ckpt` and dropped the rest,
so `runs/tcn_se_both_kd_s67` says it was distilled and from whom, and nothing
about how.

#### KD audit tab

The project's one distilled run is **worse** than the plain student at the
same seed and width — RRMSE_t 0.7161 against 0.7110. That has two
explanations and they have to be separated before either is reported: the KD
was broken, or it worked and distilling from this teacher costs accuracy. The
Evaluate tab cannot tell them apart, because both look like a student with a
worse number. The **KD audit** tab asks the two questions in order. Nothing
in it trains; it reads `runs/` and makes a few forward passes over the
validation split.

**Was it wired correctly?** Six invariants, each shown with its evidence:

| check | on `tcn_se_both_kd_s67` |
|---|---|
| teacher loads under `strict=True` | pass |
| teacher frozen and in `eval()` | pass |
| teacher is the non-causal twin | pass — `tcn_se_bi` |
| teacher reproduces its run's recorded metrics | pass — 0.5928 to 7.5e-09, whole split |
| student and teacher share a split | pass — seed 67 both |
| student's `best.pt` selected against the clean signal | pass — epoch 152, the argmin |

The reproduce check runs the whole validation split with no batch cap: the
recorded figure is a whole-split average, and a prefix measures sampling
rather than agreement — three batches of sixteen read 0.6018 against the
recorded 0.5928, which looks exactly like a mismatch and is not one. A check
that cannot run (no teacher recorded, no data) is shown as *skipped*, not as
passing.

**If it was, where did the accuracy go?** Expand each student's error about
the teacher:

```
||S−G||² = ||S−T||² + ||T−G||² + 2⟨S−T, T−G⟩
```

`||T−G||²` is the teacher's own error and the same constant for every
student, so a difference between two students lives entirely in the other two
terms. The tab finds the plain run at the same variant, seed and width
automatically and puts the two side by side:

| | ‖S−T‖² | ‖T−G‖² | 2⟨S−T,T−G⟩ | ‖S−G‖² |
|---|---|---|---|---|
| `tcn_se_both_s67_h64` (no KD) | 0.037836 | 0.079448 | −0.003476 | 0.113809 |
| `tcn_se_both_kd_s67` (KD) | 0.036861 | 0.079448 | −0.000690 | 0.115619 |

KD cut ‖S−T‖² by 0.00098 — it really did move the student onto the teacher —
and the cross term rose by 0.00279 while it did. The cross term is negative
for the plain student: its deviation from the teacher partly *cancels* the
teacher's own error. The teacher is non-causal, so `T−G` is largely the part
of its behaviour that depends on future context, which a causal student
cannot reproduce; pulling the student onto the teacher gives the cancellation
up and imports that error as noise. So the negative result is a finding, not
a failed experiment — on one seed, and with a recipe the run did not record.

Each row reports the residual between the three terms and the directly
measured ‖S−G‖², so a reader can see the identity close (~2e-9 here).

Not yet in the tab: the positive controls — distilling from the ground truth
itself, and the GT-scarce setting — need a teacher that is not a checkpoint,
which the trainer does not support yet.

Validation: `python -m unittest discover -s tests -p 'test_kd_tab.py'` covers
the recipe now being recorded, the control-run matching, the identity closing
on synthetic models, the read-off's three verdicts, and — when `runs/` and
the data are present — all six checks on the real run.

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
| `snr` | Per-SNR quantitative metrics (RRMSE_t, RRMSE_f, CC) |
| `curves` | Training loss and metric curves from history.json |
| `waveform` | Time-domain + PSD comparison plots |
| `attention` | Per-head, per-layer attention weight heatmaps |
| `snr_plot` | Per-SNR metric line charts |
| `compare_curves` | Overlay training curves from multiple runs |
| `compare_snr` | Overlay per-SNR metrics from multiple models |
| `switch` | Error vs. distance to a non-stationary artifact switch |
| `compare_switch` | Overlay switch-response curves from multiple models |
| `all` | Run all single-model tools (default) |

### Non-stationary artifact switch

Every table above scores a model on epochs whose artifact type never changes.
A deployed online denoiser does not get that: the artifact mix shifts as the
subject blinks, clenches, or stops doing either. The switch probe asks what a
model does at the moment the statistics change under it.

**The probe.** A clean EEG epoch gets one artifact type before a switch point
and the other after it. Nothing about the probe is constant enough to be
learned as a position: the switch index is drawn per sample inside
`switch_range` (default 0.25–0.75 of the epoch, so 128 samples on each side at
minimum), the two artifact types get independently drawn SNRs, and exactly half
the epochs run EOG → EMG while the other half run EMG → EOG. Each artifact is
calibrated against the clean signal of the segment it occupies, so the realised
SNR in each half is the SNR asked for regardless of where the switch landed.

It is an *evaluation* probe, not a training set — score models trained with
`artifact: BOTH` on it. It draws from the same leak-free pools as the ordinary
`BOTH` test split at the same `master_seed`, so a model's test split is its
test split here too.

**The series of seeds.** `master_seed` fixes which epochs exist in the split;
`probe_seed` fixes everything drawn on top of them — switch points, SNRs,
directions, pairings. The default series is `SWITCH_PROBE_SEEDS = (0,1,2,3,4)`,
and the curve is measured once per seed, so a bump has to survive five
independent draws of the switch positions to appear in the figure. Every draw
is made in `__init__` and stored, so `__getitem__` is a pure function of the
index: shuffling or scoring a subset cannot change what a sample contains.

**The matched control.** A raw response curve is not interpretable on its own,
because two other things happen at a switch point. The signal was *spliced* —
cut and glued to something else — which costs error whether or not the artifact
type changed. And a causal model has no left context at the start of an epoch,
so its error falls through the first few hundred ms regardless; because the
switch index varies, that within-epoch warm-up tilts the whole curve.

So every run also measures a control probe in which the splice still happens,
at the same index, but what gets spliced in is *a different epoch of the same
artifact type* at the same SNR. Everything else is identical, down to the
normalisation constant, which makes the two probes bit-identical before the
switch. The reported `delta` is switch − control, paired within each probe
seed. What it leaves is the change of artifact *statistics*, alone.

**Reading the numbers.** `eval.py --mode compare_switch` prints one row per
model and direction:

| Column | Meaning |
|--------|---------|
| `pre`, `floor` | Settled error either side of the switch, from the raw curve |
| `peak`, `oversh` | Raw peak and peak ÷ floor. Mostly the splice, not the type change |
| `leak` | RMS of Δ *before* the switch. Exactly 0.000 for a causal model — the two probes are one signal there. Reads as look-ahead only under a hard switch; `--switch_ramp` reaches back before the index by design |
| `far` | The same over the oldest half of the window. `leak > 0, far = 0` is a finite look-ahead horizon; both > 0 is look-ahead over the whole epoch |
| `Δfloor` | The regime shift alone: how much harder the new artifact is, at matched SNR, with the splice controlled for |
| `Tpeak`, `Trec`, `Tarea` | The transient (Δ minus Δfloor) — the re-adaptation cost. Peak, time to decay to 10 % of that peak, and area in relative error × ms |

```bash
# The figure: TCN-only vs. attention, five probe seeds, both directions
python eval.py --mode compare_switch --data_dir data/EEGdenoisenet/data/ \
       --compare_runs runs/tcn_both:TCN-only runs/tcn_mha_both:TCN+MHA \
                      runs/tcn_mla_both:TCN+MLA --save_dir figures/

# What future context is worth at a switch (causal vs. its non-causal twins)
python eval.py --mode compare_switch --data_dir data/EEGdenoisenet/data/ \
       --compare_runs runs/tcn_mla_both:MLA-causal runs/tcn_mla_bi_both:MLA-bidir \
                      runs/tcn_bi_both:TCN-noncausal --save_dir figures/causality/

# Controls: a soft switch instead of a hard one, and a single SNR
python eval.py --mode compare_switch --data_dir data/EEGdenoisenet/data/ \
       --compare_runs runs/tcn_both:TCN runs/tcn_mla_both:MLA \
       --switch_ramp 32 --switch_snr -3
```

All of this is also in the GUI's **Evaluate** tab (`python -m gui`), which
builds these commands for you and shows the figures inline.

**Two seeds, and they are not the same thing.** `--master_seed` is the *run's*
seed — it picks the train/val/test split, so it has to match the
`--master_seed` the checkpoint was trained with or the model is scored on the
wrong data (it defaults to 42, like the trainer). `--probe_seeds` sits on top
of it: the split stays put and only the switch points, SNRs, directions and
pairings are redrawn. Vary `--probe_seeds` for an error bar on the probe; set
`--master_seed` to match your run.

`--no-switch_control` drops the control probe and falls back to floor-referenced
numbers; `--switch_window`, `--switch_range` and `--switch_epochs` are the
other knobs. `--save_dir` writes
`switch_response.png` and a `switch_response.json` that carries the per-seed
curves, so a difference between two models can be tested seed by seed rather
than eyeballed off the band.

#### What it shows

Measured on the three `BOTH`-trained reference runs, five probe seeds, 510 test
epochs each:

| Model | Direction | peak | oversh | leak | Δfloor | Tpeak | Tarea |
|---|---|---|---|---|---|---|---|
| TCN-only | EOG→EMG | 1.007 | 1.52 | 0.000 | +0.084 | 0.029 | −27.2 |
| TCN+MHA | EOG→EMG | 0.970 | 1.44 | 0.000 | +0.065 | 0.031 | −24.7 |
| TCN+MLA | EOG→EMG | 0.974 | 1.44 | 0.000 | +0.062 | 0.026 | −22.9 |
| TCN-only | EMG→EOG | 0.960 | 1.71 | 0.000 | −0.084 | 0.318 | +21.5 |
| TCN+MHA | EMG→EOG | 0.931 | 1.67 | 0.000 | −0.088 | 0.305 | +23.6 |
| TCN+MLA | EMG→EOG | 0.944 | 1.65 | 0.000 | −0.076 | 0.306 | +22.1 |

Three things, and only the first two are effects:

1. **The switch costs a large error spike — but much of it is just the splice.**
   The raw error rises to 1.4–1.7× the settled level within 12–16 ms. The
   same-type control reproduces a good part of that. Measured as excess over
   the new-regime floor, raw peak − floor is 0.34–0.40 in both directions,
   while the type-change-specific transient is 0.31 (EMG→EOG) but only 0.03
   (EOG→EMG). Reading the raw peak as an adaptation cost therefore overstates
   it by ~1.2× arriving at EOG and by ~10–12× arriving at EMG.
2. **The cost is strongly asymmetric.** Arriving at EOG costs ~10× more
   transient than arriving at EMG. That fits the rest of the study: an EOG
   onset is a structured event the model has to respond to, while EMG onset is
   broadband and the model already handles it generically.
3. **Attention does not recover faster.** Across five probe seeds the paired
   difference in transient peak is +0.012 ± 0.007 (EMG→EOG, TCN-only worse on
   5/5 seeds) and −0.009 ± 0.005 (EOG→EMG, TCN-only *better* on 5/5). Transient
   area differs by less than its own spread in both directions. Attention does
   give a uniformly lower error level — the raw peak and the settled floors are
   both better — but that is the same advantage it has everywhere, not a faster
   response to non-stationarity. **The probe does not support a routing or
   re-adaptation account of what attention is doing here.**

#### What future context is worth at a switch

The same probe measures causality directly, because `leak` is exactly zero if
and only if the model cannot see past the switch:

| Model | Attention | Convolutions | leak | far | Δ first leaves zero |
|---|---|---|---|---|---|
| `tcn_mla_both` | causal | causal | **0.000** | **0.000** | never (at the switch) |
| `tcn_bi_both` | — | non-causal | 0.006 | 0.000 | −242 ms |
| `tcn_mla_bi_both` | bidirectional | causal | 0.015 | 0.015 | −477 ms (window edge) |

(Pooled over both directions; the per-direction rows are in the printed table.)

Three qualitatively different signatures, visible in the bottom row of the
figure. The causal model's Δ is **identically zero** at every offset before the
switch and jumps only at it — not approximately, exactly, which is what makes
the control a causality test rather than an estimate. `tcn_bi` departs from
zero 242 ms early and is exactly zero before that: its non-causal convolutions
have a finite receptive field, and that departure point measures it. The
bidirectional model never reaches zero anywhere in the measured window — its
attention reaches the whole 512-sample epoch, so it has no look-ahead horizon
to measure.

That 242 ms is a check on the method, not just a result. Perturbing a single
input sample of `tcn_bi` and asking how far back the output moves gives a
forward receptive field of exactly 62 samples — 242 ms at 256 Hz, independent
of width. The switch probe recovers the same number to the sample, from
artifact statistics alone, without touching the weights. That equality is
asserted in `tests/test_switch_probe.py`.

The non-causal models also settle ~0.11 lower, which is the familiar causality
cost, and they pay a *larger* relative overshoot (1.85–1.87 vs 1.65) — the
extra context they rely on is exactly what a switch invalidates.

Validation: `python -m unittest discover -s tests -p 'test_switch_probe.py'`
covers the probe's reproducibility and purity, the randomised switch point and
balanced directions, the SNR realised in each segment, the crossfade, the
leak-free split, the matched control's before/after behaviour, and the response
curve, its window clipping and its summary decomposition.

### Metrics

Following the EEGdenoiseNet benchmark:

- **RRMSE_t** — relative RMS error in the time domain (lower is better)
- **RRMSE_f** — relative RMS error in the frequency domain (lower is better)
- **CC** — average Pearson correlation coefficient (higher is better)

These are the names printed, plotted and used in the paper. The correlation
coefficient is **CC**, not ACC: in a denoising table "ACC" reads as *accuracy*,
which is not what it is. The spectral RRMSE is **RRMSE_f**, matching RRMSE_t
rather than mixing a domain letter with a word initial.

The internal dictionary keys are deliberately **not** renamed: every finished
run's `history.json` is keyed on `rrmse_t` / `rrmse_s` / `acc`, and so are the
checkpoints and the deploy sidecars. Renaming them would invalidate the runs
already in `runs/` for no benefit to a reader, so the keys stay and only the
labels move. The GUI's log parser accepts both spellings, so logs written
before the rename still plot.

## Deployment: ONNX on Raspberry Pi 5

The trained sandwich model (by default the 2-layer `tcn_mla`) can be exported
to a single ONNX file and run on a Raspberry Pi 5 with nothing but
`numpy` + `onnxruntime` — no PyTorch on the device.

Everything in this section is also in the GUI's **Deploy** tab (`python -m
gui`), which builds these commands, runs them as one job, and draws the
per-hop latency and the duty-cycle sweep from the traces they write — see
[Deploy tab](#deploy-tab).

### 0. The state of the machine, and why it is held still

A latency figure is a statement about a model *and* about the machine it ran
on, and the second half is usually left out. It should not be. Before anything
is timed, `deploy/preflight.py` reads the state of the three knobs that move
the numbers, judges them, prints the verdict, and writes it into every trace
the run saves — so a figure and the conditions it was taken under cannot be
separated afterwards.

```bash
python -m deploy.preflight --threads 2               # on the host
python preflight.py --threads 2 --cpus 2-3           # on the Pi
CHECK_ONLY=1 deploy/run_on_pi.sh pi@raspberrypi.local # over SSH, measuring nothing
```

| Control | What a reportable run needs | Why this one |
|---|---|---|
| **CPU governor** | `performance` on every core the process can run on | Under `ondemand` a hop that arrives after an idle gap is computed at a lower clock than one in the middle of a burst. That puts the governor's reaction time into the p99 and reports it as the network's. The duty-cycle numbers are compute ÷ period, so they scale with the clock directly. |
| **CPU affinity** | pinned, one core per thread, **counted from the top** (`--cpus 2-3`) | A Pi 5's four Cortex-A76 are identical, so migration does not land the work on a slower core — it loses cache locality and lets the process share a core with the OS. That is exactly what makes 4 threads worse than 2 in the table below (median 20.0 ms but p95 29.7 ms, against 18.2 / 18.9 at two threads). The pin counts down from the last core because **CPU 0 is where the kernel does its housekeeping** — device interrupts, the timer tick, RCU callbacks — and a forward pass sharing it gets a tail the model did not cause. The check does not take that on faith: it reads `/proc/interrupts` and reports the share each core actually carries, warning only if the pinned set includes one that is carrying a disproportionate amount. |
| **Thread count** | onnxruntime's `intra_op_num_threads`, set explicitly | The single largest effect here: 24.4 ms median at one thread, 18.2 at two, 17.4 at three, 20.0 at four. A run that does not record it is not comparable with one that does. See [onnxruntime's threading](#onnxruntimes-threading-is-not-openmps) — it is not the variable most people reach for. |
| **Throttling** | `vcgencmd get_throttled` = `0x0` since boot | A board that has browned out or capped its clock was not measured at the speed it reports. |
| **Other load** | 1-minute load average below 1.0 | Anything else running competes for the cores the measurement was pinned to. |

The verdict is one of **locked** (every control held), **partial** (a warning —
usable, but the tail is not fully reproducible) or **unlocked**. A platform that
*cannot* hold a control is `unlocked`, not excused: macOS has neither a governor
nor an affinity API, so a host run is a convenience and the board's run is the
measurement. `--require-locked` refuses to measure at all unless the verdict is
`locked`, which is what the Pi target uses by default.

Pinning is done in-process with `sched_setaffinity` rather than through
`taskset`, so the trace cannot claim an affinity the process did not have. The
governor is the one control that needs root, so it is opt-in
(`GOVERNOR=performance deploy/run_on_pi.sh …`, via passwordless sudo) and it
reports what it actually managed to do — a run taken under `ondemand` while the
report claims `performance` is worse than one that admits it. It is **not**
restored afterwards; the log prints the command to put it back.

Every trace written from now on carries an `environment` block with all of
this, including the sentence to paste into a methods section. Traces from
before this existed say *conditions not recorded*, which the GUI shows as its
own state — not as a passing one.

#### onnxruntime's threading is not OpenMP's

The thread count is the largest single effect on these numbers, and it is the
one most easily set to something that does nothing. Three facts about
onnxruntime, each measured on the build this project uses (1.30), each asserted
by a test:

- **`OMP_NUM_THREADS` governs none of it.** The wheels are built without
  OpenMP — `onnxruntime.get_build_info()` says so — so the variable has no
  effect on the timed path. It is still set, and still reported, because it
  governs numpy's BLAS in the same process, which is what scores the test set.
  The environment check used to *require* it to match the thread count and
  call that a pass; it was asserting a setting with no effect on the thing
  being measured, and now it says which is which.
- **`intra_op_num_threads = N` is N in total, including the thread making the
  call.** onnxruntime starts N−1 workers and runs the rest on the caller —
  measured as N−1 extra OS threads, and none at all at N=1. onnxruntime
  asserts the same thing itself: it refuses `intra_op_thread_affinities`
  unless given exactly N−1 entries. This is why N threads want N pinned cores,
  which is the ratio `--cpus` enforces.
- **The default is 0, which means one thread per physical core.** An unset
  thread count is not "single-threaded"; on a Pi 5 it is the four-thread row
  of the table above — the worst one.

The fourth fact is the one that changes what a duty cycle *means*. onnxruntime's
pool **spins** (busy-waits) before sleeping. Measured here over a paced replay,
one 512-sample window every 125 ms:

| `session.intra_op.allow_spinning` | median per hop | CPU used |
|---|---|---|
| `1` (onnxruntime's default) | 12.8 ms | **31 % of one core** |
| `0` (`--no-spin`, `SPIN=0`) | 14.0 ms | **18 % of one core** |

Spinning buys about a millisecond per hop and costs roughly 1.7× the CPU while
the loop is idle between them. So the duty-cycle table below is a **latency
budget** — compute wall-time ÷ hop period, which answers *can it keep up* — and
it is **not** CPU occupancy. Read "14 % duty at hop 32" as "86 % of the board is
free for an acquisition driver and a classifier" and you will be wrong by the
spin cost. Both settings are available and every trace records which one it
used; the default is left at onnxruntime's own so that figures measured before
this was exposed stay comparable.

Validation: `python -m unittest discover -s tests -p 'test_pi_tab.py'`.

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
- and, when `--data_dir` is given, reproduces the test-split RRMSE_t / RRMSE_f / CC
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
six files it needs, creates a venv there on first use, installs `numpy` +
`onnxruntime`, and runs the benchmark plus the bundled test set:

```bash
deploy/run_on_pi.sh pi@raspberrypi.local
```

`deploy/run_on_pi.sh pi@192.168.1.42 tcn_eog` tests another exported model;
`HOP=64 THREADS=3 deploy/run_on_pi.sh …` tunes the streaming hop and thread
count. Re-runs reuse the venv and take a few seconds. The script refuses to
start if the export or test set is missing and prints the command to create
it.

#### Connecting to the Pi

The script proves the connection with one cheap `ssh … true` before copying
1.2 MB of model, and when that fails it says *which* failure it was — from the
GUI they otherwise all read as "it timed out", and they need completely
different things done about them: a name that did not resolve, a key that is
not on the Pi, a host key that changed, or no answer at all.

Two settings are worth knowing about, because the defaults are chosen for a
Raspberry Pi 5 on WiFi rather than for a server in a rack:

| Variable | Default | Why |
|---|---|---|
| `SSH_CONNECT_TIMEOUT` | 20 s | This budget covers the TCP connect **and** the SSH handshake and key exchange — not just the connect. Raspberry Pi OS leaves WiFi power saving on, so the radio has to wake before any of that starts. A board on a slow link may genuinely need `SSH_CONNECT_TIMEOUT=45`. |
| `SSH_CONNECT_ATTEMPTS` | 3 | ssh tries **once** unless told otherwise. `raspberrypi.local` is answered over mDNS — a multicast query sent at the lowest WiFi rate, which is dropped routinely — so a single try turns an ordinary lost packet into a failed run. Three tries, one second apart, is roughly the system TCP timeout in total. |

The session also carries `ServerAliveInterval=15 ServerAliveCountMax=8`:
`ConnectTimeout` governs setting the connection up and nothing after it, and a
duty-cycle sweep runs for minutes. Without keepalives a link that dies
mid-measurement hangs until TCP gives up, which for an idle connection behind
NAT can be never.

If `.local` is unreliable on your network — it often is — use the address
(`hostname -I` on the Pi) rather than raising the timeout.

**A Pi on a direct Ethernet cable usually has no IPv4 at all.** With no DHCP on
that link, mDNS answers `<host>.local` with an IPv6 *link-local* address only,
and a link-local address is meaningless without the interface to send it out
of — so `ssh user@zcl.local` fails with `No route to host` even though the name
resolved perfectly. Give the address with its zone:

```bash
deploy/run_on_pi.sh 'changlong@fe80::8aa2:9eff:fe56:5033%en5'
```

`scp` needs that address bracketed where `ssh` does not (it splits
`user@host:path` on the first colon, so a bare IPv6 literal makes it try to
resolve the host `fe80`); the script brackets the scp target itself.

The script is one file but two programs: the local half, and a heredoc executed
by a different bash on the Pi, which sees only the variables the ssh line hands
across. Those are built from a single `REMOTE_VARS` list rather than spelled out
twice, because the two lists had to be kept in step by hand and were not — a
setting read by the Pi half but never sent killed the run under `set -u`, on the
board, after the files had been copied and the governor had been changed. Two
tests now check it both ways: one reads the two lists and compares them, and one
runs the whole remote half here with a stub standing in for the Pi's Python.

### 3. Manual steps on the Pi

```bash
# on the Pi (64-bit Raspberry Pi OS, Python 3.11+)
python -m venv ~/eeg-env && source ~/eeg-env/bin/activate
pip install -r requirements-raspi.txt        # numpy + onnxruntime (aarch64 wheels)
sudo cpupower frequency-set -g performance   # or let run_on_pi.sh do it: GOVERNOR=performance
export OMP_NUM_THREADS=2
python raspi_infer.py tcn_mla_eog.onnx --cpus 2-3 --require-locked --benchmark
python raspi_infer.py tcn_mla_eog.onnx --cpus 2-3 --input noisy.npy --output clean.npy      # (N, 512) epochs
python raspi_infer.py tcn_mla_eog.onnx --cpus 2-3 --input rec.npy --output clean.npy --stream --hop 32
```

`--cpus` pins the process (one core per thread, counted from the top so the
measurement stays off CPU 0 and its interrupts) and the environment check runs
before anything is timed; `--require-locked` makes a board that is not held
still refuse to be measured, and `--no-preflight` skips the check entirely —
the traces then record nothing about the conditions.

Only `raspi_infer.py`, `preflight.py`, `requirements-raspi.txt`, the `.onnx`
and its `.json` need to be copied (plus a test `.npz`, see below). From Python:

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
the host (epoch mode RRMSE_t 0.4601 / RRMSE_f 0.4643 / CC 0.8826). On an
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

This prints RRMSE_t / RRMSE_f / CC for epoch mode, streaming mode, and the
untouched noisy input as a baseline, plus a per-SNR table and timings. The
metrics are numpy replicas of `metrics.py` (they agree to 2e-7), so host and
Pi numbers are directly comparable. Reference for the shipped MLA model:
epoch mode RRMSE_t 0.460 / RRMSE_f 0.464 / CC 0.883 versus 1.41 / 4.91 /
0.595 for the raw noisy input. The streaming row is a pessimistic bound —
the epochs are independent 2-s segments, so every window straddles two
unrelated signals — and is not what a continuous recording would give.

To cut a different subset (other artifact, more epochs) on the host:

```bash
python -m deploy.make_testdata --data_dir data/EEGdenoisenet/data --artifact EMG --n 200
```

#### A test set belongs to the split it was cut from

"Test set" is not a property of a file on its own. `EEGDenoiseDataset`
permutes the epoch pool with `master_seed` and *then* slices train/val/test
out of that permutation, so the two seeds in this project do not agree about
which epochs are which: **69% of the seed-42 test epochs sit inside the
seed-67 training split** (465 of 678). Scoring a seed-67 model against
`EOG_test100.npz` does not crash, warn, or look odd — it returns a better
number than the model earns, which is the worst way for a measurement to be
wrong.

So the pairing is checked, in three places:

| Where | What it does |
|---|---|
| Deploy tab | shows the mismatch and disables **Run** before a Pi run is started |
| `/api/pi/*` | refuses the job — the page can be stale, and the API is reachable on its own |
| `raspi_infer.py --testdata` | refuses again on the board, which is where the number is actually produced |

The check reads provenance **stamped into the `.npz`** (`master_seed`,
`artifact`, `split`, `split_ratios`) rather than parsing the filename, because
a name is a convention and renaming a file changes what it claims without
changing what it holds. Sets written before this carry none, and are scored
with a warning that says the split could not be verified; re-cut them to stamp
it in. The model's side comes from the checkpoint's `run_config`, recorded in
the export sidecar as `provenance.trained_master_seed`.

A seed mismatch is refused; a cross-artifact pair (an EOG model on EMG epochs)
only warns, because that is a legitimate generalisation probe. To score a
mismatched pair deliberately, pass `--allow-split-mismatch` — the result is
then not a test score, and the flag says so.

```bash
# the matching set for a model trained at seed 67
python -m deploy.make_testdata --artifact BOTH --master_seed 67
```

Or let the Deploy tab do it. The **test set** control is a picker in the same
shape as the model one, and it labels every set against the model currently
chosen — `matches`, `wrong split`, or `unverified` — with the matching ones
sorted to the top. Its last entry is the generator: *Cut BOTH test set for
tcn_both (seed 67)*, which reads the artifact and seed off the model and runs
`make_testdata` with them (~40 ms), then selects the result. An existing
matching file is reused rather than rewritten, so a set that recorded
measurements refer to is never churned. A run whose seed was never recorded is
refused instead of cut at a guessed default — a set cut at the wrong seed
would *look* matched, which is the failure this whole section is about.

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
downstream classifier on the same board — though the duty cycle here is
compute ÷ period, a latency budget rather than CPU occupancy, and the thread
pool spins between hops by default, so the cores are busier than 14 % while the
loop is idle. `SPIN=0` trades about a millisecond per hop for that CPU back;
see [onnxruntime's threading](#onnxruntimes-threading-is-not-openmps).
Pushing below hop 8 would need a
KV cache / incremental attention so that the per-hop cost scales with the hop
instead of the window (see below), not more threads: 3–4 threads do not make
one 512-sample forward faster than ~15 ms on this SoC.

### What the export does not do

- **No dynamic INT8.** Static INT8 *is* available and is causal — see
  [Static INT8](#static-int8-the-quantisation-that-keeps-the-graph-causal)
  below. Dynamic INT8 was tried and rejected: it derives
  activation scales per tensor at run time, so a future sample changes the
  scale applied to past samples and the graph stops being causal
  (past_diff = 0.32 on the reference checkpoint). It was also slower for
  these tiny matmuls and slightly less accurate (CC 0.8754 vs 0.8772).
- **No KV cache.** Each call recomputes attention over the whole window. With
  T = 512 and d_model = 32 this is negligible; the KV-cache size advantage of
  MLA is therefore not exercised by this deployment path. It is exercised by
  [incremental inference](#incremental-inference-one-sample-at-a-time), which
  holds a real cache and measures it.

Validation: `python -m unittest discover -s tests -p 'test_export_onnx.py'`
covers architecture inference (including legacy checkpoints and `tcn_mla_r{N}` labels), export, the per-hop latency monitor,
agreement, causality, and the Pi-side epoch/streaming contract.

## Static INT8: the quantisation that keeps the graph causal

`deploy/quantize_int8.py` turns an exported fp32 graph into an INT8 one that
passes the same causality probe the fp32 export has to pass. It exists
because the earlier rejection of INT8 was a rejection of *dynamic* INT8
specifically, and the distinction is the whole point:

| | where the activation scale comes from | `past_diff` |
|---|---|---|
| dynamic INT8 | the tensor being quantised, at run time — the whole 512-sample window, future included | **0.32** |
| static INT8 | a calibration set, offline; written into the graph as constants | **0.0** |

A sample at *t* takes part in a dynamic scale applied to every sample before
it, so perturbing sample 256 moves the output at sample 0. That is what
dynamic quantisation *is*, not a tuning problem. Static calibration has
nothing at run time that depends on the window's contents, so the graph is
causal for the same reason the fp32 graph is — and every graph this module
writes is probed to prove it, at the same `1e-5` tolerance.

```bash
python -m deploy.quantize_int8 --onnx deploy/artifacts/tcn_both.onnx \
       --data_dir data/EEGdenoisenet/data --artifact BOTH --master_seed 67
python -m deploy.quantize_int8 --onnx deploy/artifacts/tcn_mla_both.onnx \
       --data_dir data/EEGdenoisenet/data --artifact BOTH --master_seed 67 --conv_only
```

Measured on the full test split, Apple-silicon host, 2 threads, QDQ,
per-channel weights, percentile calibration, 512 calibration windows from the
**train** split:

| graph | quantised | `past_diff` | size | per window | RRMSE_t | ΔRRMSE_t | ΔCC |
|---|---|---|---|---|---|---|---|
| `tcn_both` | all ops | 0.0 | 578.6 → 265.0 kB (2.18×) | 0.70 → 0.56 ms (1.26×) | 0.7245 | +0.0007 | −0.0006 |
| `tcn_se_both` | all ops | 0.0 | 673 → 352 kB (1.91×) | 1.20 → 1.12 ms (1.07×) | 0.7132 | +0.0009 | −0.0007 |
| `tcn_mla_both` | all ops | 0.0 | 1824 → 1392 kB (1.31×) | 3.42 → 5.03 ms (**0.68×**) | 0.7847 | **+0.0725** | **−0.0209** |
| `tcn_mla_both` | `Conv` only | 0.0 | 1824 → 1438 kB (1.27×) | 3.46 → 3.12 ms (1.11×) | 0.7128 | +0.0006 | −0.0004 |

Three things follow, and the third is the one to read twice.

**Convolutions quantise almost for free.** The baseline gives up 0.0007
RRMSE_t — a tenth of the seed spread — for 2.18× off the file and 1.26× off
the time. That is the deployable configuration.

**Attention does not.** Quantising the MLA sandwich end to end costs
**+0.0725 RRMSE_t**, which is 5.4× the entire gain the attention decoder buys
over the baseline in the first place (0.0134, [the capacity
axis](#what-a-millisecond-is-worth)), and it comes out **1.5× slower** than
fp32. An 8-bit MLA model is worse than the fp32 TCN baseline it was supposed
to improve on, at five times the compute. `--conv_only` leaves the attention
in fp32 and recovers all of it: +0.0006 RRMSE_t and 1.11× faster.

**The causal mask breaks two of the three calibrators.** A causal attention
graph carries a scalar `-inf` initializer — the mask value fed to `Where`
before `Softmax`, i.e. the thing that makes the model causal. `percentile`
and `entropy` build a histogram per tensor and cannot take a range of
`[-inf, inf]`, so they abort. onnxruntime reports that as *"autodetected
range … is not finite"*, which says nothing about masks, so the module
refuses first and names the cause, pointing at `--conv_only`. Causality and
quantisation are not independent here: the construct that buys the one
obstructs the other.

Notes on the settings: `uint8` and `int8` activations measured identically on
this host and both are offered, because the right one is a property of the
target CPU rather than the model. `percentile` beat `minmax` on every graph it
could run on (+0.0007 against +0.0014 on the baseline), and is the default.
Calibration draws from the **train** split at the checkpoint's own
`master_seed` — calibrating on val or test would fit the quantisation scales
to the data the model is then scored on, which is the leakage the epoch-level
split exists to prevent, one level down. A sidecar records the calibration
set, the settings, both probes, the sizes, the latencies and the accuracy of
both graphs.

The Pi runtime needs no changes — `deploy/raspi_infer.py` takes the INT8
graph as it stands:

```
$ python raspi_infer.py artifacts/tcn_both_int8.onnx --testdata testdata/BOTH_test100_s67.npz
       epoch mode                   0.7135   0.6638  0.7021      (fp32: 0.7125 / 0.7031)
$ python raspi_infer.py artifacts/tcn_mla_both_int8conv.onnx --testdata testdata/BOTH_test100_s67.npz
       epoch mode                   0.7033   0.6484  0.7136      (fp32: 0.7028 / 0.7139)
```

The latency figures above are a **host** measurement; the board's own numbers
need a run through the Deploy tab or `run_on_pi.sh`, and INT8's speedup is
the figure most likely to differ there, since the Pi 5's Cortex-A76 has
different 8-bit throughput from Apple silicon.

Validation: `python -m unittest discover -s tests -p 'test_quantize_int8.py'`
exports a small causal TCN and then asserts both halves of the claim in one
run — static quantisation keeps `past_diff` under tolerance, dynamic
quantisation of the same graph does not — plus that the probe detects a
genuinely non-causal model, that every activation type and calibrator yields
a causal graph, the mask preflight and its message, the calibration reader's
semantics, and what the sidecar records.

## Incremental inference: one sample at a time

`models/incremental.py` advances a model one sample at a time, keeping the
state a causal model needs instead of recomputing a window: a ring buffer per
dilated convolution, and an attention cache capped at the trained window.

```bash
# cache sizes and per-sample cost, on freshly built models
python -m deploy.bench_incremental --variants tcn,tcn_mha,tcn_mla

# a trained checkpoint, against the windowed path at two hop sizes
python -m deploy.bench_incremental --ckpt runs/tcn_mla_eog/best.pt --hops 32,64
```

```python
from models.incremental import IncrementalModel
stream = IncrementalModel(model)      # causal + eval(); bidirectional is refused
y = stream.run(signal)                # or stream.step(x_t) per sample
stream.cache_report()                 # state bytes, split by kind
```

### What it was built to measure

MLA was chosen for its KV cache, and until now no path in this repo held one —
the ONNX export recomputes, and `gui/params.py` could only estimate. The
streaming path holds the real thing, so the cache is read off live buffers:

| Variant | Attention | Cache floats/step | Cache over 512 steps |
|---|---|---|---|
| `tcn` | none | 0 | 0 B |
| `tcn_mha` | MHA (2 layers) | 128 | 262,144 B |
| `tcn_mla` | MLA (2 layers, rank 8) | 16 | 32,768 B |
| `tcn_mla_par` | MLA (2 layers, rank 16) | 32 | 65,536 B |

MHA keeps K **and** V, so the ratio is `2 * d_model / kv_lora_rank`. Note that
`d_model` is 32, not 64: the sandwich squeezes its 64 TCN channels by
`bottleneck_ratio` 0.5 before attention. The constructor default
`kv_lora_rank = d_model // 4` is therefore rank **8** and gives **8x**, while
the rank 16 that every YAML preset sets gives **4x**. The docstrings used to
describe `d_model // 4` as "4x", conflating the two; the arithmetic is now
asserted in `tests/test_incremental.py`.

### It is not faster in PyTorch, and the reason is worth knowing

Per emitted sample the stream does about 16x less arithmetic than recomputing a
512-wide window every 32 samples. It is nevertheless **2-3x slower** under
PyTorch eager:

| Path | ms/sample (`tcn_mla`, 1 thread) |
|---|---|
| incremental, per step | 0.360 |
| windowed, hop 32 | 0.168 |
| windowed, hop 64 | 0.084 |

The cost is not arithmetic. Growing the attention window 16x — 64 to 1024 —
moves the per-step time by 4.6% (0.364 to 0.381 ms), so the per-step cost is
almost entirely fixed overhead. Counting operators says the same thing: one
incremental step issues **1042** aten dispatches to emit one sample, while one
512-wide forward issues 4987 to emit 512 (9.7 per sample). That is ~107x more
dispatches per sample, at ~0.46 us each.

So the windowed path in `deploy/raspi_infer.py` is not a workaround — at
`d_model = 32` it is the right choice for a Python runtime, because it
amortises dispatch over 512 positions. Realising the incremental advantage
needs a runtime where a dispatch is cheap: ONNX with cache inputs and outputs,
or the hand-written numpy/C loop the Pi already runs. The arithmetic saving is
real and the cache saving is real; **PyTorch eager is simply the wrong place to
collect either.**

### Running it from the Deploy tab

Tick **incremental inference** on the Deploy tab (this machine only) and it
runs the trained checkpoint one sample at a time, writing
`deploy/artifacts/pi_latency/incremental_<model>_hop1.json` — which the Runs
tab then lists beside the windowed measurement of the same model.

Hop 1 is the literal truth: the streaming path advances one sample, so its
deadline is one sample period (3.9 ms at 256 Hz) rather than a hop of them.
That is also what makes the two comparable — expressed as a duty cycle,
compute over wall time, a windowed run at hop 32 and an incremental run at
hop 1 are the same quantity:

| | median | deadline | duty |
|---|---|---|---|
| `tcn_mla_eog` incremental | 0.51 ms/sample | 3.91 ms | **13.1 %** |
| `tcn_mla_eog` windowed, hop 32 | 2.82 ms/hop | 125 ms | **2.3 %** |

The option is **disabled for the Pi target**, with the reason on hover: this
is the PyTorch path and `run_on_pi.sh` copies onnxruntime and numpy only, so
there is no torch on the board to run it with. A split mismatch does not
block it either — it scores nothing, so it reads no test set, and the banner
says so rather than refusing.

### On the Pi: the step graph

`models/incremental.py` is PyTorch and the board has none, so the streaming
path could be measured on the host and nowhere else. `models/step_graph.py`
is the same computation written as a graph that takes its state in and hands
it back:

    (x_t, state...) -> (y_t, new_state...)

```bash
python -m deploy.export_onnx --ckpt runs/tcn_both/best.pt --name tcn_both --incremental
python deploy/raspi_infer.py tcn_both_step.onnx --incremental --testdata BOTH_test100_s67.npz
INCREMENTAL=1 deploy/run_on_pi.sh pi@raspberrypi.local tcn_both
```

The Deploy tab's **incremental inference** box now works for the Pi too; it
asks for the step graph and says how to export one if it is missing.

The ring buffers become shift-and-append — a ring needs an index that changes
per call, which a static graph cannot carry, while a shift is pure tensor
algebra. And because a graph cannot slice its cache to the part it has
written, a **validity mask** rides along as state, so the first `window`
outputs are exactly the ring-buffer path's rather than an average over
zeroed slots.

**Measured on the board**, Raspberry Pi 5, 2 threads pinned to CPU 2-3,
51,200 samples each:

| model | path | ms/sample | duty | misses | KV cache | state | RRMSE_t / CC |
|---|---|---|---|---|---|---|---|
| `tcn_both` | incremental, hop 1 | **0.240** | 6.1 % | 0 | — | 31 kB | 0.7125 / 0.7031 |
| `tcn_both` | windowed, hop 8 | 0.403 | 10.3 % | 0 | — | 0 | 0.7125 / 0.7031 |
| `tcn_both` | windowed, hop 32 | 0.101 | 2.6 % | 0 | — | 0 | 0.7125 / 0.7031 |
| `tcn_mla_both` | incremental, hop 1 | **0.760** | 19.6 % | 0 | **64 floats/step** | 192 kB | 0.7028 / 0.7139 |
| `tcn_mla_both` | windowed, hop 32 | 0.547 | 14.0 % | 0 | none | 0 | 0.7028 / 0.7139 |

Identical scores from each model's two paths, which is the point: the step
graph is the same function, not an approximation of it. For the baseline,
0.228 ms of the 0.240 is onnxruntime and 0.012 the normalise/rescale around
it; p95 0.253 ms, 15.4x inside the deadline, zero misses. The MLA row is the
first measurement in this project where **the KV cache exists on the
hardware** — 131 kB of it, allocated by onnxruntime on the board, next to
64 kB of convolution state that no attention mechanism shrinks.

It buys **latency, not throughput**. A windowed call amortises over the whole
hop, so at hop 32 — 125 ms of buffering — it is still the cheaper path per
sample. The step graph wins as the hop shrinks: it already beats hop 8, and
at a hop of one sample the windowed path would need 3.23 ms per output (one
whole window recomputed to emit one sample, 83 % duty) against the step
graph's 0.240. That is the case for it — a BCI loop that cannot afford 125 ms
of buffering has no other option — and on an attention model the KV cache is
finally a tensor onnxruntime allocates on the board rather than an estimate
made on a laptop.

Host figures for comparison, same models: 0.061 ms/sample (`tcn_both`) and
0.180 ms (`tcn_mla_both_s67`, 181 kB of state).

Nothing here is trusted because it exported cleanly. The export verifies the
graph against `models/incremental.py` — itself pinned to the batch model from
both ends — over a run longer than the attention window, and refuses to write
a sidecar if they disagree:

```
[verify] onnxruntime step loop vs models/incremental.py ...
         max|onnx-incremental| = 2.38e-07 over 640 steps
```

### How to tell which path produced a trace

Every trace now carries an `inference` block, for the same reason it carries
an `environment` one: a latency figure is a statement about a model, a
machine **and** a method, and the third was being left to the reader to infer
from which files reach the board.

```json
"inference": {"mode": "windowed", "runtime": "onnxruntime", "window": 512,
              "hop": 32, "kv_cache": false, "recompute_factor": 16.0}
"inference": {"mode": "incremental", "runtime": "pytorch 2.12.0", "hop": 1,
              "kv_cache": true, "kv_floats_per_step": 64,
              "kv_cache_bytes": 131072, "recompute_factor": 1.0}
```

Traces written before this said nothing, and all of them were windowed; the
Runs tab shows those as `windowed` without claiming the trace said so. Three
checks work on any trace, old or new:

- **the ONNX signature** — `io` is a whole window in and a whole window out,
  with no cache tensors either side, so each call recomputes everything. This
  is a property of the artifact, not of how it was invoked.
- **the directory** — windowed traces are `pi_latency/{latency,host}_*`;
  incremental ones are `pi_latency/incremental_*`.
- **the hop** — a windowed trace is keyed by a hop of many samples; an
  incremental one is hop 1.

### Equivalence

Against a single forward pass over the whole signal the stream is exact for
every variant (~3e-7). Against the sliding-window path it is exact only for the
baseline (~7e-8); the attention variants differ by ~1e-5 relative to signal RMS
(up to ~3e-3 on a trained checkpoint), because attention at the window's last
position attends over 512 cached positions whose oldest ~252 were, in the
windowed path, computed by a TCN reading the window's left zero-padding. The
stream has real history there instead.

Neither is wrong: the windowed path reproduces the training-time computation
(epochs are 512 samples with a padded left edge), while the stream gives the
oldest cached positions their fully resolved features. The gap is bounded, does
not grow with `t`, and is orders of magnitude below an RRMSE of ~0.65.

Validation: `python -m unittest discover -s tests -p 'test_incremental.py'`
covers equivalence from both ends, streaming causality (the same
perturb-and-look-back probe as `test_variants.py`), MLA's two cache strategies
against each other and against the batch model, the cache arithmetic, and the
refusal of bidirectional and TurboQuant models.

## Reproducibility

The project includes an automated reproducibility test that verifies the entire pipeline from scratch:

```bash
# Full test (builds all models, checks causality, trains 3 epochs, evaluates)
python reproduce.py

# Quick test (skips training, ~10 seconds)
python reproduce.py --quick
```

This runs **105 automated checks** covering model construction, causality (including that the `*_bi` teachers do see the future, and that the optional BottleneckSkip stays causal), seeding (same `master_seed` → same init and batch order), losses, metrics, training, checkpointing, evaluation, knowledge distillation, and the artifact-switch probe (reproducible draws, the SNR realised per segment, a control that differs only after the switch, and a response curve that reads look-ahead back out). It synthesises random data if the real dataset is not present, so it works in CI without any external downloads.

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
| Lookahead ladder (`tcn_se` → `tcn_se_la{N}` → `tcn_se_bi`) | What does *each millisecond* of that cost buy? | Same weights, recipe and seed; only how far ahead the model may read |
| Receptive-field ladder (`tcn_rf5` … `tcn_rf509`) | Was the benefit attention, or just reach? | Same recipe and seed; only how far back the convolutions see (one dilation level per rung, so parameters move with it) |
| `tcn_bi` vs `tcn_mla_bi` | Is the gain from seeing the future, or from attending to it? | Both non-causal; only one has attention |
| `X_bi` vs `X_full_bi` | How much is attention alone worth against the full offline ceiling? | Same weights; only the convolution padding differs |
| Student vs distilled student (`X` vs `X` + KD) | Can a non-causal teacher close that gap? | Same model, only the training signal differs |
| Artifact switch vs same-type splice | What does a change of artifact *statistics* cost, over and above a splice? | Same clean epochs, switch points, SNRs and normalisation; only the type changes |

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
