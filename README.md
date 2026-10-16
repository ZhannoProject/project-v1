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
│   ├── Dataloader.py           # Leak-free train/val/test splits + SNR mixing (seeded),
│   │                           #   and the non-stationary artifact-switch probe
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
│   ├── server.py               # stdlib HTTP/JSON API, training / evaluation / deploy job runners
│   ├── pi.py                   # Deploy tab: step builder, log reader, latency + duty traces
│   └── static/index.html       # the page
├── deploy/
│   ├── preflight.py            # the state of the machine a timing run is taken on, judged
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

The page has six tabs: **YAML** (the text a save would write), **Train**,
**Evaluate**, **Deploy**, **Params** and **Runs**. The Evaluate and Deploy tabs
borrow the Runs table as a run picker, so all three share one renderer.

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

  **curve** on any finished run plots its saved history in the same chart, and
  “Use as teacher” fills `teacher_ckpt` for a distillation run.

The server binds to 127.0.0.1 only and runs one training job at a time.
Validation: `python -m unittest discover -s tests -p 'test_gui.py'`.

#### Evaluate tab

The other half of the GUI: it runs `eval.py` on a finished run and shows what
comes back. Every `--mode` the script has is offered — per-SNR metrics, training
curves, waveforms, attention heatmaps, the artifact-switch probe, and the
multi-model comparisons — and the mode decides which controls are shown, so the
switch options appear for a switch run and nothing else.

- **The run is picked from the Runs table, not a dropdown.** Clicking the
  *run* field opens the Runs tab's own table in a dialog — the same groupings
  (by kind, loss, optimizer, time, or all), the same params, seed, epochs, val
  loss, RRMSE_t / RRMSE_f / CC and the same `bi` / `stale` / `partial` badges —
  and clicking a row picks it. Both are rendered by one function, so the picker
  cannot drift from the tab. The two actions that would navigate away mid-choice
  (`curve`, "Use as teacher") are dropped in the picker.

  Nothing is selected until you select it. The plain dropdown this replaced
  defaulted to whatever sorted first alphabetically — `ctrl_mla_both_bi`, a
  *non-causal control* — so evaluating the wrong model took one click and
  nothing said so. The field now reads "Choose a run…" and the command preview
  says `mode 'X' needs a run` until one is picked. The choice is remembered
  between sessions.
- **Two jobs, not one.** Training and evaluation are separate; starting an
  evaluation neither blocks a running training job nor clobbers its log.
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

- **No INT8.** Dynamic INT8 quantisation was tried and rejected: it derives
  activation scales per tensor at run time, so a future sample changes the
  scale applied to past samples and the graph stops being causal
  (past_diff = 0.32 on the reference checkpoint). It was also slower for
  these tiny matmuls and slightly less accurate (CC 0.8754 vs 0.8772).
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
