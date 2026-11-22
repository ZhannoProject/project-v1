# Bounded Lookahead for Streaming EEG Artifact Removal on an Edge Device

Core model implementation, pretrained weights, and the local training/evaluation GUI for the accompanying EEG artifact-removal manuscript. The study measures how a declared amount of future context changes reconstruction quality, output waiting time, and Raspberry Pi 5 deployment cost.

The primary model is a single-channel temporal convolutional network with ChannelNorm and squeeze-and-excitation (TCN+SE). Causal TCN and a low-rank attention model provide controls; an offline TCN+SE supplies the reference ceiling.

## Models and temporal support

- Five residual blocks, two kernel-three convolutions per block, 64 channels, and dilations `[1, 2, 4, 8, 16]` form the TCN/SE backbone.
- ChannelNorm normalizes feature channels at each time index. Input scaling uses a fixed training-derived value saved with each checkpoint.
- SE uses a reduction ratio of 8. Causal and bounded variants accumulate available history; the offline reference pools over the full input window.
- The architectural ladder allocates the total lookahead budget across convolution padding and SE pooling. Increasing lookahead also changes the convolutional past span.
- `L=186` has **no convolutional past context**. Cumulative SE can still use earlier window history.
- The attention control uses 52 channels, four causal attention layers, two heads, and KV rank 16: 127,839 trainable parameters. Saved masks and other buffers are additional tensor elements.

At 250 Hz, a horizon of `L` samples adds `4L` ms of future waiting. With hop `H=4`, the processing period is 16 ms and the conservative batching-plus-lookahead budget is `(H+L)/250` seconds. Filtering, computation, queueing, acquisition, and transport require separate accounting.

## Manuscript results

The matched synthetic comparison uses a common target interval and five seeds, except the three-seed attention control. The following RRMSE values are means across training seeds.

| Model | Lookahead | Parameters | Temporal RRMSE |
|---|---:|---:|---:|
| TCN | 0 | 117,068 | 0.751 |
| TCN+SE | 0 | 122,548 | 0.737 |
| TCN+SE | 8 | 122,548 | 0.679 |
| TCN+SE offline | full window | 122,548 | 0.626 |
| TCN+attention | 0 | 127,839 | 0.736 |

Eight future samples add 32 ms of waiting and recover about 52% of the SE family's causal-to-offline temporal-error gap. Gains diminish at larger horizons. These quality results use window-reset SE inference.

On EEG-IO, the study evaluates blink suppression and background input-output alteration across 20 participants. Background RMSE has no paired clean EEG reference and therefore measures alteration rather than verified neural-signal loss.

A separate stateful `L=8` experiment on Raspberry Pi 5 achieved single-channel hop p50/p99 of **1.84/2.04 ms**, with no missed 16 ms deadlines across a 33.7-minute full-cohort replay. The manuscript reports approximately **2.2 mJ/hop above idle** for this operating point. Future waiting and batching remain additional delay costs. Power estimates use an external meter and cooling-state/idle correction.

Replicated-stream capacity is limited by deadlines: causal TCN sustained 16 streams in the tested two-core configuration; SE variants sustained eight steady-state streams, while their 16-stream tests accumulated backlog. The attention control overloaded at eight and 16 streams. These are replicated independent single-channel streams. The eight-stream TCN measurement used the earlier one-process implementation; the other multichannel measurements used two processes.

## Released weights

[`pretrained/manifest.json`](pretrained/manifest.json) lists all 63 checkpoints, SHA-256 hashes, exact architecture metadata, normalization scales, training seeds, and saved common-interval test scores.

| IDs | Seeds | Count |
|---|---|---:|
| `tcn_s{seed}` | 42, 54, 67, 72, 88 | 5 |
| `se_l{L}_s{seed}`, `L=0,1,2,4,8,16,32,62,124,186` | 42, 54, 67, 72, 88 | 50 |
| `se_offline_s{seed}` | 42, 54, 67, 72, 88 | 5 |
| `attention_s{seed}` | 67, 72, 88 | 3 |

The release checkpoints contain model tensors, architecture and training/data configuration, and selection metadata. Optimizer state and training histories have been removed; every retained state tensor was checked against its source checkpoint.

Four ready-to-run FP32 step graphs and their verification sidecars are included under [`pretrained/onnx`](pretrained/onnx):

- `tcn_s72_step.onnx`
- `se_l0_s72_step.onnx`
- `se_l8_s67_step.onnx` — experimental bounded incremental path
- `attention_s72_step.onnx`

## Install and verify

Use Python 3.10 or newer:

```bash
python -m pip install -r requirements-core.txt
python verify_models.py
python inference.py --list
```

Verification reconstructs all 63 architectures with strict state loading, checks the four ONNX graphs against PyTorch, verifies target alignment and the `L=8` future bound, and checks that the offline reference rejects streaming.

## GUI (Windows and macOS)

From the repository root, in the same Python environment:

```bash
python -m pip install -r requirements.txt
python prepare_gui.py
python -m gui --browser
```

The GUI opens at `http://127.0.0.1:8765`. Browser mode works on Windows and macOS without a bundled desktop runtime. For an optional native window, install `pywebview>=5.0` and run `python -m gui`.

`prepare_gui.py` verifies the release hashes and copies the compact weights into the GUI's `runs/<source_run>/best.pt` layout. It also registers the four step graphs under `deploy/artifacts/`. Existing trained checkpoints are preserved. Released weights appear as **Pretrained** in the model picker, including the `tcn_mla_r16_*` attention models. Training curves and historical validation averages are not fabricated.

The included interface supports configuration editing, training, evaluation, EEG-IO comparison, Raspberry Pi replay, and 8/16-stream tests. Pi tests require an SSH-accessible Pi and local recordings. The all-subject preset plays every available subject once; replicated streams are processed by two worker processes, one inference thread per worker. Power still requires an external meter.

Datasets are supplied separately:

- EEGdenoiseNet: place `EEG_all_epochs.npy`, `EOG_all_epochs.npy`, and `EMG_all_epochs.npy` under `data/EEGdenoisenet/data/`, or set the training data directory in the GUI.
- [EEG-EyeBlinks / EEG-IO](https://gnan.ece.gatech.edu/eeg-eyeblinks): place the original subject files under `data/EEG-IO/` or `data/EEG-EyeBlinks/EEG-IO/`. The GUI reports missing recordings before evaluation.

New training outputs go to `runs/`, EEG-IO comparisons to `eegio_tests/`, and Pi/channel measurements to `deploy/artifacts/`. These generated directories are ignored by Git. Presets cover additional exploratory experiments; use each released checkpoint's recorded architecture and protocol when reproducing a manuscript model.

## Inference

Inputs are one-dimensional NumPy arrays, shape `(samples,)`, at 250 Hz. Outputs retain the input's target-sample indexing. Run each independent recording separately to reset filters and model state.

### Window-reset comparison

```bash
python inference.py --model se_l8_s67 --mode window \
  --input eeg.npy --output cleaned_window.npy
```

This mode uses 500-sample windows with 125/186 retained context margins, matching the EEG-IO ladder comparison. Unavailable edge targets are `NaN`. The synthetic results above use their own common test crop, recorded in the manifest.

### Stateful incremental inference

```bash
python inference.py --model se_l0_s72 --mode stream \
  --input eeg.npy --output cleaned_stream.npy

python inference.py --model se_l8_s67 --mode onnx --experimental-l8 \
  --input eeg.npy --output cleaned_l8.npy
```

The `L=8` stateful path emits the estimate for target `t` after sample `t+8` arrives. Its final eight targets are unavailable and remain `NaN`; no fabricated future samples are flushed. Cumulative SE persists within the recording. This differs from per-window SE resets, so the windowed accuracy scores do not directly validate stateful accuracy. Other nonzero horizons are available for window inference; this release's incremental entry point covers the causal models and the explicitly enabled `L=8` experiment.

For raw EEG-IO-style microvolt recordings, enable the paper's forward 1–80 Hz filter, 60 Hz notch, and fixed expected background level:

```bash
python inference.py --model se_l8_s67 --mode onnx --experimental-l8 \
  --raw --input-level-uv 20 --input fp1_raw.npy --output fp1_cleaned.npy
```

Without `--raw`, supply already preprocessed input. Without `--input-level-uv`, the saved training scale is used. All scaling is fixed rather than estimated from future recording samples.

For direct PyTorch access:

```python
import torch
from inference import load_pretrained

model, info = load_pretrained("se_l8_s67")
scale = model.run_config["normalization_scale"]
x = torch.zeros(1, 1, 500)  # replace with a preprocessed microvolt window
with torch.inference_mode():
    y = model(x / scale) * scale
```

## Repository layout

```text
models/               TCN, ChannelNorm, SE, attention, and stateful graphs
pretrained/pytorch/   63 compact, fixed trained checkpoints
pretrained/onnx/      Four FP32 step graphs and their metadata
pretrained/manifest.json
inference.py          Standalone functional inference entry point
verify_models.py      Weight, alignment, parity, and future-bound checks
requirements-core.txt
prepare_gui.py        Register released weights and graphs for the GUI
gui/                  Front end, local server, Windows/macOS window support
configs/              Training presets
trainer/, data/       Training and dataset-loading source (no datasets)
experiment/, utils/   Shared configuration, checkpoint and runtime helpers
deploy/               ONNX export, Pi checks and paced/parallel replay
eval.py, eegio_eval.py, metrics.py
requirements.txt      GUI, training/evaluation and export dependencies
tests/                Release/GUI integration checks
```

The manuscript's device measurements use original-speed paced replay, hop 4 at 250 Hz, fixed CPU controls, and external power measurement. This repository includes model and GUI source with the dependencies needed to run them. Datasets, historical runs, desktop application bundles, experiment archives, and manuscript working files are not part of this source release.
