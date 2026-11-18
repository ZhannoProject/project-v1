# Seven-rule methodology implementation

The strict `paper` profile is a new experiment protocol. Existing weights,
results and the architectural `tcn_se_laN` ladder remain historical/exploratory.
They are not made publication-ready by editing their metadata. Attention is
optional supporting evidence; the primary study uses TCN and SE.

## 1. Training, selection and repeats

All newly written best checkpoints are selected by validation ground-truth
MSE, including KD and training with another exploratory loss. `selection_metric`
is saved in the history and checkpoint. KD loss and teacher agreement never
select a checkpoint. The paper profile requires ordinary AdamW, 200 epochs,
fixed input normalization, causal preprocessing and H=4.

Generate a complete, reviewable plan without starting training:

```bash
python -m experiment.paper plan --out runs/paper_256 --fs 256
python -m experiment.paper plan --out runs/paper_250 --fs 250
```

The default plan includes TCN and SE at ell=0,8,16,32 and a finite-window
bidirectional SE offline reference, each at seeds 42,54,67 (27 jobs). Plans,
configs and output directories are separate from historical experiments.
Creation refuses to replace an existing plan. Review `plan.json`, then run:

```bash
python -m experiment.paper train --plan runs/paper_256/plan.json
python -m experiment.paper evaluate --plan runs/paper_256/plan.json
python -m experiment.paper report --plan runs/paper_256/plan.json
```

Evaluation requires the same run to have completed all 200 epochs and to have
used GT-MSE selection. Test loading restores its seed, pools, normalization,
rate and resampling. The reporter requires all three seeds per configuration
and reports mean ± sample SD (`ddof=1`). No training is started by `plan`.

## 2. Causality

Paper presets omit the TCN normalization layers (`tcn_norm: none`).
`tcn_norm: channel` is also causal and applies LayerNorm across channels at
each sample. Use the same choice across seeds and delays in a comparison.
Historical BatchNorm checkpoints load with their original running statistics.
The fixed input scale fitted on the training data is separate from these layers.
Convolutions use left padding. SE uses a fixed 16-sample trailing average;
this is an explicit new baseline choice, configurable and checkpointed.
Unlike cumulative SE, its dependency span is finite. Changing ell never
changes this pooling window or any model parameter.

Attention can be included with `--variants tcn tcn_se tcn_mla tcn_se_bi`.
Its optional local window is identical in batch, cached and ONNX paths;
the plan expands the shared crop for the full stacked dependency span.
With two 32-position attention layers, the sandwich context is 315 samples.
Whole-epoch attention and cumulative/global SE are refused as finite-window
paper models. Historical checkpoints retain their original behavior.

`data/resampling.py` uses causal rational FIR conversion, preserves filter
delay and records startup exclusion. The 256→250 conversion has 40 ms group
delay and 20 output samples of startup. Training and evaluation use the same
conversion. It must not be described as a zero-delay online converter.
Real-data and device replay should use native-rate models where possible.

`experiment.methodology.parity` compares the raw-amplitude batch path with
H=4 stateful inference, including forward-only filters and fixed calibration.
It checks the common valid interval at absolute max delta <1e-4. Paper
evaluation runs a 1024-sample probe with final weights, crossing cache wraps.
ONNX export separately checks step-graph agreement at the same threshold.

## 3–4. Delay and common target positions

At acquisition index u, the model estimates clean[u-ell]. In code:

```text
prediction[..., start+ell:stop+ell] vs clean[..., start:stop]
```

The convolution padding, pooling, layer count, parameter count and state
layout are unchanged across ell. Targets without real inputs are excluded,
not zero-filled. Training, GT validation, KD output/feature terms, main
evaluation and exported-backend scoring share this alignment.

For each planned architecture, derive past P and future F by composing
every convolution, finite SE gate and attention layer. The shared crop is
conservatively:

```text
start = max(P) + causal-resampler startup
stop  = N - max(F + ell)
```

Every run uses these same target positions. Empty intervals or insufficient
history/future margins are rejected. The default 256 Hz plan uses [199,410)
and the 250 Hz plan uses [219,398). Adding attention recomputes the interval.
This intentionally differs from the old R=187 assumption. TCN alone spans
125 samples; SE(window=16) spans 200. Offline SE uses a finite centered pool,
so it is not the historical global-pool offline teacher.

Algorithmic budget is (H+ell)/fs, labeled as a budget, not a measurement.
For ell=16, H=4 it is 78.125 ms at 256 Hz and 80 ms at 250 Hz. Filter delay,
I/O and inference are accounted for separately. End-to-end percentiles come
from actual per-target timestamps; component p99 values are never added.

## 5. Statistical units and attention

Evaluation saves a fingerprint of each underlying noisy/clean test pair,
the split seed, sampling rate and exact target interval. Comparisons first
pair these segment identities, then aggregate within each training seed.
Independent inferential units are training seeds, not all pooled segments.
Real-data evaluation produces one row per subject; blinks are not independent
subjects. `paired_tost` accepts only seed/subject units with matching IDs.

`python -m experiment.paper real --plan runs/paper_250/plan.json --data_dir DATA/EEG-IO`
evaluates the three repeats and groups each subject's repeated estimates
together in `real_report.json`, rather than creating three independent people.

No equivalence margin has been prespecified for attention vs SE. The reporter
therefore reports paired differences and `equivalence: not_tested`. It does
not turn a nonsignificant difference or small accuracy gain into equivalence.
TOST requires an explicit positive margin decided before interpreting the
confirmatory data. Computation cost is a separate measured endpoint.

## 6. Hardware evidence

`deploy/paper_bench.py` runs only on a controlled Pi 5, with governor, affinity
and thread checks before and after replay, including temperature/throttling.
It groups four acquired samples, processes all four with persistent state,
and records hop and target timestamps. It includes causal filtering by
default and requires the input's declared rate to match the model.

First export/exercise random weights to validate the full chain, then repeat
with final weights. `--weights random|final` records which one was measured.
The benchmark never changes the board governor automatically.

```bash
python -m deploy.paper_bench --model MODEL_step.onnx --input raw_250hz.npy \
  --input_fs 250 --out measurements/se_final --cpus 2-3 --threads 2 \
  --weights final
```

Use `--channels 2|8|16` for replicated single-channel processing. These rows
do not claim a separately trained multichannel model. Latency, processing
duty cycle, peak process RAM and end-to-end percentiles are device observations.
Host timings, cache-size arithmetic and algorithmic budgets are not Pi data.

The GUI's **EEG channels** tab runs a saved sweep of 1, 2, 8 and/or 16 EEG
input streams. Select a causal model, hop, replay duration and repeat count,
then preview and run. Hop defaults to four samples. Other hop sizes are
recorded as exploratory. Host runs are explicitly diagnostic. Pi runs
use SSH and require the existing governor/affinity checks to pass; the tab
keeps runtime threads and CPU affinity fixed across cases. Case order rotates
between repeats. Each case starts in a fresh process with independent stream
states, using archived copies of the same model, metadata and input.

By default, replay a continuous unflagged segment from EEG-IO with causal
filtering included in the timing. The recording is converted to the model's
sample rate during setup if needed. For another input, supply a finite
one-dimensional NPY recording at the model's native rate (upload limit
64 MiB), or choose a test-set timing fixture and set the filter option.
Joined test epochs are labeled timing-only. Paper models reuse their
saved training calibration; legacy models require an explicit prior scale
or **Calibrate from training data**, which uses only the training EEG pool.
The weight-provenance selector records the selected export's origin; random
weights must be exported beforehand.

Saved experiments live under `deploy/artifacts/channel_sweeps/`. The tab
shows completed repeats, mean and sample SD of per-repeat latency percentiles,
deadline misses/rate, duty cycle and peak RAM. Stopped or failed experiments
retain completed cases; missing repeats stay blank. Measurement records
include the board, thread/affinity settings, governor, temperature and
throttling checks before and after replay. Use **Add external-meter power
data** to attach synchronized active/idle CSVs to each completed case and
report idle-subtracted mW per EEG channel. Power stays blank until measured.

Power requires active AND idle external-meter CSVs (`time_s,power_mw`), plus
`--meter` describing meter model, location, supply setup and synchronization.
Capture them during matching paced replay and idle observations. The importer
integrates irregular samples in time and reports total and idle-subtracted
power/energy; the operator must verify trace alignment. PMIC rail values do
not replace this evidence. A record without external power remains incomplete.
INT8 windowed export remains distinct from a validated INT8 step deployment.

Generate device test data with the checkpoint to preserve its data protocol:

```bash
python -m deploy.make_testdata --ckpt RUN/best.pt --n 100
```

## 7. Presentation and remaining evidence

Paper report rows carry milliseconds, Hz, hop, delay and `measured=false` for
algorithmic budgets. Real-data waveform axes use seconds; SNR uses dB and
spectra use Hz. Training-epoch and architecture-sweep plots remain diagnostics.
The report command also generates `accuracy_latency.png`, using milliseconds
on its horizontal axis and seed SD error bars. Offline references are horizontal
accuracy lines and are never assigned a fabricated streaming latency.
Conclusions are scoped to single-channel EOG/EMG and the measured Pi 5
Cortex-A76 platform. A code test pass is not a successful experiment, a
three-seed result, a device measurement or a power measurement.

Fresh training, real-subject evaluation and Pi/external-meter runs are still
needed to establish the final empirical claims. Existing runs are untouched.
