# Streaming EEG Artifact Removal on Raspberry Pi 5

![Reproduce](https://github.com/ZhannoProject/project-v1/actions/workflows/reproduce.yml/badge.svg)

**Strict seven-rule methodology:** see [docs/methodology.md](docs/methodology.md).
`python -m experiment.paper plan --out runs/paper_256` creates a reviewable
three-seed study with unchanged-architecture output delays, a shared valid
interval and explicit publication checks. Planning does not start training.
Historical weights/results are retained as exploratory evidence.

Research code for measuring the accuracy, latency and energy cost of removing
EOG/EMG artifacts from a single EEG channel in a streaming system. The target
paper is an **IEEE Sensors Letters** study of the cost of causality, with
TCN+SE as the primary model and capacity, attention and knowledge distillation
as controls.

**Working paper title:** *Causality, Not Capacity: Latency–Accuracy Trade-offs
of Streaming EEG Artifact Removal on a Raspberry Pi 5.* This is the hypothesis
to test, not a conclusion established by the current repository.

The research plan follows the project roadmap dated **2026-10-28**. This README
separates the intended paper protocol from the code and evidence available
now. Historical results need re-evaluation under that protocol, particularly
after the change to fixed normalization.

- [Research question and evidence](#research-question-and-evidence)
- [Current implementation and remaining gaps](#current-implementation-and-remaining-gaps)
- [Paper protocol](#paper-protocol)
- [Setup and data](#setup-and-data)
- [Training and evaluation](#training-and-evaluation)
- [Deployment and measurements](#deployment-and-measurements)
- [Work plan and paper outputs](#work-plan-and-paper-outputs)
- [Repository map and checks](#repository-map-and-checks)

## Research question and evidence

How much denoising accuracy is lost when future samples are unavailable, and
how much of that loss can be recovered within a practical latency budget?
The intended contribution is a measured operating curve and design guidance
for a sensor system, rather than a ranking of denoising architectures.

| Comparison | Question | Role in the paper |
|---|---|---|
| Causal TCN vs causal TCN+SE | Does inexpensive channel gating help? | Primary baseline and model |
| Matched causal vs bidirectional models | What is the cost of withholding future information? | Offline reference ceiling |
| Causal models at several widths and capacities | Does capacity recover the gap? | Control for the main hypothesis |
| One controlled MLA configuration vs TCN+SE | Does attention justify its accuracy and device cost? | Supporting result; use a real KV cache |
| Plain vs distilled causal student | Can a teacher recover the gap without inference delay? | Supporting result, after implementation and positive controls |
| Fixed architecture with delayed output | What accuracy can additional latency buy? | `output_delay` and shared target crop implemented in paper mode; fresh training required |

MHA belongs in the paper only after retraining under the same protocol;
otherwise omit it from the paper comparison. MLA and KD are not assumed to be
improvements or primary contributions. A negative result requires evidence
and a defined experimental scope.

### Preliminary observations

The roadmap records the following **historical observations**, not final
results for the current fixed-normalization protocol:

| Observation recorded in the roadmap | Historical value | Required before use in the paper |
|---|---|---|
| Causal vs bidirectional gap | RRMSE_t 0.7104 vs 0.5950; gap 0.1154 | Matched configurations, three seeds and common valid interval |
| Spread across nine causal runs | RRMSE_t about 0.705–0.724 across roughly 70k–240k parameters | Remove architecture, width, loss and split confounds |
| Recovery from future context | About 72% at 16 samples, or 62.5 ms at 256 Hz | Rebuild the delayed-output experiment and re-evaluate steady state |
| KD student | RRMSE_t about 0.716, one seed | Static checks, positive controls and remaining seeds |

The 62.5 ms above is **additional future-context delay**, not total system
latency. It becomes 64 ms at 250 Hz. Do not mix these observations with new
runs or treat three seeds from different recipes as one mean and standard
deviation.

### Working hypothesis on EOG transfer (2026-11-05)

**Status: exploratory single-seed evidence; the transfer mechanism remains
unconfirmed.** Mixed EOG+EMG training may improve robustness to unfamiliar
ocular-artifact patterns by reducing dependence on the source dataset's EOG
features. The current results support investigating this hypothesis; they do
not establish that more irregular EMG necessarily generalizes to new EOG.

The comparison uses `tcn_se_eog_250hz_s54` and
`tcn_se_both_250hz_s54`. Their recorded training configurations differ only
in artifact type and output directory: both use seed 54, ChannelNorm,
200 epochs, MSE and epoch input normalization. Both resample EEGdenoiseNet
from 256 to 250 Hz before training; EEG-IO evaluation is at native 250 Hz
with identical settings for the same 20 subjects.

| Diagnostic, median | EOG-only training | BOTH training |
|---|---:|---:|
| Synthetic EOG RRMSE, lower is better | 0.466 | 0.539 |
| Clean-input RRMSE, lower is better | 0.372 | 0.461 |
| EEG-IO blink amplitude ratio, closer to 1 is preferred | 5.436 | 3.994 |
| EEG-IO background RMSE, µV, lower is better | 18.281 | 20.216 |
| EEG-IO blink count reduction, %, higher is better | 66.67 | 87.87 |

The synthetic checks use the same 510 held-out clean/EOG pairs and score
`[124,500)` in the historical epoch protocol. They are diagnostic results,
not the new paper-mode common-window evaluation. EOG-only performs better
on same-source synthetic EOG and changes clean EEG less; BOTH has better
real-data blink metrics, with 39 residual annotation-matched threshold peaks
versus 115. This ranking reversal is consistent with a source-to-real-data
distribution mismatch. Background differences on EEG-IO are not errors
against a clean target, which the dataset does not provide.

The datasets also differ in acquisition and construction: EEGdenoiseNet
extracts horizontal/vertical EOG segments for semi-synthetic mixing, whereas
EEG-IO records blink-contaminated frontal EEG. See the
[EEGdenoiseNet paper](https://doi.org/10.1088/1741-2552/ac2bf8) and
[EEG-IO dataset description](https://gnan.ece.gatech.edu/eeg-eyeblinks/).

**Competing explanation:** the current BOTH mixer adds each artifact at its
own requested SNR, so total noise is usually stronger than in EOG-only
training. A stronger suppression tendency can also improve blink metrics
while attenuating background. Diversity and total-noise-strength effects
have not yet been separated; the shared resampling step alone does not
explain the difference between these runs.

Next, compare EOG-only, EOG+EMG and EOG-only with morphology augmentation
(polarity, width, amplitude and timing), while matching the **final total
SNR**, data splits, model settings and training budget. Report clean-input
distortion alongside real blink/background metrics, and repeat across seeds
before making a general or publication-level claim.

Local evidence: `eegio_tests/eegio_20261105_160709.json` and the seed-54 BOTH
row in `eegio_tests/eegio_20261104_162544.json`. The paired synthetic probe
and detailed analysis are retained outside this Git repository under
`../diagnostics/eegio_20261105_eog/`.

### Experiment note (实验手记) on SE with matched ChannelNorm (2026-11-06)

This entry is an experiment note for two exploratory 250 Hz BOTH comparisons.
Within each seed, the plain TCN and TCN+SE use ChannelNorm and the same
training data protocol. The evaluation uses identical EEG-IO v2 settings
and scored events for the same 20 subjects.

| Seed | Blink amplitude ratio, median | Subjects with lower ratio | Background RMSE, median (µV) | Residual matched peaks, total | Paired median change in blink power suppression (dB) |
|---|---:|---:|---:|---:|---:|
| 54 | 4.34 → 3.99 | 15/20 | 20.59 → 20.22 | 36 → 39 | −0.21 |
| 67 | 4.69 → 3.45 | 18/20 | 20.45 → 20.05 | 29 → 31 | +0.26 |

Arrows show plain TCN → TCN+SE. The suppression column is the median of
per-subject differences, computed as TCN+SE minus plain TCN on the same
blink intervals. Positive values indicate greater power suppression.

SE lowered the blink amplitude ratio and background RMSE in both seed
summaries. The ratio reflects blink amplitude relative to the surrounding
baseline, so better baseline preservation can also lower it. The paired
power suppression change was small and differed in direction across the
two seeds. The observation concerns the relative blink-to-background
amplitude in these runs, with mixed evidence for additional absolute
suppression of blink-interval power.

The local plain TCN results are `eegio_tests/eegio_20261105_164435.json`
for seed 54 and `eegio_tests/eegio_20261106_123431.json` for seed 67.
The corresponding SE rows are in `eegio_tests/eegio_20261104_162544.json`.

## Current implementation and remaining gaps

| Area | Available in the repository | Remaining publication requirement |
|---|---|---|
| Models | TCN, causal cumulative-mean SE, MHA/MLA, width/depth/rank controls, bidirectional teachers | Select a small matched comparison; complete missing seed repeats |
| Training | AdamW; all new checkpoints selected by pure GT MSE; shared paper-mode crop; architecture metadata | Complete the planned 200-epoch three-seed runs |
| Data provenance | Disjoint epoch pools for a fixed seed; checkpoint-derived evaluation settings; teacher/student protocol checks | Re-score selected historical runs and retain their provenance |
| Normalization | New CLI runs use a training-calibrated fixed scale shared with evaluation and deployment | Retrain and validate accuracy under this protocol; legacy epoch normalization is offline |
| Lookahead | Paper `output_delay` shifts targets without changing the causal graph; legacy `tcn_se_la{N}` retained | Train and measure the new delayed-output family |
| Streaming | Stateful TCN/SE and MHA/MLA inference; ONNX step graph; equivalence and causality tests | Verify the entire raw-input chain at the paper operating point, including output delay |
| Real recordings | EEG-IO reader, causal filtering, annotation-gated events and subject-level metrics | Finish native-250 Hz comparisons, metric alignment and hardware-in-the-loop replay |
| Device measurements | ONNX FP32, static INT8 for windowed graphs, latency traces, preflight and paced replay | Validate the intended INT8 streaming path; complete power and multi-channel measurements |

The strict paper mode now implements label-shift/output delay and a shared
valid target interval. Paper presets omit TCN normalization and use finite
causal SE pooling. The paper architecture requires new training.
The legacy layer-allocation lookahead ladder remains exploratory.

The default convolution-only TCN spans 125 samples. The paper SE baseline adds
five 16-sample trailing pools, giving 200 samples of total context. Study plans
compose every layer's actual past/future dependency, include causal-resampler
startup, and derive one crop for all compared architectures and delays. They
reject empty/undersized valid intervals. Cumulative/global SE and time-axis
BatchNorm remain available only through the historical/exploratory paths.

The strict protocol, commands and remaining empirical requirements are detailed
in [the methodology guide](docs/methodology.md). INT8 stateful deployment,
final-weight device measurements and external-meter power evidence remain
separate acceptance requirements.

## Paper protocol

This is the protocol to satisfy before a result enters the paper. The
commands below exercise existing infrastructure; they do not fill the
implementation gaps above.

| Item | Intended protocol |
|---|---|
| Training | AdamW, 200 epochs, three seeds per included configuration; current repeat seeds are 42, 54 and 67 |
| Selection | Best validation **ground-truth MSE**; for KD, never select using teacher agreement or total KD loss |
| Test separation | Independent held-out test data; seed, artifact pools, split ratios and preprocessing must match the run |
| Aggregation | Report mean ± sample SD over seeds; distinguish run variability from repeated probe draws |
| Causality | Causal filters and online preprocessing; no full-time-axis normalization in the online path; causal SE pooling |
| Parity | Match batch and streamed/hop outputs after correct alignment on the common valid interval; target max absolute difference below `1e-4` |
| Primary operating point | Hop `H = 4`; report the sampling rate with every latency value |
| Semi-synthetic comparisons | Pair matching segments within each split; retain recording/subject grouping where applicable |
| Real-data statistics | Pair by subject; do not count individual blinks as independent subjects |
| Attention vs SE | Predefine the TOST equivalence margin; non-significance alone does not establish equivalence |
| Hardware | Measure on Pi 5; fix governor, affinity and threads; record temperature, throttling and other load |
| Energy | Use an external power meter; report idle baseline and total and incremental power |

For MSE, an ideal population predictor uses the conditional mean available
from its input history. The tower-property argument helps explain why a
future-aware teacher cannot supply future measurements at inference time. It
is not proof that KD cannot improve finite-data training: complete the GT
teacher and GT-scarcity positive controls before interpreting a negative KD
result.

### Delayed output and the common valid interval

The planned model receives samples through time `t' + ell` and estimates the
clean target at `t'`. Implement this by shifting the training targets and
emitting the result `ell` samples later, while keeping the causal architecture
and parameter count unchanged.

Using the roadmap's hop-budget convention:

$$
T_{\mathrm{hop}} = \frac{H}{f_s}, \qquad
L_{\mathrm{alg}} = \frac{H + \ell}{f_s}.
$$

| Future samples `ell` | Additional delay at 256 Hz | `L_alg` at 256 Hz, H=4 | `L_alg` at 250 Hz, H=4 |
|---|---|---|---|
| 0 | 0 ms | 15.625 ms | 16 ms |
| 8 | 31.25 ms | 46.875 ms | 48 ms |
| 16 | 62.5 ms | 78.125 ms | 80 ms |
| 32 | 125 ms | 140.625 ms | 144 ms |

These are algorithmic budgets, not measured inference or end-to-end latency.
Account for preprocessing, acquisition/I/O and inference separately, and use
actual timestamps for end-to-end percentiles. Do not add component p99 values
and call the sum a measured end-to-end p99.

For a chosen finite history length `R`, segment length `N` and the largest
lookahead in a comparison `ell_max`, the planned common target interval is

$$
t' \in [R - 1,\; N - \ell_{\max}).
$$

Use the same target positions for every model, verify that the interval is
nonempty and exclude padded/warm-up targets. Historical/exploratory runs without interval metadata retain full-epoch
scoring. Paper-mode training, validation and evaluation apply their saved
common interval and output-delay alignment. With an unchanged delayed-output architecture, inference work is
the same across `ell`; measure the implementation and confirm it with final
weights rather than assuming the device behavior.

## Setup and data

Run commands from the repository root, the directory containing `trainer/`,
`models/` and `README.md`.

### Host environment

Python 3.11+ is supported by the current code; CI uses Python 3.11. CPU PyTorch
is sufficient for functional checks. Exact repeatability is easiest on CPU;
MPS should not be described as bit-deterministic.

```bash
git clone https://github.com/ZhannoProject/project-v1.git
cd project-v1
python -m venv env
source env/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

### EEGdenoiseNet

Place the original arrays from [EEGdenoiseNet](https://github.com/ncclabsustech/EEGdenoiseNet)
in this directory; datasets and model weights are not distributed with this
repository:

```text
data/EEGdenoisenet/data/
├── EEG_all_epochs.npy
├── EOG_all_epochs.npy
└── EMG_all_epochs.npy
```

The native epochs are 512 samples at 256 Hz. A run's `master_seed` determines
its clean and artifact train/validation/test pools as well as its random
initialization and sampling. A test set from another seed can contain the
model's training epochs. Evaluation restores the checkpoint's data protocol
and rejects conflicting explicit seeds/artifacts. For legacy weights without
recorded provenance, supply the original seed and training configuration;
never infer the split from a convenient default.

In `BOTH` mode, each artifact is scaled against the clean signal. `snr_db`
records the requested per-artifact value, or their mean, whereas
`snr_mixture_db` measures the realized mixture. Equal-power uncorrelated
artifacts produce a mixture about 3 dB below the per-artifact SNR. Use the
realized mixture SNR when interpreting mixture difficulty and label existing
requested-SNR tables accurately.

The loader retains `sampling="legacy"` for historical compatibility. Its
validation/test `BOTH` artifacts share a grid SNR and it truncates some clean
pools. Set `--sampling balanced` or `sampling: balanced` to select the alternate
mode through the shared configuration resolver. Treat a sampling-mode change
as a new data protocol; the resolved mode is saved with the run.

### Normalization and rate compatibility

New `python -m trainer` runs default to `normalization: fixed`. The scale is
calibrated from the training clean EEG pool, recorded as
`normalization_scale` and reused for validation, testing and inference.
`--normalization_scale` can instead provide a previously calibrated value.

Legacy `epoch` normalization computes a standard deviation over the input
window and is an **offline** protocol. It remains available for reproducing
old results. Online interfaces require the saved fixed scale or an explicit
prior calibration (`--sigma` in the Pi runtime); they no longer infer a scale
from the future of the input stream. Prior calibration with an epoch-trained
model still needs an accuracy check. Switching a historical model to the new
training protocol requires retraining, not just editing its metadata.

`--target_fs 250` creates 500-sample training epochs and records the rate.
`resample_poly` is currently used for offline training-data conversion and
legacy EEG-IO rate conversion. It is not a validated causal live resampler.
For the intended EEG-IO deployment, train at 250 Hz and keep the live 250 Hz
signal at its native rate.

### EEG-IO

Obtain the real recordings through the
[EEG EyeBlinks dataset page](https://gnan.ece.gatech.edu/eeg-eyeblinks)
and keep the original annotations and corrupt-interval information. The reader
finds `data/EEG-IO/` or `data/EEG-EyeBlinks/EEG-IO/`. Recordings are 250 Hz;
Fp1/Fp2 are evaluated as individual channels, not as a trained multi-channel
model. There is no clean waveform reference.

## Training and evaluation

### Matched causal baselines

The following trains the current TCN and TCN+SE baselines with MSE and fixed
normalization. Use distinct output directories to preserve historical runs.
Common-window masking and the final architecture audit are still pending.

```bash
for seed in 42 54 67; do
  python -m trainer --config configs/tcn_both.yaml \
    --master_seed "$seed" --loss_fn mse --max_epochs 200 \
    --normalization fixed --save_dir "runs/paper_tcn_256_s${seed}"
  python -m trainer --config configs/tcn_se_both.yaml \
    --master_seed "$seed" --loss_fn mse --max_epochs 200 \
    --normalization fixed --save_dir "runs/paper_se_256_s${seed}"
done
```

For the 250 Hz SE baseline, start from the rate-specific preset and repeat
for the same three seeds:

```bash
python -m trainer --config configs/tcn_se_both_250hz.yaml \
  --master_seed 42 --loss_fn mse --max_epochs 200 \
  --normalization fixed --save_dir runs/paper_se_250_s42
```

For capacity controls, vary a declared width while preserving the other
settings. The roadmap's roughly 70k-parameter SE control corresponds to the
current `--hidden_channels 48` model (69,450 parameters); the 64-channel preset
has 122,548 parameters. Keep those as distinct configurations. For MLA, use
one explicit `BOTH`/MSE/200-epoch configuration and
record its width, heads, layers, KV rank, optimizer groups and parameter count.
Older EOG attention presets often use different losses and epoch counts; they
are not the paper's common recipe. Use `history.json` and checkpoint
architecture metadata to verify the built model, rather than trusting the
preset's filename. Do not derive a plain TCN by overriding a SE preset while
leaving `se_reduction` enabled.

The optional staged trainer is not needed for these baselines. Its current
implementation freezes the **attention decoder** first; some historical preset
comments describe the opposite strategy.

### Shared configuration and result protection

[`experiment/config.py`](experiment/config.py) defines model, data, training
and runtime settings and validates the resolved configuration. Named `cli`,
`gui` and `api` profiles preserve the existing launch defaults explicitly;
YAML values and explicit CLI flags override the selected profile. Use
`--profile` to select one deliberately instead of assuming that the launchers
have identical defaults.

When `save_dir` is omitted, the CLI and GUI derive
`runs/<variant>_<artifact>[_kd]_<rate>hz[_fixed][_s<seed>]` from the effective training
configuration. An omitted `target_fs` uses the native 256 Hz rate. For seed 67,
the same TCN-SE model can produce `tcn_se_both_250hz_s67` and
`tcn_se_both_256hz_s67` under epoch input scaling. Fixed input scaling adds
`_fixed`, for example `tcn_se_both_250hz_fixed_s67`, so it can coexist with
the epoch-scale run at the same rate and seed. An explicit `save_dir` is
used as written and must identify a separate folder when changing protocols.

Each training session holds an OS lock on its output directory. Concurrent
writers are rejected even with `--overwrite`; the GUI also checks active
outputs before launching a child and freezes its configuration. Reusing an
existing run requires `--overwrite`, including repeats with the same seed.
Existing checkpoint/history/configuration files are moved into
`archive/<new-run-id>/` before replacement. Failed runs with a saved
configuration also require explicit overwrite or a fresh directory.

Checkpoints, history and `resolved_config.json` are written atomically and
carry a run ID and format version. The snapshot includes the actual fitted
normalization scale and constructor/training settings. It can be supplied to
`--config` to replay a recipe; overwrite permission is reset on replay, so
choose a fresh `--save_dir` or explicitly pass `--overwrite` again.
[`experiment/checkpoints.py`](experiment/checkpoints.py) provides the shared
model reader for evaluation, KD, EEG-IO, GUI inspection and export. Legacy
unversioned checkpoints are adapted on read without rewriting their files;
unknown future versions are refused.

### Teacher and distilled student

This example uses a matched SE teacher/student at 256 Hz and the same seed.
Repeat only after completing the KD validation workstream.

```bash
python -m trainer --config configs/tcn_se_both.yaml --variant tcn_se_bi \
  --master_seed 42 --loss_fn mse --max_epochs 200 \
  --normalization fixed --save_dir runs/paper_se_bi_256_s42

python -m trainer --config configs/tcn_se_both.yaml \
  --master_seed 42 --loss_fn mse --max_epochs 200 --normalization fixed \
  --teacher_ckpt runs/paper_se_bi_256_s42/best.pt \
  --kd_alpha 0.5 --kd_feature_weight 0.1 --kd_feature_layers backbone \
  --init_from_teacher --save_dir runs/paper_se_kd_256_s42
```

Teacher loading is strict, teachers are frozen, and the student is selected
against ground truth. Before copying weights, the trainer checks the teacher
and student split seed, artifact pools, split ratios, rate and normalization.
A legacy teacher without data provenance needs its original
`--teacher_config`; it cannot silently teach a student on a different split.
The GT-teacher and GT-scarcity positive controls remain research tasks, not
completed checks inferred from successful training.

### Semi-synthetic evaluation

After training, these commands restore each checkpoint's data settings:

```bash
python eval.py --mode snr \
  --ckpt runs/paper_se_256_s42/best.pt \
  --data_dir data/EEGdenoisenet/data

python eval.py --mode compare_snr \
  --compare_runs runs/paper_tcn_256_s42:TCN runs/paper_se_256_s42:TCN+SE \
  --data_dir data/EEGdenoisenet/data --save_dir figures/paper_comparison

python eval.py --mode switch \
  --ckpt runs/paper_se_256_s42/best.pt \
  --data_dir data/EEGdenoisenet/data --save_dir figures/paper_switch
```

Available modes are `all`, `curves`, `snr`, `waveform`, `attention`, `snr_plot`,
`compare_curves`, `compare_snr`, `switch` and `compare_switch`. The switch probe
uses `BOTH`-trained models and includes a matched same-artifact splice control.
Probe seeds are repeated perturbations of a fixed split, not independent
training seeds. The existing metrics are RRMSE_t, RRMSE_f and Pearson CC;
GT MSE is the planned checkpoint-selection criterion.

Comparisons across different training seeds use each model's own held-out
split. Segment-level pairing is valid only where the underlying test segments
actually match. Current CLI evaluation is useful for diagnostics but still
needs the common steady-state crop for paper figures.

### Real-data evaluation

An exploratory native-rate evaluation of a model trained at 250 Hz is:

```bash
python eegio_eval.py --data_dir data/EEG-IO \
  --runs runs/paper_se_250_s42 --native --hop 4
```

The code uses forward `sosfilt` preprocessing by default and can retain a
zero-phase offline control. Native-rate training avoids online resampling;
normalization must match the saved run. The GUI, CLI and paper evaluator use
the `eegio_v2` measurement protocol for new results.

In the local **EEG-IO test**, enable **Full-context offline comparison** (CLI
`--offline`) to compare checkpoints on shared overlapping two-second
windows (500 samples at 250 Hz). The models use the same selected filter,
either causal or zero-phase. Each uses its saved normalization and target
delay. Only the shared interior predictions enter scoring, and the result
records the window and margins. Regular replay defaults to hop 4.

In **Lookahead**, choose **Dataset → EEG-IO** to plot blink amplitude ratio,
background RMSE, or blink count reduction on the same ladder. Choose a
**Model type** and matched **Training setup**, then **Run all seeds & rungs**.
One comparison includes every completed matching seed and available rung,
including causal and offline references. Coverage is shown before running;
missing checkpoints are not imputed, and duplicate seed/rung checkpoints
are represented once using deterministic name order, not their scores.
Different rates, normalization settings and training recipes stay separate.
All models share an offline window and scoring interval. Blink-count
reference thresholds and paired tests use each seed's own causal baseline.
Plots summarize participant medians within each seed, then show the mean
and sample SD across seeds with the actual coverage of each rung. Saved
single-seed tests remain readable and are labeled separately. Full results
remain available under **Runs → EEG-IO test**.

In the GUI, open **EEG-IO test**, set **Run on** to **Raspberry Pi 5
(SSH)**, and choose causal models trained at 250 Hz. The GUI exports each
checkpoint and copies the selected recording channel to the board over SSH.
The Pi applies forward filters and runs ONNX. **Replay speed** defaults to
**Original speed**, which follows the recording clock and measures missed
hop deadlines. **Fast replay** processes successive hops immediately using
the same filters, normalization and model. Both modes use EEG-IO v2 scoring
and save the replay mode, per-subject times and Pi conditions under
**Runs → EEG-IO test**.

For models with a verified finite receptive field shorter than the EEG-IO
window, fast replay batches the shorter ONNX inputs after the first window.
The scale for each hop still uses the original window or the checkpoint's
fixed scale. Models that need the full context run the full ONNX window.

In regular replay, historical checkpoints use the 512-sample EEG-IO window
and their saved normalization mode. Paper-protocol checkpoints use stateful inference with
the saved fixed scale, hop 4 and the declared output delay. This is an
exploratory replay measurement on prerecorded EEG.

| Metric | EEG-IO scoring rule |
|---|---|
| Blink amplitude ratio | Average output epochs aligned to annotated blink centers. Baseline-center the average using the 1.5–2 s flanks and divide its mean absolute amplitude within ±0.5 s by that in the flanks. The reference ratio is 1. [Robbins et al., §II-F](https://ieeexplore.ieee.org/ielx7/7333/9090036/09047940.pdf) |
| Background RMSE (µV) | Root mean square change from the preprocessed input outside the blink exclusion windows. Lower values indicate less change to the observed background. [EEGOAR-Net](https://uvadoc.uva.es/bitstream/handle/10324/78224/Calibration-free-ocular-artifact.pdf?isAllowed=y&sequence=1) |
| Blink count reduction (%) | `100 × (input count − output count) / input count`. For each trace, count absolute peaks above six times its mean absolute amplitude with at least 250 ms peak separation. Match peaks one-to-one to EEG-IO annotations within ±250 ms and report both counts. [E-ASR, Eq. 7–8](https://pmc.ncbi.nlm.nih.gov/articles/PMC11510769/) |

**Metric source papers:**

1. **Blink amplitude ratio:** K. A. Robbins et al. (2020),
   [*How Sensitive Are EEG Results to Preprocessing Methods: A Benchmarking Study*](https://doi.org/10.1109/TNSRE.2020.2980223).
   *IEEE Transactions on Neural Systems and Rehabilitation Engineering*,
   28(5), 1081–1090, §II-F.
2. **Background RMSE:** D. Marcos-Martínez et al. (2025),
   [*Calibration-free Ocular artifact reduction in EEG signals using a montage-independent deep learning model*](https://doi.org/10.1016/j.bspc.2025.108147)
   (EEGOAR-Net). *Biomedical Signal Processing and Control*, 110, 108147.
   Uses input/output RMSE in resting EEG to assess signal preservation.
3. **Blink count reduction:** D. Hazarika et al. (2024),
   [*Dynamical Embedding of Single-Channel Electroencephalogram for Artifact Subspace Reconstruction*](https://doi.org/10.3390/s24206734)
   (E-ASR). *Sensors*, 24(20), 6734, §3, Eq. 7–8.

These measures are adapted to EEG-IO: BAR uses its annotated blink centers,
background RMSE uses annotation-excluded background, and blink counting adds
one-to-one annotation matching. The exclusion and validity rules below are
part of this project's evaluation protocol.

EEG-IO v2 excludes the first 3 s, corrupt samples and invalid model targets.
All annotations protect the background window, including soft blinks, with at
least 0.5 s of exclusion on either side. The soft-blink setting selects the
events used in the amplitude ratio and peak counts. Each amplitude-ratio epoch
spans ±2 s and requires valid samples and flanks clear of neighboring blink
annotations. The result includes the number of
aligned epochs and detected input/output peaks for each subject. The paired
amplitude-ratio test uses distance from 1.

The **Model parity / causality** section records a deterministic normalized
input probe per checkpoint. Batch/step parity checks the maximum absolute
difference on valid aligned targets against `1e-4`. The causality check
perturbs future inputs and checks the already-due prefix against `1e-6`,
accounting for the declared output delay. Noncausal variants can be used as
controls and have no incremental parity result. The model probe covers the
graph in normalized units. The preprocessing causality probe and raw-input
hardware measurements cover their respective stages separately.

New results also save the legacy suppression dB, fidelity dB and residual
rate. Switch **Measurement** to **Legacy** in the GUI, or pass `--legacy` to
print those columns in the CLI. Saved comparisons retain their original
metric protocol, including any previously recorded Lead values.
The `ell=0` vs `ell=16` delayed-output comparison and VV/VR
hardware replay are not yet complete.

The local GUI provides training, evaluation, lookahead/RF exploration, KD
audits, EEG-IO inspection and an **EEG channels** load experiment:

```bash
python -m gui
```

For a portable Windows 11 ARM64 application, see the
[Windows build and migration guide](docs/windows-arm64.md).
The package includes a native ARM64 launcher, Python and .NET runtime,
plus optional model records and training data.

In **EEG channels**, select one causal model and replay 1, 2, 8 and/or 16
independent copies on this computer (diagnostic) or over SSH on a Pi 5.
The default source is EEG-IO, using Fp1 from the first available subject.
Choose a subject and electrode to replay a continuous unflagged segment.
Causal filtering is timed. If the model uses another rate, the recording
is resampled during setup.
The tab previews the plan, exports a stateful model when needed, saves each
repeat, and plots measured hop p99 against the selected hop deadline. Hop
defaults to four samples. See the
[measurement workflow](docs/methodology.md#6-hardware-evidence) for input,
calibration and external-meter requirements.

## Deployment and measurements

### Distinguish the inference paths

| Path | What runs | What its measurements establish |
|---|---|---|
| PyTorch incremental | TCN ring buffers, causal SE state, MHA KV or MLA latent cache | Architectural/state comparisons on the host; not a Pi ORT timing |
| ONNX windowed | Recomputes a full window and retains the newest hop outputs | Reference implementation; no persistent KV-cache claim |
| ONNX step graph | One sample plus explicit state in/out | Stateful Pi runtime for strictly causal models |
| INT8 windowed graph | Static PTQ of the exported FP32 graph | Quantized windowed accuracy and cost; not automatically INT8 stateful inference |
| Delayed-output stream | Unchanged causal model plus saved target alignment/output delay | Implemented paper path; fresh training and hardware validation needed |

Latency can be profiled with random weights before training. The existing
host benchmark is useful for checking the cache and comparing mechanisms:

```bash
mkdir -p deploy/artifacts
python -m deploy.bench_incremental --variants tcn_se,tcn_mla \
  --hops 4,32 --json deploy/artifacts/host_incremental.json
```

Validate the final weights and full raw-input chain afterwards. Do not
substitute host timings or analytical cache sizes for device measurements.

### Export on the host

The examples use the new fixed-normalized 250 Hz run above. Export its
windowed FP32 graph and its strictly causal step graph separately:

```bash
python -m deploy.export_onnx --ckpt runs/paper_se_250_s42/best.pt \
  --name paper_se_250_s42
python -m deploy.export_onnx --ckpt runs/paper_se_250_s42/best.pt \
  --name paper_se_250_s42 --incremental

python -m deploy.make_testdata --data_dir data/EEGdenoisenet/data \
  --artifact BOTH --master_seed 42 --target_fs 250 --n 100

python -m deploy.quantize_int8 \
  --onnx deploy/artifacts/paper_se_250_s42.onnx \
  --data_dir data/EEGdenoisenet/data
```

Keep each `.onnx` file with its `.json` sidecar. They describe its I/O,
normalization, sampling rate and provenance; INT8 export preserves that
protocol and calibrates on training data only. Synthetic calibration is for
functional checks, not deployment accuracy. Attention graphs may require
`--conv_only` and a compatible calibrator. The existing static INT8 path
should not be described as a completed quantized step-graph deployment.

### Run on Raspberry Pi 5

Install `deploy/requirements-raspi.txt` in a Pi environment, then copy the
model/sidecar, test data and runtime files from `deploy/` to the board.
PyTorch is not required for the ORT runtime. These examples assume those
files are together in the current Pi directory:

```bash
python preflight.py --threads 2 --cpus 2-3 --require-locked

python raspi_infer.py paper_se_250_s42.onnx \
  --testdata BOTH_test100_250hz.npz --hop 4 \
  --threads 2 --cpus 2-3 --require-locked \
  --latency --latency_out se_fp32_h4

python raspi_infer.py paper_se_250_s42_step.onnx --incremental \
  --testdata BOTH_test100_250hz.npz \
  --threads 2 --cpus 2-3 --require-locked \
  --latency --latency_out se_step
```

A step graph emits once per sample; it does not gain an `H=4` buffering contract
just because a command includes `--hop 4`. If the paper groups four samples,
implement and measure that acquisition/output schedule explicitly. Independent
benchmark epochs reset state and do not replace continuous EEG-IO replay.

The windowed runtime supports paced replay via `--input <signal.npy> --stream
--realtime --hop 4`. Use a continuous recording for hardware-in-the-loop
validation, with matching rate and fixed calibration. Compare the replay with
the corresponding offline streaming outputs and retain the timestamps.

### Table II measurement contract

| Operating conditions to measure | Required outputs |
|---|---|
| Idle baseline | Total idle power and temperature, using the same board setup |
| One channel FP32 and INT8 | p50/p99 latency (ms), deadline misses, duty cycle (%), process RAM (MiB), total power `P` and idle increment `delta P` (mW), temperature (°C) |
| INT8 at 2, 8 and 16 channels | Same outputs, with the channel scheduling/batching and shared overhead stated |
| MLA with an actual incremental cache | Same outputs plus cache/state size; include window recomputation only as a reference row |

Use an external meter such as the roadmap's KM003C/FNB58 setup and a fixed
5 V supply arrangement. Record meter location, idle subtraction, measurement
duration, governor, CPU affinity, thread count, cooling, temperature and
`get_throttled` before/after each run. PMIC rail breakdown is supplementary.
For `C` channels, state whether energy/channel/s means `P/C` or `delta P/C`;
report both when possible, with the observation duration. A duty-cycle trace
alone is not a power measurement. Multi-channel rows mean replicated
single-channel processing unless a separate multi-channel model is trained.
For hop-sized processing, duty cycle is processing time divided by `H/fs`;
include the wrapper work and use the actual schedule for a sample-wise graph.

## Work plan and paper outputs

The checklist records **publication gates**, not whether related helper code
exists. A workstream is not complete just because its GUI tab is present.

| Workstream | Next acceptance gate | Tier |
|---|---|---|
| A. Data and experiment hygiene | Common valid-window re-evaluation; three seeds for the narrow SE, causal TCN and selected MLA; matched MHA or exclusion; artifact/SNR breakdown | Minimum |
| B. KD validation | Strict/frozen teacher; reproduce teacher reference on identical student inputs; GT selection and edge mask; distance/cross-term diagnosis; GT-teacher and GT-scarcity controls; remaining seeds | Minimum |
| C. Fair attention cost | Pi comparison of incremental MLA and incremental TCN+SE, with real state/cache and matched scheduling | Minimum |
| D. EEG-IO validation | Three 250 Hz seeds each for `ell=0` and delayed `ell=16`, one offline bidirectional reference; SNR risk check; causal preprocessing; subject-paired metrics | Minimum |
| E. Pi deployment | INT8 streaming path; controlled latency/RAM/power measurements; VV/VR paced replay and parity | Minimum |
| F. Receptive field | Measure ERF; truncate dilation/history to planned P in {32, 64, 128, 187}, bound SE pooling consistently; remeasure a reduced model only if accuracy saturates early | Standard |
| G. External baselines | EEGDnet reported and reproduced accuracy distinguished; Pi whole-window latency; optional Xing–Casson DAE on EEG-IO | Standard / optional |

For the single-channel study, use a causal linear-filter baseline. ASR needs
a multi-channel subspace and is not the selected control for EEGdenoiseNet or
the two-channel EEG-IO setting. In the roadmap's idealized `R=187` window,
`ell=93` is the centered case and `ell=186` is fully anti-causal; those labels
must not be assigned to the current cumulative-SE ladder without checking
its actual dependency structure.

### Planned figures and tables

| Output | Intended evidence |
|---|---|
| Fig. 1 | System/replay pipeline and model in one diagram, with delay components identified |
| Fig. 2 | Accuracy vs parameter count, then accuracy vs `L_alg` on a logarithmic axis; KD at its unchanged inference-delay coordinate |
| Fig. 3 | Subject-averaged blink-aligned EEG-IO waveforms and a hop-latency histogram with the `T_hop` deadline |
| Table I | Offline reference, causal variants, KD, delayed `ell=8/16/32`, INT8 and external baselines; real-data metrics only where measured |
| Table II | The device operating grid, measured latency, RAM, power, energy and temperature |

The intended short paper gives the largest space to results: introduction,
methods, results and a conclusion with explicit limits. Its three intended
contributions are the measured streaming system, the controlled accuracy–delay
curve, and real-sensor/HIL validation. Restrict conclusions to the tested
single-channel EOG/EMG setting and Pi 5 Cortex-A76 platform.
The roadmap budgets four pages, at least one column for references and about
2.2 columns for results; these are manuscript planning assumptions, not a
verification of current journal submission requirements.

**Minimum** completes A–E and the main lookahead figure. **Standard** adds a
useful RF reduction, EEGDnet reproduction, fuller artifact/power breakdowns
and a documented public release. **Premium** may add real-data pseudo-label
KD, ERP CORE, QAT, synchronized GPIO/power traces and a physical frontend.
Use QAT only if PTQ leaves a material accuracy problem, particularly at high
frequencies. The roadmap's 4–5 week estimate is a planning assumption, not a
claim of remaining runtime or submission readiness. Recheck the literature
before submission; this README does not perform or claim that search.

## Repository map and checks

| Path | Purpose |
|---|---|
| [`configs/`](configs/) | Existing YAML recipes; verify their losses, rates and architectures before use |
| [`data/Dataloader.py`](data/Dataloader.py), [`data/protocol.py`](data/protocol.py) | Semi-synthetic data, normalization and split/preprocessing guards |
| [`data/EEGIO.py`](data/EEGIO.py), [`eegio_eval.py`](eegio_eval.py) | Real recordings, annotations, preprocessing and subject comparisons |
| [`models/`](models/) | TCN, SE, attention, exploratory lookahead/RF families and stateful inference |
| [`trainer/`](trainer/) | AdamW, losses, KD, validation and checkpoint/history recording |
| [`experiment/`](experiment/) | Shared configuration profiles, output leases, atomic writes and versioned checkpoint loading |
| [`eval.py`](eval.py), [`metrics.py`](metrics.py) | Diagnostics and semi-synthetic metrics; paper crop still pending |
| [`deploy/`](deploy/) | ONNX/INT8 export, step graph, Pi runtime, test data and environment checks |
| [`gui/`](gui/) | Local controls for the existing research tools |
| [`tests/`](tests/), [`reproduce.py`](reproduce.py) | Regression tests and functional reproduction checks |
| [`docs/archive/research-notes-before-roadmap.md`](docs/archive/research-notes-before-roadmap.md) | Previous README with detailed historical GUI notes, experiments and measurements; not final paper evidence |

Start with functional checks and the data/causality regressions:

```bash
python reproduce.py --quick
python -m unittest discover -s tests -p 'test_p1_protocol.py'
python -m unittest discover -s tests -p 'test_experiment_services.py'
python -m unittest discover -s tests -p 'test_distillation.py'
python -m unittest discover -s tests -p 'test_incremental.py'
python -m unittest discover -s tests -p 'test_step_graph.py'
```

`python reproduce.py` also runs short training/evaluation checks, using
synthetic arrays if benchmark data is absent. Synthetic fixtures establish
code behavior, not denoising accuracy on real data. Full GUI tests may need
local-server permission; hardware tests need the corresponding device. Set
`MPLBACKEND=Agg` for headless figure tests. Passing these checks does not
complete the publication gates or substitute for device measurements.

Checkpoint `best.pt`/`last.pt` and `history.json` carry the architecture and
recorded data/training protocol. Preserve them with evaluation outputs and
device traces. Do not overwrite historical runs or edit their metadata to
make incompatible experiments appear comparable.

TurboQuant, deeper transformers, alternate skip connections and other
exploratory branches remain available, but are outside the minimum paper
unless they answer a stated question with matched evidence. Parallel branch
concatenation does not by itself guarantee that attention is used.

## Usage terms

This repository is part of an MPhil research project. No open-source license
is declared here; contact the authors for usage terms. Dataset access and
redistribution follow the respective providers' terms. A public code release
and finalized usage terms remain part of the release work.
