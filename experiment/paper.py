"""Plan three-seed experiments, evaluate held-out segments and report mean ± SD.

python -m experiment.paper plan --out runs/paper_plan
python -m experiment.paper train --plan runs/paper_plan/plan.json
python -m experiment.paper evaluate --plan runs/paper_plan/plan.json
python -m experiment.paper report --plan runs/paper_plan/plan.json
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from experiment.storage import atomic_json


SEEDS = (42, 54, 67)
METRICS = ("gt_mse", "rrmse_t", "rrmse_s", "acc")
SCOPE = "Single-channel EOG/EMG artifact removal; hardware conclusions limited to measured Pi 5 Cortex-A76"


def make_plan(out, variants=("tcn", "tcn_se", "tcn_se_bi"), delays=(0, 8, 16, 32), fs=256., data_dir=None):
    from experiment.config import resolve_config
    from experiment.methodology import context, validate_model
    from models.variants import build_model_from_config, variant_is_causal
    from data.resampling import resampling_info
    if not variants or len(set(variants)) != len(variants) or not delays or len(set(delays)) != len(delays):
        raise ValueError("Supply nonempty unique variants and delays")
    if any(type(d) is not int or d < 0 for d in delays):
        raise ValueError("Output delays must be nonnegative integers")
    out = Path(out).resolve()
    jobs, built, frozen = [], [], []
    for variant in variants:
        cfg = resolve_config(dict(profile="paper", variant=variant,
                                  target_fs=None if fs == 256 else fs))
        if data_dir:
            cfg["data_dir"] = str(Path(data_dir).resolve())
        model = build_model_from_config(cfg)
        past, future = context(model)
        for delay in (delays if variant_is_causal(variant) else (0,)):
            item = {**cfg, "output_delay": delay}
            built.append((item, model, past, future))
    warmup = resampling_info(256., fs)["warmup_samples"]
    history = max(past for _, _, past, _ in built) + warmup + 1
    future = max(future + cfg["output_delay"] for cfg, _, _, future in built)
    n = (512 * resampling_info(256., fs)["up"] + resampling_info(256., fs)["down"] - 1) // resampling_info(256., fs)["down"]
    for cfg, model, _, _ in built:
        for seed in SEEDS:
            label = f"{cfg['variant']}_delay{cfg['output_delay']}_s{seed}"
            cfg = {**cfg, "valid_history_samples": history, "valid_future_samples": future,
                   "master_seed": seed, "save_dir": str(out / label)}
            validate_model(model, cfg, n)
            config = out / "configs" / f"{label}.json"
            jobs.append(dict(name=label, config=str(config), run_dir=cfg["save_dir"]))
            frozen.append(cfg)
            # Never replace a plan/config that may have produced existing weights.
            if config.exists():
                raise FileExistsError(f"Plan config already exists: {config}; choose a fresh output directory")
    if (out / "plan.json").exists():
        raise FileExistsError(f"Plan already exists: {out / 'plan.json'}")
    for job, cfg in zip(jobs, frozen):
        atomic_json(job["config"], cfg)
    plan = dict(schema_version=1, seeds=list(SEEDS), epochs=200, optimizer="AdamW",
                selection_metric="gt_mse", sampling_rate_hz=fs, hop_samples=4,
                common_target_interval=[history - 1, n - future], jobs=jobs,
                tost_margin=None, scope=SCOPE, status="planned_not_run", publication_ready=False)
    atomic_json(out / "plan.json", plan)
    return plan


def evaluate_run(run_dir):
    import numpy as np
    import torch
    from experiment.checkpoints import load_model, read_checkpoint
    from data.Dataloader import EEGDenoiseDataset
    from data.protocol import evaluation_data
    from experiment.methodology import aligned, target_interval, validate_model, latency, parity
    from metrics import rrmse_temporal, rrmse_spectral, acc
    root = Path(run_dir)
    model, arch, raw = load_model(root / "best.pt")
    cfg = model.run_config
    if cfg.get("protocol") != "paper" or raw.get("selection_metric") != "gt_mse":
        raise ValueError("Not a paper-protocol checkpoint selected by GT MSE")
    last = read_checkpoint(root / "last.pt")
    if last.get("run_id") != raw.get("run_id") or last["epoch"] != 200:
        raise ValueError("The same run must complete all 200 epochs before paper evaluation")
    ds = EEGDenoiseDataset(cfg["data_dir"], split="test", **evaluation_data(model))
    validate_model(model, cfg, ds.clean.shape[-1])
    records = []
    with torch.no_grad():
        for i in range(len(ds)):
            row = ds[i]
            x, y = row["noisy"][None], row["clean"][None]
            pred, clean = aligned(model(x), y, cfg)
            raw_pair = torch.cat((x, y), -1) * row["sigma"]
            segment_id = hashlib.sha256(raw_pair.numpy().tobytes()).hexdigest()
            records.append(dict(segment_id=segment_id, split_seed=cfg["master_seed"],
                                target_interval=list(target_interval(y.shape[-1], cfg)),
                                sample_rate_hz=ds.fs, gt_mse=float((pred-clean).square().mean()),
                                rrmse_t=float(rrmse_temporal(pred, clean).mean()),
                                rrmse_s=float(rrmse_spectral(pred, clean).mean()),
                                acc=float(acc(pred, clean).mean())))
    summary = {key: float(np.mean([r[key] for r in records])) for key in METRICS}
    from models.variants import variant_is_causal
    parity_result = None
    if variant_is_causal(arch["variant"]):
        # Multiple epochs concatenated only for state/cache correctness, never accuracy.
        probe = np.random.default_rng(0).normal(size=1024).astype(np.float32) * cfg["normalization_scale"]
        parity_result = parity(model, probe)
        if not parity_result["passed"]:
            raise ValueError("Raw batch/hop parity failed: result is excluded from paper")
    result = dict(run_dir=str(root), run_id=raw["run_id"], seed=cfg["master_seed"],
                  config=cfg, architecture=arch, summary=summary, segments=records,
                  parity=parity_result, latency_budget=latency(cfg, ds.fs),
                  hardware_measured=False, scope=SCOPE)
    atomic_json(root / "paper_evaluation.json", result)
    return result


def report(plan, margin=None):
    from experiment.statistics import mean_sd, pair_segments, paired_tost
    groups = {}
    for job in plan["jobs"]:
        result = json.loads((Path(job["run_dir"]) / "paper_evaluation.json").read_text())
        expected = json.loads(Path(job["config"]).read_text())
        cfg = result["config"]
        for key, value in expected.items():
            if key not in ("normalization_scale",) and cfg.get(key) != value:
                raise ValueError(f"Evaluation does not match its planned recipe: {key}")
        if result["segments"][0]["target_interval"] != plan["common_target_interval"]:
            raise ValueError("Result uses a different common target interval")
        key = (cfg["variant"], cfg["output_delay"])
        group = groups.setdefault(key, {})
        if result["seed"] in group:
            raise ValueError("Duplicate seed result")
        group[result["seed"]] = result
    rows = []
    for (variant, delay), results in groups.items():
        if set(results) != set(SEEDS):
            raise ValueError("Every included configuration requires seeds 42, 54 and 67")
        rows.append(dict(variant=variant, output_delay=delay,
                         metrics={key: mean_sd([r["summary"][key] for r in results.values()]) for key in METRICS},
                         latency_budget=next(iter(results.values()))["latency_budget"]))
    tests = []
    for (variant, delay), results in groups.items():
        base = groups.get(("tcn_se", delay))
        if not base or "mla" not in variant and "mha" not in variant:
            continue
        a, b, counts = {}, {}, {}
        for seed in SEEDS:
            a[seed], b[seed], counts[seed] = pair_segments(results[seed]["segments"], base[seed]["segments"], "rrmse_t")
        test = dict(a=variant, b="tcn_se", output_delay=delay, metric="rrmse_t",
                    paired_segments_by_seed=counts, differences_by_seed={s:a[s]-b[s] for s in SEEDS})
        if margin is None:
            test.update(equivalence="not_tested", reason="No prespecified TOST margin")
        else:
            test["tost"] = paired_tost(a, b, margin, "seed")
        tests.append(test)
    return dict(rows=rows, comparisons=tests, independent_unit="training seed", scope=SCOPE,
                hardware_status="requires device traces and external power measurements")


def evaluate_real(plan, data_dir):
    """Keep seed repeats together; never pool individual blinks as subjects."""
    import numpy as np
    from eegio_eval import (EvalConfig, evaluate_subjects, ALL_METRIC_PATHS, METRIC_PATHS,
                            LEGACY_METRIC_PATHS, METRIC_PROTOCOL, pluck)
    from experiment.statistics import mean_sd
    from experiment.checkpoints import read_checkpoint
    groups = {}
    for job in plan["jobs"]:
        root = Path(job["run_dir"])
        raw, last = read_checkpoint(root/"best.pt"), read_checkpoint(root/"last.pt")
        if raw.get("selection_metric") != "gt_mse" or last["epoch"] != 200 or last.get("run_id") != raw.get("run_id"):
            raise ValueError("Real-data paper evaluation requires completed, GT-selected runs")
        cfg = raw["run_config"]
        if cfg.get("protocol") != "paper":
            raise ValueError("Real-data paper evaluation requires the strict protocol")
        result = evaluate_subjects(data_dir, str(root), cfg=EvalConfig(target_fs=plan["sampling_rate_hz"], hop=4))
        atomic_json(root/"paper_real.json", result)
        key = (cfg["variant"], cfg["output_delay"])
        group = groups.setdefault(key, {})
        if cfg["master_seed"] in group:
            raise ValueError("Duplicate real-data seed")
        group[cfg["master_seed"]] = result
    rows = []
    for (variant, delay), results in groups.items():
        if set(results) != set(SEEDS):
            raise ValueError("Three seeds are required per real-data configuration")
        subject_sets = [{r["subject"] for r in v["rows"]} for v in results.values()]
        if any(ids != subject_sets[0] for ids in subject_sets):
            raise ValueError("Real-data seed repeats must use the same subjects")
        indexed = {seed:{r["subject"]:r for r in results[seed]["rows"]} for seed in SEEDS}
        subjects = []
        for subject in sorted(subject_sets[0]):
            metrics = {}
            for metric in ALL_METRIC_PATHS:
                values = [pluck(indexed[seed][subject], metric) for seed in SEEDS]
                if not np.isfinite(values).all():
                    metrics[metric] = None
                else:
                    metrics[metric] = mean_sd(values)
            subjects.append(dict(subject=subject, seed_repeats=3,
                                 metrics={key: metrics[key] for key in METRIC_PATHS},
                                 legacy_metrics={key: metrics[key] for key in LEGACY_METRIC_PATHS}))
        rows.append(dict(variant=variant, output_delay=delay, subjects=subjects,
                         inference_checks_by_seed={seed: result.get("inference_checks")
                                                   for seed, result in results.items()}))
    return dict(rows=rows, metric_protocol=METRIC_PROTOCOL, independent_unit="subject",
                seed_repeats=3, scope=SCOPE,
                equivalence="not_tested", reason="No prespecified TOST margin")


def plot_report(result, path):
    """Paper curve in system units; offline references have no fake latency."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    variants = sorted({r["variant"] for r in result["rows"]})
    for variant in variants:
        rows = sorted((r for r in result["rows"] if r["variant"] == variant), key=lambda r:r["output_delay"])
        if variant.endswith("_bi"):
            ax.axhline(rows[0]["metrics"]["rrmse_t"]["mean"], linestyle="--", label=f"{variant} (offline reference)")
            continue
        ax.errorbar([r["latency_budget"]["algorithmic_budget_ms"] for r in rows],
                    [r["metrics"]["rrmse_t"]["mean"] for r in rows],
                    yerr=[r["metrics"]["rrmse_t"]["sd"] for r in rows],
                    marker="o", capsize=3, label=variant)
    ax.set_xscale("log")
    ax.set_xlabel("Algorithmic latency budget (ms)")
    ax.set_ylabel("RRMSE_t (mean ± sample SD, 3 seeds)")
    ax.set_title("Single-channel EOG/EMG · H=4")
    ax.grid(alpha=.25); ax.legend()
    fig.text(.5, .01, "Algorithmic budgets; device inference and I/O require separate measurements.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1)); fig.savefig(path, dpi=180); plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", required=True)
    p.add_argument("--variants", nargs="+", default=["tcn", "tcn_se", "tcn_se_bi"])
    p.add_argument("--delays", nargs="+", type=int, default=[0, 8, 16, 32])
    p.add_argument("--fs", type=float, choices=[250., 256.], default=256.)
    p.add_argument("--data_dir")
    for mode in ("train", "evaluate", "report", "real"):
        p = sub.add_parser(mode)
        p.add_argument("--plan", required=True)
        if mode == "real":
            p.add_argument("--data_dir", required=True)
    args = parser.parse_args(argv)
    if args.mode == "plan":
        plan = make_plan(args.out, args.variants, args.delays, args.fs, args.data_dir)
        print(f"Created {len(plan['jobs'])} jobs; target interval {plan['common_target_interval']}; H=4")
        return
    path = Path(args.plan)
    plan = json.loads(path.read_text())
    if args.mode == "real":
        atomic_json(path.parent / "real_report.json", evaluate_real(plan, args.data_dir))
    elif args.mode == "report":
        result = report(plan, plan.get("tost_margin"))
        atomic_json(path.parent / "report.json", result)
        plot_report(result, path.parent / "accuracy_latency.png")
        for row in result["rows"]:
            m = row["metrics"]["rrmse_t"]
            print(f"{row['variant']} ell={row['output_delay']}: RRMSE_t {m['mean']:.5f} ± {m['sd']:.5f}")
    else:
        for job in plan["jobs"]:
            print(job["name"], flush=True)
            if args.mode == "train":
                subprocess.run([sys.executable, "-m", "trainer", "--config", job["config"]], check=True)
            else:
                evaluate_run(job["run_dir"])


if __name__ == "__main__":
    main()
