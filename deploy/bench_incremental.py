"""
Measure what incremental inference costs, and what MLA's cache actually saves.

Why this script exists
======================
Two numbers in this project were arguments rather than measurements:

1. **MLA's KV-cache compression.**  The architecture was chosen for it, but
   ``export_onnx.py`` writes a graph with no cache ("every call recomputes"),
   so no path in the repo held a cache that could be weighed.  ``gui/params.py``
   estimated it analytically.  ``models/incremental.py`` now holds the real
   thing, and this script reads its size off the live buffers.

2. **The cost of the sliding-window stream.**  ``raspi_infer.py`` recomputes
   512 positions per hop to keep ``hop`` of them, so it does ``512/hop`` times
   the necessary work.  That overhead was never quoted next to the per-sample
   cost of not paying it.

Both are reported per variant, with the environment block ``deploy/preflight.py``
produces, so a number and the conditions it was taken under stay together —
the same contract the latency traces use.

What is compared
================
``incremental``  one ``IncrementalModel.step()`` per sample: O(window) work.
``windowed``     the current path: one 512-wide forward per ``hop`` samples,
                 amortised over the ``hop`` samples it emits.

The two are not quite the same function on the attention variants (see the
equivalence section of ``models/incremental.py``), and the gap — ~1e-5 relative
— is measured here too rather than assumed, so the speedup is never quoted for
a stream that stopped agreeing with the batch model.

Usage
=====
    # Architectural comparison on freshly built models (no checkpoints needed)
    python -m deploy.bench_incremental --variants tcn,tcn_mha,tcn_mla

    # A trained checkpoint, and the hop sizes the windowed path would use
    python -m deploy.bench_incremental --ckpt runs/tcn_mla_eog/best.pt --hops 32,64

    # Write a trace next to the latency traces
    python -m deploy.bench_incremental --json deploy/artifacts/incremental.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.incremental import BYTES_FP32, IncrementalModel  # noqa: E402
from models.variants import build_model, variant_is_causal  # noqa: E402

try:
    from deploy import preflight
except ImportError:                                           # run as a script
    import preflight                                          # noqa: F401

#: Variants worth the default comparison: the baseline, and both attention
#: mechanisms in the topology the Letter reports.
DEFAULT_VARIANTS = ("tcn", "tcn_mha", "tcn_mla")

#: Sampling rate the epochs were cut at, for turning ms into a duty cycle.
FS = 256.0


def _quantiles(xs: list[float]) -> dict:
    """Median and tail of a timing sample, in milliseconds."""
    xs = sorted(xs)
    if not xs:
        return {}
    def q(p):
        return xs[min(len(xs) - 1, int(p * len(xs)))]
    return {
        "median_ms": round(statistics.median(xs) * 1e3, 5),
        "p95_ms": round(q(0.95) * 1e3, 5),
        "p99_ms": round(q(0.99) * 1e3, 5),
        "mean_ms": round(statistics.fmean(xs) * 1e3, 5),
        "n": len(xs),
    }


# ============================================================
# Timing
# ============================================================
@torch.no_grad()
def time_incremental(model, n_steps: int, warmup: int, absorb: bool = True) -> dict:
    """
    Per-sample cost of the streaming path, timed one step at a time.

    The first ``warmup`` steps are run but not timed: until the ring buffers
    and the attention cache fill, a step is doing less work than it will in
    the steady state, and reporting that as the cost would flatter the stream.
    """
    stream = IncrementalModel(model, absorb=absorb)
    torch.manual_seed(0)
    signal = torch.randn(warmup + n_steps)

    for t in range(warmup):
        stream.step(signal[t])

    samples = []
    for t in range(warmup, warmup + n_steps):
        t0 = time.perf_counter()
        stream.step(signal[t])
        samples.append(time.perf_counter() - t0)

    report = stream.cache_report()
    out = {"per_sample": _quantiles(samples), "cache": report}
    median_s = statistics.median(samples)
    out["duty_cycle"] = round(median_s * FS, 5)       # compute per sample / sample period
    out["max_fs_hz"] = round(1.0 / median_s, 1)
    return out


@torch.no_grad()
def time_windowed(model, hop: int, window: int, n_hops: int, warmup_hops: int = 3) -> dict:
    """
    Per-sample cost of the hop-and-recompute path, for the same work.

    One forward pass over ``window`` samples emits ``hop`` outputs, so the
    per-sample cost is the pass divided by ``hop``.  This is what
    ``raspi_infer.denoise_stream`` does, timed the same way it runs.
    """
    torch.manual_seed(0)
    x = torch.randn(1, 1, window)

    for _ in range(warmup_hops):
        model(x)

    per_hop = []
    for _ in range(n_hops):
        t0 = time.perf_counter()
        model(x)
        per_hop.append(time.perf_counter() - t0)

    median_hop = statistics.median(per_hop)
    return {
        "hop": hop,
        "window": window,
        "per_hop": _quantiles(per_hop),
        "per_sample": {"median_ms": round(median_hop / hop * 1e3, 5)},
        "redundancy": round(window / hop, 2),     # positions computed per position kept
        "duty_cycle": round(median_hop / (hop / FS), 5),
        "max_fs_hz": round(hop / median_hop, 1),
    }


@torch.no_grad()
def agreement(model, n: int = 1200, window: int = 512, probes=(512, 800, 1199)) -> dict:
    """
    How far the streaming output sits from the windowed output, relative to
    signal RMS.

    Reported next to the speedup so the speedup is never quoted for a stream
    that has stopped computing the same function.  For the baseline this is
    float noise; for the attention variants it is the ~1e-5 the module
    docstring accounts for.
    """
    stream = IncrementalModel(model)
    torch.manual_seed(1)
    x = torch.randn(n)
    streamed = stream.run(x)
    scale = streamed.std().item()

    worst = 0.0
    for t in probes:
        if t < window - 1:
            continue
        last = model(x[t - window + 1:t + 1].view(1, 1, window))[0, 0, -1]
        worst = max(worst, abs(last - streamed[t]).item() / scale)
    return {"max_relative_gap": float(f"{worst:.3e}"), "probes": list(probes)}


# ============================================================
# One variant
# ============================================================
def bench_variant(name: str, model, *, hops, window, n_steps, n_hops,
                  check_agreement: bool = True) -> dict:
    model.eval()
    row: dict = {"variant": name,
                 "params": sum(p.numel() for p in model.parameters())}

    inc = time_incremental(model, n_steps=n_steps, warmup=window)
    row["incremental"] = inc

    # MLA has two ways to spend its compression; both are timed, because the
    # cache size is the same and only the arithmetic differs.
    if inc["cache"]["attention"] == "mla":
        row["incremental_no_absorb"] = time_incremental(
            model, n_steps=n_steps, warmup=window, absorb=False)

    row["windowed"] = [time_windowed(model, hop, window, n_hops) for hop in hops]

    if check_agreement:
        row["agreement_vs_windowed"] = agreement(model, window=window)

    inc_ms = inc["per_sample"]["median_ms"]
    row["speedup_vs_windowed"] = {
        f"hop{w['hop']}": round(w["per_sample"]["median_ms"] / inc_ms, 2)
        for w in row["windowed"]
    }
    return row


def _print_row(row: dict) -> None:
    cache = row["incremental"]["cache"]
    inc = row["incremental"]["per_sample"]["median_ms"]
    print(f"\n  {row['variant']}  ({row['params']:,} params, "
          f"{cache['attention'] or 'no attention'})")
    print(f"    incremental      {inc:8.4f} ms/sample   "
          f"duty {row['incremental']['duty_cycle']:.4f}   "
          f"max {row['incremental']['max_fs_hz']:,.0f} Hz")
    if "incremental_no_absorb" in row:
        na = row["incremental_no_absorb"]["per_sample"]["median_ms"]
        print(f"      (no absorption){na:8.4f} ms/sample   "
              f"{na / inc:.2f}x the absorbed path")
    for w in row["windowed"]:
        print(f"    windowed hop={w['hop']:<4d} {w['per_sample']['median_ms']:8.4f} ms/sample   "
              f"duty {w['duty_cycle']:.4f}   "
              f"{row['speedup_vs_windowed']['hop' + str(w['hop'])]:>6.2f}x slower   "
              f"({w['redundancy']:.0f} positions computed per position kept)")
    print(f"    attention cache  {cache['attn_cache_bytes']:>8,} B "
          f"({cache['attn_floats_per_step']} floats/step over {cache['attention_window']} steps)")
    print(f"    conv state       {cache['conv_state_bytes']:>8,} B")
    if "agreement_vs_windowed" in row:
        print(f"    vs windowed      {row['agreement_vs_windowed']['max_relative_gap']:.1e} "
              f"relative (see models/incremental.py)")


def _print_cache_comparison(rows: list[dict]) -> None:
    """The headline table: cache per step, and MLA against MHA."""
    attn = [r for r in rows if r["incremental"]["cache"]["attention"]]
    if not attn:
        return
    print(f"\n{'=' * 72}")
    print("  KV cache per step  (the claim MLA was chosen for)")
    print(f"{'=' * 72}")
    print(f"  {'variant':<16s} {'kind':<5s} {'floats/step':>12s} {'window':>8s} {'total':>12s}")
    for r in attn:
        c = r["incremental"]["cache"]
        print(f"  {r['variant']:<16s} {c['attention']:<5s} "
              f"{c['attn_floats_per_step']:>12,d} {c['attention_window']:>8,d} "
              f"{c['attn_cache_bytes']:>10,d} B")

    mha = next((r for r in attn if r["incremental"]["cache"]["attention"] == "mha"), None)
    mla = [r for r in attn if r["incremental"]["cache"]["attention"] == "mla"]
    for r in mla:
        if mha is None:
            break
        ratio = (mha["incremental"]["cache"]["attn_cache_bytes"]
                 / r["incremental"]["cache"]["attn_cache_bytes"])
        saved = (mha["incremental"]["cache"]["attn_cache_bytes"]
                 - r["incremental"]["cache"]["attn_cache_bytes"])
        print(f"\n  {r['variant']} vs {mha['variant']}: "
              f"{ratio:.2f}x smaller cache ({saved:,} B saved per stream)")


# ============================================================
# Traces, in the shape the Deploy runs table reads
# ============================================================
#: Where the windowed traces live, so an incremental measurement of the same
#: model lands beside the one it is meant to be compared with.
TRACE_DIR = Path(__file__).resolve().parent / "artifacts" / "pi_latency"


def write_trace(row: dict, environment: dict | None, out_dir: Path = TRACE_DIR) -> Path:
    """
    One incremental measurement, written where the windowed ones are.

    Named ``incremental_<model>_hop1.json`` so the Runs tab's existing
    ``_hop<N>`` parse finds the model, and because hop 1 is the literal truth:
    the streaming path advances one sample at a time, so its deadline is one
    sample period (3.9 ms at 256 Hz) rather than a hop of them.

    That is also what makes the two comparable. A windowed run at hop 32 gets
    125 ms to do 32 samples' work; this gets 3.9 ms to do one. Expressed as a
    duty cycle -- compute over wall time -- they are the same quantity, and
    the table can put them in one column without lying about either.
    """
    inc, cache = row["incremental"], row["incremental"]["cache"]
    per = inc["per_sample"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"incremental_{row['variant']}_hop1.json"
    path.write_text(json.dumps({
        "hop": 1,
        "sample_rate_hz": FS,
        "hop_budget_ms": round(1000.0 / FS, 3),
        "n_hops": per.get("n"),
        "total_ms": {k.replace("_ms", ""): v for k, v in per.items() if k.endswith("_ms")},
        "deadline_misses": None,
        "duty_cycle": inc["duty_cycle"],
        "throughput_hops_per_s": inc["max_fs_hz"],
        "inference": {
            "mode": "incremental",
            "runtime": f"pytorch {torch.__version__}",
            "window": cache["attention_window"] or cache["receptive_field"],
            "hop": 1,
            "kv_cache": bool(cache["attn_cache_bytes"]),
            "kv_floats_per_step": cache["attn_floats_per_step"],
            "kv_cache_bytes": cache["attn_cache_bytes"],
            "conv_state_bytes": cache["conv_state_bytes"],
            "total_state_bytes": cache["total_state_bytes"],
            "absorb": cache["absorb"],
            "warmup": cache["warmup"],
            "recompute_factor": 1.0,
            "note": "one step per sample, state carried between steps; nothing is recomputed. "
                    "PyTorch on the host — models/incremental.py does not run on the Pi, "
                    "which has no torch.",
        },
        "environment": environment,
    }, indent=2))
    return path


# ============================================================
# CLI
# ============================================================
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Per-sample cost and cache size of incremental vs windowed inference.")
    p.add_argument("--variants", default=",".join(DEFAULT_VARIANTS),
                   help="comma-separated variant names to build fresh (ignored with --ckpt)")
    p.add_argument("--ckpt", help="measure a trained checkpoint instead of fresh models")
    p.add_argument("--config", help="YAML config, for a checkpoint that records no model_config")
    p.add_argument("--hops", default="32",
                   help="comma-separated hop sizes for the windowed comparison")
    p.add_argument("--window", type=int, default=512,
                   help="window length of the windowed path (the trained epoch length)")
    p.add_argument("--steps", type=int, default=2000, help="timed streaming steps")
    p.add_argument("--hops-timed", type=int, default=200, help="timed windowed forward passes")
    p.add_argument("--threads", type=int, default=1,
                   help="torch intra-op threads; recorded in the trace")
    p.add_argument("--json", help="write the full trace here")
    p.add_argument("--trace", action="store_true",
                   help="also write one trace per model into deploy/artifacts/pi_latency/, "
                        "so the Deploy runs table lists it beside the windowed measurement "
                        "of the same model")
    p.add_argument("--no-preflight", action="store_true",
                   help="skip the environment check (the trace then records nothing about "
                        "the machine, and says so)")
    a = p.parse_args(argv)

    torch.set_num_threads(a.threads)
    hops = [int(h) for h in a.hops.split(",") if h.strip()]

    env = None
    if not a.no_preflight:
        env = preflight.check(a.threads)
        preflight.emit(env)

    print(f"\n{'=' * 72}")
    print("  Incremental vs windowed inference")
    print(f"{'=' * 72}")
    print(f"  torch {torch.__version__} | {platform.machine()} | {a.threads} thread(s) | "
          f"window {a.window} | {a.steps} timed steps")

    rows = []
    if a.ckpt:
        from deploy.export_onnx import load_checkpoint_model
        model, arch, _ = load_checkpoint_model(a.ckpt, a.config)
        # The run directory, not the variant: that is the name the exports,
        # the windowed traces and the Runs tab all use, and an incremental
        # trace has to join to the same model as the windowed one beside it.
        name = Path(a.ckpt).parent.name
        rows.append(bench_variant(name, model, hops=hops, window=a.window,
                                  n_steps=a.steps, n_hops=a.hops_timed))
        _print_row(rows[-1])
    else:
        for name in [v.strip() for v in a.variants.split(",") if v.strip()]:
            if not variant_is_causal(name):
                print(f"\n  {name}: skipped — bidirectional, attends to future samples "
                      f"and has no streaming form")
                continue
            model = build_model(name)
            rows.append(bench_variant(name, model, hops=hops, window=a.window,
                                      n_steps=a.steps, n_hops=a.hops_timed))
            _print_row(rows[-1])

    _print_cache_comparison(rows)

    if a.trace:
        for row in rows:
            print(f"  trace -> {write_trace(row, env)}")

    if a.json:
        out = Path(a.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "rows": rows,
            "settings": {"window": a.window, "hops": hops, "steps": a.steps,
                         "threads": a.threads, "fs": FS,
                         "torch": torch.__version__, "machine": platform.machine(),
                         "bytes_per_float": BYTES_FP32},
            "environment": env,
        }, indent=2))
        print(f"\n  trace -> {out}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
