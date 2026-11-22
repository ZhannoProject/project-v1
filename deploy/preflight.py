"""
What the machine was doing while it was being measured.

A latency number is a statement about a model *and* about the machine it ran
on, and the second half is usually left out.  It should not be: on this board
the same 512-sample window takes 24.4 ms on one onnxruntime thread and 18.2 ms
on two, and a CPU that is free to drop its clock between hops produces a p99
that describes the governor rather than the network.  So before any timing
this records the state of the three knobs that move the numbers, judges them
against what a reportable measurement needs, and refuses to let the result be
read as if the machine had been held still when it was not:

    governor   every CPU the process can run on is at "performance", so the
               clock does not ramp between hops.  Under "ondemand" a hop that
               arrives after an idle gap is computed at a lower clock than one
               in the middle of a burst, which puts the governor's reaction
               time into the p99 and calls it the network's.  Reading is free;
               setting it needs root, which is why this reports rather than
               fixes.
    affinity   the process is pinned to a fixed set of CPUs, as many as it has
               threads.  A Pi 5's four Cortex-A76 are identical, so the cost of
               migration is not landing on a slower core but losing cache
               locality and sharing a core with the OS -- which is exactly what
               makes 4 threads worse than 2 in the table above.  On a
               heterogeneous SoC it would also be the wrong kind of core.
    threads    the thread count is set explicitly and OMP agrees with it.  It
               is the single largest effect in that table -- a quarter off the
               median and half again on the p95 -- so a run that does not
               record it is not comparable with one that does.

plus the two things that invalidate a run after the fact -- thermal throttling
and other load on the machine.

Stdlib only, and it is copied to the Pi next to ``raspi_infer.py``.  macOS has
neither a governor nor an affinity API, so there the verdict is ``unlocked``
with that as the reason: a host measurement is a convenience, and the board's
is the measurement.

    python -m deploy.preflight --threads 2            # on the host
    python preflight.py --threads 2 --cpus 0-1        # on the Pi
    python -m deploy.preflight --json                 # one line, for a tool
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

CPU_ROOT = Path("/sys/devices/system/cpu")
THERMAL = Path("/sys/class/thermal/thermal_zone0/temp")
DT_MODEL = Path("/proc/device-tree/model")

#: The governor a reportable measurement is taken under.
WANTED_GOVERNOR = "performance"

#: 1-minute load average above this means something else was running.
LOAD_WARN = 1.0

PASS, WARN, FAIL, NA = "pass", "warn", "fail", "n/a"

#: Overall verdicts, worst first.  "unlocked" is also what a host that *cannot*
#: be locked gets: the distinction that matters to a reader is whether the
#: numbers were taken under controlled conditions, not why they were not.
LOCKED, PARTIAL, UNLOCKED = "locked", "partial", "unlocked"


# ============================================================
# CPU lists:  "0-1,3"  <->  [0, 1, 3]
# ============================================================
def parse_cpu_list(text: str) -> list[int]:
    """``"0-1,3"`` -> ``[0, 1, 3]``.  Raises ValueError on anything else."""
    out: list[int] = []
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise ValueError(f"not a CPU list: {text!r}")
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        if hi < lo:
            raise ValueError(f"reversed CPU range: {part!r}")
        out.extend(range(lo, hi + 1))
    if not out:
        raise ValueError("no CPUs named")
    if len(set(out)) != len(out):
        raise ValueError(f"a CPU is named twice: {text!r}")
    return sorted(out)


def format_cpu_list(cpus) -> str:
    """``[0, 1, 3]`` -> ``"0-1,3"``."""
    cpus = sorted(cpus)
    runs, start = [], None
    for i, c in enumerate(cpus):
        if start is None:
            start = c
        if i + 1 == len(cpus) or cpus[i + 1] != c + 1:
            runs.append(str(start) if start == c else f"{start}-{c}")
            start = None
    return ",".join(runs)


def pin(cpus: str) -> list[int]:
    """
    Pin this process to ``cpus`` and return them.

    Done in-process rather than through ``taskset`` so that the pinning is part
    of the run that records it: the trace then cannot claim an affinity the
    process did not actually have.
    """
    wanted = parse_cpu_list(cpus)
    if not _has_affinity():
        raise RuntimeError(f"{platform.system()} has no CPU affinity API — "
                           "pinning is a Linux-only control")
    available = set(range(os.cpu_count() or 1))
    missing = sorted(set(wanted) - available)
    if missing:
        raise ValueError(f"no such CPU: {format_cpu_list(missing)} "
                         f"(this machine has {os.cpu_count()})")
    os.sched_setaffinity(0, wanted)
    got = sorted(os.sched_getaffinity(0))
    if got != wanted:
        raise RuntimeError(f"asked for CPUs {format_cpu_list(wanted)} but got "
                           f"{format_cpu_list(got)}")
    return got


# ============================================================
# Reading the machine
# ============================================================
def _read(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace").strip("\x00\n ")
    except OSError:
        return None


def _vcgencmd(*args) -> str | None:
    """Raspberry Pi firmware query; None anywhere else."""
    try:
        out = subprocess.run(["vcgencmd", *args], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def _online_cpus() -> list[int]:
    online = _read(CPU_ROOT / "online")
    if online:
        try:
            return parse_cpu_list(online)
        except ValueError:
            pass
    return list(range(os.cpu_count() or 1))


def _has_affinity() -> bool:
    """Whether this platform lets a process be pinned at all (Linux does)."""
    return hasattr(os, "sched_setaffinity")


def _affinity() -> list[int] | None:
    if not hasattr(os, "sched_getaffinity"):
        return None
    return sorted(os.sched_getaffinity(0))


def _cpufreq(cpu: int) -> dict:
    d = CPU_ROOT / f"cpu{cpu}" / "cpufreq"
    out = {}
    for key, name in (("governor", "scaling_governor"), ("cur_khz", "scaling_cur_freq"),
                      ("min_khz", "scaling_min_freq"), ("max_khz", "scaling_max_freq")):
        v = _read(d / name)
        if v is None:
            continue
        out[key] = v if key == "governor" else int(v)
    return out


def _throttled() -> dict:
    """
    ``vcgencmd get_throttled``: a bit field the firmware latches.  0x0 means
    the board has not been throttled or browned out since boot, which is the
    only state in which a timing run is trustworthy.
    """
    raw = _vcgencmd("get_throttled")
    if not raw or "=" not in raw:
        return {}
    value = raw.split("=", 1)[1].strip()
    try:
        bits = int(value, 16)
    except ValueError:
        return {"raw": value}
    now = {0: "under-voltage", 1: "arm frequency capped", 2: "currently throttled",
           3: "soft temperature limit"}
    ever = {16: "under-voltage has occurred", 17: "arm frequency capping has occurred",
            18: "throttling has occurred", 19: "soft temperature limit has occurred"}
    flags = [text for bit, text in {**now, **ever}.items() if bits & (1 << bit)]
    return {"raw": value, "bits": bits, "flags": flags}


def _temperature() -> float | None:
    raw = _vcgencmd("measure_temp")
    if raw:
        m = re.search(r"([\d.]+)", raw)
        if m:
            return float(m.group(1))
    v = _read(THERMAL)
    if v:
        try:
            return int(v) / 1000.0
        except ValueError:
            pass
    return None


def _interrupts() -> dict:
    """
    Cumulative interrupts handled per CPU, from ``/proc/interrupts``.

    These counts include past experiments and are not a current interrupt
    rate, CPU-time measurement or proof of OS housekeeping/isolation. Keep
    them as diagnostic provenance, never as a reason to reject fixed affinity.

    Only numbered IRQ lines are counted; the IPI and ERR rows at the bottom are
    inter-processor traffic, which is a consequence of the work rather than
    housekeeping competing with it.
    """
    try:
        lines = Path("/proc/interrupts").read_text().splitlines()
    except OSError:
        return {}
    if not lines:
        return {}
    n = len([c for c in lines[0].split() if c.upper().startswith("CPU")])
    if not n:
        return {}
    totals = [0] * n
    for line in lines[1:]:
        label, _, rest = line.partition(":")
        if not label.strip().isdigit():
            continue
        for i, field in enumerate(rest.split()[:n]):
            if field.isdigit():
                totals[i] += int(field)
    grand = sum(totals)
    if not grand:
        return {}
    return {"per_cpu": totals, "total": grand, "scope": "cumulative_since_boot",
            "share": [round(c / grand, 4) for c in totals]}


def _board() -> str:
    model = _read(DT_MODEL)
    if model:
        return model
    return f"{platform.system()} {platform.machine()}"


def collect(threads: int | None = None, runtime: dict | None = None) -> dict:
    """
    Everything that is read, before any of it is judged.

    ``runtime`` is what the inference session actually has (from
    ``EEGDenoiser.threads``) rather than what was asked for — the two can
    differ, and only the first one is a fact about the measurement.
    """
    affinity = _affinity()
    online = _online_cpus()
    # Every online CPU is read, not only the ones in use: the verdict is about
    # the cores this process can land on, but a core at "ondemand" next door is
    # worth saying out loud rather than hiding behind the affinity mask.
    in_use = affinity if affinity is not None else online
    load = os.getloadavg() if hasattr(os, "getloadavg") else None
    return {
        "board": _board(),
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine(),
                     "python": platform.python_version()},
        "supports": {"governor": (CPU_ROOT / "cpu0" / "cpufreq").is_dir(),
                     "affinity": _has_affinity()},
        "cpu": {"count": os.cpu_count(), "online": online,
                "affinity": affinity,
                "affinity_str": None if affinity is None else format_cpu_list(affinity),
                "in_use": in_use,
                "cpufreq": {str(c): _cpufreq(c) for c in online}},
        "threads": {"requested": threads,
                    "runtime": runtime,                 # what the session really has
                    "omp": os.environ.get("OMP_NUM_THREADS"),
                    "mkl": os.environ.get("MKL_NUM_THREADS")},
        "interrupts": _interrupts(),
        "thermal": {"temp_c": _temperature(), "throttled": _throttled(),
                    "arm_clock": _vcgencmd("measure_clock", "arm")},
        "load": None if load is None else {"avg1": round(load[0], 2),
                                           "avg5": round(load[1], 2),
                                           "avg15": round(load[2], 2)},
    }


# ============================================================
# Judging it
# ============================================================
def _check(name, status, detail, want):
    return {"name": name, "status": status, "detail": detail, "want": want}


def _governor_check(state: dict) -> dict:
    want = f'every CPU in use at "{WANTED_GOVERNOR}"'
    if not state["supports"]["governor"]:
        return _check("governor", NA,
                      f"{state['platform']['system']} exposes no CPU governor — "
                      "the clock cannot be pinned from software here", want)
    freq = state["cpu"]["cpufreq"]
    in_use = state["cpu"].get("in_use") or state["cpu"]["online"]
    govs = {c: freq[str(c)].get("governor") for c in in_use
            if freq.get(str(c), {}).get("governor")}
    if not govs:
        return _check("governor", FAIL, "no scaling_governor could be read", want)
    where = format_cpu_list(govs)
    # Cores this process cannot land on do not decide the verdict, but a
    # mismatch next door is said out loud: it is the usual sign that the
    # governor was set for one run and not for the machine.
    others = sorted({f.get("governor") for c, f in freq.items()
                     if int(c) not in govs and f.get("governor")} - {WANTED_GOVERNOR})
    aside = f" (CPU outside the pin set are at {', '.join(others)})" if others else ""
    unique = sorted(set(govs.values()))
    if unique == [WANTED_GOVERNOR]:
        freqs = [freq[str(c)].get("cur_khz") for c in govs if freq[str(c)].get("cur_khz")]
        at = f", running at {max(freqs) / 1e6:.2f} GHz" if freqs else ""
        return _check("governor", PASS, f"CPU {where} at performance{at}{aside}", want)
    return _check("governor", FAIL,
                  f"CPU {where} at {', '.join(unique)} — the clock is free to ramp "
                  "between hops; sudo cpupower frequency-set -g performance", want)


def housekeeping_cpus(state: dict, share: float = 0.25) -> list[int]:
    """
    Legacy diagnostic: CPUs with disproportionate cumulative IRQ counts.

    "Disproportionate" is measured, not assumed: more than ``share`` of all
    interrupts when an even spread would give 1/n each. This does not identify
    current housekeeping CPUs and must not be used to reject a measurement.
    """
    irq = state.get("interrupts") or {}
    if not irq.get("share"):
        return []
    n = len(irq["share"])
    return [cpu for cpu, s in enumerate(irq["share"]) if s > max(share, 1.5 / n)]


def default_cpus(threads: int, total: int) -> str:
    """
    The cores to pin to by default: the top ``threads`` of them.

    A reproducible default, leaving CPU 0 out where possible. Selecting these
    cores does not establish OS/IRQ isolation; that needs separate evidence.
    """
    threads = max(1, min(threads, total))
    return format_cpu_list(range(total - threads, total))


def _affinity_check(state: dict, threads: int | None) -> dict:
    want = "pinned to a fixed set of CPUs, one per inference thread"
    if not state["supports"]["affinity"]:
        return _check("affinity", NA,
                      f"{state['platform']['system']} has no CPU affinity API — "
                      "the scheduler is free to move the work", want)
    affinity = state["cpu"]["affinity"] or []
    total = state["cpu"]["count"] or len(affinity)
    where = format_cpu_list(affinity) if affinity else "?"
    irq = state.get("interrupts") or {}

    def irq_note(cpus) -> str:
        if not irq.get("share"):
            return ""
        shares = ", ".join(f"CPU {c} {100 * irq['share'][c]:.0f}%"
                           for c in cpus if c < len(irq["share"]))
        return (f"; cumulative device IRQ share: {shares} "
                "(diagnostic only, not a current-load or isolation test)") if shares else ""

    if len(affinity) >= total:
        return _check("affinity", FAIL,
                      f"not pinned — free to run on all {total} CPUs; pass "
                      f"--cpus {default_cpus(threads or 2, total)}", want)
    if threads and len(affinity) < threads:
        return _check("affinity", FAIL,
                      f"pinned to {len(affinity)} CPU(s) ({where}) but running "
                      f"{threads} threads — they will contend", want)
    if threads and len(affinity) > threads:
        return _check("affinity", WARN,
                      f"pinned to {len(affinity)} CPUs ({where}) for {threads} threads — "
                      "the spare core lets the scheduler migrate the work", want)
    return _check("affinity", PASS,
                  f"pinned to CPU {where}" + irq_note(affinity), want)


def _threads_check(state: dict, threads: int | None) -> dict:
    """
    The control here is onnxruntime's ``intra_op_num_threads``, and nothing
    else.

    It is tempting to check ``OMP_NUM_THREADS`` instead, and this did -- but
    onnxruntime is built without OpenMP (its ``get_build_info()`` says so, and
    the wheels have shipped that way for years), so that variable governs none
    of the timed path.  Requiring it to match would have been asserting a
    setting with no effect on the thing being measured.  It is still reported,
    because it does govern numpy's BLAS in the same process, which is what
    scores the test set.
    """
    want = "an explicit onnxruntime intra-op thread count"
    omp = state["threads"]["omp"]
    rt = state["threads"].get("runtime") or {}
    if rt.get('mode') == 'channel_processes':
        workers = rt.get('worker_settings') or []
        valid = len(workers) == threads and rt.get('process_count') == threads
        valid = valid and all(w.get('threads', {}).get('intra_op') == 1
                              and w.get('threads', {}).get('allow_spinning') is False for w in workers)
        if state['supports']['affinity']:
            cores = [w.get('cpu') for w in workers]
            valid = valid and len(set(cores)) == len(cores)
            valid = valid and all(w.get('affinity') == [w.get('cpu')] for w in workers)
            valid = valid and sorted(cores) == state['cpu']['affinity']
        return _check('threads', PASS if valid else FAIL,
                      f"{len(workers)} independent channel processes, each with one non-spinning ORT thread; "
                      + ('worker settings verified' if valid else 'worker count, thread or affinity mismatch'),
                      'one single-thread inference process pinned to each selected CPU')
    aside = ""
    if rt.get("openmp_in_build") is False:
        aside = (f"; OMP_NUM_THREADS={omp} governs numpy here, not onnxruntime "
                 "(this build has no OpenMP)" if omp else
                 "; OMP_NUM_THREADS is unset, which is harmless — this "
                 "onnxruntime has no OpenMP")
    elif omp:
        aside = f", OMP_NUM_THREADS={omp}"
    if threads is None:
        # Nothing has been asked to run, so there is no thread count to judge.
        # Not a pass either: a machine cannot be certified for a measurement
        # that has not been described.
        return _check("threads", NA,
                      "no thread count given — it is fixed per run, so this is "
                      "settled when a measurement is started", want)
    if rt and rt.get("intra_op") not in (None, threads):
        return _check("threads", FAIL,
                      f"asked for {threads} but the session is running "
                      f"{rt['intra_op']} intra-op threads", want)
    # ORT counts the calling thread as one of the N, so N threads want N cores
    # -- which is the ratio the affinity check enforces.
    detail = (f"{threads} onnxruntime intra-op thread{'s' if threads != 1 else ''} "
              f"({max(0, threads - 1)} worker{'s' if threads != 2 else ''} + the caller)")
    if rt.get("allow_spinning"):
        detail += ", pool spinning"
    elif rt.get("allow_spinning") is False:
        detail += ", pool not spinning"
    return _check("threads", PASS, detail + aside, want)


def _thermal_check(state: dict) -> dict:
    want = "no throttling latched since boot"
    t = state["thermal"]
    throttled, temp = t.get("throttled") or {}, t.get("temp_c")
    at = f" at {temp:.1f} °C" if temp is not None else ""
    if not throttled:
        return _check("thermal", NA, f"no throttle flag on this platform{at}", want)
    if throttled.get("bits"):
        return _check("thermal", FAIL,
                      f"{throttled['raw']}: {'; '.join(throttled['flags'])}{at}", want)
    return _check("thermal", PASS, f"never throttled since boot ({throttled['raw']}){at}", want)


def _load_check(state: dict) -> dict:
    want = f"nothing else running (1-min load < {LOAD_WARN})"
    load = state.get("load")
    if not load:
        return _check("load", NA, "no load average on this platform", want)
    if state.get('measurement_phase') == 'after_replay':
        return _check('load', NA, f"1-min load {load['avg1']} includes this replay's own work; diagnostic only", want)
    if load["avg1"] >= LOAD_WARN:
        return _check("load", WARN,
                      f"1-min load {load['avg1']} — something else is using the machine", want)
    return _check("load", PASS, f"1-min load {load['avg1']}", want)


def judge(state: dict, threads: int | None = None) -> dict:
    """
    ``state`` -> the same thing with a verdict on it.

    ``locked`` is deliberately hard to get: the three knobs that move the
    numbers have to be *controlled*, not merely readable, so a platform that
    cannot control them is ``unlocked`` rather than excused.  A warning
    downgrades to ``partial``, which is honest about being neither.
    """
    threads = threads if threads is not None else state["threads"].get("requested")
    checks = [_governor_check(state), _affinity_check(state, threads),
              _threads_check(state, threads), _thermal_check(state), _load_check(state)]
    status = {c["name"]: c["status"] for c in checks}
    controls = ("governor", "affinity", "threads")
    if any(status[c] == FAIL for c in status):
        verdict = UNLOCKED
    elif any(status[c] != PASS for c in controls):
        # n/a on a control is not a pass: the knob exists in the world, this
        # machine just has no way to hold it.
        verdict = UNLOCKED if any(status[c] == NA for c in controls) else PARTIAL
    elif any(s == WARN for s in status.values()):
        verdict = PARTIAL
    else:
        verdict = LOCKED
    reasons = [c["detail"] for c in checks if c["status"] in (FAIL, NA) and c["name"] in controls]
    return {**state, "checks": checks, "verdict": verdict,
            "preflight_policy": "fixed_controls_v2_irq_diagnostic",
            "summary": summarise(state, checks, verdict),
            "blockers": reasons}


def check(threads: int | None = None, runtime: dict | None = None, *, phase='before_replay') -> dict:
    """Read the machine and judge it in one call."""
    return judge({**collect(threads, runtime), 'measurement_phase': phase}, threads)


def summarise(state: dict, checks: list[dict], verdict: str) -> str:
    """
    One sentence for the methods section of a report.

    The reviewer's question is not "was it locked" but "what was it locked to,
    and why" — so this names the settings and is written to be pasted.
    """
    gov = sorted({f.get("governor") for f in state["cpu"]["cpufreq"].values()
                  if f.get("governor")})
    cpus = state["cpu"]["affinity_str"]
    threads = state["threads"].get("requested")
    freqs = [f.get("max_khz") for f in state["cpu"]["cpufreq"].values() if f.get("max_khz")]
    bits = [f"Measurements were taken on {state['board']}"]
    if gov:
        bits.append(f"with the CPU frequency governor set to "
                    f"{'/'.join(gov)}{f' ({max(freqs) / 1e6:.2f} GHz)' if freqs else ''}")
    else:
        bits.append("on a platform that exposes no CPU frequency governor")
    if cpus:
        bits.append(f"the process pinned to CPU {cpus}")
    elif state["supports"]["affinity"]:
        bits.append("the process not pinned to any CPU subset")
    else:
        bits.append("no CPU affinity control available")
    rt = state["threads"].get("runtime") or {}
    if rt.get('mode') == 'channel_processes':
        bits.append(f"and {rt['process_count']} independent channel processes, each with one non-spinning ORT thread")
    elif threads:
        # Named as onnxruntime names it, and with the part people get wrong:
        # the count includes the thread making the call.
        phrase = (f"and {threads} onnxruntime intra-op thread{'s' if threads != 1 else ''}"
                  + (f" ({rt['workers']} worker{'s' if rt['workers'] != 1 else ''} "
                     "and the calling thread)" if rt.get("workers") is not None else ""))
        if rt.get("allow_spinning") is not None:
            phrase += (", the thread pool spinning between inferences"
                       if rt["allow_spinning"] else ", with thread-pool spinning disabled")
        bits.append(phrase)
    tail = {
        LOCKED: ("These three settings are fixed because each of them moves the measured "
                 "latency on its own: the governor decides whether a hop after an idle "
                 "gap is computed at a lower clock than one mid-burst, the affinity "
                 "decides whether the work keeps its cache and stays off the core the OS "
                 "is using, and the thread count alone moves the median by a quarter and "
                 "the p95 by half again on this project's reference board. The inference "
                 "process/thread layout is recorded separately. They are reported so the figures "
                 "can be reproduced, not only repeated."),
        PARTIAL: ("Not every control was held: the figures below are usable but the "
                  "tail is not fully reproducible."),
        UNLOCKED: ("The machine was not held in a fixed state, so these figures are "
                   "indicative only and the tail in particular is not reproducible."),
    }[verdict]
    return ". ".join([", ".join(bits), tail])


# ============================================================
# Reporting
# ============================================================
MARK = {PASS: "ok  ", WARN: "warn", FAIL: "FAIL", NA: "n/a "}


def report(result: dict) -> str:
    lines = [f"[preflight] {result['board']} — {result['platform']['system']} "
             f"{result['platform']['release']} · {result['cpu']['count']} CPUs · "
             f"verdict: {result['verdict'].upper()}"]
    for c in result["checks"]:
        lines.append(f"[preflight]   {MARK[c['status']]}  {c['name']:<9s} {c['detail']}")
    return "\n".join(lines)


JSON_PREFIX = "[preflight-json] "


def emit(result: dict) -> None:
    """Print the human block and, after it, one line a tool can read exactly."""
    print(report(result))
    print(JSON_PREFIX + json.dumps(result, separators=(",", ":")))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threads", type=int, default=None,
                   help="the thread count the measurement will use")
    p.add_argument("--cpus", default=None, metavar="LIST",
                   help="pin to these CPUs first, e.g. 0-1 (Linux only)")
    p.add_argument("--require-locked", action="store_true",
                   help="exit non-zero unless the verdict is 'locked'")
    p.add_argument("--json", action="store_true", help="only the machine-readable line")
    a = p.parse_args(argv)

    if a.cpus:
        try:
            pin(a.cpus)
        except (ValueError, RuntimeError) as e:
            print(f"[preflight] could not pin to {a.cpus}: {e}", file=sys.stderr)
            # 3, not 2: argparse exits 2 for an unknown flag, and a caller that
            # cannot tell "I asked for something impossible" from "I asked for
            # something that does not exist" learns the wrong lesson.
            return 3
    result = check(a.threads)
    if a.json:
        print(JSON_PREFIX + json.dumps(result, separators=(",", ":")))
    else:
        emit(result)
    if a.require_locked and result["verdict"] != LOCKED:
        print(f"[preflight] refusing: the machine is {result['verdict']}, and this "
              "measurement was asked to be reportable", file=sys.stderr)
        for reason in result["blockers"]:
            print(f"[preflight]   {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
