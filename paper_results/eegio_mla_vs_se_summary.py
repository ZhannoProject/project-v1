"""
Summarise the paired EEG-IO tests written by eegio_mla_vs_se.py.

Per seed: participant-paired MLA - SE differences (median, how many participants favour MLA,
Wilcoxon signed-rank p). Across seeds: mean +- SD of the per-seed median scores.
Also compares host MLA scores with the Pi 5 runs of the same models, as a parity check.

    ../env/bin/python paper_results/eegio_mla_vs_se_summary.py
"""

import json
import math
import statistics as st
import sys
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "eegio_tests"
# metric, lower_is_better, label
METRICS = [("blink_amplitude_ratio", True, "blink amplitude ratio (toward 1 is better)"),
           ("background_rmse_uv", True, "background RMSE uV (lower alters less)"),
           ("blink_count_reduction_adjusted_pct", False, "adjusted blink-count reduction %"),
           ("suppression_db", False, "blink-interval suppression dB")]
PI = {72: "eegio_20261121_215606_531612_834961af.json", 88: "eegio_20261121_184228_243724_e101c8d6.json"}


def value(row, key):
    v = row.get(key)
    if isinstance(v, dict):
        v = v.get("value", v.get("rate"))
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None


def by_subject(result, key):
    return {r["subject"]: value(r, key) for r in result["rows"]}


def main():
    made = json.loads((ROOT / "paper_results" / "eegio_mla_vs_se_tests.json").read_text())
    per_seed = {}
    for seed, name in sorted(made.items(), key=lambda kv: int(kv[0])):
        d = json.loads((TESTS / name).read_text())
        se, mla = d["results"]
        assert se["run"].startswith("tcn_se") and "mla" in mla["run"], (se["run"], mla["run"])
        print(f"\nseed {seed}  ({name})")
        per_seed[int(seed)] = {}
        for key, lower, label in METRICS:
            a, b = by_subject(se, key), by_subject(mla, key)
            subs = [s for s in a if a[s] is not None and b.get(s) is not None]
            diff = [b[s] - a[s] for s in subs]
            better = sum((x < 0) if lower else (x > 0) for x in diff)
            p = wilcoxon(diff).pvalue if any(diff) else float("nan")
            med_se, med_mla = st.median(a[s] for s in subs), st.median(b[s] for s in subs)
            per_seed[int(seed)][key] = (med_se, med_mla)
            print(f"  {label:44} SE {med_se:8.3f}  MLA {med_mla:8.3f}  median diff {st.median(diff):+8.3f}  "
                  f"MLA better {better}/{len(subs)}  p={p:.3g}")
    print("\nacross seeds (mean +- SD of per-seed medians)")
    for key, _, label in METRICS:
        se = [v[key][0] for v in per_seed.values()]
        mla = [v[key][1] for v in per_seed.values()]
        sd = lambda x: f" +- {st.stdev(x):.3f}" if len(x) > 1 else ""
        print(f"  {label:44} SE {st.mean(se):8.3f}{sd(se)}   MLA {st.mean(mla):8.3f}{sd(mla)}")
    print("\nhost vs Pi 5 for the MLA models (median scores; the adjusted count differs by design:"
          " the Pi test set its own threshold)")
    for seed, pi in PI.items():
        if str(seed) not in made:
            continue
        host = json.loads((TESTS / made[str(seed)]).read_text())["results"][1]["summary"]
        pid = json.loads((TESTS / pi).read_text())["results"][0]["summary"]
        for key in ("blink_amplitude_ratio", "background_rmse_uv", "suppression_db"):
            print(f"  s{seed} {key:24} host {host[key]['median']:.4f}  Pi {pid[key]['median']:.4f}")


if __name__ == "__main__":
    main()
