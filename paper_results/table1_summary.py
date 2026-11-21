"""
Summarise paper_results/table1_test_scores.json into Table I rows.

Mean +- sample SD over seeds (training runs are the repeat unit). Gap recovered (Eq. 1)
is computed within seed against that seed's SE L=0 and SE offline runs, then averaged.

    ../env/bin/python paper_results/table1_summary.py [interval]   # full | se | common
"""

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = ["tcn", "se_0", "se_1", "se_2", "se_4", "se_8", "se_16", "se_32", "se_62", "se_124", "se_186",
        "se_off", "attn"]


def summarise(interval):
    data = json.loads((HERE / "table1_test_scores.json").read_text())
    by = defaultdict(dict)
    for r in data["results"].values():
        by[r["row"]][r["seed"]] = r
    out = {}
    for row in ROWS:
        seeds = sorted(by.get(row, {}))
        if not seeds:
            continue
        m = {k: [by[row][s]["mean"][interval][k] for s in seeds] for k in ("rrmse_t", "rrmse_f", "cc")}
        gaps = []
        # Every row but the offline anchor: TCN and attention are placed on the same scale,
        # anchored on that seed's TCN+SE causal (0 %) and offline (100 %) runs.
        if row != "se_off":
            for s in seeds:
                if s in by["se_0"] and s in by["se_off"]:
                    e0 = by["se_0"][s]["mean"][interval]["rrmse_t"]
                    eo = by["se_off"][s]["mean"][interval]["rrmse_t"]
                    el = by[row][s]["mean"][interval]["rrmse_t"]
                    if e0 - eo > 0:
                        gaps.append(100 * (e0 - el) / (e0 - eo))
        stat = lambda v: (st.mean(v), st.stdev(v) if len(v) > 1 else None)
        out[row] = {"seeds": seeds, "params": by[row][seeds[0]]["params"],
                    **{k: stat(v) for k, v in m.items()},
                    "gap": stat(gaps) if gaps else None,
                    "runs": [by[row][s]["run"] for s in seeds]}
    return data, out


def fmt(x, d=3):
    mean, sd = x
    return f"{mean:.{d}f}" + ("" if sd is None else f" $\\pm$ {sd:.{d}f}")


if __name__ == "__main__":
    interval = sys.argv[1] if len(sys.argv) > 1 else "common"
    data, out = summarise(interval)
    print("interval", interval, data["intervals"][interval], "| duplicates:", data["duplicates"])
    for row, r in out.items():
        gap = "--" if r["gap"] is None else fmt(r["gap"], 1)
        print(f"{row:7} n={len(r['seeds'])} {r['seeds']} params={r['params']:,} | "
              f"{fmt(r['rrmse_t'])} & {fmt(r['rrmse_f'])} & {fmt(r['cc'])} & {gap}")
