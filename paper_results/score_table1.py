"""
Test-split scores for manuscript Table I (EEGdenoiseNet BOTH, 250 Hz, fixed input scale).

Selects runs strictly: ChannelNorm, 64 channels (52 for MLA r16), MSE, no KD, 250 Hz,
fixed scale, BOTH, and a history of all 200 epochs -- a best.pt alone does not make a
run complete (interrupted runs and their _retry2 twins both have one).

Each model is scored per test segment on three target intervals:
  full    -- all 500 samples
  se      -- [124 + 20, 500 - 62): every TCN/SE row has its whole convolution context
  common  -- [252 + 20, 500 - 62): also covers the attention control's 252-sample past
(20 = polyphase 256->250 Hz resampling warm-up; 62 = offline model's future reach.)

Writes paper_results/table1_test_scores.json. Run from project v1:
    ../env/bin/python paper_results/score_table1.py
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.Dataloader import EEGDenoiseDataset          # noqa: E402
from data.protocol import evaluation_data              # noqa: E402
from experiment.checkpoints import load_model          # noqa: E402
from gui.config_io import list_runs                    # noqa: E402
from metrics import acc, rrmse_spectral, rrmse_temporal  # noqa: E402

N = 500
INTERVALS = {"full": (0, N), "se": (124 + 20, N - 62), "common": (252 + 20, N - 62)}
SE_RUNGS = (0, 1, 2, 4, 8, 16, 32, 62, 124, 186)


def epochs_done(name):
    h = json.loads((ROOT / "runs" / name / "history.json").read_text())
    return len(h) if isinstance(h, list) else len(next(iter(h.values())))


def row_key(rec):
    v = rec["variant"]
    if v == "tcn":
        return "tcn"
    if v == "tcn_se":
        return "se_0"
    if v == "tcn_se_bi":
        return "se_off"
    if v.startswith("tcn_se_la") and v[9:].isdigit() and int(v[9:]) in SE_RUNGS:
        return f"se_{int(v[9:])}"
    if v == "tcn_mla_r16":
        return "attn"
    return None


def select():
    chosen, dupes = defaultdict(list), []
    for rec in list_runs():
        key = row_key(rec)
        if key is None:
            continue
        width = 52 if key == "attn" else 64
        if not (rec.get("artifact") == "BOTH" and rec.get("train_fs") == 250.0
                and rec.get("input_normalization") == "fixed" and rec.get("tcn_norm") == "channel"
                and rec.get("hidden_channels") == width and rec.get("loss_fn") in (None, "mse")
                and not rec.get("kd")):
            continue
        try:
            if epochs_done(rec["name"]) != 200:
                continue
        except (OSError, ValueError, StopIteration):
            continue
        chosen[(key, rec["master_seed"])].append(rec["name"])
    picked = {}
    for (key, seed), names in sorted(chosen.items()):
        if len(names) > 1:
            dupes.append({"row": key, "seed": seed, "runs": sorted(names)})
        picked[(key, seed)] = sorted(names)[-1]   # the latest-named (a _retry2 sorts after its original)
    return picked, dupes


@torch.no_grad()
def score(name):
    model, arch, _ = load_model(ROOT / "runs" / name / "best.pt")
    model.eval()
    ds = EEGDenoiseDataset(model.run_config["data_dir"], split="test", **evaluation_data(model))
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    out = {k: {"rrmse_t": [], "rrmse_f": [], "cc": []} for k in INTERVALS}
    for batch in loader:
        p, y = model(batch["noisy"]), batch["clean"]
        assert p.shape[-1] == y.shape[-1] == N, p.shape
        for k, (a, b) in INTERVALS.items():
            pp, yy = p[..., a:b], y[..., a:b]
            out[k]["rrmse_t"] += rrmse_temporal(pp, yy).reshape(len(pp), -1).mean(-1).tolist()
            out[k]["rrmse_f"] += rrmse_spectral(pp, yy).reshape(len(pp), -1).mean(-1).tolist()
            out[k]["cc"] += acc(pp, yy).reshape(len(pp), -1).mean(-1).tolist()
    params = sum(p.numel() for p in model.parameters())
    return {"params": params, "n_segments": len(ds),
            "mean": {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in out.items()}}


def main():
    # --rows attn,tcn rescores only those rows and keeps every other saved result.
    only = None
    if "--rows" in sys.argv:
        only = set(sys.argv[sys.argv.index("--rows") + 1].split(","))
    picked, dupes = select()
    path = ROOT / "paper_results" / "table1_test_scores.json"
    results = {}
    if only is not None and path.exists():
        results = {k: v for k, v in json.loads(path.read_text())["results"].items() if v["row"] not in only}
    for (key, seed), name in picked.items():
        if only is not None and key not in only:
            continue
        print(f"{key:7} s{seed}  {name}", flush=True)
        results[f"{key}|{seed}"] = {"row": key, "seed": seed, "run": name, **score(name)}
    path.write_text(json.dumps({"made": datetime.now().isoformat(timespec="seconds"),
                                "intervals": INTERVALS, "duplicates": dupes,
                                "results": results}, indent=1))
    print("wrote", path)


if __name__ == "__main__":
    main()
