"""
Recover which loss function an old run was trained with.

``val_loss`` in ``history.json`` is whatever ``loss_fn`` that run used, so it is
only comparable across runs that share it.  Runs trained after the September
2026 fix record ``loss_fn`` in ``history.json["config"]`` and in the checkpoint;
older runs record it nowhere.  This tool recovers it forensically: it rebuilds
the model from ``best.pt``, evaluates it on the run's own validation split, and
computes the val loss under every loss in the registry.  Exactly one of them
reproduces the recorded number, and that identifies the loss.

    python -m trainer.identify_loss                # runs that have no recorded loss
    python -m trainer.identify_loss --all          # also re-check recorded ones
    python -m trainer.identify_loss --run tcn_eog  # one run
    python -m trainer.identify_loss --dry-run      # report, write nothing

The result is written to ``runs/<name>/loss_fn.json`` and picked up by the GUI's
"Complete by loss" view.

Two guards keep this honest, because a wrong label here is worse than no label:

  1. **The checkpoint must be the model the history describes.**  If the
     recomputed RRMSE_t does not match the RRMSE_t recorded at the best epoch,
     the directory was re-used and ``best.pt`` holds different weights — the
     candidate losses then describe a model that never produced that
     ``val_loss``, so nothing is claimed (``status: "checkpoint-mismatch"``).
  2. **The match must be unambiguous.**  ``l1`` and ``l2_combined`` can land
     within a percent of each other; a loss is only claimed when the best
     candidate is far closer than the runner-up.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch

import metrics
from data.Dataloader import build_dataloaders
from models.variants import build_model, build_model_from_config
from experiment.checkpoints import read_checkpoint
from trainer.losses import LOSS_REGISTRY, build_loss

PROJECT_ROOT = Path(_project_root)
RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "EEGdenoisenet" / "data"
SIDECAR = "loss_fn.json"

CANDIDATES = list(LOSS_REGISTRY)
REL_TOL = 0.01          # a candidate must land within 1% of the recorded val_loss
AMBIGUITY = 5.0         # ... and be this many times closer than the runner-up
METRIC_TOL = 5e-3       # recomputed RRMSE_t must match the history to this


def recorded_loss(run_dir: Path) -> str | None:
    """``loss_fn`` as saved by the run itself (history config, else checkpoint)."""
    hist = run_dir / "history.json"
    if hist.is_file():
        try:
            cfg = (json.loads(hist.read_text()).get("config") or {})
            if cfg.get("loss_fn"):
                return cfg["loss_fn"]
        except ValueError:
            pass
    best = run_dir / "best.pt"
    if best.is_file():
        try:
            ck = read_checkpoint(best)
            return (ck.get("run_config") or {}).get("loss_fn")
        except Exception:
            pass
    return None


def artifact_of(name: str) -> str | None:
    low = name.lower()
    for token in ("both", "eog", "emg"):
        if low.endswith("_" + token) or ("_" + token + "_") in low:
            return token.upper()
    return None


def _build(ck: dict) -> torch.nn.Module:
    """Rebuild the checkpoint's model: model_config when present, else weight shapes."""
    from experiment.checkpoints import load_model
    return load_model(ck)[0]


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[dict, dict]:
    """Val loss under every candidate loss, plus the three metrics."""
    crits = {n: build_loss(n).to(device) for n in CANDIDATES}
    totals = {n: 0.0 for n in CANDIDATES}
    agg = {"rrmse_t": 0.0, "rrmse_s": 0.0, "acc": 0.0}
    n_seen = 0
    for batch in loader:
        noisy, clean = batch["noisy"].to(device), batch["clean"].to(device)
        pred = model(noisy)
        bs = noisy.size(0)
        for name, crit in crits.items():
            totals[name] += crit(pred, clean).item() * bs
        for k, v in metrics.compute_all_metrics(pred, clean).items():
            agg[k] += v * bs
        n_seen += bs
    return ({n: v / n_seen for n, v in totals.items()},
            {k: v / max(n_seen, 1) for k, v in agg.items()})


def pick(candidates: dict, target: float) -> tuple[str | None, dict]:
    """The one candidate that reproduces ``target``, or None when it is ambiguous."""
    scale = max(abs(target), 1e-8)
    ranked = sorted(((abs(v - target) / scale, n) for n, v in candidates.items()))
    (best_err, best), (second_err, _) = ranked[0], ranked[1]
    ok = best_err < REL_TOL and second_err >= AMBIGUITY * max(best_err, 2e-3)
    return (best if ok else None), {"best": best, "best_rel_err": best_err,
                                    "runner_up_rel_err": second_err}


def identify(run_dir: Path, data_dir: Path, device) -> dict:
    """Identify one run's loss. Returns the sidecar record (not written here)."""
    name = run_dir.name
    # Stamp which best.pt this verdict is about: retraining the run invalidates it,
    # and config_io ignores a sidecar whose stamp no longer matches the file.
    st = (run_dir / "best.pt").stat()
    out = {"run": name, "identified_at": datetime.now().isoformat(timespec="seconds"),
           "checkpoint_mtime": st.st_mtime, "checkpoint_size": st.st_size}

    artifact = artifact_of(name)
    if artifact is None:
        return {**out, "status": "unknown-artifact",
                "detail": "cannot tell EOG/EMG/BOTH from the directory name"}

    hist_path = run_dir / "history.json"
    if not hist_path.is_file():
        return {**out, "status": "no-history", "detail": "the run never finished"}
    hist = json.loads(hist_path.read_text())
    vl = hist.get("val_loss") or []
    if not vl:
        return {**out, "status": "no-history", "detail": "history.json has no val_loss"}
    i = min(range(len(vl)), key=vl.__getitem__)
    target, rec_rrmse_t, best_epoch = vl[i], hist["rrmse_t"][i], i + 1

    ck = read_checkpoint(run_dir / "best.pt")
    try:
        model = _build(ck).to(device)
    except Exception as e:
        return {**out, "status": "unloadable", "detail": str(e)[:160]}

    _, val_loader, _ = build_dataloaders(data_dir=str(data_dir), artifact_type=artifact,
                                         batch_size=32)
    losses, mets = evaluate(model, val_loader, device)

    out.update(artifact=artifact, best_epoch=best_epoch, checkpoint_epoch=ck.get("epoch"),
               val_loss_recorded=target, rrmse_t_recorded=rec_rrmse_t,
               rrmse_t_recomputed=mets["rrmse_t"], candidates=losses)

    # Guard 1: is best.pt the model this history describes?
    if abs(mets["rrmse_t"] - rec_rrmse_t) > METRIC_TOL:
        return {**out, "status": "checkpoint-mismatch",
                "detail": f"best.pt is at epoch {ck.get('epoch')} and scores RRMSE_t "
                          f"{mets['rrmse_t']:.4f}, but the history's best epoch "
                          f"({best_epoch}) recorded {rec_rrmse_t:.4f} — the directory was "
                          f"re-used, so the loss cannot be identified from these weights"}

    # Guard 2: is the match unambiguous?
    winner, detail = pick(losses, target)
    if winner is None:
        return {**out, "status": "ambiguous",
                "detail": f"closest is {detail['best']} at {detail['best_rel_err']*100:.2f}% "
                          f"but the runner-up is only {detail['runner_up_rel_err']*100:.2f}% away"}
    return {**out, "status": "recovered", "loss_fn": winner, "source": "recovered",
            "val_loss_recomputed": losses[winner],
            "detail": f"{winner} reproduces the recorded val_loss to "
                      f"{detail['best_rel_err']*100:.3f}%; next candidate is "
                      f"{detail['runner_up_rel_err']*100:.1f}% off"}


def main(argv=None):
    p = argparse.ArgumentParser(description="Recover the loss function of past runs")
    p.add_argument("--run", action="append", default=None,
                   help="run directory name (repeatable); default: every run in runs/")
    p.add_argument("--all", action="store_true",
                   help="also re-check runs that already record their loss")
    p.add_argument("--data_dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--dry-run", action="store_true", help="report without writing sidecars")
    args = p.parse_args(argv)

    data_dir = Path(args.data_dir)
    if not (data_dir / "EEG_all_epochs.npy").is_file():
        raise SystemExit(f"Dataset not found in {data_dir} — pass --data_dir")
    device = torch.device("cpu")

    names = args.run or sorted(d.name for d in RUNS_DIR.iterdir()
                               if d.is_dir() and (d / "best.pt").is_file())
    print(f"{'run':24s}{'loss':14s}{'source':12s} detail")
    print("-" * 100)
    written = 0
    for name in names:
        run_dir = RUNS_DIR / name
        if not (run_dir / "best.pt").is_file():
            print(f"{name:24s}{'-':14s}{'-':12s} no best.pt")
            continue
        rec = recorded_loss(run_dir)
        if rec and not args.all:
            print(f"{name:24s}{rec:14s}{'recorded':12s} the run saved its own loss_fn")
            continue
        try:
            result = identify(run_dir, data_dir, device)
        except Exception as e:                       # one bad run must not stop the sweep
            result = {"run": name, "status": "error", "detail": str(e)[:160]}
        if rec:
            result["recorded_loss_fn"] = rec
            if result.get("loss_fn") and result["loss_fn"] != rec:
                result["status"] = "conflict"
                result["detail"] = (f"the run records {rec} but the weights reproduce "
                                    f"{result['loss_fn']}")
        loss = result.get("loss_fn") or "-"
        print(f"{name:24s}{loss:14s}{result['status']:12s} {result.get('detail','')[:60]}")
        if not args.dry_run:
            (run_dir / SIDECAR).write_text(json.dumps(result, indent=1))
            written += 1
    if not args.dry_run:
        print(f"\nWrote {written} × runs/<run>/{SIDECAR}. The GUI reads these for its "
              f"'Complete by loss' view.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
