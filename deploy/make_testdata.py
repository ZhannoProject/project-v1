"""
Cut a small, self-contained test set out of EEGdenoiseNet for the Pi.

    python -m deploy.make_testdata --data_dir data/EEGdenoisenet/data --artifact EOG --n 100

Writes deploy/testdata/<artifact>_test<N>.npz (about 4 KB per epoch) with

    noisy       (N, 512) float32   artifact-contaminated EEG, raw scale
    clean       (N, 512) float32   ground truth, raw scale
    snr_db      (N,)     float32   the SNR each epoch was mixed at
    fs          ()       int       256 Hz
    master_seed ()       int       the seed whose test split these came from

The epochs are the first N items of the project's own *test* split
(data/Dataloader.py, master_seed 42), so they are disjoint from anything the
models were trained or validated on, and they cycle the 10-point SNR grid
from -7 dB to 2 dB: with N a multiple of 10, every SNR is equally
represented.  They are stored at raw scale (the per-epoch normalisation the
data loader applies is undone), because that is what raspi_infer.py expects.

Score them on the Pi (or here, for a reference) with
    python deploy/raspi_infer.py <model>.onnx --testdata deploy/testdata/EOG_test100.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.Dataloader import EEGDenoiseDataset  # noqa: E402

SAMPLE_RATE_HZ = 256


def make_testdata(data_dir: str, artifact: str = "EOG", n: int = 100,
                  out_dir: str | Path = PROJECT_ROOT / "deploy" / "testdata",
                  master_seed: int = 42) -> Path:
    ds = EEGDenoiseDataset(data_dir=data_dir, artifact_type=artifact, split="test",
                           master_seed=master_seed)
    n = min(n, len(ds))
    if n % len(ds.snr_grid):
        print(f"[testdata] note: n={n} is not a multiple of the {len(ds.snr_grid)}-point SNR grid, "
              f"so SNR levels will be unevenly represented")

    noisy, clean, snr = [], [], []
    for i in range(n):
        item = ds[i]
        sigma = item["sigma"].item()                     # undo the loader's normalisation
        noisy.append(item["noisy"][0].numpy() * sigma)
        clean.append(item["clean"][0].numpy() * sigma)
        snr.append(item["snr_db"].item())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{artifact}_test{n}.npz"
    np.savez_compressed(path, noisy=np.stack(noisy).astype(np.float32),
                        clean=np.stack(clean).astype(np.float32),
                        snr_db=np.asarray(snr, dtype=np.float32), fs=SAMPLE_RATE_HZ,
                        # Which split these came from.  master_seed defines the
                        # partition, so a set cut at one seed is not held out from
                        # a run trained at another.
                        master_seed=np.asarray(master_seed, dtype=np.int64))
    levels = sorted(set(np.round(snr, 2)))
    print(f"[testdata] wrote {path}  ({path.stat().st_size / 1024:.0f} KB): {n} epochs x 512 samples, "
          f"{artifact} test split, SNR levels {levels[0]:+.1f} .. {levels[-1]:+.1f} dB "
          f"({len(levels)} levels, {n // len(levels)} epochs each)")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="data/EEGdenoisenet/data")
    p.add_argument("--artifact", default="EOG", choices=["EOG", "EMG", "BOTH"])
    p.add_argument("--n", type=int, default=100, help="epochs to cut (multiple of 10 recommended)")
    p.add_argument("--out_dir", default=str(PROJECT_ROOT / "deploy" / "testdata"))
    p.add_argument("--master_seed", type=int, default=42,
                   help="the seed whose test split to cut from; must match the master_seed of "
                        "the run you intend to score, or the epochs are not held out from it")
    a = p.parse_args(argv)
    make_testdata(a.data_dir, a.artifact, a.n, a.out_dir, master_seed=a.master_seed)


if __name__ == "__main__":
    main()
