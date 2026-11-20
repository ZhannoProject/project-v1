"""
Cut a small, self-contained test set out of EEGdenoiseNet for the Pi.

    python -m deploy.make_testdata --data_dir data/EEGdenoisenet/data --artifact EOG --n 100

Writes deploy/testdata/<artifact>_test<N>.npz (about 4 KB per epoch) with

    noisy        (N, 512) float32   artifact-contaminated EEG, raw scale
    clean        (N, 512) float32   ground truth, raw scale
    snr_db       (N,)     float32   the SNR each epoch was mixed at
    fs           ()       int       256 Hz
    master_seed  ()       int       the split this was cut from
    artifact     ()       str       EOG | EMG | BOTH
    split        ()       str       always "test"
    split_ratios (3,)     float32   the split proportions it was cut under

The last four are provenance, and they are the point of the file being
checkable at all.  Which epochs land in "test" is a function of master_seed,
artifact and split_ratios together, so a set cut under one combination is
training data under another.  That used to live only in the filename, which
nothing enforced and a rename could silently invalidate; now
``raspi_infer.py --testdata`` reads it out of the array and refuses a model it
does not match.  Files written before this carry no provenance and are scored
with a warning rather than a guarantee.

The epochs are the first N items of the project's own *test* split
(data/Dataloader.py) at --master_seed, which must be the seed the model being
scored was trained with: the split is a function of it, so a set cut at one
seed is training data for a run trained at another.  Files cut at a seed other
than 42 carry it in their name.  They cycle the 10-point SNR grid
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
                  master_seed: int = 42,
                  split_ratios: tuple = (0.7, 0.15, 0.15),
                  target_fs: float | None = None, checkpoint: str | None = None) -> Path:
    options = dict(artifact_type=artifact, split_ratios=split_ratios,
                   master_seed=master_seed, target_fs=target_fs)
    if checkpoint:
        from experiment.checkpoints import load_model
        from data.protocol import evaluation_data
        model, _, _ = load_model(checkpoint)
        options = evaluation_data(model)
        artifact, master_seed = options["artifact_type"], options["master_seed"]
        split_ratios, target_fs = options["split_ratios"], options["target_fs"]
    ds = EEGDenoiseDataset(data_dir=data_dir, artifact_type=artifact, split="test",
                           **{k:v for k,v in options.items() if k != "artifact_type"})
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
    # The seed is part of the file's identity, because it is part of the split's.
    # A set cut at one seed is training data for a run trained at another, and
    # scoring against it would be measuring the model on what it learned from.
    stem = f"{artifact}_test{n}" + ("" if master_seed == 42 else f"_s{master_seed}")
    if target_fs and target_fs != SAMPLE_RATE_HZ:
        stem += f"_{target_fs:g}hz"
    path = out_dir / f"{stem}.npz"
    # Provenance travels inside the file, not in its name: the name is a
    # convention and a rename breaks it silently, while this is read back by
    # the scorer and checked against the model it is about to grade.
    np.savez_compressed(path, noisy=np.stack(noisy).astype(np.float32),
                        clean=np.stack(clean).astype(np.float32),
                        snr_db=np.asarray(snr, dtype=np.float32), fs=ds.fs,
                        master_seed=int(master_seed), artifact=str(artifact),
                        split="test",
                        resampling=getattr(ds, "resampling", "offline"),
                        sampling=ds.sampling,
                        split_ratios=np.asarray(split_ratios, dtype=np.float32))
    levels = sorted(set(np.round(snr, 2)))
    print(f"[testdata] wrote {path}  ({path.stat().st_size / 1024:.0f} KB): {n} epochs x {len(noisy[0])} samples, "
          f"{artifact} test split at master_seed {master_seed}, "
          f"SNR levels {levels[0]:+.1f} .. {levels[-1]:+.1f} dB "
          f"({len(levels)} levels, {n // len(levels)} epochs each)")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="data/EEGdenoisenet/data")
    p.add_argument("--artifact", default="EOG", choices=["EOG", "EMG", "BOTH"])
    p.add_argument("--n", type=int, default=100, help="epochs to cut (multiple of 10 recommended)")
    p.add_argument("--target_fs", type=float, default=None, help="Match the checkpoint's training sample rate")
    p.add_argument("--ckpt", default=None, help="Restore the complete data protocol from this checkpoint")
    p.add_argument("--out_dir", default=str(PROJECT_ROOT / "deploy" / "testdata"))
    p.add_argument("--master_seed", type=int, default=42,
                   help="the split to cut from — it must be the seed the model being "
                        "scored was trained with, or the 'test' set is that model's "
                        "training data")
    a = p.parse_args(argv)
    make_testdata(a.data_dir, a.artifact, a.n, a.out_dir, a.master_seed, target_fs=a.target_fs, checkpoint=a.ckpt)


if __name__ == "__main__":
    main()
