"""Run: python -m unittest discover -s tests -p 'test_dataloader_leakage.py'.

Leakage is the one bug in a denoising study that makes every number in the
paper wrong while everything still runs green, so the invariants live here
rather than in a reviewer's head.

Two separate things are checked:

1. **The split itself.** ``_split_indices`` slices ONE permutation into three
   contiguous ranges, so disjointness is structural -- it cannot fail for any
   n, ratio or seed, and the sweep below says so explicitly.

2. **That the seed reaches the consumers.** This is where the real bug was.
   ``master_seed`` *defines* the partition, so a model trained at one seed and
   evaluated against another seed's "test" split is scored on its own training
   data: the splits of two seeds overlap ~85%. eval.py used to hardcode 42 for
   metrics and 123 for figures regardless of what the run was trained with.
"""
import hashlib
import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.Dataloader import EEGDenoiseDataset, build_dataloaders

RATIOS = (0.7, 0.15, 0.15)
SPLITS = ("train", "val", "test")


def unique_epochs(n: int, t: int = 512, tag: int = 0) -> np.ndarray:
    """Epochs that can be told apart: row i is drawn from its own seeded stream."""
    return np.stack([np.random.default_rng(10_000 * tag + i).normal(0, 1, t)
                     for i in range(n)]).astype(np.float32)


def fingerprints(rows) -> set:
    return {hashlib.sha1(np.ascontiguousarray(r).tobytes()).hexdigest() for r in rows}


class SplitIndexTests(unittest.TestCase):
    """The partition is structural: one permutation, three contiguous slices."""

    def test_the_three_splits_are_an_exact_disjoint_partition(self):
        for n_total, ratios, seed in itertools.product(
                (10, 11, 99, 100, 3400, 4514, 5598),
                (RATIOS, (0.8, 0.1, 0.1), (0.34, 0.33, 0.33), (0.98, 0.01, 0.01)),
                (0, 1, 42, 123, 2 ** 31 - 1)):
            with self.subTest(n_total=n_total, ratios=ratios, seed=seed):
                sets = {s: set(EEGDenoiseDataset._split_indices(n_total, ratios, s, seed).tolist())
                        for s in SPLITS}
                for a, b in itertools.combinations(SPLITS, 2):
                    self.assertEqual(sets[a] & sets[b], set(), f"{a} and {b} overlap")
                # Every epoch used exactly once: a partition, not just disjoint slices.
                self.assertEqual(set().union(*sets.values()), set(range(n_total)))
                self.assertEqual(sum(len(v) for v in sets.values()), n_total)

    def test_the_real_dataset_sizes_split_as_the_readme_claims(self):
        """4514 clean EEG / 3400 EOG / 5598 EMG epochs, 70/15/15."""
        for n_total in (4514, 3400, 5598):
            sizes = [len(EEGDenoiseDataset._split_indices(n_total, RATIOS, s, 42)) for s in SPLITS]
            self.assertEqual(sum(sizes), n_total)
            self.assertAlmostEqual(sizes[0] / n_total, 0.70, delta=0.01)

    def test_the_same_seed_is_reproducible(self):
        a = EEGDenoiseDataset._split_indices(4514, RATIOS, "test", 42)
        b = EEGDenoiseDataset._split_indices(4514, RATIOS, "test", 42)
        np.testing.assert_array_equal(a, b)

    def test_a_different_seed_is_a_different_split(self):
        """Why the seed has to be carried: it is not a detail, it is the split."""
        a = EEGDenoiseDataset._split_indices(4514, RATIOS, "test", 42)
        b = EEGDenoiseDataset._split_indices(4514, RATIOS, "test", 1)
        self.assertFalse(np.array_equal(a, b))

    def test_two_seeds_splits_overlap_almost_completely(self):
        """The magnitude of the hazard: evaluating a seed-1 model on seed-42's
        'test' split scores it mostly on epochs it trained on."""
        test42 = set(EEGDenoiseDataset._split_indices(4514, RATIOS, "test", 42).tolist())
        train1 = set(EEGDenoiseDataset._split_indices(4514, RATIOS, "train", 1).tolist())
        val1 = set(EEGDenoiseDataset._split_indices(4514, RATIOS, "val", 1).tolist())
        seen = test42 & (train1 | val1)
        self.assertGreater(len(seen) / len(test42), 0.75,
                           "if this ever drops, the 85% figure in the docs needs revisiting")


class DatasetContentTests(unittest.TestCase):
    """Index disjointness is necessary but not sufficient -- check the epochs."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="leak_content_"))
        # Same proportions as the real 4514 / 3400 / 5598.
        for name, n, tag in (("EEG", 500, 1), ("EOG", 380, 2), ("EMG", 620, 3)):
            np.save(cls.tmp / f"{name}_all_epochs.npy", unique_epochs(n, tag=tag))

    def _streams(self, artifact: str, split: str, seed: int) -> dict:
        ds = EEGDenoiseDataset(data_dir=str(self.tmp), artifact_type=artifact,
                               split=split, master_seed=seed)
        out = {"clean": fingerprints(ds.clean)}
        if artifact == "BOTH":
            out["eog"] = fingerprints(ds.artifact_eog)
            out["emg"] = fingerprints(ds.artifact_emg)
        else:
            out["artifact"] = fingerprints(ds.artifact)
        return out

    def test_no_epoch_is_ever_shared_between_splits(self):
        for artifact, seed in itertools.product(("EOG", "EMG", "BOTH"), (42, 1, 7)):
            with self.subTest(artifact=artifact, seed=seed):
                per = {s: self._streams(artifact, s, seed) for s in SPLITS}
                for stream in per["train"]:
                    for a, b in itertools.combinations(SPLITS, 2):
                        shared = per[a][stream] & per[b][stream]
                        self.assertEqual(shared, set(),
                                         f"{artifact} {stream}: {a} and {b} share "
                                         f"{len(shared)} epoch(s)")

    def test_no_epoch_is_repeated_within_a_split(self):
        """choice(..., replace=False) -- a duplicated epoch would silently
        reweight the training set."""
        for artifact in ("EOG", "EMG", "BOTH"):
            ds = EEGDenoiseDataset(data_dir=str(self.tmp), artifact_type=artifact,
                                   split="train", master_seed=42)
            self.assertEqual(len(fingerprints(ds.clean)), len(ds.clean))

    def test_selected_epochs_come_only_from_that_splits_index_set(self):
        """Guards the pairing step: choice() draws from clean_idx, so it cannot
        reach an epoch belonging to another split."""
        pool = unique_epochs(500, tag=1)
        for split in SPLITS:
            idx = EEGDenoiseDataset._split_indices(500, RATIOS, split, 42)
            allowed = fingerprints(pool[np.sort(idx)])
            ds = EEGDenoiseDataset(data_dir=str(self.tmp), artifact_type="EOG",
                                   split=split, master_seed=42)
            self.assertTrue(fingerprints(ds.clean) <= allowed)

    def test_clean_and_artifact_are_split_on_independent_permutations(self):
        """Different sub-seeds (master_seed vs +99), so the pairing is not
        biased by index correlation between the two arrays."""
        clean = EEGDenoiseDataset._split_indices(380, RATIOS, "test", 42)
        art = EEGDenoiseDataset._split_indices(380, RATIOS, "test", 42 + 99)
        self.assertFalse(np.array_equal(clean, art))

    def test_truncation_drops_epochs_but_never_relocates_them(self):
        """n = min(len(clean_idx), len(art_idx)) leaves some clean epochs unused;
        an unused epoch must stay unused, not reappear in another split."""
        used = {s: self._streams("EOG", s, 42)["clean"] for s in SPLITS}
        everything = set().union(*used.values())
        self.assertEqual(sum(len(v) for v in used.values()), len(everything))


class SeedPropagationTests(unittest.TestCase):
    """The consumers must evaluate on the split their run was held out from."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="leak_seed_"))
        for name, n, tag in (("EEG", 500, 1), ("EOG", 380, 2), ("EMG", 620, 3)):
            np.save(cls.tmp / f"{name}_all_epochs.npy", unique_epochs(n, tag=tag))

    def test_eval_honours_the_seed_it_is_given(self):
        """evaluate_per_snr's `seed` must reach the dataset, or passing it is
        decoration."""
        import eval as eval_mod
        seen = []
        real = eval_mod.EEGDenoiseDataset

        def spy(**kw):
            seen.append(kw.get("master_seed"))
            return real(**kw)

        eval_mod.EEGDenoiseDataset = spy
        try:
            import torch.nn as nn
            model = nn.Identity()
            model.eval = lambda: model
            model.to = lambda *a, **k: model
            eval_mod.evaluate_per_snr(model, str(self.tmp), artifact_type="EOG",
                                      snr_list=[-3], batch_size=8, seed=7)
        finally:
            eval_mod.EEGDenoiseDataset = real
        self.assertEqual(seen, [7])

    def test_no_eval_entry_point_hardcodes_a_split_seed(self):
        """The regression itself: metrics were pinned to 42 and the figures to
        123, whatever the run was trained with."""
        source = Path(__file__).resolve().parent.parent / "eval.py"
        text = source.read_text()
        self.assertNotIn("master_seed=123", text)
        for call in ("master_seed=42",):
            self.assertNotIn(call, text, "a split seed is pinned in eval.py again")
        # Every dataset built for evaluation takes the resolved seed.
        self.assertEqual(text.count("master_seed=seed"), 3)

    def test_identify_loss_rebuilds_the_runs_own_split(self):
        source = Path(__file__).resolve().parent.parent / "trainer" / "identify_loss.py"
        text = source.read_text()
        self.assertIn("master_seed", text,
                      "identify_loss must rebuild the split with the run's recorded seed")

    def test_resolve_master_seed_prefers_what_the_run_recorded(self):
        from eval import DEFAULT_MASTER_SEED, resolve_master_seed
        self.assertEqual(resolve_master_seed({"run_config": {"master_seed": 5}}), 5)
        self.assertEqual(resolve_master_seed({"run_config": {"master_seed": 5}}, override=9), 9)
        # Nothing recorded anywhere: the historical default, and the caller is warned.
        self.assertEqual(resolve_master_seed({}), DEFAULT_MASTER_SEED)

    def test_resolve_master_seed_falls_back_to_history_json(self):
        from eval import resolve_master_seed
        hist = self.tmp / "history.json"
        hist.write_text(json.dumps({"config": {"master_seed": 11}}))
        self.assertEqual(resolve_master_seed({}, str(hist)), 11)

    def test_the_trainer_records_the_seed_so_eval_can_find_it(self):
        """resolve_master_seed reads run_config; this is the field that fills it."""
        from trainer.training import Trainer
        import inspect
        self.assertIn("master_seed", inspect.getsource(Trainer.__init__))


class EvalSplitSizeTests(unittest.TestCase):
    """The evaluators have to work with the split they are actually handed."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="leak_size_"))
        # 300 EOG epochs -> a test split of ~45, far below the old fixed stride.
        for name, n, tag in (("EEG", 400, 1), ("EOG", 300, 2)):
            np.save(cls.tmp / f"{name}_all_epochs.npy", unique_epochs(n, tag=tag))

    def test_waveform_sampling_fits_any_split_size(self):
        """visualize_samples used a fixed stride of 100, so it raised IndexError
        on every split smaller than n_samples * 100 -- including the synthetic
        data reproduce.py falls back to."""
        import matplotlib
        matplotlib.use("Agg")
        import torch.nn as nn

        import eval as eval_mod
        model = nn.Identity()
        model.eval = lambda: model
        model.to = lambda *a, **k: model

        split = EEGDenoiseDataset(data_dir=str(self.tmp), artifact_type="EOG",
                                  split="test", master_seed=42)
        self.assertLess(len(split), 100, "fixture must be smaller than the old stride")
        for n_samples in (1, 4, len(split) + 15):
            with self.subTest(n_samples=n_samples):
                out = self.tmp / f"w{n_samples}.png"
                eval_mod.visualize_samples(model, str(self.tmp), artifact_type="EOG",
                                           n_samples=n_samples, save_path=str(out), seed=42)
                self.assertTrue(out.is_file())


class LoaderWiringTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="leak_wiring_"))
        for name, n, tag in (("EEG", 500, 1), ("EOG", 380, 2), ("EMG", 620, 3)):
            np.save(cls.tmp / f"{name}_all_epochs.npy", unique_epochs(n, tag=tag))

    def test_only_the_training_loader_shuffles(self):
        tr, va, te = build_dataloaders(data_dir=str(self.tmp), artifact_type="EOG", batch_size=8)
        self.assertEqual(tr.sampler.__class__.__name__, "RandomSampler")
        for loader in (va, te):
            self.assertEqual(loader.sampler.__class__.__name__, "SequentialSampler")

    def test_the_loaders_run_in_process(self):
        """sample_rng is ordinary process state: with num_workers > 0 each worker
        would inherit a copy and redraw the same SNRs."""
        for loader in build_dataloaders(data_dir=str(self.tmp), artifact_type="EOG", batch_size=8):
            self.assertEqual(loader.num_workers, 0)

    def test_val_and_test_snr_are_deterministic_but_train_is_not(self):
        kw = dict(data_dir=str(self.tmp), artifact_type="EOG", master_seed=42)
        test = EEGDenoiseDataset(split="test", **kw)
        self.assertEqual(test[0]["snr_db"].item(), test[0]["snr_db"].item())
        train = EEGDenoiseDataset(split="train", **kw)
        draws = {train[0]["snr_db"].item() for _ in range(8)}
        self.assertGreater(len(draws), 1, "training SNR is meant to be re-drawn per access")


if __name__ == "__main__":
    unittest.main()
