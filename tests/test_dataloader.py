"""
What the dataset promises, and the two places it promised more than it did.

The leak-free split is the guarantee everything else rests on, so it is
checked directly rather than trusted.  The rest of this file pins the
behaviour an audit found: the quirks of ``sampling="legacy"`` are asserted
*as quirks*, because every run in ``runs/`` was validated under them and a
silent change would make those metrics describe a dataset that no longer
exists; ``sampling="balanced"`` is asserted to fix them.

Run: python -m unittest discover -s tests -p 'test_dataloader.py'
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.Dataloader import (                                    # noqa: E402
    EEGDenoiseDataset, ArtifactSwitchDataset, SAMPLING_MODES,
    _load_epochs, artifact_gain, realised_snr_db, scale_artifact, worker_init_fn,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "EEGdenoisenet", "data")
HAVE_DATA = os.path.isfile(os.path.join(DATA_DIR, "EEG_all_epochs.npy"))
needs_data = unittest.skipUnless(HAVE_DATA, "EEGdenoiseNet .npy files not present")

SEED = 42
GRID = 10


# ============================================================
# The mixing arithmetic, which needs no dataset
# ============================================================
class TestMixing(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        self.clean = rng.normal(size=512).astype(np.float32)
        self.art = rng.normal(size=512).astype(np.float32) * 3.0

    def test_one_artifact_lands_on_the_snr_it_was_asked_for(self):
        for snr in (-7.0, -3.0, 0.0, 2.0):
            noise = scale_artifact(self.clean, self.art, snr)
            self.assertAlmostEqual(realised_snr_db(self.clean, noise), snr, places=4)

    def test_two_artifacts_at_s_db_land_about_3_db_below_s(self):
        # Each is calibrated against the same clean signal, so the mixture
        # holds twice the noise power: 10*log10(2) = 3.01 dB.  This is the
        # arithmetic behind snr_db being optimistic in BOTH mode.
        rng = np.random.default_rng(1)
        other = rng.normal(size=512).astype(np.float32)
        for snr in (-7.0, 0.0, 2.0):
            noise = (scale_artifact(self.clean, self.art, snr)
                     + scale_artifact(self.clean, other, snr))
            self.assertAlmostEqual(realised_snr_db(self.clean, noise), snr - 3.01, delta=0.6)

    def test_a_silent_artifact_gets_zero_gain_rather_than_dividing_by_zero(self):
        self.assertEqual(artifact_gain(self.clean, np.zeros(512, np.float32), 0.0), 0.0)


# ============================================================
# Load-time validation
# ============================================================
class TestLoadValidation(unittest.TestCase):
    """Each of these used to be reported far away from its cause, or not at all."""

    def _dir(self, **arrays):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        for name, arr in arrays.items():
            np.save(os.path.join(d.name, f"{name}_all_epochs.npy"), arr)
        return d.name

    def test_rejects_an_array_that_is_not_two_dimensional(self):
        d = self._dir(EEG=np.random.randn(20))
        with self.assertRaisesRegex(ValueError, "n_epochs, n_samples"):
            _load_epochs(d, "EEG")

    def test_rejects_an_artifact_whose_epoch_length_differs_from_the_clean_signal(self):
        d = self._dir(EEG=np.random.randn(20, 512), EOG=np.random.randn(20, 480))
        with self.assertRaisesRegex(ValueError, "epoch length"):
            _load_epochs(d, "EOG", 512)

    def test_rejects_a_degenerate_epoch_rather_than_training_on_all_zeros(self):
        bad = np.random.randn(20, 512)
        bad[7] = 0.0
        d = self._dir(EEG=bad)
        with self.assertRaisesRegex(ValueError, "RMS below"):
            _load_epochs(d, "EEG")

    def test_does_not_unpickle(self):
        # allow_pickle is off, so an object array cannot be loaded at all.
        d = self._dir(EEG=np.array([{"payload": 1}, {"payload": 2}], dtype=object))
        with self.assertRaises(ValueError):
            _load_epochs(d, "EEG")


# ============================================================
# The guarantee everything else rests on
# ============================================================
@needs_data
class TestLeakFreeSplits(unittest.TestCase):

    POOLS = {"EEG": (4514, SEED), "EOG": (3400, SEED + 99), "EMG": (5598, SEED + 199)}

    def test_the_three_splits_are_disjoint_and_cover_every_epoch(self):
        for name, (n_total, seed) in self.POOLS.items():
            sets = {s: set(EEGDenoiseDataset._split_indices(
                        n_total, (.7, .15, .15), s, seed).tolist())
                    for s in ("train", "val", "test")}
            with self.subTest(pool=name):
                self.assertEqual(sets["train"] & sets["val"], set())
                self.assertEqual(sets["train"] & sets["test"], set())
                self.assertEqual(sets["val"] & sets["test"], set())
                self.assertEqual(len(set().union(*sets.values())), n_total)

    def test_a_different_master_seed_gives_a_different_split(self):
        a = EEGDenoiseDataset._split_indices(4514, (.7, .15, .15), "test", SEED)
        b = EEGDenoiseDataset._split_indices(4514, (.7, .15, .15), "test", SEED + 1)
        self.assertNotEqual(set(a.tolist()), set(b.tolist()))


# ============================================================
# Determinism
# ============================================================
@needs_data
class TestDeterminism(unittest.TestCase):

    def test_val_is_a_pure_function_of_the_index(self):
        ds = EEGDenoiseDataset(DATA_DIR, "EOG", "val", master_seed=SEED)
        self.assertTrue(torch.equal(ds[3]["noisy"], ds[3]["noisy"]))

    def test_the_same_seed_rebuilds_the_same_val_set(self):
        a = EEGDenoiseDataset(DATA_DIR, "BOTH", "val", master_seed=SEED)
        b = EEGDenoiseDataset(DATA_DIR, "BOTH", "val", master_seed=SEED)
        for i in (0, 7, 101):
            self.assertTrue(torch.equal(a[i]["noisy"], b[i]["noisy"]))

    def test_a_different_seed_gives_different_data(self):
        a = EEGDenoiseDataset(DATA_DIR, "BOTH", "val", master_seed=SEED)
        c = EEGDenoiseDataset(DATA_DIR, "BOTH", "val", master_seed=SEED + 1)
        self.assertFalse(torch.equal(a[0]["noisy"], c[0]["noisy"]))


# ============================================================
# The two sampling modes
# ============================================================
@needs_data
class TestSamplingModes(unittest.TestCase):

    def test_the_mode_is_validated_and_recorded(self):
        with self.assertRaises(AssertionError):
            EEGDenoiseDataset(DATA_DIR, "EOG", "val", sampling="nonsense")
        for mode in SAMPLING_MODES:
            ds = EEGDenoiseDataset(DATA_DIR, "EOG", "val", sampling=mode)
            self.assertEqual(ds.sampling, mode)

    def test_legacy_gives_both_artifacts_the_same_val_snr(self):
        # Asserted as a quirk, not an intention: every run on disk was
        # validated under it, so it must not change without being chosen.
        ds = EEGDenoiseDataset(DATA_DIR, "BOTH", "val", master_seed=SEED, sampling="legacy")
        for i in range(40):
            self.assertEqual(ds._pick_snr(i), ds._pick_snr(i, second=True))

    def test_balanced_offsets_the_second_artifact_by_half_the_grid(self):
        ds = EEGDenoiseDataset(DATA_DIR, "BOTH", "val", master_seed=SEED, sampling="balanced")
        step = ds.snr_grid[1] - ds.snr_grid[0]
        for i in range(40):
            gap = abs(ds._pick_snr(i) - ds._pick_snr(i, second=True))
            self.assertAlmostEqual(gap, (GRID // 2) * step, places=6)

    def test_balanced_keeps_each_artifacts_marginal_over_the_whole_grid(self):
        ds = EEGDenoiseDataset(DATA_DIR, "BOTH", "val", master_seed=SEED, sampling="balanced")
        want = sorted(np.round(ds.snr_grid, 6))
        for second in (False, True):
            got = sorted({round(ds._pick_snr(i, second=second), 6) for i in range(GRID * 4)})
            self.assertEqual(got, want)

    def test_training_split_draws_the_two_snrs_independently_in_both_modes(self):
        for mode in SAMPLING_MODES:
            ds = EEGDenoiseDataset(DATA_DIR, "BOTH", "train", master_seed=SEED, sampling=mode)
            gaps = [abs(ds._pick_snr(i) - ds._pick_snr(i, second=True)) for i in range(50)]
            self.assertGreater(min(gaps), 0.0, mode)

    def test_balanced_stops_discarding_clean_training_epochs(self):
        pool = len(EEGDenoiseDataset._split_indices(4514, (.7, .15, .15), "train", SEED))
        legacy = EEGDenoiseDataset(DATA_DIR, "BOTH", "train", master_seed=SEED, sampling="legacy")
        fixed = EEGDenoiseDataset(DATA_DIR, "BOTH", "train", master_seed=SEED, sampling="balanced")
        self.assertLess(len(legacy), pool)          # the behaviour being fixed
        self.assertEqual(len(fixed), pool)

    def test_balanced_makes_the_snr_grid_divide_val_and_test(self):
        for art in ("EOG", "EMG", "BOTH"):
            for split in ("val", "test"):
                ds = EEGDenoiseDataset(DATA_DIR, art, split, master_seed=SEED,
                                       sampling="balanced")
                self.assertEqual(len(ds) % GRID, 0, f"{art} {split}")

    def test_a_balanced_val_set_still_pairs_one_to_one(self):
        ds = EEGDenoiseDataset(DATA_DIR, "BOTH", "val", master_seed=SEED, sampling="balanced")
        self.assertEqual(len(ds.artifact_eog), len(ds))
        self.assertEqual(len(ds.artifact_emg), len(ds))


# ============================================================
# Training at another sampling rate
# ============================================================
@needs_data
class TestTargetRate(unittest.TestCase):
    """
    EEGdenoiseNet arrives at 256 Hz; EEG-IO is 250. Training at the
    recording's rate removes the resampling step the evaluation would
    otherwise run, so the epochs can be brought to it at load time.
    """

    def test_the_default_leaves_the_epochs_exactly_as_they_arrive(self):
        a = EEGDenoiseDataset(DATA_DIR, "BOTH", "val", master_seed=SEED)
        b = EEGDenoiseDataset(DATA_DIR, "BOTH", "val", master_seed=SEED, target_fs=None)
        self.assertEqual(a[0]["noisy"].shape[-1], 512)
        self.assertEqual(a.fs, 256.0)
        self.assertIsNone(a.target_fs)
        for i in (0, 7, 101):
            self.assertTrue(torch.equal(a[i]["noisy"], b[i]["noisy"]))

    def test_250_hz_gives_500_sample_epochs(self):
        # 256 -> 250 is exactly 125/128, so 512 becomes 500 and no sample
        # lands between two others.
        for art in ("EOG", "EMG", "BOTH"):
            ds = EEGDenoiseDataset(DATA_DIR, art, "val", master_seed=SEED, target_fs=250.0)
            with self.subTest(artifact=art):
                self.assertEqual(ds[0]["noisy"].shape[-1], 500)
                self.assertEqual(ds.fs, 250.0)
                self.assertTrue(torch.isfinite(ds[0]["noisy"]).all())

    def test_the_split_does_not_move_with_the_rate(self):
        # Resampling happens to the pools, not the index sets, so which
        # epochs are in which split is untouched.
        a = EEGDenoiseDataset(DATA_DIR, "BOTH", "val", master_seed=SEED)
        b = EEGDenoiseDataset(DATA_DIR, "BOTH", "val", master_seed=SEED, target_fs=250.0)
        self.assertEqual(len(a), len(b))

    def test_the_snr_is_still_realised_exactly(self):
        # The mixing happens after the resampling, so the calibration is
        # done at the rate the model will see and stays exact.
        ds = EEGDenoiseDataset(DATA_DIR, "EOG", "val", master_seed=SEED, target_fs=250.0)
        for i in (0, 11, 97):
            item = ds[i]
            sigma = float(item["sigma"])
            clean = item["clean"][0].numpy() * sigma
            noise = (item["noisy"][0] - item["clean"][0]).numpy() * sigma
            self.assertAlmostEqual(realised_snr_db(clean, noise),
                                   float(item["snr_db"]), places=3)

    def test_a_model_runs_on_a_500_sample_epoch(self):
        from models.variants import build_model

        ds = EEGDenoiseDataset(DATA_DIR, "BOTH", "val", master_seed=SEED, target_fs=250.0)
        model = build_model("tcn_se", hidden_channels=16).eval()
        with torch.no_grad():
            out = model(ds[0]["noisy"].unsqueeze(0))
        self.assertEqual(tuple(out.shape), (1, 1, 500))


class TestResampleEpochs(unittest.TestCase):
    """The resampler itself, which needs no dataset."""

    def test_the_identity_case_copies_rather_than_resamples(self):
        from data.Dataloader import resample_epochs

        x = np.random.default_rng(0).normal(size=(4, 512)).astype(np.float32)
        np.testing.assert_array_equal(resample_epochs(x, 256.0, 256.0), x)

    def test_256_to_250_is_exactly_125_over_128(self):
        from data.Dataloader import resample_epochs

        x = np.random.default_rng(1).normal(size=(3, 512)).astype(np.float32)
        y = resample_epochs(x, 256.0, 250.0)
        self.assertEqual(y.shape, (3, 500))
        self.assertEqual(y.dtype, np.float32)

    def test_a_slow_sine_survives_the_conversion(self):
        # EEG and blink energy are low-frequency; what matters is that the
        # band the artifacts live in is not disturbed.
        from data.Dataloader import resample_epochs

        t = np.arange(512) / 256.0
        x = np.sin(2 * np.pi * 5.0 * t)[None, :].astype(np.float32)
        y = resample_epochs(x, 256.0, 250.0)
        t2 = np.arange(y.shape[1]) / 250.0
        want = np.sin(2 * np.pi * 5.0 * t2)
        self.assertLess(float(np.abs(y[0][40:-40] - want[40:-40]).max()), 0.02)


# ============================================================
# What a sample says about itself
# ============================================================
@needs_data
class TestReportedSNR(unittest.TestCase):

    def _measured(self, item):
        sigma = float(item["sigma"])
        clean = item["clean"][0].numpy() * sigma
        noise = (item["noisy"][0].numpy() - item["clean"][0].numpy()) * sigma
        return realised_snr_db(clean, noise)

    def test_snr_mixture_db_is_the_snr_the_sample_actually_has(self):
        for art in ("EOG", "EMG", "BOTH"):
            ds = EEGDenoiseDataset(DATA_DIR, art, "val", master_seed=SEED)
            for i in (0, 11, 97):
                self.assertAlmostEqual(self._measured(ds[i]),
                                       float(ds[i]["snr_mixture_db"]), places=3)

    def test_one_artifact_realises_exactly_what_snr_db_asked_for(self):
        ds = EEGDenoiseDataset(DATA_DIR, "EOG", "val", master_seed=SEED)
        for i in (0, 11, 97):
            self.assertAlmostEqual(float(ds[i]["snr_mixture_db"]),
                                   float(ds[i]["snr_db"]), places=3)

    def test_two_artifacts_realise_about_3_db_below_snr_db(self):
        ds = EEGDenoiseDataset(DATA_DIR, "BOTH", "val", master_seed=SEED)
        off = [float(ds[i]["snr_mixture_db"]) - float(ds[i]["snr_db"]) for i in range(60)]
        self.assertAlmostEqual(float(np.mean(off)), -3.01, delta=0.5)
        self.assertLess(max(off), 0.0)


# ============================================================
# Worker streams
# ============================================================
@needs_data
class TestWorkerStreams(unittest.TestCase):

    def test_each_worker_id_gets_its_own_augmentation_stream(self):
        ds = EEGDenoiseDataset(DATA_DIR, "EOG", "train", master_seed=SEED)
        draws = []
        for worker in range(4):
            ds.reseed_for_worker(worker)
            draws.append(tuple(float(ds.sample_rng.uniform(-7, 2)) for _ in range(6)))
        self.assertEqual(len(set(draws)), 4)

    def test_the_hook_is_a_no_op_on_an_object_without_the_method(self):
        # It reads the worker info torch provides; outside a worker there is
        # none, and it must not raise.
        worker_init_fn(0)


# ============================================================
# The switch probe's truncation
# ============================================================
@needs_data
class TestProbeTruncation(unittest.TestCase):

    def probe(self, **kw):
        kw.setdefault("master_seed", SEED)
        kw.setdefault("probe_seed", 0)
        return ArtifactSwitchDataset(DATA_DIR, split="test", **kw)

    def test_a_short_probe_is_the_head_of_the_full_one(self):
        full = self.probe()
        small = self.probe(n_epochs=64)
        self.assertEqual(len(small), 64)
        np.testing.assert_array_equal(small.clean, full.clean[:64])
        np.testing.assert_array_equal(small.switch_idx, full.switch_idx[:64])
        np.testing.assert_array_equal(small.snr_eog, full.snr_eog[:64])
        np.testing.assert_array_equal(small.snr_emg, full.snr_emg[:64])
        np.testing.assert_array_equal(small.eog_alt, full.eog_alt[:64])

    def test_a_truncated_probe_is_still_exactly_balanced(self):
        for n in (64, 65, 100):
            p = self.probe(n_epochs=n)
            self.assertEqual(int(p.eog_first.sum()), n // 2, f"n_epochs={n}")

    def test_n_epochs_above_the_split_size_is_the_whole_split(self):
        full = self.probe()
        self.assertEqual(len(self.probe(n_epochs=len(full) + 1000)), len(full))

    def test_the_control_differs_from_the_switch_even_for_one_epoch(self):
        # The collision fallback is drawn over the full split, so a one-epoch
        # probe still gets a genuinely different alternate epoch.
        a = self.probe(n_epochs=1)
        b = self.probe(n_epochs=1, control=True)
        self.assertFalse(torch.equal(a[0]["noisy"], b[0]["noisy"]))


if __name__ == "__main__":
    unittest.main()
