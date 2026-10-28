"""
The EEG-IO reader, against files written in EEG-IO's own on-disk format.

The real recordings are not redistributable, so every test here builds its
own fixtures with ``write_synthetic_recording`` -- which writes the same
';'-delimited data file and ','-delimited label file the dataset ships, with
the same ``corrupt,<n>`` convention.  What is being tested is therefore the
parsing, the masks, the windowing and the splits, not the signal.

Run: python -m unittest discover -s tests -p 'test_eegio.py'
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.EEGIO import (                                           # noqa: E402
    EEGIO_CHANNELS, EEGIO_FS, PROJECT_FS, SOFT_BLINK_CODE, EEGIOWindows,
    build_eegio_loader, list_subjects, read_recording, require_dataset,
    resample_to, split_subjects, write_synthetic_recording,
)


class _Fixture(unittest.TestCase):
    """A directory of synthetic subjects, torn down with the test."""

    N_SUBJECTS = 4
    DURATION = 20.0

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name
        self.recs = [
            write_synthetic_recording(self.dir, f"S{i:02d}", duration_s=self.DURATION,
                                      n_blinks=6, n_soft=2, seed=i)
            for i in range(self.N_SUBJECTS)
        ]


# ============================================================
# Reading the format
# ============================================================
class TestReader(_Fixture):

    def test_round_trips_a_recording_through_the_on_disk_format(self):
        rec = read_recording(self.dir, "S00")
        self.assertEqual(rec.subject, "S00")
        self.assertEqual(rec.fs, EEGIO_FS)
        self.assertEqual(rec.channels, EEGIO_CHANNELS)
        self.assertEqual(rec.signal.shape, (2, int(self.DURATION * EEGIO_FS)))
        self.assertEqual(rec.signal.dtype, np.float32)
        self.assertAlmostEqual(rec.duration_s, self.DURATION, places=6)

    def test_timestamps_are_consistent_with_the_sampling_rate(self):
        rec = read_recording(self.dir, "S00")
        self.assertAlmostEqual(float(np.median(np.diff(rec.t))), 1.0 / EEGIO_FS, places=6)

    def test_blink_rows_are_read_and_sorted(self):
        rec = read_recording(self.dir, "S00")
        self.assertEqual(len(rec.blinks), 8)                      # 6 hard + 2 soft
        self.assertTrue(np.all(np.diff(rec.blinks[:, 0]) >= 0))

    def test_soft_blinks_are_separated_by_the_documented_code(self):
        rec = read_recording(self.dir, "S00")
        hard = rec.blink_times(include_soft=False)
        both = rec.blink_times(include_soft=True)
        self.assertEqual(len(hard), 6)
        self.assertEqual(len(both), 8)
        self.assertEqual(int((rec.blinks[:, 1] == SOFT_BLINK_CODE).sum()), 2)

    def test_blink_samples_land_inside_the_recording(self):
        rec = read_recording(self.dir, "S00")
        idx = rec.blink_samples(include_soft=True)
        self.assertEqual(len(idx), 8)
        self.assertTrue((idx >= 0).all() and (idx < rec.n_samples).all())

    def test_lists_subjects_from_the_data_files(self):
        self.assertEqual(list_subjects(self.dir),
                         [f"S{i:02d}" for i in range(self.N_SUBJECTS)])

    def test_a_missing_label_file_is_named(self):
        os.remove(os.path.join(self.dir, "S01_labels.csv"))
        with self.assertRaisesRegex(FileNotFoundError, "S01_labels.csv"):
            read_recording(self.dir, "S01")


class TestCorruptIntervals(_Fixture):

    def test_an_interval_is_masked_over_its_whole_span(self):
        write_synthetic_recording(self.dir, "C0", duration_s=20.0, n_blinks=4,
                                  n_soft=0, corrupt=[(5.0, 7.0)], seed=9)
        rec = read_recording(self.dir, "C0")
        mask = rec.corrupt_mask()
        self.assertEqual(len(rec.corrupt), 1)
        self.assertTrue(mask[int(5.5 * EEGIO_FS)])
        self.assertFalse(mask[int(2.0 * EEGIO_FS)])
        self.assertFalse(mask[int(9.0 * EEGIO_FS)])

    def test_an_end_of_minus_one_means_to_the_end_of_the_recording(self):
        # The dataset's own convention, resolved by the reader so that no
        # consumer has to know it.
        write_synthetic_recording(self.dir, "C1", duration_s=12.0, n_blinks=3,
                                  n_soft=0, corrupt=[(8.0, -1)], seed=3)
        rec = read_recording(self.dir, "C1")
        self.assertAlmostEqual(rec.corrupt[0, 1], float(rec.t[-1]), places=6)
        self.assertTrue(rec.corrupt_mask()[-1])

    def test_a_file_without_a_corrupt_header_parses_as_all_blinks(self):
        rec = read_recording(self.dir, "S00")
        self.assertEqual(len(rec.corrupt), 0)
        self.assertFalse(rec.corrupt_mask().any())


# ============================================================
# Rate conversion
# ============================================================
class TestResampling(unittest.TestCase):

    def test_250_to_256_gives_the_expected_length(self):
        x = np.random.default_rng(0).normal(size=(2, 2500)).astype(np.float32)
        y = resample_to(x, EEGIO_FS, PROJECT_FS)
        self.assertEqual(y.shape[0], 2)
        self.assertEqual(y.shape[1], round(2500 * PROJECT_FS / EEGIO_FS))
        self.assertEqual(y.dtype, np.float32)

    def test_equal_rates_are_a_no_op(self):
        x = np.random.default_rng(1).normal(size=(1, 300)).astype(np.float32)
        np.testing.assert_array_equal(resample_to(x, 250.0, 250.0), x)

    def test_a_slow_sine_survives_the_rate_change(self):
        # Blink energy is low-frequency, so what matters is that the band the
        # artifact lives in is not disturbed by the conversion.
        n = 2500
        t = np.arange(n) / EEGIO_FS
        x = np.sin(2 * np.pi * 3.0 * t)[None, :].astype(np.float32)
        y = resample_to(x, EEGIO_FS, PROJECT_FS)
        t2 = np.arange(y.shape[1]) / PROJECT_FS
        want = np.sin(2 * np.pi * 3.0 * t2)
        mid = slice(100, -100)                       # ignore polyphase edges
        self.assertLess(float(np.abs(y[0][mid] - want[mid]).max()), 0.02)


# ============================================================
# Subject-level splits
# ============================================================
class TestSubjectSplits(unittest.TestCase):

    SUBS = [f"S{i:02d}" for i in range(20)]

    def test_no_subject_appears_in_two_splits(self):
        s = split_subjects(self.SUBS, seed=42)
        self.assertEqual(set(s["train"]) & set(s["val"]), set())
        self.assertEqual(set(s["train"]) & set(s["test"]), set())
        self.assertEqual(set(s["val"]) & set(s["test"]), set())

    def test_every_subject_is_placed(self):
        s = split_subjects(self.SUBS, seed=42)
        self.assertEqual(sorted(s["train"] + s["val"] + s["test"]), sorted(self.SUBS))

    def test_the_split_is_reproducible_and_seed_dependent(self):
        a = split_subjects(self.SUBS, seed=42)
        self.assertEqual(a, split_subjects(self.SUBS, seed=42))
        self.assertNotEqual(a, split_subjects(self.SUBS, seed=43))

    def test_ratios_must_sum_to_one(self):
        with self.assertRaises(AssertionError):
            split_subjects(self.SUBS, ratios=(0.5, 0.2, 0.2))


# ============================================================
# The windowed dataset
# ============================================================
class TestWindows(_Fixture):

    def ds(self, **kw):
        kw.setdefault("target_fs", PROJECT_FS)
        return EEGIOWindows(self.dir, **kw)

    def test_window_shape_and_normalisation_match_the_training_contract(self):
        ds = self.ds(win=512, hop=256)
        item = ds[0]
        self.assertEqual(tuple(item["noisy"].shape), (1, 512))
        self.assertEqual(item["noisy"].dtype, torch.float32)
        # Population std, as numpy computes it and as EEGDenoiseDataset
        # divides by; torch's .std() defaults to the sample estimator and so
        # reads 1.001 on a 512-sample window.
        self.assertAlmostEqual(float(item["noisy"].std(unbiased=False)), 1.0, places=3)
        self.assertGreater(float(item["sigma"]), 0.0)

    def test_there_is_no_clean_key(self):
        # The point of the module: a real recording has no target, and a
        # scoring loop must fail rather than invent one.
        self.assertNotIn("clean", self.ds()[0])

    def test_two_channels_can_be_requested(self):
        ds = self.ds(channels=("Fp1", "Fp2"), win=256, hop=256)
        self.assertEqual(tuple(ds[0]["noisy"].shape), (2, 256))
        self.assertEqual(ds.channel_names, ("Fp1", "Fp2"))

    def test_channels_can_be_given_as_indices(self):
        ds = self.ds(channels=(1,), win=256, hop=256)
        self.assertEqual(ds.channel_names, ("Fp2",))

    def test_window_count_follows_win_and_hop(self):
        per_rec = int(self.DURATION * PROJECT_FS)
        for win, hop in ((512, 256), (256, 128), (512, 512)):
            ds = self.ds(win=win, hop=hop, drop_corrupt=False)
            expect = self.N_SUBJECTS * len(range(0, per_rec - win + 1, hop))
            self.assertEqual(len(ds), expect, f"win={win} hop={hop}")

    def test_native_rate_is_available(self):
        ds = self.ds(target_fs=None, win=500, hop=500)
        self.assertEqual(ds.fs, EEGIO_FS)
        self.assertEqual(tuple(ds[0]["noisy"].shape), (1, 500))

    def test_the_blink_mask_marks_the_annotated_instants(self):
        ds = self.ds(win=512, hop=256, drop_corrupt=False, blink_halfwidth_s=0.2)
        total = sum(int(ds[i]["blink_mask"].sum()) for i in range(len(ds)))
        self.assertGreater(total, 0)
        # Some window somewhere must contain a whole blink event.
        self.assertTrue(any(int(ds[i]["n_blinks"]) >= 1 for i in range(len(ds))))

    def test_soft_blinks_only_enter_the_mask_when_asked_for(self):
        hard = self.ds(win=512, hop=512, drop_corrupt=False, include_soft=False)
        both = self.ds(win=512, hop=512, drop_corrupt=False, include_soft=True)
        n_hard = sum(int(hard[i]["blink_mask"].sum()) for i in range(len(hard)))
        n_both = sum(int(both[i]["blink_mask"].sum()) for i in range(len(both)))
        self.assertGreater(n_both, n_hard)

    def test_a_zero_halfwidth_marks_single_samples(self):
        ds = self.ds(win=512, hop=256, drop_corrupt=False, blink_halfwidth_s=0.0)
        runs = sum(int(ds[i]["n_blinks"]) for i in range(len(ds)))
        marked = sum(int(ds[i]["blink_mask"].sum()) for i in range(len(ds)))
        self.assertEqual(runs, marked)

    def test_corrupt_windows_are_dropped_and_counted(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        write_synthetic_recording(d.name, "S00", duration_s=20.0, n_blinks=5,
                                  n_soft=0, corrupt=[(4.0, 9.0)], seed=1)
        kept = EEGIOWindows(d.name, win=512, hop=256, drop_corrupt=True)
        allw = EEGIOWindows(d.name, win=512, hop=256, drop_corrupt=False)
        self.assertLess(len(kept), len(allw))
        self.assertEqual(kept.dropped_corrupt, len(allw) - len(kept))
        self.assertEqual(allw.dropped_corrupt, 0)

    def test_getitem_is_a_pure_function_of_the_index(self):
        ds = self.ds(win=512, hop=256)
        self.assertTrue(torch.equal(ds[3]["noisy"], ds[3]["noisy"]))

    def test_subjects_can_be_restricted(self):
        ds = self.ds(subjects=["S01", "S02"], win=512, hop=512)
        self.assertEqual(sorted(set(ds[i]["subject"] for i in range(len(ds)))),
                         ["S01", "S02"])

    def test_summary_describes_the_split(self):
        ds = self.ds(win=512, hop=512, drop_corrupt=False)
        s = ds.summary()
        self.assertEqual(s["fs"], PROJECT_FS)
        self.assertEqual(len(s["subjects"]), self.N_SUBJECTS)
        self.assertEqual(s["n_windows"], len(ds))
        self.assertGreater(s["n_blink_events"], 0)
        self.assertAlmostEqual(s["seconds"], self.N_SUBJECTS * self.DURATION, delta=1.0)


class TestLoader(_Fixture):

    def test_a_batch_keeps_subject_as_a_list_of_names(self):
        dl = build_eegio_loader(self.dir, batch_size=4, win=512, hop=512)
        batch = next(iter(dl))
        self.assertEqual(tuple(batch["noisy"].shape), (4, 1, 512))
        self.assertEqual(tuple(batch["blink_mask"].shape), (4, 512))
        self.assertIsInstance(batch["subject"], list)
        self.assertIsInstance(batch["subject"][0], str)

    def test_a_split_selects_subjects_not_windows(self):
        dl = build_eegio_loader(self.dir, split="train", batch_size=2,
                                win=512, hop=512, split_ratios=(0.5, 0.25, 0.25))
        want = split_subjects(list_subjects(self.dir), (0.5, 0.25, 0.25), 42)["train"]
        self.assertEqual(sorted(dl.dataset.subjects), sorted(want))


# ============================================================
# Absent data
# ============================================================
class TestMissingDataset(unittest.TestCase):

    def test_the_error_says_where_to_get_it(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(FileNotFoundError) as cm:
                require_dataset(empty)
            msg = str(cm.exception)
            self.assertIn("gnan.ece.gatech.edu", msg)
            self.assertIn("_data.csv", msg)
            self.assertIn("write_synthetic_recording", msg)


# ============================================================
# A checkpoint can actually be run on a window
# ============================================================
class TestModelCompatibility(_Fixture):

    def test_a_project_model_accepts_an_eegio_window_unchanged(self):
        # The reason target_fs defaults to 256: at the native 250 Hz a window
        # is still the right shape, but it is not the rate anything was
        # trained at.  This checks the tensor contract, not the denoising.
        from models.variants import build_model

        model = build_model("tcn_se", hidden_channels=16).eval()
        ds = self.ds_factory()
        with torch.no_grad():
            out = model(ds[0]["noisy"].unsqueeze(0))
        self.assertEqual(tuple(out.shape), (1, 1, 512))
        self.assertTrue(torch.isfinite(out).all())

    def ds_factory(self):
        return EEGIOWindows(self.dir, win=512, hop=512, target_fs=PROJECT_FS)


if __name__ == "__main__":
    unittest.main()
