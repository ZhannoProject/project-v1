"""Run: python -m unittest discover -s tests -p 'test_eeg_io.py'.

EEG-IO is request-gated and not in the repo, so these tests build fixtures in
the format the dataset documents (meagmohit/BLINK, data/EEG-IO/README.md) --
``;``-delimited data with a header row, and a labels file whose corrupt-interval
count has to be honoured to parse the blink rows correctly.  Same tactic
reproduce.py uses for a missing EEGdenoiseNet: synthesise the format, so the
reader is exercised even with no real recordings present.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.eeg_io import (BLINK_CODES, CHANNEL_NAMES, NATIVE_RATE_HZ, PROJECT_RATE_HZ, WINDOW,
                         cut_epochs, list_subjects, load_dataset, load_subject, read_labels,
                         read_signal, resample_to_project_rate, save_epochs)


def write_data(path: Path, fp1: np.ndarray, fp2: np.ndarray, *,
               fs: float = NATIVE_RATE_HZ, extra_column: bool = False) -> None:
    """Write ``S<ID>_data.csv``: ';' delimited, one header row, time;Fp1;Fp2."""
    t = np.arange(len(fp1)) / fs
    head = "time;channel1;channel2" + (";channel3" if extra_column else "")
    rows = [head]
    for i in range(len(fp1)):
        row = f"{t[i]:.6f};{fp1[i]:.6f};{fp2[i]:.6f}"
        if extra_column:
            row += ";0.0"
        rows.append(row)
    path.write_text("\n".join(rows) + "\n")


def write_labels(path: Path, blinks, corrupt=()) -> None:
    """Write ``S<ID>_labels.csv`` in the documented three-part layout."""
    lines = [f"corrupt,{len(corrupt)}"]
    lines += [f"{start},{end}" for start, end in corrupt]
    lines.append("blinks")
    lines += [f"{t},{code}" for t, code in blinks]
    path.write_text("\n".join(lines) + "\n")


def blinky_signal(n: int, blink_times, *, fs: float = NATIVE_RATE_HZ,
                  amplitude: float = 120.0, seed: int = 0) -> np.ndarray:
    """Small pink-ish background with a large negative deflection at each blink."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    sig = 5.0 * np.sin(2 * np.pi * 10.0 * t) + rng.normal(0, 2.0, n)
    for t0 in blink_times:
        # Blinks in EEG-IO are annotated at the minimum; ~200 ms wide.
        sig -= amplitude * np.exp(-0.5 * ((t - t0) / 0.1) ** 2)
    return sig


class LabelParsingTests(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="eeg_io_labels_"))

    def test_reads_blinks_corrupt_intervals_and_codes(self):
        path = self.tmp / "S01_labels.csv"
        write_labels(path, blinks=[(1.5, 0), (3.25, 1), (5.0, 2)],
                     corrupt=[(0.0, 0.5), (9.0, 9.5)])
        labels = read_labels(path, duration_s=10.0)
        self.assertEqual(labels.corrupt, [(0.0, 0.5), (9.0, 9.5)])
        np.testing.assert_allclose(labels.blink_times, [1.5, 3.25, 5.0])
        np.testing.assert_array_equal(labels.blink_codes, [0, 1, 2])

    def test_corrupt_end_of_minus_one_means_to_the_end_of_the_recording(self):
        path = self.tmp / "S02_labels.csv"
        write_labels(path, blinks=[(2.0, 0)], corrupt=[(7.0, -1)])
        self.assertEqual(read_labels(path, duration_s=12.0).corrupt, [(7.0, 12.0)])

    def test_the_corrupt_count_is_what_separates_intervals_from_blinks(self):
        """An interval row is '<start>,<end>' and a blink row is '<time>,<code>' --
        only the promised count tells them apart, so a miscount misreads both."""
        path = self.tmp / "S03_labels.csv"
        write_labels(path, blinks=[(4.0, 0), (6.0, 1)], corrupt=[(1.0, 2.0)])
        labels = read_labels(path, duration_s=10.0)
        self.assertEqual(len(labels.corrupt), 1)
        np.testing.assert_allclose(labels.blink_times, [4.0, 6.0])

    def test_blinks_are_returned_in_time_order(self):
        path = self.tmp / "S04_labels.csv"
        write_labels(path, blinks=[(9.0, 0), (2.0, 1), (5.0, 0)])
        labels = read_labels(path, duration_s=10.0)
        np.testing.assert_allclose(labels.blink_times, [2.0, 5.0, 9.0])
        np.testing.assert_array_equal(labels.blink_codes, [1, 0, 0])

    def test_soft_blinks_can_be_excluded(self):
        path = self.tmp / "S05_labels.csv"
        write_labels(path, blinks=[(1.0, 0), (2.0, 2), (3.0, 1)])
        labels = read_labels(path, duration_s=10.0)
        np.testing.assert_allclose(labels.blinks(include_soft=True), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(labels.blinks(include_soft=False), [1.0, 3.0])

    def test_an_unknown_blink_code_is_refused(self):
        """Codes are 0/1/2; a 3 means the file is not what we think it is."""
        path = self.tmp / "S06_labels.csv"
        write_labels(path, blinks=[(1.0, 3)])
        with self.assertRaisesRegex(ValueError, "blink code 3"):
            read_labels(path, duration_s=10.0)

    def test_a_count_larger_than_the_intervals_given_is_refused(self):
        """Promising 2 intervals and giving 1 would otherwise silently swallow
        the 'blinks' marker and read the first blink as an interval."""
        path = self.tmp / "S07_labels.csv"
        path.write_text("corrupt,2\n1.0,2.0\nblinks\n4.0,0\n")
        with self.assertRaisesRegex(ValueError, "still promised"):
            read_labels(path, duration_s=10.0)

    def test_a_file_truncated_mid_interval_is_refused(self):
        path = self.tmp / "S07b_labels.csv"
        path.write_text("corrupt,3\n1.0,2.0\n")
        with self.assertRaisesRegex(ValueError, "promised 2 more"):
            read_labels(path, duration_s=10.0)

    def test_blank_lines_are_tolerated(self):
        path = self.tmp / "S08_labels.csv"
        path.write_text("corrupt,1\n\n1.0,2.0\n\nblinks\n\n4.0,0\n\n")
        labels = read_labels(path, duration_s=10.0)
        self.assertEqual(labels.corrupt, [(1.0, 2.0)])
        np.testing.assert_allclose(labels.blink_times, [4.0])

    def test_every_documented_code_is_known(self):
        self.assertEqual(BLINK_CODES, {0: "normal", 1: "stimulated", 2: "soft"})


class SignalReadingTests(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="eeg_io_signal_"))

    def test_reads_semicolon_delimited_data_past_the_header(self):
        fp1 = np.arange(10, dtype=float)
        fp2 = -np.arange(10, dtype=float)
        write_data(self.tmp / "S01_data.csv", fp1, fp2)
        t, eeg = read_signal(self.tmp / "S01_data.csv")
        self.assertEqual(eeg.shape, (2, 10))
        np.testing.assert_allclose(eeg[0], fp1)
        np.testing.assert_allclose(eeg[1], fp2)
        np.testing.assert_allclose(t[:3], [0, 1 / NATIVE_RATE_HZ, 2 / NATIVE_RATE_HZ], atol=1e-6)

    def test_columns_past_the_third_are_ignored(self):
        """BLINK's own reader takes usecols=(0,1,2); OpenBCI files carry more."""
        fp1, fp2 = np.ones(8), 2 * np.ones(8)
        write_data(self.tmp / "S02_data.csv", fp1, fp2, extra_column=True)
        _, eeg = read_signal(self.tmp / "S02_data.csv")
        self.assertEqual(eeg.shape, (2, 8))
        np.testing.assert_allclose(eeg[1], fp2)

    def test_a_wrong_sample_rate_is_refused_not_resampled(self):
        """Silently resampling a 500 Hz file would hide that it is not EEG-IO."""
        n = 600
        write_data(self.tmp / "S03_data.csv", np.zeros(n), np.zeros(n), fs=500.0)
        write_labels(self.tmp / "S03_labels.csv", blinks=[(1.0, 0)])
        with self.assertRaisesRegex(ValueError, "implies 500"):
            load_subject(self.tmp, "S03")


class ResampleTests(unittest.TestCase):

    def test_250_to_256_is_an_exact_rational_ratio(self):
        """256/250 = 128/125, so the resample carries no accumulating phase error."""
        n = 2500                                    # 10 s at 250 Hz
        x = blinky_signal(n, [2.0, 5.0, 8.0])[None, :]
        y = resample_to_project_rate(x, NATIVE_RATE_HZ, PROJECT_RATE_HZ)
        self.assertEqual(y.shape[1], n * 128 // 125)
        self.assertAlmostEqual(y.shape[1] / PROJECT_RATE_HZ, n / NATIVE_RATE_HZ, places=6)

    def test_a_blink_stays_where_the_annotation_says_it_is(self):
        """Annotations are in seconds; after resampling, time * new_fs must still
        land on the blink, or every blink-centred window is off."""
        n, t_blink = 2500, 5.0
        x = blinky_signal(n, [t_blink])[None, :]
        y = resample_to_project_rate(x, NATIVE_RATE_HZ, PROJECT_RATE_HZ)
        found = int(np.argmin(y[0]))
        self.assertAlmostEqual(found / PROJECT_RATE_HZ, t_blink, delta=0.02)

    def test_a_10hz_rhythm_survives_the_resample(self):
        n = 2500
        t = np.arange(n) / NATIVE_RATE_HZ
        x = np.sin(2 * np.pi * 10.0 * t)[None, :]
        y = resample_to_project_rate(x, NATIVE_RATE_HZ, PROJECT_RATE_HZ)[0]
        # Dominant rfft bin, away from the polyphase edge transients.
        core = y[128:-128]
        freqs = np.fft.rfftfreq(len(core), 1 / PROJECT_RATE_HZ)
        self.assertAlmostEqual(freqs[np.argmax(np.abs(np.fft.rfft(core)))], 10.0, delta=0.5)


class LoadingTests(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="eeg_io_load_"))
        self.n = 7500                                # 30 s at 250 Hz
        self.blinks = [5.0, 10.0, 15.0, 20.0]
        for sub in ("S01", "S02"):
            write_data(self.tmp / f"{sub}_data.csv",
                       blinky_signal(self.n, self.blinks, seed=1),
                       blinky_signal(self.n, self.blinks, seed=2))
            write_labels(self.tmp / f"{sub}_labels.csv",
                         blinks=[(t, 1) for t in self.blinks])

    def test_lists_only_subjects_that_have_both_files(self):
        (self.tmp / "S09_data.csv").write_text("time;channel1;channel2\n0;0;0\n")   # no labels
        self.assertEqual(list_subjects(self.tmp), ["S01", "S02"])

    def test_a_missing_directory_names_the_request_gated_source(self):
        with self.assertRaisesRegex(FileNotFoundError, "gnan.ece.gatech.edu"):
            list_subjects(self.tmp / "nope")

    def test_loads_at_the_project_rate_by_default(self):
        rec = load_subject(self.tmp, "S01")
        self.assertEqual(rec.fs, PROJECT_RATE_HZ)
        self.assertEqual(rec.resampled_from, NATIVE_RATE_HZ)
        self.assertAlmostEqual(rec.duration_s, self.n / NATIVE_RATE_HZ, places=3)

    def test_native_rate_is_available_unresampled(self):
        rec = load_subject(self.tmp, "S01", resample=False)
        self.assertEqual(rec.fs, NATIVE_RATE_HZ)
        self.assertIsNone(rec.resampled_from)
        self.assertEqual(rec.eeg.shape, (2, self.n))

    def test_channels_are_addressable_by_name(self):
        rec = load_subject(self.tmp, "S01")
        self.assertEqual(CHANNEL_NAMES, ("Fp1", "Fp2"))
        np.testing.assert_array_equal(rec.channel("Fp1"), rec.eeg[0])
        np.testing.assert_array_equal(rec.channel("Fp2"), rec.eeg[1])

    def test_corrupt_mask_covers_the_annotated_interval(self):
        write_labels(self.tmp / "S02_labels.csv", blinks=[(5.0, 1)], corrupt=[(2.0, 4.0)])
        rec = load_subject(self.tmp, "S02")
        mask = rec.corrupt_mask()
        self.assertTrue(mask[int(3.0 * rec.fs)])
        self.assertFalse(mask[int(1.0 * rec.fs)])
        self.assertFalse(mask[int(5.0 * rec.fs)])

    def test_load_dataset_returns_every_subject(self):
        self.assertEqual([r.subject for r in load_dataset(self.tmp)], ["S01", "S02"])


class EpochCuttingTests(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="eeg_io_cut_"))
        self.n = 15000                               # 60 s at 250 Hz
        self.blinks = [5.0, 15.0, 25.0, 35.0, 45.0]
        write_data(self.tmp / "S01_data.csv",
                   blinky_signal(self.n, self.blinks, seed=1),
                   blinky_signal(self.n, self.blinks, seed=2))
        write_labels(self.tmp / "S01_labels.csv", blinks=[(t, 1) for t in self.blinks])
        self.recs = load_dataset(self.tmp)

    def test_blink_windows_are_centred_on_the_annotation(self):
        ep = cut_epochs(self.recs, channels=["Fp1"]).blink()
        self.assertEqual(len(ep), len(self.blinks))
        for row in ep.x:
            # The annotation marks the minimum, so it should sit mid-window.
            self.assertAlmostEqual(int(np.argmin(row)), WINDOW // 2, delta=WINDOW // 16)

    def test_both_populations_are_produced_and_tagged(self):
        ep = cut_epochs(self.recs, channels=["Fp1"])
        self.assertEqual(int(ep.is_blink.sum()), len(self.blinks))
        self.assertGreater(int((~ep.is_blink).sum()), 0)
        self.assertEqual(len(ep), len(ep.blink()) + len(ep.clean()))

    def test_blink_free_windows_really_contain_no_blink(self):
        """The whole point of the pair: if 'clean' windows held blinks too, a
        suppression number could not be read against them."""
        ep = cut_epochs(self.recs, channels=["Fp1"]).clean()
        self.assertGreater(len(ep), 0)
        np.testing.assert_array_equal(ep.n_blinks, 0)
        # A blink deflection is ~120 uV against a ~5 uV background.
        self.assertLess(np.abs(ep.x).max(), 60.0)

    def test_blink_windows_are_the_ones_with_the_large_deflection(self):
        ep = cut_epochs(self.recs, channels=["Fp1"])
        self.assertGreater(np.abs(ep.blink().x).max(), 3 * np.abs(ep.clean().x).max())

    def test_each_channel_becomes_its_own_windows(self):
        """The model takes (B, 1, T), so Fp1 and Fp2 cannot share a window."""
        one = cut_epochs(self.recs, channels=["Fp1"])
        both = cut_epochs(self.recs, channels=["Fp1", "Fp2"])
        self.assertEqual(len(both), 2 * len(one))
        self.assertEqual(set(both.channel), {"Fp1", "Fp2"})
        self.assertEqual(both.x.shape[1], WINDOW)

    def test_corrupt_intervals_are_excluded(self):
        write_labels(self.tmp / "S01_labels.csv", blinks=[(t, 1) for t in self.blinks],
                     corrupt=[(0.0, 30.0)])
        recs = load_dataset(self.tmp)
        ep = cut_epochs(recs, channels=["Fp1"])
        fs = recs[0].fs
        self.assertTrue((ep.start / fs >= 30.0 - 1e-9).all())
        # The blinks at 5/15/25 s fall inside the corrupt span, so only 35/45 survive.
        self.assertEqual(len(ep.blink()), 2)

    def test_the_guard_band_keeps_blink_free_windows_clear(self):
        """A window abutting a blink can hold its tail; guard_s pushes it away."""
        fs = self.recs[0].fs
        tight = cut_epochs(self.recs, channels=["Fp1"], guard_s=0.0).clean()
        loose = cut_epochs(self.recs, channels=["Fp1"], guard_s=3.0).clean()
        self.assertGreater(len(tight), len(loose))
        for start in loose.start:
            gap = np.abs(np.asarray(self.blinks) - (start + WINDOW / 2) / fs)
            self.assertGreater(gap.min(), 1.0)

    def test_soft_blinks_can_be_dropped(self):
        write_labels(self.tmp / "S01_labels.csv",
                     blinks=[(5.0, 0), (15.0, 2), (25.0, 1)])
        recs = load_dataset(self.tmp)
        self.assertEqual(len(cut_epochs(recs, channels=["Fp1"]).blink()), 3)
        self.assertEqual(len(cut_epochs(recs, channels=["Fp1"], include_soft=False).blink()), 2)

    def test_blink_free_windows_can_be_capped(self):
        ep = cut_epochs(self.recs, channels=["Fp1", "Fp2"], max_clean_per_channel=3)
        self.assertLessEqual(len(ep.clean()), 6)

    def test_windows_running_off_the_end_are_dropped(self):
        """A blink in the first second has no room for a centred 512-sample window."""
        write_data(self.tmp / "S02_data.csv", blinky_signal(self.n, [0.1], seed=3),
                   blinky_signal(self.n, [0.1], seed=4))
        write_labels(self.tmp / "S02_labels.csv", blinks=[(0.1, 1)])
        ep = cut_epochs([load_subject(self.tmp, "S02")], channels=["Fp1"])
        self.assertEqual(len(ep.blink()), 0)

    def test_mixed_sample_rates_are_refused(self):
        native = load_subject(self.tmp, "S01", resample=False)
        resampled = load_subject(self.tmp, "S01", resample=True)
        with self.assertRaisesRegex(ValueError, "disagree on sample rate"):
            cut_epochs([native, resampled])


class ExportTests(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="eeg_io_export_"))
        n = 15000
        write_data(self.tmp / "S01_data.csv", blinky_signal(n, [5.0, 15.0], seed=1),
                   blinky_signal(n, [5.0, 15.0], seed=2))
        write_labels(self.tmp / "S01_labels.csv", blinks=[(5.0, 1), (15.0, 1)])
        self.epochs = cut_epochs(load_dataset(self.tmp), max_clean_per_channel=4)

    def test_npz_round_trips_every_field(self):
        path = save_epochs(self.epochs, self.tmp / "out.npz")
        with np.load(path, allow_pickle=False) as d:
            np.testing.assert_allclose(d["x"], self.epochs.x)
            np.testing.assert_array_equal(d["is_blink"], self.epochs.is_blink)
            np.testing.assert_array_equal(d["n_blinks"], self.epochs.n_blinks)
            np.testing.assert_array_equal(d["subject"], self.epochs.subject)
            np.testing.assert_array_equal(d["channel"], self.epochs.channel)
            self.assertEqual(float(d["fs"]), PROJECT_RATE_HZ)

    def test_the_export_carries_no_clean_array(self):
        """EEG-IO has no ground truth; anything finding a 'clean' key here would
        treat it as one, and every metric computed off it would be fiction."""
        path = save_epochs(self.epochs, self.tmp / "out.npz")
        with np.load(path, allow_pickle=False) as d:
            self.assertNotIn("clean", d.files)
            self.assertIn("x", d.files)

    def test_windows_are_raw_microvolts_not_normalised(self):
        """raspi_infer.py normalises each window by its own std; storing raw uV
        keeps this consistent with deploy/testdata/*.npz."""
        self.assertGreater(np.abs(self.epochs.blink().x).max(), 50.0)


if __name__ == "__main__":
    unittest.main()
