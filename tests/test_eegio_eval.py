"""
Evaluating on real recordings: the filter, the detector, the four metrics.

None of this needs EEG-IO. The filters are probed the way the project probes
a model, the detector is run on signals with blinks planted at known
samples, and each metric is given a model whose answer is known in advance --
identity, silence, a pure delay, an anticipator. That is the only way to
check a metric: on a model you already know the right answer for.

Run: python -m unittest discover -s tests -p 'test_eegio_eval.py'
"""

import json
import os
import socket
import sys
import tempfile
import threading
import unittest
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eegio_eval as E                                            # noqa: E402
from data.EEGIO import write_synthetic_recording                  # noqa: E402

FS = 256.0
N = 4096


def _blinky(n=N, fs=FS, at=(500, 1500, 2600, 3500), width=0.15, amp=40.0, seed=0):
    """Background noise with Hanning bumps at known samples."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=n).astype(np.float32)
    w = int(width * fs)
    for c in at:
        a, b = max(c - w // 2, 0), min(c + w // 2, n)
        x[a:b] += (np.hanning(b - a) * amp).astype(np.float32)
    return x, list(at)


# ============================================================
# Causal preprocessing
# ============================================================
class TestPreprocessing(unittest.TestCase):

    def test_the_default_filter_is_causal(self):
        r = E.preprocess_is_causal(FS)
        self.assertEqual(r["past_diff"], 0.0)
        self.assertGreater(r["future_diff"], 0.0, "the probe perturbed nothing")

    def test_the_zero_phase_control_is_not(self):
        # sosfiltfilt runs the filter backwards as well; it is the control
        # precisely because it cannot be deployed.
        r = E.preprocess_is_causal(FS, zero_phase=True)
        self.assertGreater(r["past_diff"], 0.0)

    def test_the_band_comes_from_the_training_data(self):
        self.assertEqual(E.TRAIN_BAND_HZ, (1.0, 80.0))
        self.assertEqual(E.POWERLINE_HZ, 60.0)

    def test_three_stages_by_default_and_fewer_when_asked(self):
        self.assertEqual(len(E.design_filters(FS)), 3)
        self.assertEqual(len(E.design_filters(FS, notch_hz=None)), 2)
        self.assertEqual(len(E.design_filters(FS, band=(0.0, 80.0), notch_hz=None)), 1)

    def test_a_lowpass_at_or_above_nyquist_is_left_out_rather_than_clipped(self):
        self.assertEqual(len(E.design_filters(100.0, band=(1.0, 80.0), notch_hz=None)), 1)

    def test_it_removes_dc_and_keeps_the_eeg_band(self):
        t = np.arange(N) / FS
        drift = np.ones(N, dtype=np.float32) * 10.0
        tone = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)
        y = E.preprocess(drift + tone, FS)
        settled = y[N // 2:]
        self.assertLess(abs(float(settled.mean())), 0.1)             # DC gone
        self.assertGreater(float(settled.std()), 0.5)                # 10 Hz kept

    def test_the_powerline_tone_is_attenuated(self):
        t = np.arange(N) / FS
        hum = np.sin(2 * np.pi * E.POWERLINE_HZ * t).astype(np.float32)
        with_notch = E.preprocess(hum, FS)[N // 2:]
        without = E.preprocess(hum, FS, notch_hz=None)[N // 2:]
        self.assertLess(float(np.abs(with_notch).max()), float(np.abs(without).max()))


# ============================================================
# Blink intervals
# ============================================================
class TestDetection(unittest.TestCase):

    def test_it_finds_the_planted_blinks(self):
        x, at = _blinky()
        found = E.detect_blinks(x, FS)
        self.assertGreaterEqual(len(found), len(at) - 1)
        for c in at:
            self.assertTrue(any(a <= c < b for a, b in found), f"missed the blink at {c}")

    def test_a_flat_signal_yields_nothing(self):
        self.assertEqual(E.detect_blinks(np.zeros(N, dtype=np.float32), FS), [])

    def test_only_the_annotated_detections_survive(self):
        x, at = _blinky()
        found = E.detect_blinks(x, FS)
        kept = E.keep_annotated(found, at[:2])
        self.assertEqual(len(kept), 2)
        self.assertTrue(all(any(a <= c < b for c in at[:2]) for a, b in kept))

    def test_no_annotations_keeps_nothing(self):
        x, _ = _blinky()
        self.assertEqual(E.keep_annotated(E.detect_blinks(x, FS), []), [])

    def test_an_excluded_stretch_produces_no_detections(self):
        x, at = _blinky()
        bad = np.zeros(N, dtype=bool)
        bad[400:700] = True                       # covers the first blink
        found = E.detect_blinks(x, FS, exclude=bad)
        self.assertFalse(any(a <= at[0] < b for a, b in found))

    def test_background_excludes_the_blinks_a_guard_band_and_the_corrupt(self):
        corrupt = np.zeros(N, dtype=bool)
        corrupt[3000:3200] = True
        bg = E.background_mask(N, [(500, 600)], FS, guard_s=0.25, exclude=corrupt)
        self.assertFalse(bg[550])                                  # the blink
        self.assertFalse(bg[500 - int(0.1 * FS)])                  # inside the guard
        self.assertFalse(bg[3100])                                 # corrupt
        self.assertTrue(bg[2000])                                  # ordinary background


# ============================================================
# The metrics, on models whose answer is known
# ============================================================
class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.x, self.at = _blinky()
        self.kept = E.keep_annotated(E.detect_blinks(self.x, FS), self.at)
        self.bg = E.background_mask(N, self.kept, FS)

    def test_doing_nothing_suppresses_nothing(self):
        self.assertAlmostEqual(E.suppression_db(self.x, self.x, self.kept), 0.0, places=6)

    def test_removing_the_blinks_shows_as_suppression(self):
        # Attenuated rather than zeroed: zero power in the intervals is an
        # infinite ratio, which _db reports as not-a-number rather than
        # inventing a figure, and no real model does it anyway.
        y = self.x.copy()
        for a, b in self.kept:
            y[a:b] *= 0.01
        self.assertGreater(E.suppression_db(self.x, y, self.kept), 10.0)

    def test_an_untouched_background_is_perfectly_faithful(self):
        y = self.x.copy()
        for a, b in self.kept:
            y[a:b] = 0.0
        self.assertTrue(np.isnan(E.fidelity_db(self.x, y, self.bg)))   # no distortion at all

    def test_mangling_the_background_destroys_fidelity(self):
        rng = np.random.default_rng(1)
        clean = E.fidelity_db(self.x, self.x + 0.01 * rng.normal(size=N), self.bg)
        dirty = E.fidelity_db(self.x, self.x + 1.00 * rng.normal(size=N), self.bg)
        self.assertGreater(clean, dirty)

    def test_suppression_alone_can_be_gamed_which_is_why_fidelity_is_beside_it(self):
        # A model that attenuates everything scores well on suppression and
        # badly on fidelity. If that ever stops being true the pair is useless.
        y = (self.x * 0.1).astype(np.float32)
        self.assertGreater(E.suppression_db(self.x, y, self.kept), 10.0)
        self.assertLess(E.fidelity_db(self.x, y, self.bg), 10.0)

    def test_residual_detection_is_total_for_a_model_that_does_nothing(self):
        r = E.residual_detectable(self.x, FS, self.kept)
        self.assertEqual(r["rate"], 1.0)

    def test_residual_detection_is_zero_once_the_blinks_are_gone(self):
        y = self.x.copy()
        for a, b in self.kept:
            y[a:b] = self.x[self.bg][:b - a] * 0.1 if self.bg.sum() >= (b - a) else 0.0
        self.assertEqual(E.residual_detectable(y, FS, self.kept)["rate"], 0.0)

    def test_residual_on_no_intervals_is_not_a_number_rather_than_zero(self):
        self.assertTrue(np.isnan(E.residual_detectable(self.x, FS, [])["rate"]))


class TestStandardMetrics(unittest.TestCase):
    def test_adjusted_counts_use_the_reference_output_threshold_for_both_signals(self):
        fs = 100
        x = np.ones(1000)
        centers = np.array([300, 600])
        x[centers] = [12, 100]
        valid = np.ones(len(x), dtype=bool)
        baseline = E.standard_metrics(
            x, x * 0.5, fs, centers, centers, valid, .25, adjusted_blink_count=True,
        )
        reference = baseline["metric_support"]["adjusted_blink_count"]
        self.assertAlmostEqual(reference["threshold_uv"], 3.33)
        result = E.standard_metrics(
            x, x * 0.02, fs, centers, centers, valid, .25,
            adjusted_blink_count=True, reference_threshold_uv=reference["threshold_uv"],
        )
        self.assertEqual(result["blink_count_reduction_pct"], 0)
        self.assertEqual(result[E.ADJUSTED_BLINK_COUNT], 100)
        self.assertEqual(result["metric_support"]["adjusted_blink_count"],
                         {**reference, "output_blink_count": 0})
        self.assertEqual(reference["input_blink_count"], 2)
        self.assertEqual(E.summarise([result])[E.ADJUSTED_BLINK_COUNT]["median"], 100)

    def test_known_blinks_and_background_gain(self):
        fs = 100
        background = np.tile([-1., 1.], 450)
        centers = np.array([300, 620])
        x = background.copy()
        x[centers] = 100
        y = background * 0.5
        valid = np.ones(len(x), dtype=bool)
        result = E.standard_metrics(x, y, fs, centers, centers, valid, .25)

        self.assertAlmostEqual(result["blink_amplitude_ratio"], 1, places=3)
        self.assertEqual(result["background_rmse_uv"], 0.5)
        self.assertEqual(result["blink_count_reduction_pct"], 100)
        support = result["metric_support"]
        self.assertEqual((support["bar_epochs"], support["input_blink_count"],
                          support["output_blink_count"]), (2, 2, 0))
        self.assertGreater(support["input_blink_amplitude_ratio"], result["blink_amplitude_ratio"])

    def test_background_excludes_unscored_labels_and_invalid_samples(self):
        fs = 100
        x = np.ones(1000)
        y = x.copy()
        y[:300] = 100
        y[450:551] = 100
        valid = np.arange(len(x)) >= 300
        annotated = np.array([], dtype=int)
        result = E.standard_metrics(x, y, fs, annotated, np.array([500]), valid, .25)

        self.assertEqual(result["background_rmse_uv"], 0)
        self.assertEqual(result["metric_support"]["background_samples"], 599)
        self.assertTrue(np.isnan(result["blink_amplitude_ratio"]))
        self.assertTrue(np.isnan(result["blink_count_reduction_pct"]))

    def test_pairing_uses_distance_to_one_for_blink_amplitude_ratio(self):
        def comparison(run, values):
            return {"run": run, "rows": [{"subject": str(i), "blink_amplitude_ratio": value}
                                         for i, value in enumerate(values)]}
        baseline = comparison("baseline", [2., .4, 1.8])
        candidate = comparison("candidate", [.5, .8, 1.2])
        result = E.paired_wilcoxon(baseline, candidate, "blink_amplitude_ratio")
        self.assertEqual(result["tested_quantity"], "distance_to_one")
        self.assertEqual(result["b_better_on"], 3)
        self.assertAlmostEqual(result["median_diff"], -.5)


class TestOnsetLeakage(unittest.TestCase):
    """
    The measurement the tab exists for, given models whose lead is known
    exactly: the correction *is* the planted bump, shifted by a known number
    of samples. Shift zero is a model that tracks the artifact as it arrives;
    a positive shift is one that is already correcting before it does.
    """

    def setUp(self):
        self.x, self.at = _blinky()
        self.kept = E.keep_annotated(E.detect_blinks(self.x, FS), self.at)
        self.bg = E.background_mask(N, self.kept, FS)
        bump = np.zeros(N, dtype=np.float32)
        w = int(0.15 * FS)
        for c in self.at:
            a, b = max(c - w // 2, 0), min(c + w // 2, N)
            bump[a:b] += (np.hanning(b - a) * 40.0).astype(np.float32)
        self.bump = bump

    def _output(self, shift):
        corr = np.roll(self.bump, -shift) if shift else self.bump
        return (self.x - corr).astype(np.float32)

    def test_a_model_that_tracks_the_artifact_leads_by_about_nothing(self):
        r = E.onset_leakage(self.x, self._output(0), FS, self.kept, self.bg)
        self.assertGreaterEqual(r["n_measurable"], 1)
        self.assertLessEqual(abs(r["median"]), 2.0)

    def test_a_shifted_correction_measures_the_shift(self):
        for shift in (8, 16, 32):
            r = E.onset_leakage(self.x, self._output(shift), FS, self.kept, self.bg)
            with self.subTest(shift=shift):
                self.assertAlmostEqual(r["median"], shift, delta=2.0)

    def test_a_model_that_reacts_late_measures_negative(self):
        r = E.onset_leakage(self.x, self._output(-16), FS, self.kept, self.bg)
        self.assertLess(r["median"], 0.0)

    def test_it_reports_milliseconds_consistent_with_samples(self):
        r = E.onset_leakage(self.x, self._output(16), FS, self.kept, self.bg)
        self.assertAlmostEqual(r["median_ms"], r["median"] / FS * 1e3, places=6)

    def test_no_intervals_is_not_a_number(self):
        r = E.onset_leakage(self.x, self.x, FS, [], self.bg)
        self.assertEqual(r["n"], 0)
        self.assertTrue(np.isnan(r["median"]))


class TestRiskCheck(unittest.TestCase):

    def test_a_stronger_blink_estimates_a_worse_snr(self):
        out = []
        for amp in (10.0, 80.0):
            x, at = _blinky(amp=amp)
            kept = E.keep_annotated(E.detect_blinks(x, FS), at)
            bg = E.background_mask(N, kept, FS)
            out.append(E.blink_snr_estimate(x, kept, bg)["median_db"])
        self.assertGreater(out[0], out[1])

    def test_it_reports_the_training_range_it_is_checked_against(self):
        x, at = _blinky()
        kept = E.keep_annotated(E.detect_blinks(x, FS), at)
        r = E.blink_snr_estimate(x, kept, E.background_mask(N, kept, FS))
        self.assertEqual(r["train_range_db"], list(E.TRAIN_SNR_DB))
        self.assertTrue(0.0 <= r["inside_train_range"] <= 1.0)


# ============================================================
# Running a model along a recording
# ============================================================
class _Identity:
    def __call__(self, t):
        return t

    def eval(self):
        return self


class TestStreamApply(unittest.TestCase):

    def test_an_identity_model_returns_the_signal(self):
        x = _blinky()[0]
        np.testing.assert_allclose(E.stream_apply(_Identity(), x, win=512, hop=64), x,
                                   rtol=1e-4, atol=1e-4)

    def test_it_works_when_the_signal_is_shorter_than_one_window(self):
        x = _blinky(n=200)[0]
        y = E.stream_apply(_Identity(), x, win=512, hop=64)
        self.assertEqual(len(y), 200)
        np.testing.assert_allclose(y, x, rtol=1e-4, atol=1e-4)

    def test_every_hop_leaves_the_length_alone(self):
        x = _blinky(n=1500)[0]
        for hop in (1, 32, 64, 512):
            self.assertEqual(len(E.stream_apply(_Identity(), x, 512, hop)), 1500, f"hop={hop}")

    def test_offline_apply_uses_full_windows_and_aligns_delayed_interior_outputs(self):
        class Probe:
            run_config = {"output_delay": 1}

            def __call__(self, x):
                self.inputs = x.clone()
                return x.roll(-2, dims=-1)

        x = np.random.default_rng(0).normal(size=29).astype(np.float32)
        probe = Probe()
        y = E.offline_apply(probe, x, E.OfflineWindow(12, 3, 4))
        windows = np.stack([x[start:start + 12] for start in (0, 5, 10, 15, 17)])
        np.testing.assert_allclose(probe.inputs[:, 0], windows / windows.std(axis=1, keepdims=True))
        np.testing.assert_allclose(y[3:-4], x[6:-1], atol=1e-6)
        self.assertTrue(np.isnan(y[np.r_[:3, -4:0]]).all())


# ============================================================
# Pairing and the summary
# ============================================================
class TestPairing(unittest.TestCase):

    def result(self, run, values):
        rows = [{"subject": s, "suppression_db": v, "fidelity_db": 0.0,
                 "residual": {"rate": 0.0}, "onset_leakage": {"median_ms": 0.0}}
                for s, v in values.items()]
        return {"run": run, "rows": rows, "summary": E.summarise(rows)}

    def test_it_pairs_on_the_subjects_both_models_have(self):
        a = self.result("a", {"S1": 1.0, "S2": 2.0, "S3": 3.0})
        b = self.result("b", {"S2": 4.0, "S3": 5.0, "S9": 9.0})
        t = E.paired_wilcoxon(a, b, "suppression_db")
        self.assertEqual(t["subjects"], ["S2", "S3"])
        self.assertEqual(t["n_pairs"], 2)

    def test_a_consistent_improvement_is_counted_on_every_subject(self):
        a = self.result("a", {f"S{i}": float(i) for i in range(6)})
        b = self.result("b", {f"S{i}": float(i) + 1.0 for i in range(6)})
        t = E.paired_wilcoxon(a, b, "suppression_db")
        self.assertEqual(t["b_better_on"], 6)
        self.assertAlmostEqual(t["median_diff"], 1.0)
        self.assertLess(t["p_value"], 0.05)

    def test_too_few_pairs_says_so_rather_than_inventing_a_p_value(self):
        a = self.result("a", {"S1": 1.0})
        b = self.result("b", {"S1": 2.0})
        t = E.paired_wilcoxon(a, b, "suppression_db")
        self.assertTrue(np.isnan(t["p_value"]))
        self.assertIn("too few", t["note"])

    def test_summarise_counts_only_the_finite_values(self):
        rows = [{"subject": "a", "suppression_db": 1.0, "fidelity_db": float("nan"),
                 "residual": {"rate": 0.5}, "onset_leakage": {"median_ms": 2.0}}]
        s = E.summarise(rows)
        self.assertEqual(s["suppression_db"]["n"], 1)
        self.assertEqual(s["fidelity_db"]["n"], 0)


# ============================================================
# Running at the native rate instead of the training rate
# ============================================================
class TestTargetRate(unittest.TestCase):
    """
    EEG-IO is 250 Hz and the models were trained at 256, so one of the two
    has to give. Both settings are offered and both have to work.
    """

    def _recording(self, d, dur=12.0):
        return write_synthetic_recording(d, "S00", duration_s=dur, n_blinks=4,
                                         n_soft=0, seed=5, amplitude=60.0, drift=0.0)

    def test_the_default_resamples_to_the_training_rate(self):
        self.assertEqual(E.EvalConfig().target_fs, E.PROJECT_FS)
        self.assertEqual(E.PROJECT_FS, 256.0)
        self.assertEqual(E.EEGIO_FS, 250.0)

    def test_the_native_rate_skips_the_conversion(self):
        with tempfile.TemporaryDirectory() as d:
            rec = self._recording(d)
            out = E.evaluate_subjects(d, "identity", ["S00"],
                                      E.EvalConfig(target_fs=E.EEGIO_FS, hop=256),
                                      apply_fn=lambda x: x)
        row = out["rows"][0]
        self.assertEqual(row["fs"], E.EEGIO_FS)
        self.assertEqual(row["n_samples"], rec.n_samples)      # untouched

    def test_none_means_the_native_rate_too(self):
        with tempfile.TemporaryDirectory() as d:
            rec = self._recording(d)
            out = E.evaluate_subjects(d, "identity", ["S00"],
                                      E.EvalConfig(target_fs=None, hop=256),
                                      apply_fn=lambda x: x)
        self.assertEqual(out["rows"][0]["n_samples"], rec.n_samples)

    def test_resampling_changes_the_length_by_the_rate_ratio(self):
        with tempfile.TemporaryDirectory() as d:
            rec = self._recording(d)
            out = E.evaluate_subjects(d, "identity", ["S00"],
                                      E.EvalConfig(target_fs=E.PROJECT_FS, hop=256),
                                      apply_fn=lambda x: x)
        self.assertEqual(out["rows"][0]["fs"], E.PROJECT_FS)
        self.assertEqual(out["rows"][0]["n_samples"],
                         round(rec.n_samples * E.PROJECT_FS / E.EEGIO_FS))

    def test_both_rates_find_the_same_blinks(self):
        # The conversion must not move the annotations off the artifacts.
        with tempfile.TemporaryDirectory() as d:
            self._recording(d, dur=20.0)
            kept = {}
            for fs in (E.EEGIO_FS, E.PROJECT_FS):
                out = E.evaluate_subjects(d, "identity", ["S00"],
                                          E.EvalConfig(target_fs=fs, hop=256),
                                          apply_fn=lambda x: x)
                kept[fs] = out["rows"][0]["n_kept"]
        self.assertGreater(kept[E.EEGIO_FS], 0)
        self.assertEqual(kept[E.EEGIO_FS], kept[E.PROJECT_FS])

    def test_the_preprocessing_is_causal_at_either_rate(self):
        for fs in (E.EEGIO_FS, E.PROJECT_FS):
            with self.subTest(fs=fs):
                self.assertEqual(E.preprocess_is_causal(fs)["past_diff"], 0.0)

    def test_the_tab_offers_both_and_reports_which_was_used(self):
        from gui.eegio_tab import meta

        settings = meta()
        rates = {r["value"] for r in settings["rates"]}
        self.assertEqual(rates, {E.EEGIO_FS, E.PROJECT_FS})
        self.assertEqual(settings["defaults"]["n_subjects"], 20)
        self.assertEqual(settings["defaults"]["hop"], 4)

    def test_every_run_carries_the_rate_it_trained_at(self):
        # A directory name no longer implies a rate: runs/tcn_mla_both_s67
        # held a 256 Hz run and now holds a 250 Hz one, so the picker has to
        # read it from the run and say whether the run said it or this is an
        # inference from an older run recording nothing.
        from gui.eegio_tab import available_runs

        rows = available_runs()
        self.assertTrue(rows)
        for r in rows:
            with self.subTest(run=r["name"]):
                self.assertTrue(np.isfinite(r["train_fs"]) and r["train_fs"] > 0)
                self.assertIsInstance(r["train_fs_recorded"], bool)
                if not r["train_fs_recorded"]:
                    self.assertEqual(r["train_fs"], E.PROJECT_FS)

    def test_picker_and_eegio_use_the_same_checkpoint_first_training_rates(self):
        from gui import config_io, eegio_tab

        cases = {
            "tcn_256hz": (250, 256, (250, True)),
            "history_only": (None, 250, (250, True)),
            "legacy": (None, None, (256, False)),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, (_, history_rate, _) in cases.items():
                run = root / name
                run.mkdir()
                (run / "best.pt").touch()
                (run / "history.json").write_text(json.dumps({
                    "config": {"target_fs": history_rate}, "val_loss": [1.0],
                    "rrmse_t": [0.7], "rrmse_s": [0.6], "acc": [0.8],
                }))

            def checkpoint_info(path):
                if not path.is_file():
                    return {}
                rate = cases[path.parent.name][0]
                return {"epoch": 1, "run_config": {} if rate is None else {"target_fs": rate}}

            with patch.object(config_io, "RUNS_DIR", root), patch.object(
                config_io, "_checkpoint_info", side_effect=checkpoint_info
            ):
                picker_rows = config_io.list_runs()
                eegio_rows = eegio_tab.available_runs()

        expected = {name: case[2] for name, case in cases.items()}
        for rows in (picker_rows, eegio_rows):
            self.assertEqual({r["name"]: (r["train_fs"], r["train_fs_recorded"])
                              for r in rows}, expected)


# ============================================================
# Keeping the comparisons
# ============================================================
class TestSavedResults(unittest.TestCase):
    """
    A measurement that vanishes when the page reloads cannot be compared
    with next week's, so each comparison is written to eegio_tests/ and the
    tab can reopen it.
    """

    def setUp(self):
        from gui import eegio_tab

        self.mod = eegio_tab
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._old = eegio_tab.RESULTS_DIR
        eegio_tab.RESULTS_DIR = Path(self._tmp.name)
        self.addCleanup(lambda: setattr(eegio_tab, "RESULTS_DIR", self._old))

    def _result(self, run="tcn_se_both", n=2, hop=32):
        return {"saved": {"at": "2026-10-30T12:00:00", "hop": hop, "n_subjects": n,
                          "models": [{"run": run, "train_fs": 256.0}]},
                "results": [{"run": run, "summary": {"suppression_db": {"median": 1.0}}}],
                "subjects": [f"S{i:02d}" for i in range(n)],
                "baseline": run, "hop": hop, "target_fs": 256.0, "rate_mismatch": []}

    def test_a_saved_result_carries_its_own_file_name(self):
        # Without it the page cannot select the thing it just produced in
        # its own history list.
        out = self._result()
        name = self.mod.save_result(out)
        self.assertEqual(out["name"], name)
        self.assertEqual(json.loads((self.mod.RESULTS_DIR / name).read_text())["name"], name)

    def test_results_are_listed_newest_first(self):
        import time

        names = []
        for i in range(3):
            names.append(self.mod.save_result(self._result(n=i + 1)))
            time.sleep(1.01)                 # the name is second-resolution
        listed = [r["name"] for r in self.mod.list_results()]
        self.assertEqual(listed, sorted(names, reverse=True))

    def test_the_listing_says_enough_to_choose_between_two(self):
        self.mod.save_result(self._result(run="tcn_se_la16_both", n=5, hop=64))
        row = self.mod.list_results()[0]
        self.assertEqual(row["runs"], ["tcn_se_la16_both"])
        self.assertEqual(row["n_subjects"], 5)
        self.assertEqual(row["hop"], 64)
        self.assertEqual(row["target_fs"], 256.0)
        self.assertIn("suppression_db", row["summary"]["tcn_se_la16_both"])

    def test_a_result_reads_back_whole(self):
        name = self.mod.save_result(self._result(n=4))
        self.assertEqual(self.mod.read_result(name)["subjects"],
                         ["S00", "S01", "S02", "S03"])

    def test_a_name_outside_the_directory_is_refused(self):
        # The GUI has no authentication; this is the only place a request
        # gets to name a file.
        for bad in ("../../etc/passwd", "/etc/passwd", ""):
            with self.subTest(name=bad):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.mod.read_result(bad)

    def test_a_missing_result_is_a_not_found_rather_than_a_crash(self):
        with self.assertRaises(FileNotFoundError):
            self.mod.read_result("eegio_20000101_000000.json")

    def test_an_unparseable_file_does_not_break_the_whole_listing(self):
        self.mod.save_result(self._result())
        (self.mod.RESULTS_DIR / "eegio_99999999_999999.json").write_text("{ not json")
        rows = self.mod.list_results()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["error"], "could not be parsed")

    def test_only_the_newest_are_kept(self):
        import time

        self.mod.MAX_RESULTS, keep = 3, self.mod.MAX_RESULTS
        self.addCleanup(lambda: setattr(self.mod, "MAX_RESULTS", keep))
        for i in range(5):
            self.mod.save_result(self._result(n=i + 1))
            time.sleep(1.01)
        self.assertEqual(len(self.mod.list_results()), 3)

    def test_meta_carries_the_history_so_the_tab_can_draw_it(self):
        self.mod.save_result(self._result())
        self.assertEqual(len(self.mod.meta()["history"]), 1)


# ============================================================
# The tab's glue
# ============================================================
class TestTabGlue(unittest.TestCase):

    def test_non_finite_numbers_are_nulled_so_the_json_stays_parseable(self):
        # json.dumps writes bare NaN, which JSON.parse rejects outright -- the
        # page then sees an empty object instead of an error.
        from gui.eegio_tab import _finite

        got = _finite({"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": 2.5}})
        self.assertIsNone(got["a"])
        self.assertEqual(got["b"], [1.0, None])
        self.assertEqual(got["c"]["d"], 2.5)

        import json
        json.loads(json.dumps(got, allow_nan=False))      # raises if any slipped through

    def test_an_absent_dataset_is_reported_with_where_to_get_it(self):
        from gui.eegio_tab import dataset_status

        with tempfile.TemporaryDirectory() as empty:
            st = dataset_status(empty)
        self.assertFalse(st["present"])
        self.assertIn("gnan.ece.gatech.edu", st["url"])

    def test_a_present_dataset_lists_its_subjects(self):
        from gui.eegio_tab import dataset_status

        with tempfile.TemporaryDirectory() as d:
            for i in range(2):
                write_synthetic_recording(d, f"S{i:02d}", duration_s=4.0, n_blinks=2,
                                          n_soft=0, seed=i)
            st = dataset_status(d)
        self.assertTrue(st["present"])
        self.assertEqual(st["n"], 2)

    def test_evaluate_refuses_without_a_run(self):
        from gui.eegio_tab import evaluate

        with self.assertRaises(ValueError):
            evaluate({"runs": []})

    def test_the_suggested_comparison_is_the_ladder_and_its_two_ends(self):
        from gui.eegio_tab import SUGGESTED

        self.assertEqual(SUGGESTED[0], "tcn_se_both")          # causal baseline
        self.assertTrue(any("la16" in s for s in SUGGESTED))   # the 62.5 ms rung
        self.assertTrue(any(s.endswith("_bi_both") for s in SUGGESTED))


# ============================================================
# End to end on synthetic recordings
# ============================================================
class TestEndToEnd(unittest.TestCase):

    def test_offline_mode_shares_filter_and_scores_the_same_interval(self):
        from gui import eegio_tab

        with tempfile.TemporaryDirectory() as tmp:
            rec = write_synthetic_recording(tmp, "S00", duration_s=6, n_blinks=1)
            delayed = _Identity()
            delayed.run_config = {"valid_history_samples": 150}
            pair = {"lookahead": 16, "master_seed": 42, "runs": ["a", "b"]}
            progress = []
            with patch.object(eegio_tab, "RESULTS_DIR", Path(tmp) / "results"), \
                 patch.object(eegio_tab, "available_runs", return_value=[]), \
                 patch.object(E, "load_run", side_effect=[(_Identity(), {}), (delayed, {})]), \
                 patch.object(E, "monotonic", side_effect=count()), \
                 patch("experiment.methodology.context", return_value=(0, 0)), \
                 patch("experiment.methodology.model_stream_checks", return_value={}):
                out = eegio_tab.evaluate({"runs": pair["runs"], "data_dir": tmp, "target_fs": 250,
                                          "offline": True, "zero_phase": True},
                                         lookahead_comparison=pair,
                                         on_progress=lambda update: progress.append(json.dumps(update)))
                entry = eegio_tab.list_results()[0]
                self.assertEqual(eegio_tab.read_result(out["name"])["saved"]["lookahead_comparison"], pair)
            self.assertTrue(any(json.loads(message)["phase"] == "inference" for message in progress))
            self.assertEqual(entry["lookahead_comparison"], pair)
            self.assertEqual(entry["offline_window"], {"samples": 500, "left": 149, "right": 125})
            self.assertIsNone(entry["hop"])
            self.assertEqual(out["saved"]["blink_count_adjustment"]["reference_run"], "a")
            reference = out["results"][0]["rows"][0]["metric_support"]["adjusted_blink_count"]
            for r in out["results"]:
                self.assertTrue(r["config"]["zero_phase"])
                self.assertEqual(r["config"]["offline_window"], entry["offline_window"])
                self.assertEqual(r["rows"][0]["valid_target_interval"], [149, rec.n_samples - 125])
                self.assertLess(r["rows"][0]["background_rmse_uv"], 1e-5)
                self.assertEqual(r["rows"][0]["metric_support"]["adjusted_blink_count"], reference)
                self.assertIn(E.ADJUSTED_BLINK_COUNT, entry["summary"][r["run"]])

    def test_gui_stream_reports_progress_before_returning_the_saved_result(self):
        from gui import eegio_tab
        from gui.server import Handler

        with tempfile.TemporaryDirectory() as tmp:
            for subject in ("S00", "S01"):
                write_synthetic_recording(tmp, subject, duration_s=4, n_blinks=1)
            results_dir = Path(tmp) / "results"
            body = json.dumps({"runs": ["first", "second"], "data_dir": tmp}).encode()
            request = (f"POST /api/eegio/evaluate?progress=1 HTTP/1.0\r\n"
                       f"Content-Length: {len(body)}\r\n\r\n").encode() + body
            proceed = threading.Event()
            original_read = E.read_recording

            def read_recording(*args):
                if not proceed.wait(10):
                    raise TimeoutError("No progress arrived before reading the recording")
                return original_read(*args)

            client, server = socket.socketpair()
            checks = {"parity": {"passed": True}, "causality": {"passed": True}}
            with client, server, patch.object(eegio_tab, "RESULTS_DIR", results_dir), \
                 patch.object(eegio_tab, "available_runs", return_value=[]), \
                 patch.object(E, "load_run", return_value=(_Identity(), {})), \
                 patch("experiment.methodology.model_stream_checks", return_value=checks), \
                 patch.object(E, "monotonic", side_effect=count()), \
                 patch.object(E, "read_recording", side_effect=read_recording):
                def handle():
                    with server:
                        Handler(server, ("local", 0), SimpleNamespace(verbose=False))

                thread = threading.Thread(target=handle, daemon=True)
                thread.start()
                client.settimeout(15)
                try:
                    client.sendall(request)
                    with client.makefile("rb") as stream:
                        self.assertIn(b"200 OK", stream.readline())
                        while stream.readline() not in (b"\r\n", b""):
                            pass
                        self.assertEqual(json.loads(stream.readline())["progress"]["phase"], "loading_model")
                        first = json.loads(stream.readline())["progress"]
                        self.assertEqual((first["subject"], first["completed"], first["total"]), ("S00", 0, 4))
                        proceed.set()
                        events = [json.loads(line) for line in stream]
                finally:
                    proceed.set()
                    client.close()
                    thread.join(15)

            progress = [event["progress"] for event in events[:-1]]
            self.assertTrue(any(0 < p.get("samples", 0) < p.get("total_samples", 0) for p in progress))
            subjects = [p for p in progress if p["phase"] == "subject"]
            self.assertEqual([(p["model"], p["subject"], p["completed"]) for p in subjects],
                             [("first", "S01", 1), ("second", "S00", 2), ("second", "S01", 3)])
            self.assertEqual(progress[-1], {"phase": "saving", "completed": 4, "total": 4})
            out = events[-1]["result"]
            self.assertEqual((out["hop"], out["saved"]["n_subjects"]), (4, 2))
            self.assertEqual(out["metric_protocol"], E.METRIC_PROTOCOL)
            self.assertEqual(set(out["metrics"]), set(E.METRIC_PATHS))
            self.assertEqual(set(out["legacy_metrics"]), set(E.LEGACY_METRIC_PATHS))
            self.assertEqual({t["metric"] for t in out["tests"]}, set(E.ALL_METRIC_PATHS))
            self.assertEqual([r["inference_checks"] for r in out["results"]], [checks, checks])
            self.assertNotIn("onset_leakage_ms", out["results"][0]["summary"])
            self.assertTrue((results_dir / out["name"]).is_file())

    def test_a_whole_evaluation_runs_and_reports_every_metric(self):
        with tempfile.TemporaryDirectory() as d:
            subs = [write_synthetic_recording(d, f"S{i:02d}", duration_s=20.0,
                                              n_blinks=6, n_soft=0, seed=i,
                                              amplitude=60.0, drift=0.0).subject
                    for i in range(2)]
            cfg = E.EvalConfig(hop=256)
            out = E.evaluate_subjects(d, "identity", subs, cfg,
                                      apply_fn=lambda x: x * 0.5)
        self.assertEqual(len(out["rows"]), 2)
        self.assertEqual(out["causality"]["past_diff"], 0.0)
        for row in out["rows"]:
            self.assertGreater(row["n_kept"], 0)
            self.assertIn("suppression_db", row)
            self.assertIn("rate", row["residual"])
            self.assertNotIn("onset_leakage", row)
            self.assertIn("median_db", row["risk"])
        for metric in E.METRIC_PATHS:
            self.assertIn(metric, out["summary"])

    def test_the_corrupt_stretches_are_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            write_synthetic_recording(d, "S00", duration_s=20.0, n_blinks=6, n_soft=0,
                                      corrupt=[(4.0, 10.0)], seed=3,
                                      amplitude=60.0, drift=0.0)
            cfg = E.EvalConfig(hop=256)
            out = E.evaluate_subjects(d, "identity", ["S00"], cfg, apply_fn=lambda x: x)
        self.assertGreater(out["rows"][0]["corrupt_fraction"], 0.2)


if __name__ == "__main__":
    unittest.main()
