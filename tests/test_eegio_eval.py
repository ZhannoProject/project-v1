"""
Evaluating on real recordings: the filter, the detector, the four metrics.

None of this needs EEG-IO. The filters are probed the way the project probes
a model, the detector is run on signals with blinks planted at known
samples, and each metric is given a model whose answer is known in advance --
identity, silence, a pure delay, an anticipator. That is the only way to
check a metric: on a model you already know the right answer for.

Run: python -m unittest discover -s tests -p 'test_eegio_eval.py'
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

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
            self.assertIn("median_ms", row["onset_leakage"])
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
