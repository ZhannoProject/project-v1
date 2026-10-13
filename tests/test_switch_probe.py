"""
The non-stationary artifact-switch probe and the response curve measured on it.

Covers what the experiment's conclusions rest on: that the probe is
reproducible and leak-free, that the switch really is where it says it is at
the SNR it says, that the matched control differs from the probe only after
the switch point, and that the response curve and its summary read those
properties back out correctly.

Runs on small synthetic .npy files, so no dataset download is needed.
"""

import sys
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import eval as evaltools
from data.Dataloader import (ArtifactSwitchDataset, EEGDenoiseDataset,
                             SWITCH_PROBE_SEEDS, artifact_gain,
                             build_switch_probes)

T = 512


def _make_data(data_dir: Path, n=200, seed=7):
    """Random epochs with per-type character, so a switch is detectable."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi * 8, T)
    # Clean EEG: smooth-ish.  EOG: a localised low-frequency bump (blink-like)
    # anywhere in the epoch, on a small floor — like the real thing, an EOG
    # epoch is strongly non-stationary but never exactly silent.  EMG:
    # broadband throughout.
    eeg = rng.standard_normal((n, T)).astype(np.float32)
    eog = 0.2 * rng.standard_normal((n, T)).astype(np.float32)
    for i in range(n):
        c = rng.integers(64, T - 64)
        eog[i] += np.exp(-((np.arange(T) - c) ** 2) / (2 * 40.0 ** 2)) * rng.uniform(5, 15)
    emg = (rng.standard_normal((n, T)) * np.cos(t)).astype(np.float32)

    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "EEG_all_epochs.npy", eeg)
    np.save(data_dir / "EOG_all_epochs.npy", eog)
    np.save(data_dir / "EMG_all_epochs.npy", emg)


class _Passthrough(torch.nn.Module):
    """A causal 'model' that returns its input — enough to exercise the curve."""

    def forward(self, x):
        return x


class _Delay(torch.nn.Module):
    """Causal: output at t depends only on t and t-1."""

    def forward(self, x):
        return 0.5 * (x + torch.nn.functional.pad(x, (1, 0))[..., :-1])


class _Future(torch.nn.Module):
    """Non-causal: output at t peeks one sample ahead."""

    def forward(self, x):
        return 0.5 * (x + torch.nn.functional.pad(x, (0, 1))[..., 1:])


class ProbeTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="switch_probe_"))
        _make_data(cls.tmp / "data")
        cls.data_dir = str(cls.tmp / "data")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def probe(self, **kw):
        kw.setdefault("n_epochs", 16)
        return ArtifactSwitchDataset(self.data_dir, **kw)

    # ---------- reproducibility ----------

    def test_same_seed_is_identical(self):
        a, b = self.probe(probe_seed=0), self.probe(probe_seed=0)
        np.testing.assert_array_equal(a.switch_idx, b.switch_idx)
        np.testing.assert_array_equal(a.eog_first, b.eog_first)
        for i in (0, 3, 7):
            self.assertTrue(torch.equal(a[i]["noisy"], b[i]["noisy"]))

    def test_probe_seed_moves_the_draws(self):
        a, b = self.probe(probe_seed=0), self.probe(probe_seed=1)
        self.assertFalse(np.array_equal(a.switch_idx, b.switch_idx))
        self.assertFalse(np.allclose(a.snr_eog, b.snr_eog))

    def test_getitem_is_pure(self):
        """Access order must not change what a sample contains."""
        ds = self.probe(probe_seed=0)
        first = ds[2]["noisy"].clone()
        for j in (9, 4, 2, 11):
            ds[j]
        self.assertTrue(torch.equal(first, ds[2]["noisy"]))

    def test_series_covers_the_seeds(self):
        seeds = [s for s, _ in build_switch_probes(self.data_dir, n_epochs=8)]
        self.assertEqual(seeds, list(SWITCH_PROBE_SEEDS))

    # ---------- the design the reviewer asked for ----------

    def test_switch_point_is_randomised_inside_the_range(self):
        ds = self.probe(probe_seed=0, n_epochs=40, switch_range=(0.25, 0.75))
        lo, hi = ds.switch_bounds
        self.assertEqual((lo, hi), (128, 384))
        self.assertTrue((ds.switch_idx >= lo).all() and (ds.switch_idx <= hi).all())
        self.assertGreater(len(np.unique(ds.switch_idx)), 10, "switch point is not varying")

    def test_directions_are_exactly_balanced(self):
        ds = self.probe(probe_seed=0, n_epochs=None)
        self.assertEqual(ds.eog_first.sum(), len(ds) // 2)

    def test_order_can_be_pinned(self):
        self.assertTrue(self.probe(order="eog_first").eog_first.all())
        self.assertFalse(self.probe(order="emg_first").eog_first.any())

    def test_snrs_are_drawn_independently(self):
        ds = self.probe(probe_seed=0, n_epochs=40)
        self.assertFalse(np.allclose(ds.snr_eog, ds.snr_emg))
        self.assertLess(abs(np.corrcoef(ds.snr_eog, ds.snr_emg)[0, 1]), 0.5)

    def test_fixed_snr_overrides_both(self):
        ds = self.probe(fixed_snr_db=-3.0)
        np.testing.assert_allclose(ds.snr_eog, -3.0)
        np.testing.assert_allclose(ds.snr_emg, -3.0)

    def test_each_segment_realises_its_own_snr(self):
        """The requested SNR must hold *within* the segment it was drawn for."""
        ds = self.probe(probe_seed=0, n_epochs=24)
        for i in range(len(ds)):
            k = int(ds.switch_idx[i])
            s = ds[i]
            clean = ds.clean[i]
            artifact = s["noisy"].squeeze().numpy() * s["sigma"].item() - clean

            def snr(a, b):
                return 20 * np.log10(np.sqrt(np.mean(a ** 2)) / np.sqrt(np.mean(b ** 2)))

            want_pre = ds.snr_eog[i] if ds.eog_first[i] else ds.snr_emg[i]
            want_post = ds.snr_emg[i] if ds.eog_first[i] else ds.snr_eog[i]
            self.assertAlmostEqual(snr(clean[:k], artifact[:k]), want_pre, places=3)
            self.assertAlmostEqual(snr(clean[k:], artifact[k:]), want_post, places=3)

    def test_ramp_softens_the_boundary(self):
        """A ramp must spread the step, and must not move the switch index."""
        hard = self.probe(probe_seed=0, ramp_samples=0)
        soft = self.probe(probe_seed=0, ramp_samples=32)
        np.testing.assert_array_equal(hard.switch_idx, soft.switch_idx)
        w_hard = hard._switch_weight(200)
        w_soft = soft._switch_weight(200)
        self.assertEqual(set(np.unique(w_hard)), {0.0, 1.0})
        self.assertGreater(len(np.unique(w_soft)), 10)
        self.assertTrue(np.all(np.diff(w_soft) >= -1e-6), "ramp must be monotone")
        self.assertAlmostEqual(float(w_soft[0]), 0.0)
        self.assertAlmostEqual(float(w_soft[-1]), 1.0)

    # ---------- the matched control ----------

    def test_control_matches_the_probe_before_the_switch(self):
        """
        The whole method rests on this: probe and control are one signal on
        [0, k), so a causal model's difference there is exactly zero.
        """
        probe = self.probe(probe_seed=0)
        ctrl = self.probe(probe_seed=0, control=True)
        for i in range(len(probe)):
            k = int(probe.switch_idx[i])
            a, b = probe[i], ctrl[i]
            self.assertEqual(a["sigma"].item(), b["sigma"].item())
            np.testing.assert_array_equal(a["noisy"].squeeze().numpy()[:k],
                                          b["noisy"].squeeze().numpy()[:k])
            np.testing.assert_array_equal(a["clean"].numpy(), b["clean"].numpy())

    def test_control_differs_after_the_switch(self):
        probe = self.probe(probe_seed=0)
        ctrl = self.probe(probe_seed=0, control=True)
        diffs = []
        for i in range(len(probe)):
            k = int(probe.switch_idx[i])
            diffs.append(np.abs(probe[i]["noisy"].squeeze().numpy()[k:]
                                - ctrl[i]["noisy"].squeeze().numpy()[k:]).max())
        self.assertGreater(min(diffs), 0.0, "control must splice something different")

    def test_control_splices_a_different_epoch_of_the_same_type(self):
        ds = self.probe(probe_seed=0, n_epochs=40)
        for i in range(len(ds)):
            self.assertFalse(np.array_equal(ds.eog_alt[i], ds.eog[i]))
            self.assertFalse(np.array_equal(ds.emg_alt[i], ds.emg[i]))

    # ---------- leak-free ----------

    def test_draws_only_from_its_own_split(self):
        """
        Probe epochs must come from the same test split the model was scored
        on, and never from the training split.
        """
        clean_all = np.load(Path(self.data_dir) / "EEG_all_epochs.npy")
        test_idx = EEGDenoiseDataset._split_indices(len(clean_all), (0.7, 0.15, 0.15),
                                                   "test", 42)
        train_idx = EEGDenoiseDataset._split_indices(len(clean_all), (0.7, 0.15, 0.15),
                                                     "train", 42)
        test_rows = {clean_all[i].tobytes() for i in test_idx}
        train_rows = {clean_all[i].tobytes() for i in train_idx}
        for seed in (0, 1):
            ds = self.probe(probe_seed=seed, n_epochs=None)
            for row in ds.clean:
                self.assertIn(row.tobytes(), test_rows)
                self.assertNotIn(row.tobytes(), train_rows)

    def test_rejects_a_range_with_no_room(self):
        with self.assertRaises(ValueError):
            self.probe(switch_range=(0.49, 0.51), ramp_samples=512)


class ResponseCurveTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="switch_curve_"))
        _make_data(cls.tmp / "data")
        cls.data_dir = str(cls.tmp / "data")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def evaluate(self, model, **kw):
        kw.setdefault("probe_seeds", (0, 1))
        kw.setdefault("n_epochs", 16)
        return evaltools.evaluate_switch_response(
            model, self.data_dir, batch_size=8, device=torch.device("cpu"),
            verbose=False, **kw)

    def test_curve_shape_and_alignment(self):
        res = self.evaluate(_Passthrough())
        off = res["offsets"]
        self.assertEqual(off[0], -off[-1])            # symmetric about the switch
        self.assertIn(0, off)
        self.assertEqual(len(off), len(res["all"]["err"]))
        np.testing.assert_allclose(res["t_ms"], off / 256 * 1000)
        self.assertEqual(res["all"]["err"].shape, res["all"]["err_std"].shape)
        for d in ("all", "eog_to_emg", "emg_to_eog"):
            self.assertIn(d, res)
            self.assertEqual(len(res[d]["per_seed"]), 2)

    def test_window_is_clipped_to_what_the_range_allows(self):
        res = self.evaluate(_Passthrough(), window=10_000)
        self.assertEqual(res["offsets"][-1], 127)      # min(128, 511-384)

    def test_a_causal_model_leaks_nothing_before_the_switch(self):
        res = self.evaluate(_Delay())
        s = res["all"]["summary"]
        self.assertEqual(s["pre_leak"], 0.0)
        self.assertEqual(s["pre_leak_far"], 0.0)
        np.testing.assert_allclose(res["all"]["delta"][res["offsets"] < 0], 0.0, atol=0)

    def test_the_leak_measures_the_real_look_ahead(self):
        """
        The probe's answer must agree with the model's actual forward
        receptive field, measured by perturbing one input sample.
        """
        from models.variants import build_model
        from utils.utils import seed_everything

        seed_everything(0)
        model = build_model("tcn_bi", hidden_channels=16).eval()

        x = torch.randn(1, 1, 512)
        with torch.no_grad():
            base = model(x)
            bumped = x.clone()
            bumped[0, 0, 256] += 10.0
            moved = (base - model(bumped)).abs().squeeze()
        # Exactly zero, not a tolerance: outside the receptive field the two
        # forward passes do identical arithmetic on identical numbers.  A
        # tolerance clips the edge of the field and costs a sample or two.
        first = int((moved > 0).nonzero().min())
        lookahead = 256 - first
        self.assertGreater(lookahead, 0, "tcn_bi should not be causal")

        res = self.evaluate(model, probe_seeds=(0,), n_epochs=24)
        off, delta = res["offsets"], np.abs(res["all"]["delta"])
        pre = off < 0
        departs = off[pre][np.argmax(delta[pre] > 0)]
        self.assertEqual(int(-departs), lookahead,
                         "the probe's leak onset must be the receptive field")

    def test_a_non_causal_model_does(self):
        res = self.evaluate(_Future())
        self.assertGreater(res["all"]["summary"]["pre_leak"], 0.0)
        # One sample of look-ahead: the leak is at the last pre-switch offset
        # only, so the far half of the window stays clean.
        self.assertEqual(res["all"]["summary"]["pre_leak_far"], 0.0)
        self.assertGreater(abs(res["all"]["delta"][res["offsets"] == -1][0]), 0.0)

    def test_no_control_means_no_delta_fields(self):
        res = self.evaluate(_Passthrough(), control=False)
        self.assertNotIn("delta", res["all"])
        self.assertNotIn("pre_leak", res["all"]["summary"])
        self.assertIn("floor_transient_area", res["all"]["summary"])

    def test_summary_separates_the_transient_from_the_regime_offset(self):
        offsets = np.arange(-8, 9)
        err = np.full(17, 0.5)
        # A curve that steps to a new level and spikes on the way.
        err[offsets >= 0] = 0.8
        err[offsets == 0] = 2.0
        delta = err - 0.5
        s = evaltools._switch_summary(offsets, err, fs=256, recovery_tol=0.1,
                                      floor_frac=0.5, delta=delta)
        self.assertAlmostEqual(s["post_floor"], 0.8, places=6)
        self.assertAlmostEqual(s["delta_floor"], 0.3, places=6)     # the regime offset
        self.assertAlmostEqual(s["transient_peak"], 1.2, places=6)  # 2.0 - 0.8
        self.assertAlmostEqual(s["transient_recovery_ms"], 1 / 256 * 1000, places=6)
        self.assertGreater(s["transient_area"], 0.0)

    def test_result_round_trips_through_json(self):
        import json
        res = self.evaluate(_Delay())
        blob = json.loads(json.dumps(evaltools._switch_result_to_json(res)))
        self.assertEqual(blob["probe_seeds"], [0, 1])
        self.assertEqual(len(blob["all"]["err"]), len(res["offsets"]))
        self.assertIn("delta", blob["all"])
        self.assertIn("per_seed_delta", blob["all"])


if __name__ == "__main__":
    unittest.main()
