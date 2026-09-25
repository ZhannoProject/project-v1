"""Run: python -m unittest discover -s tests -p 'test_identify_loss.py'."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from trainer.identify_loss import REL_TOL, artifact_of, pick, recorded_loss


class PickTests(unittest.TestCase):
    """A wrong loss label is worse than none, so pick() only claims a clear winner."""

    # Real candidate values, measured on runs/tcn_mha_par_emg/best.pt
    REAL = {"mse": 0.1300, "l1": 0.2786, "stft": 2.9112,
            "l2_combined": 0.2755, "l1_combined": 0.4242}

    def test_identifies_the_loss_that_reproduces_the_recorded_value(self):
        for name, target in (("mse", 0.1300), ("l1_combined", 0.4242), ("stft", 2.9112)):
            winner, detail = pick(self.REAL, target)
            self.assertEqual(winner, name)
            self.assertLess(detail["best_rel_err"], REL_TOL)

    def test_small_float_drift_is_still_a_match(self):
        """Recomputing on a different device drifts the last digits, as on
        runs/tcn_mla_par_bi_emg: recorded 0.4376 vs recomputed 0.4379."""
        real = {**self.REAL, "l1_combined": 0.4379}
        winner, detail = pick(real, 0.4376)
        self.assertEqual(winner, "l1_combined")
        self.assertLess(detail["best_rel_err"], 0.002)

    def test_drift_beyond_the_tolerance_is_refused(self):
        """3% off is not float noise — it is a different model or a different loss."""
        self.assertIsNone(pick(self.REAL, 0.4379)[0])   # candidates say l1_combined = 0.4242

    def test_two_close_candidates_are_refused(self):
        """l1 (0.2786) and l2_combined (0.2755) are 1% apart — neither may be claimed."""
        winner, detail = pick(self.REAL, 0.2770)
        self.assertIsNone(winner)
        self.assertLess(detail["runner_up_rel_err"], 0.05)

    def test_a_value_no_candidate_produces_is_refused(self):
        self.assertIsNone(pick(self.REAL, 0.85)[0])

    def test_zero_target_does_not_divide_by_zero(self):
        self.assertIsNone(pick(self.REAL, 0.0)[0])


class RecordedLossTests(unittest.TestCase):
    def test_reads_history_config_then_checkpoint_then_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self.assertIsNone(recorded_loss(d))

            torch.save({"epoch": 1, "model_state": {}, "run_config": {"loss_fn": "mse"}},
                       d / "best.pt")
            self.assertEqual(recorded_loss(d), "mse")       # from the checkpoint

            (d / "history.json").write_text(json.dumps(
                {"val_loss": [1.0], "config": {"loss_fn": "l1_combined"}}))
            self.assertEqual(recorded_loss(d), "l1_combined")   # history wins

    def test_legacy_run_records_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            torch.save({"epoch": 1, "model_state": {}}, d / "best.pt")
            (d / "history.json").write_text(json.dumps({"val_loss": [1.0], "rrmse_t": [0.7]}))
            self.assertIsNone(recorded_loss(d))


class ArtifactTests(unittest.TestCase):
    def test_artifact_comes_from_the_directory_name(self):
        for name, want in (("tcn_mla_eog", "EOG"), ("tcn_mha_par_bi_emg", "EMG"),
                           ("ctrl_mla_both_causal", "BOTH"), ("tcn_mla_eog_kd", "EOG"),
                           ("something_else", None)):
            self.assertEqual(artifact_of(name), want, name)


if __name__ == "__main__":
    unittest.main()
