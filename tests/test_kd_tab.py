"""
The KD audit tab, and the recording fix it depends on.

Most of this runs without the dataset: the identity the decomposition rests
on is checked on small synthetic models, the read-off and the recipe on
hand-written rows. One class re-audits the project's real distilled run and
is skipped when ``runs/`` or the data is absent.

Run: python -m unittest discover -s tests -p 'test_kd_tab.py'
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui import kd                                               # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REAL_KD = ROOT / "runs" / "tcn_se_both_kd_s67" / "best.pt"
HAVE_DATA = (ROOT / "data" / "EEGdenoisenet" / "data" / "EEG_all_epochs.npy").is_file()


# ============================================================
# The recording fix
# ============================================================
class TestRecipeIsRecorded(unittest.TestCase):
    """trainer/training.py used to drop every kd_* key except teacher_ckpt."""

    def test_a_distilled_run_writes_its_whole_recipe_into_history(self):
        from trainer.training import Trainer

        recipe = {"teacher_ckpt": "runs/t/best.pt", "kd_alpha": 0.3,
                  "kd_feature_weight": 0.1, "kd_feature_layers": ["decoder"],
                  "kd_output_loss": "mse", "kd_feature_loss": "l1",
                  "init_from_teacher": True, "teacher_config": None,
                  "master_seed": 7}
        dl = DataLoader(TensorDataset(torch.zeros(2, 1, 8)), batch_size=1)
        with tempfile.TemporaryDirectory() as tmp:
            t = Trainer(nn.Conv1d(1, 1, 1), dl, dl, save_dir=tmp, loss_fn="mse",
                        device=torch.device("cpu"), run_config=recipe)
        cfg = t.history["config"]
        for key, value in recipe.items():
            with self.subTest(key=key):
                self.assertIn(key, cfg)
                self.assertEqual(cfg[key], value)


# ============================================================
# What the tab reads off disk
# ============================================================
class TestRecipeReading(unittest.TestCase):

    def _with_history(self, cfg):
        return mock.patch.object(kd, "_history", return_value={"config": cfg})

    def test_an_unrecorded_recipe_is_reported_as_such_not_defaulted(self):
        with self._with_history({"teacher_ckpt": "x"}):
            r = kd._recorded_recipe("anything")
        self.assertFalse(r["recorded"])
        self.assertIsNone(r["kd_alpha"])           # not the 0.5 default

    def test_a_recorded_recipe_is_passed_through(self):
        with self._with_history({"kd_alpha": 0.25, "init_from_teacher": False}):
            r = kd._recorded_recipe("anything")
        self.assertTrue(r["recorded"])
        self.assertEqual(r["kd_alpha"], 0.25)
        self.assertIs(r["init_from_teacher"], False)


class TestControlMatching(unittest.TestCase):
    """The control is the same model, seed and width, trained without a teacher."""

    BASE = {"variant": "tcn_se", "artifact": "BOTH", "master_seed": 67,
            "hidden_channels": 64, "status": "complete", "kd": False}

    def row(self, name, **kw):
        return {**self.BASE, "name": name, **kw}

    def test_it_finds_the_matching_plain_run(self):
        student = self.row("s_kd", kd=True)
        rows = [student, self.row("other_seed", master_seed=42),
                self.row("narrow", hidden_channels=48), self.row("the_control")]
        self.assertEqual(kd._control_for(student, rows)["name"], "the_control")

    def test_another_kd_run_is_never_a_control(self):
        student = self.row("s_kd", kd=True)
        rows = [student, self.row("another_kd", kd=True)]
        self.assertIsNone(kd._control_for(student, rows))

    def test_a_stale_run_is_never_a_control(self):
        student = self.row("s_kd", kd=True)
        rows = [student, self.row("stale", status="stale")]
        self.assertIsNone(kd._control_for(student, rows))


# ============================================================
# The decomposition
# ============================================================
class _Scale(nn.Module):
    """A deterministic stand-in denoiser: y = a * x + b."""

    def __init__(self, a, b=0.0):
        super().__init__()
        self.a, self.b = a, b

    def forward(self, x):
        return self.a * x + self.b


class TestDecomposition(unittest.TestCase):

    def setUp(self):
        g = torch.Generator().manual_seed(0)
        clean = torch.randn(64, 1, 32, generator=g)
        noisy = clean + 0.5 * torch.randn(64, 1, 32, generator=g)
        self.batches = [{"noisy": noisy[i:i + 16], "clean": clean[i:i + 16]}
                        for i in range(0, 64, 16)]
        self.models = {"T": _Scale(0.9), "plain": _Scale(0.7), "kd": _Scale(0.85, 0.01)}

    def run_decompose(self, students):
        hist = {"plain": {"config": {}}, "kd": {"config": {"teacher_ckpt": "T"}}}
        with mock.patch.object(kd, "_load", side_effect=lambda r: (self.models[r], {})), \
             mock.patch.object(kd, "_loader", return_value=self.batches), \
             mock.patch.object(kd, "_history", side_effect=lambda r: hist.get(r, {})):
            return kd.decompose("T", students, max_batches=100)

    def test_the_three_terms_sum_to_the_direct_error(self):
        out = self.run_decompose(["plain", "kd"])
        for r in out["rows"]:
            with self.subTest(run=r["run"]):
                self.assertLess(r["residual"], 1e-6)
                self.assertAlmostEqual(r["sum"], r["s_minus_g"], places=6)

    def test_the_teacher_term_is_one_constant_for_every_student(self):
        out = self.run_decompose(["plain", "kd"])
        self.assertEqual(len({round(r["t_minus_g"], 12) for r in out["rows"]}), 1)

    def test_the_control_is_listed_first_and_a_delta_is_read_off(self):
        out = self.run_decompose(["kd", "plain"])
        self.assertEqual([r["run"] for r in out["rows"]], ["plain", "kd"])
        d = out["delta"]
        self.assertEqual((d["control"], d["kd"]), ("plain", "kd"))
        self.assertAlmostEqual(d["d_s_minus_g"], d["d_s_minus_t"] + d["d_cross"], places=6)

    def test_a_student_identical_to_the_teacher_has_no_distance_and_no_cross_term(self):
        self.models["copy"] = self.models["T"]
        out = self.run_decompose(["copy"])
        r = out["rows"][0]
        self.assertAlmostEqual(r["s_minus_t"], 0.0, places=12)
        self.assertAlmostEqual(r["cross"], 0.0, places=12)
        self.assertNotIn("delta", out)        # one student: nothing to compare


class TestReadOff(unittest.TestCase):

    def rows(self, st_ctrl, st_kd, sg_ctrl, sg_kd):
        mk = lambda name, st, sg: {"run": name, "s_minus_t": st, "cross": sg - st - 1.0,
                                   "s_minus_g": sg}
        return mk("c", st_ctrl, sg_ctrl), mk("k", st_kd, sg_kd)

    def test_closer_and_worse_reads_as_kd_costing_accuracy(self):
        d = kd._read_off(*self.rows(0.04, 0.03, 1.10, 1.12))
        self.assertIn("cost accuracy", d["verdict"])

    def test_not_closer_reads_as_the_distillation_doing_nothing(self):
        d = kd._read_off(*self.rows(0.03, 0.04, 1.10, 1.12))
        self.assertIn("did not move toward the teacher", d["verdict"])

    def test_closer_and_better_reads_as_kd_helping(self):
        d = kd._read_off(*self.rows(0.04, 0.03, 1.10, 1.08))
        self.assertIn("improved it", d["verdict"])


# ============================================================
# The tab's metadata
# ============================================================
class TestMeta(unittest.TestCase):

    def test_meta_names_all_six_checks_and_the_identity(self):
        with mock.patch.object(kd, "kd_runs", return_value=[]):
            m = kd.meta()
        self.assertEqual(len(m["check_names"]), 6)
        self.assertIn("2<S-T, T-G>", m["identity"])
        self.assertEqual(m["kd_runs"], [])


# ============================================================
# The project's own distilled run, when it is on disk
# ============================================================
@unittest.skipUnless(REAL_KD.is_file() and HAVE_DATA, "runs/ or the dataset is absent")
class TestTheRealRun(unittest.TestCase):

    def test_all_six_checks_hold_and_the_teacher_reproduces_exactly(self):
        c = kd.static_checks("tcn_se_both_kd_s67")
        failed = [x for x in c["checks"] if x["ok"] is False]
        self.assertEqual(failed, [], json.dumps(failed, indent=1))
        reproduce = next(x for x in c["checks"]
                         if x["name"] == "teacher reproduces its recorded metrics")
        self.assertIs(reproduce["ok"], True)

    def test_the_run_is_found_with_its_teacher_and_its_control(self):
        row = next(r for r in kd.kd_runs() if r["name"] == "tcn_se_both_kd_s67")
        self.assertEqual(row["teacher_run"], "tcn_se_bi_both_s67")
        self.assertEqual(row["control"], "tcn_se_both_s67_h64")
        # Whether the recipe is recorded depends on when this directory was
        # last trained into, so it is not asserted either way -- only that
        # the question has an answer.
        self.assertIsInstance(row["recipe"]["recorded"], bool)

    def test_the_audit_scores_the_teacher_at_the_rate_the_run_trained_at(self):
        # A run trained with target_fs 250 was validated on 500-sample
        # epochs. Re-scoring its teacher on the 512-sample ones compares two
        # different datasets, and the "reproduces its recorded metrics"
        # check would report a mismatch for a reason that has nothing to do
        # with the teacher.
        row = next(r for r in kd.kd_runs() if r["name"] == "tcn_se_both_kd_s67")
        checks = {c["name"]: c for c in kd.static_checks(row["name"])["checks"]}
        self.assertIs(checks["teacher reproduces its recorded metrics"]["ok"], True,
                      checks["teacher reproduces its recorded metrics"]["detail"])


if __name__ == "__main__":
    unittest.main()
