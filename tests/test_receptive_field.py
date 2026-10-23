"""The receptive-field sweep: every rung must reach what its name says.

Run: PYTHONPATH=. python -m unittest discover -s tests -p 'test_receptive_field.py'

The sweep's whole content is a curve of error against reach, so the x-axis has
to be right. A schedule that does not produce the field it claims would put
every point in the wrong place and the curve would still look plausible --
which is why the architectural number is checked against a model that answers
for itself, twice over, by two independent methods.
"""
import os
import pathlib
import sys
import unittest

ROOT_DIR = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT_DIR))

import torch

from models.incremental import IncrementalModel
from models.receptive_field import (DEFAULT_DILATIONS, RF_LADDER, architectural, field,
                                    measured_field, ms, rung_table)
from models.variants import VARIANT_REGISTRY, base_variant, build_model, resolved_arch


class LadderTests(unittest.TestCase):
    def test_every_rung_reaches_what_the_formula_says(self):
        """
        Three ways of asking, which must agree: the formula, the sum over the
        model's own convolutions, and a perturbation that watches how far one
        sample carries.
        """
        for dil in RF_LADDER:
            with self.subTest(dilations=dil):
                model = build_model("tcn", dilations=list(dil)).eval()
                expected = field(dil)
                self.assertEqual(architectural(model), expected)
                self.assertEqual(measured_field(model, seq_len=4 * expected + 64), expected)

    def test_the_streaming_wrapper_agrees(self):
        """
        ``IncrementalModel`` arrives at the field independently -- by counting
        the ring buffers it must allocate -- so it is a fourth opinion.
        """
        for dil in ((1, 2), (1, 2, 4, 8, 16), (1, 2, 4, 8, 16, 32, 64)):
            with self.subTest(dilations=dil):
                model = build_model("tcn", dilations=list(dil)).eval()
                self.assertEqual(IncrementalModel(model).receptive_field, field(dil))

    def test_the_ladder_doubles_and_spans_the_window(self):
        fields = [field(d) for d in RF_LADDER]
        self.assertEqual(fields, sorted(fields), "the ladder has to be ordered by reach")
        self.assertEqual(fields[0], 5)
        self.assertGreaterEqual(fields[-1], 500,
                                "the top rung should reach about as far as the attention window")
        for a, b in zip(fields, fields[1:]):
            self.assertGreater(b, 1.8 * a, "each rung should roughly double the reach")

    def test_the_default_is_on_the_ladder(self):
        """
        The rung every existing run used has to be one of the points, or the
        sweep cannot be compared with anything already measured.
        """
        self.assertIn(DEFAULT_DILATIONS, RF_LADDER)
        self.assertEqual(field(DEFAULT_DILATIONS), 125)
        self.assertEqual(architectural(build_model("tcn")), 125)

    def test_ms_converts_at_the_sampling_rate(self):
        self.assertAlmostEqual(ms(256), 1000.0, places=1)


class VariantTests(unittest.TestCase):
    def test_each_rung_has_a_variant_named_for_its_reach(self):
        """A run cannot then be mislabelled about the one axis being swept."""
        for base in ("tcn", "tcn_se"):
            for dil in RF_LADDER:
                name = f"{base}_rf{field(dil)}"
                with self.subTest(variant=name):
                    self.assertIn(name, VARIANT_REGISTRY)
                    model = build_model(name)
                    self.assertEqual(architectural(model), field(dil))
                    self.assertEqual(base_variant(name), base)

    def test_the_variant_records_its_schedule(self):
        arch = resolved_arch(build_model("tcn_rf253"))
        self.assertEqual(arch["dilations"], [1, 2, 4, 8, 16, 32])

    def test_the_default_variant_records_the_schedule_it_took(self):
        """
        A run that passes no ``--dilations`` still has to write down which
        ones it used, or the sweep cannot place the point it already has.
        Checked through JSON, because that is how a run records it.
        """
        import json
        arch = json.loads(json.dumps(resolved_arch(build_model("tcn"))))
        self.assertEqual(arch["dilations"], list(DEFAULT_DILATIONS))

    def test_the_default_variant_is_unchanged(self):
        """Every checkpoint already trained has to go on loading."""
        self.assertEqual(sum(p.numel() for p in build_model("tcn").parameters()), 117068)

    def test_a_longer_reach_costs_one_block(self):
        rows = {r["receptive_field"]: r for r in rung_table()}
        sizes = [rows[f]["params"] for f in sorted(rows)]
        steps = {b - a for a, b in zip(sizes, sizes[1:])}
        self.assertEqual(len(steps), 1, "each rung should add the same one residual block")


class SweepPlanTests(unittest.TestCase):
    """What the tab will actually run."""

    def setUp(self):
        from gui import rfsweep
        self.rfsweep = rfsweep

    def test_it_plans_one_run_per_rung_shortest_first(self):
        steps = self.rfsweep.build_steps(
            {"variant": "tcn", "artifact": "BOTH", "master_seed": 67,
             "max_epochs": 5, "rungs": [509, 29, 125]})
        self.assertEqual(len(steps), 3)
        order = [s["label"] for s in steps]
        self.assertTrue(order[0].startswith("receptive field 29 "), order)
        self.assertTrue(order[-1].startswith("receptive field 509 "), order)

    def test_each_command_pins_the_rung_and_its_own_directory(self):
        for step in self.rfsweep.build_steps(
                {"variant": "tcn", "artifact": "BOTH", "master_seed": 67, "max_epochs": 5}):
            cmd = step["command"]
            variant = cmd[cmd.index("--variant") + 1]
            save = cmd[cmd.index("--save_dir") + 1]
            with self.subTest(variant=variant):
                self.assertIn(variant, VARIANT_REGISTRY)
                self.assertIn(variant.split("_rf")[1], save,
                              "the run directory has to name the reach it holds")

    def test_each_command_trains_from_its_family_recipe(self):
        """
        Without a config the trainer falls back to l1_combined at lr 5e-3 for
        150 epochs, where every run on disk is mse at 5e-4 for 200. A rung
        launched that way would differ from the 125-sample point it is plotted
        against in the loss, the learning rate and the length, and the curve
        would read as a reach effect.
        """
        for variant, config in (("tcn", "configs/tcn_both.yaml"),
                                ("tcn_se", "configs/tcn_se_both.yaml")):
            for step in self.rfsweep.build_steps({"variant": variant, "rungs": [29, 509]}):
                cmd = step["command"]
                with self.subTest(variant=variant, cmd=" ".join(cmd[-6:])):
                    self.assertEqual(cmd[cmd.index("--config") + 1], config)

    def test_the_two_recipes_differ_only_in_the_gate(self):
        """
        The ladder is run for both families and the curves are read side by
        side, so the two configs have to be the same recipe. Compared as text
        rather than parsed, because a stray key is exactly what this is for.
        """
        def recipe(path):
            out = {}
            for line in (ROOT_DIR / path).read_text().splitlines():
                line = line.split("#")[0].strip()
                if ":" in line:
                    k, v = line.split(":", 1)
                    out[k.strip()] = v.strip()
            return out

        plain = recipe("configs/tcn_both.yaml")
        se = recipe("configs/tcn_se_both.yaml")
        self.assertNotIn("se_reduction", plain,
                         "a gate setting here would put SE on the baseline")
        self.assertEqual(se.get("se_reduction"), "8")
        expected = {"variant", "se_reduction", "save_dir", "attn_lr_scale"}
        self.assertEqual({k for k in set(plain) | set(se) if plain.get(k) != se.get(k)},
                         expected,
                         "the two ladder recipes differ in more than the gate")

    def test_rungs_off_the_ladder_are_refused(self):
        with self.assertRaisesRegex(ValueError, "not rungs of this ladder"):
            self.rfsweep.build_steps({"variant": "tcn", "rungs": [7]})

    def test_it_is_for_the_tcn_families(self):
        with self.assertRaisesRegex(ValueError, "TCN families"):
            self.rfsweep.build_steps({"variant": "tcn_mla", "rungs": [125]})

    def test_a_run_that_recorded_no_schedule_sits_on_the_default_rung(self):
        """
        Every `tcn` run already on disk predates the sweep, and the point they
        measured -- 125 samples -- is the one the sweep most needs, since it
        is where the existing study sits. They have to land on the curve
        without being retrained.
        """
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            run = pathlib.Path(tmp)
            (run / "history.json").write_text(json.dumps({"config": {"n_params": 117068}}))
            self.assertEqual(self.rfsweep._run_rf(run, "tcn"), 125)
            self.assertEqual(self.rfsweep._run_rf(run, "tcn_se"), 125)
            # A reach claimed by a name but recorded nowhere is not evidence.
            self.assertIsNone(self.rfsweep._run_rf(run, "tcn_rf509"))
            self.assertIsNone(self.rfsweep._run_rf(run, "tcn_mla"))
            # And what a run did record outranks all of it.
            (run / "history.json").write_text(
                json.dumps({"config": {"arch": {"dilations": [1, 2, 4]}}}))
            self.assertEqual(self.rfsweep._run_rf(run, "tcn"), 29)

    def test_results_place_a_run_by_what_it_recorded(self):
        """
        A point's position on the x-axis comes from the architecture the run
        wrote down, not from its directory name -- a run can be called
        anything.
        """
        import inspect
        source = inspect.getsource(self.rfsweep._run_rf)
        self.assertIn("history.json", source)
        self.assertIn("dilations", source)


class AggregationTests(unittest.TestCase):
    """
    A rung arrives with several runs on it the moment the tab opens -- the 125
    rung holds every ``tcn``/``tcn_se`` run the project has done -- so what
    goes on the point matters more here than on a sweep nobody has run yet.
    """

    def results(self, runs, **kw):
        from unittest import mock

        from gui import config_io, rfsweep
        with mock.patch.object(config_io, "list_runs", return_value=runs), \
             mock.patch.object(rfsweep, "_run_rf", lambda d, v=None: 125):
            return rfsweep.results(**kw)

    @staticmethod
    def a_run(name, **kw):
        row = {"name": name, "variant": "tcn_se", "artifact": "BOTH", "status": "complete",
               "master_seed": 42, "hidden_channels": 64, "params": 123838, "epochs": 200,
               "val_loss": 0.113, "rrmse_t": 0.708, "rrmse_s": 0.645, "acc": 0.711}
        row.update(kw)
        return row

    def test_seeds_at_one_reach_become_a_mean_and_a_spread(self):
        rows = self.results([self.a_run("a", master_seed=42, rrmse_t=0.70),
                             self.a_run("b", master_seed=54, rrmse_t=0.72)],
                            variant="tcn_se")
        self.assertEqual(len(rows), 1, "two seeds at one reach are one point")
        point = rows[0]
        self.assertEqual(point["n_runs"], 2)
        self.assertEqual(point["seeds"], [42, 54])
        self.assertAlmostEqual(point["rrmse_t"]["mean"], 0.71)
        self.assertAlmostEqual(point["rrmse_t"]["sd"], 0.0141, places=3)

    def test_a_run_of_another_width_is_excluded_and_named(self):
        """
        The mistake this guards against already happened once: a 48-channel
        run averaged in with two 64-channel ones made a three-seed error bar
        out of two seeds and a width experiment.
        """
        rows = self.results([self.a_run("wide", rrmse_t=0.70),
                             self.a_run("narrow", hidden_channels=48, rrmse_t=0.90,
                                      master_seed=67)],
                            variant="tcn_se")
        point = rows[0]
        self.assertEqual(point["n_runs"], 1)
        self.assertAlmostEqual(point["rrmse_t"]["mean"], 0.70)
        self.assertEqual([m["name"] for m in point["mismatched"]], ["narrow"])
        self.assertIn("48 channels, not 64", point["mismatched"][0]["why"])

    def test_a_distilled_run_is_not_a_seed_repeat(self):
        """
        ``runs/tcn_se_both_kd_s67`` is on the 125 rung and had a bidirectional
        teacher. That is a different training signal -- its own row in the
        ablation table -- so averaging it in would put the KD effect on the
        reach axis.
        """
        rows = self.results([self.a_run("plain", rrmse_t=0.70),
                             self.a_run("distilled", kd=True, rrmse_t=0.60, master_seed=67)],
                            variant="tcn_se")
        self.assertEqual(rows[0]["runs"], ["plain"])
        self.assertIn("distilled from a teacher", rows[0]["mismatched"][0]["why"])

    def test_a_run_trained_on_another_loss_is_excluded(self):
        """Its val loss is not on the same scale as the rest of the curve."""
        rows = self.results([self.a_run("mse"),
                             self.a_run("l1", loss_fn="l1_combined", master_seed=54)],
                            variant="tcn_se")
        self.assertEqual(rows[0]["runs"], ["mse"])
        self.assertIn("l1_combined", rows[0]["mismatched"][0]["why"])

    def test_the_loss_the_curve_demands_is_the_one_it_trains(self):
        """The constant and the configs cannot drift apart."""
        from gui import rfsweep
        for config in rfsweep.LADDER_CONFIGS.values():
            with self.subTest(config=config):
                self.assertIn(f'loss_fn: "{rfsweep.LADDER_LOSS}"',
                              (ROOT_DIR / config).read_text())

    def test_a_stale_run_is_on_neither_side(self):
        """Its metrics describe weights that are gone."""
        rows = self.results([self.a_run("good"), self.a_run("stale", status="stale")],
                            variant="tcn_se")
        self.assertEqual(rows[0]["runs"], ["good"])
        self.assertEqual(rows[0]["mismatched"], [])

    def test_a_rung_whose_runs_disagree_on_size_reports_none(self):
        """
        The parameter count is the rung's cost, so a number true of only some
        of its runs would be read as the rung's.
        """
        rows = self.results([self.a_run("a", params=123838),
                             self.a_run("b", params=117068, master_seed=54)],
                            variant="tcn_se")
        self.assertIsNone(rows[0]["params"])


if __name__ == "__main__":
    unittest.main()
