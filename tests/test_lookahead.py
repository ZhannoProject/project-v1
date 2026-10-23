"""
Tests for bounded lookahead: models/lookahead.py and the tcn_se_la{N} ladder.

Run: python -m unittest discover -s tests -p 'test_lookahead.py'

The point of these is that the knob cannot quietly lie. A lookahead budget is
a latency claim -- L samples is L/fs of output delay -- so the two things that
must hold are that the model reads no further ahead than it was licensed to
(or the claim is unsafe) and that the two ends of the ladder are the models
that already exist (or the curve is not measuring what it says it is).
"""

import contextlib
import inspect
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.TCN_backbone import CausalConv1d, CausalSE, CausalTCNBackbone, ResidualBlock
from models.lookahead import (LOOKAHEAD_LADDER, allocate, architectural, capacity,
                              forward_receptive_field, ms)
from models.variants import (LOOKAHEAD_VARIANTS, VARIANT_REGISTRY, TCNOnly, TCNWithSE,
                             base_variant, build_model, pinned_kwargs, resolved_arch,
                             variant_is_causal)

DILATIONS = [1, 2, 4, 8, 16]


@contextlib.contextmanager
def fake_runs(*taken):
    """
    A runs/ tree holding exactly the directories named, and nothing else.

    ``train_spec`` asks the filesystem whether a rung has been trained, so a
    test that uses the real tree is really testing what experiments happen to
    be on this machine today -- it passed until someone trained la124.
    """
    from gui import lookahead
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for save_dir in taken:
            (root / save_dir).mkdir(parents=True, exist_ok=True)
            (root / save_dir / "best.pt").write_bytes(b"")
        with mock.patch.object(lookahead, "PROJECT_ROOT", root):
            yield root


class TestAllocate(unittest.TestCase):
    """The budget is spent exactly, and never beyond a site's capacity."""

    def test_none_passes_through(self):
        # None is the legacy centred/global setting, not a budget.
        self.assertIsNone(allocate([2, 4, 8], None))

    def test_zero_gives_every_site_nothing(self):
        self.assertEqual(allocate([2, 4, 8], 0), [0, 0, 0])

    def test_the_budget_is_spent_to_the_sample(self):
        caps = [2, 2, 4, 4, 8, 8, 16, 16, 32, 32]
        for budget in range(0, sum(caps) + 1):
            with self.subTest(budget=budget):
                out = allocate(caps, budget)
                self.assertEqual(sum(out), budget)
                for o, c in zip(out, caps):
                    self.assertGreaterEqual(o, 0)
                    self.assertLessEqual(o, c)

    def test_a_request_over_capacity_is_clamped_not_refused(self):
        caps = [2, 4, 8]
        self.assertEqual(sum(allocate(caps, 1000)), sum(caps))

    def test_it_is_proportional_not_front_loaded(self):
        # Spread, so that the budget's extent is the only thing that varies
        # across the ladder and not also the depth it is spent at.
        caps = [2, 32]
        self.assertEqual(allocate(caps, 34), [2, 32])
        out = allocate(caps, 17)
        self.assertGreater(out[1], out[0])

    def test_bad_input_is_refused(self):
        with self.assertRaises(ValueError):
            allocate([2, 4], -1)
        with self.assertRaises(TypeError):
            allocate([2, 4], 3.5)
        with self.assertRaises(TypeError):
            allocate([2, 4], True)

    def test_capacity_matches_the_stack_it_describes(self):
        # Two convolutions per block, each able to move (k-1)*d taps right.
        self.assertEqual(capacity(DILATIONS), 2 * sum(2 * d for d in DILATIONS))
        self.assertEqual(capacity(DILATIONS), 124)
        # Plus one SE gate per block at the same per-block capacity.
        self.assertEqual(capacity(DILATIONS, se=True), 186)
        self.assertEqual(LOOKAHEAD_LADDER[-1], capacity(DILATIONS, se=True))


class TestConvAndGate(unittest.TestCase):
    """The two sites, on their own."""

    def test_conv_lookahead_is_the_right_hand_padding(self):
        c = CausalConv1d(1, 1, 3, dilation=4, causal=False, lookahead=5)
        self.assertEqual(c.pad_right, 5)
        self.assertEqual(c.pad_left, (3 - 1) * 4 - 5)
        self.assertFalse(c.causal)

    def test_conv_lookahead_cannot_exceed_its_taps(self):
        c = CausalConv1d(1, 1, 3, dilation=2, causal=False, lookahead=99)
        self.assertEqual(c.pad_right, (3 - 1) * 2)
        self.assertEqual(c.pad_left, 0)

    def test_a_zero_budget_conv_is_the_causal_conv(self):
        a = CausalConv1d(1, 1, 3, dilation=4, causal=False, lookahead=0)
        b = CausalConv1d(1, 1, 3, dilation=4)
        self.assertEqual((a.pad_left, a.pad_right), (b.pad_left, b.pad_right))
        self.assertTrue(a.causal)

    def test_legacy_conv_settings_are_untouched(self):
        causal = CausalConv1d(1, 1, 3, dilation=4)
        centred = CausalConv1d(1, 1, 3, dilation=4, causal=False)
        self.assertEqual((causal.pad_left, causal.pad_right), (8, 0))
        self.assertEqual((centred.pad_left, centred.pad_right), (4, 4))

    def test_the_gate_reads_the_running_mean_l_samples_later(self):
        se = CausalSE(4, reduction=2, causal=False, lookahead=3).eval()
        x = torch.randn(1, 4, 16)
        got = se.squeeze(x)
        counts = torch.arange(1, 17, dtype=x.dtype)
        running = x.cumsum(dim=2) / counts.view(1, 1, -1)
        for t in range(16):
            torch.testing.assert_close(got[:, :, t], running[:, :, min(t + 3, 15)])

    def test_the_gate_clamps_at_the_end_of_the_epoch(self):
        # No future left there: the last positions all read the final running
        # mean rather than wrapping or changing rule.
        se = CausalSE(4, reduction=2, causal=False, lookahead=5).eval()
        x = torch.randn(1, 4, 8)
        got = se.squeeze(x)
        torch.testing.assert_close(got[:, :, 3], got[:, :, 7])

    def test_zero_lookahead_gate_is_the_causal_gate(self):
        torch.manual_seed(0)
        a = CausalSE(8, reduction=2, causal=False, lookahead=0).eval()
        torch.manual_seed(0)
        b = CausalSE(8, reduction=2).eval()
        x = torch.randn(2, 8, 32)
        torch.testing.assert_close(a.squeeze(x), b.squeeze(x))
        self.assertTrue(b.causal)

    def test_the_global_pool_is_still_the_global_pool(self):
        se = CausalSE(4, reduction=2, causal=False).eval()
        x = torch.randn(1, 4, 16)
        torch.testing.assert_close(se.squeeze(x), x.mean(dim=2, keepdim=True).expand_as(x))
        self.assertTrue(se.global_pool)
        self.assertFalse(se.causal)


class TestLicensedAgainstMeasured(unittest.TestCase):
    """
    The claim a lookahead model makes, against what it does.

    ``architectural()`` is what the graph is licensed to read and is what sets
    the output delay; ``forward_receptive_field()`` is what the weights
    actually reach. The first must bound the second, always -- a model reading
    further than licensed would make the latency figure unsafe.
    """

    def test_the_convolution_stack_realises_its_budget_exactly(self):
        # Nothing on this path can zero a dependence: PReLU has a slope either
        # side and BatchNorm is affine in eval mode. So licensed == measured,
        # which is what makes the probe a boundary test rather than an
        # estimate -- and it is how the 62 samples of tcn_bi are known.
        for budget in (0, 1, 7, 31, 62, 124):
            with self.subTest(budget=budget):
                b = CausalTCNBackbone(1, 8, DILATIONS, causal=False, lookahead=budget)
                self.assertEqual(architectural(b), budget)
                self.assertEqual(forward_receptive_field(b), budget)

    def test_tcn_bi_measures_the_number_the_switch_probe_recovers(self):
        # 62 samples = 242 ms at 256 Hz, the figure README quotes from the
        # artifact-switch probe. Two independent instruments, one number.
        model = build_model("tcn_bi", hidden_channels=16)
        self.assertEqual(forward_receptive_field(model), 62)
        self.assertEqual(architectural(model), 62)

    def test_an_se_model_never_reads_further_than_licensed(self):
        for variant in LOOKAHEAD_VARIANTS:
            with self.subTest(variant=variant):
                model = build_model(variant, hidden_channels=64)
                licensed, measured = architectural(model), forward_receptive_field(model)
                self.assertEqual(licensed, pinned_kwargs(variant)["lookahead"])
                self.assertIsNotNone(measured)
                self.assertLessEqual(measured, licensed)

    def test_a_dead_relu_gate_is_why_measured_can_fall_short(self):
        # The gate is sigmoid(W2 @ relu(W1 @ s)): where W1 @ s is negative the
        # gate stops depending on s, so a licensed peek contributes nothing.
        # With channels // reduction == 1 there is a single hidden unit and
        # about half of all inits are dead, which is the clearest possible
        # demonstration that the shortfall is this and not a probe artefact.
        reaches = set()
        for seed in range(8):
            torch.manual_seed(seed)
            block = ResidualBlock(1, 8, 3, dilation=4, se_reduction=8, causal=False,
                                  conv_lookahead=(0, 0), se_lookahead=8)
            self.assertEqual(architectural(block), 8)
            reaches.add(forward_receptive_field(block))
        # Some inits realise the full 8, some realise nothing at all.
        self.assertIn(8, reaches)
        self.assertIn(0, reaches)

    def test_a_block_adds_its_sites_up(self):
        torch.manual_seed(0)
        block = ResidualBlock(1, 8, 3, dilation=4, se_reduction=8, causal=False,
                              conv_lookahead=(8, 8), se_lookahead=8)
        self.assertEqual(architectural(block), 24)
        self.assertEqual(forward_receptive_field(block), 24)

    def test_an_unbounded_reach_is_reported_as_unbounded(self):
        # A global-pool gate reads the whole epoch; no finite number says that,
        # and None is a different answer from a large one.
        for variant in ("tcn_se_bi", "tcn_mla_bi", "tcn_mla_full_bi"):
            with self.subTest(variant=variant):
                model = build_model(variant, hidden_channels=16)
                self.assertIsNone(architectural(model))
                self.assertIsNone(forward_receptive_field(model))

    def test_causal_models_read_nothing_ahead(self):
        for variant in ("tcn", "tcn_se", "tcn_mha", "tcn_mla", "tcn_mla_r8"):
            with self.subTest(variant=variant):
                model = build_model(variant, hidden_channels=16)
                self.assertEqual(architectural(model), 0)
                self.assertEqual(forward_receptive_field(model), 0)

    def test_a_nonzero_tolerance_reads_short(self):
        # Documented rather than worked around: beyond the reach the outputs
        # are bit-identical, so 0 is the right threshold and anything larger
        # cuts off a chain of small sensitivities that is really there.
        model = build_model("tcn_se_la186", hidden_channels=64)
        self.assertLessEqual(forward_receptive_field(model, tol=1e-6),
                             forward_receptive_field(model, tol=0.0))

    def test_training_mode_is_not_measured_in(self):
        # BatchNorm's batch statistics span time, which couples every position
        # to every other; the probe must not report that as a reach.
        model = build_model("tcn_se", hidden_channels=16)
        model.train()
        self.assertEqual(forward_receptive_field(model), 0)
        self.assertTrue(model.training, "the probe must leave the mode as it found it")


class TestTheLadder(unittest.TestCase):
    """The registry entries, and the two ends of the ladder."""

    def test_every_rung_is_registered_and_pins_its_budget(self):
        self.assertEqual(len(LOOKAHEAD_VARIANTS), len(LOOKAHEAD_LADDER))
        for budget in LOOKAHEAD_LADDER:
            name = f"tcn_se_la{budget}"
            self.assertIn(name, VARIANT_REGISTRY)
            self.assertEqual(pinned_kwargs(name), {"causal": False, "lookahead": budget})

    def test_a_rung_is_not_causal(self):
        # Fewer future samples than a teacher, but not none -- and everything
        # downstream (the exporter, the streaming paths) needs that answer.
        for name in LOOKAHEAD_VARIANTS:
            with self.subTest(name=name):
                self.assertFalse(variant_is_causal(name))

    def test_a_rung_cannot_be_talked_out_of_its_budget(self):
        with self.assertRaises(ValueError):
            build_model("tcn_se_la32", lookahead=8)
        with self.assertRaises(ValueError):
            build_model("tcn_se_la32", causal=True)

    def test_a_rung_shares_the_weights_of_tcn_se(self):
        # The ladder changes where padding goes and when a mean is read, not
        # what is learned, so every rung loads into every other and into the
        # causal model -- which is what lets one teach another.
        causal = build_model("tcn_se", hidden_channels=32)
        for name in ("tcn_se_la8", "tcn_se_la186", "tcn_se_bi"):
            with self.subTest(name=name):
                rung = build_model(name, hidden_channels=32)
                self.assertEqual(sum(p.numel() for p in rung.parameters()),
                                 sum(p.numel() for p in causal.parameters()))
                rung.load_state_dict(causal.state_dict())

    def test_a_rung_resolves_to_tcn_se(self):
        for name in LOOKAHEAD_VARIANTS:
            with self.subTest(name=name):
                self.assertEqual(base_variant(name), "tcn_se")

    def test_a_run_records_the_budget_it_trained(self):
        arch = resolved_arch(build_model("tcn_se_la32"))
        self.assertEqual(arch["lookahead"], 32)
        self.assertEqual(arch["variant"], "tcn_se_la32")
        self.assertIs(arch["causal"], False)

    def test_the_bottom_of_the_ladder_is_tcn_se_itself(self):
        # Not an la variant with L=0: that would be a second name for a model
        # that already has one, and two names for one thing is how a ladder
        # ends up with two slightly different bottom rungs.
        self.assertNotIn("tcn_se_la0", VARIANT_REGISTRY)
        self.assertEqual(forward_receptive_field(build_model("tcn_se", hidden_channels=16)), 0)

    def test_a_rung_runs_and_keeps_its_shape(self):
        x = torch.randn(2, 1, 512)
        for name in ("tcn_se_la1", "tcn_se_la32", "tcn_se_la186"):
            with self.subTest(name=name):
                model = build_model(name, hidden_channels=16).eval()
                with torch.no_grad():
                    self.assertEqual(model(x).shape, x.shape)

    def test_lookahead_contradicting_causal_is_refused(self):
        from models.variants import TCNWithSE
        with self.assertRaises(ValueError):
            TCNWithSE(causal=True, lookahead=8)

    def test_latency_is_the_budget_over_the_sample_rate(self):
        self.assertAlmostEqual(ms(32), 125.0, places=6)      # the deploy tab's hop 32
        self.assertAlmostEqual(ms(62), 242.1875, places=4)   # tcn_bi's 242 ms
        self.assertIsNone(ms(None))


class TestStreamingRefusesToLookAhead(unittest.TestCase):
    """A model that needs the future cannot be streamed, and says so."""

    def test_the_incremental_path_refuses_a_rung(self):
        from models.incremental import IncrementalModel
        with self.assertRaises(ValueError):
            IncrementalModel(build_model("tcn_se_la32", hidden_channels=16).eval())

    def test_the_step_graph_refuses_a_rung(self):
        from models.step_graph import StepGraph
        with self.assertRaises(ValueError):
            StepGraph(build_model("tcn_se_la32", hidden_channels=16).eval())

    def test_the_causal_model_is_still_streamable(self):
        from models.incremental import IncrementalModel
        IncrementalModel(build_model("tcn_se", hidden_channels=16).eval())


class TestTheTabsNumbers(unittest.TestCase):
    """gui/lookahead.py: what the Lookahead tab is handed."""

    def test_a_rung_averages_only_runs_of_its_own_width(self):
        # The reason this exists: runs/tcn_se_both_s67 held a 48-channel run
        # beside two 64-channel ones, and averaging the three made a spread of
        # two seeds and a width experiment. An excluded run is reported, not
        # dropped -- it exists, and the reader should see why it is not in.
        from gui.lookahead import _runs_on
        by_rung = {0: [
            {"name": "a", "status": "complete", "hidden_channels": 64, "master_seed": 42,
             "rrmse_t": 0.70, "rrmse_s": 0.65, "acc": 0.71},
            {"name": "b", "status": "complete", "hidden_channels": 64, "master_seed": 54,
             "rrmse_t": 0.72, "rrmse_s": 0.66, "acc": 0.70},
            {"name": "odd", "status": "complete", "hidden_channels": 48, "master_seed": 67,
             "rrmse_t": 0.90, "rrmse_s": 0.80, "acc": 0.60},
        ]}
        out = _runs_on(0, 64, by_rung)
        self.assertEqual(out["n_runs"], 2)
        self.assertEqual(out["seeds"], [42, 54])
        self.assertAlmostEqual(out["rrmse_t"]["mean"], 0.71, places=9)
        self.assertEqual([m["name"] for m in out["mismatched"]], ["odd"])
        self.assertIn("48 channels, not 64", out["mismatched"][0]["why"])

    def test_a_stale_run_is_on_no_curve_at_all(self):
        # Its metrics describe weights that were overwritten, so there is
        # nothing for a point to stand on -- not even as an excluded row.
        from gui.lookahead import _runs_on
        by_rung = {0: [{"name": "s", "status": "stale", "hidden_channels": 64,
                        "master_seed": 42, "rrmse_t": 0.5, "rrmse_s": 0.5, "acc": 0.9}]}
        out = _runs_on(0, 64, by_rung)
        self.assertEqual(out["n_runs"], 0)
        self.assertEqual(out["mismatched"], [])

    def test_one_seed_reports_no_spread_rather_than_zero(self):
        from gui.lookahead import _seed_stats
        self.assertIsNone(_seed_stats([0.7])["sd"])
        self.assertEqual(_seed_stats([0.7])["n"], 1)
        self.assertEqual(_seed_stats([])["n"], 0)

    def test_the_ladder_is_measured_end_to_end(self):
        from gui.lookahead import measure
        out = measure({"hidden_channels": 16})
        rungs = [r["lookahead"] for r in out["rows"]]
        self.assertEqual(rungs[0], 0)                      # tcn_se anchors the bottom
        self.assertIsNone(rungs[-1])                       # tcn_se_bi the top
        self.assertEqual(rungs[1:-1], list(LOOKAHEAD_LADDER))
        for row in out["rows"]:
            with self.subTest(rung=row["lookahead"]):
                if row["unbounded"]:
                    self.assertIsNone(row["licensed"])
                    continue
                self.assertEqual(row["licensed"], row["lookahead"])
                self.assertAlmostEqual(row["latency_ms"], ms(row["lookahead"]), places=6)
                self.assertLessEqual(row["measured"], row["licensed"])

    def test_an_unregistered_rung_is_measurable_but_not_trainable(self):
        from gui.lookahead import measure
        row = next(r for r in measure({"hidden_channels": 16, "budgets": [33]})["rows"]
                   if r["lookahead"] == 33)
        self.assertEqual(row["licensed"], 33)
        self.assertFalse(row["trainable"])
        self.assertIn("LOOKAHEAD_LADDER", row["note"])

    def test_a_budget_over_capacity_is_refused_by_the_api(self):
        from gui.lookahead import measure
        with self.assertRaises(ValueError):
            measure({"budgets": [10_000]})
        with self.assertRaises(ValueError):
            measure({"budgets": ["32"]})
        with self.assertRaises(ValueError):
            measure({"hidden_channels": 0})

    def test_training_a_rung_goes_to_its_own_directory(self):
        from gui.lookahead import train_spec
        s = train_spec({"lookahead": 32, "master_seed": 67})
        self.assertEqual(s["variant"], "tcn_se_la32")
        self.assertEqual(s["save_dir"], "runs/tcn_se_la32_both_s67")
        self.assertIn("--variant tcn_se_la32", s["extra"])
        self.assertIn("--master_seed 67", s["extra"])
        self.assertEqual(s["config"], "tcn_se_both.yaml")

    def test_the_default_seed_keeps_the_bare_name(self):
        # Same rule as the trainer's: seed 42 keeps the name every existing
        # directory already has, so the two cannot disagree about a path.
        from gui.lookahead import train_spec
        self.assertEqual(train_spec({"lookahead": 32})["save_dir"], "runs/tcn_se_la32_both")

    def test_the_causal_rung_trains_tcn_se_itself(self):
        from gui.lookahead import train_spec
        s = train_spec({"lookahead": 0, "master_seed": 54})
        self.assertEqual(s["variant"], "tcn_se")
        self.assertEqual(s["save_dir"], "runs/tcn_se_both_s54")

    def test_an_off_ladder_rung_cannot_be_trained(self):
        # A checkpoint needs a name that states its budget, or it misdescribes
        # itself the moment anyone reads the directory.
        from gui.lookahead import train_spec
        with self.assertRaises(ValueError) as ctx:
            train_spec({"lookahead": 33})
        self.assertIn("tcn_se_la33", str(ctx.exception))

    def test_bad_train_input_is_refused(self):
        from gui.lookahead import train_spec
        for spec in ({}, {"lookahead": "32"}, {"lookahead": 32, "artifact": "ECG"}):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                train_spec(spec)

    def test_a_rung_can_be_planned_at_several_seeds(self):
        """
        A rung trained once is a point with no spread, so the unit of work is
        a rung at three seeds. Each one has to be the command it would have
        been on its own, or the batch is a second entry point.
        """
        from gui.lookahead import train_plan, train_spec
        plan = train_plan({"lookahead": 124, "master_seeds": [67, 42, 54, 42]})
        self.assertEqual(plan["seeds"], [42, 54, 67], "sorted, and 42 asked for twice is once")
        for run in plan["runs"]:
            alone = train_spec({"lookahead": 124, "master_seed": run["master_seed"]})
            self.assertEqual(run["extra"], alone["extra"])
        self.assertEqual(len({r["save_dir"] for r in plan["runs"]}), 3,
                         "three seeds must not land in one directory")

    def test_one_seed_still_works_the_old_way(self):
        from gui.lookahead import train_plan
        plan = train_plan({"lookahead": 124, "master_seed": 54})
        self.assertEqual(plan["seeds"], [54])
        self.assertEqual(plan["runs"][0]["save_dir"], "runs/tcn_se_la124_both_s54")

    def test_the_plan_says_which_seeds_are_already_taken(self):
        """
        So the tab can untick those and train the rest, instead of the whole
        request being refused because one of three has been done before.
        """
        from gui.lookahead import train_plan
        with fake_runs("runs/tcn_se_la32_both", "runs/tcn_se_la32_both_s67"):
            plan = train_plan({"lookahead": 32, "master_seeds": [42, 54, 67]})
            self.assertEqual(plan["taken"],
                             ["runs/tcn_se_la32_both", "runs/tcn_se_la32_both_s67"])
            self.assertEqual(plan["fresh"], [54])
            for run in plan["runs"]:
                self.assertEqual(run["exists"], run["save_dir"] in plan["taken"])

    def test_overwrite_lands_only_where_there_is_something_to_overwrite(self):
        """
        A blanket --overwrite would also switch off the trainer's guard for
        the seeds that did not need it, which is where a mistyped seed
        quietly lands on top of a finished run.

        Run against a made-up runs tree, not the real one: which rungs are
        trained is whatever the person has been doing this week, and a test
        that reads it passes or fails on that.
        """
        from gui.lookahead import train_plan
        with fake_runs("runs/tcn_se_la32_both"):
            plan = train_plan({"lookahead": 32, "master_seeds": [42, 54], "overwrite": True})
            self.assertEqual([r["exists"] for r in plan["runs"]], [True, False])
            for run in plan["runs"]:
                with self.subTest(seed=run["master_seed"]):
                    self.assertEqual("--overwrite" in run["extra"], run["exists"])
            asked_not = train_plan({"lookahead": 32, "master_seeds": [42]})
            self.assertNotIn("--overwrite", asked_not["runs"][0]["extra"])

    def test_bad_seed_lists_are_refused(self):
        from gui.lookahead import train_plan
        for seeds in ([], ["42"], [42, None], [4.0], [True]):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                train_plan({"lookahead": 124, "master_seeds": seeds})

    def test_the_config_the_ladder_trains_from_exists(self):
        from gui.lookahead import LADDER_CONFIG
        from gui.config_io import config_path
        self.assertTrue(config_path(LADDER_CONFIG).is_file())


class TestRunsTabTellsTheRungsApart(unittest.TestCase):
    """
    A rung is not a teacher, and the Runs tab must not call it one.

    ``model_config.causal is False`` is true of a bounded rung and of a *_bi
    twin alike, so the flag alone badges every rung of the ladder as a
    non-deployable teacher. That is a run labelled as something it is not,
    which is the failure mode this project keeps building guards against --
    so the row carries all three answers and the tab reads the right one.
    """

    @classmethod
    def setUpClass(cls):
        import shutil
        import tempfile
        from unittest import mock

        from gui import config_io

        cls.tmp = Path(tempfile.mkdtemp(prefix="la_runs_"))
        for name, variant in (("tcn_se_la32_both", "tcn_se_la32"),
                              ("tcn_se_both", "tcn_se"),
                              ("tcn_se_bi_both", "tcn_se_bi")):
            d = cls.tmp / name
            d.mkdir()
            model = build_model(variant, hidden_channels=8)
            torch.save({"epoch": 2, "model_config": model.model_config,
                        "model_state": model.state_dict(),
                        "run_config": {"master_seed": 42}}, d / "best.pt")
            shutil.copy(d / "best.pt", d / "last.pt")
            (d / "history.json").write_text(
                '{"train_loss": [1, 0.5], "val_loss": [0.9, 0.4], "rrmse_t": [0.9, 0.7],'
                ' "rrmse_s": [0.9, 0.6], "acc": [0.1, 0.7],'
                ' "config": {"artifact": "BOTH", "master_seed": 42}}')
        cls.patch = mock.patch.object(config_io, "RUNS_DIR", cls.tmp)
        cls.patch.start()
        config_io._CKPT_CACHE.clear()
        cls.rows = {r["name"]: r for r in config_io.list_runs()}

    @classmethod
    def tearDownClass(cls):
        import shutil
        cls.patch.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)
        from gui import config_io
        config_io._CKPT_CACHE.clear()

    def test_a_rung_is_marked_as_a_rung(self):
        row = self.rows["tcn_se_la32_both"]
        self.assertEqual(row["lookahead"], 32)
        self.assertEqual(row["variant"], "tcn_se_la32")

    def test_a_rung_is_not_called_bidirectional(self):
        # The whole point: it reads 32 samples ahead, not the epoch.
        self.assertFalse(self.rows["tcn_se_la32_both"]["bidirectional"])

    def test_a_rung_is_still_not_causal(self):
        # It reads samples that have not arrived, so every tool that refuses a
        # teacher must refuse it too -- for a different stated reason.
        self.assertFalse(self.rows["tcn_se_la32_both"]["causal"])

    def test_a_teacher_is_still_a_teacher(self):
        row = self.rows["tcn_se_bi_both"]
        self.assertTrue(row["bidirectional"])
        self.assertIsNone(row["lookahead"])
        self.assertFalse(row["causal"])

    def test_the_causal_baseline_is_unmarked(self):
        row = self.rows["tcn_se_both"]
        self.assertFalse(row["bidirectional"])
        self.assertIsNone(row["lookahead"])
        self.assertTrue(row["causal"])

    def test_a_rung_groups_as_its_own_experiment(self):
        # "by kind" groups on variant x artifact, and a rung is a variant, so
        # it never shares a group -- or an average -- with the causal model.
        self.assertEqual(self.rows["tcn_se_la32_both"]["kind"], "tcn_se_la32 · BOTH")
        self.assertNotEqual(self.rows["tcn_se_la32_both"]["kind"],
                            self.rows["tcn_se_both"]["kind"])

    def test_the_ladder_tab_reads_the_rung_off_the_row(self):
        from gui.lookahead import _rung_of
        self.assertEqual(_rung_of(self.rows["tcn_se_la32_both"]), 32)
        self.assertEqual(_rung_of(self.rows["tcn_se_both"]), 0)
        self.assertIsNone(_rung_of(self.rows["tcn_se_bi_both"]))

    def test_the_exporter_refuses_a_rung_and_says_why(self):
        from unittest import mock

        from gui import pi
        with mock.patch.object(pi.config_io, "list_runs", return_value=list(self.rows.values())), \
             mock.patch.object(pi, "RUNS_DIR", self.tmp):
            models = {m["name"]: m for m in pi.list_models()}
            rung = models["tcn_se_la32_both"]
            self.assertFalse(rung["exportable"])
            self.assertFalse(rung["deployable"])
            self.assertEqual(rung["lookahead"], 32)
            self.assertTrue(models["tcn_se_both"]["exportable"])
            self.assertFalse(models["tcn_se_bi_both"]["exportable"])

            with self.assertRaises(ValueError) as ctx:
                pi._export_step({"model": "tcn_se_la32_both"})
            self.assertIn("32 samples", str(ctx.exception))
            self.assertIn("125.0 ms", str(ctx.exception))

    def test_a_model_row_with_neither_flag_is_exportable(self):
        # Rows built elsewhere (and the tests that mock them) carry only
        # ``bidirectional``; absence of a lookahead must not read as one.
        from unittest import mock

        from gui import pi
        with mock.patch.object(pi.config_io, "list_runs",
                               return_value=[{"name": "legacy", "bidirectional": False}]), \
             mock.patch.object(pi, "RUNS_DIR", self.tmp):
            self.assertTrue({m["name"]: m for m in pi.list_models()}["legacy"]["exportable"])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class WindowIsSlidNotWidenedTests(unittest.TestCase):
    """
    What the ladder actually buys. The tab reads as a price list -- pay
    latency, get accuracy -- and that reading is only safe while the rung is
    still looking at the past as well.
    """

    @staticmethod
    def window(name):
        from models.TCN_backbone import CausalConv1d
        model = build_model(name)
        convs = [m for m in model.modules() if isinstance(m, CausalConv1d)]
        return sum(c.pad_left for c in convs), sum(c.pad_right for c in convs)

    def test_a_tap_given_to_the_future_is_taken_from_the_past(self):
        """
        pad_left + pad_right is fixed by the kernel, so the window is a fixed
        size and the budget only moves it. This is the whole reason the price
        curve turns around at the top rung instead of flattening.
        """
        span = capacity([1, 2, 4, 8, 16])
        for name in ["tcn_se"] + [f"tcn_se_la{n}" for n in LOOKAHEAD_LADDER] + ["tcn_se_bi"]:
            with self.subTest(variant=name):
                back, ahead = self.window(name)
                self.assertEqual(back + ahead, span,
                                 "the convolution window changed size, so the budget is not "
                                 "purely a shift and this ladder means something else")

    def test_the_full_budget_leaves_no_past_at_all(self):
        """
        186 is the mirror of the causal baseline, not a better-informed
        version of it: every convolution padded entirely on the right.
        """
        from models.lookahead import backward_taps
        self.assertEqual(self.window("tcn_se_la186"), (0, 124))
        self.assertEqual(backward_taps(build_model("tcn_se_la186")), 0)
        self.assertEqual(self.window("tcn_se"), (124, 0), "and the baseline is its mirror")

    def test_the_forward_only_model_is_the_causal_one_reading_backwards(self):
        """
        What the top rung actually *is*. A causal layer computes
        ``w0·x[t-2d] + w1·x[t-d] + w2·x[t]``; move all the padding right and it
        computes ``w0·x[t] + w1·x[t+d] + w2·x[t+2d]``. That is the same filter
        applied to the reversed signal with its kernel read backwards -- and
        since the kernel is learned, reading it backwards costs nothing. So
        the anti-causal model is not a degenerate model: it is this model
        class run on the recording backwards, which is well posed, and
        undeployable online for the obvious reason rather than a subtle one.

        Checked to float precision, so it is an identity rather than a
        resemblance.
        """
        x = torch.randn(1, 1, 512)
        flip = lambda t: torch.flip(t, [-1])
        causal = TCNOnly().eval()
        anti = TCNOnly(causal=False, lookahead=capacity([1, 2, 4, 8, 16])).eval()
        anti.load_state_dict({k: (torch.flip(v, [-1]) if v.dim() == 3 and v.shape[-1] > 1
                                  else v.clone())
                              for k, v in causal.state_dict().items()})
        with torch.no_grad():
            got, want = anti(x), flip(causal(flip(x)))
        self.assertLess((got - want).abs().max().item(), 1e-5 * want.abs().max().item() + 1e-6)

    def test_the_se_gate_is_the_one_part_that_does_not_reverse(self):
        """
        Its squeeze is a running mean from sample 0, so it still accumulates
        forwards however the convolutions are padded. That is why the mirror
        rung is not simply the baseline's score reflected: it has the past's
        statistics and the future's detail at once.
        """
        x = torch.randn(1, 1, 512)
        flip = lambda t: torch.flip(t, [-1])
        causal = TCNWithSE().eval()
        anti = TCNWithSE(causal=False, lookahead=capacity([1, 2, 4, 8, 16], se=True)).eval()
        anti.load_state_dict({k: (torch.flip(v, [-1]) if v.dim() == 3 and v.shape[-1] > 1
                                  else v.clone())
                              for k, v in causal.state_dict().items()})
        with torch.no_grad():
            got, want = anti(x), flip(causal(flip(x)))
        self.assertGreater((got - want).abs().max().item(), 1e-5,
                           "the SE gate reversed too, so the mirror is exact after all")

    def test_the_ceiling_is_a_centred_window_not_an_unbounded_one(self):
        """
        ``tcn_se_bi`` is called the unbounded ceiling because its SE gate
        pools globally, but its convolutions see 62 either side -- the same
        125 samples as every rung, centred. Worth stating, because "unbounded"
        invites reading its score as what infinite context is worth.
        """
        self.assertEqual(self.window("tcn_se_bi"), (62, 62))

    def test_backward_taps_counts_only_the_convolutions(self):
        """
        The SE squeeze is a running mean from 0, so it reaches the whole past
        at every rung; counting it would make every rung look like it kept
        its past.
        """
        from models.lookahead import backward_taps
        self.assertEqual(backward_taps(build_model("tcn_se_la186")), 0)
        # Two convolutions per block, so a one-level stack holds 4 taps, all
        # of them behind t while it is causal.
        self.assertEqual(backward_taps(build_model("tcn_rf5")), capacity([1]))

    def test_the_budget_is_a_share_of_the_licence_and_the_top_rungs_are_thirds(self):
        """
        "la186" on a 124-tap window is an arithmetic that does not work out
        until the denominator is named: capacity is 124 taps the convolutions
        can move plus 62 the gates can peek. Against that the ladder's top
        three rungs are exactly the thirds, which is why they are the rungs.
        """
        from gui.lookahead import share_label
        total = capacity([1, 2, 4, 8, 16], se=True)
        self.assertEqual(share_label(62, total), "1/3")
        self.assertEqual(share_label(124, total), "2/3")
        self.assertEqual(share_label(186, total), "1/1", "reads as the last of the thirds")
        self.assertEqual(share_label(16, total), "9%")
        self.assertEqual(share_label(1, total), "0.5%")
        self.assertEqual(share_label(0, total), "", "the causal end is not a share of anything")

    def test_the_names_on_disk_are_left_alone(self):
        """
        27 run directories and the architecture recorded inside each of them
        name the variant. The number is also the latency the deployment
        carries, in samples, which is the contract the ladder is priced in —
        so the share is shown beside the name, not instead of it.
        """
        for budget in LOOKAHEAD_LADDER:
            with self.subTest(budget=budget):
                self.assertIn(f"tcn_se_la{budget}", VARIANT_REGISTRY)

    def test_the_rung_row_reports_what_the_budget_cost(self):
        from gui.lookahead import measure
        rows = {r["variant"]: r for r in measure({"artifact": "BOTH"})["rows"]}
        self.assertEqual((rows["tcn_se"]["past"], rows["tcn_se"]["ahead"]), (124, 0))
        self.assertEqual((rows["tcn_se_la186"]["past"], rows["tcn_se_la186"]["ahead"]), (0, 124))
        self.assertEqual(rows["tcn_se_la124"]["share"], "2/3")
        for r in rows.values():
            if r["past"] is not None:
                self.assertEqual(r["past"] + r["ahead"], r["window"])
        self.assertTrue(rows["tcn_se_la186"]["one_sided"])
        self.assertFalse(rows["tcn_se_la124"]["one_sided"])
        self.assertFalse(rows["tcn_se"]["one_sided"])


class RungMembershipTests(unittest.TestCase):
    """
    Which runs a rung averages. The causal rung is the anchor every Delta in
    the table is measured from, so a run that does not belong there moves the
    whole ladder.
    """

    @staticmethod
    def a_run(name, **kw):
        row = {"name": name, "status": "complete", "master_seed": 42,
               "hidden_channels": 64, "loss_fn": "mse", "kd": False,
               "rrmse_t": 0.71, "rrmse_s": 0.65, "acc": 0.71}
        row.update(kw)
        return row

    def runs_on(self, rows):
        from gui.lookahead import _runs_on
        return _runs_on(0, 64, {0: rows})

    def test_a_distilled_run_is_not_a_seed_repeat(self):
        """
        ``runs/tcn_se_both_kd_s67`` sat in the causal anchor: it moved it from
        0.7104 to 0.7118 and widened the spread from 0.0020 to 0.0033, so
        every "what a millisecond is worth" Delta was measured from a point
        that was part KD.
        """
        out = self.runs_on([self.a_run("plain", rrmse_t=0.70),
                            self.a_run("distilled", kd=True, master_seed=67, rrmse_t=0.76)])
        self.assertEqual(out["n_runs"], 1)
        self.assertAlmostEqual(out["rrmse_t"]["mean"], 0.70)
        self.assertIn("distilled from a teacher", out["mismatched"][0]["why"])

    def test_another_width_is_still_excluded(self):
        out = self.runs_on([self.a_run("wide"),
                            self.a_run("narrow", hidden_channels=48, master_seed=67)])
        self.assertEqual(out["runs"][0]["name"], "wide")
        self.assertIn("48 channels, not 64", out["mismatched"][0]["why"])

    def test_another_loss_is_excluded(self):
        out = self.runs_on([self.a_run("mse"),
                            self.a_run("l1", loss_fn="l1_combined", master_seed=54)])
        self.assertEqual(out["runs"][0]["name"], "mse")
        self.assertIn("l1_combined", out["mismatched"][0]["why"])

    def test_the_rows_carry_what_the_rule_asks_about(self):
        """
        The rule reads ``kd`` off a row, and the rows are a projection of the
        Runs tab's own. It was the projection dropping that key -- not the
        rule -- that let a distilled run into the anchor.
        """
        from gui.lookahead import ladder_runs
        for row in ladder_runs("BOTH"):
            self.assertIn("kd", row)

    def test_both_ladders_ask_the_same_question(self):
        """
        The receptive-field sweep has the same membership problem, and two
        copies of this rule would answer it two ways on the same run.
        """
        import gui.rfsweep as rfsweep
        from gui.lookahead import why_excluded
        src = inspect.getsource(rfsweep.results)
        self.assertIn("why_excluded", src)
        self.assertIs(sys.modules["gui.lookahead"].why_excluded, why_excluded)


