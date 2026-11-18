"""Scientific contracts: alignment, independence, raw pipeline parity and provenance."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from experiment.config import resolve_config
from experiment.methodology import aligned, context, latency, parity, target_interval, validate_model
from models.variants import build_model, build_model_from_config


class MethodologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)

    def model(self, variant="tcn_se", tcn_norm="none", **kw):
        return build_model(variant, hidden_channels=8, dropout=0, tcn_norm=tcn_norm,
                           se_window=16, attention_window=16, **kw)

    def test_paper_profile_rejects_protocol_drift(self):
        cfg = resolve_config(dict(profile="paper"))
        self.assertEqual((cfg["max_epochs"], cfg["loss_fn"], cfg["hop"]), (200, "mse", 4))
        self.assertEqual(cfg["tcn_norm"], "none")
        for override in (dict(max_epochs=199), dict(loss_fn="l1"), dict(normalization="epoch"),
                         dict(tcn_norm="batch"), dict(resampling="offline"), dict(hop=32),
                         dict(variant="tcn_se_la16"), dict(output_delay=103)):
            with self.subTest(override=override), self.assertRaises(ValueError):
                resolve_config(dict(profile="paper", **override))

    def test_delay_direction_and_excluded_targets_have_no_gradient(self):
        # Identity at acquisition t+2 is the perfect predictor of clean[t].
        prediction = torch.arange(12., requires_grad=True)
        clean = torch.arange(12.) + 2
        cfg = dict(output_delay=2, valid_history_samples=4, valid_future_samples=3)
        a, b = aligned(prediction, clean, cfg)
        self.assertTrue(torch.equal(a, b))
        a.sum().backward()
        self.assertTrue(torch.equal(prediction.grad, torch.tensor([0.]*5+[1.]*6+[0.])))
        self.assertEqual(target_interval(12, cfg), (3, 9))
        with self.assertRaises(ValueError):
            target_interval(5, cfg)

    def test_changing_delay_preserves_architecture_weights_and_count(self):
        models = []
        for delay in (0, 8, 16, 32):
            cfg = resolve_config(dict(profile="paper", output_delay=delay))
            torch.manual_seed(42)
            models.append(build_model_from_config(cfg))
        for model in models[1:]:
            self.assertEqual(models[0].model_config, model.model_config)
            for key, tensor in models[0].state_dict().items():
                self.assertTrue(torch.equal(tensor, model.state_dict()[key]))

    def test_finite_context_and_training_causality(self):
        model = self.model().train()
        from models.causal_config import ChannelNorm
        self.assertFalse(any(isinstance(layer, (ChannelNorm, torch.nn.BatchNorm1d))
                             for layer in model.modules()))
        self.assertEqual(context(model), (199, 0))
        self.assertEqual(context(self.model("tcn_se_bi")), (97, 102))
        x = torch.randn(2, 1, 512)
        changed = x.clone(); changed[..., 300:] += 20
        with torch.no_grad():
            self.assertLess(float((model(x)[..., :300]-model(changed)[..., :300]).abs().max()), 1e-6)
        cfg = resolve_config(dict(profile="paper", valid_history_samples=199))
        with self.assertRaisesRegex(ValueError, "history"):
            validate_model(model, cfg, 512)
        cfg["valid_history_samples"] = 200
        validate_model(model, cfg, 512)
        with self.assertRaises(ValueError):
            context(build_model("tcn_se"))

    def test_raw_filter_batch_vs_hop_parity_crosses_all_cache_boundaries(self):
        for variant in ("tcn", "tcn_se", "tcn_mla", "tcn_mha_par"):
            model = self.model(variant)
            model.run_config = dict(protocol="paper", normalization="fixed", normalization_scale=3.,
                                    output_delay=16, valid_history_samples=316, valid_future_samples=32)
            with self.subTest(variant=variant):
                result = parity(model, np.random.default_rng(5).normal(size=800).astype(np.float32)*3)
                self.assertTrue(result["passed"], result)
                self.assertTrue(result["includes_causal_filters"])
                self.assertEqual(result["hop_samples"], 4)

    def test_step_graph_preserves_finite_se_and_channel_normalization(self):
        from models.step_graph import StepGraph
        for variant in ("tcn_se", "tcn_mla"):
            model = self.model(variant, tcn_norm="channel").eval()
            graph = StepGraph(model)
            signal = torch.randn(1, 1, 350)
            state = graph.initial_state(1)
            output = []
            with torch.no_grad():
                reference = model(signal)
                for i in range(signal.shape[-1]):
                    values = graph(signal[..., i:i+1], *state)
                    y, state = values[0], values[1:]
                    output.append(y)
            self.assertLess(float((torch.cat(output, -1)-reference).abs().max()), 1e-4)

    def test_model_checks_use_the_aligned_prefix_and_separate_noncausal_controls(self):
        from experiment.methodology import model_stream_checks

        model = self.model("tcn_se")
        model.run_config = dict(output_delay=16, valid_history_samples=200, valid_future_samples=16)
        model.model_config["max_seq_len"] = 480
        result = model_stream_checks(model)
        self.assertEqual(result["n_samples"], 480)
        self.assertTrue(result["parity"]["passed"], result)
        self.assertEqual(result["parity"]["domain"], "model_input")
        causal = result["causality"]
        self.assertTrue(causal["passed"], causal)
        self.assertEqual(causal["output_delay_samples"], 16)
        self.assertEqual(causal["checked_targets"], causal["changed_at_sample"] - 199 - 16)

        class FutureReader(torch.nn.Module):
            causal = False

            def __init__(self):
                super().__init__()
                self.gain = torch.nn.Parameter(torch.ones(()))

            def forward(self, x):
                return x.roll(-1, dims=-1) * self.gain

        result = model_stream_checks(FutureReader())
        self.assertIsNone(result["parity"]["passed"])
        self.assertFalse(result["causality"]["passed"], result)

    def test_checkpoint_rebuilds_new_blocks_but_legacy_still_loads(self):
        from experiment.checkpoints import load_model
        from models.variants import resolved_arch
        x = torch.randn(1, 1, 512)
        for norm in ("none", "channel", "batch"):
            model = self.model(tcn_norm=norm).eval()
            raw = dict(model_state=model.state_dict(), model_config=resolved_arch(model))
            del raw["model_config"]["tcn_norm"]
            rebuilt, arch, _ = load_model(raw)
            self.assertEqual(arch["tcn_norm"], norm)
            with torch.no_grad():
                self.assertTrue(torch.equal(model(x), rebuilt(x)))

    def test_paper_onnx_step_export_preserves_raw_parity_and_delay_metadata(self):
        from deploy.export_onnx import export_checkpoint
        from models.variants import resolved_arch
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            model = self.model().eval()
            cfg = resolve_config(dict(profile="paper", hidden_channels=8, output_delay=16,
                                      normalization_scale=3.))
            path = Path(tmp)/"best.pt"
            torch.save(dict(model_state=model.state_dict(), model_config=resolved_arch(model),
                            run_config=cfg, epoch=0, val_loss=0.), path)
            sidecar = export_checkpoint(path, out_dir=tmp, name="paper", incremental=True)
            self.assertEqual(sidecar["preprocessing"]["output_delay_samples"], 16)
            self.assertEqual(sidecar["verification"]["domain"], "raw_amplitude")
            self.assertLess(sidecar["verification"]["max_abs_diff_vs_incremental"], 1e-4)

    def test_real_data_offline_chunks_match_the_full_finite_reference(self):
        from eegio_eval import stream_apply
        model = self.model("tcn_se_bi").eval()
        model.run_config = dict(protocol="paper", normalization="fixed", normalization_scale=3.,
                                valid_history_samples=200, valid_future_samples=102)
        signal = np.random.default_rng(9).normal(size=4500).astype(np.float32)
        with torch.no_grad():
            reference = model(torch.from_numpy(signal/3)[None,None])[0,0].numpy()*3
        output = stream_apply(model, signal, hop=4)
        np.testing.assert_allclose(output[199:-102],reference[199:-102],atol=1e-5,rtol=0)
        self.assertTrue(np.isnan(output[:199]).all())

    def test_validation_selects_pure_gt_mse_even_if_training_uses_l1(self):
        from trainer.training import Trainer
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            batch = dict(noisy=torch.ones(2, 1, 12)*2, clean=torch.zeros(2, 1, 12))
            trainer = Trainer(torch.nn.Conv1d(1, 1, 1), [batch], [batch], save_dir=tmp,
                              max_epochs=1, loss_fn="l1", device=torch.device("cpu"))
            with torch.no_grad():
                trainer.model.weight.fill_(1); trainer.model.bias.zero_()
            loss, _ = trainer._validate(0)
            self.assertAlmostEqual(loss, 4.)
            self.assertEqual(trainer.history["config"]["selection_metric"], "gt_mse")

    def test_kd_terms_use_the_common_target_interval(self):
        from trainer.distillation import KDTrainer
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            model, teacher = self.model(), self.model("tcn_se_bi")
            batch = dict(noisy=torch.randn(2, 1, 512), clean=torch.randn(2, 1, 512))
            cfg = dict(output_delay=16, valid_history_samples=200, valid_future_samples=102)
            trainer = KDTrainer(model, teacher, [batch], [batch], save_dir=tmp, max_epochs=1,
                                run_config=cfg, kd_feature_layers=["backbone"], kd_feature_weight=.1,
                                device=torch.device("cpu"), loss_fn="mse")
            actual = trainer.kd_criterion.forward
            seen = []
            def capture(student, teacher, clean, sf, tf):
                seen.append((student.shape[-1], teacher.shape[-1], clean.shape[-1],
                             sf["backbone"].shape[-1], tf["backbone"].shape[-1]))
                return actual(student, teacher, clean, sf, tf)
            with patch.object(trainer.kd_criterion, "forward", side_effect=capture):
                trainer._train_one_epoch(1)
            self.assertEqual(seen, [(211,)*5])
            self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
            trainer._cleanup_training()

    def test_causal_resampling_does_not_use_future_and_preserves_delay(self):
        from data.resampling import causal_resample, resampling_info
        x = np.random.default_rng(7).normal(size=512)
        for rate in (250., 512.):
            whole = causal_resample(x, 256., rate)
            prefix = causal_resample(x[:101], 256., rate)
            np.testing.assert_array_equal(whole[:len(prefix)], prefix)
            self.assertGreater(resampling_info(256., rate)["delay_ms"], 0)

    def test_filter_chunk_partition_does_not_change_outputs(self):
        from deploy.preprocessing import CausalFilter
        x = np.random.default_rng(1).normal(size=700)
        reference = CausalFilter(250).process(x)
        stream = CausalFilter(250)
        chunks = np.concatenate([stream.process(x[i:i+4]) for i in range(0, len(x), 4)])
        np.testing.assert_array_equal(chunks, reference)

    def test_algorithmic_delay_is_not_mislabeled_as_measured_latency(self):
        result = latency(dict(hop=4, output_delay=16), 250)
        self.assertEqual(result["algorithmic_budget_ms"], 80.)
        self.assertFalse(result["measured"])

    def test_plan_freezes_three_seeds_same_crop_and_refuses_replacement(self):
        from experiment.paper import make_plan
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(tmp, fs=250.)
            self.assertEqual(len(plan["jobs"]), 27)
            self.assertEqual(plan["common_target_interval"], [219, 398])
            for job in plan["jobs"]:
                cfg = json.loads(Path(job["config"]).read_text())
                self.assertEqual(cfg["max_epochs"], 200)
                self.assertIn(cfg["master_seed"], (42,54,67))
                self.assertFalse((Path(job["run_dir"])/"best.pt").exists())
            with self.assertRaises(FileExistsError):
                make_plan(tmp, fs=250.)


class StatisticsAndHardwareTests(unittest.TestCase):
    def test_sample_sd_and_tost_require_independent_units(self):
        from experiment.statistics import mean_sd, paired_tost
        self.assertEqual(mean_sd([1,2,3])["sd"], 1.)
        a, b = {42: .7, 54: .701, 67: .699}, {42:.7,54:.7,67:.7}
        self.assertTrue(paired_tost(a,b,.02,"seed")["equivalent"])
        self.assertFalse(paired_tost({k:v+.2 for k,v in a.items()},b,.02,"seed")["equivalent"])
        with self.assertRaises(ValueError):
            paired_tost(a,b,.02,"segment")
        with self.assertRaises(ValueError):
            paired_tost(a,b,0,"subject")

    def test_pairing_requires_identity_and_identical_target_positions(self):
        from experiment.statistics import pair_segments
        a = [dict(segment_id="one", target_interval=[20,400], sample_rate_hz=250, error=.5)]
        self.assertEqual(pair_segments(a,a,"error"), (.5,.5,1))
        with self.assertRaises(ValueError):
            pair_segments(a,[{**a[0],"segment_id":"other"}],"error")
        with self.assertRaises(ValueError):
            pair_segments(a,[{**a[0],"target_interval":[0,512]}],"error")

    def test_external_meter_is_time_weighted_and_idle_subtracted(self):
        from deploy.paper_bench import power_summary
        with tempfile.TemporaryDirectory() as tmp:
            active, idle = Path(tmp)/"active.csv", Path(tmp)/"idle.csv"
            active.write_text("time_s,power_mw\n0,1000\n1,2000\n3,2000\n")
            idle.write_text("time_s,power_mw\n0,500\n3,500\n")
            result = power_summary(active,idle,3)
            self.assertAlmostEqual(result["total_power_mw"], 5500/3)
            self.assertAlmostEqual(result["incremental_power_mw"], 4000/3)
            with self.assertRaises(ValueError):
                power_summary(active,idle,4)


if __name__ == "__main__":
    unittest.main()
