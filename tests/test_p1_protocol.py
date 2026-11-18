"""Regression checks for held-out provenance and raw-signal online causality."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch

import numpy as np
import torch

import eval as evaluation
from data.Dataloader import EEGDenoiseDataset, build_dataloaders
from data.protocol import (bind_run_config, dataset_protocol, evaluation_data,
                           validate_distillation_data)
from deploy.export_onnx import _preprocessing
from deploy.raspi_infer import (EEGDenoiser, IncrementalDenoiser, StreamingDenoiser,
                                normalization_scale)
from models.variants import build_model
from trainer.distillation import KDTrainer, load_teacher
from trainer.training import Trainer


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        rng = np.random.default_rng(7)
        for name in ("EEG", "EOG", "EMG"):
            np.save(self.data / f"{name}_all_epochs.npy",
                    (10 * rng.normal(size=(80, 512))).astype(np.float32))

    def loaders(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return build_dataloaders(str(self.data), artifact_type="BOTH", batch_size=8,
                                     normalization="fixed", **kwargs)

    def model(self, cfg=None, variant="tcn"):
        model = build_model(variant, hidden_channels=8, dropout=0)
        return bind_run_config(model, {"run_config": cfg or dict(master_seed=67, artifact="BOTH",
                                                                 target_fs=250)})

    def test_evaluation_defaults_to_checkpoint_seed_rate_and_scale(self):
        cfg = dict(master_seed=67, artifact="BOTH", target_fs=250,
                   normalization="fixed", normalization_scale=10.)
        model = self.model(cfg)
        data = evaluation_data(model)
        self.assertEqual(data["master_seed"], 67)
        self.assertEqual(data["target_fs"], 250)
        self.assertEqual(data["normalization_scale"], 10.)
        with contextlib.redirect_stdout(io.StringIO()):
            results, _ = evaluation.evaluate_per_snr(model, str(self.data), snr_list=[0],
                                                     device=torch.device("cpu"))
        self.assertTrue(np.isfinite(list(results[0].values())).all())

    def test_all_single_model_evaluators_use_the_same_checkpoint_protocol(self):
        model = self.model(variant="tcn_mha")
        class Captured(Exception):
            pass
        for function in (evaluation.evaluate_per_snr, evaluation.visualize_samples,
                         evaluation.plot_attention_heatmaps):
            def capture(**kwargs):
                self.assertEqual(kwargs["master_seed"], 67)
                self.assertEqual(kwargs["target_fs"], 250)
                self.assertEqual(kwargs["artifact_type"], "BOTH")
                raise Captured
            with self.subTest(function=function.__name__), patch.object(
                    evaluation, "EEGDenoiseDataset", side_effect=capture):
                with self.assertRaises(Captured), contextlib.redirect_stdout(io.StringIO()):
                    function(model, str(self.data), device=torch.device("cpu"))
        for _, module in evaluation._get_attn_modules(model):
            module.capture_weights(False)

    def test_explicit_mismatched_seed_and_artifact_are_rejected(self):
        model = self.model()
        with self.assertRaisesRegex(ValueError, "different split"):
            evaluation_data(model, seed=42)
        with self.assertRaisesRegex(ValueError, "artifact pools"):
            evaluation_data(model, artifact="EMG")

    def test_legacy_seed_must_be_explicit_and_checkpoint_beats_yaml(self):
        model = self.model({"normalization": "epoch"})
        with self.assertRaisesRegex(ValueError, "no recorded master_seed"):
            evaluation_data(model)
        self.assertEqual(evaluation_data(model, seed=67)["master_seed"], 67)
        bind_run_config(model, {"run_config": {"master_seed": 67}}, {"master_seed": 42})
        self.assertEqual(evaluation_data(model)["master_seed"], 67)

    def test_compare_snr_resolves_each_checkpoints_seed(self):
        models = [self.model(dict(master_seed=s, artifact="BOTH")) for s in (42, 67)]
        seen = []
        def score(model, *args, seed=None, **kwargs):
            seen.append(evaluation_data(model, seed)["master_seed"])
            return {0: dict(rrmse_t=1., rrmse_s=1., acc=0.)}, {}
        with patch.object(evaluation, "load_run_model", side_effect=[(m, "tcn") for m in models]), \
             patch.object(evaluation, "evaluate_per_snr", side_effect=score), \
             patch.object(evaluation.plt, "show"), contextlib.redirect_stdout(io.StringIO()):
            evaluation.compare_snr_curves(["a", "b"], str(self.data))
        evaluation.plt.close("all")
        self.assertEqual(seen, [42, 67])

    def test_switch_probe_restores_training_rate_and_normalization(self):
        model = self.model(dict(master_seed=67, artifact="BOTH", target_fs=250,
                                normalization="fixed", normalization_scale=10.))
        with contextlib.redirect_stdout(io.StringIO()):
            result = evaluation.evaluate_switch_response(model, str(self.data), probe_seeds=(0,),
                                                         n_epochs=4, verbose=False,
                                                         device=torch.device("cpu"))
        self.assertEqual(result["fs"], 250)
        with self.assertRaisesRegex(ValueError, "different split"):
            evaluation.evaluate_switch_response(model, str(self.data), master_seed=42)

    def test_one_training_scale_is_shared_by_all_splits_and_saved(self):
        train, val, test = self.loaders(master_seed=67, target_fs=250)
        scale = train.dataset.normalization_scale
        self.assertGreater(scale, 0)
        for loader in (train, val, test):
            self.assertEqual(loader.dataset.normalization_scale, scale)
            self.assertEqual(loader.dataset[0]["sigma"].item(), np.float32(scale))
        with contextlib.redirect_stdout(io.StringIO()):
            trainer = Trainer(build_model("tcn", hidden_channels=8, dropout=0), train, val,
                              save_dir=str(self.root / "run"), max_epochs=1, loss_fn="mse",
                              device=torch.device("cpu"))
            trainer.fit()
        ckpt = torch.load(self.root / "run" / "best.pt", weights_only=False)
        self.assertEqual(ckpt["run_config"]["normalization_scale"], scale)
        self.assertEqual(ckpt["run_config"]["master_seed"], 67)
        self.assertEqual(ckpt["run_config"]["target_fs"], 250)
        self.assertEqual(json.loads((self.root / "run" / "history.json").read_text())
                         ["config"]["normalization_scale"], scale)

    def test_training_and_evaluation_cli_round_trip(self):
        from trainer.__main__ import main as train_main
        run = self.root / "cli_run"
        args = ["trainer", "--data_dir", str(self.data), "--save_dir", str(run),
                "--variant", "tcn", "--artifact", "BOTH", "--hidden_channels", "8",
                "--master_seed", "67", "--target_fs", "250", "--device", "cpu",
                "--max_epochs", "1", "--loss_fn", "mse"]
        with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
            train_main()
        ckpt = torch.load(run / "best.pt", weights_only=False)
        self.assertEqual(ckpt["run_config"]["normalization"], "fixed")
        self.assertGreater(ckpt["run_config"]["normalization_scale"], 0)
        with patch.object(sys, "argv", ["eval", "--ckpt", str(run / "best.pt"),
                                        "--data_dir", str(self.data), "--mode", "snr"]), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            evaluation.main()
        self.assertIn("500 samples", output.getvalue())
        self.assertIn("AVG", output.getvalue())

    def test_all_mode_skips_cross_artifact_switch_probe_for_eog(self):
        model = build_model("tcn", hidden_channels=8, dropout=0)
        path = self.root / "eog.pt"
        torch.save(dict(model_config=model.model_config, model_state=model.state_dict(), epoch=1,
                        run_config=dict(master_seed=67, artifact="EOG")), path)
        with patch.object(sys, "argv", ["eval", "--ckpt", str(path), "--data_dir", str(self.data)]), \
             patch.object(evaluation, "evaluate_per_snr", return_value=({}, {})), \
             patch.object(evaluation, "visualize_samples"), \
             patch.object(evaluation, "plot_attention_heatmaps"), \
             patch.object(evaluation, "plot_snr_curves"), \
             patch.object(evaluation, "evaluate_switch_response") as switch, \
             contextlib.redirect_stdout(io.StringIO()):
            evaluation.main()
        switch.assert_not_called()

    def test_calibration_does_not_use_held_out_data(self):
        train, _, _ = self.loaders()
        original = train.dataset.normalization_scale
        path = self.data / "EEG_all_epochs.npy"
        raw = np.load(path)
        held_out = np.concatenate([EEGDenoiseDataset._split_indices(80, (.7, .15, .15), s, 42)
                                   for s in ("val", "test")])
        raw[held_out] *= 1000
        np.save(path, raw)
        again, _, _ = self.loaders()
        self.assertEqual(again.dataset.normalization_scale, original)
        with self.assertRaisesRegex(ValueError, "training normalization_scale"):
            EEGDenoiseDataset(str(self.data), "BOTH", "val", normalization="fixed")

    def test_raw_signal_future_changes_do_not_change_past_predictions(self):
        train, _, _ = self.loaders()
        scale = train.dataset.normalization_scale
        torch.manual_seed(8)
        x = 10 * torch.randn(2, 1, 512)
        changed = x.clone()
        changed[..., 256:] *= 20
        for variant in ("tcn", "tcn_se", "tcn_mha", "tcn_mla"):
            model = build_model(variant, hidden_channels=8, dropout=0).eval()
            with self.subTest(variant=variant), torch.no_grad():
                y1, y2 = model(x / scale) * scale, model(changed / scale) * scale
                torch.testing.assert_close(y1[..., :256], y2[..., :256], atol=0, rtol=0)

    def teacher(self, cfg):
        model = build_model("tcn_bi", hidden_channels=8, dropout=0)
        path = self.root / "teacher.pt"
        torch.save(dict(model_config=model.model_config, model_state=model.state_dict(),
                        run_config=cfg, val_loss=.1), path)
        with contextlib.redirect_stdout(io.StringIO()):
            return load_teacher(path)

    def test_kd_rejects_mismatch_before_weight_initialization_or_writes(self):
        train, val, _ = self.loaders(master_seed=67)
        cfg = dataset_protocol(train.dataset)
        teacher = self.teacher(dict(cfg, master_seed=42))
        student = build_model("tcn", hidden_channels=8, dropout=0)
        before = {k: v.clone() for k, v in student.state_dict().items()}
        with self.assertRaisesRegex(ValueError, "master_seed"):
            KDTrainer(student, teacher, train, val, init_from_teacher=True,
                      save_dir=str(self.root / "rejected"), loss_fn="mse")
        self.assertFalse((self.root / "rejected").exists())
        for key, value in student.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_kd_checks_missing_provenance_rate_artifact_and_normalization(self):
        train, val, _ = self.loaders(master_seed=67)
        cfg = dataset_protocol(train.dataset)
        cases = [({}, "master_seed"), (dict(cfg, target_fs=250), "target_fs"),
                 (dict(cfg, artifact="EOG"), "artifact"),
                 (dict(cfg, normalization="epoch", normalization_scale=None), "normalization"),
                 (dict(cfg, split_ratios=[.6, .2, .2]), "split_ratios")]
        for teacher_cfg, key in cases:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_distillation_data(self.teacher(teacher_cfg), train, val)

    def test_matching_kd_protocol_trains_and_teacher_stays_frozen(self):
        train, val, _ = self.loaders(master_seed=67)
        teacher = self.teacher(dataset_protocol(train.dataset))
        with contextlib.redirect_stdout(io.StringIO()):
            kd = KDTrainer(build_model("tcn", hidden_channels=8, dropout=0), teacher, train, val,
                           save_dir=str(self.root / "kd"), max_epochs=1, loss_fn="mse",
                           init_from_teacher=True, device=torch.device("cpu"))
            kd.fit()
        self.assertTrue((self.root / "kd" / "best.pt").exists())
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in teacher.parameters()))

    def test_online_interfaces_never_infer_scale_from_future_input(self):
        meta = dict(preprocessing=dict(normalization="fixed", normalization_scale=10.))
        # Exercise raw-scale wrappers with a deterministic causal backend, no ONNX dependency.
        d = EEGDenoiser.__new__(EEGDenoiser)
        d.meta, d.window = meta, 512
        d._run = lambda x: x + .25
        x = np.random.default_rng(4).normal(size=512).astype(np.float32)
        changed = x.copy()
        changed[256:] *= 100
        np.testing.assert_array_equal(d.denoise_stream(x)[:256], d.denoise_stream(changed)[:256])
        inc = IncrementalDenoiser.__new__(IncrementalDenoiser)
        inc.meta = meta
        inc.initial_state = lambda: {}
        inc.step = lambda x, state: (x + .25, state)
        np.testing.assert_array_equal(inc.denoise_stream(x)[:256], inc.denoise_stream(changed)[:256])
        for obj in (d, inc):
            obj.meta = {}
            with self.assertRaisesRegex(ValueError, "offline only"):
                obj.denoise_stream(x)
        np.testing.assert_allclose(d.denoise_epochs(x), x + .25 * x.std(), atol=1e-6)
        with self.assertRaisesRegex(ValueError, "offline only"):
            StreamingDenoiser(d)
        self.assertEqual(normalization_scale({}, sigma=10., online=True), 10.)
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                normalization_scale({}, sigma=value, online=True)
        with self.assertRaisesRegex(ValueError, "differs"):
            normalization_scale(meta, sigma=20., online=True)

    def test_sidecar_records_rate_scale_and_legacy_offline_contract(self):
        raw = dict(run_config=dict(target_fs=250, normalization="fixed", normalization_scale=10.))
        cfg = _preprocessing(raw, 500, "BOTH", "step")
        self.assertEqual(cfg["sample_rate_hz"], 250)
        self.assertEqual(cfg["normalization_scale"], 10.)
        self.assertTrue(cfg["causal"])
        self.assertFalse(_preprocessing({}, 512, "EOG", "step")["causal"])

    def test_int8_preserves_fixed_preprocessing_and_calibration_split(self):
        from deploy import quantize_int8 as quant
        fp32 = self.root / "model.onnx"
        cfg = dict(master_seed=67, artifact="BOTH", target_fs=250,
                   normalization="fixed", normalization_scale=10.)
        source = dict(data_protocol=cfg, provenance=dict(trained_master_seed=67, trained_artifact="BOTH"),
                      preprocessing=_preprocessing(dict(run_config=cfg), 500, "BOTH", "window"))
        fp32.with_suffix(".json").write_text(json.dumps(source))
        with patch.object(quant, "TrainSplitCalibration") as calibrate, \
             patch.object(quant, "quantize_graph", return_value={}), \
             patch.object(quant, "verify_int8", return_value={}):
            calibrate.return_value.n_windows = 4
            result = quant.quantize_checkpoint_graph(fp32, data_dir=str(self.data))
            self.assertEqual(calibrate.call_args.kwargs["data_options"]["master_seed"], 67)
            self.assertEqual(calibrate.call_args.kwargs["data_options"]["normalization_scale"], 10.)
            self.assertEqual(calibrate.call_args.kwargs["data_options"]["target_fs"], 250)
            self.assertEqual(result["preprocessing"], source["preprocessing"])
            self.assertEqual(result["provenance"], source["provenance"])
            with self.assertRaisesRegex(ValueError, "different split"):
                quant.quantize_checkpoint_graph(fp32, data_dir=str(self.data), master_seed=42)


if __name__ == "__main__":
    unittest.main()
