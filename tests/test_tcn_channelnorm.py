"""Contracts for the explicitly named, no-SE ChannelNorm training control."""
import contextlib
import io
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs import load_config
from experiment.checkpoints import load_model
from experiment.config import resolve_config
from models.TCN_backbone import CausalSE, CausalTCNBackbone
from models.causal_config import ChannelNorm
from models.variants import base_variant, build_model, build_model_from_config, resolved_arch


class TCNChannelNormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_backbone_defaults_to_channelnorm_and_legacy_checkpoint_keeps_batchnorm(self):
        backbone = CausalTCNBackbone(1, 8, [1, 2])
        self.assertEqual(sum(isinstance(m, ChannelNorm) for m in backbone.modules()), 4)
        legacy = build_model("tcn", hidden_channels=8, tcn_norm="batch").eval()
        raw = {"model_state": legacy.state_dict(), "model_config": {"variant": "tcn"}}
        loaded, arch, _ = load_model(raw)
        self.assertEqual(arch["tcn_norm"], "batch")
        self.assertTrue(any(isinstance(m, torch.nn.BatchNorm1d) for m in loaded.modules()))
        signal = torch.randn(1, 1, 32)
        with torch.no_grad():
            torch.testing.assert_close(loaded(signal), legacy(signal), rtol=0, atol=0)

    def test_selected_variant_clears_inherited_se_and_normalization_settings(self):
        # A user switches models while editing an existing SE/other preset.
        inherited = dict(variant="tcn_channelnorm", hidden_channels=8,
                         se_reduction=8, tcn_norm="none", dropout=0.)
        cfg = resolve_config(inherited, profile="gui")
        self.assertEqual(cfg["tcn_norm"], "channel")
        self.assertIsNone(cfg["se_reduction"])
        with contextlib.redirect_stdout(io.StringIO()):
            model = build_model_from_config(inherited)
        self.assertFalse(any(isinstance(m, (CausalSE, torch.nn.BatchNorm1d)) for m in model.modules()))
        self.assertEqual(sum(isinstance(m, torch.nn.LayerNorm) for m in model.modules()), 10)
        x = torch.randn(2, 1, 32)
        prediction = model(x)
        prediction.square().mean().backward()
        self.assertEqual(prediction.shape, x.shape)
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_direct_construction_cannot_mislabel_another_architecture(self):
        for settings in ({"tcn_norm": "none"}, {"tcn_norm": "batch"}, {"se_reduction": 8}):
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, "contradicts"):
                build_model("tcn_channelnorm", hidden_channels=8, **settings)

    def test_checkpoint_reload_keeps_name_and_matches_plain_channelnorm_tcn(self):
        model = build_model("tcn_channelnorm", hidden_channels=8, dropout=0.).eval()
        raw = dict(model_state=model.state_dict(), model_config=resolved_arch(model),
                   run_config=dict(variant="tcn_channelnorm", tcn_norm="channel", master_seed=42))
        loaded, arch, _ = load_model(raw)
        self.assertEqual(arch["variant"], "tcn_channelnorm")
        self.assertEqual(arch["tcn_norm"], "channel")
        self.assertIsNone(arch["se_reduction"])
        self.assertEqual(base_variant(arch["variant"]), "tcn")
        plain = build_model("tcn", hidden_channels=8, dropout=0., tcn_norm="channel").eval()
        plain.load_state_dict(model.state_dict(), strict=True)
        x = torch.randn(2, 1, 40)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(x), loaded(x)))
            self.assertTrue(torch.equal(model(x), plain(x)))

    def test_250hz_preset_matches_the_exploratory_se_comparison_protocol(self):
        root = Path(__file__).resolve().parents[1]
        cfg = resolve_config(load_config(root / "configs/tcn_channelnorm_both_250hz.yaml"), profile="gui")
        self.assertEqual(cfg["variant"], "tcn_channelnorm")
        self.assertEqual(cfg["target_fs"], 250.)
        self.assertEqual(cfg["master_seed"], 42)
        self.assertEqual(cfg["normalization"], "epoch")
        self.assertEqual(cfg["resampling"], "offline")
        self.assertEqual(cfg["artifact"], "BOTH")
        self.assertEqual((cfg["max_epochs"], cfg["loss_fn"], cfg["batch_size"]), (200, "mse", 32))
        self.assertEqual(cfg["save_dir"], "./runs/tcn_channelnorm_both_250hz")


if __name__ == "__main__":
    unittest.main()
