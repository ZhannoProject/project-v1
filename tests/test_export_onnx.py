"""Run: python -m unittest discover -s tests -p 'test_export_onnx.py'."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from models.variants import build_model
from trainer.checkpoint import save_checkpoint

from deploy.export_onnx import (export_checkpoint, infer_arch_from_state_dict,
                                load_checkpoint_model)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))
from raspi_infer import EEGDenoiser, StreamingDenoiser  # noqa: E402


def _checkpoint(tmp, variant, **kwargs):
    """Save a freshly initialised model the way the trainer does."""
    model = build_model(variant, hidden_channels=64, **kwargs)
    opt = torch.optim.Adam(model.parameters())
    sched = torch.optim.lr_scheduler.StepLR(opt, 1)
    run_dir = Path(tmp) / f"{variant}_test"
    run_dir.mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        save_checkpoint(run_dir, 0, model, opt, sched, 0.0, {}, is_best=True)
    return model, run_dir / "best.pt"


class ExportOnnxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_arch_inference_matches_construction(self):
        cases = [("tcn", {}), ("tcn_mha", {"n_attn_layers": 3}),
                 ("tcn_mla", {"kv_lora_rank": 2, "n_attn_layers": 2}),
                 ("tcn_mla_par", {"kv_lora_rank": 8}),
                 ("tcn_mha", {"hidden_channels": 96, "bottleneck_ratio": 1 / 3})]
        for variant, kw in cases:
            kw = {"hidden_channels": 64, **kw}
            sd = build_model(variant, **kw).state_dict()
            arch = infer_arch_from_state_dict(sd)
            self.assertEqual(arch["variant"], variant)
            rebuilt = build_model(**arch, n_heads=4) if variant != "tcn" else build_model(**arch)
            self.assertEqual({k: v.shape for k, v in rebuilt.state_dict().items()},
                             {k: v.shape for k, v in sd.items()}, variant)

    def test_legacy_checkpoint_without_model_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, ckpt = _checkpoint(tmp, "tcn_mla", kv_lora_rank=2)
            raw = torch.load(ckpt, map_location="cpu", weights_only=False)
            del raw["model_config"]                     # what pre-metadata checkpoints look like
            torch.save(raw, ckpt)
            loaded, arch, _ = load_checkpoint_model(ckpt)
            self.assertEqual(arch["kv_lora_rank"], 2)
            x = torch.randn(2, 1, 512)
            with torch.no_grad():
                self.assertTrue(torch.allclose(model.eval()(x), loaded(x)))

    def test_mla_2layer_export_verify_and_pi_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, ckpt = _checkpoint(tmp, "tcn_mla", n_attn_layers=2, kv_lora_rank=8)
            with contextlib.redirect_stdout(io.StringIO()):
                sidecar = export_checkpoint(ckpt, out_dir=tmp, name="m")   # asserts agreement + causality
            onnx_path = Path(tmp) / "m.onnx"
            self.assertTrue(onnx_path.exists())
            self.assertLess(sidecar["verification"]["max_abs_diff_vs_torch"], 1e-4)
            self.assertEqual(sidecar["verification"]["causality_past_diff"], 0.0)
            self.assertEqual(sidecar["model"]["n_attn_layers"], 2)
            self.assertEqual(json.loads((Path(tmp) / "m.json").read_text())["io"]["input"]["shape"], ["batch", 1, 512])

            # Pi-side runtime: raw-scale contract and streaming equivalence
            d = EEGDenoiser(onnx_path, n_threads=1)
            raw = 25.0 * np.random.default_rng(0).standard_normal((3, 512)).astype(np.float32)
            sigma = raw.std(axis=1, keepdims=True)
            with torch.no_grad():
                ref = model.eval()(torch.from_numpy(raw / sigma)[:, None, :]).numpy()[:, 0, :] * sigma
            np.testing.assert_allclose(d.denoise_epochs(raw), ref, atol=1e-3)

            sig = raw.reshape(-1)
            offline = d.denoise_stream(sig, hop=32)
            streamer = StreamingDenoiser(d, hop=32)
            online = np.concatenate([streamer.push(sig[i:i + 45]) for i in range(0, len(sig), 45)])
            np.testing.assert_array_equal(offline, online)
            np.testing.assert_array_equal(offline[:512], d.denoise_epochs(sig[:512]))


if __name__ == "__main__":
    unittest.main()
