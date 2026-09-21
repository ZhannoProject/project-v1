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
import raspi_infer  # noqa: E402
from raspi_infer import EEGDenoiser, StreamingDenoiser  # noqa: E402
import metrics as torch_metrics  # noqa: E402


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
                 ("tcn_mha", {"hidden_channels": 96, "bottleneck_ratio": 1 / 3}),
                 ("tcn_mla", {"kv_lora_rank": 4, "skip_bottleneck_ratio": 0.5}),
                 ("tcn_mha_par", {"skip_bottleneck_ratio": 0.25, "n_attn_layers": 2})]
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

    def test_rank_alias_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            # A tcn_mla_r4 checkpoint has tcn_mla's weight layout; it must load, keep
            # its own label, and the rank the label promises must match the weights.
            model, ckpt = _checkpoint(tmp, "tcn_mla_r4", n_attn_layers=2)
            loaded, arch, _ = load_checkpoint_model(ckpt)
            self.assertEqual((arch["variant"], arch["kv_lora_rank"]), ("tcn_mla_r4", 4))
            x = torch.randn(2, 1, 512)
            with torch.no_grad():
                self.assertTrue(torch.allclose(model.eval()(x), loaded(x)))

            # What the old rank-sweep bug produced: labelled r8, weights rank 16.
            _, bad = _checkpoint(tmp, "tcn_mla", kv_lora_rank=16)
            raw = torch.load(bad, map_location="cpu", weights_only=False)
            raw["model_config"]["variant"] = "tcn_mla_r8"
            torch.save(raw, bad)
            with self.assertRaisesRegex(ValueError, "labelled 'tcn_mla_r8'.*kv_lora_rank=8.*imply kv_lora_rank=16"):
                load_checkpoint_model(bad)

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

    def test_numpy_metrics_match_torch(self):
        rng = np.random.default_rng(0)
        clean = 30.0 * rng.standard_normal((6, 512)).astype(np.float32)
        pred = clean + 10.0 * rng.standard_normal((6, 512)).astype(np.float32)
        got = raspi_infer.metrics(pred, clean)
        p, c = torch.from_numpy(pred)[:, None, :], torch.from_numpy(clean)[:, None, :]
        for key, fn in [("rrmse_t", torch_metrics.rrmse_temporal),
                        ("rrmse_s", torch_metrics.rrmse_spectral), ("acc", torch_metrics.acc)]:
            np.testing.assert_allclose(got[key], fn(p, c).numpy(), rtol=1e-5, atol=1e-6, err_msg=key)

    def test_testdata_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, ckpt = _checkpoint(tmp, "tcn_mla", n_attn_layers=2, kv_lora_rank=8)
            with contextlib.redirect_stdout(io.StringIO()):
                export_checkpoint(ckpt, out_dir=tmp, name="m")
            rng = np.random.default_rng(0)
            clean = 30.0 * rng.standard_normal((20, 512)).astype(np.float32)
            npz = Path(tmp) / "t.npz"
            np.savez(npz, noisy=clean + 20.0 * rng.standard_normal(clean.shape).astype(np.float32),
                     clean=clean, snr_db=np.tile(np.linspace(-7, 2, 10), 2).astype(np.float32), fs=256)
            with contextlib.redirect_stdout(io.StringIO()):
                r = raspi_infer.evaluate_testdata(EEGDenoiser(Path(tmp) / "m.onnx", n_threads=1), npz)
            self.assertEqual(set(r), {"noisy input (no model)", "epoch mode", "stream mode (hop 32)"})
            self.assertTrue(all(np.isfinite(list(m.values())).all() for m in r.values()))


if __name__ == "__main__":
    unittest.main()
