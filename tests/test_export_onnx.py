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
from raspi_infer import (EEGDenoiser, LatencyMonitor, StreamingDenoiser,  # noqa: E402
                         duty_cycle_check, save_duty_rows, stream_realtime)
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

    def test_latency_monitor(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, ckpt = _checkpoint(tmp, "tcn_mla", n_attn_layers=2, kv_lora_rank=8)
            with contextlib.redirect_stdout(io.StringIO()):
                export_checkpoint(ckpt, out_dir=tmp, name="m")
            d = EEGDenoiser(Path(tmp) / "m.onnx", n_threads=1)
            hop, n_hops = 32, 6
            sig = np.random.default_rng(1).standard_normal(512 + hop * n_hops).astype(np.float32)
            ref = d.denoise_stream(sig, hop)

            # Push path: one record per hop, output untouched, budget arithmetic right.
            m = LatencyMonitor(hop, d.sample_rate)
            s = StreamingDenoiser(d, hop, m)
            out = np.concatenate([s.push(sig[i:i + 50]) for i in range(0, len(sig), 50)])
            np.testing.assert_array_equal(out, ref)
            self.assertEqual(len(m.total_ms), n_hops)
            self.assertIsNotNone(m.first_window_ms)
            summ = m.summary()
            self.assertEqual((summ["n_hops"], summ["hop"], summ["hop_budget_ms"]), (n_hops, hop, 125.0))
            self.assertTrue(all(i <= t for i, t in zip(m.infer_ms, m.total_ms)))   # run time is part of the hop
            self.assertEqual(summ["deadline_misses"], 0)
            for key in ("infer_ms", "total_ms", "overhead_ms", "interval_ms"):
                self.assertEqual(set(summ[key]), {"median", "mean", "p95", "p99", "max", "min", "std"})
            self.assertNotIn("e2e_latency_ms", summ)                            # no schedule attached

            # Offline path records the same hops; an impossible budget makes every hop a miss.
            m2 = LatencyMonitor(hop, sample_rate=1e9)                           # budget ~0 ms
            np.testing.assert_array_equal(d.denoise_stream(sig, hop, m2), ref)
            self.assertEqual((len(m2.total_ms), m2.summary()["deadline_misses"], m2.summary()["miss_rate"]),
                             (n_hops, n_hops, 1.0))

            # Paced replay (fast fake sample rate): identical output, e2e latency reported, files written.
            d.sample_rate = 200_000.0
            m3 = LatencyMonitor(hop, d.sample_rate, report_every=3)
            with contextlib.redirect_stdout(io.StringIO()) as log:
                np.testing.assert_array_equal(stream_realtime(d, sig, hop, m3), ref)
                summ3 = m3.print_summary()
            self.assertEqual(log.getvalue().count("[latency] hop"), n_hops // 3)   # rolling lines
            self.assertIn("e2e_latency_ms", summ3)
            self.assertIn("e2e_drift_ms", summ3)
            self.assertGreaterEqual(summ3["e2e_latency_ms"]["min"], summ3["total_ms"]["min"] * 0.99)
            expect = summ3["total_ms"]["mean"] / summ3["hop_budget_ms"]        # both sides are rounded
            self.assertAlmostEqual(summ3["duty_cycle"]["mean"], expect, delta=1e-3 * expect + 1e-3)
            js, csv = m3.save(Path(tmp) / "lat" / "run")
            self.assertEqual(json.loads(js.read_text())["n_hops"], n_hops)
            rows = csv.read_text().strip().splitlines()
            self.assertEqual(rows[0], "hop,infer_ms,total_ms,t_s,temp_c,e2e_latency_ms")
            self.assertEqual(len(rows) - 1, n_hops)
            self.assertEqual(rows[1].split(",")[0], "1")

    def test_duty_cycle_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, ckpt = _checkpoint(tmp, "tcn_mla", n_attn_layers=2, kv_lora_rank=8)
            with contextlib.redirect_stdout(io.StringIO()):
                export_checkpoint(ckpt, out_dir=tmp, name="m")
            d = EEGDenoiser(Path(tmp) / "m.onnx", n_threads=1)
            d.sample_rate = 2000.0                    # fake fs keeps the paced replay short
            sig = np.random.default_rng(2).standard_normal(1000).astype(np.float32)
            # hop 128 -> 64 ms period, far above one forward pass: OK.
            # hop 1   -> 0.5 ms period, far below it: must be flagged, with misses and a growing backlog.
            with contextlib.redirect_stdout(io.StringIO()) as log:
                rows = duty_cycle_check(d, sig, hops=(128, 1), seconds=0.5)
            by_hop = {r["hop"]: r for r in rows}
            self.assertEqual(set(by_hop), {128, 1})
            self.assertEqual(by_hop[128]["verdict"], "OK")
            self.assertEqual(by_hop[1]["verdict"], "FAIL")
            self.assertGreater(by_hop[1]["deadline_misses"], 0)
            self.assertGreater(by_hop[1]["duty_mean"], 1.0)
            self.assertGreater(by_hop[1]["e2e_drift_ms"], by_hop[1]["hop_budget_ms"])
            self.assertAlmostEqual(by_hop[128]["total_latency_median_ms"],
                                   by_hop[128]["hop_budget_ms"] + by_hop[128]["e2e_median_ms"], places=1)
            self.assertIn("smallest hop that is comfortably real-time: 128", log.getvalue())
            # A sweep carries the state of the machine it was measured on: a
            # duty cycle is only a number about the model if the clock was
            # held still, so the two are written together.
            js, csv = save_duty_rows(rows, Path(tmp) / "duty",
                                     environment={"verdict": "locked"})
            saved = json.loads(js.read_text())
            self.assertEqual([r["hop"] for r in saved["rows"]], [128, 1])
            self.assertEqual(saved["environment"], {"verdict": "locked"})
            lines = csv.read_text().strip().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertTrue(lines[0].startswith("hop,hop_budget_ms,") and lines[0].endswith(",verdict"))

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
            # The scores go into the trace beside the timings, so the return
            # carries what they were measured on as well as what they were:
            # a number without its test set and split is not reportable.
            self.assertEqual(set(r), {"testdata", "n_epochs", "hop", "split", "modes"})
            self.assertEqual((r["testdata"], r["n_epochs"], r["hop"]), ("t.npz", 20, 32))
            self.assertEqual(set(r["split"]), {"artifact", "master_seed", "source"})
            self.assertEqual(set(r["modes"]),
                             {"noisy input (no model)", "epoch mode", "stream mode (hop 32)"})
            self.assertTrue(all(np.isfinite(list(m.values())).all()
                                for m in r["modes"].values()))


class ModeLabellingTests(unittest.TestCase):
    """
    Each row of the scoring table has to name the computation that produced it.

    The rows are assigned in a branch, and the numbers are plausible either
    way, so a mislabelled row is not something a reader could catch. Two
    things are pinned: that a label matches its own independent computation,
    and that the step graph does not borrow the windowed path's name for a
    mechanism it does not use.
    """

    def _fixture(self, tmp):
        rng = np.random.default_rng(0)
        clean = 30.0 * rng.standard_normal((20, 512)).astype(np.float32)
        npz = Path(tmp) / "t.npz"
        np.savez(npz, noisy=clean + 20.0 * rng.standard_normal(clean.shape).astype(np.float32),
                 clean=clean, snr_db=np.tile(np.linspace(-7, 2, 10), 2).astype(np.float32),
                 fs=256, master_seed=42, artifact="EOG", split="test")
        return npz, clean

    def test_each_windowed_row_is_the_computation_it_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, ckpt = _checkpoint(tmp, "tcn")
            with contextlib.redirect_stdout(io.StringIO()):
                export_checkpoint(ckpt, out_dir=tmp, name="m")
            npz, clean = self._fixture(tmp)
            d = EEGDenoiser(Path(tmp) / "m.onnx", n_threads=1)
            noisy = np.load(npz)["noisy"]
            for hop in (8, 32):
                with self.subTest(hop=hop):
                    with contextlib.redirect_stdout(io.StringIO()):
                        got = raspi_infer.evaluate_testdata(d, npz, hop,
                                                            strict_split=False)["modes"]
                    want_epoch = raspi_infer.metrics(d.denoise_epochs(noisy), clean)
                    want_stream = raspi_infer.metrics(
                        d.denoise_stream(noisy.reshape(-1), hop).reshape(noisy.shape), clean)
                    for key, want in (("epoch mode", want_epoch),
                                      (f"stream mode (hop {hop})", want_stream)):
                        self.assertIn(key, got, f"no row called {key!r}")
                        for metric, v in want.items():
                            self.assertAlmostEqual(got[key][metric], float(v.mean()), places=9,
                                                   msg=f"{key}:{metric} is not what it claims")
                    # the hop in the label is the hop that ran
                    self.assertNotIn(f"stream mode (hop {64 if hop == 8 else 8})", got)

    def test_the_step_graph_does_not_claim_a_batched_epoch(self):
        """
        It steps a sample at a time and resets between epochs. The scoring
        protocol matches the windowed path's -- which is what keeps the two
        numbers comparable -- but the mechanism does not, and the label says so.
        """
        with tempfile.TemporaryDirectory() as tmp:
            _, ckpt = _checkpoint(tmp, "tcn")
            with contextlib.redirect_stdout(io.StringIO()):
                export_checkpoint(ckpt, out_dir=tmp, name="m")
                export_checkpoint(ckpt, out_dir=tmp, name="m", incremental=True)
            npz, _ = self._fixture(tmp)

            with contextlib.redirect_stdout(io.StringIO()):
                stepped = raspi_infer.evaluate_testdata(
                    raspi_infer.IncrementalDenoiser(Path(tmp) / "m_step.onnx", n_threads=1),
                    npz, 32, strict_split=False)["modes"]
                batched = raspi_infer.evaluate_testdata(
                    EEGDenoiser(Path(tmp) / "m.onnx", n_threads=1),
                    npz, 32, strict_split=False)["modes"]

        self.assertIn("epoch mode (stepped)", stepped)
        self.assertNotIn("epoch mode", stepped,
                         "the step graph borrowed the batched path's row name")
        # No stream row either: for a step graph it would be the same numbers.
        self.assertFalse([k for k in stepped if k.startswith("stream")])

        a, b = stepped["epoch mode (stepped)"], batched["epoch mode"]
        for metric in a:
            self.assertAlmostEqual(a[metric], b[metric], places=4,
                                   msg=f"the two mechanisms disagree on {metric}")


if __name__ == "__main__":
    unittest.main()
