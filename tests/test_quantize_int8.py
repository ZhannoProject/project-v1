"""
Static INT8, and the one claim the whole module rests on.

This project refused INT8 once, because ``quantize_dynamic`` derives each
activation's scale from the tensor it is quantizing -- the whole 512-sample
window -- so a sample at *t* takes part in the scale applied to samples
before it.  ``test_dynamic_quantization_breaks_causality`` reproduces that
failure and ``test_static_quantization_stays_causal`` shows the static graph
does not have it, on the same exported model, in the same test run.  Those
two together are why ``deploy/quantize_int8.py`` exists; everything else here
is the scaffolding around them.

Run: python -m unittest discover -s tests -p 'test_quantize_int8.py'
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deploy.export_onnx import (                                   # noqa: E402
    INPUT_NAME, OUTPUT_NAME, _session, export_to_onnx,
)
from deploy.quantize_int8 import (                                 # noqa: E402
    ACTIVATION_TYPES, CALIBRATORS, CAUSALITY_TOL, CONV_ONLY,
    _synthetic_calibration, probe_causality, quantize_checkpoint_graph,
    quantize_graph, verify_int8,
)
from models.variants import build_model                            # noqa: E402

SEQ_LEN = 128           # long enough for the probe, short enough to be quick
HIDDEN = 8


def _runner(path):
    sess = _session(path, 2)
    return lambda x: sess.run([OUTPUT_NAME], {INPUT_NAME: x})[0]


class _ExportedModel(unittest.TestCase):
    """A small causal TCN exported to fp32 ONNX once per test class."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls._tmp.name)
        cls.fp32 = cls.dir / "tiny.onnx"
        model = build_model("tcn", hidden_channels=HIDDEN).eval()
        export_to_onnx(model, cls.fp32, seq_len=SEQ_LEN)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _static(self, name="static.onnx", **kw):
        out = self.dir / name
        quantize_graph(self.fp32, out, _synthetic_calibration(seq_len=SEQ_LEN), **kw)
        return out


# ============================================================
# The claim
# ============================================================
class TestCausality(_ExportedModel):

    def test_the_fp32_graph_is_causal(self):
        r = probe_causality(_runner(self.fp32), SEQ_LEN)
        self.assertEqual(r["past_diff"], 0.0)
        self.assertGreater(r["future_diff"], 0.0)

    def test_static_quantization_stays_causal(self):
        # Scales are graph constants, so nothing at run time depends on the
        # window's contents and a perturbation cannot reach backwards.
        r = probe_causality(_runner(self._static()), SEQ_LEN)
        self.assertLess(r["past_diff"], CAUSALITY_TOL)
        self.assertGreater(r["future_diff"], 0.0,
                           "the probe perturbed nothing downstream either")

    def test_dynamic_quantization_breaks_causality(self):
        # The documented reason INT8 was rejected the first time. If this
        # ever starts passing, onnxruntime changed how dynamic quantization
        # picks its scales and the module's premise needs re-checking.
        from onnxruntime.quantization import QuantType, quantize_dynamic

        dyn = self.dir / "dynamic.onnx"
        quantize_dynamic(str(self.fp32), str(dyn), weight_type=QuantType.QInt8)
        r = probe_causality(_runner(dyn), SEQ_LEN)
        self.assertGreater(
            r["past_diff"], CAUSALITY_TOL,
            "dynamic INT8 was expected to leak the future into the past")

    def test_the_probe_detects_a_non_causal_graph(self):
        # Pointed at a model that really does read ahead, the probe must say
        # so -- otherwise a past_diff of 0 proves nothing about the method.
        import torch

        model = build_model("tcn_bi", hidden_channels=HIDDEN).eval()
        with torch.no_grad():
            path = self.dir / "bi.onnx"
            export_to_onnx(model, path, seq_len=SEQ_LEN)
        r = probe_causality(_runner(path), SEQ_LEN)
        self.assertGreater(r["past_diff"], CAUSALITY_TOL)


# ============================================================
# Quantization settings
# ============================================================
class TestQuantizeGraph(_ExportedModel):

    def test_it_writes_a_graph_whose_weights_are_int8(self):
        # Not "a smaller file": QDQ adds a Quantize/Dequantize pair and a
        # scale and zero-point initializer per quantized tensor, and that
        # overhead is per-tensor rather than per-weight.  On this 8-channel
        # test model it outweighs the saving and the file grows; on the real
        # 64-channel baseline the weights dominate and 578.6 kB becomes
        # 265.0 kB.  What is invariant at any width is that the weights
        # themselves are now 8-bit.
        import onnx

        out = self._static()
        self.assertTrue(out.is_file())
        model = onnx.load(str(out))
        dtypes = {i.data_type for i in model.graph.initializer}
        self.assertIn(onnx.TensorProto.INT8, dtypes)
        self.assertTrue(any(n.op_type == "QuantizeLinear" for n in model.graph.node))
        self.assertTrue(any(n.op_type == "DequantizeLinear" for n in model.graph.node))

    def test_the_report_describes_what_was_done(self):
        out = self.dir / "described.onnx"
        r = quantize_graph(self.fp32, out, _synthetic_calibration(seq_len=SEQ_LEN),
                           activation="int8", per_channel=False,
                           calibrate_method="minmax")
        self.assertEqual(r["quant_format"], "QDQ")
        self.assertEqual(r["activation_type"], "int8")
        self.assertEqual(r["weight_type"], "int8")
        self.assertFalse(r["per_channel"])
        self.assertEqual(r["calibrate_method"], "minmax")
        self.assertEqual(r["op_types_to_quantize"], "all supported")
        self.assertGreater(r["calibration_windows"], 0)

    def test_every_activation_type_and_calibrator_produces_a_causal_graph(self):
        for act in ACTIVATION_TYPES:
            for calib in CALIBRATORS:
                with self.subTest(activation=act, calibrate_method=calib):
                    out = self._static(f"{act}_{calib}.onnx",
                                       activation=act, calibrate_method=calib)
                    r = probe_causality(_runner(out), SEQ_LEN)
                    self.assertLess(r["past_diff"], CAUSALITY_TOL)

    def test_conv_only_is_recorded_as_such(self):
        out = self.dir / "convonly.onnx"
        r = quantize_graph(self.fp32, out, _synthetic_calibration(seq_len=SEQ_LEN),
                           op_types=CONV_ONLY)
        self.assertEqual(r["op_types_to_quantize"], ["Conv"])

    def test_bad_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            quantize_graph(self.fp32, self.dir / "x.onnx",
                           _synthetic_calibration(seq_len=SEQ_LEN), activation="fp8")
        with self.assertRaises(ValueError):
            quantize_graph(self.fp32, self.dir / "x.onnx",
                           _synthetic_calibration(seq_len=SEQ_LEN),
                           calibrate_method="vibes")


# ============================================================
# The causal mask versus a histogram calibrator
# ============================================================
class TestMaskPreflight(unittest.TestCase):
    """
    A causal attention graph carries a scalar -inf -- the mask value before
    softmax -- and a histogram calibrator cannot take a range of [-inf, inf].
    onnxruntime reports that as "autodetected range ... is not finite", which
    says nothing about masks; the module refuses first and names the cause.
    """

    def _graph_with_an_infinity(self, path: Path):
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        neg_inf = numpy_helper.from_array(
            np.array(-np.inf, dtype=np.float32), name="mask_value")
        node = helper.make_node("Add", [INPUT_NAME, "mask_value"], [OUTPUT_NAME])
        graph = helper.make_graph(
            [node], "masked",
            [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, [1, 1, 8])],
            [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, [1, 1, 8])],
            [neg_inf])
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]),
                  str(path))

    def test_a_histogram_calibrator_over_a_masked_graph_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "masked.onnx"
            self._graph_with_an_infinity(src)
            for calib in ("percentile", "entropy"):
                with self.assertRaises(ValueError) as cm:
                    quantize_graph(src, Path(tmp) / "out.onnx",
                                   _synthetic_calibration(seq_len=8),
                                   calibrate_method=calib)
                msg = str(cm.exception)
                self.assertIn("causal attention mask", msg)
                self.assertIn("conv_only", msg)

    def test_conv_only_and_minmax_are_not_refused(self):
        # Both are ways round the histogram problem, so neither should trip
        # the preflight; they fail or succeed on their own merits afterwards.
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "masked.onnx"
            self._graph_with_an_infinity(src)
            from deploy.quantize_int8 import _refuse_histogram_over_a_mask
            _refuse_histogram_over_a_mask(src, "minmax", None)
            _refuse_histogram_over_a_mask(src, "percentile", CONV_ONLY)


# ============================================================
# Calibration
# ============================================================
class TestCalibration(unittest.TestCase):

    def test_the_reader_yields_then_stops_then_rewinds(self):
        r = _synthetic_calibration(n_batches=3, batch_size=2, seq_len=16)
        got = [r.get_next() for _ in range(3)]
        self.assertTrue(all(g is not None for g in got))
        self.assertEqual(tuple(got[0][INPUT_NAME].shape), (2, 1, 16))
        self.assertIsNone(r.get_next())
        r.rewind()
        self.assertIsNotNone(r.get_next())

    def test_windows_are_sigma_normalised_like_the_training_data(self):
        r = _synthetic_calibration(n_batches=1, batch_size=8, seq_len=512)
        x = r.get_next()[INPUT_NAME]
        self.assertAlmostEqual(float(x.std()), 1.0, places=1)


# ============================================================
# Verification and the sidecar
# ============================================================
class TestVerifyAndSidecar(_ExportedModel):

    def test_verify_reports_size_latency_agreement_and_both_probes(self):
        out = self._static("verified.onnx")
        r = verify_int8(self.fp32, out, seq_len=SEQ_LEN)
        self.assertLess(r["causality"]["past_diff"], CAUSALITY_TOL)
        self.assertEqual(r["causality"]["fp32_reference"]["past_diff"], 0.0)
        self.assertGreater(r["size_kb"]["fp32"], 0)
        self.assertGreater(r["latency"]["int8"]["median_ms"], 0)
        self.assertIn("rms_diff_rel", r["agreement"])
        self.assertNotIn("test_metrics", r)          # no data_dir was given

    def test_the_sidecar_says_the_calibration_was_synthetic(self):
        r = quantize_checkpoint_graph(self.fp32, out_dir=self.dir, seq_len=SEQ_LEN)
        meta = json.loads(Path(r["json"]).read_text())
        calib = meta["quantization"]["calibration"]
        self.assertEqual(calib["source"], "synthetic")
        self.assertIn("do not deploy", calib["warning"])
        self.assertEqual(meta["quantization"]["scheme"],
                         "static (scales are graph constants)")
        self.assertEqual(meta["io"]["input"]["name"], INPUT_NAME)

    def test_conv_only_gets_its_own_file_name(self):
        plain = quantize_checkpoint_graph(self.fp32, out_dir=self.dir, seq_len=SEQ_LEN)
        conv = quantize_checkpoint_graph(self.fp32, out_dir=self.dir, seq_len=SEQ_LEN,
                                         op_types=CONV_ONLY)
        self.assertTrue(str(plain["onnx"]).endswith("_int8.onnx"))
        self.assertTrue(str(conv["onnx"]).endswith("_int8conv.onnx"))
        self.assertNotEqual(plain["onnx"], conv["onnx"])


if __name__ == "__main__":
    unittest.main()
