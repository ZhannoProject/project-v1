"""The step graph must compute the same function as the stream it ports.

Run: PYTHONPATH=. python -m unittest discover -s tests -p 'test_step_graph.py'

``models/step_graph.py`` exists so incremental inference can leave PyTorch and
run on the Pi, where there is no torch.  That is only worth anything if the
graph computes what ``models/incremental.py`` computes, so that module is the
oracle here -- it is already pinned to the batch model from both ends, which
makes it the stricter reference and the thing this is a port of.

Both ends are checked: the torch ``StepGraph`` against ``IncrementalModel``,
and the exported ONNX against ``StepGraph``.  A graph that exports cleanly and
computes something else would otherwise pass unnoticed, which is the failure
mode that matters — it produces plausible numbers.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.incremental import IncrementalModel
from models.step_graph import MASK_FILL, StepGraph
from models.variants import build_model

#: Longer than the 512-sample attention window, so the validity mask is
#: exercised both filling up and full.  A graph that only agreed after warmup
#: would pass a shorter probe.
N = 640

STREAMABLE = ["tcn", "tcn_se", "tcn_se_r4", "tcn_mha", "tcn_mla", "tcn_mha_par",
              "tcn_mla_par", "tcn_mla_r8"]


def _signal(n=N, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n)


def _run_graph(graph, x):
    state = graph.initial_state(1)
    out = torch.empty_like(x)
    with torch.no_grad():
        for t in range(x.shape[0]):
            y, *state = graph(x[t].view(1, 1, 1), *state)
            out[t] = y.reshape(())
    return out


class EquivalenceTests(unittest.TestCase):
    def test_it_matches_the_streaming_model_it_ports(self):
        x = _signal()
        for variant in STREAMABLE:
            with self.subTest(variant=variant):
                model = build_model(variant).eval()
                ref = IncrementalModel(model).run(x)
                got = _run_graph(StepGraph(model).eval(), x)
                self.assertLess((ref - got).abs().max().item(), 1e-5)

    def test_it_matches_the_batch_model(self):
        """Checked directly too, not only through the oracle."""
        x = _signal(300, seed=1)
        for variant in ("tcn", "tcn_mla", "tcn_mha_par"):
            with self.subTest(variant=variant):
                model = build_model(variant).eval()
                with torch.no_grad():
                    ref = model(x.view(1, 1, -1))[0, 0]
                self.assertLess((ref - _run_graph(StepGraph(model).eval(), x)).abs().max().item(),
                                1e-5)

    def test_the_mask_makes_warmup_exact(self):
        """
        A ring buffer slices itself to what it has written; a static graph
        cannot, so a validity mask does the same job. Without it the attention
        would average over zeroed slots and the first outputs would be wrong —
        which is precisely the stretch this checks.
        """
        x = _signal(120, seed=2)                      # far shorter than the window
        model = build_model("tcn_mla").eval()
        ref = IncrementalModel(model).run(x)
        got = _run_graph(StepGraph(model).eval(), x)
        self.assertLess((ref - got).abs().max().item(), 1e-5)

    def test_the_initial_mask_is_closed(self):
        graph = StepGraph(build_model("tcn_mla").eval())
        masks = [t for s, t in zip(graph.state_spec, graph.initial_state(1))
                 if s["name"].startswith("attn_mask")]
        self.assertEqual(len(masks), 1, "the decoder shares one mask across its blocks")
        self.assertTrue(bool((masks[0] == MASK_FILL).all()),
                        "the stream must start seeing nothing, not a window of zeros")


class StateShapeTests(unittest.TestCase):
    def test_mla_declares_a_smaller_cache_than_mha(self):
        """
        The claim, as a property of the graph signature rather than of a
        measurement taken about it.
        """
        def cache_floats(variant):
            spec = StepGraph(build_model(variant).eval()).state_spec
            return sum(int(torch.tensor(s["shape"]).prod()) for s in spec
                       if s["name"].startswith("kv_"))
        self.assertLess(cache_floats("tcn_mla"), cache_floats("tcn_mha"))

    def test_the_spec_matches_what_forward_takes(self):
        for variant in STREAMABLE:
            with self.subTest(variant=variant):
                graph = StepGraph(build_model(variant).eval())
                state = graph.initial_state(1)
                self.assertEqual(len(state), len(graph.state_spec))
                for tensor, spec in zip(state, graph.state_spec):
                    self.assertEqual(list(tensor.shape)[1:], spec["shape"])
                self.assertEqual(len(set(s["name"] for s in graph.state_spec)),
                                 len(graph.state_spec), "state names must be unique")

    def test_bidirectional_and_turboquant_are_refused(self):
        for variant in ("tcn_mla_bi", "tcn_bi", "tcn_mha_full_bi"):
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                StepGraph(build_model(variant).eval())
        with self.assertRaises(ValueError):
            StepGraph(build_model("tcn_mla", turboquant=True).eval())


class OnnxExportTests(unittest.TestCase):
    """The exported graph, driven the way the Pi drives it."""

    def _export_and_run(self, model, x):
        import numpy as np
        import onnxruntime as ort
        from deploy.export_onnx import export_step_to_onnx

        graph = StepGraph(model).eval()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "step.onnx"
            info = export_step_to_onnx(model, path)
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            names = [i.name for i in sess.get_inputs()]
            self.assertEqual(names[1:], [s["name"] for s in info["state"]])
            state = {n: t.numpy() for n, t in zip(names[1:], graph.initial_state(1))}
            out = np.empty(len(x), dtype=np.float32)
            for t in range(len(x)):
                r = sess.run(None, {"x_t": x[t].view(1, 1, 1).numpy(), **state})
                out[t] = r[0].reshape(())
                state = dict(zip(names[1:], r[1:]))
        return torch.from_numpy(out), info

    def test_the_exported_graph_reproduces_the_stream(self):
        x = _signal(200, seed=3)
        for variant in ("tcn", "tcn_mla", "tcn_mha"):
            with self.subTest(variant=variant):
                # One model for both sides: build_model is not seeded, so two
                # calls give the same architecture with different weights and
                # the comparison would be meaningless.
                model = build_model(variant).eval()
                got, _ = self._export_and_run(model, x)
                ref = IncrementalModel(model).run(x)
                self.assertLess((ref - got).abs().max().item(), 1e-4)

    def test_the_graph_declares_its_state_in_and_out(self):
        """What makes it runnable without torch: the cache is in the signature."""
        import onnxruntime as ort
        from deploy.export_onnx import export_step_to_onnx
        model = build_model("tcn_mla").eval()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "step.onnx"
            info = export_step_to_onnx(model, path)
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            ins = [i.name for i in sess.get_inputs()]
            outs = [o.name for o in sess.get_outputs()]
        self.assertEqual(ins[0], "x_t")
        self.assertEqual(outs[0], "y_t")
        # Every state tensor comes back, or the stream could not continue.
        self.assertEqual(outs[1:], [f"{n}_out" for n in ins[1:]])
        self.assertEqual(info["mode"], "incremental")
        self.assertTrue(any(n.startswith("kv_c") for n in ins), "MLA should cache a latent")


if __name__ == "__main__":
    unittest.main()
