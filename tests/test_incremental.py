"""The streaming path must compute the same function as the batch path.

Run: PYTHONPATH=. python -m unittest discover -s tests -p 'test_incremental.py'

``models/incremental.py`` exists to make MLA's KV-cache claim measurable, and a
cache that is cheap but wrong measures nothing.  So the equivalence is pinned
from both ends: against a whole-signal forward pass (exact, ~1e-6), and against
the sliding-window path the Pi runtime actually uses (exact for the baseline,
~1e-4 relative for the attention variants, for the architectural reason set out
in the module docstring).  The two tolerances are asserted separately on
purpose — folding them into one loose number would let a real regression in the
convolution buffers hide behind attention's expected mismatch.

Also checked: that streaming cannot leak the future, that MLA's two cache
strategies agree, that the cache sizes are the ones the architecture promises,
and that bidirectional teachers are refused rather than quietly streamed.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.incremental import IncrementalModel, MHAStepper, MLAStepper
from models.variants import build_model

#: Every deployable topology.  The rank aliases go through the same MLA code
#: path, so one of them stands in for the sweep.
STREAMABLE = ["tcn", "tcn_se", "tcn_se_r4", "tcn_mha", "tcn_mla", "tcn_mha_par",
              "tcn_mla_par", "tcn_mla_r8"]

WINDOW = 512


def _signal(n: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(n)


class WholeSignalEquivalenceTests(unittest.TestCase):
    """Stream vs. one forward pass over the same samples: exact."""

    def test_matches_batch_forward(self):
        x = _signal(300)
        for variant in STREAMABLE:
            with self.subTest(variant=variant):
                model = build_model(variant).eval()
                with torch.no_grad():
                    ref = model(x.view(1, 1, -1))[0, 0]
                got = IncrementalModel(model).run(x)
                self.assertLess((ref - got).abs().max().item(), 1e-5,
                                "streaming diverged from the batch forward pass")

    def test_matches_batch_forward_batched(self):
        """Several independent streams at once must not contaminate each other."""
        torch.manual_seed(1)
        x = torch.randn(4, 256)
        for variant in ("tcn_mla", "tcn_mha_par"):
            with self.subTest(variant=variant):
                model = build_model(variant).eval()
                with torch.no_grad():
                    ref = model(x.unsqueeze(1))[:, 0]
                got = IncrementalModel(model).run(x)
                self.assertLess((ref - got).abs().max().item(), 1e-5)

    def test_batched_equals_one_at_a_time(self):
        model = build_model("tcn_mla").eval()
        torch.manual_seed(2)
        x = torch.randn(3, 200)
        together = IncrementalModel(model).run(x)
        stream = IncrementalModel(model)
        for row in range(x.shape[0]):
            alone = stream.run(x[row])
            self.assertLess((together[row] - alone).abs().max().item(), 1e-6,
                            "a stream's output depended on its neighbours in the batch")

    def test_reset_restores_initial_state(self):
        """A reset stream reproduces its own first run, bit for bit."""
        model = build_model("tcn_mla").eval()
        x = _signal(180, seed=3)
        stream = IncrementalModel(model)
        first = stream.run(x).clone()
        second = stream.run(x)
        self.assertTrue(torch.equal(first, second),
                        "reset() left state behind: the second run differed")


class SqueezeExcitationStreamingTests(unittest.TestCase):
    """
    The SE gate has temporal state, and skipping it is silent.

    ``ResidualBlockStepper`` walks a block's parts by name. When SE was added
    it was simply not among them, so the stream returned plausible numbers
    from a model without the gate -- a 0.2 absolute error and no complaint.
    These pin the two things that would hide it again.
    """

    def test_a_gated_model_does_not_stream_as_an_ungated_one(self):
        x = _signal(256, seed=11)
        gated = build_model("tcn_se").eval()
        plain = build_model("tcn").eval()
        plain.load_state_dict({k: v for k, v in gated.state_dict().items()
                               if ".se." not in k}, strict=True)
        # Same convolutions, so if the gate were dropped the two would agree.
        self.assertGreater((IncrementalModel(gated).run(x)
                            - IncrementalModel(plain).run(x)).abs().max().item(), 1e-3,
                           "the streamed gated model matched the ungated one: SE was skipped")

    def test_the_running_summary_is_the_whole_state(self):
        """Two numbers per channel, against a 512-position cache for attention."""
        se = IncrementalModel(build_model("tcn_se").eval()).cache_report()
        attn = IncrementalModel(build_model("tcn_mla").eval()).cache_report()
        self.assertEqual(se["attn_cache_bytes"], 0)
        self.assertLess(se["total_state_bytes"], attn["total_state_bytes"])

    def test_a_non_causal_gate_is_refused(self):
        with self.assertRaises(ValueError):
            IncrementalModel(build_model("tcn_se_bi").eval())


class SlidingWindowEquivalenceTests(unittest.TestCase):
    """
    Stream vs. the hop-and-recompute path in ``deploy/raspi_infer.py``.

    The baseline agrees to float noise.  The attention variants do not, and the
    module docstring says why: attention at the window's last position attends
    over 512 cached positions whose oldest ~252 were, in the windowed path,
    computed from the window's left zero-padding.  The bound is asserted so
    that the known mismatch stays small and stays bounded.
    """

    def _worst_relative_gap(self, variant, n=1200, probes=(512, 700, 1000, 1199)):
        model = build_model(variant).eval()
        stream = IncrementalModel(model)
        x = _signal(n, seed=4)
        streamed = stream.run(x)
        scale = streamed.std().item()
        worst = 0.0
        with torch.no_grad():
            for t in probes:
                last = model(x[t - WINDOW + 1:t + 1].view(1, 1, WINDOW))[0, 0, -1]
                worst = max(worst, abs(last - streamed[t]).item() / scale)
        return worst

    def test_baseline_matches_sliding_window_exactly(self):
        """No attention, receptive field 125 < 512: nothing mixes in the padding."""
        self.assertLess(self._worst_relative_gap("tcn"), 1e-6)

    def test_attention_sliding_window_gap_is_small_and_bounded(self):
        """
        A "stays small" check, not a precision claim, so the bound is loose on
        purpose.  How much of the gap survives depends on how much attention
        weight the trained model puts on the oldest cached positions, and that
        varies: these randomly initialised models land near 5e-4, while the
        trained ``runs/tcn_mha_eog`` checkpoint reaches 2.8e-3.  The bound has
        to cover both without being so tight that a different torch version or
        init reshuffles it into a failure.

        Even the largest of those is two orders of magnitude below the ~0.65
        RRMSE these models actually achieve, so it cannot move a reported
        metric.  The assertions that would catch a genuine state bug are the
        exact ones: the baseline test above, the whole-signal equivalence
        tests, and ``test_gap_does_not_grow_with_time``.
        """
        for variant in ("tcn_mha", "tcn_mla", "tcn_mla_par"):
            with self.subTest(variant=variant):
                gap = self._worst_relative_gap(variant)
                self.assertLess(gap, 1e-2,
                                "the known window/stream mismatch grew beyond its bound")

    def test_gap_does_not_grow_with_time(self):
        """A mismatch that accumulated would make long streams unusable."""
        early = self._worst_relative_gap("tcn_mla", n=1200, probes=(512, 600, 700))
        late = self._worst_relative_gap("tcn_mla", n=4000, probes=(3700, 3850, 3999))
        self.assertLess(late, max(10 * early, 1e-4),
                        "the window/stream gap grows with t: state is drifting")


class StreamingCausalityTests(unittest.TestCase):
    """
    The stream must not leak the future either.

    ``tests/test_variants.py`` proves this for the batch models by perturbing a
    sample and watching the outputs before it.  A cache bug — a stale ring
    slot, an off-by-one write pointer — could reintroduce a leak on the
    streaming path alone, so the same probe is run through it.
    """

    def test_future_samples_cannot_change_past_outputs(self):
        split = 128
        x1 = _signal(256, seed=5)
        x2 = x1.clone()
        x2[split] += 10.0
        for variant in STREAMABLE:
            with self.subTest(variant=variant):
                model = build_model(variant).eval()
                y1 = IncrementalModel(model).run(x1)
                y2 = IncrementalModel(model).run(x2)
                before = (y1[:split] - y2[:split]).abs().max().item()
                at_and_after = (y1[split:] - y2[split:]).abs().max().item()
                self.assertEqual(before, 0.0,
                                 "a later sample changed an earlier output: the stream leaks")
                self.assertGreater(at_and_after, 0.0,
                                   "the perturbation changed nothing; the probe is not probing")

    def test_bidirectional_models_are_refused(self):
        for variant in ("tcn_mla_bi", "tcn_bi", "tcn_mha_full_bi"):
            with self.subTest(variant=variant):
                model = build_model(variant).eval()
                with self.assertRaises(ValueError):
                    IncrementalModel(model)


class MLACacheTests(unittest.TestCase):
    """MLA's two cache strategies, and the compression they are chosen for."""

    def test_absorbed_matches_decompressed(self):
        """
        Projection absorption is algebra, not an approximation.

        ``absorb=True`` folds the key up-projection into the query and the
        value up-projection past the weighted sum, so K and V are never built.
        It must agree with decompressing the latent cache the obvious way.
        """
        x = _signal(400, seed=6)
        for variant in ("tcn_mla", "tcn_mla_par", "tcn_mla_r8"):
            with self.subTest(variant=variant):
                model = build_model(variant).eval()
                absorbed = IncrementalModel(model, absorb=True).run(x)
                naive = IncrementalModel(model, absorb=False).run(x)
                self.assertLess((absorbed - naive).abs().max().item(), 1e-5)

    def test_absorbed_matches_batch_forward(self):
        """Both MLA paths are checked against the batch model, not just each other."""
        x = _signal(300, seed=7)
        model = build_model("tcn_mla").eval()
        with torch.no_grad():
            ref = model(x.view(1, 1, -1))[0, 0]
        for absorb in (True, False):
            with self.subTest(absorb=absorb):
                got = IncrementalModel(model, absorb=absorb).run(x)
                self.assertLess((ref - got).abs().max().item(), 1e-5)

    def test_mla_caches_the_latent_not_kv(self):
        """
        The headline number: per step MLA holds ``kv_lora_rank`` floats per
        layer where MHA holds ``2 * d_model``.
        """
        mha = IncrementalModel(build_model("tcn_mha").eval())
        mla = IncrementalModel(build_model("tcn_mla").eval())

        d_model = mha.model.decoder.blocks[0].attn.d_model
        rank = mla.model.decoder.blocks[0].attn.kv_lora_rank
        layers = len(mha.model.decoder.blocks)

        self.assertEqual(mha.cache_report()["attn_floats_per_step"], 2 * d_model * layers)
        self.assertEqual(mla.cache_report()["attn_floats_per_step"], rank * layers)
        self.assertLess(mla.cache_report()["attn_cache_bytes"],
                        mha.cache_report()["attn_cache_bytes"],
                        "MLA's cache was not smaller than MHA's")

    def test_compression_ratio_against_mha(self):
        """
        The cache ratio is ``2 * d_model / kv_lora_rank``, because MHA has to
        keep both K and V while MLA keeps one latent.

        ``d_model`` is 32 here, not 64: the sandwich squeezes 64 TCN channels
        by ``bottleneck_ratio`` 0.5 before attention.  So the constructor
        default ``d_model // 4`` is rank **8**, giving 8x, while the rank 16
        that every trained checkpoint and every YAML preset actually sets gives
        4x — the number the README reports.

        Both are asserted, because the two were conflated in the docstrings:
        ``kv_lora_rank = d_model // 4`` was described as "4x compression", which
        is the ratio for rank 16 rather than for the default it was attached to.
        """
        mha = IncrementalModel(build_model("tcn_mha").eval()).cache_report()

        default = IncrementalModel(build_model("tcn_mla").eval()).cache_report()
        self.assertEqual(default["attn_floats_per_step"] // 2, 8,
                         "the constructor default is d_model // 4 = 32 // 4 = 8")
        self.assertAlmostEqual(
            mha["attn_cache_bytes"] / default["attn_cache_bytes"], 8.0, places=6)

        trained = IncrementalModel(build_model("tcn_mla", kv_lora_rank=16).eval()).cache_report()
        self.assertAlmostEqual(
            mha["attn_cache_bytes"] / trained["attn_cache_bytes"], 4.0, places=6,
            msg="rank 16 is what the checkpoints use; its ratio is the reported 4x")

    def test_rank_sets_the_cache_size(self):
        """A rank sweep must actually move the cache, or the sweep measures nothing."""
        sizes = {}
        for rank_variant in ("tcn_mla_r4", "tcn_mla_r8", "tcn_mla_r16"):
            report = IncrementalModel(build_model(rank_variant).eval()).cache_report()
            sizes[rank_variant] = report["attn_floats_per_step"]
        self.assertLess(sizes["tcn_mla_r4"], sizes["tcn_mla_r8"])
        self.assertLess(sizes["tcn_mla_r8"], sizes["tcn_mla_r16"])


class CacheGeometryTests(unittest.TestCase):
    """The reported state has to describe the model, not a guess about it."""

    def test_window_comes_from_the_trained_mask(self):
        model = build_model("tcn_mla", max_seq_len=256).eval()
        self.assertEqual(IncrementalModel(model).attention_window, 256)

    def test_baseline_has_no_attention_cache(self):
        report = IncrementalModel(build_model("tcn").eval()).cache_report()
        self.assertEqual(report["attn_cache_bytes"], 0)
        self.assertEqual(report["attn_floats_per_step"], 0)
        self.assertGreater(report["conv_state_bytes"], 0)

    def test_receptive_field_matches_a_perturbation_probe(self):
        """
        The reported receptive field is used to set ``warmup``; measure it
        instead of trusting the arithmetic.  Perturb one sample of the
        baseline and see how far the influence reaches.
        """
        # Seeded: build_model draws from the global RNG, so the weights differ
        # with test ordering. With an unlucky draw a tap's contribution falls
        # under the threshold below and the measured reach comes up short --
        # the test failed in a full run and passed on its own.
        torch.manual_seed(1234)
        model = build_model("tcn").eval()
        n, hit = 400, 100
        x1 = _signal(n, seed=8)
        x2 = x1.clone()
        x2[hit] += 10.0
        y1 = IncrementalModel(model).run(x1)
        y2 = IncrementalModel(model).run(x2)
        # Anything that moved at all: a strict threshold measures the weights'
        # magnitudes rather than the receptive field.
        touched = (y1 - y2).abs() > 0
        reach = int(touched.nonzero().max()) - hit + 1
        self.assertEqual(reach, IncrementalModel(model).receptive_field)

    def test_cache_bytes_scale_with_batch_size(self):
        stream = IncrementalModel(build_model("tcn_mla").eval())
        stream.reset(batch_size=1)
        one = stream.cache_report()["total_state_bytes"]
        stream.reset(batch_size=8)
        eight = stream.cache_report()["total_state_bytes"]
        self.assertEqual(eight, 8 * one)

    def test_step_rejects_a_batch_it_was_not_reset_for(self):
        stream = IncrementalModel(build_model("tcn").eval())
        stream.reset(batch_size=2)
        with self.assertRaises(ValueError):
            stream.step(torch.zeros(5))


class TurboQuantRefusalTests(unittest.TestCase):
    """
    TurboQuant fake-quantises K and V inside a batch forward pass.  Streaming
    would quantise a growing cache once per step, which is a different
    computation — so it is refused rather than silently approximated.
    """

    def test_turboquant_models_are_refused(self):
        for variant in ("tcn_mha", "tcn_mla"):
            with self.subTest(variant=variant):
                model = build_model(variant, turboquant=True).eval()
                with self.assertRaises(ValueError):
                    IncrementalModel(model)


if __name__ == "__main__":
    unittest.main()
