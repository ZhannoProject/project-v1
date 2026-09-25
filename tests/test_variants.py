"""Causality is the one property these variants exist to trade away.

Run: PYTHONPATH=. python tests/test_variants.py

Every deployable variant must be unable to see the future, and every ``*_bi``
teacher must actually be able to — a teacher that is secretly causal teaches
the student nothing it does not already know, and a "causal" model that leaks
one sample ahead cannot run online at all.  Neither failure shows up in the
loss, so it is checked directly: perturb one sample and look at the outputs
before it.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.TCN_backbone import CausalConv1d, CausalTCNBackbone
from models.variants import (BIDIRECTIONAL_VARIANTS, VARIANT_REGISTRY, build_model,
                             base_variant, variant_is_causal)

B, T, SPLIT = 2, 512, 256


def leak(model) -> float:
    """Largest change before SPLIT when the sample at SPLIT changes."""
    torch.manual_seed(0)
    x1 = torch.randn(B, 1, T)
    x2 = x1.clone()
    x2[:, :, SPLIT] += 10.0
    model.eval()
    with torch.no_grad():
        return (model(x1)[:, :, :SPLIT] - model(x2)[:, :, :SPLIT]).abs().max().item()


class CausalityTests(unittest.TestCase):
    def test_every_deployable_variant_is_causal(self):
        for name in VARIANT_REGISTRY:
            if not variant_is_causal(name):
                continue
            with self.subTest(variant=name):
                self.assertLess(leak(build_model(name, hidden_channels=64)), 1e-6)

    def test_every_bi_variant_sees_the_future(self):
        for name in BIDIRECTIONAL_VARIANTS:
            with self.subTest(variant=name):
                self.assertGreater(leak(build_model(name, hidden_channels=64)), 0.0)

    def test_tcn_bi_is_non_causal_in_the_convolutions(self):
        """The attention twins keep causal convolutions; tcn_bi does not, which
        is the only way a model with no attention can see ahead."""
        self.assertIn("tcn_bi", BIDIRECTIONAL_VARIANTS)
        self.assertEqual(base_variant("tcn_bi"), "tcn")
        self.assertFalse(build_model("tcn_bi", hidden_channels=64).backbone.causal)
        self.assertTrue(build_model("tcn_mla_bi", hidden_channels=64).tcn_pt1.causal)

    def test_a_twin_is_the_same_model_with_different_padding(self):
        """Same shapes both ways, or the teacher cannot initialise the student."""
        for causal_name in ("tcn", "tcn_mla", "tcn_mla_par"):
            with self.subTest(variant=causal_name):
                a = build_model(causal_name, hidden_channels=64).state_dict()
                b = build_model(causal_name + "_bi", hidden_channels=64).state_dict()
                self.assertEqual({k: v.shape for k, v in a.items()},
                                 {k: v.shape for k, v in b.items()})

    def test_the_name_and_the_flag_cannot_disagree(self):
        with self.assertRaisesRegex(ValueError, "tcn_bi"):
            build_model("tcn", hidden_channels=64, causal=False)
        with self.assertRaisesRegex(ValueError, "contradicts"):
            build_model("tcn_bi", hidden_channels=64, causal=True)


class PaddingTests(unittest.TestCase):
    def test_padding_moves_but_never_changes_the_output_length(self):
        for kernel, dilation in ((3, 1), (3, 4), (5, 2), (1, 1)):
            for causal in (True, False):
                with self.subTest(kernel=kernel, dilation=dilation, causal=causal):
                    conv = CausalConv1d(1, 4, kernel, dilation, causal=causal)
                    total = (kernel - 1) * dilation
                    self.assertEqual(conv.pad_left + conv.pad_right, total)
                    self.assertEqual(conv.pad_right, 0 if causal else total - total // 2)
                    self.assertEqual(conv(torch.randn(1, 1, 64)).shape[-1], 64)

    def test_the_backbone_records_which_form_it_is(self):
        self.assertTrue(CausalTCNBackbone(1, 8, [1, 2]).causal)
        self.assertFalse(CausalTCNBackbone(1, 8, [1, 2], causal=False).causal)


if __name__ == "__main__":
    unittest.main()
