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

from models.TCN_backbone import CausalConv1d, CausalSE, CausalTCNBackbone
from models.variants import (BIDIRECTIONAL_VARIANTS, FULLY_BIDIRECTIONAL_VARIANTS,
                             VARIANT_REGISTRY, build_model, base_variant, resolved_arch,
                             variant_is_causal)

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

    def test_full_bi_is_non_causal_everywhere_and_bi_is_not(self):
        """The two axes must stay separable: X_bi looks ahead only through
        attention, X_full_bi looks ahead in the convolutions too.  If the TCN
        stages ever stopped being causal in X_bi, the control it provides
        would be gone and nothing else would notice."""
        for full in FULLY_BIDIRECTIONAL_VARIANTS:
            attn_only = full.replace("_full_bi", "_bi")
            with self.subTest(variant=full):
                a, b = build_model(attn_only, hidden_channels=64), build_model(full, hidden_channels=64)
                self.assertTrue(a.causal_tcn)
                self.assertFalse(b.causal_tcn)
                self.assertFalse(a.causal)
                self.assertFalse(b.causal)
                self.assertTrue(a.tcn_pt1.causal and a.tcn_pt2.causal)
                self.assertFalse(b.tcn_pt1.causal or b.tcn_pt2.causal)
                # Both see ahead, but unmasking every layer leaks far more than
                # unmasking attention alone — orders of magnitude, not a margin.
                self.assertGreater(leak(b), 100 * leak(a))

    def test_tcn_has_no_full_bi_because_the_axes_collapse(self):
        """With no attention there is only one thing to make non-causal, so
        tcn_bi already is what tcn_full_bi would be."""
        self.assertNotIn("tcn_full_bi", VARIANT_REGISTRY)
        self.assertEqual([v for v in FULLY_BIDIRECTIONAL_VARIANTS if base_variant(v) == "tcn"], [])

    def test_a_twin_is_the_same_model_with_different_padding(self):
        """Same shapes both ways, or the teacher cannot initialise the student."""
        for causal_name in ("tcn", "tcn_mla", "tcn_mla_par"):
            with self.subTest(variant=causal_name):
                shapes = []
                for suffix in ("", "_bi", "_full_bi"):
                    name = causal_name + suffix
                    if name not in VARIANT_REGISTRY:
                        continue
                    sd = build_model(name, hidden_channels=64).state_dict()
                    shapes.append({k: v.shape for k, v in sd.items()})
                for other in shapes[1:]:
                    self.assertEqual(shapes[0], other)

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




class RecordedArchitectureTests(unittest.TestCase):
    """
    A run has to record the shape of the thing it trained, not the config that
    was meant to build it.

    This project has the scar.  Every MLA checkpoint here was trained with 4
    attention layers and 2 heads while its YAML said 2 and 4, so the headline
    MHA/MLA ablation differs in depth, head count and parameter count — and
    nothing in any run recorded either number, because the trainer's history
    filtered them out and ``model_config`` holds only what was explicitly
    passed.  ``resolved_arch`` closes both halves of that.
    """

    def test_defaults_are_resolved_not_omitted(self):
        """
        ``model_config`` records only what was passed, so a run that took a
        default recorded nothing — and if a default later moves, that run
        becomes ambiguous in hindsight.
        """
        model = build_model("tcn")
        self.assertNotIn("hidden_channels", model.model_config)
        self.assertEqual(resolved_arch(model)["hidden_channels"], 64)

    def test_it_records_what_was_passed_over_the_default(self):
        arch = resolved_arch(build_model("tcn_mla", hidden_channels=48, n_attn_layers=4,
                                         n_heads=2, kv_lora_rank=16))
        self.assertEqual((arch["hidden_channels"], arch["n_attn_layers"],
                          arch["n_heads"], arch["kv_lora_rank"]), (48, 4, 2, 16))

    def test_a_pinned_rank_wins_over_a_passed_one(self):
        """``tcn_mla_r8`` means rank 8 whatever the config says alongside it."""
        self.assertEqual(resolved_arch(build_model("tcn_mla_r8"))["kv_lora_rank"], 8)

    def test_it_distinguishes_the_bidirectional_twins(self):
        """
        A teacher and its causal student share a state dict, so the weights
        cannot tell them apart — only the recorded architecture can.
        """
        self.assertIs(resolved_arch(build_model("tcn_mla"))["causal"], True)
        self.assertIs(resolved_arch(build_model("tcn_mla_bi"))["causal"], False)
        full = resolved_arch(build_model("tcn_mla_full_bi"))
        self.assertIs(full["causal"], False)
        self.assertIs(full["causal_tcn"], False)

    def test_it_separates_the_pair_that_went_unnoticed(self):
        """
        The concrete regression: a 2-layer/4-head MHA and a 4-layer/2-head MLA
        must not produce records that look like the same experiment with one
        variable changed.
        """
        mha = resolved_arch(build_model("tcn_mha", hidden_channels=64,
                                        n_attn_layers=2, n_heads=4))
        mla = resolved_arch(build_model("tcn_mla", hidden_channels=64,
                                        n_attn_layers=4, n_heads=2, kv_lora_rank=16))
        differing = [k for k in ("n_attn_layers", "n_heads")
                     if mha.get(k) != mla.get(k)]
        self.assertEqual(sorted(differing), ["n_attn_layers", "n_heads"],
                         "the record no longer shows the depth/head difference")


class CausalSETests(unittest.TestCase):
    """
    Squeeze-and-Excitation has to be rebuilt to be usable here.

    Standard SE averages over the whole sequence, so the gate applied at t=0
    depends on the last sample of the epoch.  That is a future leak through
    the channel weights while every convolution around it stays causal, and
    it would not show up in the loss -- only later, as a model that cannot be
    deployed.  The running-average squeeze is the fix, and these are the
    checks that it is actually a fix rather than a rename.
    """

    def test_the_squeeze_is_a_running_mean(self):
        se = CausalSE(8, reduction=2).eval()
        x = torch.randn(2, 8, 32)
        s = se.squeeze(x)
        for t in (0, 1, 7, 31):
            with self.subTest(t=t):
                self.assertTrue(torch.allclose(s[:, :, t], x[:, :, :t + 1].mean(dim=2),
                                               atol=1e-6))

    def test_the_causal_gate_cannot_see_ahead(self):
        torch.manual_seed(0)
        se = CausalSE(16, reduction=4, causal=True).eval()
        x1 = torch.randn(2, 16, 128)
        x2 = x1.clone()
        x2[:, :, 64] += 10.0
        with torch.no_grad():
            y1, y2 = se(x1), se(x2)
        self.assertEqual((y1[:, :, :64] - y2[:, :, :64]).abs().max().item(), 0.0,
                         "the SE gate leaked a later sample into an earlier output")
        self.assertGreater((y1[:, :, 64:] - y2[:, :, 64:]).abs().max().item(), 0.0,
                           "the perturbation changed nothing; the probe is not probing")

    def test_the_non_causal_gate_does_see_ahead(self):
        """
        The teacher's pooling is the standard global one. If this ever stopped
        leaking, the causal check above would be passing for free.
        """
        torch.manual_seed(0)
        se = CausalSE(16, reduction=4, causal=False).eval()
        x1 = torch.randn(2, 16, 128)
        x2 = x1.clone()
        x2[:, :, 64] += 10.0
        with torch.no_grad():
            past = (se(x1)[:, :, :64] - se(x2)[:, :, :64]).abs().max().item()
        self.assertGreater(past, 0.0)

    def test_both_poolings_share_one_parameter_set(self):
        """A teacher and its causal twin must still be able to swap weights."""
        a = CausalSE(16, reduction=4, causal=True)
        b = CausalSE(16, reduction=4, causal=False)
        self.assertEqual(list(a.state_dict()), list(b.state_dict()))
        b.load_state_dict(a.state_dict())

    def test_the_variant_is_causal_end_to_end(self):
        model = build_model("tcn_se").eval()
        self.assertEqual(leak(model), 0.0, "tcn_se leaked the future")
        self.assertGreater(leak(build_model("tcn_se_bi").eval()), 0.0,
                           "tcn_se_bi should see ahead; it is a teacher")

    def test_se_is_off_unless_asked_for(self):
        """
        The baseline has to be untouched, or every number already measured
        with it would silently belong to a different model.
        """
        plain = build_model("tcn")
        self.assertTrue(all(b.se is None for b in plain.backbone.network))
        self.assertEqual(list(plain.state_dict()),
                         list(build_model("tcn", se_reduction=None).state_dict()))
        gated = build_model("tcn_se")
        self.assertTrue(all(b.se is not None for b in gated.backbone.network))

    def test_it_costs_a_fraction_of_what_attention_costs(self):
        """The reason to try it at all, asserted so it stays true."""
        n = lambda v: sum(p.numel() for p in build_model(v).parameters())
        base, se, mla = n("tcn"), n("tcn_se"), n("tcn_mla")
        self.assertLess(se - base, 0.1 * base, "SE should be a few percent, not a redesign")
        self.assertLess((se - base) * 5, mla - base)


if __name__ == "__main__":
    unittest.main()
