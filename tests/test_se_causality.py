"""
Is the SE squeeze causal *at training time*, and is it the same path as at
inference?

Everything the project already asserts about causality is asserted in
``eval()`` -- ``test_variants.py``'s perturbation probe, the lookahead
ladder's reach measurements, the ONNX exporter's refusal. That is the right
mode to deploy in and the wrong mode to stop at, because a gate that pooled
globally while training and ran as a running mean afterwards would pass every
one of those tests and still have been trained on the future. The whole
``tcn_se`` ladder, and the project's headline result, sits on this gate.

So this file re-asks the question in ``train()``, and asks it two ways: by
perturbation (does an output move when a later input moves?) and by gradient
(is d out_t / d in_s exactly zero for s > t?). The second is the one that
matters for training, because gradients are what training is made of.

It also pins the one place where the historical model is not causal in
train mode, and shows it is not the gate: ``BatchNorm1d`` normalises over
(batch, time), so its batch statistics couple every position to every other.
That is ordinary, it disappears in ``eval()`` where the statistics are frozen
constants, and ``tests/test_lookahead.py`` already documents the probe's side
of it. What is asserted here is that it is the *only* such coupling.

Run: python -m unittest discover -s tests -p 'test_se_causality.py'
"""

import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.TCN_backbone import CausalSE, ResidualBlock          # noqa: E402
from models.variants import build_model                          # noqa: E402

C, T = 16, 64
T0 = T // 2                      # the sample the probes perturb


def _x(seed=0, b=2):
    return torch.randn(b, C, T, generator=torch.Generator().manual_seed(seed))


def _se(seed=0, **kw):
    """
    A gate built under a fixed seed.

    Not fussiness: ``nn.Conv1d`` draws its init from the global RNG, so an
    unseeded gate depends on how many tests ran before it, and a gate whose
    ReLU happens to land dead carries no gradient through the squeeze at all
    -- which would let a *global pool* pass a test meant to catch one. The
    README describes the same dead-ReLU effect in the lookahead ladder's
    ``measured`` column.
    """
    torch.manual_seed(seed)
    return CausalSE(C, **kw)


def _gate_varies_in_time(se, x) -> float:
    """
    How much the gate changes across t. Zero means the squeeze is reaching
    the excitation as a constant, so nothing but ``x_t`` can affect
    ``out_t`` and a causality test over the gate would be vacuous.
    """
    with torch.no_grad():
        g = torch.sigmoid(se.fc2(se.act(se.fc1(se.squeeze(x)))))
    return float(g.std())


def _perturbed(x, t0=T0, amount=5.0):
    y = x.clone()
    y[:, :, t0] += amount
    return y


def _past_future(a, b, t0=T0):
    """How far a perturbation at t0 moved the output, before and after it."""
    a, b = a.detach(), b.detach()
    return (float((a[..., :t0] - b[..., :t0]).abs().max()),
            float((a[..., t0:] - b[..., t0:]).abs().max()))


# ============================================================
# The squeeze is the cumulative mean, in either mode
# ============================================================
class TestSqueezeIsACumulativeMean(unittest.TestCase):

    def test_it_matches_the_formula_in_both_modes(self):
        se = _se(reduction=4)
        x = _x()
        want = x.cumsum(dim=2) / torch.arange(1, T + 1, dtype=x.dtype).view(1, 1, -1)
        for training in (True, False):
            se.train(training)
            with self.subTest(training=training):
                torch.testing.assert_close(se.squeeze(x), want)

    def test_position_t_uses_nothing_after_t(self):
        # Directly: the squeeze at t computed on the full sequence must equal
        # the squeeze at t computed on the prefix [0..t] alone.
        se = _se(reduction=4).train()
        x = _x()
        full = se.squeeze(x)
        for t in (0, 1, 7, T // 3, T - 1):
            prefix = se.squeeze(x[:, :, : t + 1])
            torch.testing.assert_close(full[:, :, t], prefix[:, :, t])


# ============================================================
# Train/eval parity: the SE path cannot differ, and does not
# ============================================================
class TestTrainEvalParity(unittest.TestCase):

    def test_the_module_holds_nothing_that_behaves_differently_when_training(self):
        # Structural, so a future edit that adds a BatchNorm or a Dropout
        # inside the gate fails here rather than silently in a run.
        se = _se(reduction=4)
        for name, m in se.named_modules():
            self.assertNotIsInstance(m, (nn.BatchNorm1d, nn.Dropout, nn.Dropout1d), name)

    def test_the_gate_is_bit_identical_in_train_and_eval(self):
        se = _se(reduction=4)
        x = _x(1)
        se.train()
        a = se(x)
        se.eval()
        b = se(x)
        self.assertTrue(torch.equal(a, b),
                        "the SE gate behaves differently while training")

    def test_parity_holds_for_every_pooling_setting(self):
        for kw in ({"causal": True}, {"causal": False},
                   {"causal": False, "lookahead": 0},
                   {"causal": False, "lookahead": 8}):
            se = _se(reduction=4, **kw)
            x = _x(2)
            se.train()
            a = se(x)
            se.eval()
            with self.subTest(**kw):
                self.assertTrue(torch.equal(a, se(x)))


# ============================================================
# Causality in train mode, by perturbation
# ============================================================
class TestPerturbationInTrainMode(unittest.TestCase):

    def test_the_causal_gate_does_not_move_an_earlier_output(self):
        se = _se(reduction=4, causal=True).train()
        x = _x(3)
        past, future = _past_future(se(x), se(_perturbed(x)))
        self.assertEqual(past, 0.0, "the SE gate read ahead while training")
        self.assertGreater(future, 0.0, "the perturbation did nothing; the probe is idle")

    def test_the_squeeze_alone_does_not_move_an_earlier_position(self):
        se = _se(reduction=4, causal=True).train()
        x = _x(4)
        past, future = _past_future(se.squeeze(x), se.squeeze(_perturbed(x)))
        self.assertEqual(past, 0.0)
        self.assertGreater(future, 0.0)

    def test_the_global_pool_does_move_an_earlier_output(self):
        # The control. Without it, a past_diff of 0 above would say nothing
        # about the probe -- this is the gate the *_bi teachers use, and it
        # must be caught.
        se = _se(reduction=4, causal=False).train()
        x = _x(5)
        past, _ = _past_future(se(x), se(_perturbed(x)))
        self.assertGreater(past, 0.0,
                           "the global pool should leak the future into every position")

    def test_a_bounded_gate_peeks_exactly_its_budget_while_training(self):
        for L in (1, 4, 11):
            se = _se(reduction=4, causal=False, lookahead=L).train()
            x = _x(6)
            a, b = se.squeeze(x), se.squeeze(_perturbed(x))
            moved = (a - b).abs().amax(dim=(0, 1))          # per position
            first = int(torch.nonzero(moved > 0)[0]) if (moved > 0).any() else T
            with self.subTest(lookahead=L):
                self.assertEqual(first, T0 - L,
                                 f"a gate licensed {L} ahead reached {T0 - first}")


# ============================================================
# Causality in train mode, by gradient
# ============================================================
class TestGradientInTrainMode(unittest.TestCase):
    """
    The decisive form of the question. Training moves weights along
    d loss / d theta, and that path runs through d out_t / d in_s. If any
    future input had a non-zero derivative into an earlier output, the gate
    would be *trained* on the future whatever the forward pass looked like.
    """

    def _future_sensitivity(self, module, t=T0):
        x = _x(7, b=1).requires_grad_(True)
        if isinstance(module, CausalSE):
            assert _gate_varies_in_time(module, _x(7, b=1)) > 1e-6, \
                "the gate is constant in t, so this test would prove nothing"
        out = module(x)
        out[0, :, t].sum().backward()
        g = x.grad[0].abs().sum(dim=0)               # per input position
        return float(g[t + 1:].max()), float(g[: t + 1].max())

    def test_no_future_input_has_a_gradient_into_an_earlier_output(self):
        se = _se(seed=1, reduction=4, causal=True).train()
        future, past = self._future_sensitivity(se)
        self.assertEqual(future, 0.0, "a future sample has a gradient path into out_t")
        self.assertGreater(past, 0.0, "nothing had a gradient; the test is vacuous")

    def test_the_global_pool_does_have_one(self):
        se = _se(seed=1, reduction=4, causal=False).train()
        future, _ = self._future_sensitivity(se)
        self.assertGreater(future, 0.0)

    def test_a_residual_block_with_the_gate_is_causal_by_gradient(self):
        block = ResidualBlock(C, C, 3, dilation=2, dropout=0.0, se_reduction=4).train()
        future, past = self._future_sensitivity(block)
        self.assertEqual(future, 0.0)
        self.assertGreater(past, 0.0)


# ============================================================
# The whole model, and what the one train-mode coupling is
# ============================================================
class TestWholeModelInTrainMode(unittest.TestCase):

    def _model(self, **kwargs):
        m = build_model("tcn_se", hidden_channels=16, **kwargs)
        for mod in m.modules():                       # dropout would mask the probe
            if isinstance(mod, (nn.Dropout, nn.Dropout1d)):
                mod.p = 0.0
        return m

    def test_in_eval_it_is_exactly_causal(self):
        m = self._model().eval()
        x = torch.randn(2, 1, T, generator=torch.Generator().manual_seed(8))
        past, future = _past_future(m(x), m(_perturbed(x)))
        self.assertEqual(past, 0.0)
        self.assertGreater(future, 0.0)

    def test_in_train_mode_batchnorm_is_the_only_thing_that_couples_time(self):
        # Train mode everywhere -> the probe is non-zero, because BatchNorm
        # normalises over (batch, time). Put *only* the BatchNorms back in
        # eval, leave the rest training, and it is exactly causal again. So
        # the coupling is BatchNorm's, not the gate's.
        x = torch.randn(2, 1, T, generator=torch.Generator().manual_seed(9))

        m = self._model(tcn_norm="batch").train()
        past_all_train, _ = _past_future(m(x), m(_perturbed(x)))
        self.assertGreater(past_all_train, 0.0,
                           "expected BatchNorm's batch statistics to couple time")

        m = self._model(tcn_norm="batch").train()
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm1d):
                mod.eval()
        past_bn_eval, future = _past_future(m(x), m(_perturbed(x)))
        self.assertEqual(past_bn_eval, 0.0,
                         "something other than BatchNorm couples time in train mode")
        self.assertGreater(future, 0.0)

    def test_default_model_is_causal_in_train_mode(self):
        model = self._model().train()
        self.assertFalse(any(isinstance(layer, nn.BatchNorm1d) for layer in model.modules()))
        x = torch.randn(2, 1, T, generator=torch.Generator().manual_seed(11))
        past, future = _past_future(model(x), model(_perturbed(x)))
        self.assertEqual(past, 0.0)
        self.assertGreater(future, 0.0)

    def test_the_bidirectional_twin_is_caught_in_train_mode_too(self):
        m = build_model("tcn_se_bi", hidden_channels=16).train()
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm1d):
                mod.eval()
            if isinstance(mod, (nn.Dropout, nn.Dropout1d)):
                mod.p = 0.0
        x = torch.randn(2, 1, T, generator=torch.Generator().manual_seed(10))
        past, _ = _past_future(m(x), m(_perturbed(x)))
        self.assertGreater(past, 0.0, "the non-causal twin should still read ahead")


if __name__ == "__main__":
    unittest.main()
