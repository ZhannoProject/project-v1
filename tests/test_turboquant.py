"""Run: python -m unittest discover -s tests -p 'test_turboquant.py'."""
import contextlib
import io
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from models.AttentionBlocks.turboquant import TurboQuant, _pack, _unpack
from models.variants import build_model, build_model_from_config
from trainer.checkpoint import save_checkpoint

ROOT = Path(__file__).resolve().parents[1]


class TurboQuantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_bit_packing(self):
        for bits in range(1, 9):
            codes = torch.randint(0, 2**bits, (2, 3, 13))
            packed = _pack(codes, bits)
            self.assertEqual(packed.shape[-1], (13 * bits + 7) // 8)
            self.assertTrue(torch.equal(codes, _unpack(packed, 13, bits)))

    def test_zero_and_shapes(self):
        for mode in ('mse', 'prod'):
            for bits in (2, 3, 4, 8):
                codec = TurboQuant(8, bits, mode)
                x = torch.zeros(2, 4, 9, 8)
                self.assertTrue(torch.equal(codec.decode(codec.encode(x)), x))
                x = torch.randn_like(x)
                encoded = codec.encode(x)
                self.assertEqual(codec.decode(encoded).shape, x.shape)
                self.assertTrue(torch.isfinite(codec.decode(encoded)).all())
                self.assertGreater(encoded.nbytes, 0)

    def test_distortion_and_rng(self):
        torch.manual_seed(7)
        x = torch.randn(1000, 16)
        state = torch.random.get_rng_state()
        a = TurboQuant(16, 3, 'mse', 17)
        b = TurboQuant(16, 3, 'mse', 17)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(a(x), b(x)))
        error3 = (a(x) - x).square().mean() / x.square().mean()
        error4 = (TurboQuant(16, 4, 'mse', 17)(x) - x).square().mean() / x.square().mean()
        self.assertLess(error3, .06)
        self.assertLess(error4, error3)
        print(f'MSE distortion: 3-bit={error3:.4f}, 4-bit={error4:.4f}')

    def test_qjl_unbiased_over_seeds(self):
        x = torch.arange(1, 9, dtype=torch.float32).reshape(1, 8)
        x = x / x.norm()
        estimates = torch.stack([TurboQuant(8, 3, 'prod', seed)(x) for seed in range(300)])
        self.assertLess((estimates.mean(0) - x).norm().item(), .065)

    def test_disabled_compatibility_and_parameters(self):
        for variant in ('tcn_mha', 'tcn_mla', 'tcn_mha_par', 'tcn_mla_par'):
            torch.manual_seed(5)
            plain = build_model(variant).eval()
            torch.manual_seed(5)
            disabled = build_model(variant, turboquant=False).eval()
            x = torch.randn(1, 1, 16)
            with torch.no_grad():
                self.assertTrue(torch.equal(plain(x), disabled(x)))
            enabled = build_model(variant, turboquant=True)
            enabled.load_state_dict(plain.state_dict(), strict=True)
            self.assertEqual(sum(p.numel() for p in plain.parameters()),
                             sum(p.numel() for p in enabled.parameters()))
            self.assertEqual(set(plain.state_dict()), set(enabled.state_dict()))
            # Same weights, no quantization during normal training.
            torch.manual_seed(15)
            y1 = plain.train()(x)
            torch.manual_seed(15)
            y2 = enabled.train()(x)
            self.assertTrue(torch.equal(y1, y2))

    def test_all_architectures_depths_and_causality(self):
        for variant in ('tcn_mha', 'tcn_mla', 'tcn_mha_par', 'tcn_mla_par'):
            for depth in (10, 20, 50):
                model = build_model(variant, n_attn_layers=depth, turboquant=True).eval()
                x = torch.randn(1, 1, 16)
                altered = x.clone()
                altered[..., 8:] += 5
                with torch.no_grad():
                    y, z = model(x), model(altered)
                self.assertEqual(y.shape, x.shape)
                self.assertTrue(torch.isfinite(y).all())
                torch.testing.assert_close(y[..., :8], z[..., :8], atol=1e-5, rtol=1e-5)

    def test_ste_gradients(self):
        for variant in ('tcn_mha', 'tcn_mla', 'tcn_mha_par', 'tcn_mla_par'):
            model = build_model(variant, turboquant=True, turboquant_train_mode='ste')
            model(torch.randn(2, 1, 16)).square().mean().backward()
            grads = [p.grad for p in model.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            attn = model.decoder.blocks[0].attn
            projection = attn.qkv_proj if 'mha' in variant else attn.kv_down_proj
            self.assertGreater(projection.weight.grad.abs().sum().item(), 0)

    def test_validation(self):
        for kwargs in ({'bits': 1}, {'bits': 3.5}, {'mode': 'bad'}, {'seed': -1}):
            with self.assertRaises(ValueError): TurboQuant(8, **kwargs)
        with self.assertRaises(ValueError): build_model('tcn', turboquant=True)
        with self.assertRaises(ValueError): build_model('tcn_mha', turboquant='true')
        with self.assertRaises(ValueError): build_model('tcn_mha', turboquant=True, turboquant_seed=-1)

    def test_checkpoint_and_eval_restore(self):
        config = dict(variant='tcn_mla', n_attn_layers=10, turboquant=True,
                      turboquant_bits=4, turboquant_seed=123)
        model = build_model_from_config(config).eval()
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            save_checkpoint(directory, 1, model, optimizer, scheduler, .1, {}, True)
            ckpt = torch.load(directory/'best.pt', weights_only=False)
            restored = build_model_from_config(ckpt['model_config']).eval()
            restored.load_state_dict(ckpt['model_state'], strict=True)
            x = torch.randn(1, 1, 16)
            with torch.no_grad(): torch.testing.assert_close(model(x), restored(x), atol=0, rtol=0)
            ns = runpy.run_path(str(ROOT/'eval.py'))
            def check_eval(m, *args, **kwargs):
                self.assertEqual(len(m.decoder.blocks), 10)
                self.assertIsNotNone(m.decoder.blocks[0].attn.turboquant)
            ns['main'].__globals__['evaluate_per_snr'] = check_eval
            with patch.object(sys, 'argv', ['eval', '--ckpt', str(directory/'best.pt'),
                                           '--data_dir', directory.as_posix(), '--mode', 'snr']):
                ns['main']()

    def test_training_config_wiring(self):
        ns = runpy.run_path(str(ROOT/'trainer/__main__.py'))
        for variant in ('tcn_mha', 'tcn_mla'):
            observed = []
            class FakeTrainer:
                def __init__(self, model, **kwargs): observed.append(model)
                def fit(self): pass
            ns['main'].__globals__['Trainer'] = FakeTrainer
            ns['main'].__globals__['build_dataloaders'] = lambda **kwargs: (None, None, None)
            argv = ['trainer', '--config', str(ROOT/f'configs/{variant}_eog_turboquant.yaml'),
                    '--n_attn_layers', '10', '--turboquant_bits', '4']
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()):
                ns['main']()
            self.assertEqual(len(observed[0].decoder.blocks), 10)
            self.assertEqual(observed[0].decoder.blocks[0].attn.turboquant.keys.bits, 4)
            with patch.object(sys, 'argv', argv+['--no-turboquant']), contextlib.redirect_stdout(io.StringIO()):
                ns['main']()
            self.assertIsNone(observed[-1].decoder.blocks[0].attn.turboquant)


if __name__ == '__main__':
    unittest.main()
