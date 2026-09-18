"""Run: python -m unittest discover -s tests -p 'test_distillation.py'."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from models.AttentionBlocks.attention import CausalMHA, CausalMLA, CausalTransformerDecoder
from models.variants import (BIDIRECTIONAL_VARIANTS, build_model, build_model_from_config,
                             variant_is_causal)
from trainer.checkpoint import save_checkpoint
from trainer.distillation import (DistillationLoss, FeatureProjector, FeatureTap, KDTrainer,
                                  init_student_from_teacher, load_teacher,
                                  parse_feature_layers, probe_feature_shapes)
from trainer.losses import build_loss

T = 128
SMALL = dict(hidden_channels=16, max_seq_len=T)


def _small(variant, **kwargs):
    """Tiny model of any variant; MLA variants get a rank-2 KV bottleneck."""
    kw = {**SMALL, **kwargs}
    if "mla" in variant:
        kw.setdefault("kv_lora_rank", 2)
    return build_model(variant, **kw)


def _past_diff(model, t=T // 2):
    """Max change in outputs before t when the input is perturbed at t."""
    model.eval()
    x1 = torch.randn(2, 1, T)
    x2 = x1.clone()
    x2[:, :, t] += 10.0
    with torch.no_grad():
        return (model(x1)[:, :, :t] - model(x2)[:, :, :t]).abs().max().item()


class _SyntheticEEG(Dataset):
    """Dict batches shaped like data/Dataloader.py, no files needed."""
    def __init__(self, n=32, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.clean = torch.randn(n, 1, T, generator=g)
        self.noisy = self.clean + 0.5 * torch.randn(n, 1, T, generator=g)

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, i):
        return {"noisy": self.noisy[i], "clean": self.clean[i],
                "sigma": torch.tensor(1.0), "snr_db": torch.tensor(0.0)}


def _checkpoint(tmp, variant, **kwargs):
    model = _small(variant, **kwargs)
    opt = torch.optim.Adam(model.parameters())
    sched = torch.optim.lr_scheduler.StepLR(opt, 1)
    run_dir = Path(tmp) / f"{variant}_test"
    run_dir.mkdir()
    with contextlib.redirect_stdout(io.StringIO()):
        save_checkpoint(run_dir, 3, model, opt, sched, 0.25, {}, is_best=True)
    return model, run_dir / "best.pt"


class BidirectionalAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_causal_flag_controls_the_mask(self):
        for attention_type in ("mha", "mla"):
            for causal in (True, False):
                dec = CausalTransformerDecoder(d_model=16, n_heads=4, n_layers=1, max_seq_len=T,
                                               attention_type=attention_type, causal=causal).eval()
                x1 = torch.randn(2, 16, T)
                x2 = x1.clone()
                x2[:, :, T // 2] += 10.0
                with torch.no_grad():
                    past = (dec(x1)[:, :, :T // 2] - dec(x2)[:, :, :T // 2]).abs().max().item()
                if causal:
                    self.assertEqual(past, 0.0, attention_type)
                else:
                    self.assertGreater(past, 0.0, attention_type)

    def test_bi_variants_see_the_future_and_causal_twins_do_not(self):
        for name in BIDIRECTIONAL_VARIANTS:
            base = name[:-3]
            self.assertFalse(variant_is_causal(name))
            self.assertTrue(variant_is_causal(base))
            self.assertGreater(_past_diff(_small(name)), 0.0, name)
            self.assertEqual(_past_diff(_small(base)), 0.0, base)

    def test_bi_variants_share_state_dict_with_causal_twin(self):
        for name in BIDIRECTIONAL_VARIANTS:
            t_sd = _small(name).state_dict()
            s_sd = _small(name[:-3]).state_dict()
            self.assertEqual(list(t_sd), list(s_sd), name)
            for k in t_sd:
                self.assertEqual(t_sd[k].shape, s_sd[k].shape, f"{name}.{k}")

    def test_all_attention_modules_carry_the_flag(self):
        for name in ("tcn_mla_bi", "tcn_mha_par_bi"):
            model = _small(name)
            attn = [m for m in model.modules() if isinstance(m, (CausalMHA, CausalMLA))]
            self.assertTrue(attn)
            self.assertTrue(all(not m.causal for m in attn), name)
            self.assertFalse(model.causal)

    def test_model_config_round_trips_the_variant(self):
        model = _small("tcn_mla_bi")
        self.assertEqual(model.model_config["variant"], "tcn_mla_bi")
        rebuilt = build_model_from_config(model.model_config)
        self.assertFalse(rebuilt.causal)
        rebuilt.load_state_dict(model.state_dict())

    def test_contradictory_causal_kwarg_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "tcn_mla_bi"):
            _small("tcn_mla", causal=False)
        with self.assertRaisesRegex(ValueError, "'tcn_mla'"):
            _small("tcn_mla_bi", causal=True)
        _small("tcn_mla_bi", causal=False)  # redundant but consistent


class DistillationLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.s = torch.randn(2, 1, T, requires_grad=True)
        self.t = torch.randn(2, 1, T)
        self.c = torch.randn(2, 1, T)

    def test_terms_combine_as_documented(self):
        mse = build_loss("mse")
        crit = DistillationLoss(mse, build_loss("l1"), alpha=0.3, feature_weight=0.2)
        sf = {"decoder": torch.randn(2, 8, T, requires_grad=True)}
        tf = {"decoder": torch.randn(2, 8, T)}
        total, parts = crit(self.s, self.t, self.c, sf, tf)
        expected = (0.7 * mse(self.s, self.c) + 0.3 * build_loss("l1")(self.s, self.t)
                    + 0.2 * torch.nn.functional.mse_loss(sf["decoder"], tf["decoder"]))
        self.assertTrue(torch.allclose(total, expected))
        self.assertEqual(set(parts), {"task", "output", "feature"})
        self.assertFalse(parts["task"].requires_grad)

    def test_alpha_zero_is_plain_training(self):
        mse = build_loss("mse")
        total, parts = DistillationLoss(mse, mse, alpha=0.0)(self.s, self.t, self.c)
        self.assertTrue(torch.allclose(total, mse(self.s, self.c)))
        self.assertEqual(parts["feature"].item(), 0.0)

    def test_teacher_gets_no_gradient(self):
        t = self.t.clone().requires_grad_(True)
        total, _ = DistillationLoss(build_loss("mse"), build_loss("mse"), alpha=1.0)(self.s, t, self.c)
        total.backward()
        self.assertIsNone(t.grad)
        self.assertIsNotNone(self.s.grad)

    def test_feature_shape_mismatch_is_reported(self):
        crit = DistillationLoss(build_loss("mse"), build_loss("mse"), feature_weight=1.0)
        with self.assertRaisesRegex(ValueError, "shapes must match"):
            crit(self.s, self.t, self.c, {"decoder": torch.zeros(2, 8, T)},
                 {"decoder": torch.zeros(2, 16, T)})
        with self.assertRaisesRegex(ValueError, "no features"):
            crit(self.s, self.t, self.c)

    def test_bad_weights_are_rejected(self):
        mse = build_loss("mse")
        with self.assertRaises(ValueError):
            DistillationLoss(mse, mse, alpha=1.5)
        with self.assertRaises(ValueError):
            DistillationLoss(mse, mse, feature_weight=-1)

    def test_feature_tap_records_decoder_output(self):
        model = _small("tcn_mla")
        tap = FeatureTap(model, ["decoder", "decoder.blocks.0"])
        model(torch.randn(2, 1, T))
        self.assertEqual(tuple(tap.features["decoder"].shape), (2, 8, T))
        self.assertEqual(tuple(tap.features["decoder.blocks.0"].shape), (2, T, 8))
        self.assertTrue(tap.features["decoder"].requires_grad)
        tap.remove()
        tap.clear()
        model(torch.randn(2, 1, T))
        self.assertEqual(tap.features, {})
        with self.assertRaisesRegex(ValueError, "not a submodule"):
            FeatureTap(model, ["encoder"])


class TeacherStudentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_init_from_teacher_copies_every_tensor(self):
        teacher = _small("tcn_mha_bi")
        student = _small("tcn_mha")
        with contextlib.redirect_stdout(io.StringIO()):
            n_copied, n_skipped = init_student_from_teacher(student, teacher)
        self.assertEqual(n_skipped, 0)
        self.assertEqual(n_copied, len(student.state_dict()))
        for k, v in teacher.state_dict().items():
            self.assertTrue(torch.equal(v, student.state_dict()[k]), k)
        # ... and yet the student is still causal
        self.assertEqual(_past_diff(student), 0.0)

    def test_load_teacher_is_frozen_eval_and_refused_by_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, ckpt = _checkpoint(tmp, "tcn_mla_bi")
            with contextlib.redirect_stdout(io.StringIO()):
                teacher = load_teacher(ckpt)
            self.assertFalse(teacher.training)
            self.assertFalse(teacher.causal)
            self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, teacher.state_dict()[k]), k)

            from deploy.export_onnx import load_checkpoint_model
            with self.assertRaisesRegex(ValueError, "bidirectional teacher"):
                load_checkpoint_model(ckpt)

    def test_load_teacher_legacy_checkpoint_without_model_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, ckpt = _checkpoint(tmp, "tcn_mla", hidden_channels=64, kv_lora_rank=2,
                                      max_seq_len=512)
            raw = torch.load(ckpt, weights_only=False)
            raw.pop("model_config")
            torch.save(raw, ckpt)
            with contextlib.redirect_stdout(io.StringIO()):
                teacher = load_teacher(ckpt)
            self.assertEqual(teacher.model_config["variant"], "tcn_mla")
            self.assertEqual(teacher.model_config["kv_lora_rank"], 2)

    def test_kd_trainer_end_to_end(self):
        loader = DataLoader(_SyntheticEEG(32), batch_size=8)
        teacher = _small("tcn_mla_bi").eval()
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                kd = KDTrainer(
                    model=_small("tcn_mla"), teacher=teacher,
                    train_loader=loader, val_loader=loader,
                    kd_alpha=0.5, kd_feature_weight=0.1, init_from_teacher=True,
                    save_dir=tmp, max_epochs=2, log_every=999, loss_fn="mse",
                    device=torch.device("cpu"),
                )
                kd.fit()
            history = json.loads((Path(tmp) / "history.json").read_text())
            for k in ("train_loss", "val_loss", "kd_task", "kd_output", "kd_feature"):
                self.assertEqual(len(history[k]), 2, k)
            self.assertEqual(history["teacher"]["variant"], "tcn_mla_bi")
            self.assertIn("acc", history["teacher"])
            self.assertTrue(all(v > 0 for v in history["kd_feature"]))

            ckpt = torch.load(Path(tmp) / "best.pt", weights_only=False)
            self.assertEqual(ckpt["model_config"]["variant"], "tcn_mla")
            student = build_model_from_config(ckpt["model_config"])
            student.load_state_dict(ckpt["model_state"])
            self.assertEqual(_past_diff(student), 0.0)
            # teacher untouched by training
            self.assertTrue(all(not p.requires_grad for p in kd.teacher.parameters()))

    def test_kd_trainer_without_feature_term_installs_no_taps(self):
        loader = DataLoader(_SyntheticEEG(16), batch_size=8)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            kd = KDTrainer(
                model=_small("tcn_mha"), teacher=_small("tcn_mha_bi"),
                train_loader=loader, val_loader=loader, kd_alpha=1.0, kd_feature_weight=0.0,
                save_dir=tmp, max_epochs=1, log_every=999, loss_fn="l1",
                device=torch.device("cpu"),
            )
            self.assertIsNone(kd._student_tap)
            kd.fit()
            self.assertEqual(kd.history["kd_feature"], [0.0])


class CompressedStudentTests(unittest.TestCase):
    """The student may be narrower / shallower than the teacher."""

    def setUp(self):
        torch.manual_seed(0)
        self.teacher = _small("tcn_mla_bi", hidden_channels=32, n_attn_layers=3, kv_lora_rank=4).eval()
        self.student = _small("tcn_mla", hidden_channels=16, n_attn_layers=2, kv_lora_rank=2)
        self.loader = DataLoader(_SyntheticEEG(), batch_size=8, shuffle=True)

    def test_parse_feature_layers(self):
        self.assertEqual(parse_feature_layers(["decoder", " decoder.blocks.0 : decoder.blocks.2 "]),
                         [("decoder", "decoder", "decoder"),
                          (" decoder.blocks.0 : decoder.blocks.2 ", "decoder.blocks.0", "decoder.blocks.2")])
        with self.assertRaisesRegex(ValueError, "empty student path"):
            parse_feature_layers([":decoder"])

    def test_init_from_teacher_copies_leading_slices(self):
        n_copied, n_skipped = init_student_from_teacher(self.student, self.teacher)
        self.assertEqual(n_skipped, 0)                 # every student tensor has a teacher twin
        s_sd, t_sd = self.student.state_dict(), self.teacher.state_dict()
        k = "tcn_pt1.network.0.conv_block.0.conv.weight"        # (H, 1, 3): H differs
        self.assertTrue(torch.equal(s_sd[k], t_sd[k][:16]))
        k = "decoder.blocks.1.attn.q_proj.weight"                # (d, d): both axes differ
        self.assertTrue(torch.equal(s_sd[k], t_sd[k][:8, :8]))
        self.assertNotIn("decoder.blocks.2.attn.q_proj.weight", s_sd)   # deeper teacher block skipped

    def test_projector_bridges_channel_axis_and_stays_causal(self):
        pairs = parse_feature_layers(["decoder", "decoder.blocks.0:decoder.blocks.2"])
        shapes = probe_feature_shapes(self.student, self.teacher, pairs, T)
        self.assertEqual(shapes["decoder"], ((8, T), (16, T)))                      # (B, C, T) tap
        self.assertEqual(shapes["decoder.blocks.0:decoder.blocks.2"], ((T, 8), (T, 16)))  # (B, T, C) tap
        for label, (s_shape, t_shape) in shapes.items():
            proj = FeatureProjector((1, *s_shape), (1, *t_shape))
            self.assertFalse(proj.is_identity)
            x = torch.randn(2, *s_shape)
            self.assertEqual(tuple(proj(x).shape), (2, *t_shape))
            # pointwise in time: perturbing t only changes output at t
            t_axis = s_shape.index(T)
            x2 = x.clone()
            x2.select(t_axis + 1, T // 2).add_(10.0)
            delta = (proj(x) - proj(x2)).abs().sum(dim=[d for d in range(3) if d != t_axis + 1])
            self.assertEqual((delta > 0).nonzero().flatten().tolist(), [T // 2])
        self.assertTrue(FeatureProjector((1, 8, T), (1, 8, T)).is_identity)
        with self.assertRaisesRegex(ValueError, "more than one axis"):
            FeatureProjector((1, 8, T), (1, 16, T // 2))

    def test_kd_trainer_trains_a_smaller_student(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            kd = KDTrainer(model=self.student, teacher=self.teacher, train_loader=self.loader,
                           val_loader=self.loader, kd_alpha=0.5, kd_feature_weight=0.1,
                           kd_feature_layers=["decoder", "decoder.blocks.0:decoder.blocks.2"],
                           init_from_teacher=True, save_dir=tmp, max_epochs=2, log_every=999,
                           device=torch.device("cpu"), loss_fn="mse")
            self.assertEqual(len(kd.kd_projectors), 2)
            proj_ids = {id(p) for p in kd.kd_projectors.parameters()}
            in_opt = {id(p) for g in kd.optimizer.param_groups for p in g["params"]}
            self.assertTrue(proj_ids <= in_opt, "projector params must be optimised")
            before = [p.detach().clone() for p in kd.kd_projectors.parameters()]
            kd.staged_fit(warmup_epochs=1, sched_warmup_epochs=0)       # rebuilds the optimizer twice
            in_opt = {id(p) for g in kd.optimizer.param_groups for p in g["params"]}
            self.assertTrue(proj_ids <= in_opt, "projectors must survive the staged rebuild")
            self.assertTrue(any(not torch.equal(a, b) for a, b in
                                zip(before, kd.kd_projectors.parameters())), "projectors did not train")
            self.assertEqual(len(kd.history["kd_feature"]), 2)
            self.assertEqual(kd.history["student"]["n_params"],
                             sum(p.numel() for p in self.student.parameters()))
            self.assertLess(kd.history["student"]["n_params"], kd.history["teacher"]["n_params"])
            ckpt = torch.load(Path(tmp) / "best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(ckpt["model_config"]["hidden_channels"], 16)
            self.assertEqual(set(ckpt["model_state"]), set(self.student.state_dict()))
            rebuilt = build_model_from_config(ckpt["model_config"])
            rebuilt.load_state_dict(ckpt["model_state"], strict=True)   # no projector keys leaked
        self.assertLess(_past_diff(self.student), 1e-5)

    def test_same_size_student_gets_identity_projectors(self):
        twin = _small("tcn_mla", hidden_channels=32, n_attn_layers=3, kv_lora_rank=4)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            kd = KDTrainer(model=twin, teacher=self.teacher, train_loader=self.loader,
                           val_loader=self.loader, kd_feature_weight=0.1, save_dir=tmp,
                           max_epochs=1, log_every=999, device=torch.device("cpu"), loss_fn="mse")
            self.assertEqual(len(kd.kd_projectors), 0)
            self.assertTrue(kd._projectors["decoder"].is_identity)
            self.assertEqual([g["name"] for g in kd.optimizer.param_groups], ["all"])


if __name__ == "__main__":
    unittest.main()
