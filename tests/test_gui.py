"""Run: python -m unittest discover -s tests -p 'test_gui.py'."""
import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

import yaml

from gui import config_io, server
from gui.config_io import (coerce, coerce_all, config_path, create_config, read_config,
                           save_config, update_yaml_text, yaml_scalar)
from gui.server import GUIServer, TrainingJob, build_train_command

try:
    import torch
except ImportError:                      # the GUI (and most of its tests) run without torch
    torch = None

ROOT = Path(__file__).resolve().parents[1]

SAMPLE = '''# ============================================================
# Ablation 1 — TCN + MLA Transformer (EOG artifact)
# ============================================================

# --- Model ---
variant: "tcn_mla"
hidden_channels: 64
kv_lora_rank: 16             # default: d_model // 4 = 16

# --- Data ---
artifact: "EOG"
data_dir: "./data/EEGdenoisenet/data/"   # path with # inside "quotes # stays"
snr_range_db: [-7.0, 2.0]
rank_sweep: [4, 8, 16, 32]

# --- Training ---
learning_rate: 5.0e-4
eta_min: 1.0e-6
staged: false
loss_fn: "l1_combined"              # "l1_combined", "l2_combined", "mse", "l1", or "stft"
save_dir: "./runs/tcn_mla_eog"
'''


class YamlWriterTests(unittest.TestCase):
    def test_unchanged_values_keep_their_original_text(self):
        out = update_yaml_text(SAMPLE, {"learning_rate": 5e-4, "eta_min": 1e-6, "hidden_channels": 64,
                                        "snr_range_db": [-7.0, 2.0], "staged": False})
        self.assertEqual(out, SAMPLE)           # byte-identical: 5.0e-4 not rewritten as 0.0005

    def test_changed_value_keeps_comment_and_spacing(self):
        out = update_yaml_text(SAMPLE, {"kv_lora_rank": 8, "loss_fn": "mse"})
        self.assertIn('kv_lora_rank: 8             # default: d_model // 4 = 16', out)
        self.assertIn('loss_fn: "mse"              # "l1_combined", "l2_combined", "mse", "l1", or "stft"', out)
        self.assertEqual(yaml.safe_load(out)["kv_lora_rank"], 8)
        # everything else untouched
        self.assertEqual(len(out.splitlines()), len(SAMPLE.splitlines()))
        self.assertIn("# --- Model ---", out)
        self.assertIn("rank_sweep: [4, 8, 16, 32]", out)

    def test_hash_inside_quotes_is_not_a_comment(self):
        out = update_yaml_text(SAMPLE, {"data_dir": "/new/path"})
        self.assertIn('data_dir: "/new/path"   # path with # inside "quotes # stays"', out)

    def test_missing_keys_are_appended_by_section(self):
        out = update_yaml_text(SAMPLE, {"teacher_ckpt": "./runs/t/best.pt", "kd_alpha": 0.7,
                                        "dropout": 0.2})
        self.assertIn("# --- Knowledge distillation (added by the training GUI) ---\n"
                      'teacher_ckpt: "./runs/t/best.pt"\nkd_alpha: 0.7', out)
        self.assertIn("# --- Model (added by the training GUI) ---\ndropout: 0.2", out)
        parsed = yaml.safe_load(out)
        self.assertEqual(parsed["kd_alpha"], 0.7)
        self.assertEqual(parsed["variant"], "tcn_mla")

    def test_null_bool_list_and_float_rendering(self):
        self.assertEqual(yaml_scalar(None), "null")
        self.assertEqual(yaml_scalar(True), "true")
        self.assertEqual(yaml_scalar(["decoder", "decoder.blocks.0"]), '["decoder", "decoder.blocks.0"]')
        self.assertEqual(yaml_scalar(0.5), "0.5")
        for v in (5e-4, 1e-6, 3.0, 12345.678):
            self.assertEqual(yaml.safe_load(yaml_scalar(v)), v)   # PyYAML reads back a float
        self.assertEqual(yaml_scalar('a "b"'), '"a \\"b\\""')

    def test_round_trip_through_real_preset(self):
        for name in ("tcn_mla_eog.yaml", "tcn_mla_eog_kd.yaml", "mla_rank_sweep_eog.yaml"):
            cfg = read_config(name)
            self.assertEqual(update_yaml_text(cfg["text"], cfg["values"]), cfg["text"], name)


class CoercionTests(unittest.TestCase):
    def test_types_and_ranges(self):
        self.assertEqual(coerce("hidden_channels", "64"), 64)
        self.assertEqual(coerce("learning_rate", "5e-4"), 5e-4)
        self.assertIs(coerce("staged", "true"), True)
        self.assertEqual(coerce("snr_range_db", "-7, 2"), [-7.0, 2.0])
        self.assertEqual(coerce("kd_feature_layers", "decoder, decoder.blocks.0"), ["decoder", "decoder.blocks.0"])
        self.assertIsNone(coerce("teacher_ckpt", ""))
        self.assertIsNone(coerce("kd_output_loss", None))
        with self.assertRaisesRegex(ValueError, "required"):
            coerce("data_dir", "")
        with self.assertRaisesRegex(ValueError, "≥"):
            coerce("kd_alpha", -0.1)
        with self.assertRaisesRegex(ValueError, "≤"):
            coerce("turboquant_bits", 9)
        with self.assertRaisesRegex(ValueError, "low must be"):
            coerce("snr_range_db", [3, 1])
        with self.assertRaisesRegex(ValueError, "not a known setting"):
            coerce_all({"learnign_rate": 1})

    def test_config_names_are_sandboxed(self):
        for bad in ("../x.yaml", "sub/x.yaml", "x.txt", ".hidden.yaml", ""):
            with self.assertRaises(ValueError, msg=bad):
                config_path(bad)
        self.assertEqual(config_path("tcn_eog.yaml").parent, (ROOT / "configs").resolve())


class _TempConfigs:
    """Point config_io at a scratch copy of two presets and an empty runs dir."""
    def __enter__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gui_test_"))
        cfg_dir, runs_dir = self.tmp / "configs", self.tmp / "runs"
        cfg_dir.mkdir(); runs_dir.mkdir()
        for name in ("tcn_eog.yaml", "tcn_mla_eog_kd.yaml"):
            shutil.copy(ROOT / "configs" / name, cfg_dir / name)
        (runs_dir / "demo_bi_eog").mkdir()
        (runs_dir / "demo_bi_eog" / "best.pt").write_bytes(b"")
        (runs_dir / "demo_bi_eog" / "history.json").write_text(json.dumps(
            {"train_loss": [1, .5], "val_loss": [.9, .4], "rrmse_t": [.9, .5],
             "rrmse_s": [.9, .5], "acc": [.1, .8], "kd_output": [.3, .2]}))
        self.patches = [mock.patch.object(config_io, "CONFIG_DIR", cfg_dir),
                        mock.patch.object(config_io, "RUNS_DIR", runs_dir),
                        mock.patch.object(server, "RUNS_DIR", runs_dir)]
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)


class ConfigIOTests(unittest.TestCase):
    def test_read_save_create(self):
        with _TempConfigs() as t:
            cfg = read_config("tcn_mla_eog_kd.yaml")
            self.assertEqual(cfg["values"]["variant"], "tcn_mla")
            self.assertIn("teacher_ckpt", cfg["present"])
            self.assertEqual(cfg["extra"], {})

            text = save_config("tcn_mla_eog_kd.yaml", {**cfg["values"], "max_epochs": 7})
            self.assertEqual((t.tmp / "configs" / "tcn_mla_eog_kd.yaml").read_text(), text)
            self.assertEqual(yaml.safe_load(text)["max_epochs"], 7)
            self.assertIn("# --- Knowledge distillation", text)   # comments survived

            create_config("new.yaml", template="tcn_eog.yaml")
            self.assertTrue((t.tmp / "configs" / "new.yaml").read_text().startswith("# Created by the training GUI from tcn_eog.yaml"))
            with self.assertRaises(FileExistsError):
                create_config("new.yaml")
            create_config("minimal.yaml")
            self.assertEqual(yaml.safe_load((t.tmp / "configs" / "minimal.yaml").read_text())["variant"], "tcn_mla")
            self.assertEqual(config_io.list_configs(), ["minimal.yaml", "new.yaml", "tcn_eog.yaml", "tcn_mla_eog_kd.yaml"])

            runs = config_io.list_runs()
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["name"], "demo_bi_eog")
            self.assertTrue(runs[0]["bidirectional"] and runs[0]["kd"])
            self.assertEqual((runs[0]["epochs"], runs[0]["best_epoch"], runs[0]["acc"]), (2, 2, .8))


class TrainingJobTests(unittest.TestCase):
    def test_lines_are_buffered_incrementally_and_stop_works(self):
        job = TrainingJob()
        job.start([sys.executable, "-c", "import time\nfor i in range(3): print('line', i)\ntime.sleep(30)"], ROOT)
        for _ in range(100):
            s = job.status(0)
            if len(s["lines"]) == 3:
                break
            time.sleep(0.05)
        self.assertEqual(s["lines"], ["line 0", "line 1", "line 2"])
        self.assertEqual(job.status(since=2)["lines"], ["line 2"])
        self.assertTrue(job.running)
        with self.assertRaises(RuntimeError):
            job.start([sys.executable, "-c", "pass"], ROOT)
        self.assertTrue(job.stop())
        time.sleep(0.2)
        self.assertFalse(job.running)
        self.assertNotEqual(job.status(0)["returncode"], 0)

    def test_train_command_uses_this_interpreter(self):
        cmd = build_train_command("tcn_eog.yaml", "--max_epochs 2 --staged")
        self.assertEqual(cmd, [sys.executable, "-m", "trainer", "--config", "configs/tcn_eog.yaml",
                               "--max_epochs", "2", "--staged"])
        with self.assertRaises(ValueError):
            build_train_command("../evil.yaml")


class MetricsParsingTests(unittest.TestCase):
    """The chart's data: per-epoch validation metrics read back out of the log."""

    LOG = [
        "[KD] Teacher on the validation split:",
        "  → val_loss   = 0.3000  RRMSE_t=0.500  RRMSE_s=0.510  ACC=0.850",
        "[fit | Epoch 1/3]",
        "  [E01] step   20/75  loss=1.3000",
        "  → train_loss = 1.2000  (10.0s)",
        "  → val_loss   = 0.9000  RRMSE_t=2.000  RRMSE_s=3.000  ACC=-0.010",
        "[fit | Epoch 2/3]",
        "  → train_loss = 0.6000  [task=0.5000  output=0.3000  feature=0.2000]  (9.0s)",
        "  → val_loss   = 0.5000  RRMSE_t=0.800  RRMSE_s=0.700  ACC=0.700",
        "Teacher reference: val_loss=0.3000  RRMSE_t=0.500  RRMSE_s=0.510  ACC=0.850",
    ]

    def test_epochs_are_parsed_and_teacher_lines_ignored(self):
        job = TrainingJob()
        for line in self.LOG:
            job._scan(line)
        m = job.metrics()
        # The teacher's reference pass (before epoch 1) and the closing summary
        # have the same numbers in them but are not epochs of this run.
        self.assertEqual([r["epoch"] for r in m], [1, 2])
        self.assertEqual(job._max_epochs, 3)
        self.assertEqual(m[0], {"epoch": 1, "phase": "fit", "train_loss": 1.2,
                                "val_loss": 0.9, "rrmse_t": 2.0, "rrmse_s": 3.0, "acc": -0.01})
        self.assertEqual(m[1]["train_loss"], 0.6)          # KD's bracketed breakdown
        self.assertEqual(m[1]["acc"], 0.7)

    def test_status_exposes_metrics_and_start_clears_them(self):
        job = TrainingJob()
        for line in self.LOG:
            job._scan(line)
        st = job.status(0)
        self.assertEqual(len(st["metrics"]), 2)
        self.assertEqual(st["max_epochs"], 3)
        job.start([sys.executable, "-c", "pass"], ROOT)
        for _ in range(100):
            if not job.running:
                break
            time.sleep(0.02)
        self.assertEqual(job.status(0)["metrics"], [])

    def test_real_training_log_round_trips(self):
        """Parse a log this project actually produced, if one is lying around."""
        logs = sorted((ROOT / "runs").glob("gui_train_*.log"))
        if not logs:
            self.skipTest("no runs/gui_train_*.log to parse")
        job = TrainingJob()
        for line in logs[-1].read_text(errors="replace").splitlines():
            job._scan(line)
        m = job.metrics()
        if not m:
            self.skipTest("that log has no completed epoch")
        self.assertEqual([r["epoch"] for r in m], list(range(1, len(m) + 1)))
        self.assertTrue(all("val_loss" in r and "rrmse_t" in r for r in m))


class RunsListingTests(unittest.TestCase):
    """complete / stale / interrupted, and the grouping the Runs tab shows."""

    def _run_dir(self, root, name, hist_epochs=None, last_epoch=None, model_config=None,
                 loss_fn=None, sidecar_loss=None):
        d = root / name
        d.mkdir()
        (d / "best.pt").write_bytes(b"")
        if last_epoch is not None and torch is not None:
            torch.save({"epoch": last_epoch, "model_state": {},
                        "model_config": model_config or {}}, d / "last.pt")
            torch.save({"epoch": last_epoch, "model_state": {},
                        "model_config": model_config or {}}, d / "best.pt")
        if hist_epochs:
            n = hist_epochs
            h = {"train_loss": [1.0] * n, "val_loss": [1.0 - i / 100 for i in range(n)],
                 "rrmse_t": [0.7] * n, "rrmse_s": [0.6] * n, "acc": [0.8] * n}
            if loss_fn:
                h["config"] = {"loss_fn": loss_fn, "artifact": "EOG"}
            (d / "history.json").write_text(json.dumps(h))
        if sidecar_loss:
            (d / "loss_fn.json").write_text(json.dumps(
                {"run": name, "status": "recovered", "loss_fn": sidecar_loss}))
        return d

    def test_status_classification_and_kinds(self):
        tmp = Path(tempfile.mkdtemp(prefix="gui_runs_"))
        runs = tmp / "runs"; runs.mkdir()
        self._run_dir(runs, "tcn_mla_eog", hist_epochs=3, last_epoch=3,
                      model_config={"variant": "tcn_mla", "n_attn_layers": 2, "kv_lora_rank": 4})
        self._run_dir(runs, "tcn_mla_both", hist_epochs=3, last_epoch=1,
                      model_config={"variant": "tcn_mla"})
        self._run_dir(runs, "tcn_mha_eog")                       # no history at all
        try:
            with mock.patch.object(config_io, "RUNS_DIR", runs):
                rows = {r["name"]: r for r in config_io.list_runs()}
                self.assertEqual(rows["tcn_mha_eog"]["status"], "interrupted")
                if torch is None:
                    self.skipTest("torch needed to write a checkpoint with an epoch")
                self.assertEqual(rows["tcn_mla_eog"]["status"], "complete")
                # history says 3 epochs, last.pt sits at 1: the directory was re-used
                self.assertEqual(rows["tcn_mla_both"]["status"], "stale")
                self.assertEqual(rows["tcn_mla_eog"]["kind"], "tcn_mla · EOG")
                self.assertEqual(rows["tcn_mla_eog"]["kv_lora_rank"], 4)

                kinds = config_io.runs_by_kind()
                self.assertEqual([g["kind"] for g in kinds], ["tcn_mla · EOG"])
                self.assertEqual([r["name"] for r in kinds[0]["runs"]], ["tcn_mla_eog"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_loss_is_recorded_recovered_or_unknown(self):
        """Where each run's loss_fn comes from, and that an unverified one is never claimed."""
        tmp = Path(tempfile.mkdtemp(prefix="gui_loss_"))
        runs = tmp / "runs"; runs.mkdir()
        self._run_dir(runs, "new_eog", hist_epochs=3, last_epoch=3, loss_fn="l1_combined")
        self._run_dir(runs, "old_eog", hist_epochs=3, last_epoch=3, sidecar_loss="mse")
        self._run_dir(runs, "mystery_eog", hist_epochs=3, last_epoch=3)
        # a sidecar that failed to identify must not be believed
        d = self._run_dir(runs, "stale_eog", hist_epochs=3, last_epoch=1)
        (d / "loss_fn.json").write_text(json.dumps(
            {"run": "stale_eog", "status": "checkpoint-mismatch", "detail": "re-used dir"}))
        try:
            with mock.patch.object(config_io, "RUNS_DIR", runs):
                rows = {r["name"]: r for r in config_io.list_runs()}
                self.assertEqual((rows["new_eog"]["loss_fn"], rows["new_eog"]["loss_source"]),
                                 ("l1_combined", "recorded"))
                self.assertEqual((rows["old_eog"]["loss_fn"], rows["old_eog"]["loss_source"]),
                                 ("mse", "recovered"))
                self.assertEqual((rows["mystery_eog"]["loss_fn"], rows["mystery_eog"]["loss_source"]),
                                 (None, "unknown"))
                self.assertIsNone(rows["stale_eog"]["loss_fn"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_optimizer_is_read_from_the_checkpoint_state(self):
        """Adam and AdamW differ only by decoupled_weight_decay; the schedule by its state."""
        from gui.config_io import optimizer_info

        def ck(dwd, groups, sched):
            return {"optimizer_state": {"param_groups": [
                        {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0,
                         "decoupled_weight_decay": dwd, **g} for g in groups]},
                    "scheduler_state": sched}

        cosine = {"_schedulers": [{"end_factor": 1.0}, {"T_max": 95}], "_milestones": [5]}
        info = optimizer_info(ck(True, [{"name": "tcn", "initial_lr": 5e-4},
                                        {"name": "decoder", "initial_lr": 1.5e-3}], cosine))
        self.assertEqual(info["family"], "AdamW")
        self.assertEqual(info["scheduler"], "cosine+warmup")
        self.assertEqual(info["lr"], 5e-4)
        self.assertTrue(info["lr_exact"])
        self.assertEqual(info["attn_lr_scale"], 3.0)
        self.assertIn("AdamW", info["label"])
        self.assertIn("×3", info["label"])

        # plateau run with no initial_lr: the rate is the decayed one, not the configured one
        plateau = optimizer_info(ck(False, [{"name": "tcn", "lr": 1.25e-3},
                                            {"name": "decoder", "lr": 1.25e-4}],
                                    {"factor": 0.5, "patience": 8, "mode": "min"}))
        self.assertEqual(plateau["family"], "Adam")
        self.assertEqual(plateau["scheduler"], "plateau")
        self.assertFalse(plateau["lr_exact"])
        self.assertEqual(plateau["attn_lr_scale"], 0.1)   # exact: both groups decay together
        self.assertIn("decayed", plateau["label"])

        self.assertEqual(optimizer_info(ck(True, [{"name": "all", "initial_lr": 5e-3}], {}))
                         ["scheduler"], "none")
        self.assertEqual(optimizer_info({})  , {})
        # an older torch that does not record the flag must not be called AdamW
        self.assertEqual(optimizer_info({"optimizer_state": {"param_groups": [
            {"betas": (0.9, 0.999), "weight_decay": 0.0, "lr": 1e-3}]}})["family"],
            "Adam/AdamW")
        self.assertEqual(optimizer_info({"optimizer_state": {"param_groups": [
            {"momentum": 0.9, "lr": 0.1}]}})["family"], "SGD")

    def test_runs_by_optimizer_groups_runs_optimised_the_same_way(self):
        if torch is None:
            self.skipTest("torch needed to write a checkpoint")
        tmp = Path(tempfile.mkdtemp(prefix="gui_opt_"))
        runs = tmp / "runs"; runs.mkdir()

        def make(name, dwd, lr, scale, sched):
            d = self._run_dir(runs, name, hist_epochs=3, last_epoch=3, loss_fn="mse")
            groups = [{"name": "tcn", "initial_lr": lr, "betas": (0.9, 0.999),
                       "weight_decay": 0.0, "decoupled_weight_decay": dwd},
                      {"name": "decoder", "initial_lr": lr * scale, "betas": (0.9, 0.999),
                       "weight_decay": 0.0, "decoupled_weight_decay": dwd}]
            for f in ("best.pt", "last.pt"):
                torch.save({"epoch": 3, "model_state": {}, "model_config": {"variant": "tcn_mla"},
                            "optimizer_state": {"param_groups": groups},
                            "scheduler_state": sched}, d / f)

        cosine = {"_schedulers": [{"end_factor": 1.0}, {"T_max": 95}], "_milestones": [5]}
        make("a_eog", True, 5e-4, 3.0, cosine)
        make("b_eog", True, 5e-4, 3.0, cosine)          # same recipe as a
        make("c_eog", True, 5e-3, 0.1, cosine)          # different LR and ratio
        make("d_eog", False, 5e-4, 3.0, {"factor": 0.5, "patience": 8, "mode": "min"})
        try:
            with mock.patch.object(config_io, "RUNS_DIR", runs):
                groups = config_io.runs_by_optimizer()
                by = {g["key"]: [r["name"] for r in g["runs"]] for g in groups}
                self.assertEqual(len(groups), 3)
                adamw_same = [v for k, v in by.items() if "AdamW" in k and "0.0005" in k][0]
                self.assertEqual(sorted(adamw_same), ["a_eog", "b_eog"])
                self.assertTrue(any("Adam ·" in k and "plateau" in k for k in by))
                self.assertTrue(all(g["known"] for g in groups))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_runs_by_loss_groups_only_what_is_comparable(self):
        """val_loss is comparable within one loss x artifact group, so that is the grouping."""
        if torch is None:
            self.skipTest("torch needed to write a checkpoint with an epoch")
        tmp = Path(tempfile.mkdtemp(prefix="gui_loss_"))
        runs = tmp / "runs"; runs.mkdir()
        self._run_dir(runs, "a_eog", hist_epochs=3, last_epoch=3, loss_fn="mse")
        self._run_dir(runs, "b_eog", hist_epochs=4, last_epoch=4, loss_fn="mse")
        self._run_dir(runs, "c_eog", hist_epochs=3, last_epoch=3, loss_fn="l1_combined")
        self._run_dir(runs, "d_emg", hist_epochs=3, last_epoch=3, loss_fn="mse")
        self._run_dir(runs, "e_eog", hist_epochs=3, last_epoch=3)              # unknown loss
        self._run_dir(runs, "f_eog", hist_epochs=3, last_epoch=1, loss_fn="mse")  # stale
        try:
            with mock.patch.object(config_io, "RUNS_DIR", runs):
                groups = config_io.runs_by_loss()
                keys = [g["key"] for g in groups]
                # same loss but a different artifact is a different group
                self.assertEqual(keys, ["l1_combined · EOG", "mse · EMG", "mse · EOG",
                                        "unknown · EOG"])
                by = {g["key"]: g for g in groups}
                self.assertEqual([r["name"] for r in by["mse · EOG"]["runs"]], ["b_eog", "a_eog"])
                self.assertTrue(by["mse · EOG"]["comparable"])
                self.assertFalse(by["unknown · EOG"]["comparable"])
                # the stale run is excluded from every group, as in the kind view
                self.assertNotIn("f_eog", [r["name"] for g in groups for r in g["runs"]])
                # ranked by val_loss: legitimate inside a group, which is the point
                vals = [r["val_loss"] for r in by["mse · EOG"]["runs"]]
                self.assertEqual(vals, sorted(vals))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_variant_falls_back_to_the_directory_name(self):
        """Checkpoints written before model_config existed still group correctly."""
        tmp = Path(tempfile.mkdtemp(prefix="gui_runs_"))
        runs = tmp / "runs"; runs.mkdir()
        self._run_dir(runs, "tcn_mha_par_emg", hist_epochs=2)
        try:
            with mock.patch.object(config_io, "RUNS_DIR", runs):
                row = config_io.list_runs()[0]
                self.assertEqual((row["variant"], row["artifact"]), ("tcn_mha_par", "EMG"))
                self.assertEqual(row["kind"], "tcn_mha_par · EMG")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_read_run_history_is_sandboxed(self):
        tmp = Path(tempfile.mkdtemp(prefix="gui_runs_"))
        runs = tmp / "runs"; runs.mkdir()
        self._run_dir(runs, "tcn_mla_eog", hist_epochs=4)
        self._run_dir(runs, "half_done")
        try:
            with mock.patch.object(config_io, "RUNS_DIR", runs):
                h = config_io.read_run_history("tcn_mla_eog")
                self.assertEqual(h["max_epochs"], 4)
                self.assertEqual([r["epoch"] for r in h["metrics"]], [1, 2, 3, 4])
                self.assertIn("rrmse_t", h["metrics"][0])
                with self.assertRaises(ValueError):
                    config_io.read_run_history("../configs")
                with self.assertRaises(ValueError):
                    config_io.read_run_history("")
                with self.assertRaises(FileNotFoundError):
                    config_io.read_run_history("half_done")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = _TempConfigs().__enter__()
        cls.srv = GUIServer("127.0.0.1", 0)
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        cls.thread = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.thread.start()
        server.JOB = TrainingJob()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown(); cls.srv.server_close()
        cls.tmp.__exit__(None, None, None)

    def call(self, path, body=None):
        req = urllib.request.Request(self.base + path, method="POST" if body is not None else "GET",
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_page_and_meta(self):
        with urllib.request.urlopen(self.base + "/") as r:
            self.assertIn(b"<title>Training settings</title>", r.read())
        status, meta = self.call("/api/meta")
        self.assertEqual(status, 200)
        self.assertIn("tcn_mla_bi", meta["choices"]["variants"])
        self.assertEqual([s["title"] for s in meta["schema"]][:2], ["Model", "Data"])
        self.assertTrue({"tcn_eog.yaml", "tcn_mla_eog_kd.yaml"} <= set(meta["configs"]))
        self.assertEqual(meta["datalists"]["checkpoints"], ["./runs/demo_bi_eog/best.pt"])

    def test_runs_endpoints_expose_kinds_and_history(self):
        status, payload = self.call("/api/runs")
        self.assertEqual(status, 200)
        self.assertEqual([r["name"] for r in payload["runs"]], ["demo_bi_eog"])
        self.assertIn("kinds", payload)
        self.assertEqual(sum(g["n"] for g in payload["kinds"]), 1)
        self.assertIn("losses", payload)

        status, losses = self.call("/api/runs/losses")
        self.assertEqual(status, 200)
        self.assertEqual(sum(g["n"] for g in losses["losses"]), 1)
        self.assertTrue(all("comparable" in g for g in losses["losses"]))

        self.assertIn("optimizers", payload)
        status, opt = self.call("/api/runs/optimizers")
        self.assertEqual(status, 200)
        self.assertTrue(all("label" in g and "detail" in g for g in opt["optimizers"]))

        status, hist = self.call("/api/runs/history?name=demo_bi_eog")
        self.assertEqual(status, 200)
        self.assertEqual(hist["max_epochs"], 2)
        self.assertEqual([r["epoch"] for r in hist["metrics"]], [1, 2])
        self.assertEqual(hist["metrics"][1]["acc"], 0.8)

        status, err = self.call("/api/runs/history?name=../configs")
        self.assertEqual(status, 400)
        self.assertIn("invalid run name", err["error"])

    def test_config_preview_save_new(self):
        status, cfg = self.call("/api/config?name=tcn_eog.yaml")
        self.assertEqual(status, 200)
        self.assertEqual(cfg["values"]["variant"], "tcn")

        status, prev = self.call("/api/preview", {"name": "tcn_eog.yaml", "values": {"max_epochs": 3}})
        self.assertEqual(status, 200)
        self.assertIn("max_epochs: 3", prev["text"])
        self.assertIn("max_epochs: 100", read_config("tcn_eog.yaml")["text"])   # preview did not write

        status, err = self.call("/api/preview", {"name": "tcn_eog.yaml", "values": {"batch_size": ""}})
        self.assertEqual(status, 400)
        self.assertIn("batch_size", err["error"])

        status, saved = self.call("/api/config", {"name": "tcn_eog.yaml", "values": {"max_epochs": 3}})
        self.assertEqual(status, 200)
        self.assertEqual(yaml.safe_load(read_config("tcn_eog.yaml")["text"])["max_epochs"], 3)

        status, made = self.call("/api/config/new", {"name": "exp.yaml", "template": "tcn_eog.yaml"})
        self.assertEqual(status, 200)
        self.assertIn("exp.yaml", made["configs"])
        status, _ = self.call("/api/config/new", {"name": "exp.yaml"})
        self.assertEqual(status, 409)
        status, _ = self.call("/api/config?name=../x.yaml")
        self.assertEqual(status, 400)
        status, _ = self.call("/api/nope")
        self.assertEqual(status, 404)

    def test_train_endpoints_with_stubbed_command(self):
        fake = [sys.executable, "-c", "print('hello'); print('world')"]
        with mock.patch.object(server, "build_train_command", return_value=fake):
            status, s = self.call("/api/train", {"name": "tcn_eog.yaml"})
        self.assertEqual(status, 200)
        for _ in range(100):
            status, s = self.call("/api/train/status?since=0")
            if not s["running"]:
                break
            time.sleep(0.05)
        self.assertEqual(s["returncode"], 0)
        self.assertEqual(s["lines"], ["hello", "world"])
        self.assertTrue(s["log_path"].endswith(".log"))
        self.assertTrue(Path(s["log_path"]).read_text().startswith("hello\nworld\n"))

        slow = [sys.executable, "-c", "import time; print('x', flush=True); time.sleep(30)"]
        with mock.patch.object(server, "build_train_command", return_value=slow):
            self.assertEqual(self.call("/api/train", {"name": "tcn_eog.yaml"})[0], 200)
            self.assertEqual(self.call("/api/train", {"name": "tcn_eog.yaml"})[0], 409)
        status, s = self.call("/api/train/stop", {})
        self.assertTrue(s["stopped"])
        time.sleep(0.3)
        self.assertFalse(self.call("/api/train/status")[1]["running"])


class NativeAndBundleTests(unittest.TestCase):
    def test_cli_mode_selection(self):
        from gui import __main__ as cli
        with mock.patch.object(cli, "run_native") as native, mock.patch.object(cli, "serve") as browser:
            with mock.patch.object(cli, "native_available", return_value=True):
                cli.main([])                                   # default → native, free port
                native.assert_called_once_with("127.0.0.1", 0, verbose=False)
                cli.main(["--browser", "--no-browser"])        # explicit browser mode
                browser.assert_called_once_with("127.0.0.1", 8765, open_browser=False, verbose=False)
            native.reset_mock(); browser.reset_mock()
            with mock.patch.object(cli, "native_available", return_value=False):
                cli.main(["--port", "9000"])                   # no pywebview → browser fallback
                browser.assert_called_once_with("127.0.0.1", 9000, open_browser=True, verbose=False)
                native.assert_not_called()

    @unittest.skipUnless(sys.platform == "darwin", "app bundles are macOS-only")
    def test_make_app_builds_a_launcher_bundle(self):
        import plistlib
        from gui.make_app import build_app
        with tempfile.TemporaryDirectory() as tmp:
            app = build_app(Path(tmp), python="/venv/bin/python", root=Path("/proj/ect"))
            self.assertEqual(app.name, "EEG Training GUI.app")
            launch = app / "Contents" / "MacOS" / "launch"
            self.assertTrue(launch.stat().st_mode & 0o111)
            cfg = (app / "Contents" / "Resources" / "launch.cfg").read_text()
            self.assertEqual(cfg, "root=/proj/ect\npython=/venv/bin/python\n")
            if shutil.which("cc"):
                self.assertTrue(launch.read_bytes()[:4] in (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe"),
                                "expected a Mach-O launcher when a compiler is available")
            else:
                self.assertIn("exec '/venv/bin/python' -m gui --native", launch.read_text())
            with open(app / "Contents" / "Info.plist", "rb") as f:
                info = plistlib.load(f)
            self.assertEqual(info["CFBundleExecutable"], "launch")
            self.assertEqual(info["CFBundleIdentifier"], "local.eeg-denoise.training-gui")
            self.assertIn("NSDesktopFolderUsageDescription", info)
            if shutil.which("iconutil"):
                self.assertTrue((app / "Contents" / "Resources" / "icon.icns").is_file())
                self.assertEqual(info["CFBundleIconFile"], "icon")
            # rebuilding replaces the previous bundle
            build_app(Path(tmp))
            self.assertIn("root=", (app / "Contents" / "Resources" / "launch.cfg").read_text())


if __name__ == "__main__":
    unittest.main()
