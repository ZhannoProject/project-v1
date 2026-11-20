"""Run: python -m unittest discover -s tests -p 'test_gui.py'."""
import contextlib
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from unittest import mock

import yaml

from gui import config_io, params, server
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
    def test_training_sequence_keeps_the_log_and_resets_the_curve_per_model(self):
        job = TrainingJob()
        steps = []
        for label, loss in [("first", 0.1), ("second", 0.2)]:
            log = f"[fit | Epoch 1/1]\n→ val_loss = {loss} RRMSE_t=0.7 RRMSE_f=0.6 CC=0.8"
            steps.append({"label": label, "command": [sys.executable, "-c", f"print({log!r})"]})
        job.start_steps(steps, ROOT)
        for _ in range(100):
            if not job.running:
                break
            time.sleep(.02)
        status = job.status()
        self.assertEqual(status["returncode"], 0)
        self.assertEqual([s["returncode"] for s in status["steps"]], [0, 0])
        self.assertEqual([m["val_loss"] for m in status["metrics"]], [0.2])
        self.assertIn("=== step 1/2: first ===", status["lines"])
        self.assertIn("=== step 2/2: second ===", status["lines"])

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
                               "--profile", "gui",
                               "--max_epochs", "2", "--staged"])
        with self.assertRaises(ValueError):
            build_train_command("../evil.yaml")


class MetricsParsingTests(unittest.TestCase):
    """The chart's data: per-epoch validation metrics read back out of the log."""

    # Epoch 1 is in the pre-rename spelling (RRMSE_s / ACC) and epoch 2 in the
    # current one (RRMSE_f / CC): the parser has to read both, or the logs
    # already sitting in runs/ stop plotting.
    LOG = [
        "[KD] Teacher on the validation split:",
        "  → val_loss   = 0.3000  RRMSE_t=0.500  RRMSE_f=0.510  CC=0.850",
        "[fit | Epoch 1/3]",
        "  [E01] step   20/75  loss=1.3000",
        "  → train_loss = 1.2000  (10.0s)",
        "  → val_loss   = 0.9000  RRMSE_t=2.000  RRMSE_s=3.000  ACC=-0.010",
        "[fit | Epoch 2/3]",
        "  → train_loss = 0.6000  [task=0.5000  output=0.3000  feature=0.2000]  (9.0s)",
        "  → val_loss   = 0.5000  RRMSE_t=0.800  RRMSE_f=0.700  CC=0.700",
        "Teacher reference: val_loss=0.3000  RRMSE_t=0.500  RRMSE_f=0.510  CC=0.850",
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
                 loss_fn=None, sidecar_loss=None, saved_at=None, mtime=None):
        d = root / name
        d.mkdir()
        (d / "best.pt").write_bytes(b"")
        if last_epoch is not None and torch is not None:
            ck = {"epoch": last_epoch, "model_state": {}, "model_config": model_config or {}}
            if saved_at:
                ck["saved_at"] = saved_at
            torch.save(ck, d / "last.pt")
            torch.save(ck, d / "best.pt")
        if mtime is not None:
            for f in ("best.pt", "last.pt"):
                if (d / f).is_file():
                    os.utime(d / f, (mtime, mtime))
        if hist_epochs:
            n = hist_epochs
            h = {"train_loss": [1.0] * n, "val_loss": [1.0 - i / 100 for i in range(n)],
                 "rrmse_t": [0.7] * n, "rrmse_s": [0.6] * n, "acc": [0.8] * n}
            if loss_fn:
                # The artifact a run records has to agree with its own name:
                # list_runs prefers the recorded value (a run knows what it
                # trained on; the name only encodes it by convention), so a
                # fixture that writes EOG into a directory called *_emg is
                # describing a run that cannot exist.
                artifact = ("EMG" if name.endswith("_emg")
                            else "BOTH" if name.endswith("_both") else "EOG")
                h["config"] = {"loss_fn": loss_fn, "artifact": artifact}
            (d / "history.json").write_text(json.dumps(h))
        if sidecar_loss:
            (d / "loss_fn.json").write_text(json.dumps(
                {"run": name, "status": "recovered", "loss_fn": sidecar_loss}))
        return d

    @unittest.skipIf(torch is None, "needs real checkpoints to carry a timestamp")
    def test_runs_by_time_is_newest_first_and_keeps_the_dead_ones(self):
        """The time view is the log, not the results: a run that died is the
        main thing you come here to find, so unlike the other groupings it must
        not filter by status."""
        tmp = Path(tempfile.mkdtemp(prefix="gui_time_"))
        runs = tmp / "runs"; runs.mkdir()
        day = datetime(2026, 9, 25, 14, 30)
        older = datetime(2026, 9, 23, 9, 0)
        self._run_dir(runs, "tcn_mla_eog", hist_epochs=3, last_epoch=3,
                      saved_at=day.isoformat(timespec="seconds"))
        self._run_dir(runs, "tcn_mha_eog", hist_epochs=3, last_epoch=3,
                      saved_at=day.replace(hour=8).isoformat(timespec="seconds"))
        # no history: interrupted, and the mtime is all there is to go on
        self._run_dir(runs, "tcn_eog", last_epoch=1, mtime=older.timestamp())
        try:
            with mock.patch.object(config_io, "RUNS_DIR", runs):
                groups = config_io.runs_by_time()
                self.assertEqual([g["date"] for g in groups], ["2026-09-25", "2026-09-23"])
                self.assertEqual([r["name"] for r in groups[0]["runs"]],
                                 ["tcn_mla_eog", "tcn_mha_eog"])       # newest first
                self.assertEqual(groups[0]["sources"], ["recorded"])
                # the interrupted run is kept, and its time is flagged as a guess
                dead = groups[1]["runs"][0]
                self.assertEqual(dead["name"], "tcn_eog")
                self.assertEqual(dead["status"], "interrupted")
                self.assertEqual(dead["finished_at_source"], "file")
                self.assertEqual(dead["finished_at"][:10], "2026-09-23")
                self.assertIsNone(dead["master_seed"])          # never recorded one
                # every run appears exactly once
                self.assertEqual(sum(g["n"] for g in groups), 3)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipIf(torch is None, "needs a real checkpoint")
    def test_the_trainer_writes_the_timestamp_the_view_reads(self):
        """The two halves of this feature live in different files; if
        save_checkpoint ever stops writing saved_at, every new run silently
        falls back to mtimes and nothing else notices."""
        from trainer.checkpoint import save_checkpoint
        tmp = Path(tempfile.mkdtemp(prefix="gui_stamp_"))
        try:
            model = torch.nn.Linear(2, 2)
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
            sched = torch.optim.lr_scheduler.StepLR(opt, 1)
            before = datetime.now().replace(microsecond=0)
            save_checkpoint(tmp, 1, model, opt, sched, 0.5, {}, is_best=True)
            for name in ("best.pt", "last.pt"):
                ck = torch.load(tmp / name, map_location="cpu", weights_only=False)
                self.assertIn("saved_at", ck, name)
                self.assertGreaterEqual(datetime.fromisoformat(ck["saved_at"]), before)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipIf(torch is None, "needs a real checkpoint")
    def test_the_seed_is_read_from_either_place_the_trainer_writes_it(self):
        """history.json is the cheap source and works without torch; the
        checkpoint's run_config is the only one an interrupted run has."""
        tmp = Path(tempfile.mkdtemp(prefix="gui_seed_"))
        runs = tmp / "runs"; runs.mkdir()
        both = self._run_dir(runs, "tcn_mla_eog", hist_epochs=2, last_epoch=2)
        h = json.loads((both / "history.json").read_text())
        h["config"] = {"loss_fn": "mse", "master_seed": 7}
        (both / "history.json").write_text(json.dumps(h))
        # no history at all: the checkpoint has to answer
        ckpt_only = self._run_dir(runs, "tcn_mla_emg", last_epoch=1)
        for f in ("best.pt", "last.pt"):
            torch.save({"epoch": 1, "model_state": {}, "model_config": {},
                        "run_config": {"master_seed": 13}}, ckpt_only / f)
        try:
            with mock.patch.object(config_io, "RUNS_DIR", runs):
                rows = {r["name"]: r for r in config_io.list_runs()}
                self.assertEqual(rows["tcn_mla_eog"]["master_seed"], 7)
                self.assertEqual(rows["tcn_mla_emg"]["master_seed"], 13)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipIf(torch is None, "needs a real checkpoint")
    def test_a_recorded_time_beats_the_file_mtime(self):
        """A copied or restored directory has a truthful-looking mtime that
        describes the copy.  When the run wrote its own time, use that."""
        tmp = Path(tempfile.mkdtemp(prefix="gui_time2_"))
        runs = tmp / "runs"; runs.mkdir()
        self._run_dir(runs, "tcn_mla_eog", hist_epochs=2, last_epoch=2,
                      saved_at="2026-01-02T03:04:05",
                      mtime=datetime(2026, 9, 25).timestamp())
        try:
            with mock.patch.object(config_io, "RUNS_DIR", runs):
                row = config_io.list_runs()[0]
                self.assertEqual(row["finished_at"], "2026-01-02T03:04:05")
                self.assertEqual(row["finished_at_source"], "recorded")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

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
        cls.addClassCleanup(cls.tmp.__exit__, None, None, None)
        cls.srv = GUIServer("127.0.0.1", 0)
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        cls.thread = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.thread.start()
        server.TRAIN_JOBS.reset()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown(); cls.srv.server_close()

    @contextlib.contextmanager
    def no_runs(self, *taken):
        """A runs/ tree holding exactly the directories named, and nothing else."""
        from gui import lookahead
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for save_dir in taken:
                (root / save_dir).mkdir(parents=True, exist_ok=True)
                (root / save_dir / "best.pt").write_bytes(b"")
            with mock.patch.object(lookahead, "PROJECT_ROOT", root):
                yield root

    def call_raw(self, path):
        """Like call(), but returns the body bytes — the figure route is not JSON."""
        try:
            with urllib.request.urlopen(self.base + path) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

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
        self.assertIn("tcn_channelnorm", meta["choices"]["variants"])
        self.assertEqual(meta["variant_settings"]["tcn_channelnorm"],
                         {"se_reduction": None, "tcn_norm": "channel"})
        self.assertEqual([s["title"] for s in meta["schema"]][:2], ["Model", "Data"])
        self.assertTrue({"tcn_eog.yaml", "tcn_mla_eog_kd.yaml"} <= set(meta["configs"]))
        self.assertEqual(meta["datalists"]["checkpoints"], ["./runs/demo_bi_eog/best.pt"])

    def test_runs_endpoints_expose_kinds_and_history(self):
        status, payload = self.call("/api/runs")
        self.assertEqual(status, 200)
        self.assertIn("times", payload)
        status, times = self.call("/api/runs/times")
        self.assertEqual(status, 200)
        self.assertEqual(sum(g["n"] for g in times["times"]), 1)
        self.assertTrue(all("date" in g and "sources" in g for g in times["times"]))
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

    @unittest.skipIf(torch is None, "parameter counting builds a real model")
    def test_params_endpoints(self):
        status, r = self.call("/api/params", {"values": {"variant": "tcn_mla_par",
                                                         "n_attn_layers": 2}})
        self.assertEqual(status, 200)
        self.assertEqual(r["variant"], "tcn_mla_par")
        self.assertEqual(sum(g["params"] for g in r["groups"]), r["trainable"])
        status, r = self.call("/api/params", {"values": {"variant": "nope"}})
        self.assertEqual(status, 400)
        self.assertIn("Unknown variant", r["error"])
        status, r = self.call("/api/params/run?name=../etc")
        self.assertEqual(status, 400)

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
            # Several at once is the point: a seed repeat is n runs of one
            # config, and waiting for each before starting the next made that
            # a sequence of sittings. The cap is the only refusal left.
            ids = []
            for _ in range(server.TrainingJobs.MAX_ACTIVE):
                status, s = self.call("/api/train", {"name": "tcn_eog.yaml"})
                self.assertEqual(status, 200)
                ids.append(s["id"])
            self.assertEqual(len(set(ids)), server.TrainingJobs.MAX_ACTIVE,
                             "each run needs an id of its own, or they share a log")
            # Past the cap a run waits rather than being refused ...
            status, queued = self.call("/api/train", {"name": "tcn_eog.yaml"})
            self.assertEqual(status, 200)
            self.assertTrue(queued["queued"])
            self.assertFalse(queued["running"])
            # ... and stopping it takes it out of the queue without touching the four.
            status, s = self.call("/api/train/stop", {"id": queued["id"]})
            self.assertTrue(s["stopped"])
            self.assertFalse(s["queued"])
            self.assertEqual(s["returncode"], -1)

        # Every run is listed, and each is addressable by id.
        status, s = self.call(f"/api/train/status?id={ids[0]}")
        self.assertEqual(s["id"], ids[0])
        self.assertEqual(len([j for j in s["jobs"] if j["running"]]),
                         server.TrainingJobs.MAX_ACTIVE)
        # The run that finished earlier in this test is still listed: a
        # finished run is kept so its log and curve can be read back, which is
        # most of why the switcher is worth having.
        self.assertTrue(any(not j["running"] for j in s["jobs"]))
        self.assertEqual(s["jobs"][0]["id"], queued["id"], "newest first")

        # Stopping one leaves the others alone.
        status, s = self.call("/api/train/stop", {"id": ids[0]})
        self.assertTrue(s["stopped"])
        time.sleep(0.4)
        self.assertFalse(self.call(f"/api/train/status?id={ids[0]}")[1]["running"])
        self.assertTrue(self.call(f"/api/train/status?id={ids[1]}")[1]["running"])

        for job_id in ids[1:]:
            self.call("/api/train/stop", {"id": job_id})
        time.sleep(0.4)
        self.assertFalse(any(j["running"] for j in
                             self.call("/api/train/status")[1]["jobs"]))

    def test_a_status_without_an_id_means_the_newest_running_run(self):
        """What a page that has only ever shown one run asks for."""
        server.TRAIN_JOBS.reset()
        slow = [sys.executable, "-c", "import time; print('x', flush=True); time.sleep(30)"]
        try:
            with mock.patch.object(server, "build_train_command", return_value=slow):
                first = self.call("/api/train", {"name": "tcn_eog.yaml"})[1]["id"]
                second = self.call("/api/train", {"name": "tcn_eog.yaml"})[1]["id"]
            self.assertNotEqual(first, second)
            self.assertEqual(self.call("/api/train/status")[1]["id"], second)
        finally:
            server.TRAIN_JOBS.stop_all()

    def test_a_rung_trains_at_every_seed_asked_for(self):
        """
        One run is a point with no spread, so the tab offers the seeds
        together. They start as ordinary training runs, which is what lets the
        Train tab hold them side by side.
        """
        server.TRAIN_JOBS.reset()
        slow = [sys.executable, "-c", "import time; print('x', flush=True); time.sleep(30)"]
        try:
            # An empty runs tree: whether a rung is trained is whatever the
            # person has been doing this week, and a test that reads the real
            # one passes until they train that rung.
            with self.no_runs(), \
                 mock.patch.object(server, "build_train_command", return_value=slow):
                code, body = self.call("/api/lookahead/train",
                                       {"lookahead": 124, "master_seeds": [42, 54, 67]})
            self.assertEqual(code, 200, body)
            self.assertEqual(len(body["started"]), 3)
            self.assertEqual(len(set(body["started"])), 3, "three runs, three jobs")
            self.assertEqual(body["id"], body["started"][0],
                             "the page opens on the first of them")
            self.assertEqual([r["master_seed"] for r in body["planned"]["runs"]], [42, 54, 67])
        finally:
            server.TRAIN_JOBS.stop_all()

    def test_a_seed_that_already_has_a_run_is_named_not_just_refused(self):
        """And nothing starts, so the request is all or none."""
        server.TRAIN_JOBS.reset()
        with self.no_runs("runs/tcn_se_both"):
            code, body = self.call("/api/lookahead/train",
                                   {"lookahead": 0, "master_seeds": [42, 54, 67]})
        self.assertEqual(code, 409)
        self.assertIn("runs/tcn_se_both", body["error"])
        self.assertIn("overwrite", body["error"])
        self.assertEqual(server.TRAIN_JOBS.summaries(), [])

    def test_more_seeds_than_room_refuses_before_starting_any(self):
        """
        Starting two of three and refusing the third leaves a half-done
        request that reads like a failure of the runs that did start.
        """
        server.TRAIN_JOBS.reset()
        slow = [sys.executable, "-c", "import time; print('x', flush=True); time.sleep(30)"]
        try:
            with self.no_runs(), \
                 mock.patch.object(server.TRAIN_JOBS, "MAX_ACTIVE", 1), \
                 mock.patch.object(server.TRAIN_JOBS, "MAX_QUEUED", 1), \
                 mock.patch.object(server, "build_train_command", return_value=slow):
                code, body = self.call("/api/lookahead/train",
                                       {"lookahead": 124, "master_seeds": [42, 54, 67]})
            self.assertEqual(code, 409)
            self.assertIn("more than there is room for", body["error"])
            self.assertEqual(server.TRAIN_JOBS.summaries(), [],
                             "a refused batch must not leave runs going")
        finally:
            server.TRAIN_JOBS.stop_all()

    def test_seeds_past_the_running_cap_are_queued_not_refused(self):
        server.TRAIN_JOBS.reset()
        slow = [sys.executable, "-c", "import time; print('x', flush=True); time.sleep(30)"]
        try:
            with self.no_runs(), \
                 mock.patch.object(server.TRAIN_JOBS, "MAX_ACTIVE", 2), \
                 mock.patch.object(server, "build_train_command", return_value=slow):
                code, body = self.call("/api/lookahead/train",
                                       {"lookahead": 124, "master_seeds": [42, 54, 67]})
            self.assertEqual(code, 200, body)
            by_id = {j["id"]: j for j in body["jobs"]}
            states = [(by_id[i]["running"], by_id[i]["queued"]) for i in body["started"]]
            self.assertEqual(states, [(True, False), (True, False), (False, True)])
            self.assertEqual(by_id[body["started"][2]]["queue_position"], 1)
        finally:
            server.TRAIN_JOBS.stop_all()

    def test_overwrite_lets_a_taken_seed_be_retrained(self):
        """Otherwise the 409 telling the reader to tick overwrite names a box
        that does not exist."""
        server.TRAIN_JOBS.reset()
        slow = [sys.executable, "-c", "import time; print('x', flush=True); time.sleep(30)"]
        try:
            with self.no_runs("runs/tcn_se_both"), \
                 mock.patch.object(server, "build_train_command", return_value=slow):
                code, body = self.call("/api/lookahead/train",
                                       {"lookahead": 0, "master_seeds": [42],
                                        "overwrite": True})
            self.assertEqual(code, 200, body)
            self.assertIn("--overwrite", body["planned"]["runs"][0]["extra"])
        finally:
            server.TRAIN_JOBS.stop_all()

    def test_a_finished_run_is_kept_but_the_oldest_are_dropped(self):
        """Its log and curve stay readable; the list does not grow forever."""
        jobs = server.TrainingJobs()
        jobs.MAX_KEPT = 2
        quick = [sys.executable, "-c", "pass"]
        for _ in range(4):
            job = jobs.start(quick, config_io.PROJECT_ROOT, Path(tempfile.gettempdir()))
            for _ in range(200):
                if not job.running:
                    break
                time.sleep(0.01)
        self.assertLessEqual(len(jobs.summaries()), 2)

    def test_a_running_job_is_never_dropped(self):
        jobs = server.TrainingJobs()
        jobs.MAX_KEPT = 1
        slow = [sys.executable, "-c", "import time; time.sleep(30)"]
        quick = [sys.executable, "-c", "pass"]
        try:
            held = jobs.start(slow, config_io.PROJECT_ROOT, Path(tempfile.gettempdir()))
            for _ in range(3):
                done = jobs.start(quick, config_io.PROJECT_ROOT, Path(tempfile.gettempdir()))
                for _ in range(200):
                    if not done.running:
                        break
                    time.sleep(0.01)
            self.assertIn(held.id, [j["id"] for j in jobs.summaries()])
        finally:
            jobs.stop_all()

    @staticmethod
    def _wait(predicate, timeout=10.0):
        end = time.time() + timeout
        while time.time() < end:
            if predicate():
                return True
            time.sleep(0.02)
        return False

    def test_queued_runs_start_in_order_as_slots_free_up(self):
        jobs = server.TrainingJobs()
        jobs.MAX_ACTIVE = 2
        tmp = Path(tempfile.gettempdir())
        root = config_io.PROJECT_ROOT
        slow = [sys.executable, "-c", "import time; print('go', flush=True); time.sleep(30)"]
        quick = [sys.executable, "-c", "print('done')"]
        try:
            a = jobs.start(slow, root, tmp)
            b = jobs.start(slow, root, tmp)
            c = jobs.start(quick, root, tmp)
            d = jobs.start(slow, root, tmp)
            self.assertTrue(a.running and b.running)
            self.assertTrue(c.queued and d.queued)
            self.assertFalse(c.running or d.running)
            self.assertEqual(jobs.counts(), (2, 2))
            self.assertEqual([j["queue_position"] for j in jobs.summaries() if j["queued"]], [2, 1],
                             "newest first, so the later one is further back")
            self.assertTrue(jobs.running, "the close guard must see waiting runs")
            a.stop(grace=0.5)
            # c takes a's slot, finishes at once, and hands its slot to d.
            self.assertTrue(self._wait(lambda: c.returncode == 0 and d.running))
            self.assertIn("done", c.status(0)["lines"])
            self.assertEqual(jobs.counts(), (2, 0))
        finally:
            jobs.stop_all()

    def test_a_full_queue_refuses_and_stop_all_drops_the_queue_first(self):
        jobs = server.TrainingJobs()
        jobs.MAX_ACTIVE, jobs.MAX_QUEUED = 1, 1
        tmp = Path(tempfile.gettempdir())
        slow = [sys.executable, "-c", "import time; time.sleep(30)"]
        try:
            running = jobs.start(slow, config_io.PROJECT_ROOT, tmp)
            waiting = jobs.start(slow, config_io.PROJECT_ROOT, tmp)
            with self.assertRaises(server.TooManyJobs):
                jobs.start(slow, config_io.PROJECT_ROOT, tmp)
            self.assertEqual(jobs.room(), 0)
            jobs.stop_all()
            self.assertTrue(self._wait(lambda: not running.running))
            time.sleep(0.2)
            self.assertFalse(waiting.running, "a stopped batch must not promote the queue")
            self.assertEqual(waiting.returncode, -1)
            self.assertFalse(jobs.running)
        finally:
            jobs.stop_all()

    def test_two_live_runs_deriving_one_directory_get_different_pills(self):
        """They will overwrite each other, which the switcher cannot fix --
        but it must not show two identical labels while it happens."""
        jobs = server.TrainingJobs()
        slow = [sys.executable, "-c", "import time; time.sleep(30)"]
        try:
            a = jobs.start(slow, config_io.PROJECT_ROOT, Path(tempfile.gettempdir()), title="same")
            b = jobs.start(slow, config_io.PROJECT_ROOT, Path(tempfile.gettempdir()), title="same")
            self.assertEqual(a.title, "same")
            self.assertEqual(b.title, "same #2")
        finally:
            jobs.stop_all()


    def test_eval_endpoints_with_stubbed_command(self):
        server.EVAL_JOB = server.EvalJob()
        with tempfile.TemporaryDirectory() as tmp:
            figs = Path(tmp) / "figures"
            figs.mkdir()
            (figs / "shot.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            # An absolute path, because this test points FIGURES_DIR at a temp
            # directory while figure_rel_path resolves relative paths against
            # PROJECT_ROOT — in production the one is inside the other.
            fake = [sys.executable, "-c",
                    f"print('[Switch] Figure saved: {figs / 'shot.png'}')"]
            with mock.patch.object(server, "FIGURES_DIR", figs), \
                 mock.patch.object(server, "build_eval_command",
                                   return_value=(fake, "figures/out")):
                status, s = self.call("/api/eval", {"mode": "curves", "run": "x"})
                self.assertEqual(status, 200)
                self.assertEqual(s["save_dir"], "figures/out")
                for _ in range(100):
                    status, s = self.call("/api/eval/status?since=0")
                    if not s["running"]:
                        break
                    time.sleep(0.05)
                self.assertEqual(s["returncode"], 0)
                self.assertEqual(s["artifacts"], ["shot.png"])
                self.assertTrue(s["log_path"].endswith(".log"))

                # the figure the run reported is servable, and nothing else is
                status, body = self.call_raw("/api/eval/figure?path=shot.png")
                self.assertEqual(status, 200)
                self.assertEqual(body, b"\x89PNG\r\n\x1a\n")
                self.assertEqual(self.call("/api/eval/figure?path=../secret")[0], 400)
                self.assertEqual(self.call("/api/eval/figure?path=missing.png")[0], 400)

    def test_eval_preview_reports_the_command_and_its_refusals(self):
        status, s = self.call("/api/eval/preview", {"mode": "curves", "run": "nope"})
        self.assertEqual(status, 400)
        self.assertIn("nope", s["error"])
        status, s = self.call("/api/eval/preview", {"mode": "bogus"})
        self.assertEqual(status, 400)

    def test_an_evaluation_does_not_block_training(self):
        """The two jobs are independent; neither 409s the other."""
        server.EVAL_JOB = server.EvalJob()
        server.TRAIN_JOBS.reset()
        slow = [sys.executable, "-c", "import time; print('x', flush=True); time.sleep(30)"]
        try:
            with mock.patch.object(server, "build_eval_command", return_value=(slow, "figures/o")):
                self.assertEqual(self.call("/api/eval", {"mode": "curves", "run": "x"})[0], 200)
                self.assertEqual(self.call("/api/eval", {"mode": "curves", "run": "x"})[0], 409)
            with mock.patch.object(server, "build_train_command", return_value=slow):
                self.assertEqual(self.call("/api/train", {"name": "tcn_eog.yaml"})[0], 200)
        finally:
            self.call("/api/eval/stop", {})
            self.call("/api/train/stop", {})


@unittest.skipIf(torch is None, "parameter counting builds a real model")
class ParamReportTests(unittest.TestCase):
    """The Params tab: the breakdown has to add up and reconcile with the Runs tab."""

    def test_groups_sum_to_the_total_and_items_sum_to_their_group(self):
        r = params.report({"variant": "tcn_mla_par", "hidden_channels": 64,
                           "n_attn_layers": 4, "kv_lora_rank": 16})
        self.assertEqual(sum(g["params"] for g in r["groups"]), r["trainable"])
        for g in r["groups"]:
            if g["items"]:
                self.assertEqual(sum(i["params"] for i in g["items"]), g["params"], g["key"])

    def test_checkpoint_count_matches_what_the_runs_tab_reports(self):
        """Both numbers are state_dict-minus-causal_mask; they must not drift apart."""
        from models.variants import build_model_from_config
        cfg = {"variant": "tcn_mla", "hidden_channels": 64, "n_attn_layers": 2}
        model = build_model_from_config(params.resolve(cfg))
        listed = sum(v.numel() for k, v in model.state_dict().items()
                     if not k.endswith("causal_mask"))
        self.assertEqual(params.report(cfg)["checkpoint"], listed)

    def test_bidirectional_twin_costs_nothing(self):
        """*_bi only drops the mask; a teacher and its student are the same size."""
        base = {"hidden_channels": 64, "n_attn_layers": 4, "kv_lora_rank": 16}
        student = params.report({**base, "variant": "tcn_mla_par"})
        teacher = params.report({**base, "variant": "tcn_mla_par_bi"})
        self.assertEqual(student["trainable"], teacher["trainable"])
        self.assertTrue(student["attention"]["causal"])
        self.assertFalse(teacher["attention"]["causal"])

    def test_sandwich_runs_attention_at_the_bottleneck_width(self):
        """bottleneck_ratio shrinks d_model for the sandwich and is absent from
        the parallel models, which is where their size difference comes from."""
        base = {"hidden_channels": 64, "n_attn_layers": 4, "kv_lora_rank": 16}
        sandwich = params.report({**base, "variant": "tcn_mla", "bottleneck_ratio": 0.25})
        parallel = params.report({**base, "variant": "tcn_mla_par", "bottleneck_ratio": 0.25})
        self.assertEqual(sandwich["attention"]["d_model"], 16)
        self.assertEqual(parallel["attention"]["d_model"], 64)
        self.assertLess(sandwich["trainable"], parallel["trainable"])
        self.assertIn("bottleneck_ratio", " ".join(parallel["notes"]))

    def test_a_pinned_rank_is_reported_not_obeyed(self):
        r = params.report({"variant": "tcn_mla_r8", "hidden_channels": 64, "kv_lora_rank": 32})
        self.assertEqual(r["attention"]["kv_lora_rank"], 8)
        self.assertIn("fixes kv_lora_rank = 8", " ".join(r["notes"]))

    def test_the_measured_cache_confirms_the_analytic_one(self):
        """
        The Params tab reports the KV cache twice: derived from the
        architecture, and weighed off the streaming model's live buffers.  The
        whole point is that the two agree, so a drift between the formula in
        params.py and the code in models/incremental.py shows up here rather
        than in the Letter.
        """
        for variant in ("tcn_mha", "tcn_mla", "tcn_mla_par", "tcn_mla_r8"):
            with self.subTest(variant=variant):
                r = params.report({"variant": variant, "hidden_channels": 64,
                                   "n_attn_layers": 2, "kv_lora_rank": 16})
                self.assertTrue(r["streaming"]["available"])
                self.assertEqual(r["streaming"]["kv_per_step"],
                                 r["attention"]["kv_per_step"],
                                 "measured cache disagrees with the analytic formula")
                self.assertEqual(r["streaming"]["kv_bytes_per_step"],
                                 r["attention"]["kv_bytes_per_step"])

    def test_mla_measures_a_smaller_cache_than_mha(self):
        base = {"hidden_channels": 64, "n_attn_layers": 2, "n_heads": 4, "kv_lora_rank": 16}
        mha = params.report({**base, "variant": "tcn_mha"})["streaming"]
        mla = params.report({**base, "variant": "tcn_mla"})["streaming"]
        self.assertEqual(mha["kv_per_step"] / mla["kv_per_step"], 4.0)
        # The end-to-end number an embedded budget needs is smaller than 4x,
        # because the convolution ring buffers are the same in both and the
        # KV arithmetic does not count them.
        self.assertLess(mla["total_state_bytes"], mha["total_state_bytes"])
        self.assertLess(mha["total_state_bytes"] / mla["total_state_bytes"], 4.0)
        self.assertEqual(mha["conv_state_bytes"], mla["conv_state_bytes"])

    def test_the_baseline_reports_conv_state_and_no_cache(self):
        r = params.report({"variant": "tcn", "hidden_channels": 64})
        self.assertIsNone(r["attention"])
        self.assertTrue(r["streaming"]["available"])
        self.assertEqual(r["streaming"]["attn_cache_bytes"], 0)
        self.assertGreater(r["streaming"]["conv_state_bytes"], 0)

    def test_unstreamable_models_report_a_reason_not_an_error(self):
        """
        The Params tab is where the bidirectional teachers and the TurboQuant
        presets get inspected, and neither can stream.  That has to come back
        as a fact about the model, or the tab breaks on exactly the variants it
        exists to explain.
        """
        for variant in ("tcn_mla_bi", "tcn_bi", "tcn_mha_full_bi"):
            with self.subTest(variant=variant):
                r = params.report({"variant": variant, "hidden_channels": 64})
                self.assertFalse(r["streaming"]["available"])
                self.assertTrue(r["streaming"]["reason"])
        r = params.report({"variant": "tcn_mla", "hidden_channels": 64,
                           "turboquant": True})
        self.assertFalse(r["streaming"]["available"])
        self.assertIn("TurboQuant", r["streaming"]["reason"])

    def test_the_form_does_not_pin_a_feature_layer_the_model_lacks(self):
        """
        ``kd_feature_layers`` must default to empty, not to "decoder".

        A hard default is written into every config the GUI saves, so the
        trainer never sees it unset and never gets to resolve it from the
        model. ``decoder`` is attention-only, so a GUI run of tcn/tcn_se with
        feature distillation switched on died with "'decoder' is not a
        submodule of TCNWithSE" -- the one knob that makes KD do more than
        chase the teacher's output was unusable from the interface.
        """
        field = next(f for s in config_io.SCHEMA for f in s.fields
                     if f.key == "kd_feature_layers")
        self.assertIsNone(field.default,
                          "a pinned default is sent on every run and blocks the model's own choice")
        self.assertTrue(field.nullable, "it has to be clearable")
        self.assertIn("empty", field.help.lower(),
                      "the help has to say what empty does")

    def test_a_saved_config_leaves_the_layer_for_the_model_to_pick(self):
        from trainer.distillation import default_feature_layers
        from models.variants import build_model
        text = config_io.update_yaml_text(
            'variant: "tcn"\nkd_feature_weight: 0.0\nkd_feature_layers: ["decoder"]\n',
            {"variant": "tcn_se", "kd_feature_weight": 0.5, "kd_feature_layers": None})
        self.assertIn("kd_feature_layers: null", text)
        # ...and what the trainer then picks is a module the model has.
        self.assertEqual(default_feature_layers(build_model("tcn_se")), ["backbone"])

    def test_an_impossible_shape_is_a_message_not_a_traceback(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            params.report({"variant": "tcn_mla", "hidden_channels": 64, "n_heads": 7})
        with self.assertRaisesRegex(ValueError, "Unknown variant"):
            params.report({"variant": "tcn_nonesuch"})

    def test_run_lookup_is_sandboxed_and_needs_a_recorded_architecture(self):
        for name in ("../secrets", "..", ".", "", "runs/x", "a/../../etc"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                params.report_for_run(name)
        with self.assertRaises(FileNotFoundError):
            params.report_for_run("no_such_run")

    def test_a_half_typed_form_is_a_message_not_a_traceback(self):
        """This recomputes on every keystroke, so a field mid-edit is normal
        traffic.  Everything here used to reach the page as a 500."""
        for label, values in (
            ("empty depth box", {"variant": "tcn_mla", "n_attn_layers": 0}),
            ("negative depth", {"variant": "tcn_mla", "n_attn_layers": -3}),
            ("depth not whole", {"variant": "tcn_mla", "n_attn_layers": 2.5}),
            ("variant not a name", {"variant": ["tcn_mla"]}),
            ("values not an object", "tcn_mla"),
        ):
            with self.subTest(case=label), self.assertRaises(ValueError):
                params.report(values)

    def test_an_absurd_width_is_refused_before_anything_is_allocated(self):
        """One stray digit in hidden_channels would otherwise ask torch for tens
        of gigabytes.  A non-numeric field must not switch the ceiling off: that
        is exactly the combination that slipped through."""
        with self.assertRaisesRegex(ValueError, "counter stops at"):
            params.report({"variant": "tcn_mla", "hidden_channels": 6400})
        with self.assertRaisesRegex(ValueError, "counter stops at"):
            params.report({"variant": "tcn_mha_par_bi", "hidden_channels": 100000,
                           "bottleneck_ratio": "half"})
        with self.assertRaisesRegex(ValueError, "counter stops at"):
            params.report({"variant": "tcn_mla", "hidden_channels": 100000,
                           "n_heads": "4"})
        # and the real models stay comfortably under it
        self.assertLess(params.report({"variant": "tcn_mla_par_bi"})["trainable"],
                        params.MAX_PARAMS)

    def test_causality_is_reported_on_both_axes(self):
        for variant, causal, causal_tcn in (("tcn", True, True),
                                            ("tcn_bi", False, False),
                                            ("tcn_mla", True, True),
                                            ("tcn_mla_bi", False, True),
                                            ("tcn_mla_full_bi", False, False)):
            with self.subTest(variant=variant):
                r = params.report({"variant": variant, "hidden_channels": 64})
                self.assertEqual((r["causal"], r["causal_tcn"]), (causal, causal_tcn))


class RunsTablePageTests(unittest.TestCase):
    """The runs table is built by string concatenation in three places, and a
    column added to one of them shows up as a quietly ragged table rather than
    an error.  Nothing else checks that they still agree."""

    PAGE = (ROOT / "gui" / "static" / "index.html").read_text()

    def test_header_row_and_group_spans_agree_on_the_column_count(self):
        """
        Four tables live in this page now — the training runs, the Deploy
        runs view, which has its own columns because a measurement has a
        machine and a deadline and no epochs, the EEG-IO comparisons, which
        have a rate and several models at once and no RRMSE, and the
        receptive-field ladder, whose untrained rungs span the columns they
        have no numbers for. Each one's colspans are checked against its own
        header rather than all four against one number, so a column added to
        any of them still shows up here.
        """
        def columns(name):
            head = re.search(rf"const {name} = `(.*?)`;", self.PAGE, re.S)
            self.assertIsNotNone(head, f"{name} row not found")
            return len(re.findall(r"<th", head.group(1)))

        def carve(opening, closing):
            """Take a block out of the page so what is left is judged alone."""
            start = self.PAGE.index(opening)
            end = self.PAGE.index(closing, start + len(opening))
            return self.PAGE[start:end], (start, end)

        # The deploy and EEG-IO branches of runsMarkup and the ladder's
        # renderer, split out so each table's spans are judged against the
        # header it uses.  The branches are carved in page order, each one
        # ending where the next begins.
        deploy_block, deploy_at = carve('} else if (view === "deploy") {',
                                        '} else if (view === "eegio") {')
        eegio_block, eegio_at = carve('} else if (view === "eegio") {', "} else {")
        rf_block, rf_at = carve("function renderRfTable() {", "\n  }")
        rest = self.PAGE
        for start, end in sorted((deploy_at, eegio_at, rf_at), reverse=True):
            rest = rest[:start] + rest[end:]

        self.assertEqual(set(re.findall(r'colspan="(\d+)"', deploy_block)),
                         {str(columns("DEPLOY_HEAD"))},
                         "the Deploy runs view's colspans do not match DEPLOY_HEAD")
        self.assertEqual(set(re.findall(r'colspan="(\d+)"', eegio_block)),
                         {str(columns("EEGIO_HEAD"))},
                         "the EEG-IO runs view's colspans do not match EEGIO_HEAD")
        # The ladder's untrained rungs still print the three columns that are
        # known without training — reach, schedule, parameter cost — and span
        # only the rest, so each row is measured by its total width.
        wide = columns("RF_HEAD")
        for row in rf_block.split("<tr")[1:]:
            spans = [int(n) for n in re.findall(r'colspan="(\d+)"', row)]
            width = len(re.findall(r"<td", row)) - len(spans) + sum(spans)
            self.assertEqual(width, wide,
                             f"a receptive-field row is {width} columns wide, "
                             f"RF_HEAD has {wide}")
        self.assertEqual(set(re.findall(r'colspan="(\d+)"', rest)),
                         {str(columns("HEAD"))},
                         "the runs table's colspans do not match HEAD")

    def test_the_receptive_field_table_costs_a_rung_one_way(self):
        """
        A run's ``params`` counts every tensor in its state_dict, including
        BatchNorm's running statistics; the ladder counts trainable
        parameters. They differ by 258 per block at width 64, so a column
        that took whichever was available would show a rung getting more
        expensive the moment it was trained.
        """
        block = re.search(r"function renderRfTable\(\) \{(.*?)\n  \}", self.PAGE, re.S)
        self.assertIsNotNone(block, "renderRfTable not found")
        cells = re.findall(r'<td class="num"[^>]*>\$\{fmtN\((.*?)\)\}', block.group(1))
        self.assertTrue(cells, "no parameter cell found")
        self.assertNotIn("r.params", cells, "the cost column reads a checkpoint's state size")

    def test_opening_a_deploy_row_loads_every_trace_of_that_measurement(self):
        """
        The Deploy tab draws two panels from two slots keyed by trace kind
        (``latency`` and ``duty``), so opening only the per-hop trace left the
        duty-cycle chart showing the previous measurement's sweep, or nothing.

        A row therefore carries all of its traces and the handler loads the
        set, clearing both slots first — a row with no duty sweep must not
        keep another model's, which would attribute one model's duty cycle to
        another.
        """
        row = re.search(r"function deployRow\(r\) \{(.*?)\n  \}", self.PAGE, re.S)
        self.assertIsNotNone(row, "deployRow not found")
        self.assertIn("data-traces=", row.group(1),
                      "the row does not carry the measurement's other traces")

        handler = re.search(r'querySelectorAll\("button\[data-trace\]"\)(.*?)\}\)\);',
                            self.PAGE, re.S)
        self.assertIsNotNone(handler, "the data-trace click handler was not found")
        body = handler.group(1)
        self.assertIn("dataset.traces", body, "the handler ignores the row's other traces")
        self.assertIn("{ latency: null, duty: null }", body,
                      "the handler does not clear the panels, so a stale duty sweep survives")
        self.assertIn("loadPiTrace", body)

    def test_a_run_row_fills_every_column(self):
        row = re.search(r"return `<tr\$\{rowAttrs\}>(.*?)</tr>`;", self.PAGE, re.S)
        self.assertIsNotNone(row, "runRow's row template not found")
        head = re.search(r"const HEAD = `(.*?)`;", self.PAGE, re.S)
        columns = len(re.findall(r"<th", head.group(1)))
        # <td ...> written inline, plus the interpolations that each stand for
        # exactly one cell: the two cell helpers and the trailing action cell,
        # which differs between the Runs tab and the picker.
        helpers = len(re.findall(r"\$\{(?:paramCell|seedCell|action)\}?\(?", row.group(1)))
        cells = len(re.findall(r"<td", row.group(1))) + helpers
        self.assertEqual(cells, columns,
                         f"runRow emits {cells} cells for {columns} columns")

    def test_both_run_row_modes_emit_exactly_one_action_cell(self):
        """The Runs tab's actions and the picker's Select share the last column."""
        action = re.search(r"const action = mode === \"pick\"\n(.*?);\n", self.PAGE, re.S)
        self.assertIsNotNone(action, "runRow's action cell not found")
        for branch in action.group(1).split("      : "):
            self.assertEqual(len(re.findall(r"<td", branch)), 1,
                             f"an action branch does not emit exactly one cell: {branch.strip()}")


class LookaheadPageTests(unittest.TestCase):
    """The ladder's training controls, which are built in the page."""

    PAGE = (ROOT / "gui" / "static" / "index.html").read_text()

    def test_seeds_are_ticked_together_not_chosen_one_at_a_time(self):
        """
        A rung trained at one seed is a point the chart draws hollow, because
        the spread there is unknown. Offering one seed at a time made the
        ordinary unit of work three sittings.
        """
        self.assertNotIn('id="laSeed"', self.PAGE, "the one-of-three select is back")
        self.assertIn('id="laSeeds"', self.PAGE)
        self.assertRegex(self.PAGE, r'master_seeds: picked\.map')

    def test_the_tab_has_the_overwrite_the_refusal_names(self):
        """The server's 409 says to tick overwrite, so there has to be one."""
        self.assertIn('id="laOverwrite"', self.PAGE)
        self.assertRegex(self.PAGE, r'overwrite: \$\("#laOverwrite"\)\.checked')

    def test_the_anchor_is_not_counted_as_a_rung(self):
        """
        "10 / 9 rungs priced" — the 0 ms baseline and the unbounded ceiling
        are the two ends the ladder is measured between, not steps on it.
        It only showed once every rung had been trained.
        """
        self.assertIn("const rungs = rows.filter(r => !r.unbounded && r.lookahead !== 0)",
                      self.PAGE)
        self.assertIn("${priced.length} / ${rungs.length}", self.PAGE)
        self.assertNotIn("rows.length - 2", self.PAGE)

    def test_a_rung_with_no_past_is_off_the_price_curve(self):
        """
        The line is read as "pay this latency, get this error". A rung whose
        window has left the past entirely is not one more step along it — it
        is the baseline's mirror — so joining it to the rest would draw a
        purchase nobody can make.
        """
        fn = re.search(r"function renderLookaheadChart\(.*?\n  \}", self.PAGE, re.S)
        self.assertIsNotNone(fn, "renderLookaheadChart not found")
        body = fn.group(0)
        self.assertIn("const curve = pts.filter(r => !r.one_sided)", body)
        self.assertRegex(body, r"const line = curve\.map")
        self.assertIn("no past", body, "the chart does not say why the point is off the curve")

    def test_the_table_shows_the_whole_window_not_just_what_was_bought(self):
        """
        Only what the budget bought was ever shown. The two numbers come out
        of one fixed window, and showing both is also what answers "186 of
        what?" — the window is 124 taps, the rung number is the licence.
        """
        self.assertIn(">window</th>", self.PAGE)
        self.assertIn("${r.past} / ${r.ahead}", self.PAGE)

    def test_the_share_rides_with_the_name_in_both_places(self):
        """
        `la186` on a 124-tap window invites an arithmetic that does not work
        out. The picker is where a rung is chosen and the table is where the
        rungs are compared, so both carry the share — one is not enough, and
        two spellings of it would be worse than none.
        """
        self.assertIn("family.shares", self.PAGE)
        self.assertIn("${n} — ${variant}_la${n}${tail}", self.PAGE)
        label = re.search(r"const label = r\.unbounded(.*?);\n", self.PAGE, re.S)
        self.assertIsNotNone(label, "the rung label was not found")
        self.assertIn("share", label.group(0), "the rung column drops the share")

    def test_the_preview_asks_about_every_seed_not_the_ticked_ones(self):
        """Which directories are taken is what decides the ticks."""
        fn = re.search(r"async function previewLookahead\(.*?\n  \}", self.PAGE, re.S)
        self.assertIsNotNone(fn, "previewLookahead not found")
        self.assertIn("state.la.meta.seeds", fn.group(0))

    def test_a_trained_seed_is_unticked_rather_than_hidden(self):
        """With overwrite it is a legitimate thing to ask for, and hiding it
        would hide that the rung has been trained before."""
        fn = re.search(r"function renderLaTrain\(.*?\n  \}", self.PAGE, re.S)
        self.assertIsNotNone(fn, "renderLaTrain not found")
        self.assertIn("cb.checked = !r.exists", fn.group(0))
        self.assertNotIn("cb.disabled", fn.group(0))

    def test_several_commands_are_shown_as_several_lines(self):
        """`.cmd` wraps on word boundaries, so three commands joined by
        newlines would print as one unreadable paragraph."""
        self.assertIn("#laCommand { white-space: pre-wrap; }", self.PAGE)


class RunPickerPageTests(unittest.TestCase):
    """The Evaluate tab picks its run from the Runs table rather than a bare
    dropdown, which is only true for as long as the two share a renderer."""

    PAGE = (ROOT / "gui" / "static" / "index.html").read_text()

    def test_the_tab_and_the_picker_share_one_renderer(self):
        for caller in ("function renderRuns()", "function renderRunPicker()"):
            block = self.PAGE[self.PAGE.index(caller):]
            self.assertIn("runsMarkup(", block[:400],
                          f"{caller} does not build its table with runsMarkup")
        self.assertIn('runsMarkup(state.runsView, "tab")', self.PAGE)
        self.assertIn('runsMarkup(state.pickView, "pick")', self.PAGE)

    def test_the_picker_offers_the_runs_tab_views_minus_the_unpickable(self):
        """
        The picker and the tab share ``runsMarkup``, so every view the picker
        offers must be one the tab has.

        They are no longer identical: the tab also lists Deploy runs and
        EEG-IO comparisons, and neither is something the picker can return —
        it picks a checkpoint to evaluate, distil from or deploy, and a
        latency trace and a past comparison are results, not checkpoints.
        """
        def views(container):
            block = re.search(rf'id="{container}">(.*?)</span>', self.PAGE, re.S)
            self.assertIsNotNone(block, f"{container} not found")
            return set(re.findall(r'data-view="([^"]+)"', block.group(1)))

        pick, tab = views("pickView"), views("runsView")
        self.assertTrue(pick <= tab,
                        f"the picker offers views the tab does not: {sorted(pick - tab)}")
        self.assertEqual(tab - pick, {"deploy", "eegio"},
                         "the tab gained a view the picker skips without a reason recorded here")

    def test_the_picker_does_not_open_on_a_view_it_cannot_pick_from(self):
        """
        The picker starts on the Runs tab's view, and the tab has two the
        picker does not — Deploy runs and EEG-IO comparisons. Reading the
        tab's view unconditionally opened the dialog on a table of
        measurements with nothing in it to return, so the view is taken only
        when the picker offers it.
        """
        block = self.PAGE[self.PAGE.index("function openRunPicker("):]
        block = block[:block.index("\n  }")]
        self.assertNotIn("state.pickView = state.pickView || state.runsView;", block,
                         "the picker still inherits the tab's view unchecked")
        self.assertIn('$("#pickView").querySelectorAll("button")', block)
        self.assertIn("offered.includes(state.runsView)", block,
                      "the inherited view is not checked against the ones the picker offers")

    def test_the_picker_drops_the_actions_that_would_navigate_away(self):
        """curve and "Use as teacher" leave the tab; mid-choice that loses the
        choice, so pick mode must not render them."""
        self.assertIn('const plot = (mode === "tab"', self.PAGE)
        self.assertIn('mode === "pick"\n      ? `<td><button class="primary" data-pick=', self.PAGE)

    def test_no_run_is_selected_by_default(self):
        """The dropdown this replaced defaulted to whatever sorted first — a
        non-causal control run — so evaluating the wrong model took one click
        and said nothing.  Nothing is chosen until the user chooses it."""
        self.assertRegex(self.PAGE, r"evalRun: null")
        self.assertIn('out.run = state.evalRun || "";', self.PAGE)
        self.assertIn("Choose a run…", self.PAGE)
        self.assertNotIn('$("#evalRun").value', self.PAGE)

    def test_run_rows_are_not_built_with_a_bare_map(self):
        """``.map(runRow)`` hands runRow the array index as its second
        argument, which is the time-stamp slot — every row in the four
        non-time views then rendered its index in front of the run name.
        The views that pass no stamp must call it explicitly."""
        self.assertNotRegex(self.PAGE, r"\.map\(runRow\)")

    def test_an_unrecorded_seed_is_blank_not_assumed(self):
        """Runs from before the trainer wrote its config down have no seed.
        Filling the default in would put invented points into an error bar,
        which is the one place that does real damage."""
        self.assertIn("seedCell(r.master_seed)", self.PAGE)
        self.assertRegex(self.PAGE, r"function seedCell\(v\) \{\s*\n\s*if \(v == null\)")
        self.assertNotRegex(self.PAGE, r"master_seed\s*(\?\?|\|\|)\s*42")

    def test_the_params_column_is_served_to_the_page(self):
        """The cell reads r.params, which list_runs has to keep providing."""
        self.assertIn("paramCell(r.params)", self.PAGE)
        self.assertIn('"params": info.get("params")',
                      (ROOT / "gui" / "config_io.py").read_text())


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
        from importlib.util import find_spec
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
            if find_spec("PIL"):
                from PIL import Image
                with Image.open(app / "Contents" / "Resources" / "icon.icns") as icon:
                    self.assertEqual(icon.format, "ICNS")
                    self.assertEqual(icon.size, (1024, 1024))
                    self.assertGreater(len(icon.info["sizes"]), 1)
                self.assertEqual(info["CFBundleIconFile"], "icon")
            # rebuilding replaces the previous bundle
            build_app(Path(tmp))
            self.assertIn("root=", (app / "Contents" / "Resources" / "launch.cfg").read_text())


class EvalCommandTests(unittest.TestCase):
    """The Evaluate tab's contract: what it will run, and what it refuses to."""

    def build(self, **spec):
        cmd, save_dir = server.build_eval_command(spec)
        self.assertEqual(cmd[0], sys.executable)
        self.assertEqual(cmd[1], "eval.py")
        return cmd, save_dir

    def flag(self, cmd, name, n=1):
        i = cmd.index(name)
        return cmd[i + 1] if n == 1 else cmd[i + 1:i + 1 + n]

    # ---- the modes the page offers must be the modes eval.py has ----
    def test_mode_table_matches_eval_py(self):
        import eval as evaltools
        self.assertEqual(set(server.EVAL_MODES), set(evaltools.MODES),
                         "the GUI's mode table has drifted from eval.py's --mode choices")

    def test_single_model_mode_passes_a_checkpoint(self):
        cmd, save_dir = self.build(mode="switch", run="tcn_mla_both",
                                   data_dir="./data/x", artifact="BOTH", master_seed=7,
                                   probe_seeds="0 1")
        self.assertEqual(self.flag(cmd, "--ckpt"), "runs/tcn_mla_both/best.pt")
        self.assertEqual(self.flag(cmd, "--artifact"), "BOTH")
        self.assertEqual(self.flag(cmd, "--master_seed"), "7")
        self.assertEqual(self.flag(cmd, "--probe_seeds", 2), ["0", "1"])
        self.assertTrue(save_dir.startswith("figures/"))

    def test_compare_mode_passes_labelled_runs(self):
        cmd, _ = self.build(mode="compare_switch", data_dir="./data/x",
                            runs=[{"name": "tcn_both", "label": "TCN-only"},
                                  {"name": "tcn_mla_both", "label": ""}])
        self.assertEqual(self.flag(cmd, "--compare_runs", 2),
                         ["runs/tcn_both:TCN-only", "runs/tcn_mla_both"])
        self.assertNotIn("--ckpt", cmd)

    def test_curves_uses_history_and_needs_no_data(self):
        cmd, _ = self.build(mode="curves", run="tcn_mla_both")
        self.assertEqual(self.flag(cmd, "--history"), "runs/tcn_mla_both/history.json")
        for absent in ("--ckpt", "--data_dir", "--artifact", "--master_seed"):
            self.assertNotIn(absent, cmd)

    def test_switch_options_are_only_sent_when_set(self):
        cmd, _ = self.build(mode="switch", run="tcn_mla_both", data_dir="d",
                            switch_ramp=0, switch_snr=None, switch_epochs=None,
                            switch_control=True)
        for absent in ("--switch_ramp", "--switch_snr", "--switch_epochs",
                       "--no-switch_control", "--switch_window"):
            self.assertNotIn(absent, cmd)

        cmd, _ = self.build(mode="switch", run="tcn_mla_both", data_dir="d",
                            switch_ramp=32, switch_snr=-3.0, switch_epochs=64,
                            switch_range=[0.4, 0.6], switch_control=False)
        self.assertEqual(self.flag(cmd, "--switch_ramp"), "32")
        self.assertEqual(self.flag(cmd, "--switch_snr"), "-3.0")
        self.assertEqual(self.flag(cmd, "--switch_epochs"), "64")
        self.assertEqual(self.flag(cmd, "--switch_range", 2), ["0.4", "0.6"])
        self.assertIn("--no-switch_control", cmd)

    def test_switch_options_are_not_sent_to_unrelated_modes(self):
        cmd, _ = self.build(mode="snr", run="tcn_mla_both", data_dir="d",
                            probe_seeds="0 1", switch_ramp=32, vis_snr=-3)
        for absent in ("--probe_seeds", "--switch_ramp", "--vis_snr"):
            self.assertNotIn(absent, cmd)

    def test_extra_args_are_appended(self):
        cmd, _ = self.build(mode="curves", run="tcn_mla_both",
                            extra_args="--n_samples 2")
        self.assertEqual(cmd[-2:], ["--n_samples", "2"])

    def test_save_dir_defaults_under_figures_and_is_pinned_there(self):
        _, save_dir = self.build(mode="curves", run="tcn_mla_both")
        self.assertRegex(save_dir, r"^figures/gui_eval_\d{8}_\d{6}$")
        cmd, save_dir = self.build(mode="curves", run="tcn_mla_both",
                                   save_dir="figures/mine")
        self.assertEqual(save_dir, "figures/mine")
        for escape in ("../runs", "/tmp/anywhere", "figures/../../etc"):
            with self.assertRaises(ValueError):
                self.build(mode="curves", run="tcn_mla_both", save_dir=escape)

    def test_refusals(self):
        cases = {
            "unknown mode": dict(mode="nope", run="tcn_mla_both"),
            "no run": dict(mode="switch", data_dir="d"),
            "one run in a compare": dict(mode="compare_snr", data_dir="d",
                                         runs=[{"name": "tcn_both"}]),
            "traversal in a run name": dict(mode="switch", run="../../etc", data_dir="d"),
            "absent run": dict(mode="switch", run="definitely_not_a_run", data_dir="d"),
            "colon in a label": dict(mode="compare_snr", data_dir="d",
                                     runs=[{"name": "tcn_both", "label": "a:b"},
                                           {"name": "tcn_mla_both"}]),
            "junk probe seed": dict(mode="switch", run="tcn_mla_both", data_dir="d",
                                    probe_seeds="0 x"),
            "no data_dir": dict(mode="switch", run="tcn_mla_both", data_dir=""),
        }
        for why, spec in cases.items():
            with self.subTest(why):
                with self.assertRaises((ValueError, FileNotFoundError, KeyError)):
                    server.build_eval_command(spec)

    def test_a_run_without_the_file_the_mode_needs_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            (runs / "empty").mkdir(parents=True)
            with mock.patch.object(server, "RUNS_DIR", runs):
                with self.assertRaises(FileNotFoundError):
                    server.build_eval_command({"mode": "curves", "run": "empty"})
                with self.assertRaises(FileNotFoundError):
                    server.build_eval_command({"mode": "switch", "run": "empty",
                                               "data_dir": "d"})


class FigureServingTests(unittest.TestCase):
    """figures/ is the only tree the GUI serves, and it stays that way."""

    def test_rel_path_accepts_only_paths_inside_figures(self):
        self.assertEqual(server.figure_rel_path("figures/a/b.png"), "a/b.png")
        self.assertEqual(
            server.figure_rel_path(str(server.FIGURES_DIR / "a.png")), "a.png")
        for outside in ("/etc/passwd", "runs/tcn_both/best.pt", "../figures/x.png"):
            self.assertIsNone(server.figure_rel_path(outside))

    def test_figure_path_refuses_escapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            figs = Path(tmp) / "figures"
            (figs / "sub").mkdir(parents=True)
            (figs / "sub" / "ok.png").write_bytes(b"\x89PNG")
            (Path(tmp) / "secret.txt").write_text("no")
            with mock.patch.object(server, "FIGURES_DIR", figs):
                self.assertEqual(server.figure_path("sub/ok.png").read_bytes(), b"\x89PNG")
                for bad in ("", "../secret.txt", "sub/../../secret.txt", "missing.png"):
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        server.figure_path(bad)


class EvalJobTests(unittest.TestCase):
    """The eval job collects the files eval.py says it wrote."""

    LOG = [
        "[Switch] Evaluating: TCN-only (tcn)",
        "[Switch] Figure saved: figures/gui_eval_1/switch_response.png",
        "[Switch] Numbers saved: figures/gui_eval_1/switch_response.json",
        "[Switch] Figure saved: figures/gui_eval_1/switch_response.png",   # same file again
        "[Curves] Figure saved: /tmp/elsewhere/other.png",                 # outside figures/
        "nothing saved here",
    ]

    def test_artifacts_are_collected_deduped_and_confined(self):
        job = server.EvalJob()
        for line in self.LOG:
            job._scan(line)
        self.assertEqual(job.artifacts(),
                         ["gui_eval_1/switch_response.png",
                          "gui_eval_1/switch_response.json"])
        self.assertEqual(job.status(0)["artifacts"], job.artifacts())
        self.assertEqual(job.status(0)["kind"], "eval")

    def test_starting_again_clears_the_previous_run(self):
        job = server.EvalJob()
        job._scan("[Switch] Figure saved: figures/old.png")
        self.assertTrue(job.artifacts())
        job.start([sys.executable, "-c", "pass"], config_io.PROJECT_ROOT)
        for _ in range(100):
            if not job.running:
                break
            time.sleep(0.02)
        self.assertEqual(job.artifacts(), [])

    def test_training_and_evaluation_are_separate_jobs(self):
        """One must not lock out or overwrite the other."""
        # Training is a registry now (several runs at once) and evaluation is
        # still one job; what must hold either way is that neither can end up
        # writing into the other's buffer or log.
        self.assertIsNot(server.TRAIN_JOBS, server.EVAL_JOB)
        self.assertEqual(server.TRAIN_JOBS._factory().KIND, "train")
        self.assertEqual(server.EVAL_JOB.KIND, "eval")
        train, ev = server.TrainingJob(), server.EvalJob()
        train._scan("[fit | Epoch 1/3]")
        train._scan("  → val_loss   = 0.5  RRMSE_t=0.8  RRMSE_f=0.7  CC=0.7")
        ev._scan("[Switch] Figure saved: figures/x.png")
        self.assertEqual(len(train.metrics()), 1)
        self.assertEqual(ev.artifacts(), ["x.png"])
        self.assertNotIn("metrics", ev.status(0))
        self.assertNotIn("artifacts", train.status(0))


if __name__ == "__main__":
    unittest.main()
