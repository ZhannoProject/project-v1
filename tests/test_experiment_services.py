"""Output ownership, lifecycle, configuration and checkpoint compatibility contracts."""

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import torch

from configs import load_config
from experiment.config import defaults, resolve_config
from experiment.checkpoints import load_model, read_checkpoint
from experiment.storage import RunLease, atomic_json, atomic_write
from gui import server, config_io, native
from models.variants import build_model
from trainer.checkpoint import save_checkpoint
from trainer.training import Trainer

ROOT = Path(__file__).resolve().parents[1]


class ExperimentServicesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def checkpoint(self, variant="tcn", legacy=False):
        model = build_model(variant, hidden_channels=8, dropout=0,
                            tcn_norm="batch" if legacy else "channel")
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        with contextlib.redirect_stdout(io.StringIO()):
            save_checkpoint(self.tmp, 1, model, optimizer, scheduler, .1,
                            {"config": {"master_seed": 67, "artifact": "BOTH"}}, True)
        path = self.tmp / "best.pt"
        if legacy:
            raw = torch.load(path, weights_only=False)
            raw.pop("schema_version")
            raw.pop("model_config")
            torch.save(raw, path)
        return model.eval(), path

    def trainer(self, name="run", overwrite=False):
        batch = {"noisy": torch.randn(2, 1, 512), "clean": torch.randn(2, 1, 512)}
        with contextlib.redirect_stdout(io.StringIO()):
            return Trainer(build_model("tcn", hidden_channels=8, dropout=0), [batch], [batch],
                           save_dir=str(self.tmp / name), max_epochs=1, loss_fn="mse",
                           device=torch.device("cpu"), overwrite=overwrite)

    def test_browser_shutdown_stops_every_kind_and_closes_server(self):
        fake = MagicMock(server_address=("127.0.0.1", 1234))
        fake.serve_forever.side_effect = KeyboardInterrupt
        jobs = [MagicMock(running=True) for _ in range(4)]
        with patch.object(server, "GUIServer", return_value=fake), \
             patch.object(server.TASKS, "entries", return_value=list(enumerate(jobs))), \
             patch.object(server.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
            server.serve(open_browser=False)
        for job in jobs:
            job.shutdown.assert_called_once()
        fake.server_close.assert_called_once()

    def test_shutdown_continues_after_one_task_fails(self):
        bad, good = MagicMock(running=True), MagicMock(running=True)
        bad.shutdown.side_effect = RuntimeError("stop failed")
        fake = MagicMock(server_address=("127.0.0.1", 1234))
        fake.serve_forever.side_effect = KeyboardInterrupt
        with patch.object(server, "GUIServer", return_value=fake), \
             patch.object(server.TASKS, "entries", return_value=[("bad", bad), ("good", good)]), \
             patch.object(server.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "stop failed"):
                server.serve(open_browser=False)
        good.shutdown.assert_called_once()
        fake.server_close.assert_called_once()

    def test_native_and_browser_share_shutdown(self):
        fake = MagicMock(server_address=("127.0.0.1", 1234))
        webview = MagicMock()
        with patch.dict(sys.modules, {"webview": webview}), \
             patch.object(native, "GUIServer", return_value=fake), \
             patch.object(native.threading, "Thread"), patch.object(native, "_brand_process_macos"), \
             patch.object(server.TASKS, "stop_all") as stop, contextlib.redirect_stdout(io.StringIO()):
            native.run_native()
        stop.assert_called_once()
        fake.shutdown.assert_called_once()
        fake.server_close.assert_called_once()

    def test_lease_rejects_another_process_then_releases(self):
        code = """from experiment.storage import RunLease
import sys
try:
    with RunLease(sys.argv[1]): pass
except FileExistsError:
    sys.exit(17)
"""
        with RunLease(self.tmp):
            result = subprocess.run([sys.executable, "-c", code, str(self.tmp)], cwd=ROOT,
                                     capture_output=True, text=True)
            self.assertEqual(result.returncode, 17, result.stderr)
        result = subprocess.run([sys.executable, "-c", code, str(self.tmp)], cwd=ROOT,
                                 capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_lease_is_released_after_process_is_killed(self):
        code = """from experiment.storage import RunLease
import sys,time
with RunLease(sys.argv[1]):
    print('locked',flush=True)
    time.sleep(30)
"""
        child = subprocess.Popen([sys.executable, "-c", code, str(self.tmp)], cwd=ROOT,
                                 stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), "locked")
            with self.assertRaises(FileExistsError):
                with RunLease(self.tmp):
                    pass
        finally:
            child.kill()
            child.wait(timeout=5)
            child.stdout.close()
        with RunLease(self.tmp):
            pass

    def test_failed_atomic_write_preserves_previous_artifact(self):
        path = self.tmp / "history.json"
        atomic_json(path, {"epoch": 1})
        def fail(stream):
            stream.write(b"broken")
            raise RuntimeError("interrupted")
        with self.assertRaises(RuntimeError):
            atomic_write(path, fail)
        self.assertEqual(json.loads(path.read_text()), {"epoch": 1})
        self.assertEqual(list(self.tmp.glob("*.tmp")), [])

    def test_training_failure_releases_lease(self):
        trainer = self.trainer()
        with patch.object(trainer, "_train_one_epoch", side_effect=RuntimeError("failed")), \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                trainer.fit()
        with RunLease(trainer.save_dir):
            pass

    def test_interrupted_run_recovers_seed_from_configuration_only(self):
        from trainer.checkpoint import recorded_seed
        atomic_json(self.tmp / "resolved_config.json",
                    dict(schema_version=1, run_config=dict(master_seed=67)))
        seed, files = recorded_seed(self.tmp)
        self.assertEqual(seed, 67)
        self.assertEqual(files, ["resolved_config.json"])

    def test_complete_snapshot_and_explicit_overwrite_backup(self):
        trainer = self.trainer()
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.fit()
        path = trainer.save_dir / "best.pt"
        original = path.read_bytes()
        raw = read_checkpoint(path)
        snapshot = json.loads((trainer.save_dir / "resolved_config.json").read_text())
        self.assertEqual(snapshot["run_id"], raw["run_id"])
        for key in ("learning_rate", "weight_decay", "max_epochs", "loss_fn", "device", "arch"):
            self.assertIn(key, snapshot["run_config"])
        with self.assertRaises(FileExistsError):
            self.trainer()
        with contextlib.redirect_stdout(io.StringIO()):
            again = self.trainer(overwrite=True)
            again.fit()
        archives = list((trainer.save_dir / "archive").glob("*/best.pt"))
        self.assertEqual(len(archives), 1)
        self.assertEqual(archives[0].read_bytes(), original)
        self.assertNotEqual(read_checkpoint(path)["run_id"], raw["run_id"])
        replay = resolve_config(load_config(trainer.save_dir / "resolved_config.json"))
        self.assertEqual(replay["learning_rate"], again.history["config"]["learning_rate"])
        self.assertEqual(replay["hidden_channels"], 8)
        self.assertFalse(replay["overwrite"])

    def test_gui_reserves_output_and_freezes_resolved_configuration(self):
        class FakeJob:
            def __init__(self): self.running = False
            def start(self, command, cwd, log_path):
                self.command, self.running = command, True
        registry = server.TrainingJobs(factory=FakeJob)
        cfg = self.tmp / "cfg.yaml"
        cfg.write_text("variant: tcn\nmaster_seed: 67\nsave_dir: runs/test\n")
        command = [sys.executable, "-m", "trainer", "--config", str(cfg)]
        job = registry.start(command, self.tmp, self.tmp / "logs")
        with self.assertRaises(FileExistsError):
            registry.start(command, self.tmp, self.tmp / "logs", title="different title")
        cfg.write_text("master_seed: 42\n")
        frozen = load_config(job.command[-1])
        self.assertEqual(frozen["master_seed"], 67)
        self.assertEqual(frozen["save_dir"], "runs/test")
        self.assertEqual(job.title, "test")
        self.assertIsInstance(frozen["eta_min"], float)
        resolve_config(frozen)

    def test_gui_uses_rate_qualified_paths_and_titles_beside_legacy_runs(self):
        class FakeJob:
            def __init__(self): self.running = False
            def start(self, command, cwd, log_path):
                self.command, self.running = command, True

        registry = server.TrainingJobs(factory=FakeJob)
        runs = self.tmp / "runs"
        old = runs / "tcn_se_both_s67" / "best.pt"
        old.parent.mkdir(parents=True)
        old.write_bytes(b"legacy weights")
        cfg = self.tmp / "cfg.yaml"
        cfg.write_text("variant: tcn_se\nartifact: BOTH\nmaster_seed: 67\ntarget_fs: 250\n")
        command = [sys.executable, "-m", "trainer", "--config", str(cfg), "--profile", "gui"]
        for rate, flags in ((250, []), (256, ["--target_fs", "256"])):
            job = registry.start(command + flags, self.tmp, runs)
            name = f"tcn_se_both_{rate}hz_s67"
            self.assertEqual(job.title, name)
            self.assertEqual(job.output_dir, (runs / name).resolve())
            self.assertEqual(load_config(job.command[-1])["save_dir"], f"./runs/{name}")
        self.assertEqual(old.read_bytes(), b"legacy weights")

    def test_simultaneous_gui_launches_cannot_claim_the_same_output(self):
        entered, release = threading.Event(), threading.Event()
        class FakeJob:
            def __init__(self): self.running = False
            def start(self, command, cwd, log_path):
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test timed out")
                self.running = True
        registry = server.TrainingJobs(factory=FakeJob)
        command = [sys.executable, "-m", "trainer", "--variant", "tcn", "--save_dir", "same"]
        results = []
        def launch():
            try:
                results.append(registry.start(command, self.tmp, self.tmp / "logs"))
            except Exception as exc:
                results.append(exc)
        first, second = threading.Thread(target=launch), threading.Thread(target=launch)
        first.start()
        try:
            self.assertTrue(entered.wait(5))
            second.start()
        finally:
            release.set()
            first.join(5)
            if second.ident is not None:
                second.join(5)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(sum(isinstance(r, FileExistsError) for r in results), 1)
        self.assertEqual(sum(isinstance(r, FakeJob) for r in results), 1)

    def test_named_profiles_are_shared_with_gui_fields_and_trainer(self):
        for key, field in config_io.FIELDS.items():
            if key in defaults("gui"):
                self.assertEqual(field.default, defaults("gui")[key], key)
        import inspect
        signature = inspect.signature(Trainer)
        for key in ("learning_rate", "max_epochs", "log_every", "loss_fn"):
            self.assertEqual(signature.parameters[key].default, defaults("api")[key])
        for path in (ROOT / "configs").glob("*.yaml"):
            resolve_config(load_config(path))

    def test_invalid_configuration_fails_before_training(self):
        for values in ({"typo_learning_rate": 1}, {"batch_size": 0}, {"max_epochs": True},
                       {"normalization_scale": float("nan")}, {"learning_rate": float("inf")},
                       {"artifact": "wrong"}, {"split_ratios": [1, 1, 1]}, {"tcn_norm": "batch"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                resolve_config(values)

    def test_shutdown_refuses_new_work_until_reopened(self):
        registry = server.TrainingJobs()
        registry.shutdown()
        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            registry.start([sys.executable, "-c", "pass"], self.tmp, self.tmp)
        job = server.TrainingJob()
        job.shutdown()
        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            job.start([sys.executable, "-c", "pass"], self.tmp)
        registry.reopen()
        job.reopen()
        self.assertFalse(registry._closing)
        self.assertFalse(job._closing)

    def test_one_training_stop_failure_does_not_skip_remaining_jobs(self):
        registry = server.TrainingJobs()
        bad, good = MagicMock(running=True), MagicMock(running=True)
        bad.stop.side_effect = RuntimeError("failure")
        registry._jobs.update(bad=bad, good=good)
        with self.assertRaisesRegex(RuntimeError, "failure"):
            registry.shutdown()
        good.stop.assert_called_once()

    def test_old_and_new_checkpoints_use_same_loader_and_preserve_predictions(self):
        for legacy in (False, True):
            original, path = self.checkpoint(legacy=legacy)
            model, arch, raw = load_model(path)
            self.assertEqual(arch["tcn_norm"], "batch" if legacy else "channel")
            self.assertEqual(raw["schema_version"], 1)
            self.assertEqual(raw["source_schema_version"], 0 if legacy else 1)
            x = torch.randn(2, 1, 512)
            with torch.no_grad():
                torch.testing.assert_close(model(x), original(x), atol=0, rtol=0)
            self.assertEqual(model.run_config["master_seed"], 67)

    def test_future_checkpoint_version_is_refused(self):
        _, path = self.checkpoint()
        raw = torch.load(path, weights_only=False)
        raw["schema_version"] = 999
        torch.save(raw, path)
        with self.assertRaisesRegex(ValueError, "schema_version"):
            read_checkpoint(path)

    def test_legacy_none_rank_is_resolved_from_actual_weights(self):
        model = build_model("tcn_mla", hidden_channels=32, dropout=0).eval()
        path = self.tmp / "rank.pt"
        torch.save(dict(model_state=model.state_dict(), model_config={**model.model_config,
                                                                     "kv_lora_rank": None}), path)
        restored, arch, _ = load_model(path)
        self.assertEqual(arch["kv_lora_rank"], 4)
        with torch.no_grad():
            x = torch.randn(1, 1, 64)
            torch.testing.assert_close(restored(x), model(x), atol=0, rtol=0)

    def test_gui_uses_recorded_artifact_for_arbitrary_directory_names(self):
        _, path = self.checkpoint()
        root = self.tmp / "library"
        run = root / "arbitrary_name"
        run.mkdir(parents=True)
        path.rename(run / "best.pt")
        with patch.object(config_io, "RUNS_DIR", root):
            rows = config_io.list_runs()
        self.assertEqual(rows[0]["artifact"], "BOTH")
        self.assertEqual(rows[0]["master_seed"], 67)

    def test_all_model_entry_points_use_same_architecture_and_data_protocol(self):
        _, path = self.checkpoint()
        from eval import load_run_model
        from eegio_eval import load_run
        from trainer.distillation import load_teacher
        from deploy.export_onnx import load_checkpoint_model
        from gui import kd
        with contextlib.redirect_stdout(io.StringIO()), patch.object(kd, "_runs_dir", return_value=self.tmp.parent):
            models = [load_run_model(str(self.tmp))[0], load_run(str(self.tmp))[0],
                      load_teacher(path), load_checkpoint_model(path)[0], kd._load(self.tmp.name)[0]]
        for model in models:
            self.assertEqual(model.model_config["variant"], "tcn")
            self.assertEqual(model.run_config["master_seed"], 67)
            self.assertEqual(model.run_config["artifact"], "BOTH")


if __name__ == "__main__":
    unittest.main()
