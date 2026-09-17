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
            text = launch.read_text()
            self.assertIn("cd '/proj/ect'", text)
            self.assertIn("exec '/venv/bin/python' -m gui --native", text)
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
            self.assertIn("-m gui --native", launch.read_text())


if __name__ == "__main__":
    unittest.main()
