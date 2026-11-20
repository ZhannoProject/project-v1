"""Exercise Pi EEG-IO scoring with real ONNX graphs and loopback transport."""

import contextlib
from http.client import HTTPResponse
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import eegio_eval
from data.EEGIO import read_recording, write_synthetic_recording
from deploy import eegio_replay, preflight
from experiment.config import resolve_config
from experiment.methodology import aligned_output
from gui import config_io, eegio_pi, eegio_tab
from models.variants import build_model, resolved_arch


class LoopbackRemote:
    def __init__(self):
        self.traces = []
        self.cleaned = 0
        self.uploads = {}

    def ssh(self, _command):
        pass

    def stage(self, model, **_):
        self.model = model

    def upload(self, source, name):
        self.uploads[name] = source

    def run_case(self, args, local, stem, *, suffixes):
        args = args[2:]
        args[args.index("--model") + 1] = str(self.model)
        for flag in ("--input", "--tail-model"):
            if flag in args:
                index = args.index(flag) + 1
                args[index] = str(self.uploads[args[index]])
        args[args.index("--out") + 1] = str(local / stem)
        eegio_replay.main(args)
        with np.load(local / (stem + ".npz"), allow_pickle=False) as result:
            self.traces.append((result["filtered"], result["output"]))

    def stop(self, *, remove_workspace):
        self.cleaned += int(remove_workspace)


class EEGIOPiTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.enterContext(patch.object(eegio_eval, "PROJECT_ROOT", self.root))
        self.enterContext(patch.object(config_io, "RUNS_DIR", self.root / "runs"))
        self.enterContext(patch.object(eegio_tab, "RESULTS_DIR", self.root / "results"))
        self.data = self.root / "eegio"
        write_synthetic_recording(self.data, duration_s=16.012, n_blinks=1, n_soft=0, amplitude=60)
        self.enterContext(patch.object(eegio_tab, "DATA_DIR_CANDIDATES", (self.data,)))
        self.runs = ["channelnorm_both_250hz)_s67", "paper_delay16"]
        self.enterContext(patch.object(eegio_tab, "available_runs", return_value=[
            {"name": run, "train_fs": 250., "causal": True, "train_fs_recorded": True,
             "training_protocol": "paper" if run == self.runs[1] else "legacy"}
            for run in self.runs
        ]))
        self.spec = dict(runs=self.runs, target_fs=250, n_subjects=1, hop=4, ssh="pi@host")

    def test_invalid_pi_requests_fail_before_ssh(self):
        for change in (
            {"ssh": ""}, {"ssh": "host;bad"}, {"target_fs": 256},
            {"zero_phase": True}, {"runs": ["unavailable"]}, {"hop": 8},
            {"cpus": "0-999999"}, {"cpus": "2,2"}, {"cpus": "2", "threads": 2},
            {"replay_mode": "2x"},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                eegio_pi.build_command({**self.spec, **change})

    def test_windowed_and_delayed_step_traces_reach_the_same_scoring_path(self):
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = build_model("tcn_se", hidden_channels=8, tcn_norm="channel", se_window=16)
        protocol = resolve_config(dict(profile="paper", variant="tcn_se", hidden_channels=8,
                                       target_fs=250, normalization_scale=2., output_delay=16,
                                       valid_history_samples=220, valid_future_samples=16))
        for run, cfg in zip(self.runs, ({"target_fs": 250, "normalization": "epoch"}, protocol)):
            folder = config_io.RUNS_DIR / run
            folder.mkdir(parents=True)
            torch.save(dict(model_state=model.state_dict(), model_config=resolved_arch(model),
                            run_config=cfg, epoch=0), folder / "best.pt")

        def sleep(seconds):
            nonlocal clock
            self.assertEqual(mode, "paced", "Fast replay must process hops without waiting")
            clock += seconds

        timer = SimpleNamespace(perf_counter=lambda: clock, sleep=sleep)
        board = {"board": "Raspberry Pi 5 Model B (test fixture)", "verdict": "locked"}
        runs = {}
        for mode in ("paced", "fast"):
            remote = LoopbackRemote()
            clock = 0.
            spec = self.spec if mode == "paced" else {**self.spec, "replay_mode": "fast"}
            command = eegio_pi.build_command(spec)
            stdout = io.StringIO()
            with patch.object(eegio_pi, "Remote", return_value=remote), \
                 patch.object(preflight, "pin"), patch.object(preflight, "check", return_value=board), \
                 patch.object(eegio_replay, "time", timer), contextlib.redirect_stdout(stdout):
                self.assertEqual(eegio_pi.main([command[-1]]), 0, stdout.getvalue())
            self.assertEqual(remote.cleaned, len(self.runs))
            entry = eegio_tab.list_results()[0]
            self.assertEqual((entry["target"], entry["replay_mode"]), ("pi5", mode))
            result = eegio_tab.read_result(entry["name"])
            self.assertEqual(result["saved"]["replay_mode"], mode)
            self.assertIn("[eegio-result] " + result["name"], stdout.getvalue())
            self.assertEqual(result["metric_protocol"], eegio_eval.METRIC_PROTOCOL)
            runs[mode] = (result["results"], remote.traces)

        cfg = eegio_eval.EvalConfig(target_fs=250)
        rec = read_recording(str(self.data), "S00")
        host_input = eegio_eval.preprocess(rec.signal[0], rec.fs)
        paced_results, paced_traces = runs["paced"]
        fast_results, fast_traces = runs["fast"]
        for index, run in enumerate(self.runs):
            with self.subTest(run=run):
                filtered, output = paced_traces[index]
                fast_input, fast_output = fast_traces[index]
                np.testing.assert_array_equal(filtered, host_input)
                np.testing.assert_array_equal(fast_input, filtered)
                np.testing.assert_allclose(fast_output, output, atol=2e-4, rtol=1e-5)
                loaded, _ = eegio_eval.load_run(run)
                delayed = run == self.runs[1]
                if delayed:
                    output = aligned_output(torch.from_numpy(output), protocol).numpy()
                np.testing.assert_allclose(output, eegio_eval.stream_apply(loaded, host_input, hop=cfg.hop),
                                           atol=2e-4, rtol=1e-5)
                host_row = eegio_eval.evaluate_subjects(str(self.data), run, ["S00"], cfg)["rows"][0]
                row = paced_results[index]["rows"][0]
                fast_row = fast_results[index]["rows"][0]
                self.assertEqual(row["valid_target_interval"], host_row["valid_target_interval"])
                self.assertEqual(fast_row["valid_target_interval"], row["valid_target_interval"])
                for metric in eegio_eval.METRIC_PATHS:
                    self.assertAlmostEqual(row[metric], host_row[metric], places=4)
                    self.assertAlmostEqual(fast_row[metric], row[metric], places=4)
                report = row["pi_replay"]
                self.assertEqual(report["backend"], "onnx_step" if delayed else "onnx_windowed")
                self.assertTrue(report["paced"])
                self.assertEqual(report["deadline_misses"], 0)
                self.assertGreater(report["steady_hops"], 0)
                self.assertAlmostEqual(report["observation_s"], rec.duration_s)
                fast = fast_row["pi_replay"]
                self.assertFalse(fast["paced"])
                self.assertIsNone(fast["deadline_misses"])
                self.assertIsNone(fast["hop_budget_ms"])
                if delayed:
                    self.assertEqual(fast["steady_hops"], report["steady_hops"])
                else:
                    self.assertLess(fast["inference_window_samples"], cfg.win)
                    self.assertGreater(fast["window_batch_size"], 1)
                    self.assertIsNone(fast["processing_p99_ms"])
                self.assertEqual(fast["observation_s"], 0.)


class EEGIOPiLookaheadAndStepTests(unittest.TestCase):
    """
    A bounded-lookahead model replayed windowed, and a causal model replayed
    through its one-sample step graph at H=1, both through the GUI runner.

    The reference is independent of the replay code: the PyTorch model run
    once over the whole preprocessed recording. A causal conv model with
    finite context gives the same output at every sample whichever window it
    is computed in, provided the window holds its past and its L future
    samples -- which is exactly what the emission rule promises. A cumulative
    SE run through the step graph carries its running mean from the first
    sample, which is again the whole-recording forward pass.
    """

    SCALE = 2.0

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.enterContext(patch.object(eegio_eval, "PROJECT_ROOT", self.root))
        self.enterContext(patch.object(config_io, "RUNS_DIR", self.root / "runs"))
        self.enterContext(patch.object(eegio_tab, "RESULTS_DIR", self.root / "results"))
        self.data = self.root / "eegio"
        write_synthetic_recording(self.data, duration_s=16.012, n_blinks=1, n_soft=0, amplitude=60)
        self.enterContext(patch.object(eegio_tab, "DATA_DIR_CANDIDATES", (self.data,)))
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        cfg = {"target_fs": 250, "normalization": "fixed", "normalization_scale": self.SCALE}
        self.models = {}
        for run, variant, lookahead in (("causal_cumulative_se", "tcn_se", None),
                                        ("lookahead8", "tcn_la8", 8)):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(11)
                model = build_model(variant, hidden_channels=8, tcn_norm="channel").eval()
            folder = config_io.RUNS_DIR / run
            folder.mkdir(parents=True)
            torch.save(dict(model_state=model.state_dict(), model_config=resolved_arch(model),
                            run_config=cfg, epoch=0), folder / "best.pt")
            self.models[run] = (model, lookahead)
        self.enterContext(patch.object(eegio_tab, "available_runs", return_value=[
            {"name": run, "train_fs": 250., "causal": lookahead is None, "lookahead": lookahead,
             "bidirectional": False, "train_fs_recorded": True, "training_protocol": "exploratory"}
            for run, (_, lookahead) in self.models.items()
        ]))
        rec = read_recording(str(self.data), "S00")
        self.filtered = eegio_eval.preprocess(rec.signal[0], rec.fs)
        self.duration_s = rec.duration_s

    def reference(self, run):
        model = self.models[run][0]
        with torch.no_grad():
            x = torch.from_numpy(self.filtered / self.SCALE)[None, None].float()
            return model(x)[0, 0].numpy() * self.SCALE

    def replay(self, runs, **change):
        clock = 0.

        def sleep(seconds):
            nonlocal clock
            clock += seconds

        timer = SimpleNamespace(perf_counter=lambda: clock, sleep=sleep)
        board = {"board": "Raspberry Pi 5 Model B (test fixture)", "verdict": "locked"}
        remote = LoopbackRemote()
        spec = {**dict(runs=runs, target_fs=250, n_subjects=1, hop=4, ssh="pi@host"), **change}
        stdout = io.StringIO()
        with patch.object(eegio_pi, "Remote", return_value=remote), \
             patch.object(preflight, "pin"), patch.object(preflight, "check", return_value=board), \
             patch.object(eegio_replay, "time", timer), contextlib.redirect_stdout(stdout):
            self.assertEqual(eegio_pi.main([eegio_pi.build_command(spec)[-1]]), 0, stdout.getvalue())
        result = eegio_tab.read_result(eegio_tab.list_results()[0]["name"])
        return result, remote.traces

    def test_requests_the_runner_must_refuse(self):
        base = dict(target_fs=250, n_subjects=1, hop=4, ssh="pi@host")
        with self.assertRaisesRegex(ValueError, "causal-only"):
            eegio_pi.build_command({**base, "runs": ["lookahead8"], "incremental": True})
        with self.assertRaises(ValueError):
            eegio_pi.build_command({**base, "runs": ["causal_cumulative_se"], "incremental": "yes"})
        with patch.object(eegio_tab, "available_runs", return_value=[
                {"name": "teacher", "train_fs": 250., "causal": False, "lookahead": None,
                 "bidirectional": True, "train_fs_recorded": True}]):
            with self.assertRaisesRegex(ValueError, "bounded-lookahead"):
                eegio_pi.build_command({**base, "runs": ["teacher"]})
        command = eegio_pi.build_command({**base, "runs": ["causal_cumulative_se", "lookahead8"]})
        self.assertFalse(json.loads(command[-1])["incremental"])

    def test_lookahead_outputs_match_the_whole_recording_forward_pass(self):
        reference = self.reference("lookahead8")
        for mode, hop in (("fast", 1), ("fast", 4), ("paced", 4)):
            with self.subTest(mode=mode, hop=hop):
                result, traces = self.replay(["lookahead8"], hop=hop, replay_mode=mode)
                filtered, output = traces[0]
                np.testing.assert_array_equal(filtered, self.filtered)
                # Every sample, the final L flushed from the last window included.
                np.testing.assert_allclose(output, reference, atol=2e-4, rtol=1e-4)
                report = result["results"][0]["rows"][0]["pi_replay"]
                self.assertEqual(report["lookahead_samples"], 8)
                self.assertAlmostEqual(report["lookahead_wait_ms"], 32.0)
                self.assertEqual(report["backend"], "onnx_windowed")
                self.assertEqual(report["paced"], mode == "paced")

    def test_emitting_without_the_lookahead_delay_would_be_caught(self):
        # Guards the test above: the newest hop of each window, as a causal
        # model emits it, is not the whole-recording output for an L=8 model.
        model, _ = self.models["lookahead8"]
        x = torch.from_numpy(self.filtered[:500] / self.SCALE)[None, None].float()
        with torch.no_grad():
            window = model(x)[0, 0].numpy() * self.SCALE
        reference = self.reference("lookahead8")
        self.assertGreater(np.abs(window[-4:] - reference[496:500]).max(), 1e-3)

    def test_step_graph_at_hop_one_carries_state_across_the_recording(self):
        reference = self.reference("causal_cumulative_se")
        for mode in ("fast", "paced"):
            with self.subTest(mode=mode):
                result, traces = self.replay(["causal_cumulative_se"], hop=1,
                                             replay_mode=mode, incremental=True)
                np.testing.assert_allclose(traces[0][1], reference, atol=2e-4, rtol=1e-4)
                self.assertTrue(result["saved"]["incremental"])
                self.assertTrue(eegio_tab.list_results()[0]["incremental"])
                report = result["results"][0]["rows"][0]["pi_replay"]
                self.assertEqual((report["backend"], report["hop_samples"]), ("onnx_step", 1))
                self.assertEqual(report["lookahead_samples"], 0)
                if mode == "paced":
                    self.assertEqual(report["deadline_misses"], 0)

    def test_adjusted_blink_count_uses_the_baseline_threshold_for_every_model(self):
        result, _ = self.replay(["causal_cumulative_se", "lookahead8"], hop=4, replay_mode="fast")
        key = eegio_eval.ADJUSTED_BLINK_COUNT
        self.assertEqual(result["saved"]["blink_count_adjustment"]["reference_run"], "causal_cumulative_se")
        base, rung = (r["rows"][0] for r in result["results"])
        for row in (base, rung):
            self.assertIn(key, row)
            self.assertIn(key, result["results"][0]["summary"])
        threshold = base["metric_support"]["adjusted_blink_count"]["threshold_uv"]
        self.assertTrue(np.isfinite(threshold))
        self.assertEqual(rung["metric_support"]["adjusted_blink_count"]["threshold_uv"], threshold)
        # The input is counted against the same cutoff for both models.
        self.assertEqual(rung["metric_support"]["adjusted_blink_count"]["input_blink_count"],
                         base["metric_support"]["adjusted_blink_count"]["input_blink_count"])
        self.assertIn(key, {t["metric"] for t in result["tests"]})

    def test_fast_full_windows_are_the_paced_computation_batched(self):
        # A cumulative SE has no valid short graph, so fast replay batches
        # full windows; it must give exactly what the paced hop loop gives.
        outputs = {}
        for mode in ("paced", "fast"):
            result, traces = self.replay(["causal_cumulative_se"], hop=2, replay_mode=mode)
            outputs[mode] = traces[0][1]
            report = result["results"][0]["rows"][0]["pi_replay"]
            if mode == "fast":
                self.assertIsNone(report["inference_window_samples"])
                self.assertEqual(report["window_batch_size"], eegio_replay.FAST_WINDOW_BATCH)
        np.testing.assert_allclose(outputs["fast"], outputs["paced"], atol=2e-5, rtol=1e-5)


class EEGIOPiApiTests(unittest.TestCase):
    def request(self, path, body=None):
        from gui.server import Handler

        method = "GET" if body is None else "POST"
        payload = b"" if body is None else json.dumps(body).encode()
        client, handler = socket.socketpair()
        with client:
            client.settimeout(5)
            with handler:
                client.sendall(f"{method} {path} HTTP/1.0\r\nContent-Length: {len(payload)}\r\n\r\n".encode() + payload)
                Handler(handler, ("local", 0), SimpleNamespace(verbose=False))
            response = HTTPResponse(client)
            response.begin()
            return response.status, json.loads(response.read())

    def test_run_progress_and_stop_are_bound_to_the_pi_job(self):
        from gui import server

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        job = server.EEGIOPiJob()
        self.addCleanup(job.shutdown)
        self.enterContext(patch.object(server, "EEGIO_PI_JOB", job))
        self.enterContext(patch.object(server, "RUNS_DIR", Path(tmp.name)))
        worker = (
            "import time; "
            "print('[eegio-progress] {\"phase\": \"inference\", \"subject\": \"S00\"}'); "
            "print('[eegio-pi-samples] 4 250'); time.sleep(20)"
        )
        with patch.object(eegio_pi, "build_command", return_value=[sys.executable, "-u", "-c", worker]):
            status, started = self.request("/api/eegio/pi/run", {})
            self.assertEqual(status, 200)
            self.assertTrue(started["running"])
            self.assertEqual(self.request("/api/pi/run", {})[0], 409)
            self.assertEqual(self.request("/api/eegio/evaluate", {"target": "pi5"})[0], 400)
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline:
                _, polled = self.request("/api/eegio/pi/status")
                if (polled.get("progress") or {}).get("samples") == 4:
                    break
                time.sleep(.05)
            self.assertEqual(polled["progress"]["subject"], "S00")
            self.assertEqual(polled["progress"]["total_samples"], 250)
            self.assertEqual(self.request("/api/eegio/pi/stop", {"id": "stale"})[0], 409)
            status, stopped = self.request("/api/eegio/pi/stop", {"id": started["id"]})
            self.assertEqual(status, 200)
            self.assertTrue(stopped["stopped"])
            self.assertFalse(stopped["running"])
