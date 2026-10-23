"""The Deploy tab: commands, log reading, traces, routes, page.

Run: python -m unittest discover -s tests -p 'test_pi_tab.py'.
"""
import contextlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from deploy import preflight
from gui import pi, server
from gui.server import DeployJob, GUIServer

ROOT = Path(__file__).resolve().parents[1]


# The output of a real run of deploy/raspi_infer.py, copied verbatim.  The
# parser has no other contract than these strings: nothing in raspi_infer
# emits a machine-readable stream, and the tab fills in from this while a run
# is going.  (The board and thermal lines come from run_on_pi.sh, which only
# prints them on the Pi.)
REAL_LOG = """\
==> running on pi@raspberrypi.local
--> Raspberry Pi 5 Model B Rev 1.1 | Python 3.13.5 | onnxruntime 1.30 | 4 cores
--> before: 52.4°C  1500 MHz  throttled=0x0
[raspi] deploy/artifacts/tcn_mla_eog.onnx: window=512 fs=256 Hz  providers=['CPUExecutionProvider']  intra_op=2 (1 worker + the caller)  spinning=on
[bench] median 2.77 ms | p95 2.87 ms | max 3.14 ms per 512-sample window;  hop 32 = 125.0 ms budget -> 43.6x real-time headroom
[latency/testdata] hop    800 | last   2.84 ms (run   2.82) | last 64: median   2.83  p95   2.92  max   3.07 ms | misses 0/800 of 125.0 ms budget
[test] EOG_test100.npz: 100 epochs x 512 samples, SNR -7.0..+2.0 dB
                                   RRMSE_t  RRMSE_f      CC
       noisy input (no model)       1.4076   4.9135  0.5950
       epoch mode                   0.4601   0.4643  0.8826
       stream mode (hop 32)         0.5615   0.6582  0.8405
       epoch mode: 4.30 ms/epoch (batched)   stream mode: 2.79 ms/hop, 4.47 s for 200 s of signal
       note: the stream row concatenates independent epochs, so every window straddles two unrelated signals at different scales; it is a pessimistic bound, not what a continuous recording would give.
       per SNR (epoch mode):   SNR dB   n  RRMSE_t  RRMSE_f      CC
                                 -7.0  10   0.6534   0.7141  0.7798
                                 +2.0  10   0.4148   0.4931  0.9110
[latency/testdata] 1584 hops of 32 samples (125.0 ms budget each), 4.469 s wall, first window 3.141 ms
           per hop   : median 2.811 | p95 2.898 | p99 2.94 | max 3.071 ms  (jitter p95-median 0.087 ms)
           of which  : onnxruntime 2.792 ms median / 2.877 p95;  pre/post-processing 0.018 ms median
           real time : 0 deadline misses (0.00%), 43.13x headroom at p95, 354.43 hops/s, duty cycle 2.2% mean / 2.4% p99
           end-to-end: sample available -> output emitted  median 16.974 | p95 17.564 | max 18.121 ms  (drift over run +0.01 ms)
           SoC temp  : 52.4 -> 54.5 °C (peak 56.2)
[latency/testdata] wrote /home/pi/eeg/latency_tcn_mla_eog_hop32.json and /home/pi/eeg/latency_tcn_mla_eog_hop32.csv
[duty] 4 s paced replay per hop at 256 Hz, window 512, 2 onnxruntime threads
        hop   period hops/s |            compute ms |          duty % |          e2e ms   drift | misses | verdict
                  ms        | median    p99    max |   mean     p99 |  median     max      ms |        |
         32   125.00    8.0 |  13.60  15.06  15.11 |   10.6    12.0 |   21.32   25.21   -0.46 |      0 | OK
          8    31.25   32.0 |   3.75   5.42   6.44 |   12.2    17.3 |   12.84   14.08   +0.33 |      0 | OK
          4    15.62   64.0 |   2.90   5.20   8.81 |   19.1    33.3 |    7.13   12.66   -0.16 |      0 | OK
          2     7.81  128.0 |   2.76   5.66   8.96 |   37.2    72.4 |    3.77   15.85   -0.02 |      1 | FAIL
[duty] smallest hop that is comfortably real-time: 4 samples = 15.6 ms buffering + 7.1 ms compute = 22.8 ms total latency at 19% duty cycle
--> after:  54.5°C  1500 MHz  throttled=0x0   (peak during test: 56.2°C)
==> per-hop latency traces:
{trace_dir}/latency_tcn_mla_eog_hop32.json
{trace_dir}/latency_tcn_mla_eog_hop32_duty.json
==> done
"""


def scan(text: str) -> dict:
    s = pi.LogScan()
    for line in text.splitlines():
        s.feed(line)
    return s.state()


class CommandTests(unittest.TestCase):
    """What one submission of the form turns into, and what it refuses."""

    HOST = dict(target="host", model="demo", testdata="demo.npz", measure=True,
                threads=2, hop=32, latency_every=400)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "artifacts" / "pi_latency").mkdir(parents=True)
        (self.tmp / "testdata").mkdir()
        (self.tmp / "artifacts" / "demo.onnx").write_bytes(b"not really an onnx")
        (self.tmp / "testdata" / "demo.npz").write_bytes(b"not really an npz")
        self.patches = [
            mock.patch.object(pi, "ARTIFACTS_DIR", self.tmp / "artifacts"),
            mock.patch.object(pi, "TESTDATA_DIR", self.tmp / "testdata"),
            mock.patch.object(pi, "TRACE_DIR", self.tmp / "artifacts" / "pi_latency"),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_measure_and_sweep_are_two_invocations(self):
        """raspi_infer returns straight after --duty_cycle, so one call could
        not also score the test set."""
        steps = pi.build_steps({**self.HOST, "duty": True, "duty_hops": "64,32,8",
                                "duty_seconds": 20})
        self.assertEqual(len(steps), 2)
        first, second = (s["command"] for s in steps)
        self.assertEqual(first[:3], [sys.executable, "deploy/raspi_infer.py",
                                     "deploy/artifacts/demo.onnx"])
        self.assertIn("--benchmark", first)
        self.assertIn("--latency", first)
        self.assertEqual(first[first.index("--testdata") + 1], "deploy/testdata/demo.npz")
        self.assertNotIn("--duty_cycle", first)
        self.assertEqual(second[second.index("--duty_cycle") + 1], "64,32,8")
        self.assertEqual(second[second.index("--duty_seconds") + 1], "20")
        self.assertNotIn("--benchmark", second)
        # Both write to one stem, so the sweep's _duty.json lands beside the
        # per-hop trace of the same model and hop.
        stem = "deploy/artifacts/pi_latency/host_demo_hop32"
        self.assertEqual(first[first.index("--latency_out") + 1], stem)
        self.assertEqual(second[second.index("--latency_out") + 1], stem)

    def test_a_host_trace_is_named_so_it_cannot_pass_for_the_board(self):
        cmd = pi.build_steps(self.HOST)[0]["command"]
        self.assertTrue(cmd[cmd.index("--latency_out") + 1]
                        .startswith("deploy/artifacts/pi_latency/host_"))

    def test_the_export_names_the_artifact_after_its_run(self):
        """So the model the next step tests is the one just written."""
        with mock.patch.object(pi, "RUNS_DIR", self.tmp / "runs"):
            (self.tmp / "runs" / "tcn_mla_both").mkdir(parents=True)
            (self.tmp / "runs" / "tcn_mla_both" / "best.pt").write_bytes(b"x")
            steps = pi.build_steps({**self.HOST, "model": "", "export": True,
                                    "export_run": "tcn_mla_both"})
            self.assertEqual(len(steps), 2)
            self.assertEqual(steps[0]["command"][:4],
                             [sys.executable, "-m", "deploy.export_onnx", "--ckpt"])
            self.assertEqual(steps[0]["command"][4], "runs/tcn_mla_both/best.pt")
            self.assertEqual(steps[0]["command"][steps[0]["command"].index("--name") + 1],
                             "tcn_mla_both")
            # ... and the measurement runs against that file, which does not
            # exist yet: the step before it writes it.
            self.assertIn("deploy/artifacts/tcn_mla_both.onnx", steps[1]["command"])

    def test_the_pi_runs_one_script_with_its_settings_in_the_environment(self):
        steps = pi.build_steps({**self.HOST, "target": "pi", "ssh": "pi@raspberrypi.local",
                                "duty": True, "duty_hops": "64,8", "duty_seconds": 20,
                                "realtime": True, "realtime_s": 30,
                                "cpus": "0-1", "governor": "performance",
                                "require_locked": True})
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["command"],
                         ["bash", "deploy/run_on_pi.sh", "pi@raspberrypi.local", "demo"])
        self.assertEqual(steps[0]["env"], {
            # Empty unless the step graph was asked for: the Pi half reads it
            # under `set -u`, so it is always sent, never omitted.
            "INCREMENTAL": "",
            "REMOTE_DIR": "eeg", "TESTDATA": "demo.npz", "HOP": "32", "THREADS": "2",
            "LATENCY_EVERY": "400", "REALTIME_S": "30", "DUTY_HOPS": "64,8", "DUTY_S": "20",
            "CPUS": "0-1", "GOVERNOR": "performance", "REQUIRE_LOCKED": "1",
            "CHECK_ONLY": "", "SPIN": "1"})

    def test_the_sweep_is_off_by_an_empty_variable_not_a_missing_one(self):
        """run_on_pi.sh tests `-n "$DUTY_HOPS"`, so the empty string is how a
        run says 'no sweep'."""
        steps = pi.build_steps({**self.HOST, "target": "pi", "ssh": "pi@host"})
        self.assertEqual(steps[0]["env"]["DUTY_HOPS"], "")
        self.assertEqual(steps[0]["env"]["REALTIME_S"], "0")

    def test_the_preview_shows_the_environment_it_will_set(self):
        steps = pi.build_steps({**self.HOST, "target": "pi", "ssh": "pi@host"})
        line = pi.describe(steps)[0]["command"]
        self.assertTrue(line.startswith("REMOTE_DIR=eeg TESTDATA=demo.npz HOP=32"))
        self.assertIn("bash deploy/run_on_pi.sh pi@host demo", line)
        self.assertNotIn("DUTY_HOPS=", line)          # empty: not shown

    def test_refusals(self):
        bad = [
            # (spec, something the message has to say)
            ({**self.HOST, "model": "nope"}, "export it first"),
            ({**self.HOST, "model": "../../etc/passwd"}, "not an export name"),
            ({**self.HOST, "testdata": "missing.npz"}, "make_testdata"),
            ({**self.HOST, "measure": False}, "nothing to run"),
            ({**self.HOST, "duty": True, "duty_hops": "32,abc"}, "whole numbers"),
            ({**self.HOST, "duty": True, "duty_hops": ""}, "at least one hop"),
            ({**self.HOST, "threads": 999}, "between 1 and 16"),
            ({**self.HOST, "target": "elsewhere"}, "unknown target"),
            ({**self.HOST, "target": "pi", "ssh": ""}, "address is needed"),
            # A value that would arrive at ssh as a flag, or split into two
            # arguments, is not a host name.
            ({**self.HOST, "target": "pi", "ssh": "-oProxyCommand=x"}, "not a [user@]host"),
            ({**self.HOST, "target": "pi", "ssh": "pi@host; rm -rf /"}, "not a [user@]host"),
            ({**self.HOST, "target": "pi", "ssh": "a@b@c"}, "not a [user@]host"),
            ({**self.HOST, "target": "pi", "ssh": "pi@host", "remote_dir": "../etc"},
             "not a directory name"),
            # run_on_pi.sh takes two positional arguments and nothing else, so
            # flags meant for raspi_infer would be silently dropped.
            ({**self.HOST, "target": "pi", "ssh": "pi@host", "extra_args": "--benchmark"},
             "no extra flags"),
            ({**self.HOST, "export": True, "export_run": "", "model": ""}, "pick a model"),
            ({**self.HOST, "export": True, "export_run": "../evil"}, "invalid run name"),
        ]
        for spec, expected in bad:
            with self.subTest(expected=expected):
                with self.assertRaises((ValueError, FileNotFoundError)) as cm:
                    pi.build_steps(spec)
                self.assertIn(expected, str(cm.exception))

    def test_a_pi_on_a_direct_link_can_be_addressed(self):
        """mDNS gives a Pi on a direct Ethernet cable an IPv6 link-local
        address and no IPv4, and a link-local address needs the interface
        after a %.  Refusing that form meant the tab could not drive the one
        setup it was built for."""
        for ssh in ("changlong@fe80::8aa2:9eff:fe56:5033%en5",
                    "changlong@[fe80::8aa2:9eff:fe56:5033%en5]",
                    "pi@192.168.1.42", "pi@raspberrypi.local", "raspberrypi.local"):
            with self.subTest(ssh=ssh):
                steps = pi.build_steps({**self.HOST, "target": "pi", "ssh": ssh})
                self.assertEqual(steps[0]["command"][2], ssh)
        # ... and the things that must still be refused.
        for ssh in ("-oProxyCommand=x", "pi@host; rm -rf /", "pi@host`id`",
                    "pi@host $(id)", "a@b@c", "pi@host|tee"):
            with self.subTest(ssh=ssh), self.assertRaises(ValueError):
                pi.build_steps({**self.HOST, "target": "pi", "ssh": ssh})

    def test_extra_flags_are_appended_to_the_local_call(self):
        cmd = pi.build_steps({**self.HOST, "extra_args": "--max_batches 2"})[0]["command"]
        self.assertEqual(cmd[-2:], ["--max_batches", "2"])

    # ---- the locked state the reviewer asked for ----
    def test_every_measuring_step_carries_the_lock_settings(self):
        """Pinning and the thread count have to reach *every* process that
        times something, or one row of the report was taken differently from
        the next."""
        steps = pi.build_steps({**self.HOST, "cpus": "0-1", "require_locked": True,
                                "duty": True, "duty_hops": "32,8"})
        self.assertEqual(len(steps), 2)
        for step in steps:
            self.assertIn("--cpus", step["command"])
            self.assertEqual(step["command"][step["command"].index("--cpus") + 1], "0-1")
            self.assertIn("--require-locked", step["command"])
            # OMP is read when the kernels load, long before any flag is
            # parsed, so it has to be in the environment rather than on argv.
            self.assertEqual(step["env"]["OMP_NUM_THREADS"], "2")
            self.assertEqual(step["env"]["MKL_NUM_THREADS"], "2")

    def test_not_pinning_is_allowed_but_has_to_be_said(self):
        cmd = pi.build_steps({**self.HOST, "cpus": ""})[0]["command"]
        self.assertNotIn("--cpus", cmd)
        self.assertNotIn("--require-locked", cmd)

    def test_a_state_check_asks_for_nothing_it_does_not_need(self):
        """It is what you run *before* choosing a model, so it must not demand
        one — nor a test set, nor an export that does not exist yet."""
        steps = pi.build_steps({"target": "host", "check_only": True,
                                "threads": 2, "cpus": "0-1"})
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["command"],
                         [sys.executable, "-m", "deploy.preflight", "--threads", "2",
                          "--cpus", "0-1"])

    def test_a_state_check_on_the_pi_measures_nothing(self):
        steps = pi.build_steps({**self.HOST, "target": "pi", "ssh": "pi@host",
                                "check_only": True, "measure": True, "duty": True,
                                "duty_hops": "32"})
        self.assertEqual(steps[0]["env"]["CHECK_ONLY"], "1")
        self.assertIn("nothing is measured", steps[0]["note"])

    def test_the_lock_settings_are_validated_before_they_reach_a_process(self):
        bad = [
            ({**self.HOST, "cpus": "0-1; rm -rf /"}, "not a CPU list"),
            ({**self.HOST, "cpus": "3-1"}, "reversed CPU range"),
            # Two threads on one core contend, which is the opposite of what
            # pinning is for.
            ({**self.HOST, "cpus": "0", "threads": 2}, "would contend"),
            ({**self.HOST, "target": "pi", "ssh": "pi@host", "governor": "a; reboot"},
             "not a governor name"),
        ]
        for spec, expected in bad:
            with self.subTest(expected=expected):
                with self.assertRaises(ValueError) as cm:
                    pi.build_steps(spec)
                self.assertIn(expected, str(cm.exception))


class ModelListingTests(unittest.TestCase):
    """What the Deploy tab offers as a model.

    It used to list ``deploy/artifacts/*.onnx`` — two files, against
    twenty-nine trained runs, so twenty-seven models could not be chosen at
    all.  A run that has not been exported yet is still a model you can
    deploy; it just needs the export step first."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "artifacts").mkdir()
        (self.tmp / "runs").mkdir()
        self.patches = [mock.patch.object(pi, "ARTIFACTS_DIR", self.tmp / "artifacts"),
                        mock.patch.object(pi, "RUNS_DIR", self.tmp / "runs")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def export(self, name, age=0.0):
        p = self.tmp / "artifacts" / f"{name}.onnx"
        p.write_bytes(b"x")
        os.utime(p, (time.time() - age, time.time() - age))

    def run_dir(self, name, age=0.0):
        d = self.tmp / "runs" / name
        d.mkdir(exist_ok=True)
        (d / "best.pt").write_bytes(b"x")
        os.utime(d / "best.pt", (time.time() - age, time.time() - age))

    def models(self, runs=()):
        with mock.patch.object(pi.config_io, "list_runs", return_value=list(runs)):
            return {m["name"]: m for m in pi.list_models()}

    def test_a_trained_run_is_offered_even_with_no_export(self):
        self.run_dir("tcn_mla_both")
        m = self.models([{"name": "tcn_mla_both", "variant": "tcn_mla", "artifact": "BOTH",
                          "params": 109573, "bidirectional": False}])["tcn_mla_both"]
        self.assertTrue(m["run"])
        self.assertFalse(m["exported"])
        self.assertEqual(m["needs"], "export")
        self.assertTrue(m["deployable"])
        self.assertEqual(m["variant"], "tcn_mla")      # described from the run

    def test_an_export_and_its_run_are_one_model(self):
        self.run_dir("tcn_eog", age=600)
        self.export("tcn_eog")
        rows = self.models([{"name": "tcn_eog", "variant": "tcn", "artifact": "EOG",
                             "bidirectional": False}])
        self.assertEqual(len(rows), 1, "the export and its run listed twice")
        self.assertEqual(rows["tcn_eog"]["needs"], "none")

    def test_a_checkpoint_newer_than_its_export_is_stale(self):
        """The export no longer describes the run it is named after, and the
        numbers measured from it are not that run's."""
        self.export("tcn_mla_eog", age=600)
        self.run_dir("tcn_mla_eog")
        m = self.models([{"name": "tcn_mla_eog", "bidirectional": False}])["tcn_mla_eog"]
        self.assertTrue(m["exported"])
        self.assertTrue(m["stale"])
        self.assertEqual(m["needs"], "re-export")

    def test_a_non_causal_twin_is_listed_but_not_deployable(self):
        self.run_dir("tcn_bi_both")
        m = self.models([{"name": "tcn_bi_both", "variant": "tcn_bi",
                          "bidirectional": True}])["tcn_bi_both"]
        self.assertTrue(m["bidirectional"])
        self.assertFalse(m["deployable"])

    def test_an_export_with_no_run_still_lists(self):
        self.export("copied_in")
        self.assertIn("copied_in", self.models([]))

    def test_choosing_an_unexported_run_without_exporting_says_which_it_is(self):
        self.run_dir("tcn_mla_both")
        (self.tmp / "testdata").mkdir()
        (self.tmp / "testdata" / "d.npz").write_bytes(b"x")
        with mock.patch.object(pi, "TESTDATA_DIR", self.tmp / "testdata"), \
             mock.patch.object(pi.config_io, "list_runs",
                               return_value=[{"name": "tcn_mla_both", "bidirectional": False}]):
            with self.assertRaises(ValueError) as cm:
                pi.build_steps({"target": "host", "model": "tcn_mla_both",
                                "testdata": "d.npz", "measure": True})
            self.assertIn("no ONNX export yet", str(cm.exception))
            self.assertIn("tick 'export'", str(cm.exception))
            # ... and with it ticked, the export is of that same model.
            steps = pi.build_steps({"target": "host", "model": "tcn_mla_both",
                                    "testdata": "d.npz", "measure": True, "export": True})
            self.assertEqual(len(steps), 2)
            self.assertIn("runs/tcn_mla_both/best.pt", steps[0]["command"])
            self.assertIn("deploy/artifacts/tcn_mla_both.onnx", steps[1]["command"])

    def test_a_non_causal_model_is_refused_before_the_export_starts(self):
        self.run_dir("tcn_bi_both")
        with mock.patch.object(pi.config_io, "list_runs",
                               return_value=[{"name": "tcn_bi_both", "bidirectional": True}]):
            with self.assertRaises(ValueError) as cm:
                pi.build_steps({"target": "host", "model": "tcn_bi_both",
                                "export": True, "measure": True})
        self.assertIn("non-causal", str(cm.exception))


class LogScanTests(unittest.TestCase):
    """What the tab shows while a run is going, read out of its output."""

    def setUp(self):
        # The traces run_on_pi.sh copies back, where a real run would leave
        # them.  A path with spaces in it on purpose: this project's own
        # directory has one, and the Pi script announces absolute paths.
        self.tmp = Path(tempfile.mkdtemp()) / "a project" / "pi_latency"
        self.tmp.mkdir(parents=True)
        for name in ("latency_tcn_mla_eog_hop32.json", "latency_tcn_mla_eog_hop32_duty.json"):
            (self.tmp / name).write_text("{}")
        self.patch = mock.patch.object(pi, "TRACE_DIR", self.tmp)
        self.patch.start()
        self.state = scan(REAL_LOG.replace("{trace_dir}", str(self.tmp)))

    def tearDown(self):
        self.patch.stop()
        shutil.rmtree(self.tmp.parents[1], ignore_errors=True)

    def test_the_board_and_the_model_it_is_running(self):
        self.assertIn("Raspberry Pi 5 Model B", self.state["board"])
        self.assertEqual(self.state["model"],
                         {"onnx": "deploy/artifacts/tcn_mla_eog.onnx", "window": 512,
                          "sample_rate_hz": 256.0, "providers": "CPUExecutionProvider",
                          "threads": 2, "spinning": True})

    def test_the_older_shape_of_that_line_still_parses(self):
        """It used to read 'threads=2', before the difference between
        onnxruntime's intra-op count and OMP's mattered.  Logs already sitting
        in runs/ were written that way."""
        old = ("[raspi] m.onnx: window=512 fs=256 Hz  "
               "providers=['CPUExecutionProvider']  threads=2")
        model = scan(old)["model"]
        self.assertEqual(model["threads"], 2)
        self.assertIsNone(model["spinning"])
        self.assertEqual(self.state["thermal"]["before"]["temp_c"], 52.4)
        self.assertEqual(self.state["thermal"]["after"]["temp_c"], 54.5)

    def test_the_benchmark_line(self):
        self.assertEqual(self.state["bench"], {
            "median_ms": 2.77, "p95_ms": 2.87, "max_ms": 3.14, "window": 512,
            "hop": 32, "hop_budget_ms": 125.0, "realtime_headroom_x": 43.6})

    def test_the_latency_summary_including_the_parts_only_a_paced_run_has(self):
        lat = self.state["latency"][-1]
        self.assertEqual(lat["label"], "latency/testdata")
        self.assertEqual((lat["n_hops"], lat["hop"], lat["hop_budget_ms"]), (1584, 32, 125.0))
        self.assertEqual(lat["total_ms"], {"median": 2.811, "p95": 2.898,
                                           "p99": 2.94, "max": 3.071})
        self.assertEqual(lat["infer_ms"]["median"], 2.792)
        self.assertEqual(lat["overhead_ms"]["median"], 0.018)
        self.assertEqual(lat["deadline_misses"], 0)
        self.assertEqual(lat["realtime_headroom_x"], 43.13)
        self.assertEqual(lat["duty_cycle"], {"mean": 0.022, "p99": 0.024})
        self.assertEqual(lat["e2e_latency_ms"]["median"], 16.974)
        self.assertEqual(lat["e2e_drift_ms"], 0.01)
        self.assertEqual(lat["temp_c"], {"start": 52.4, "end": 54.5, "max": 56.2})

    def test_the_rolling_line_is_read_before_any_summary_exists(self):
        """It is the only sign of life during a long run, and it prints long
        before the summary that would otherwise create the record."""
        partial = scan(REAL_LOG.split("[test]")[0])
        self.assertEqual(partial["latency"], [])
        self.assertEqual(partial["progress"]["hops"], 800)
        self.assertEqual(partial["progress"]["deadline_misses"], 0)
        self.assertEqual(partial["progress"]["hop_budget_ms"], 125.0)

    def test_the_duty_rows_carry_the_fields_the_trace_file_does(self):
        """The panel draws either source with one renderer, which only works
        for as long as the parsed row and save_duty_rows' row agree."""
        rows = self.state["duty"]["rows"]
        self.assertEqual([r["hop"] for r in rows], [32, 8, 4, 2])
        self.assertEqual([r["verdict"] for r in rows], ["OK", "OK", "OK", "FAIL"])
        self.assertEqual(rows[-1], {
            "hop": 2, "hop_budget_ms": 7.81, "hops_per_s": 128.0,
            "compute_median_ms": 2.76, "compute_p99_ms": 5.66, "compute_max_ms": 8.96,
            "duty_mean": 0.372, "duty_p99": 0.724, "e2e_median_ms": 3.77,
            "e2e_max_ms": 15.85, "e2e_drift_ms": -0.02, "deadline_misses": 1,
            "verdict": "FAIL", "total_latency_median_ms": 11.58})
        trace = ROOT / "deploy" / "artifacts" / "pi_latency" / "latency_tcn_mla_eog_hop32_duty.json"
        if not trace.is_file():
            self.skipTest("no duty trace on this machine to compare the field names with")
        from_file = json.loads(trace.read_text())[0]
        missing = (set(from_file) - set(rows[0])) - {"n_hops", "temp_c"}
        self.assertEqual(missing, set(), f"the trace has fields the parser does not: {missing}")

    def test_the_table_header_is_not_mistaken_for_a_row(self):
        """Two header lines sit above the rows and both have pipes in them."""
        for line in REAL_LOG.splitlines():
            if "|" in line and "verdict" in line:
                self.assertIsNone(pi.LogScan._duty_row(line))
        self.assertIsNone(pi.LogScan._duty_row(
            "                  ms        | median    p99    max |   mean     p99 |"
            "  median     max      ms |        |"))

    def test_the_test_set_rows_and_the_per_snr_rows_stay_apart(self):
        t = self.state["test"]
        self.assertEqual((t["file"], t["n_epochs"], t["window"]), ("EOG_test100.npz", 100, 512))
        self.assertEqual([r["name"] for r in t["rows"]],
                         ["noisy input (no model)", "epoch mode", "stream mode (hop 32)"])
        self.assertEqual(t["rows"][1], {"name": "epoch mode", "rrmse_t": 0.4601,
                                        "rrmse_s": 0.4643, "acc": 0.8826})
        self.assertEqual([r["snr_db"] for r in t["per_snr"]], [-7.0, 2.0])
        self.assertEqual(t["per_snr"][0]["n"], 10)

    def test_only_traces_that_landed_here_are_offered(self):
        """A run on the Pi writes its traces there and run_on_pi.sh copies them
        back; the Pi-side path (/home/pi/eeg/...) names a file on the board,
        which is not one this GUI can read."""
        self.assertEqual(self.state["traces"],
                         ["latency_tcn_mla_eog_hop32.json",
                          "latency_tcn_mla_eog_hop32_duty.json"])

    def test_a_trace_that_is_not_there_is_not_offered(self):
        (self.tmp / "latency_tcn_mla_eog_hop32.json").unlink()
        state = scan(REAL_LOG.replace("{trace_dir}", str(self.tmp)))
        self.assertEqual(state["traces"], ["latency_tcn_mla_eog_hop32_duty.json"])

    def test_the_printer_and_the_parser_agree(self):
        """Parse what LatencyMonitor actually prints, not a copy of it."""
        sys.path.insert(0, str(ROOT / "deploy"))
        try:
            import raspi_infer
        except ImportError:                      # numpy / onnxruntime not installed
            self.skipTest("deploy/raspi_infer.py cannot be imported here")
        finally:
            sys.path.pop(0)
        m = raspi_infer.LatencyMonitor(hop=32, sample_rate=256, label="latency/stream")
        m.record_first_window(0.0031)
        for i in range(50):
            m.record_hop(0.0028 + i * 1e-6, 0.0030 + i * 1e-6)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            summary = m.print_summary()
        got = scan(buf.getvalue())["latency"][-1]
        self.assertEqual(got["label"], "latency/stream")
        self.assertEqual(got["n_hops"], summary["n_hops"])
        self.assertEqual(got["hop_budget_ms"], summary["hop_budget_ms"])
        self.assertEqual(got["total_ms"], {k: summary["total_ms"][k]
                                           for k in ("median", "p95", "p99", "max")})
        self.assertEqual(got["infer_ms"]["median"], summary["infer_ms"]["median"])
        self.assertEqual(got["deadline_misses"], summary["deadline_misses"])
        self.assertEqual(got["realtime_headroom_x"], summary["realtime_headroom_x"])
        self.assertAlmostEqual(got["duty_cycle"]["mean"], summary["duty_cycle"]["mean"],
                               delta=0.001)

    def test_a_real_deploy_log_round_trips(self):
        """Parse a log this GUI actually produced, if one is lying around."""
        logs = [p for p in sorted((ROOT / "runs").glob("gui_deploy_*.log"))
                if "[raspi] " in p.read_text(errors="replace")]
        if not logs:
            self.skipTest("no runs/gui_deploy_*.log with a measurement in it")
        for log in logs:
            st = scan(log.read_text(errors="replace"))
            self.assertIsNotNone(st["model"], f"{log.name}: the model line was not read")
            self.assertTrue(st["bench"] or st["latency"] or st["duty"]["rows"],
                            f"{log.name}: no measurement was read out of it")


class PreflightTests(unittest.TestCase):
    """The check itself: what it reads, and what it refuses to call locked."""

    def tree(self, governors, cur_khz=1_500_000):
        """A fake /sys/devices/system/cpu for a machine with these governors."""
        root = Path(tempfile.mkdtemp())
        for cpu, gov in enumerate(governors):
            d = root / f"cpu{cpu}" / "cpufreq"
            d.mkdir(parents=True)
            (d / "scaling_governor").write_text(gov + "\n")
            (d / "scaling_cur_freq").write_text(f"{cur_khz}\n")
            (d / "scaling_max_freq").write_text("1500000\n")
            (d / "scaling_min_freq").write_text("1500000\n")
        (root / "online").write_text(f"0-{len(governors) - 1}\n")
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        return root

    RUNTIME = {"intra_op": 2, "workers": 1, "inter_op": 1, "execution_mode": "sequential",
               "allow_spinning": True, "openmp_in_build": False, "onnxruntime": "1.30.0"}

    def state(self, governors=("performance",) * 4, affinity=(2, 3), threads=2,
              omp="2", throttled="0x0", load=0.1, runtime=None, interrupts=None):
        with mock.patch.object(preflight, "CPU_ROOT", self.tree(list(governors))), \
             mock.patch.object(preflight, "_has_affinity", return_value=True), \
             mock.patch.object(preflight, "_affinity", return_value=list(affinity)), \
             mock.patch.object(preflight, "_throttled",
                               return_value={"raw": throttled, "bits": int(throttled, 16),
                                             "flags": []} if throttled else {}), \
             mock.patch.object(preflight, "_temperature", return_value=51.2), \
             mock.patch.object(os, "cpu_count", return_value=len(governors)), \
             mock.patch.dict(os.environ, {"OMP_NUM_THREADS": omp} if omp else {}, clear=False), \
             mock.patch.object(preflight, "_interrupts",
                               return_value=preflight._interrupts() if interrupts is None
                               else self._parse_irq(interrupts)), \
             mock.patch.object(os, "getloadavg", return_value=(load, load, load)):
            if not omp:
                os.environ.pop("OMP_NUM_THREADS", None)
            return preflight.check(threads, runtime=self.RUNTIME if runtime is None else runtime)

    @staticmethod
    def _parse_irq(text):
        with mock.patch.object(preflight.Path, "read_text", return_value=text):
            return preflight._interrupts()

    def test_cpu_lists_round_trip(self):
        for text, cpus in [("0-1", [0, 1]), ("0,2", [0, 2]), ("0-3", [0, 1, 2, 3]),
                           ("2", [2]), ("0-1,4", [0, 1, 4])]:
            self.assertEqual(preflight.parse_cpu_list(text), cpus)
            self.assertEqual(preflight.format_cpu_list(cpus), text)
        for bad in ("", "a", "0-", "1-0", "0,0", "0;1", "-1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                preflight.parse_cpu_list(bad)

    def test_a_board_that_is_held_still_passes(self):
        r = self.state()
        self.assertEqual(r["verdict"], preflight.LOCKED)
        self.assertEqual({c["name"]: c["status"] for c in r["checks"]},
                         {"governor": "pass", "affinity": "pass", "threads": "pass",
                          "thermal": "pass", "load": "pass"})

    def test_a_governor_that_can_ramp_fails(self):
        r = self.state(governors=("ondemand",) * 4)
        self.assertEqual(r["verdict"], preflight.UNLOCKED)
        gov = next(c for c in r["checks"] if c["name"] == "governor")
        self.assertEqual(gov["status"], "fail")
        self.assertIn("cpupower frequency-set -g performance", gov["detail"])

    def test_a_stray_core_outside_the_pin_set_is_said_but_does_not_decide(self):
        """The process cannot land on CPU 3, so its governor cannot move these
        numbers — but it is the usual sign that the governor was set for one
        run and not for the machine, so it is reported."""
        r = self.state(governors=("performance", "performance", "performance", "ondemand"),
                       affinity=(0, 1))
        self.assertEqual(r["verdict"], preflight.LOCKED)
        gov = next(c for c in r["checks"] if c["name"] == "governor")
        self.assertIn("CPU 0-1 at performance", gov["detail"])
        self.assertIn("outside the pin set are at ondemand", gov["detail"])

    def test_a_stray_core_does_decide_when_nothing_is_pinned(self):
        r = self.state(governors=("performance", "performance", "performance", "ondemand"),
                       affinity=(0, 1, 2, 3))
        self.assertEqual(r["verdict"], preflight.UNLOCKED)
        gov = next(c for c in r["checks"] if c["name"] == "governor")
        self.assertEqual(gov["status"], "fail")

    def test_not_being_pinned_fails(self):
        r = self.state(affinity=(0, 1, 2, 3))
        self.assertEqual(r["verdict"], preflight.UNLOCKED)
        aff = next(c for c in r["checks"] if c["name"] == "affinity")
        self.assertEqual(aff["status"], "fail")
        self.assertIn("--cpus", aff["detail"])

    # ---- staying off the core the kernel keeps busy ----
    INTERRUPTS = ("           CPU0       CPU1       CPU2       CPU3\n"
                  " 11:     840000       1000       1200        900   GICv2  timer\n"
                  " 38:     160000        500        400        300   GICv2  eth0\n"
                  "IPI0:      5000       5000       5000       5000   Rescheduling\n"
                  "ERR:          0\n")

    def test_the_housekeeping_core_is_measured_not_assumed(self):
        """'CPU 0 handles the OS chores' is received wisdom; /proc/interrupts
        is this board's own answer."""
        with mock.patch.object(preflight.Path, "read_text", return_value=self.INTERRUPTS):
            irq = preflight._interrupts()
        # The IPI row is inter-processor traffic, a consequence of the work
        # rather than housekeeping competing with it, so it is not counted.
        self.assertEqual(irq["per_cpu"], [1000000, 1500, 1600, 1200])
        self.assertAlmostEqual(irq["share"][0], 0.9957, places=3)
        state = {"interrupts": irq}
        self.assertEqual(preflight.housekeeping_cpus(state), [0])

    def test_pinning_onto_that_core_is_a_warning_with_the_number_in_it(self):
        r = self.state(affinity=(0, 1), interrupts=self.INTERRUPTS)
        aff = next(c for c in r["checks"] if c["name"] == "affinity")
        self.assertEqual(aff["status"], "warn")
        self.assertIn("housekeeping", aff["detail"])
        self.assertIn("CPU 0 100%", aff["detail"])      # this board's own share
        self.assertIn("--cpus 2-3", aff["detail"])      # and what to do instead
        self.assertEqual(r["verdict"], preflight.PARTIAL)

    def test_the_top_cores_pass_and_say_they_are_clear_of_it(self):
        r = self.state(affinity=(2, 3), interrupts=self.INTERRUPTS)
        aff = next(c for c in r["checks"] if c["name"] == "affinity")
        self.assertEqual(aff["status"], "pass")
        self.assertIn("clear of the housekeeping core", aff["detail"])
        self.assertEqual(r["verdict"], preflight.LOCKED)

    def test_the_default_counts_down_from_the_top(self):
        self.assertEqual(preflight.default_cpus(2, 4), "2-3")
        self.assertEqual(preflight.default_cpus(1, 4), "3")
        self.assertEqual(preflight.default_cpus(3, 4), "1-3")
        # More threads than cores cannot avoid CPU 0, and should not pretend to.
        self.assertEqual(preflight.default_cpus(4, 4), "0-3")
        self.assertEqual(preflight.default_cpus(8, 4), "0-3")

    def test_a_board_that_spreads_its_interrupts_has_no_housekeeping_core(self):
        even = ("           CPU0       CPU1       CPU2       CPU3\n"
                " 11:       1000       1000       1000       1000   GICv2  timer\n")
        with mock.patch.object(preflight.Path, "read_text", return_value=even):
            irq = preflight._interrupts()
        self.assertEqual(preflight.housekeeping_cpus({"interrupts": irq}), [])

    def test_no_interrupt_file_is_simply_no_evidence(self):
        self.assertEqual(preflight.housekeeping_cpus({"interrupts": {}}), [])
        r = self.state(affinity=(2, 3))                 # macOS: no /proc at all
        self.assertEqual(next(c for c in r["checks"]
                              if c["name"] == "affinity")["status"], "pass")

    def test_more_cores_than_threads_is_a_warning_not_a_pass(self):
        r = self.state(affinity=(0, 1, 2), threads=2)
        self.assertEqual(r["verdict"], preflight.PARTIAL)

    def test_fewer_cores_than_threads_fails(self):
        r = self.state(affinity=(0,), threads=2)
        self.assertEqual(r["verdict"], preflight.UNLOCKED)

    def test_omp_is_reported_but_is_not_the_control(self):
        """onnxruntime is built without OpenMP, so OMP_NUM_THREADS governs
        none of the timed path — requiring it to match would have been
        asserting a setting that does nothing to what is being measured.  It
        still governs numpy's BLAS in the same process, so it is reported."""
        rt = {"intra_op": 2, "workers": 1, "allow_spinning": True, "openmp_in_build": False}
        for omp in ("8", None):
            r = self.state(omp=omp, runtime=rt)
            self.assertEqual(r["verdict"], preflight.LOCKED, f"OMP_NUM_THREADS={omp}")
            threads = next(c for c in r["checks"] if c["name"] == "threads")
            self.assertEqual(threads["status"], "pass")
            self.assertIn("no OpenMP" if omp is None else "not onnxruntime", threads["detail"])

    def test_the_control_is_the_intra_op_count_the_session_really_has(self):
        rt = {"intra_op": 8, "workers": 7, "allow_spinning": True, "openmp_in_build": False}
        r = self.state(threads=2, runtime=rt)
        threads = next(c for c in r["checks"] if c["name"] == "threads")
        self.assertEqual(threads["status"], "fail")
        self.assertIn("running 8", threads["detail"])
        self.assertEqual(r["verdict"], preflight.UNLOCKED)

    def test_the_count_is_reported_as_onnxruntime_counts_it(self):
        """intra_op_num_threads = N is N in total: N-1 workers plus the thread
        making the call.  That is why N threads want N pinned cores, and it is
        the part that is easy to get wrong."""
        rt = {"intra_op": 2, "workers": 1, "allow_spinning": False, "openmp_in_build": False}
        detail = next(c for c in self.state(runtime=rt)["checks"]
                      if c["name"] == "threads")["detail"]
        self.assertIn("1 worker + the caller", detail)
        self.assertIn("not spinning", detail)

    def test_a_board_that_has_throttled_fails(self):
        r = self.state(throttled="0x50000")
        self.assertEqual(r["verdict"], preflight.UNLOCKED)

    def test_a_machine_with_no_knobs_is_unlocked_not_excused(self):
        """macOS has neither control. 'n/a' on a control must not read as a
        pass: the figures are still not reproducible."""
        with mock.patch.object(preflight, "CPU_ROOT", Path("/nonexistent")), \
             mock.patch.object(preflight, "_affinity", return_value=None), \
             mock.patch.object(preflight, "_throttled", return_value={}), \
             mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "2"}):
            r = preflight.check(2)
        self.assertEqual(r["verdict"], preflight.UNLOCKED)
        self.assertEqual([c["status"] for c in r["checks"] if c["name"] == "governor"], ["n/a"])
        self.assertIn("indicative only", r["summary"])

    def test_the_summary_is_written_to_be_pasted_into_a_report(self):
        r = self.state()
        self.assertIn("performance", r["summary"])
        self.assertIn("pinned to CPU 2-3", r["summary"])
        self.assertIn("2 onnxruntime intra-op threads", r["summary"])
        self.assertIn("the calling thread", r["summary"])
        self.assertIn("because", r["summary"])          # it says why, not only what

    def test_the_printed_block_and_the_json_line_say_the_same_thing(self):
        r = self.state()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            preflight.emit(r)
        lines = buf.getvalue().splitlines()
        self.assertTrue(lines[0].startswith("[preflight] "))
        self.assertIn("LOCKED", lines[0])
        payload = [l for l in lines if l.startswith(preflight.JSON_PREFIX)]
        self.assertEqual(len(payload), 1)
        self.assertEqual(json.loads(payload[0][len(preflight.JSON_PREFIX):])["verdict"],
                         r["verdict"])

    def test_the_gui_reads_the_verdict_off_that_line(self):
        """The one contract between the checker and the tab."""
        r = self.state()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            preflight.emit(r)
        got = scan(buf.getvalue())["environment"]
        self.assertEqual(got["verdict"], preflight.LOCKED)
        self.assertEqual([c["name"] for c in got["checks"]],
                         ["governor", "affinity", "threads", "thermal", "load"])

    def test_the_governor_the_form_offers_is_the_one_that_is_checked(self):
        self.assertEqual(pi.WANTED_GOVERNOR, preflight.WANTED_GOVERNOR)
        self.assertEqual(pi.meta()["defaults"]["governor"], preflight.WANTED_GOVERNOR)

    def test_pinning_refuses_a_cpu_that_is_not_there(self):
        if not hasattr(os, "sched_setaffinity"):
            self.skipTest("no affinity API on this platform")
        with self.assertRaises(ValueError):
            preflight.pin(f"{(os.cpu_count() or 1) + 99}")


class OnnxThreadingTests(unittest.TestCase):
    """What onnxruntime's threading actually does, asserted against the real
    runtime rather than against what OpenMP would have done.

    Everything the preflight check says about threads rests on these, and all
    of them are things it is easy to assume wrongly."""

    @classmethod
    def setUpClass(cls):
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("onnxruntime is not installed")
        cls.onnx = ROOT / "deploy" / "artifacts" / "tcn_mla_eog.onnx"
        if not cls.onnx.is_file():
            raise unittest.SkipTest("no exported model on this machine")
        sys.path.insert(0, str(ROOT / "deploy"))
        try:
            import raspi_infer
        finally:
            sys.path.pop(0)
        cls.raspi_infer = raspi_infer

    def test_the_wheel_carries_no_openmp(self):
        """Which is why OMP_NUM_THREADS is reported and not required."""
        import onnxruntime as ort
        self.assertNotIn("openmp", ort.get_build_info().lower())

    def test_the_thread_count_includes_the_thread_making_the_call(self):
        """onnxruntime starts intra_op_num_threads - 1 workers and runs the
        rest on the caller — so N threads occupy N cores, which is the ratio
        the --cpus check enforces.  onnxruntime asserts the same thing itself:
        it refuses an affinity list that is not exactly N-1 long."""
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads = 2
        o.add_session_config_entry("session.intra_op_thread_affinities", "1;2")   # 2, not 1
        o.log_severity_level = 4          # the refusal is the point; don't print it
        with self.assertRaises(Exception) as cm:
            ort.InferenceSession(str(self.onnx), o, providers=["CPUExecutionProvider"])
        self.assertIn("thread_pool_size", str(cm.exception))

    def test_the_session_records_what_it_really_has(self):
        d = self.raspi_infer.EEGDenoiser(self.onnx, n_threads=3, spin=False)
        self.assertEqual(d.threads["intra_op"], 3)
        self.assertEqual(d.threads["workers"], 2)          # 3 - 1, the caller is the third
        self.assertFalse(d.threads["allow_spinning"])
        self.assertFalse(d.threads["openmp_in_build"])
        self.assertEqual(d.threads["execution_mode"], "sequential")

    def test_one_thread_spawns_no_workers(self):
        d = self.raspi_infer.EEGDenoiser(self.onnx, n_threads=1)
        self.assertEqual(d.threads["workers"], 0)

    def test_spinning_is_recorded_either_way(self):
        """It does not change whether the loop keeps up, but it changes how
        much of the board is left for anything else — so a trace that does not
        say which way it ran cannot be read as a deployment claim."""
        for spin in (True, False):
            d = self.raspi_infer.EEGDenoiser(self.onnx, n_threads=2, spin=spin)
            self.assertIs(d.threads["allow_spinning"], spin)


class TraceTests(unittest.TestCase):
    """The files a run leaves behind, which are what the panels read once the
    run is over."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.patch = mock.patch.object(pi, "TRACE_DIR", self.tmp)
        self.patch.start()
        (self.tmp / "host_demo_hop32.json").write_text(json.dumps({
            "hop": 32, "n_hops": 4, "hop_budget_ms": 125.0,
            "total_ms": {"median": 2.8}, "deadline_misses": 0}))
        (self.tmp / "host_demo_hop32.csv").write_text(
            "hop,infer_ms,total_ms,t_s,temp_c\n"
            "1,1.0,2.0,0.1,50.0\n2,1.1,9.9,0.2,50.5\n3,1.2,2.2,0.3,nan\n4,1.3,2.3,0.4,nan\n")
        (self.tmp / "host_demo_hop32_duty.json").write_text(json.dumps([
            {"hop": 32, "verdict": "OK"}, {"hop": 4, "verdict": "FAIL"}]))

    def tearDown(self):
        self.patch.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_a_sweep_and_a_per_hop_trace_are_told_apart(self):
        rows = {r["name"]: r for r in pi.list_traces()}
        self.assertEqual(rows["host_demo_hop32.json"]["kind"], "latency")
        self.assertEqual(rows["host_demo_hop32.json"]["median_ms"], 2.8)
        self.assertTrue(rows["host_demo_hop32.json"]["csv"])
        self.assertTrue(all(r["host"] for r in rows.values()))
        self.assertEqual(rows["host_demo_hop32_duty.json"]["kind"], "duty")
        self.assertEqual(rows["host_demo_hop32_duty.json"]["best_hop"], 32)

    def test_the_series_keeps_the_outliers(self):
        """Bucketing down to a drawable number of points by plain decimation
        would drop exactly the stalls the curve is read for."""
        t = pi.read_trace("host_demo_hop32.json", max_points=2)
        self.assertEqual(t["series"]["n_hops"], 4)
        self.assertEqual(t["series"]["bucket"], 2)
        points = t["series"]["points"]
        self.assertEqual(len(points), 2)
        self.assertEqual(points[0]["total_ms_max"], 9.9)
        self.assertEqual(points[0]["total_ms"], 9.9)       # median of two is the upper one
        self.assertEqual([p["hop"] for p in points], [2, 4])
        # A board with no thermal zone writes nan; those are dropped rather
        # than drawn as zero.
        self.assertEqual(points[0]["temp_c"], 50.5)
        self.assertNotIn("temp_c", points[1])

    def test_a_sweep_comes_back_as_rows(self):
        t = pi.read_trace("host_demo_hop32_duty.json")
        self.assertEqual(t["kind"], "duty")
        self.assertEqual([r["hop"] for r in t["rows"]], [32, 4])
        self.assertNotIn("series", t)

    def test_a_name_outside_the_trace_directory_is_refused(self):
        for bad in ("", "../../etc/passwd", "sub/../../x.json", "nope.json"):
            with self.subTest(name=bad):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    pi.read_trace(bad)


class StepSequenceTests(unittest.TestCase):
    """A deploy run is several commands but one job: one log to read, and a
    step that fails stops the ones after it."""

    def steps(self, *scripts):
        return [{"label": f"step {i}", "command": [sys.executable, "-c", s]}
                for i, s in enumerate(scripts)]

    def wait(self, job, timeout=10):
        for _ in range(int(timeout / 0.05)):
            if not job.running:
                return
            time.sleep(0.05)
        self.fail("the job did not finish")

    def test_steps_run_in_order_and_share_one_log(self):
        job = DeployJob()
        job.start_steps(self.steps("print('one')", "print('two')"), ROOT)
        self.wait(job)
        s = job.status(0)
        self.assertEqual(s["returncode"], 0)
        self.assertEqual([l for l in s["lines"] if l in ("one", "two")], ["one", "two"])
        self.assertEqual([st["returncode"] for st in s["steps"]], [0, 0])
        self.assertIn("=== step 1/2: step 0 ===", s["lines"])

    def test_a_failing_step_stops_the_ones_after_it(self):
        job = DeployJob()
        job.start_steps(self.steps("import sys; sys.exit(3)", "print('never')"), ROOT)
        self.wait(job)
        s = job.status(0)
        self.assertEqual(s["returncode"], 3)
        self.assertNotIn("never", s["lines"])
        self.assertEqual([st["returncode"] for st in s["steps"]], [3, None])
        self.assertTrue(any("the remaining steps were not run" in l for l in s["lines"]))

    def test_stop_between_steps_does_not_let_the_next_one_start(self):
        job = DeployJob()
        job.start_steps(self.steps("import time; print('x', flush=True); time.sleep(30)",
                                   "print('never')"), ROOT)
        for _ in range(100):
            if "x" in job.status(0)["lines"]:
                break
            time.sleep(0.05)
        self.assertTrue(job.stop())
        self.wait(job)
        self.assertNotIn("never", job.status(0)["lines"])
        self.assertFalse(job.running)

    def test_a_stop_that_lands_before_the_first_process_exists_is_still_a_stop(self):
        """start_steps spawns on its own thread, so a Stop can arrive with no
        process to terminate yet — and must still prevent the run."""
        job = DeployJob()
        job.start_steps(self.steps("print('never')"), ROOT)
        job.stop()
        self.wait(job)
        self.assertNotIn("never", job.status(0)["lines"])

    # ---- stopping a step that is not one process ----
    # A Deploy run is the only job that runs a shell script, and the script
    # forks: run_on_pi.sh starts ssh and scp.  Those inherit the job's output
    # pipe, and the pipe does not close while any of them holds it — so a
    # SIGTERM to the direct child left the reader blocked, the job stuck on
    # "running", and the Stop button with nothing further to do.
    FORKING = ["bash", "-c", "sleep 60 & echo started; sleep 60"]
    STUBBORN = ["bash", "-c", "trap '' TERM; echo started; sleep 60"]

    def started(self, job, timeout=10):
        for _ in range(int(timeout / 0.05)):
            if any(l for l in job.status(0)["lines"]):
                return
            time.sleep(0.05)
        self.fail("the step never produced a line")

    def test_a_step_that_forks_still_stops(self):
        job = DeployJob()
        job.start_steps([{"label": "forks", "command": self.FORKING}], ROOT)
        self.started(job)
        t0 = time.perf_counter()
        self.assertTrue(job.stop())
        self.wait(job)
        # Before the fix this took 60 s — the lifetime of the orphan — because
        # nothing closed the pipe.
        self.assertLess(time.perf_counter() - t0, 8)
        self.assertNotEqual(job.status(0)["returncode"], 0)

    def test_a_step_that_ignores_sigterm_is_killed(self):
        job = DeployJob()
        job.start_steps([{"label": "stubborn", "command": self.STUBBORN}], ROOT)
        self.started(job)
        self.assertTrue(job.stop(grace=0.5))
        self.wait(job)
        self.assertFalse(job.running)

    def test_a_forking_step_is_stopped_mid_sequence(self):
        job = DeployJob()
        job.start_steps([{"label": "forks", "command": self.FORKING},
                         {"label": "after", "command": [sys.executable, "-c", "print('never')"]}],
                        ROOT)
        self.started(job)
        self.assertTrue(job.stop())
        self.wait(job)
        self.assertNotIn("never", job.status(0)["lines"])

    def test_the_server_is_not_in_the_group_that_gets_signalled(self):
        """The one way this fix could go badly: if the step did not get a
        session of its own, its process group is the GUI's, and signalling the
        group would take the server down with it."""
        job = DeployJob()
        job.start_steps([{"label": "x", "command": [sys.executable, "-c",
                                                    "import time; print('x', flush=True); time.sleep(30)"]}],
                        ROOT)
        self.started(job)
        proc = job._proc
        self.assertIsNotNone(proc)
        self.assertEqual(job._session_of(proc), proc.pid,
                         "the step is not its own session leader")
        self.assertNotEqual(job._session_of(proc), os.getpgid(0))
        job.stop()
        self.wait(job)

    def test_a_reader_that_cannot_be_freed_is_let_go_of_not_left_running(self):
        """Anything that daemonises escapes the session and keeps the pipe.
        A job that cannot be stopped is the worse outcome, so the reader is
        abandoned — and the generation counter keeps its output out of
        whatever runs next."""
        escape = ("import os, time\n"
                  "print('started', flush=True)\n"
                  "if os.fork() == 0:\n"
                  "    os.setsid()\n"
                  "    for i in range(60):\n"
                  "        print('ORPHAN', flush=True); time.sleep(0.1)\n"
                  "    os._exit(0)\n"
                  "time.sleep(60)\n")
        job = DeployJob()
        job.start_steps([{"label": "escapes", "command": [sys.executable, "-c", escape]}], ROOT)
        self.started(job)
        self.assertTrue(job.stop(grace=0.2))
        self.wait(job, timeout=5)
        self.assertFalse(job.running)
        self.assertTrue(any("holding its output pipe" in l for l in job.status(0)["lines"]))

        job.start_steps([{"label": "next", "command": [sys.executable, "-c",
                          "import time; time.sleep(1); print('clean')"]}], ROOT)
        self.wait(job)
        self.assertNotIn("ORPHAN", " ".join(job.status(0)["lines"]))

    def test_a_step_cannot_wait_on_the_servers_stdin(self):
        """A subprocess that unexpectedly asks a question — ssh wanting a
        password — would otherwise read the GUI's own stdin and sit there."""
        job = DeployJob()
        job.start_steps([{"label": "reads stdin", "command": [
            sys.executable, "-c", "import sys; print(repr(sys.stdin.read()))"]}], ROOT)
        self.wait(job, timeout=10)
        self.assertEqual(job.status(0)["returncode"], 0)
        self.assertIn("''", job.status(0)["lines"])

    def test_stop_on_an_idle_job_says_so(self):
        job = DeployJob()
        self.assertFalse(job.stop())

    def test_the_environment_of_one_step_is_its_own(self):
        job = DeployJob()
        job.start_steps([{"label": "env", "env": {"PI_TEST_VAR": "42"},
                          "command": [sys.executable, "-c",
                                      "import os; print(os.environ.get('PI_TEST_VAR'))"]},
                         {"label": "no env", "command": [sys.executable, "-c",
                                                         "import os; print(os.environ.get('PI_TEST_VAR'))"]}],
                        ROOT)
        self.wait(job)
        lines = [l for l in job.status(0)["lines"] if l in ("42", "None")]
        self.assertEqual(lines, ["42", "None"])


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = GUIServer("127.0.0.1", 0)
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        server.DEPLOY_JOB = DeployJob()
        # The route mirrors every job to runs/gui_deploy_<stamp>.log; a test
        # run has no business leaving those among the real ones.
        self.logs = Path(tempfile.mkdtemp())
        self.patch = mock.patch.object(server, "RUNS_DIR", self.logs)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        shutil.rmtree(self.logs, ignore_errors=True)

    def call(self, path, body=None):
        req = urllib.request.Request(
            self.base + path, method="POST" if body is not None else "GET",
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_meta_lists_what_is_on_disk(self):
        status, m = self.call("/api/pi/meta")
        self.assertEqual(status, 200)
        for key in ("exports", "testdata", "traces", "defaults"):
            self.assertIn(key, m)
        self.assertTrue(m["runner"], "deploy/run_on_pi.sh should be in the repo")

    def test_the_preview_reports_the_commands_and_its_refusals(self):
        status, r = self.call("/api/pi/preview", {"target": "host", "model": "",
                                                  "measure": True})
        self.assertEqual(status, 400)
        self.assertIn("pick an exported model", r["error"])
        status, r = self.call("/api/pi/preview", {"target": "pi", "model": "x",
                                                  "ssh": "pi@host; evil"})
        self.assertEqual(status, 400)
        self.assertIn("not a [user@]host", r["error"])

    def test_a_run_streams_its_steps_and_refuses_a_second_one(self):
        fake = [{"label": "first", "command": [sys.executable, "-c", "print('hello')"]},
                {"label": "second", "command": [sys.executable, "-c", "print('world')"]}]
        with mock.patch.object(server.pi, "build_steps", return_value=fake):
            status, s = self.call("/api/pi/run", {})
            self.assertEqual(status, 200)
            self.assertEqual([st["label"] for st in s["planned"]], ["first", "second"])
            for _ in range(200):
                status, s = self.call("/api/pi/status?since=0")
                if not s["running"]:
                    break
                time.sleep(0.05)
        self.assertEqual(s["returncode"], 0)
        self.assertIn("hello", s["lines"])
        self.assertIn("world", s["lines"])
        self.assertIn("parsed", s)
        self.assertTrue(s["log_path"].endswith(".log"))

        slow = [{"label": "slow", "command": [sys.executable, "-c",
                                              "import time; print('x', flush=True); time.sleep(30)"]}]
        with mock.patch.object(server.pi, "build_steps", return_value=slow):
            self.assertEqual(self.call("/api/pi/run", {})[0], 200)
            self.assertEqual(self.call("/api/pi/run", {})[0], 409)
        self.assertTrue(self.call("/api/pi/stop", {})[1]["stopped"])
        time.sleep(0.3)
        self.assertFalse(self.call("/api/pi/status")[1]["running"])

    def test_a_deploy_run_does_not_block_training(self):
        slow = [{"label": "slow", "command": [sys.executable, "-c",
                                              "import time; time.sleep(30)"]}]
        with mock.patch.object(server.pi, "build_steps", return_value=slow):
            self.assertEqual(self.call("/api/pi/run", {})[0], 200)
        self.assertFalse(self.call("/api/train/status")[1]["running"])
        self.call("/api/pi/stop", {})

    def test_the_trace_route_is_sandboxed(self):
        status, r = self.call("/api/pi/trace?name=../../../etc/passwd")
        self.assertEqual(status, 400)
        self.assertIn("outside", r["error"])


class PiRunnerScriptTests(unittest.TestCase):
    """deploy/run_on_pi.sh: how it connects, and what it says when it cannot.

    The GUI runs this with no terminal, so everything here has to work with
    nobody there to answer a prompt — and when it fails it has to say which
    failure it was, since from the tab they all look like "it timed out"."""

    SCRIPT = ROOT / "deploy" / "run_on_pi.sh"
    TEXT = SCRIPT.read_text()

    def test_one_dropped_packet_does_not_fail_the_run(self):
        """ConnectionAttempts is 1 unless it is said otherwise. A .local name
        is multicast, sent at the lowest WiFi rate and dropped routinely, so a
        single try turns an ordinary lost packet into a failed measurement."""
        m = re.search(r'SSH_CONNECT_ATTEMPTS="\$\{SSH_CONNECT_ATTEMPTS:-(\d+)\}"', self.TEXT)
        self.assertIsNotNone(m, "the connection is made with however many attempts ssh defaults to")
        self.assertGreater(int(m.group(1)), 1)
        self.assertIn('-o "ConnectionAttempts=$SSH_CONNECT_ATTEMPTS"', self.TEXT)

    def test_the_connect_budget_fits_a_key_exchange_too(self):
        """ConnectTimeout covers the TCP connect *and* the SSH handshake and
        key exchange — a Pi 5 waking its radio has to do all of it inside the
        budget, so 10 s was too tight and produced intermittent failures."""
        m = re.search(r'SSH_CONNECT_TIMEOUT="\$\{SSH_CONNECT_TIMEOUT:-(\d+)\}"', self.TEXT)
        self.assertIsNotNone(m, "the connect timeout is not configurable")
        self.assertGreaterEqual(int(m.group(1)), 20)
        # Worst case for a genuinely dead host should stay near the OS default
        # it replaced (~75 s), or a wrong address becomes a long wait.
        attempts = int(re.search(r'SSH_CONNECT_ATTEMPTS:-(\d+)', self.TEXT).group(1))
        self.assertLessEqual(int(m.group(1)) * attempts, 90)

    def test_a_link_that_dies_mid_measurement_is_noticed(self):
        """ConnectTimeout governs setting the connection up and nothing after
        it. A sweep runs for minutes; without keepalives a dropped link hangs
        until TCP gives up, which for an idle connection can be never."""
        self.assertIn("-o ServerAliveInterval=", self.TEXT)
        self.assertIn("-o ServerAliveCountMax=", self.TEXT)

    def test_prompts_are_refused_only_when_nobody_could_answer_them(self):
        block = re.search(r'if \[\[ ! -t 0 \]\]; then\n(.*?)\nfi', self.TEXT, re.S)
        self.assertIsNotNone(block, "BatchMode is not tied to there being no terminal")
        self.assertIn("BatchMode=yes", block.group(1))

    def test_the_connection_is_proved_before_anything_is_copied(self):
        probe = self.TEXT.index('"$PI" true')
        copy = self.TEXT.index("copying ${#FILES[@]} files")
        self.assertLess(probe, copy, "the 1.2 MB copy starts before the connection is proved")

    def test_each_way_of_failing_gets_its_own_answer(self):
        """They are four different problems with four different fixes."""
        for hint in ("ssh-copy-id", "ssh-keygen -R", "hostname -I", "SSH_CONNECT_TIMEOUT=45"):
            self.assertIn(hint, self.TEXT)

    # ---- the two halves of the script have to agree ----
    # It is one file but two programs: the local half, and a heredoc executed
    # by a different bash on the Pi with only the variables the ssh line hands
    # across. Nothing in bash checks that the two agree, and when they did not
    # the run died at "line 45: SPIN: unbound variable" — on the board, after
    # the files had been copied and the governor had been changed.
    @property
    def remote_body(self):
        return self.TEXT.split("<<'REMOTE'\n", 1)[1].split("\nREMOTE\n", 1)[0]

    def test_every_variable_the_pi_half_reads_is_sent_to_it(self):
        body = self.remote_body
        sent = set(re.search(r"REMOTE_VARS=\((.*?)\)", self.TEXT, re.S).group(1).split())
        read = {v for v in re.findall(r'\$\{?([A-Za-z_]\w*)', body)}
        assigned = (set(re.findall(r'(?:^|[;&|(]|\bdo\b|\bthen\b)\s*(?:export\s+)?'
                                   r'([A-Za-z_]\w*)=', body, re.M))
                    | set(re.findall(r'^\s*([A-Za-z_]\w*)\(\)', body, re.M))
                    | set(re.findall(r'for\s+([A-Za-z_]\w*)\s+in', body))
                    | {"HOME", "PATH", "BASH_VERSION", "IFS"})
        missing = read - assigned - sent
        self.assertEqual(missing, set(),
                         f"the Pi half reads {sorted(missing)} but the ssh line never sends it; "
                         "under `set -u` that kills the run partway through")

    def test_the_preamble_is_built_from_that_one_list(self):
        """Spelled out a second time in the ssh command, the two lists drift —
        which is exactly what happened."""
        ssh_line = next(l for l in self.TEXT.splitlines() if l.startswith("ssh ") and "bash -s" in l)
        self.assertIn("$REMOTE_ENV", ssh_line)
        self.assertNotRegex(ssh_line, r"\w+='\$\w+'")

    def test_the_thermal_sampler_cannot_outlive_the_run(self):
        """It is a `while :` loop in the background with nothing to stop it.
        A measurement that fails under `set -e` would leave it reading the
        thermal zone every two seconds on the board, and the temp file with
        it — once per failed run, forever."""
        self.assertRegex(self.remote_body, r"trap '[^']*\$SAMPLER[^']*' EXIT")
        self.assertRegex(self.remote_body, r"trap '[^']*PEAK_FILE[^']*' EXIT")

    def test_the_state_check_is_only_given_flags_it_accepts(self):
        """LOCK_ARGS goes to preflight.py and RUN_ARGS to raspi_infer.py.
        While they were one list, a state check with spinning disabled was
        handed --no-spin, which preflight.py does not take."""
        check_call = next(l for l in self.remote_body.splitlines() if "preflight.py" in l)
        self.assertIn("LOCK_ARGS", check_call)
        self.assertNotIn("RUN_ARGS", check_call)
        self.assertNotIn("--no-spin", self.TEXT.split("LOCK_ARGS=(")[1].split("RUN_ARGS")[0])
        # The real contract: every flag that list can carry, against the real parser.
        for flags in ([], ["--cpus", "0-1"], ["--require-locked"],
                      ["--cpus", "0-1", "--require-locked"]):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    rc = preflight.main([*flags, "--threads", "2", "--json"])
                except SystemExit as e:                 # argparse exits 2 on a bad flag
                    self.fail(f"preflight.py refuses {flags}: exit {e.code}")
            # 0 locked, 1 not locked, 3 could not pin (no affinity API here).
            # 2 would mean the flag itself was not understood.
            self.assertNotEqual(rc, 2, f"preflight.py does not understand {flags}")
            self.assertIn(rc, (0, 1, 3), f"preflight.py refuses {flags}")

    def test_the_pi_half_runs_through_for_every_combination(self):
        """Run the remote half here, with a stub standing in for the Pi's
        python, and check it reaches the end and composes the right calls.
        The unbound-variable failure was invisible to every other test because
        that code only ever ran on the board."""
        stub = ('\nmkdir -p "$HOME/$REMOTE_DIR/env/bin"\n'
                'printf \'#!/bin/sh\\necho "[py] $*"\\n\' > "$HOME/$REMOTE_DIR/env/bin/python"\n'
                'chmod +x "$HOME/$REMOTE_DIR/env/bin/python"\n'
                ': > "$HOME/$REMOTE_DIR/requirements-raspi.txt"\n')
        values = dict(REMOTE_DIR="eeg", MODEL="m", GRAPH="m", INCREMENTAL="",
                      TESTDATA="t.npz", HOP="32", THREADS="2",
                      LATENCY_EVERY="400", LATENCY_STEM="lat", REALTIME_S="0",
                      DUTY_HOPS="", DUTY_S="20", CPUS="0-1", GOVERNOR="",
                      REQUIRE_LOCKED="", CHECK_ONLY="", SPIN="1")
        # Composed from the script's *own* list, not from a copy of it, so a
        # variable the body reads but REMOTE_VARS omits reaches the body unset
        # here exactly as it reached the Pi.
        names = re.search(r"REMOTE_VARS=\((.*?)\)", self.TEXT, re.S).group(1).split()
        self.assertEqual(set(names) - set(values), set(),
                         "a new remote variable — give it a value in this test")
        home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, home, ignore_errors=True)

        def run(_stub=None, **over):
            env = {**values, **over}
            pre = " ".join(f"{k}={shlex.quote(env[k])}" for k in names)
            r = subprocess.run(["bash", "-c", f"{pre} bash -s"],
                               input=(_stub or stub) + self.remote_body,
                               text=True, capture_output=True,
                               env={**os.environ, "HOME": str(home)}, timeout=60)
            self.assertEqual(r.returncode, 0, f"{over or 'defaults'}: {r.stderr.strip()[:200]}")
            return [l for l in r.stdout.splitlines() if l.startswith("[py] ")]

        measure = " ".join(run())
        self.assertIn("raspi_infer.py m.onnx --threads 2 --hop 32 --cpus 0-1 --benchmark", measure)
        self.assertNotIn("--no-spin", measure)

        self.assertIn("--no-spin", " ".join(run(SPIN="0")))

        check = " ".join(run(CHECK_ONLY="1", SPIN="0"))
        self.assertIn("preflight.py --threads 2 --cpus 0-1", check)
        self.assertNotIn("--no-spin", check)            # preflight.py does not take it
        self.assertNotIn("raspi_infer.py", check)       # nothing is measured

        sweep = run(DUTY_HOPS="64,8", REALTIME_S="5", SPIN="0")
        self.assertTrue(any("--duty_cycle 64,8" in c for c in sweep))
        self.assertTrue(any("--realtime" in c for c in sweep))

        # The step graph: a different file and a flag, same script. STEP_ARGS
        # is an array and cannot ride through the scalar env, so it is rebuilt
        # on the far side -- which only this harness ever exercises.
        step = " ".join(run(INCREMENTAL="1", GRAPH="m_step"))
        self.assertIn("raspi_infer.py m_step.onnx", step)
        self.assertIn("--incremental", step)
        self.assertNotIn("raspi_infer.py m.onnx", step)
        # ...and the windowed default must not pick the flag up.
        self.assertNotIn("--incremental", measure)

        # An emptied pin field means "do not pin" — and then the command must
        # not claim otherwise.
        self.assertNotIn("--cpus", " ".join(run(CPUS="")))

        # The default: resolved on the board, where the core count is known,
        # to the top THREADS cores — off CPU 0 and its interrupts.
        # The real nproc honours OMP_NUM_THREADS unless --all is passed, and the
        # script exports it: a stub that ignores the flag cannot catch the bug
        # that put the measurement on CPU 0-1 of a four-core board.
        stub_nproc = stub + ('nproc() { if [[ "$*" == *--all* || -z "${OMP_NUM_THREADS:-}" ]]; '
                             'then echo 4; else echo "$OMP_NUM_THREADS"; fi; }\n'
                             'export -f nproc\n')
        auto = run(CPUS="auto", _stub=stub_nproc)
        self.assertIn("--cpus 2-3", " ".join(auto))
        self.assertIn("--cpus 1-3", " ".join(run(CPUS="auto", THREADS="3", _stub=stub_nproc)))
        self.assertIn("--cpus 3", " ".join(run(CPUS="auto", THREADS="1", _stub=stub_nproc)))

    def test_everyone_agrees_on_how_many_files_go_over(self):
        """The count is written out in the script's header, in the README and
        in the GUI's hint, and it has drifted twice already — once when
        preflight.py joined the list."""
        files = re.search(r"^FILES=\((.*?)^\)", self.TEXT, re.S | re.M).group(1)
        n = len([l for l in files.splitlines() if l.strip().startswith('"')])
        self.assertEqual(n, 6)
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8}
        # Only counts that are talking about the copy: "two files" elsewhere in
        # the README is about something else entirely.
        said = re.compile(r"(?:cop(?:y|ies|ied)|checks)\s+(?:the\s+)?(\w+)\s+files"
                          r"|(\w+)\s+files\s+are\s+copied")
        found = 0
        for name, text in (("run_on_pi.sh", self.TEXT),
                           ("README.md", (ROOT / "README.md").read_text()),
                           ("index.html", (ROOT / "gui" / "static" / "index.html").read_text())):
            for m in said.finditer(text):
                word = (m.group(1) or m.group(2)).lower()
                if word not in words:
                    continue
                found += 1
                self.assertEqual(words[word], n,
                                 f"{name} says '{m.group(0)}' but FILES has {n}")
        self.assertGreaterEqual(found, 3, "the count is no longer written down anywhere")

    def test_a_host_that_does_not_resolve_is_named_as_such(self):
        """End to end, with no network needed: the name cannot resolve, so the
        script must diagnose it and stop before copying anything."""
        env = {**os.environ, "SSH_CONNECT_TIMEOUT": "1", "SSH_CONNECT_ATTEMPTS": "1"}
        try:
            out = subprocess.run(
                ["bash", str(self.SCRIPT), "pi@no-such-pi-for-tests.invalid", "tcn_mla_eog"],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                text=True, timeout=60)
        except subprocess.TimeoutExpired:
            self.fail("the script hung instead of reporting that it could not connect")
        if "missing:" in out.stderr:
            self.skipTest("no export on this machine to run the script against")
        self.assertNotEqual(out.returncode, 0)
        self.assertNotIn("copying", out.stdout)
        self.assertRegex(out.stderr, r"did not resolve|no usable answer")


class PageTests(unittest.TestCase):
    PAGE = (ROOT / "gui" / "static" / "index.html").read_text()

    def test_the_tab_button_and_the_panel_agree(self):
        self.assertIn('<button data-tab="deploy"', self.PAGE)
        self.assertIn('<div class="tab" id="tab-deploy">', self.PAGE)

    def test_the_train_tab_can_show_several_runs_at_once(self):
        """The switcher only means anything if each run keeps its own buffer:
        one shared log would interleave two runs' output into nonsense."""
        self.assertIn('id="jobTabs"', self.PAGE)
        self.assertIn("function selectJob(", self.PAGE)
        self.assertIn("function jobState(", self.PAGE)
        # Per-run read offset, log and curve.
        block = self.PAGE[self.PAGE.index("function jobState("):]
        block = block[:block.index("function renderJobTabs(")]
        for field in ("next:", "metrics:", "log:"):
            self.assertIn(field, block)
        # The poll has to name the run it is asking about, or every run's
        # lines land in whichever log is on screen.
        poll = self.PAGE[self.PAGE.index("async function poll()"):][:600]
        self.assertIn("&id=", poll)
        self.assertIn("state.job.next", poll)

    def test_the_switcher_hides_itself_when_there_is_one_run(self):
        """One run is the ordinary case and needs no chooser."""
        block = self.PAGE[self.PAGE.index("function renderJobTabs("):][:1400]
        self.assertIn("list.length > 1", block)

    def test_the_switcher_is_a_queue_not_a_history(self):
        """Finished runs leave it — their logs and curves are still in the
        Runs tab — except the one being read, which must not vanish mid-read."""
        block = self.PAGE[self.PAGE.index("function renderJobTabs("):][:1400]
        self.assertIn("j.running || j.id === state.jobId", block)

    def test_stopping_means_the_run_you_are_looking_at(self):
        block = self.PAGE[self.PAGE.index("async function stop()"):][:400]
        self.assertIn("state.jobId", block)

    def test_the_duty_table_header_and_rows_have_the_same_columns(self):
        """The sweep table is built by string concatenation in two places, and
        a column added to one of them reads as a quietly ragged table."""
        block = self.PAGE[self.PAGE.index("function renderPiDuty("):]
        block = block[:block.index("function renderPiTables(")]
        head = re.search(r"const head = `(.*?)`;", block, re.S)
        self.assertIsNotNone(head, "the duty table's header was not found")
        columns = len(re.findall(r"<th", head.group(1)))
        body = re.search(r"table\.innerHTML = head \+ rows\.map\(r => (.*?)\)\.join\(\"\"\);",
                         block, re.S)
        self.assertIsNotNone(body, "the duty table's rows were not found")
        self.assertEqual(len(re.findall(r"<td", body.group(1))), columns)

    def test_both_sources_are_wired_and_the_file_wins(self):
        """A trace file has every hop in it at full precision; the parsed log
        is the live preview that fills the tab until one exists."""
        for fn in ("function piLatencySource()", "function piDutySource()"):
            block = self.PAGE[self.PAGE.index(fn):][:600]
            self.assertIn("state.pi.loaded", block)
            self.assertIn("state.pi.parsed", block)
            self.assertLess(block.index("state.pi.loaded"), block.index("state.pi.parsed"),
                            f"{fn} prefers the log over the trace file")

    def test_the_model_is_chosen_from_the_runs_table(self):
        """A dropdown of deploy/artifacts/ offered two of twenty-nine models,
        and showed none of the numbers you would choose between them on. The
        field is the same picker the Evaluate tab uses, for the same reason."""
        self.assertIn('id="piModelBtn"', self.PAGE)
        self.assertNotRegex(self.PAGE, r'<select id="piModel"')
        self.assertIn('openRunPicker("deploy")', self.PAGE)
        self.assertIn('"Choose a model to deploy"', self.PAGE)

    def test_the_picker_says_which_runs_can_already_be_run(self):
        block = self.PAGE[self.PAGE.index("const deployBadge"):][:1200]
        for state in ("onnx", "export first", "no export"):
            self.assertIn(state, block)
        # A bounded-lookahead rung is refused like a teacher but for its own
        # reason, and the badge has to give that reason rather than the
        # teacher's: the horizon is finite, and what it costs is a delay.
        self.assertIn("r.lookahead", block)
        self.assertIn("bounded lookahead", block)
        self.assertIn('state.pickFor === "deploy"', block,
                      "the deploy badge would show in the Evaluate picker too")

    def test_the_export_box_follows_the_model_rather_than_carrying_over(self):
        """Picked after an unexported run, an exported one would otherwise be
        re-exported without being asked."""
        block = self.PAGE[self.PAGE.index("function setPiModel("):][:900]
        self.assertIn('$("#piExport").checked = !m.exported || m.stale', block)

    def test_the_run_picker_fills_whichever_field_asked_for_it(self):
        self.assertIn("const PICK_FOR = {", self.PAGE)
        self.assertIn('openRunPicker("deploy")', self.PAGE)
        self.assertIn('openRunPicker("eval")', self.PAGE)

    def test_no_measurement_is_shown_without_the_conditions_it_was_taken_under(self):
        """The reviewer's point: a latency figure is a statement about a model
        *and* a machine. Both chart titles carry the verdict of their own
        source, so neither can be read on its own."""
        for fn in ("function renderPiLatency(", "function renderPiDuty("):
            block = self.PAGE[self.PAGE.index(fn):][:1400]
            self.assertIn("lockBadge(", block, f"{fn} draws a title with no verdict on it")
        self.assertIn('$("#piEnv")', self.PAGE)
        self.assertIn("For the report:", self.PAGE)

    def test_an_unrecorded_state_is_not_shown_as_a_passing_one(self):
        """Traces from before the check existed have unknown conditions, which
        is a different thing from a machine that was measured and found
        unlocked."""
        block = self.PAGE[self.PAGE.index("function lockBadge("):][:800]
        self.assertIn("conditions not recorded", block)
        self.assertRegex(block, r"if \(!v\)")

    def test_a_control_the_platform_cannot_hold_is_not_offered(self):
        # The whole function, not a fixed slice of it: a byte count silently
        # stops covering what it was written to cover as soon as anything is
        # added above the line it was reaching for.
        start = self.PAGE.index("function syncPiTarget()")
        block = self.PAGE[start:self.PAGE.index("\n  function ", start + 1)]
        self.assertIn("supports", block)
        self.assertIn("cpus.disabled", block)
        # ... and a disabled field must not still send its value.
        self.assertIn('$("#piCpus").disabled ? "" :', self.PAGE)


class DutySweepAndSteppingTests(unittest.TestCase):
    """
    The duty-cycle sweep and incremental inference cannot be combined.

    The sweep asks which hop size keeps up. A step graph has no hop -- it
    emits on every sample -- so there is nothing to sweep, and its duty cycle
    is the one number the latency run already reports.

    Combining them did not fail: ``duty_cycle_check`` wraps the denoiser in
    the windowed ``StreamingDenoiser``, which hands it a whole window per
    hop, so the step graph re-stepped 125 samples for every emitted hop --
    recomputing precisely what carrying state exists to avoid -- and printed
    the result as a clean OK/FAIL table. Wrong numbers that look right are
    the reason this is refused in three places rather than one.
    """

    PAGE = (ROOT / "gui" / "static" / "index.html").read_text()
    SPEC = {"target": "host", "model": "tcn_se_both", "hop": 32, "threads": 2,
            "testdata": "BOTH_test100.npz", "duty_hops": "64,32,16,8,4",
            "allow_split_mismatch": True}

    def test_the_api_refuses_the_pair(self):
        with self.assertRaisesRegex(ValueError, "no hop"):
            pi.build_steps({**self.SPEC, "incremental": True, "duty": True})

    def test_each_on_its_own_is_fine(self):
        for flags in ({"incremental": True}, {"duty": True}):
            with self.subTest(**flags):
                self.assertTrue(pi.build_steps({**self.SPEC, **flags}))

    def test_the_runtime_refuses_it_too(self):
        """The board is where the numbers are produced, so it checks as well."""
        source = (ROOT / "deploy" / "raspi_infer.py").read_text()
        i = source.index("if a.duty_cycle and a.incremental:")
        guard = source[i:i + 1200]
        self.assertIn("return 2", guard, "the refusal has to be a non-zero exit")
        self.assertIn("--duty_cycle", guard)

    def test_no_incremental_row_claims_a_duty_sweep(self):
        """
        A step graph has no hop to sweep, so a duty trace under its stem is
        never part of that measurement.

        It got there because the copy-back from the Pi is a glob --
        ``<stem>*.json`` -- so a ``_duty`` file left on the board by an
        earlier run came home with a fresh timestamp and was read as part of
        the new one. A run with the sweep switched off still produced a row
        reading "duty sweep · safe from hop 16".
        """
        for r in pi.list_deploy_runs():
            if r["mode"] == "incremental":
                with self.subTest(model=r["model"], where=r["where"]):
                    self.assertIsNone(r["duty"],
                                      "an incremental row absorbed a duty-sweep trace")

    def test_the_board_is_cleared_before_it_is_measured(self):
        """
        The cause, not the symptom: clearing the stem on the Pi first is what
        stops the glob picking up a previous run's output.
        """
        script = (ROOT / "deploy" / "run_on_pi.sh").read_text()
        clear = script.index('rm -f "$LATENCY_STEM"')
        measure = script.index('--latency_out "$LATENCY_STEM"')
        self.assertLess(clear, measure, "the clear has to happen before the measurement")
        self.assertIn('"$LATENCY_STEM"_*.json', script[clear:clear + 200],
                      "the siblings (_duty, _realtime) have to go too")

    def test_the_page_will_not_let_both_be_ticked(self):
        block = self.PAGE[self.PAGE.index("const dutyBox = $(\"#piDutyOn\")"):][:1400]
        self.assertIn("dutyBox.disabled = inc.checked", block)
        self.assertIn("inc.disabled = true", block,
                      "ticking the sweep must disable stepping as well")


class ControlGridLayoutTests(unittest.TestCase):
    """
    A control that cannot shrink wrecks the row it is in.

    Both control grids are ``auto-fit, minmax(190px, 1fr)``. A grid item's
    default ``min-width`` is ``auto``, which refuses to shrink below the
    content's minimum size -- so one field whose text is ``nowrap`` with
    nothing allowed to clip it sizes its column to that text, overflows its
    cell and lands on top of the field beside it. That is what the test-set
    button did: 237px of content in a 218px cell, printing its hint through
    the model's.

    Checked as CSS rather than by rendering, because there is no browser
    here; the geometry itself was verified in the pane.
    """

    PAGE = (ROOT / "gui" / "static" / "index.html").read_text()

    def _rule(self, selector):
        i = self.PAGE.index(selector)
        return self.PAGE[i:self.PAGE.index("}", i)]

    def test_a_nowrap_button_label_is_allowed_to_clip(self):
        """
        Every picker button's metadata span is nowrap, so each one needs the
        overflow handling that lets the button shrink.
        """
        for selector in ("#piModelBtn .meta", "#piTestdataBtn .meta"):
            with self.subTest(selector=selector):
                rule = self._rule(selector)
                self.assertIn("white-space: nowrap", rule)
                self.assertIn("overflow: hidden", rule,
                              f"{selector} is nowrap with no clipping: it cannot shrink")
                self.assertIn("text-overflow: ellipsis", rule)
                self.assertIn("min-width: 0", rule)

    def test_grid_items_may_shrink_below_their_content(self):
        for tab in ("#tab-deploy", "#tab-eval"):
            with self.subTest(tab=tab):
                self.assertIn("min-width: 0", self._rule(f"{tab} .efield {{"),
                              f"{tab}'s fields keep min-width:auto and cannot shrink")
                self.assertIn("max-width: 100%", self._rule(f"{tab} .efield > *"),
                              f"a child of a {tab} field may still exceed its cell")


class RunsPayloadFreshnessTests(unittest.TestCase):
    """
    Everything /api/runs returns has to reach the page.

    ``refreshRuns`` used to copy a hand-written list of keys, so a grouping
    added to the endpoint stayed invisible until a reload -- which reads as
    missing data, not stale data. It cost the loss and optimizer views once,
    and the Deploy-runs view again when those rows were added to the payload
    and not to the list. The list is now iterated from the payload, so the
    two cannot drift apart.
    """

    PAGE = (ROOT / "gui" / "static" / "index.html").read_text()

    def _fn(self, name):
        i = self.PAGE.index(f"async function {name}")
        return self.PAGE[i:self.PAGE.index("\n  }", i)]

    def test_the_page_copies_whatever_the_endpoint_returns(self):
        body = self._fn("refreshRuns")
        self.assertIn("Object.entries(payload)", body,
                      "refreshRuns copies a hand-written key list; a new grouping will be "
                      "invisible until the page is reloaded")

    def test_a_finished_deploy_updates_the_runs_view(self):
        """
        A measurement is a row in the Runs tab as well as a trace in the
        Deploy tab, and only the Deploy tab refreshes when one finishes.
        """
        body = self._fn("refreshPiMeta")
        self.assertIn("/api/runs/deploys", body)
        self.assertIn("state.meta.deploys", body)

    def test_the_endpoint_returns_the_deploy_rows(self):
        """The server half of the same contract."""
        import gui.server as server
        payload_keys = {"runs", "kinds", "times", "losses", "optimizers",
                        "deploys", "deploy_logs"}
        source = (ROOT / "gui" / "server.py").read_text()
        handler = source[source.index('if url.path == "/api/runs":'):][:700]
        for key in payload_keys - {"runs"}:
            with self.subTest(key=key):
                self.assertIn(f'"{key}"', handler)


class DeployModelListFreshnessTests(unittest.TestCase):
    """
    The Deploy tab keeps its own model list, and it has to learn about runs
    that finish while the page is open.

    It did not: ``refreshRuns`` updated every Runs-tab grouping and left
    ``state.meta.pi`` alone, so a newly trained run was selectable from the
    Runs table but unknown to the Deploy tab. Everything keyed on finding it
    then failed *quietly* -- no metadata line, no split check, and a test-set
    generator that said "pick a model first" about a model that was picked.
    """

    PAGE = (ROOT / "gui" / "static" / "index.html").read_text()

    def test_refreshing_the_runs_also_refreshes_the_deploy_list(self):
        block = self.PAGE[self.PAGE.index("async function refreshRuns"):]
        block = block[:block.index("\n  async function", 1)
                      if "\n  async function" in block[1:] else 4000]
        self.assertIn("refreshPiMeta()", block,
                      "a run that finished while the page was open never reaches the Deploy tab")

    def test_an_unknown_selection_is_reported_not_swallowed(self):
        """
        A model the tab does not know and no model at all are different
        situations; reporting both as "pick one" is what made a stale list
        look like a broken tab.
        """
        hint = self.PAGE[self.PAGE.index("function renderPiModelHint"):][:1600]
        self.assertIn("not in this tab's list", hint)

        split = self.PAGE[self.PAGE.index("function piSplitProblem"):][:1400]
        self.assertIn("not in this tab's list", split,
                      "an unknown model must not read as a clean split check")

    def test_every_trained_run_the_tab_lists_can_be_paired(self):
        """
        What the bug report was really about: a new variant has to arrive with
        everything the tab needs -- an artifact and a seed -- or the generator
        cannot cut it a test set.
        """
        for m in pi.list_models():
            if not (m["run"] and m["deployable"]):
                continue
            with self.subTest(model=m["name"]):
                self.assertIsNotNone(m["variant"], f"{m['name']} has no variant")
                if m["master_seed"] is None:
                    continue          # pre-dates the seed being recorded; refused loudly
                self.assertIn(m["artifact"], ("EOG", "EMG", "BOTH"),
                              f"{m['name']} records no artifact, so no test set can be cut")


class TestdataGeneratorTests(unittest.TestCase):
    """
    Cutting the matching set from the model.

    The pairing rules turn "which set do I pick" into a small research question
    whose answer the run already records, so the generator reads the artifact
    and seed off the model instead of asking.  What it must never do is guess:
    a set cut at a default seed for a run whose seed is unknown would look
    matched and would not be, which is the exact failure the check exists for.
    """

    def test_it_reuses_an_existing_matching_set(self):
        """
        An existing file is already the right epochs. Rewriting it would churn
        a file that recorded measurements may refer to.
        """
        model = next((m for m in pi.list_models()
                      if m.get("master_seed") == 67 and m.get("artifact") == "BOTH"), None)
        if model is None:
            self.skipTest("no seed-67 BOTH model in the working tree")
        before = (pi.TESTDATA_DIR / "BOTH_test100_s67.npz").stat().st_mtime
        out = pi.cut_testdata_for(model["name"], n=100)
        self.assertEqual(out["name"], "BOTH_test100_s67.npz")
        self.assertFalse(out["created"])
        self.assertEqual((pi.TESTDATA_DIR / "BOTH_test100_s67.npz").stat().st_mtime, before,
                         "an existing matching set was rewritten instead of reused")

    def test_it_refuses_a_run_with_no_recorded_seed(self):
        model = next((m for m in pi.list_models() if m.get("master_seed") is None), None)
        if model is None:
            self.skipTest("every run records its seed in this tree")
        with self.assertRaisesRegex(ValueError, "does not record the seed"):
            pi.cut_testdata_for(model["name"])

    def test_it_refuses_an_unknown_model_and_a_silly_size(self):
        with self.assertRaisesRegex(ValueError, "no model called"):
            pi.cut_testdata_for("no_such_model_anywhere")
        model = next((m for m in pi.list_models() if m.get("master_seed") is not None), None)
        if model is None:
            self.skipTest("no model with a recorded seed")
        for n in (0, 9, 10_000):
            with self.subTest(n=n), self.assertRaisesRegex(ValueError, "between 10 and 2000"):
                pi.cut_testdata_for(model["name"], n=n)

    def test_a_cut_set_matches_the_model_it_was_cut_for(self):
        """The point of the whole feature: what it writes must pass the check."""
        data_dir = Path("data/EEGdenoisenet/data")
        if not (data_dir / "EEG_all_epochs.npy").is_file():
            self.skipTest("EEGdenoiseNet .npy files are not present")
        model = next((m for m in pi.list_models()
                      if m.get("master_seed") is not None and m.get("artifact") == "EMG"), None)
        if model is None:
            self.skipTest("no EMG model with a recorded seed")

        out = pi.cut_testdata_for(model["name"], n=20)
        created = pi.TESTDATA_DIR / out["name"]
        try:
            self.assertIsNone(pi.split_mismatch(model["name"], out["name"]),
                              "the generator produced a set its own model would be refused on")
            # And the API accepts the pair it just made.
            self.assertEqual(
                pi._testdata_name({"model": model["name"], "testdata": out["name"],
                                   "_enforce_split": True}),
                out["name"])
        finally:
            if out.get("created"):
                created.unlink(missing_ok=True)


class DeployRunListingTests(unittest.TestCase):
    """
    The Runs tab's other half: the measurements, not the training runs.

    One row per ``(where, model, hop)``, because the three traces a single run
    can write -- per-hop latency, the duty sweep, the real-time replay -- are
    one run's output and should read as one row.
    """

    def test_a_runs_three_traces_become_one_row(self):
        rows = pi.list_deploy_runs()
        if not rows:
            self.skipTest("no traces in deploy/artifacts/pi_latency/")
        keys = [(r["where"], r["model"], r["hop"]) for r in rows]
        self.assertEqual(len(keys), len(set(keys)), "a measurement was split across rows")
        multi = [r for r in rows if len(r["traces"]) > 1]
        if multi:
            self.assertTrue(any(r["duty"] or r["realtime"] for r in multi),
                            "extra traces were gathered but neither duty nor realtime was set")

    def test_the_primary_trace_is_the_latency_one(self):
        """
        'Open' should land on the measurement, not on whichever sibling the
        directory listing happened to return first.
        """
        for r in pi.list_deploy_runs():
            with self.subTest(model=r["model"], where=r["where"]):
                self.assertIn(r["primary"], r["traces"])
                if any(not t.endswith(("_duty.json", "_realtime.json")) for t in r["traces"]):
                    self.assertFalse(r["primary"].endswith(("_duty.json", "_realtime.json")))

    def test_the_model_name_survives_underscores_and_digits(self):
        """
        ``tcn_mla_both_s67`` is a run name with both. Anchoring the parse on
        the leading underscore would cut it in the wrong place.
        """
        rows = {r["model"] for r in pi.list_deploy_runs()}
        if "tcn_mla_both_s67" not in {r for r in rows}:
            self.skipTest("no trace for a model whose name carries a seed suffix")
        self.assertIn("tcn_mla_both_s67", rows)

    def test_rows_carry_the_deadline_and_the_environment(self):
        for r in pi.list_deploy_runs():
            with self.subTest(model=r["model"]):
                # hop / 256 Hz, in ms -- what the median has to beat.
                self.assertAlmostEqual(r["budget_ms"], r["hop"] / 256.0 * 1000, places=2)
                if r["median_ms"] is not None:
                    self.assertEqual(r["realtime_ok"], r["median_ms"] < r["budget_ms"])
                    self.assertAlmostEqual(r["duty_cycle"], r["median_ms"] / r["budget_ms"],
                                           places=3)
                # A trace from before the environment was recorded must come
                # back as None, which the page shows as "unknown" -- not as a
                # pass, and not as "unlocked".
                self.assertIn(r["verdict"], (None, "locked", "partial", "unlocked"))

    def test_host_runs_are_labelled_not_hidden(self):
        rows = pi.list_deploy_runs()
        if not any(r["where"] == "host" for r in rows):
            self.skipTest("no host traces in this tree")
        self.assertTrue(all(r["where"] in ("host", "pi") for r in rows))

    def test_newest_first(self):
        times = [r["mtime"] for r in pi.list_deploy_runs()]
        self.assertEqual(times, sorted(times, reverse=True))

    def test_deploy_logs_are_listed_newest_first(self):
        logs = pi.list_deploy_logs()
        self.assertEqual([x["mtime"] for x in logs],
                         sorted((x["mtime"] for x in logs), reverse=True))
        self.assertTrue(all(x["name"].startswith("gui_deploy_") for x in logs))


class TestScoreRecordingTests(unittest.TestCase):
    """
    A measurement's accuracy belongs in the same file as its timing.

    ``evaluate_testdata`` computed RRMSE_t / RRMSE_f / CC, printed them, and
    returned them to a caller that threw the return away -- so the trace saved
    the milliseconds and lost the thing the milliseconds were spent on. The
    monitor now carries them into its summary, which is what ``save()`` writes.
    """

    def test_the_monitor_writes_scores_into_its_trace(self):
        from deploy import raspi_infer          # numpy + onnxruntime
        m = raspi_infer.LatencyMonitor(hop=32, sample_rate=256, label="latency/test")
        self.assertNotIn("scores", m.summary(),
                         "a run that scored nothing should not claim an empty result")
        m.scores = {"testdata": "EOG_test100.npz", "n_epochs": 100,
                    "modes": {"epoch mode": {"rrmse_t": 0.46, "rrmse_s": 0.464, "acc": 0.883}}}
        self.assertEqual(m.summary()["scores"]["modes"]["epoch mode"]["acc"], 0.883)

        with tempfile.TemporaryDirectory() as td:
            js, _ = m.save(Path(td) / "t")
            written = json.loads(js.read_text())
        self.assertEqual(written["scores"]["testdata"], "EOG_test100.npz")

    def test_scores_are_recovered_from_a_job_log_for_older_traces(self):
        """
        The measurements already on disk predate scores being saved, and their
        numbers are not lost: the job log that produced them also names the
        traces it wrote, which is the join.
        """
        recovered = pi._scores_from_logs()
        if not recovered:
            self.skipTest("no deploy job logs with a scored test set")
        for name, sc in recovered.items():
            with self.subTest(trace=name):
                self.assertEqual(sc["source"], "log")
                self.assertTrue(sc["from_log"].startswith("gui_deploy_"))
                # Matched by prefix: a windowed run logs "epoch mode", a step
                # graph "epoch mode (stepped)". Both are the per-epoch score.
                epoch = next((k for k in sc["modes"] if k.startswith("epoch mode")), None)
                self.assertIsNotNone(epoch, f"{name} recorded no epoch-mode row")
                for key in ("rrmse_t", "rrmse_s", "acc"):
                    self.assertIsInstance(sc["modes"][epoch][key], float)

    def test_a_row_says_where_its_numbers_came_from(self):
        for r in pi.list_deploy_runs():
            sc = r.get("scores")
            if sc is None:
                continue
            with self.subTest(model=r["model"], where=r["where"]):
                self.assertIn(sc["source"], ("trace", "log"))
                if sc["source"] == "log":
                    self.assertIn("from_log", sc)

    def test_a_trace_recorded_score_beats_one_scraped_from_a_log(self):
        """Data saved by the run wins over text parsed out of its log."""
        import copy
        traces = pi.list_traces()
        if not traces:
            self.skipTest("no traces")
        marker = {"testdata": "FROM_TRACE.npz",
                  "modes": {"epoch mode": {"rrmse_t": 0.1, "rrmse_s": 0.1, "acc": 0.99}}}
        patched = copy.deepcopy(traces)
        for t in patched:
            if not t["name"].endswith(("_duty.json", "_realtime.json")):
                t["scores"] = marker
        with mock.patch.object(pi, "list_traces", return_value=patched):
            rows = [r for r in pi.list_deploy_runs() if r.get("scores")]
        self.assertTrue(rows, "no row picked up the trace-recorded score")
        for r in rows:
            self.assertEqual(r["scores"]["source"], "trace")
            self.assertEqual(r["scores"]["testdata"], "FROM_TRACE.npz")


class InferenceModeTests(unittest.TestCase):
    """
    A trace has to say how its milliseconds were produced, not only what they
    were.

    The environment block exists so a latency figure cannot be separated from
    the machine; the same argument applies to the algorithm. Without it a
    reader has to know which files ``run_on_pi.sh`` copies to work out that a
    trace is the hop-and-recompute path and not a cached one — and once a
    cached export exists, old and new traces would be indistinguishable.
    """

    def test_a_windowed_trace_says_it_recomputes(self):
        from deploy import raspi_infer
        m = raspi_infer.LatencyMonitor(hop=32, sample_rate=256, model_window=512)
        inf = m.summary()["inference"]
        self.assertEqual(inf["mode"], "windowed")
        self.assertIs(inf["kv_cache"], False)
        self.assertEqual(inf["recompute_factor"], 16.0)   # 512 / 32
        self.assertEqual(inf["runtime"], "onnxruntime")

    def test_the_recompute_factor_follows_the_hop(self):
        from deploy import raspi_infer
        for hop, expected in ((8, 64.0), (64, 8.0), (512, 1.0)):
            with self.subTest(hop=hop):
                m = raspi_infer.LatencyMonitor(hop=hop, sample_rate=256, model_window=512)
                self.assertEqual(m.summary()["inference"]["recompute_factor"], expected)

    def test_rows_default_to_windowed_for_traces_that_predate_the_field(self):
        """
        Every trace written before this said nothing, and all of them were
        windowed — but the row must not claim the trace said so.
        """
        for r in pi.list_deploy_runs():
            with self.subTest(model=r["model"], where=r["where"]):
                self.assertIn(r["mode"], ("windowed", "incremental"))
                if r["inference"] is None:
                    self.assertEqual(r["mode"], "windowed")

    def test_an_incremental_trace_is_listed_as_its_own_measurement(self):
        rows = [r for r in pi.list_deploy_runs() if r["mode"] == "incremental"]
        if not rows:
            self.skipTest("no incremental traces in deploy/artifacts/pi_latency/")
        for r in rows:
            with self.subTest(model=r["model"]):
                # Hop 1 is the literal truth and what makes the duty cycle
                # comparable with a windowed run at any hop.
                self.assertEqual(r["hop"], 1)
                self.assertAlmostEqual(r["budget_ms"], 1000 / 256.0, places=2)
                # Both machines now: the host runs it in torch, the Pi runs
                # the exported step graph. Which one is read from the trace's
                # environment block, not from its filename.
                self.assertIn(r["where"], ("host", "pi"))
                self.assertEqual(r["inference"]["recompute_factor"], 1.0)
                self.assertIn("kv_cache", r["inference"])

    def test_only_the_attention_variants_hold_a_kv_cache(self):
        for r in pi.list_deploy_runs():
            inf = r.get("inference") or {}
            if inf.get("mode") != "incremental":
                continue
            with self.subTest(model=r["model"]):
                has_attention = bool(inf.get("kv_floats_per_step"))
                self.assertEqual(inf["kv_cache"], has_attention)


class IncrementalStepTests(unittest.TestCase):
    """The Deploy tab's incremental step: host only, and from the checkpoint."""

    SPEC = {"target": "host", "model": "tcn_mla_eog", "hop": 32, "threads": 2,
            "testdata": "EOG_test100.npz"}

    def test_it_runs_the_checkpoint_not_the_export(self):
        steps = pi.build_steps({**self.SPEC, "incremental": True})
        self.assertEqual(len(steps), 1)
        cmd = steps[0]["command"]
        self.assertIn("deploy.bench_incremental", cmd)
        self.assertIn("--ckpt", cmd)
        self.assertIn("runs/tcn_mla_eog/best.pt", cmd)
        self.assertIn("--trace", cmd)
        self.assertNotIn("--testdata", cmd)   # it scores nothing

    def test_the_pi_needs_a_step_graph_and_says_how_to_make_one(self):
        """
        The Pi runs ``<model>_step.onnx``, not ``models/incremental.py`` --
        it has no torch. Without that graph the request is refused with the
        command that produces it, rather than failing on the board.
        """
        # One that is otherwise ready to deploy, so the step graph is the only
        # thing missing and the refusal is about that rather than the export.
        name = next((m["name"] for m in pi.list_models()
                     if m["exported"] and not m["bidirectional"]
                     and not (pi.ARTIFACTS_DIR / f"{m['name']}_step.onnx").is_file()), None)
        if name is None:
            self.skipTest("every exported model already has a step graph")
        with self.assertRaises(FileNotFoundError) as caught:
            pi.build_steps({**self.SPEC, "model": name, "target": "pi",
                            "ssh": "pi@host", "incremental": True, "measure": True,
                            "allow_split_mismatch": True})
        self.assertIn("--incremental", str(caught.exception))

    def test_the_pi_is_told_to_run_the_step_graph(self):
        name = next((m["name"] for m in pi.list_models()
                     if (pi.ARTIFACTS_DIR / f"{m['name']}_step.onnx").is_file()), None)
        if name is None:
            self.skipTest("no step graph exported in this tree")
        steps = pi.build_steps({**self.SPEC, "model": name, "target": "pi",
                                "ssh": "pi@host", "incremental": True, "measure": True,
                                "testdata": pi.list_testdata()[0]["name"],
                                "allow_split_mismatch": True})
        self.assertEqual(steps[-1]["env"]["INCREMENTAL"], "1")

    def test_a_windowed_pi_run_does_not_ask_for_one(self):
        steps = pi.build_steps({**self.SPEC, "model": "tcn_mla_eog", "target": "pi",
                                "ssh": "pi@host", "measure": True,
                                "allow_split_mismatch": True})
        self.assertEqual(steps[-1]["env"]["INCREMENTAL"], "")

    def test_it_counts_as_something_to_run(self):
        """On its own it is a measurement, not an empty submission."""
        steps = pi.build_steps({**self.SPEC, "incremental": True,
                                "measure": False, "duty": False})
        self.assertTrue(steps)

    def test_it_needs_a_checkpoint(self):
        with self.assertRaises((FileNotFoundError, ValueError)):
            pi.build_steps({**self.SPEC, "model": "no_such_run", "incremental": True})


if __name__ == "__main__":
    unittest.main()


class SplitProvenanceTests(unittest.TestCase):
    """
    A test set is only a test set for a model trained at the same seed.

    ``EEGDenoiseDataset`` permutes the epoch pool with ``master_seed`` before
    slicing train/val/test out of it, so the seed-42 test epochs are 69% of
    them inside the seed-67 *training* split.  Scoring across that line does
    not crash or look odd — it returns a better number than the model earns.
    The pairing is therefore checked in three places (the page, the API, and
    the scorer on the board) and all three are covered here.
    """

    def test_provenance_is_read_from_the_file_when_recorded(self):
        import numpy as np
        from deploy import raspi_infer
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "BOTH_test40_s67.npz"
            np.savez_compressed(p, noisy=np.zeros((2, 512), np.float32),
                                clean=np.zeros((2, 512), np.float32),
                                snr_db=np.zeros(2, np.float32), fs=256,
                                master_seed=7, artifact="EMG", split="test")
            prov = raspi_infer.testdata_provenance(p)
        # The recorded values win over everything the name claims.
        self.assertEqual(prov["source"], "recorded")
        self.assertEqual((prov["master_seed"], prov["artifact"]), (7, "EMG"))

    def test_provenance_falls_back_to_the_filename(self):
        import numpy as np
        from deploy import raspi_infer
        with tempfile.TemporaryDirectory() as td:
            for name, seed, art in [("EOG_test100.npz", 42, "EOG"),
                                    ("BOTH_test100_s67.npz", 67, "BOTH")]:
                p = Path(td) / name
                np.savez_compressed(p, noisy=np.zeros((1, 512), np.float32))
                prov = raspi_infer.testdata_provenance(p)
                with self.subTest(name=name):
                    self.assertEqual(prov["source"], "filename")
                    self.assertEqual((prov["master_seed"], prov["artifact"]), (seed, art))

    def test_an_unrecognisable_file_is_unknown_not_a_guess(self):
        import numpy as np
        from deploy import raspi_infer
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "whatever.npz"
            np.savez_compressed(p, noisy=np.zeros((1, 512), np.float32))
            self.assertEqual(raspi_infer.testdata_provenance(p)["source"], "unknown")

    def test_the_scorer_refuses_a_seed_mismatch(self):
        from deploy import raspi_infer
        meta = {"provenance": {"trained_master_seed": 67, "trained_artifact": "BOTH"}}
        prov = {"source": "recorded", "master_seed": 42, "artifact": "BOTH", "split": "test"}
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "LEAK"):
                raspi_infer.check_split(meta, prov, strict=True)
            # ...and scores it anyway when told to, which is what the flag is for.
            problems = raspi_infer.check_split(meta, prov, strict=False)
        self.assertTrue(any("LEAK" in p for p in problems))

    def test_the_scorer_accepts_a_matching_pair(self):
        from deploy import raspi_infer
        meta = {"provenance": {"trained_master_seed": 67, "trained_artifact": "BOTH"}}
        prov = {"source": "recorded", "master_seed": 67, "artifact": "BOTH", "split": "test"}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(raspi_infer.check_split(meta, prov, strict=True), [])

    def test_unverifiable_pairs_warn_rather_than_refuse(self):
        """
        A run from before the seed was recorded cannot be checked.  That is a
        warning, not a refusal: refusing would make every legacy checkpoint
        unscoreable, and the warning says exactly what is unverified.
        """
        from deploy import raspi_infer
        meta = {"provenance": {"trained_master_seed": None, "trained_artifact": None}}
        prov = {"source": "recorded", "master_seed": 42, "artifact": "EOG", "split": "test"}
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(raspi_infer.check_split(meta, prov, strict=True), [
                p for p in raspi_infer.check_split(meta, prov, strict=True)])
        self.assertIn("cannot verify the split", out.getvalue())

    def test_a_non_test_split_is_refused(self):
        from deploy import raspi_infer
        meta = {"provenance": {"trained_master_seed": 42, "trained_artifact": "EOG"}}
        prov = {"source": "recorded", "master_seed": 42, "artifact": "EOG", "split": "train"}
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "not the test split"):
                raspi_infer.check_split(meta, prov, strict=True)

    def test_make_testdata_stamps_what_it_cut(self):
        """The provenance has to describe the cut, not the default."""
        import numpy as np
        from deploy.make_testdata import make_testdata
        data_dir = Path("data/EEGdenoisenet/data")
        if not (data_dir / "EEG_all_epochs.npy").is_file():
            self.skipTest("EEGdenoiseNet .npy files are not present")
        with tempfile.TemporaryDirectory() as td:
            with contextlib.redirect_stdout(io.StringIO()):
                p = make_testdata(str(data_dir), "EMG", 10, out_dir=td, master_seed=13)
            d = np.load(p)
            self.assertEqual((int(d["master_seed"]), str(d["artifact"]), str(d["split"])),
                             (13, "EMG", "test"))
            self.assertEqual(p.name, "EMG_test10_s13.npz")

    def test_the_bundled_sets_carry_provenance(self):
        """They are shipped in the repo, so they are the ones most likely reused."""
        for t in pi.list_testdata():
            with self.subTest(name=t["name"]):
                self.assertEqual(t["source"], "recorded",
                                 f"{t['name']} has no recorded provenance; re-cut it")

    def test_the_api_refuses_a_leaking_pair(self):
        """
        Checked server-side too: the page can be stale and the API is reachable
        on its own, so the browser's banner cannot be the only guard.
        """
        sets = {t["name"]: t for t in pi.list_testdata()}
        models = {m["name"]: m for m in pi.list_models() if m.get("master_seed") is not None}
        if not sets or not models:
            self.skipTest("no exported models or test sets to pair")

        pair = next(((m, t) for m in models.values() for t in sets.values()
                     if int(m["master_seed"]) != int(t["master_seed"])), None)
        if pair is None:
            self.skipTest("no mismatched pair available to test with")
        model, testset = pair

        spec = {"model": model["name"], "testdata": testset["name"]}

        # The run path refuses: this is the call that goes on to produce a score.
        with self.assertRaises(ValueError):
            pi._testdata_name({**spec, "_enforce_split": True})

        # Preview does not. It is read-only, the page already shows the banner,
        # and blocking it would hide the command from someone who means to copy
        # it and add --allow-split-mismatch.
        self.assertEqual(pi._testdata_name(spec), testset["name"])

        # The override exists because a deliberate probe is a legitimate thing
        # to run; it just must not be the default.
        self.assertEqual(
            pi._testdata_name({**spec, "_enforce_split": True,
                               "allow_split_mismatch": True}),
            testset["name"])

    def test_split_mismatch_levels(self):
        matched = next((t for t in pi.list_testdata() if t["master_seed"] == 67), None)
        mismatched = next((t for t in pi.list_testdata() if t["master_seed"] == 42), None)
        model = next((m for m in pi.list_models() if m.get("master_seed") == 67), None)
        if not (matched and mismatched and model):
            self.skipTest("the fixtures this needs are not in the working tree")
        self.assertIsNone(pi.split_mismatch(model["name"], matched["name"]))
        bad = pi.split_mismatch(model["name"], mismatched["name"])
        self.assertEqual(bad["level"], "error")
        self.assertIn("master_seed", bad["text"])


if __name__ == "__main__":
    unittest.main()
