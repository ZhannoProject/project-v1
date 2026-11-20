"""Family-wide EEG-IO selection and matched-seed statistics."""

import unittest
from unittest.mock import patch

from gui import lookahead, eegio_tab


class LookaheadBatchTests(unittest.TestCase):
    def setUp(self):
        self.records = []
        for seed in (42, 54, 67):
            for budget in (0, 8, None):
                variant = "tcn_se" if budget == 0 else "tcn_se_bi" if budget is None else "tcn_se_la8"
                self.records.append(dict(name=f"s{seed}_{budget}", variant=variant, artifact="BOTH",
                                         status="complete", hidden_channels=64, params=123838,
                                         tcn_norm="channel", train_fs=250., input_normalization="epoch",
                                         master_seed=seed, epochs=200, loss_fn="mse", kd=False))
        self.records += [dict(self.records[1], name="s42_1", variant="tcn_se_la1"),
                         dict(self.records[1], name="zz_duplicate"),
                         dict(self.records[1], name="unfinished", status="interrupted"),
                         dict(self.records[1], name="wrong_rate", train_fs=256.),
                         dict(self.records[1], name="wrong_norm", tcn_norm="batch"),
                         dict(self.records[1], name="unpaired_seed", master_seed=99)]

    def info(self, path):
        run = next(r for r in self.records if r["name"] == path.parent.name)
        budget = lookahead._rung_of(run)
        return dict(model_config={"causal": budget == 0, "se_reduction": 8},
                    run_config={"batch_size": 32}, optimizer={"name": "AdamW"})

    def family(self):
        return next(f for f in lookahead._se_families(self.records).values()
                    if f.get("tcn_norm") == "channel" and f["sample_rate_hz"] == 250.)

    def test_batch_contains_all_seeds_and_available_rungs_with_no_duplicate_weighting(self):
        with patch.object(lookahead, "list_runs", return_value=self.records), \
             patch.object(lookahead, "_checkpoint_info", side_effect=self.info):
            family = self.family()
            groups = lookahead._eegio_groups(lookahead.ladder_runs("BOTH", family, records=self.records), family, 64)
            self.assertEqual(len(groups), 1)
            group = groups[0]
            self.assertEqual(group["seeds"], [42, 54, 67])
            self.assertEqual(len(group["runs"]), 10)
            self.assertEqual(group["runs"][:3], ["s42_0", "s54_0", "s67_0"])
            self.assertEqual(next(c["seeds"] for c in group["coverage"] if c["lookahead"] == 1), [42])
            self.assertEqual(group["duplicates"], ["zz_duplicate"])
            spec, comparison = lookahead.eegio_spec({"variant": family["id"], "comparison_id": group["id"]})
            self.assertEqual(spec["runs"], group["runs"])
            self.assertTrue(spec["offline"])
            self.assertEqual(comparison["baseline_by_run"]["s67_8"], "s67_0")
            self.assertEqual(comparison["rungs"]["s54_None"], None)
            for bad in ("obsolete", None):
                with self.assertRaises(ValueError):
                    lookahead.eegio_spec({"variant": family["id"], "comparison_id": bad})

    def test_different_training_recipes_are_separate_choices(self):
        self.records += [dict(self.records[0], name="short_base", epochs=100),
                         dict(self.records[1], name="short_rung", epochs=100)]
        with patch.object(lookahead, "_checkpoint_info", side_effect=self.info):
            family = self.family()
            groups = lookahead._eegio_groups(lookahead.ladder_runs("BOTH", family, records=self.records), family, 64)
        self.assertEqual(len(groups), 2)
        self.assertEqual(set(groups[1]["runs"]), {"short_base", "short_rung"})
        self.assertIn("100 epochs", groups[1]["label"])

    def test_statistics_repeat_over_seed_medians_not_pooled_subjects(self):
        comparison = {"members": [{"run": "a", "master_seed": 42, "lookahead": 8},
                                   {"run": "b", "master_seed": 54, "lookahead": 8},
                                   {"run": "c", "master_seed": 67, "lookahead": 16}]}
        results = [dict(run="a", summary={"metric": {"median": 2., "n": 20}}),
                   dict(run="b", summary={"metric": {"median": 4., "n": 10}}),
                   dict(run="c", summary={"metric": {"median": 1., "n": 20}})]
        summary = eegio_tab._lookahead_seed_summary(results, comparison)
        metric = summary[0]["metrics"]["metric"]
        self.assertEqual(metric["n"], 2)
        self.assertEqual(metric["mean"], 3.)
        self.assertAlmostEqual(metric["sd"], 2 ** .5)
        self.assertIsNone(summary[1]["metrics"]["metric"]["sd"])

    def test_batch_uses_each_seeds_own_blink_reference_and_paired_comparisons(self):
        import eegio_eval as E

        names = ["base42", "base54", "rung42", "rung54"]
        members = [{"run": name, "master_seed": 42 if name.endswith("42") else 54,
                    "lookahead": 0 if name.startswith("base") else 8} for name in names]
        references = {name: "base" + name[-2:] for name in names}
        comparison = {"schema_version": 2, "members": members,
                      "baselines": {"42": "base42", "54": "base54"}, "baseline_by_run": references}
        received, compared = {}, []

        def evaluate(data_dir, name, subjects, cfg, **kwargs):
            received[name] = kwargs.get("reference_thresholds_uv")
            threshold = 10. if name.endswith("42") else 20.
            return dict(run=name, rows=[{"subject": "S00", "metric_support": {
                "adjusted_blink_count": {"threshold_uv": threshold}}}],
                summary={"metric": {"median": threshold, "n": 1}},
                config={"offline_window": {"samples": 500, "left": 124, "right": 125}}, causality={})

        def compare(results):
            compared.append([r["run"] for r in results])
            return [{"a": results[0]["run"], "b": results[1]["run"]}]

        with patch.object(eegio_tab, "dataset_status", return_value={"present": True, "subjects": ["S00"]}), \
             patch.object(eegio_tab, "available_runs", return_value=[]), \
             patch.object(eegio_tab, "save_result"), \
             patch.object(E, "prepare_offline", side_effect=lambda runs, cfg: (cfg, dict.fromkeys(runs))), \
             patch.object(E, "compare", side_effect=compare):
            out = eegio_tab.evaluate({"runs": names, "offline": True, "target_fs": 250},
                                     evaluate_model=evaluate, lookahead_comparison=comparison)
        self.assertIsNone(received["base42"])
        self.assertIsNone(received["base54"])
        self.assertEqual(received["rung42"], {"S00": 10.})
        self.assertEqual(received["rung54"], {"S00": 20.})
        self.assertEqual(compared, [["base42", "rung42"], ["base54", "rung54"]])
        self.assertEqual([t["master_seed"] for t in out["tests"]], [42, 54])
        self.assertEqual(eegio_tab._describe(out, "test")["lookahead_summary"], out["lookahead_summary"])

    def test_offline_reference_alone_can_read_zero_phase_input(self):
        import eegio_eval as E

        names = ["base42", "rung42", "off42"]
        members = [{"run": "base42", "master_seed": 42, "lookahead": 0},
                   {"run": "rung42", "master_seed": 42, "lookahead": 8},
                   {"run": "off42", "master_seed": 42, "lookahead": None}]
        comparison = {"schema_version": 2, "members": members, "baselines": {"42": "base42"},
                      "baseline_by_run": dict.fromkeys(names, "base42")}
        filters = {}

        def evaluate(data_dir, name, subjects, cfg, **kwargs):
            filters[name] = cfg.zero_phase
            return dict(run=name, rows=[{"subject": "S00", "metric_support": {
                "adjusted_blink_count": {"threshold_uv": 10.}}}],
                summary={"metric": {"median": 1., "n": 1}},
                config={"offline_window": {"samples": 500, "left": 125, "right": 125}}, causality={})

        def run(spec):
            with patch.object(eegio_tab, "dataset_status", return_value={"present": True, "subjects": ["S00"]}), \
                 patch.object(eegio_tab, "available_runs", return_value=[]), \
                 patch.object(eegio_tab, "save_result"), \
                 patch.object(E, "prepare_offline", side_effect=lambda runs, cfg: (cfg, dict.fromkeys(runs))), \
                 patch.object(E, "compare", return_value=[]):
                return eegio_tab.evaluate({"runs": names, "offline": True, "target_fs": 250, **spec},
                                          evaluate_model=evaluate, lookahead_comparison=comparison)

        out = run({"offline_zero_phase": True})
        self.assertEqual(filters, {"base42": False, "rung42": False, "off42": True})
        self.assertEqual((out["saved"]["zero_phase"], out["saved"]["zero_phase_runs"]), (False, ["off42"]))
        described = eegio_tab._describe(out, "test")
        self.assertEqual((described["offline_zero_phase"], described["zero_phase_runs"]), (True, ["off42"]))
        # Saved results without the option read as before.
        out = run({})
        self.assertNotIn("offline_zero_phase", out["saved"])
        self.assertFalse(eegio_tab._describe(out, "test")["offline_zero_phase"])
        self.assertEqual(set(filters.values()), {False})
        with self.assertRaises(ValueError):
            run({"offline_zero_phase": True, "zero_phase": True})
        comparison["members"] = members[:2]
        with self.assertRaises(ValueError):
            run({"offline_zero_phase": True})

    def test_offline_reference_runs_for_every_comparison_shape(self):
        self.assertEqual(eegio_tab._offline_reference_runs({"rungs": {"a": 0, "b": 8, "c": None}}, ["a", "b", "c"]), ["c"])
        self.assertEqual(eegio_tab._offline_reference_runs({"lookahead": None}, ["a", "b"]), ["b"])
        self.assertEqual(eegio_tab._offline_reference_runs({"lookahead": 8}, ["a", "b"]), [])


class LookaheadBatchApiTests(unittest.TestCase):
    setUp = LookaheadBatchTests.setUp
    info = LookaheadBatchTests.info
    family = LookaheadBatchTests.family
    def test_family_request_resolves_every_seed_on_the_server(self):
        import json
        import threading
        import urllib.request
        from gui.server import GUIServer

        with patch.object(lookahead, "list_runs", return_value=self.records), \
             patch.object(lookahead, "_checkpoint_info", side_effect=self.info):
            family = self.family()
            groups = lookahead._eegio_groups(lookahead.ladder_runs("BOTH", family, records=self.records), family, 64)
            payload = {"variant": family["id"], "comparison_id": groups[0]["id"], "n_subjects": 5}
            with GUIServer("127.0.0.1", 0) as server, \
                 patch.object(eegio_tab, "evaluate", return_value={"ok": True}) as evaluate:
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    request = urllib.request.Request(f"http://127.0.0.1:{server.server_address[1]}/api/lookahead/eegio",
                                                     data=json.dumps(payload).encode(),
                                                     headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(request) as response:
                        self.assertEqual(json.load(response), {"ok": True})
                finally:
                    server.shutdown()
                    thread.join(timeout=2)
            spec = evaluate.call_args.args[0]
            comparison = evaluate.call_args.kwargs["lookahead_comparison"]
            self.assertEqual(len(spec["runs"]), 10)
            self.assertEqual(spec["n_subjects"], 5)
            self.assertEqual(comparison["seeds"], [42, 54, 67])


class LookaheadBatchPageTests(unittest.TestCase):
    def test_page_submits_family_scope_and_keeps_legacy_history_separate(self):
        import shutil
        import subprocess
        import os
        from pathlib import Path

        node = os.environ.get("NODE_BINARY") or shutil.which("node")
        if not node:
            self.skipTest("Node is unavailable for JavaScript logic checks")
        root = Path(__file__).resolve().parents[1]
        script = r'''
const fs = require('fs'), vm = require('vm'), assert = require('assert/strict');
const page = fs.readFileSync('gui/static/index.html', 'utf8');
const extract = name => {
  const match = page.match(new RegExp('(?:async )?function ' + name + '\\([^]*?\\n  \\}'));
  assert.ok(match, name); return match[0];
};
const group = {id:'recipe', target_fs:250, runs:['a42','a54','b42','b54'], baselines:{42:'a42',54:'a54'}};
const entry = (name, comparison, runs=group.runs) => ({name, lookahead_comparison:comparison,
    runs, target_fs:250, zero_phase:false});
const state = {la:{data:{eegio_groups:[group]}, eegio:{}}, meta:{eegio_tests:[
  entry('batch',{schema_version:2,id:'recipe'}),
  entry('legacy',{master_seed:42},['a42','b42']),
  entry('other',{schema_version:2,id:'different'}),
]}};
let sent;
const controls = {'#laEegioN':{value:'20',reportValidity:()=>true},
  '#btnLaEegio':{disabled:false}, '#laEegioSaved':{value:''}};
const context = {state, $:id=>{assert.ok(controls[id],id);return controls[id]},
  lookaheadSettings:()=>({variant:'tcn_se/channel/250',artifact:'BOTH',hidden_channels:64}),
  laEegioZeroPhase:()=>false, renderLookahead:()=>{},renderLaEegio:()=>{},
  renderLaEegioProgress:()=>{}, refreshEegioHistory:async()=>{},
  streamEegio:async(spec,progress,endpoint)=>{sent={spec,endpoint};return {name:'new-test'}}};
vm.runInNewContext(['laEegioGroup','syncLaEegioSelection','laEegioHistory','runLaEegio'].map(extract).join('\n'),context);
(async()=>{
  context.syncLaEegioSelection(); assert.equal(state.la.eegio.comparison,'recipe');
  assert.equal(context.laEegioHistory(group).map(r=>r.name).join(','),'batch,legacy');
  await context.runLaEegio();
  assert.equal(sent.spec.comparison_id,'recipe'); assert.equal(sent.spec.n_subjects,20);
  assert.equal(sent.endpoint,'/api/lookahead/eegio'); assert.equal('runs' in sent.spec,false);
  assert.equal(state.la.eegio.busy,false); assert.equal(controls['#laEegioSaved'].value,'new-test');
  state.la.data.eegio_groups=[{...group,id:'another'}];
  context.syncLaEegioSelection(); assert.equal(state.la.eegio.comparison,'another');
})().catch(error=>{console.error(error);process.exitCode=1});
'''
        result = subprocess.run([node, "-e", script], cwd=root, capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
