"""Special experiment families stay separate without relocating any runs."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gui import config_io


class RunGroupTests(unittest.TestCase):
    def test_rf_default_rung_is_still_a_sweep_not_direct_training(self):
        self.assertEqual(config_io.run_category("tcn_rf125"), "receptive")
        self.assertEqual(config_io.run_category("tcn_se_rf125"), "receptive")
        self.assertEqual(config_io.run_category("tcn"), "direct")
        self.assertEqual(config_io.run_category("tcn_se_bi"), "direct")

    def test_recorded_model_wins_over_a_misleading_folder_name(self):
        self.assertEqual(config_io.run_category("tcn_rf509", {"variant":"tcn"}), "direct")
        self.assertEqual(config_io.run_category("renamed", {"variant":"tcn_se_la16"}), "lookahead")
        self.assertEqual(config_io.run_category("renamed", {}, {"arch":{"variant":"tcn_rf29"}}), "receptive")

    def test_delayed_output_and_legacy_lookahead_share_their_own_group(self):
        self.assertEqual(config_io.run_category("tcn_se", {}, {"output_delay":16}), "lookahead")
        self.assertEqual(config_io.run_category("tcn_se", {}, {"output_delay":0}), "direct")
        self.assertEqual(config_io.run_category("tcn_se", {}, {"arch":{"lookahead":8}}), "lookahead")

    def test_distillation_and_rank_experiments_are_separate(self):
        self.assertEqual(config_io.run_category("tcn_se", {}, {"teacher_ckpt":"teacher/best.pt"}), "distillation")
        self.assertEqual(config_io.run_category("tcn_se", kd=True), "distillation")
        self.assertEqual(config_io.run_category("tcn_mla_r16"), "other")
        self.assertEqual(config_io.run_category("tcn_se_r4"), "other")

    def test_custom_dilation_is_rf_but_default_metadata_is_not(self):
        self.assertEqual(config_io.run_category("tcn", {"dilations":[1,2,4]}), "receptive")
        self.assertEqual(config_io.run_category("tcn", {"dilations":[1,2,4,8,16]}), "direct")

    def test_listing_keeps_every_checkpoint_and_classifies_interrupted_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("plain", "sweep", "delayed", "kd_interrupted"):
                d=root/name; d.mkdir(); (d/"best.pt").write_bytes(b"fixture")
            (root/"sweep"/"history.json").write_text(json.dumps({
                "config":{"arch":{"variant":"tcn_rf125"}}, "val_loss":[1.],
                "rrmse_t":[.7], "rrmse_s":[.6], "acc":[.8]}))
            def info(path):
                cfg = ({"output_delay":16} if path.parent.name=="delayed" else
                       {"teacher_ckpt":"teacher/best.pt"} if path.parent.name=="kd_interrupted" else {})
                return {"epoch":1, "run_config":cfg}
            with patch.object(config_io,"RUNS_DIR",root), patch.object(config_io,"_checkpoint_info",side_effect=info):
                rows = config_io.list_runs()
            self.assertEqual({r['name']:r['category'] for r in rows},
                             dict(plain="direct",sweep="receptive",delayed="lookahead",kd_interrupted="distillation"))
            self.assertEqual(next(r for r in rows if r['name']=="kd_interrupted")['status'],"interrupted")
            self.assertEqual(sum(g['n'] for g in config_io.run_categories(rows)),4)
            self.assertEqual(sorted(p.name for p in root.iterdir()),["delayed","kd_interrupted","plain","sweep"])

    def test_protocol_comes_from_the_checkpoint_and_is_separate_from_run_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("legacy", "exploratory", "paper", "unknown"):
                run = root / name
                run.mkdir()
                (run / "best.pt").touch()
                (run / "history.json").write_text(json.dumps({
                    "config": {"protocol": "paper"}, "val_loss": [1.0],
                    "rrmse_t": [0.7], "rrmse_s": [0.6], "acc": [0.8],
                }))

            def info(path):
                name = path.parent.name
                if not path.is_file() or name == "unknown":
                    return {}
                return {"epoch": 1, "run_config": {} if name == "legacy" else {"protocol": name}}

            with patch.object(config_io, "RUNS_DIR", root), patch.object(config_io, "_checkpoint_info", side_effect=info):
                rows = config_io.list_runs()

            self.assertEqual({r["name"]: r["training_protocol"] for r in rows}, {
                "legacy": "legacy", "exploratory": "exploratory", "paper": "paper", "unknown": "unknown",
            })
            self.assertTrue(all(r["status"] == "complete" for r in rows))

    def test_qualified_requires_matching_legacy_checkpoints_and_validation_history(self):
        recipe = dict(artifact="BOTH", master_seed=42, max_epochs=200,
                      optimizer="AdamW", loss_fn="mse")
        curve = [1.0, 0.5] + [0.6] * 198
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("verified", "different_curve", "missing_last", "paper"):
                run = root / name
                run.mkdir()
                (run / "best.pt").touch()
                if name != "missing_last":
                    (run / "last.pt").touch()
                (run / "history.json").write_text(json.dumps({
                    "config": recipe, "val_loss": curve,
                    "rrmse_t": [0.7] * 200, "rrmse_s": [0.6] * 200, "acc": [0.8] * 200,
                }))

            def info(path):
                if not path.is_file():
                    return {}
                epoch = 2 if path.name == "best.pt" else 200
                saved_curve = curve[:epoch]
                if path.parent.name == "different_curve":
                    saved_curve[0] = 0.9
                cfg = {**recipe, "protocol": "paper"} if path.parent.name == "paper" else recipe
                return {"epoch": epoch, "run_config": cfg, "optimizer": {"family": "AdamW"},
                        "val_loss": curve[epoch - 1], "validation_history": saved_curve}

            with patch.object(config_io, "RUNS_DIR", root), patch.object(config_io, "_checkpoint_info", side_effect=info):
                rows = config_io.list_runs()

        self.assertEqual([r["name"] for r in rows if r["qualified"]], ["verified"])


if __name__ == "__main__":
    unittest.main()
