"""
Runs are kept apart by seed.

A checkpoint directory is named after the variant and artifact, so without
this two runs of the same model at different seeds land in the same folder and
the second destroys the first — in place, with no copy anywhere.  That is not
hypothetical: it happened to this project's own BOTH runs twice, taking the
seed-42 and then the seed-55 weights with it.

Two things keep it from happening again and both are checked here: the default
path carries the seed, and starting on top of another run is refused.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainer.checkpoint import ensure_writable, recorded_seed


def _run_dir(root: Path, name: str, seed=None, files=("history.json",)) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f == "history.json":
            cfg = {"master_seed": seed} if seed is not None else {}
            (d / f).write_text(json.dumps({"config": cfg, "train_loss": [1.0]}))
        else:
            (d / f).write_bytes(b"not really a checkpoint")
    return d


class RecordedSeedTests(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="run_iso_"))

    def test_an_empty_directory_holds_no_run(self):
        self.assertEqual(recorded_seed(self.tmp / "nothing"), (None, []))
        (self.tmp / "empty").mkdir()
        self.assertEqual(recorded_seed(self.tmp / "empty"), (None, []))

    def test_a_directory_with_only_loose_files_holds_no_run(self):
        d = self.tmp / "loose"
        d.mkdir()
        (d / "loss_fn.json").write_text("{}")
        self.assertEqual(recorded_seed(d), (None, []))

    def test_the_seed_is_read_from_history(self):
        d = _run_dir(self.tmp, "seeded", seed=55)
        self.assertEqual(recorded_seed(d), (55, ["history.json"]))

    def test_a_legacy_run_is_a_run_with_an_unknown_seed(self):
        """Not the same as an empty directory, and the caller must not confuse
        the two: one is safe to write to, the other is somebody's results."""
        d = _run_dir(self.tmp, "legacy", seed=None, files=("history.json", "best.pt"))
        seed, files = recorded_seed(d)
        self.assertIsNone(seed)
        self.assertEqual(sorted(files), ["best.pt", "history.json"])


class EnsureWritableTests(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="run_iso_"))

    def test_an_unused_directory_is_fine(self):
        ensure_writable(self.tmp / "fresh", 42, False)       # must not raise

    def test_the_same_seed_requires_explicit_overwrite(self):
        d = _run_dir(self.tmp, "again", seed=42)
        with self.assertRaises(FileExistsError):
            ensure_writable(d, 42, False)
        ensure_writable(d, 42, True)

    def test_a_different_seed_is_refused(self):
        """The exact thing that lost this project's seed-55 runs."""
        d = _run_dir(self.tmp, "clash", seed=55, files=("history.json", "best.pt", "last.pt"))
        with self.assertRaises(FileExistsError) as cm:
            ensure_writable(d, 67, False)
        msg = str(cm.exception)
        self.assertIn("master_seed=55", msg)
        self.assertIn("67", msg)
        self.assertIn("--overwrite", msg)
        # the message has to say what to do instead, not just complain
        self.assertIn("save_dir", msg)

    def test_an_unrecorded_seed_is_refused_too(self):
        """It cannot be shown to be the same run, so it is treated as another."""
        d = _run_dir(self.tmp, "legacy", seed=None, files=("history.json", "best.pt"))
        with self.assertRaises(FileExistsError):
            ensure_writable(d, 42, False)

    def test_overwrite_is_the_way_through(self):
        for seed in (None, 55):
            d = _run_dir(self.tmp, f"ow_{seed}", seed=seed)
            ensure_writable(d, 67, True)                     # must not raise

    def test_nothing_is_created_or_removed_by_the_check(self):
        d = _run_dir(self.tmp, "intact", seed=55)
        before = sorted(p.name for p in d.iterdir())
        with self.assertRaises(FileExistsError):
            ensure_writable(d, 67, False)
        self.assertEqual(sorted(p.name for p in d.iterdir()), before)
        self.assertFalse((self.tmp / "never_made").exists())
        ensure_writable(self.tmp / "never_made", 42, False)
        self.assertFalse((self.tmp / "never_made").exists(),
                         "the check must not create the directory it is asked about")


class DefaultSaveDirTests(unittest.TestCase):
    """An automatic path identifies the training rate and seed."""

    def derive(self, variant, artifact, seed, distil=False, rate=None):
        from experiment.config import resolve_save_dir
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(variant=variant, artifact=artifact, master_seed=seed, target_fs=rate,
                       teacher_ckpt="teacher.pt" if distil else None)
            return resolve_save_dir(cfg, Path(tmp))["save_dir"]

    def test_resolver_is_shared_with_the_cli(self):
        from trainer.__main__ import resolve_save_dir as cli_resolve
        from experiment.config import resolve_save_dir
        self.assertIs(cli_resolve, resolve_save_dir)

    def test_default_seed_omits_the_seed_suffix(self):
        from trainer.__main__ import DEFAULT_MASTER_SEED
        self.assertEqual(self.derive("tcn_mla", "EOG", DEFAULT_MASTER_SEED),
                         "./runs/tcn_mla_eog_256hz")

    def test_other_seeds_get_their_own_folder(self):
        self.assertEqual(self.derive("tcn", "BOTH", 55), "./runs/tcn_both_256hz_s55")
        self.assertEqual(self.derive("tcn", "BOTH", 67), "./runs/tcn_both_256hz_s67")

    def test_250_hz_is_distinct_and_explicit_256_uses_the_native_path(self):
        self.assertEqual(self.derive("tcn_se", "BOTH", 67, rate=250), "./runs/tcn_se_both_250hz_s67")
        self.assertEqual(self.derive("tcn_se", "BOTH", 67, rate=250.0), "./runs/tcn_se_both_250hz_s67")
        self.assertEqual(self.derive("tcn_se", "BOTH", 67, rate=256), self.derive("tcn_se", "BOTH", 67))

    def test_two_seeds_of_one_model_never_collide(self):
        paths = {self.derive("tcn_mla", "BOTH", s) for s in (42, 55, 67)}
        self.assertEqual(len(paths), 3)

    def test_distillation_keeps_its_own_suffix(self):
        self.assertEqual(self.derive("tcn_mla", "EOG", 7, distil=True),
                         "./runs/tcn_mla_eog_kd_256hz_s7")


class DefaultSeedFindsAFreeFolderTests(unittest.TestCase):
    """
    The default seed keeps the bare name only while the bare name is its own.

    When the rate-qualified folder is already used by a different seed,
    derive a path with the default seed spelled out.

    Diverting is only right when the clash is certain. A folder whose seed was
    never recorded cannot be shown to be a different run, and inventing a
    second folder for what may be the same one is a guess, so that case is
    still handed to the guard.
    """

    def _derive(self, runs, variant, artifact, seed, default=42):
        from experiment.config import resolve_save_dir
        cfg = resolve_save_dir(dict(variant=variant, artifact=artifact, master_seed=seed), runs.parent)
        return str(runs.parent / cfg["save_dir"])

    def _run(self, runs, name, seed):
        d = runs / name
        d.mkdir(parents=True)
        (d / "history.json").write_text(json.dumps(
            {"train_loss": [1.0], "val_loss": [1.0], "rrmse_t": [0.7], "rrmse_s": [0.6],
             "acc": [0.8], "config": {"master_seed": seed} if seed is not None else {}}))
        return d

    def test_the_default_seed_steps_aside_for_a_known_other_seed(self):
        tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        runs = tmp / "runs"; runs.mkdir()
        self._run(runs, "tcn_both_256hz", seed=67)
        self.assertEqual(self._derive(runs, "tcn", "BOTH", 42),
                         str(runs / "tcn_both_256hz_s42"),
                         "seed 42 was sent at a folder held by seed 67")

    def test_a_free_or_own_folder_still_gets_the_bare_name(self):
        tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        runs = tmp / "runs"; runs.mkdir()
        self.assertEqual(self._derive(runs, "tcn", "EOG", 42), str(runs / "tcn_eog_256hz"))
        self._run(runs, "tcn_eog_256hz", seed=42)
        self.assertEqual(self._derive(runs, "tcn", "EOG", 42), str(runs / "tcn_eog_256hz"),
                         "its own folder is still its own")

    def test_an_unrecorded_seed_is_left_to_the_guard(self):
        """
        Not divertable: it may be the same run. Refusing asks the operator,
        which is the whole point of the guard.
        """
        tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        runs = tmp / "runs"; runs.mkdir()
        self._run(runs, "tcn_eog_256hz", seed=None)
        self.assertEqual(self._derive(runs, "tcn", "EOG", 42), str(runs / "tcn_eog_256hz"))

    def test_a_non_default_seed_keeps_its_seed_suffix(self):
        tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        runs = tmp / "runs"; runs.mkdir()
        self.assertEqual(self._derive(runs, "tcn", "BOTH", 67), str(runs / "tcn_both_256hz_s67"))

    def test_the_trainer_uses_this_rule(self):
        from trainer.__main__ import resolve_save_dir as cli_resolve
        from experiment.config import resolve_save_dir
        self.assertIs(cli_resolve, resolve_save_dir)


if __name__ == "__main__":
    unittest.main()
