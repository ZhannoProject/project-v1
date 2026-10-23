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

    def test_the_same_seed_is_the_ordinary_retrain(self):
        d = _run_dir(self.tmp, "again", seed=42)
        ensure_writable(d, 42, False)                        # must not raise

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
    """The seed is part of a run's identity, so it is part of its path."""

    def derive(self, variant, artifact, seed, distil=False):
        from trainer.__main__ import DEFAULT_MASTER_SEED
        suffix = "" if seed == DEFAULT_MASTER_SEED else f"_s{seed}"
        return f"./runs/{variant}_{artifact.lower()}" + ("_kd" if distil else "") + suffix

    def test_the_source_formula_is_the_one_tested_here(self):
        """
        Keeps this test honest if the derivation in __main__ moves.

        The formula is no longer one line: the bare name goes to the default
        seed only when nothing else holds it, so the source also consults
        what is on disk. Both halves are pinned.
        """
        src = (Path(__file__).resolve().parents[1] / "trainer" / "__main__.py").read_text()
        self.assertIn('seeded = f"{base}_s{cfg[\'master_seed\']}"', src)
        self.assertIn('+ ("_kd" if distil else "")', src)
        self.assertIn('recorded_seed(Path(base))', src)

    def test_the_default_seed_keeps_the_bare_path(self):
        """Every directory already in runs/, and every path in the README and
        the deploy scripts, has to go on meaning what it meant."""
        from trainer.__main__ import DEFAULT_MASTER_SEED
        self.assertEqual(self.derive("tcn_mla", "EOG", DEFAULT_MASTER_SEED),
                         "./runs/tcn_mla_eog")

    def test_other_seeds_get_their_own_folder(self):
        self.assertEqual(self.derive("tcn", "BOTH", 55), "./runs/tcn_both_s55")
        self.assertEqual(self.derive("tcn", "BOTH", 67), "./runs/tcn_both_s67")

    def test_two_seeds_of_one_model_never_collide(self):
        paths = {self.derive("tcn_mla", "BOTH", s) for s in (42, 55, 67)}
        self.assertEqual(len(paths), 3)

    def test_distillation_keeps_its_own_suffix(self):
        self.assertEqual(self.derive("tcn_mla", "EOG", 7, distil=True),
                         "./runs/tcn_mla_eog_kd_s7")


class DefaultSeedFindsAFreeFolderTests(unittest.TestCase):
    """
    The default seed keeps the bare name only while the bare name is its own.

    The rule "seed 42 keeps runs/<variant>_<artifact>" assumed that folder was
    free. It is not always: runs/tcn_both holds a *seed-67* run, trained
    before this convention existed. So the default seed derived a path it was
    not allowed to write, and a multi-seed trial -- change the seed, press
    Train -- was refused for the one seed that should have been easiest.

    Diverting is only right when the clash is certain. A folder whose seed was
    never recorded cannot be shown to be a different run, and inventing a
    second folder for what may be the same one is a guess, so that case is
    still handed to the guard.
    """

    def _derive(self, runs, variant, artifact, seed, default=42):
        """The derivation under test, mirrored from trainer/__main__."""
        from trainer.checkpoint import recorded_seed
        base = runs / f"{variant}_{artifact.lower()}"
        if seed != default:
            return f"{base}_s{seed}"
        held, files = recorded_seed(base)
        return str(f"{base}_s{seed}") if (files and held is not None and held != seed) else str(base)

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
        self._run(runs, "tcn_both", seed=67)
        self.assertEqual(self._derive(runs, "tcn", "BOTH", 42),
                         str(runs / "tcn_both_s42"),
                         "seed 42 was sent at a folder held by seed 67")

    def test_a_free_or_own_folder_still_gets_the_bare_name(self):
        """Every path already in runs/, the README and the deploy scripts."""
        tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        runs = tmp / "runs"; runs.mkdir()
        self.assertEqual(self._derive(runs, "tcn", "EOG", 42), str(runs / "tcn_eog"))
        self._run(runs, "tcn_eog", seed=42)
        self.assertEqual(self._derive(runs, "tcn", "EOG", 42), str(runs / "tcn_eog"),
                         "its own folder is still its own")

    def test_an_unrecorded_seed_is_left_to_the_guard(self):
        """
        Not divertable: it may be the same run. Refusing asks the operator,
        which is the whole point of the guard.
        """
        tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        runs = tmp / "runs"; runs.mkdir()
        self._run(runs, "tcn_eog", seed=None)
        self.assertEqual(self._derive(runs, "tcn", "EOG", 42), str(runs / "tcn_eog"))

    def test_a_non_default_seed_is_unchanged(self):
        tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        runs = tmp / "runs"; runs.mkdir()
        self.assertEqual(self._derive(runs, "tcn", "BOTH", 67), str(runs / "tcn_both_s67"))

    def test_the_trainer_uses_this_rule(self):
        """The mirror above has to stay a mirror."""
        source = (Path(__file__).resolve().parent.parent / "trainer" / "__main__.py").read_text()
        block = source[source.index('if cfg.get("save_dir") is None:'):][:2000]
        self.assertIn("recorded_seed(Path(base))", block)
        self.assertIn("held is not None and held != cfg[\"master_seed\"]", block)


if __name__ == "__main__":
    unittest.main()
