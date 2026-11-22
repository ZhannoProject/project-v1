"""Release registration and model-picker integration, with no EEG data or Pi."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import prepare_gui
from gui import channels, config_io, eegio_tab, pi, server

ROOT = Path(__file__).resolve().parents[1]


class ReleaseRegistration(unittest.TestCase):
    def test_existing_training_checkpoint_is_preserved(self):
        item = next(r for r in json.loads((ROOT/'pretrained/manifest.json').read_text())['models']
                    if r['id'] == 'se_l8_s67')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'pretrained').mkdir()
            (root/'pretrained/manifest.json').write_text(json.dumps({'models': [item]}))
            source = root/item['path']
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/item['path'], source)
            checkpoint = root/'runs'/item['source_run']/'best.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'existing training checkpoint')
            before = checkpoint.read_bytes()
            result = prepare_gui.prepare(root)
            self.assertEqual(result['preserved'], 1)
            self.assertEqual(checkpoint.read_bytes(), before)
            self.assertFalse((root/'deploy').exists())

    def test_released_models_are_selectable_without_fabricated_history(self):
        rows = config_io.list_runs()
        manifest = json.loads((ROOT/'pretrained/manifest.json').read_text())['models']
        expected = {r['source_run'] for r in manifest}
        released = {r['name']: r for r in rows if r['name'] in expected}
        self.assertEqual(set(released), expected)
        for name, row in released.items():
            self.assertEqual(row['status'], 'pretrained')
            self.assertFalse(row['qualified'])
            self.assertIsNone(row['epochs'])
            self.assertFalse((ROOT/'runs'/name/'history.json').exists())
        self.assertEqual({r['name'] for r in eegio_tab.available_runs()}, expected)
        self.assertTrue(any('tcn_mla_r16_' in name for name in expected))
        self.assertEqual(config_io.runs_by_kind(rows), [])

    def test_prepared_step_graphs_match_gui_channel_settings(self):
        items = json.loads((ROOT/'pretrained/manifest.json').read_text())['models']
        for item in items:
            if not item.get('onnx_path'):
                continue
            settings = channels.model_settings(item['source_run'])
            self.assertTrue(settings['step_exported'])
            self.assertFalse(settings['step_stale'])
            self.assertEqual(settings['sample_rate_hz'], 250)
            self.assertEqual(settings['lookahead'], item['lookahead_samples'])
            self.assertEqual(settings['experimental_bounded'], item['lookahead_samples'] == 8)

    def test_registration_is_repeatable(self):
        result = prepare_gui.prepare(ROOT)
        self.assertEqual(result, dict(registered=0, existing=63, preserved=0, step_graphs=0))

    def test_gui_metadata_loads_with_no_datasets(self):
        result = server.meta()
        self.assertTrue(result['configs'])
        self.assertEqual(len(result['runs']), 63)
        self.assertFalse(eegio_tab.dataset_status()['present'])
        self.assertEqual(len(pi.list_models()), 63)


if __name__ == '__main__':
    unittest.main()
