"""Register bundled weights for the GUI without creating training histories."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_new(source, target):
    """Copy a release file without replacing an existing checkpoint."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.release-', dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copy2(source, temporary)
        # Exclusive creation also protects a training job that started meanwhile.
        with target.open('xb') as output, temporary.open('rb') as input_file:
            shutil.copyfileobj(input_file, output)
    except FileExistsError:
        return False
    finally:
        temporary.unlink(missing_ok=True)
    return True


def prepare(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root/'pretrained/manifest.json').read_text(encoding='utf-8'))
    counts = dict(registered=0, existing=0, preserved=0, step_graphs=0)
    for item in manifest['models']:
        name = item['source_run']
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
            raise ValueError(f'Invalid run identifier: {name!r}')
        source = root/item['path']
        if digest(source) != item['sha256']:
            raise ValueError(f'Checksum mismatch: {item["id"]}')
        target = root/'runs'/name/'best.pt'
        if target.exists():
            if digest(target) != item['sha256']:
                counts['preserved'] += 1
                continue
            counts['existing'] += 1
        elif _copy_new(source, target):
            counts['registered'] += 1
        else:
            counts['preserved'] += 1
            continue
        if not item.get('onnx_path'):
            continue
        graph = root/item['onnx_path']
        if digest(graph) != item['onnx_sha256']:
            raise ValueError(f'Graph checksum mismatch: {item["id"]}')
        exported = root/'deploy/artifacts'/f'{name}_step.onnx'
        sidecar = exported.with_suffix('.json')
        # Keep independently exported graphs and their metadata together.
        if exported.exists() or sidecar.exists():
            continue
        metadata = json.loads(graph.with_suffix('.json').read_text(encoding='utf-8'))
        metadata['model']['checkpoint'] = f'runs/{name}/best.pt'
        if _copy_new(graph, exported):
            with sidecar.open('x', encoding='utf-8') as output:
                json.dump(metadata, output, indent=2, allow_nan=False)
                output.write('\n')
            counts['step_graphs'] += 1
    return counts


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(prepare(), indent=2))
    print('Start the GUI with: python -m gui --browser')


if __name__ == '__main__':
    main()
