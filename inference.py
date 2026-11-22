"""Minimal CPU inference for the paper's released single-channel models."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from models.variants import build_model_from_config

ROOT = Path(__file__).resolve().parent


def model_catalog():
    return json.loads((ROOT/'pretrained/manifest.json').read_text())['models']


def resolve_model(name):
    entry = next((r for r in model_catalog() if r['id'] == str(name)), None)
    if entry is not None:
        return ROOT/entry['path'], dict(entry)
    path = Path(name).expanduser()
    if not path.is_file():
        raise ValueError(f'Unknown model {name!r}; use --list to see released IDs')
    return path, {'id': path.stem}


def load_pretrained(name):
    """Load a compact checkpoint and reconstruct its exact saved architecture."""
    path, info = resolve_model(name)
    if 'sha256' in info and hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
        raise ValueError(f'Checkpoint checksum mismatch: {path}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    architecture = checkpoint['model_config']
    config = {**checkpoint['run_config'], **architecture}
    model = build_model_from_config(config)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.model_config = architecture
    model.run_config = checkpoint['run_config']
    model.eval()
    return model, info


def _input(signal):
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Input must be one finite, nonempty EEG channel: shape (samples,)')
    return x


def _scale(config, input_level_uv):
    value = config['normalization_scale'] if input_level_uv is None else input_level_uv
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Input scale must be finite and positive')
    return float(value)


def causal_preprocess(signal, fs=250.):
    """Forward-only 1--80 Hz filtering and a 60 Hz notch, reset per recording."""
    from scipy.signal import butter, iirnotch, sosfilt, tf2sos
    x = np.asarray(_input(signal), dtype=np.float64)
    b, a = iirnotch(60., 30., fs=fs)
    for sos in [butter(4, 1., btype='highpass', fs=fs, output='sos'),
                tf2sos(b, a), butter(4, 80., btype='lowpass', fs=fs, output='sos')]:
        x = sosfilt(sos, x)
    return x.astype(np.float32)


@torch.inference_mode()
def predict_window(model, signal, *, input_level_uv=None):
    """Matched EEG-IO window path: 500 samples, retain 125/186 context margins.

    SE is reset per window. Unavailable edge targets are NaN. This is a
    functional scoring layout; it does not schedule real-time acquisition.
    """
    x = _input(signal)
    win, left, right = 500, 125, 186
    if len(x) < win:
        raise ValueError('Window inference needs at least 500 samples')
    scale = _scale(model.run_config, input_level_uv)
    stride = win-left-right
    starts = np.minimum(np.arange(0, len(x)-win+stride, stride), len(x)-win)
    windows = np.lib.stride_tricks.sliding_window_view(x, win)
    output = np.full_like(x, np.nan)
    stop = left
    for first in range(0, len(starts), 16):
        offsets = starts[first:first+16]
        batch = torch.from_numpy(windows[offsets]/scale)[:, None, :]
        predictions = model(batch).cpu().numpy()[:, 0]*scale
        for offset, prediction in zip(offsets, predictions):
            begin, stop = stop, int(offset)+win-right
            output[begin:stop] = prediction[begin-int(offset):stop-int(offset)]
    return output


def _stream_delay(model, experimental_l8):
    lookahead = model.model_config.get('lookahead') or 0
    if lookahead:
        if lookahead != 8 or not experimental_l8:
            raise ValueError('Stateful lookahead inference requires L=8 and --experimental-l8')
        return 8
    if not getattr(model, 'causal', True) or not getattr(model, 'causal_tcn', True):
        raise ValueError('The offline reference requires window inference')
    return 0


@torch.inference_mode()
def predict_stream(model, signal, *, input_level_uv=None, experimental_l8=False):
    """Stateful sample-by-sample inference, with target-aligned delayed output."""
    from models.incremental import IncrementalModel
    from models.bounded_step import BoundedStepGraph
    x = _input(signal)
    scale = _scale(model.run_config, input_level_uv)
    delay = _stream_delay(model, experimental_l8)
    output = np.full_like(x, np.nan)
    if delay:
        stream = BoundedStepGraph(model)
        state = stream.initial_state()
        for i, sample in enumerate(x):
            y, *state = stream(torch.tensor([[[sample/scale]]], dtype=torch.float32), *state)
            if i >= delay:
                output[i-delay] = float(y.reshape(()))*scale
    else:
        stream = IncrementalModel(model)
        for i, sample in enumerate(x):
            output[i] = float(stream.step(float(sample)/scale).reshape(()))*scale
    return output


def predict_onnx(name, signal, *, input_level_uv=None, experimental_l8=False, threads=1):
    """Run a released stateful ONNX graph; reset all state for this recording."""
    import onnxruntime as ort
    _, entry = resolve_model(name)
    if 'onnx_path' not in entry:
        raise ValueError('No bundled ONNX graph for this ID; see --list')
    path = ROOT/entry['onnx_path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['onnx_sha256']:
        raise ValueError('ONNX checksum mismatch')
    meta = json.loads(path.with_suffix('.json').read_text())
    delay = int(meta['preprocessing'].get('output_delay_samples') or 0)
    if delay and (delay != 8 or not experimental_l8):
        raise ValueError('The bounded graph requires --experimental-l8')
    scale = _scale(meta['data_protocol'], input_level_uv)
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    inputs = session.get_inputs()
    mask_fill = meta['export'].get('mask_fill', -1e9)
    state = {v.name: np.full([d if isinstance(d, int) else 1 for d in v.shape],
                 mask_fill if v.name.startswith('attn_mask') else 0., dtype=np.float32) for v in inputs[1:]}
    x = _input(signal)
    output = np.full_like(x, np.nan)
    for i, sample in enumerate(x):
        values = session.run(None, {inputs[0].name: np.array([[[sample/scale]]], dtype=np.float32), **state})
        state = dict(zip(state, values[1:]))
        if i >= delay:
            output[i-delay] = float(values[0].reshape(()))*scale
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--model', default='se_l8_s67')
    parser.add_argument('--mode', choices=['window', 'stream', 'onnx'], default='window')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--raw', action='store_true', help='Apply the paper causal filters to raw microvolt input')
    parser.add_argument('--input-level-uv', type=float, help='Preset expected background level, e.g. 20 for EEG-IO')
    parser.add_argument('--experimental-l8', action='store_true')
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.list:
        for r in model_catalog():
            print(f"{r['id']:20} L={str(r['lookahead_samples']):4} params={r['parameters']:7} ONNX={'yes' if 'onnx_path' in r else 'no'}")
        return
    if args.input is None or args.output is None:
        parser.error('--input and --output are required for inference')
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    signal = _input(np.load(args.input, allow_pickle=False))
    if args.raw:
        signal = causal_preprocess(signal)
    if args.mode == 'onnx':
        result = predict_onnx(args.model, signal, input_level_uv=args.input_level_uv,
                              experimental_l8=args.experimental_l8, threads=args.threads)
    else:
        model, _ = load_pretrained(args.model)
        result = (predict_window(model, signal, input_level_uv=args.input_level_uv) if args.mode == 'window'
                  else predict_stream(model, signal, input_level_uv=args.input_level_uv,
                                      experimental_l8=args.experimental_l8))
    np.save(args.output, result)
    print(f'Saved {len(result)} target-aligned samples; {np.isfinite(result).sum()} available targets to {args.output}')


if __name__ == '__main__':
    main()
