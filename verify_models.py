"""Verify released weights, exact architectures, streaming parity and future bounds."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from inference import ROOT, model_catalog, load_pretrained, predict_stream, predict_onnx, predict_window


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(67)
    signal = rng.normal(size=500).astype(np.float32)*20
    entries = model_catalog()
    for entry in entries:
        path = ROOT/entry['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
        model, _ = load_pretrained(entry['id'])
        assert sum(p.numel() for p in model.parameters()) == entry['parameters']
        with torch.inference_mode():
            output = model(torch.from_numpy(signal/model.run_config['normalization_scale'])[None,None])
        assert output.shape == (1,1,500) and torch.isfinite(output).all()
    print(f'Loaded all {len(entries)} checkpoints with strict state matching.',flush=True)
    parity=[]
    for name in ['tcn_s72','se_l0_s72','se_l8_s67','attention_s72']:
        model,entry=load_pretrained(name)
        scale=model.run_config['normalization_scale']
        with torch.inference_mode():reference=model(torch.from_numpy(signal/scale)[None,None]).numpy()[0,0]*scale
        streamed=predict_stream(model,signal,experimental_l8=True)
        onnx=predict_onnx(name,signal,experimental_l8=True)
        delay=entry['lookahead_samples']
        stop=len(signal)-delay if delay else len(signal)
        # Tolerances are expressed in normalized model-input units.
        torch_error=float(np.max(np.abs(streamed[:stop]-reference[:stop]))/scale)
        onnx_error=float(np.max(np.abs(onnx[:stop]-reference[:stop]))/scale)
        assert torch_error<1e-4 and onnx_error<1e-4,(name,torch_error,onnx_error)
        assert np.isnan(streamed[stop:]).all() and np.isnan(onnx[stop:]).all()
        window=predict_window(model,signal)
        np.testing.assert_allclose(window[125:314],reference[125:314],rtol=1e-5,atol=1e-4)
        assert np.isnan(window[:125]).all() and np.isnan(window[314:]).all()
        parity.append(dict(model=name,pytorch_error=torch_error,onnx_error=onnx_error))
    model,_=load_pretrained('se_l8_s67')
    changed=signal.copy();changed[250:]+=100
    a=predict_stream(model,signal,experimental_l8=True)
    b=predict_stream(model,changed,experimental_l8=True)
    np.testing.assert_array_equal(a[:242],b[:242])
    try:predict_stream(load_pretrained('se_offline_s67')[0],signal)
    except ValueError:pass
    else:raise AssertionError('Offline reference accepted streaming')
    print(json.dumps(dict(checkpoints=len(entries),streaming_graphs=4,parity=parity,future_bound='passed',offline_stream_guard='passed'),indent=2))


if __name__=='__main__':main()
