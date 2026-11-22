"""Run inside the shipped Python (ARM64 or x64) on a Windows device or CI runner."""

import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import threading
import time
import urllib.request


def check(bundle=None):
    assert sys.platform == 'win32', 'Run the device check on Windows'
    machine=platform.machine().lower()
    assert machine in ('arm64','aarch64','amd64','x86_64'), f'Unexpected Windows architecture: {machine}'
    manifest=Path(__file__).resolve().parents[2]/'manifest.json'
    if manifest.is_file():
        built=json.loads(manifest.read_text(encoding='utf-8')).get('arch','arm64')
        native='arm64' if machine in ('arm64','aarch64') else 'x64'
        assert built==native, f'This is the {built} package running under {native} Python'
    import numpy as np
    import torch
    import scipy.signal
    import onnx
    import onnxruntime as ort
    import onnxscript
    import matplotlib
    import yaml
    import metrics
    from gui import config_io, server
    from deploy.memory import peak_ram_mib
    from experiment.storage import RunLease
    from models.variants import build_model
    assert Path(metrics.__file__).resolve().parent==Path(__file__).resolve().parents[1], 'metrics must come from the portable app'
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='EEG ARM 中文 ') as directory:
        root=Path(directory)
        with RunLease(root):
            try:
                with RunLease(root): pass
            except FileExistsError: pass
            else: raise AssertionError('A second run acquired the directory')
        model=build_model('tcn_se',hidden_channels=8,tcn_norm='channel',se_window=16).eval()
        signal=torch.randn(2,1,32)
        model(signal).square().mean().backward()
        assert peak_ram_mib()>0
        # Exercise a real exported stateful graph and Windows memory collection.
        from deploy.export_onnx import export_checkpoint
        from models.variants import resolved_arch
        from experiment.config import resolve_config
        cfg=resolve_config(dict(profile='paper',hidden_channels=8,normalization_scale=2.,target_fs=250,valid_history_samples=220))
        torch.save(dict(model_state=model.state_dict(),model_config=resolved_arch(model),run_config=cfg,epoch=0),root/'best.pt')
        export_checkpoint(root/'best.pt',out_dir=root,name='smoke',incremental=True)
        np.save(root/'input.npy',np.ones(320,dtype=np.float32))
        from deploy.paper_bench import main as replay
        replay(['--model',str(root/'smoke_step.onnx'),'--input',str(root/'input.npy'),
                '--input_fs','250','--input_stage','preprocessed','--seconds','1.2',
                '--channels','2','--threads','1','--weights','random','--target','host','--out',str(root/'timing')])
        result=json.loads((root/'timing.json').read_text())
        assert result['replica_max_abs_delta']==0 and not result['measured_on_device']
        httpd=server.GUIServer('127.0.0.1',0)
        threading.Thread(target=httpd.serve_forever,daemon=True).start()
        try:
            base=f'http://127.0.0.1:{httpd.server_address[1]}'
            with urllib.request.urlopen(base+'/') as response:
                assert b'EEG channels' in response.read()
            with urllib.request.urlopen(base+'/api/meta') as response:
                assert json.load(response)['configs']
            with urllib.request.urlopen(base+'/api/channels/meta') as response:
                assert json.load(response)['defaults']['channels']==[1,2,8,16]
        finally:
            httpd.shutdown();httpd.server_close()
        # A normal Windows task and its child process must both stop.
        child_file=root/'child.pid'
        code="import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); pathlib.Path(sys.argv[1]).write_text(str(p.pid)); print('ready',flush=True); time.sleep(60)"
        job=server.Job()
        try:
            job.start([sys.executable,'-X','utf8','-u','-c',code,str(child_file)],config_io.PROJECT_ROOT)
            until=time.monotonic()+20
            while not child_file.exists() and time.monotonic()<until: time.sleep(.05)
            assert child_file.exists(), job.status(0)
            pid=int(child_file.read_text())
            assert job.stop(grace=.2) and not job.running
            import ctypes
            from ctypes import wintypes
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
            kernel.OpenProcess.restype=wintypes.HANDLE
            handle=kernel.OpenProcess(0x1000,False,pid)
            if handle:
                status=wintypes.DWORD()
                kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
                kernel.CloseHandle.argtypes=[wintypes.HANDLE]
                kernel.GetExitCodeProcess(handle,ctypes.byref(status));kernel.CloseHandle(handle)
                assert status.value!=259, 'Child process survived Stop'
        finally: job.stop(grace=.2)
    if bundle:
        from gui.windows_start import load_desktop_runtime
        load_desktop_runtime(bundle)
        from webview.platforms import winforms
        assert winforms._is_chromium(), 'Microsoft Edge WebView2 Runtime is missing'
    report=dict(status='passed',platform=platform.platform(),architecture=platform.machine(),
                python=sys.version,torch=torch.__version__,onnxruntime=ort.__version__)
    if bundle:
        directory=Path(bundle)/'logs';directory.mkdir(exist_ok=True)
        (directory/'windows-selfcheck.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    from gui.windows_start import configure_runtime
    bundle=Path(__file__).resolve().parents[2]
    configure_runtime(bundle)
    check(bundle)
