"""EEG input-channel experiments: planning, saved results and calibration."""

import io
import json
import math
from pathlib import Path
import statistics
import sys

from gui import config_io, eegio_tab, pi
from deploy.channel_sweep import ALLOWED_CHANNELS, channel_counts, eegio_input, NAME_RE, SSH_RE

PROJECT_ROOT=config_io.PROJECT_ROOT
RESULTS_DIR=PROJECT_ROOT/"deploy/artifacts/channel_sweeps"
INPUT_DIR=PROJECT_ROOT/"deploy/artifacts/channel_inputs"


def model_settings(name):
    if not NAME_RE.fullmatch(name or ""):
        raise ValueError("Choose a model")
    step=pi.ARTIFACTS_DIR/f"{name}_step.onnx"
    side=step.with_suffix('.json')
    info=config_io._checkpoint_info(pi.RUNS_DIR/name/"best.pt")
    saved=json.loads(side.read_text()) if side.is_file() else {}
    cfg=info.get("run_config") or saved.get("data_protocol") or {}
    prep=saved.get("preprocessing") or {}
    checkpoint=pi.RUNS_DIR/name/"best.pt"
    stale=step.is_file() and checkpoint.is_file() and step.stat().st_mtime+1<checkpoint.stat().st_mtime
    return dict(name=name, sample_rate_hz=cfg.get("target_fs") or prep.get("sample_rate_hz") or 256.,
                tcn_norm=(info.get("model_config") or {}).get("tcn_norm") or (saved.get("model") or {}).get("tcn_norm"),
                normalization=cfg.get("normalization") or prep.get("normalization") or "epoch",
                normalization_scale=cfg.get("normalization_scale") or prep.get("normalization_scale"),
                protocol=cfg.get("protocol","legacy"), step_exported=step.is_file() and side.is_file(), step_stale=stale,
                config=cfg)


def _number(spec,key,default,lo,hi,integer=False):
    value=spec.get(key)
    if value in (None,""): value=default
    if isinstance(value,bool): raise ValueError(f"{key} must be a number")
    try: value=float(value)
    except (ValueError,TypeError): raise ValueError(f"Invalid {key}") from None
    if not math.isfinite(value) or not lo<=value<=hi or integer and value!=int(value):
        raise ValueError(f"{key} must be {'an integer ' if integer else ''}between {lo} and {hi}")
    return int(value) if integer else value


def build_steps(spec):
    name=pi._model_name(spec)
    models={m['name']:m for m in pi.list_models()}
    if name not in models or not models[name].get('exportable',True):
        raise ValueError("Choose a causal model; non-causal architectural lookahead cannot be streamed")
    settings=model_settings(name)
    target=spec.get('target','host')
    if target not in ('host','pi'): raise ValueError("Choose this computer or Raspberry Pi")
    counts=channel_counts(spec.get('channels',[1,2,8,16]))
    repeats=_number(spec,'repeats',3,1,10,True)
    seconds=_number(spec,'seconds',20,1,600)
    hop = _number(spec, "hop", 4, 1, pi.MAX_HOP, True)
    threads=_number(spec,'threads',2,1,16,True)
    fs=_number(spec,'input_fs',settings['sample_rate_hz'],1,100000)
    if fs!=settings['sample_rate_hz']: raise ValueError("Input rate must match the model's training rate")
    cpus=pi._cpu_list(spec,threads)
    if target=='pi' and not cpus: raise ValueError("Choose fixed CPU cores for the Pi measurement")
    sigma=None
    if settings['normalization']!='fixed':
        if spec.get('sigma') in (None,''):
            raise ValueError("Legacy model needs a fixed prior scale; use Calibrate from training data")
        sigma=_number(spec,'sigma',None,1e-12,1e12)
    elif spec.get('sigma') not in (None,''):
        supplied=_number(spec,'sigma',None,1e-12,1e12)
        if not math.isclose(supplied,settings['normalization_scale'],rel_tol=1e-7):
            raise ValueError("Calibration differs from the model's saved fixed scale")
    source_kind = spec.get("source", "testdata" if spec.get("testdata") else "eegio")
    source_note = ""
    import numpy as np
    if source_kind=='eegio':
        data_dir = str(eegio_tab.find_data_dir())
        values, recording = eegio_input(data_dir, spec.get("eegio_subject"),
                                        spec.get("eegio_electrode") or "Fp1", fs)
        samples = len(values)
        source_args = ['--eegio_dir', data_dir, '--eegio_subject', recording["subject"],
                       '--eegio_electrode', recording["electrode"]]
        source_note = f' EEG-IO {recording["subject"]} · {recording["electrode"]}.'
        if recording["resampling"]:
            source_note += f' Converted from {recording["native_fs"]:g} Hz before timed replay.'
    elif source_kind=='testdata':
        testdata=pi._testdata_name(spec)
        source=pi.TESTDATA_DIR/testdata
        with np.load(source,allow_pickle=False) as data:
            if 'fs' not in data or float(data['fs'])!=fs:
                raise ValueError("Test-set sampling rate differs from the model")
            samples=data['noisy'].size
        source_args=['--testdata',str(source)]
    elif source_kind=='recording':
        source=Path(str(spec.get('input') or '')).expanduser().resolve()
        if not source.is_file() or source.suffix.lower()!='.npy': raise ValueError("Choose a 1-D NPY recording")
        array=np.load(source,allow_pickle=False,mmap_mode='r')
        if array.ndim!=1 or array.dtype.kind not in 'fiu': raise ValueError("Recording must contain one real-valued EEG channel")
        samples=array.size
        source_args=['--input',str(source)]
    else: raise ValueError("Choose EEG-IO, a recording, or a test-set timing fixture")
    n = int(seconds * fs) // hop * hop
    if samples<n: raise ValueError(f"Input has {samples/fs:.1f} seconds; requested {seconds:g}")
    cfg=settings['config']
    first_steady_sample = (cfg.get("valid_history_samples") or 512) - 1 + cfg.get("output_delay", 0)
    if n - hop < first_steady_sample:
        raise ValueError("Use a longer replay to include complete steady-state hops")
    stage = "raw" if source_kind == "eegio" else spec.get(
        'input_stage', 'preprocessed' if source_kind == 'testdata' else 'raw')
    if stage not in ('raw','preprocessed'): raise ValueError("Unknown input stage")
    weights=spec.get('weights','final')
    if weights not in ('random','final'): raise ValueError("Unknown weights label")
    steps=[]
    if spec.get('export'):
        step=pi._export_step(dict(model=name))
        step.pop('name',None);step['command']+=['--incremental']
        step['label']='Export and verify the stateful model'
        steps.append(step)
    elif not settings['step_exported'] or settings['step_stale']:
        raise ValueError("This model needs a stateful export; enable Export first")
    command=[sys.executable,'-m','deploy.channel_sweep','--model',str(pi.ARTIFACTS_DIR/f'{name}_step.onnx'),
             *source_args,'--input_fs',str(fs),'--input_stage',stage,'--channels',','.join(map(str,counts)),
             '--seconds',str(seconds),'--hop',str(hop),'--repeats',str(repeats),'--threads',str(threads),
             '--target',target,'--weights',weights,'--out_dir',str(RESULTS_DIR)]
    if cpus: command+=['--cpus',cpus]
    if sigma is not None: command+=['--sigma',str(sigma)]
    if settings['protocol']!='paper' or hop != 4: command+=['--exploratory']
    if target=='pi':
        ssh=str(spec.get('ssh') or '').strip(); remote=str(spec.get('remote_dir') or 'eeg').strip()
        if not SSH_RE.fullmatch(ssh) or ssh.count('@')>1: raise ValueError("Enter the Pi's SSH address")
        if not NAME_RE.fullmatch(remote): raise ValueError("Remote directory must be one directory name")
        command+=['--ssh',ssh,'--remote_dir',remote]
    steps.append(dict(label=f"Measure {', '.join(map(str,counts))} EEG channels × {repeats} repeats",
                      command=command,env={'OMP_NUM_THREADS':str(threads),'OPENBLAS_NUM_THREADS':str(threads)},
                      note=f"H={hop} · {1000*hop/fs:g} ms deadline · {n/fs*len(counts)*repeats:g} s paced replay, plus setup. "
                           +('Host diagnostic.' if target=='host' else 'Pi 5; locked governor and CPU affinity required.')
                           + source_note))
    return steps


def calibrate(name):
    from data.Dataloader import EEGDenoiseDataset
    from data.protocol import evaluation_data
    from types import SimpleNamespace
    cfg=model_settings(name)['config']
    options=evaluation_data(SimpleNamespace(run_config=cfg,checkpoint_loaded=True))
    options.update(normalization='fixed',normalization_scale=None)
    directory=Path(cfg.get('data_dir') or PROJECT_ROOT/'data/EEGdenoisenet/data')
    if not directory.is_absolute(): directory=PROJECT_ROOT/directory
    data=EEGDenoiseDataset(str(directory),split='train',**options)
    return dict(sigma=data.normalization_scale,source=f"Training EEG pool only · seed {options['master_seed']}")


def save_input(payload):
    import numpy as np
    import hashlib
    if len(payload)>64*1024*1024: raise ValueError("NPY upload limit is 64 MiB")
    array=np.load(io.BytesIO(payload),allow_pickle=False)
    if not isinstance(array,np.ndarray) or array.ndim!=1 or array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
        raise ValueError("Upload a finite, real-valued, single-channel NPY recording")
    INPUT_DIR.mkdir(parents=True,exist_ok=True)
    path=INPUT_DIR/f"input_{hashlib.sha256(payload).hexdigest()[:16]}.npy"
    if not path.exists():
        from experiment.storage import atomic_write
        atomic_write(path,lambda stream:stream.write(payload))
    return dict(path=str(path),samples=array.size)


def list_results():
    rows=[]
    for path in sorted(RESULTS_DIR.glob('*/experiment.json'),reverse=True):
        try:
            d=json.loads(path.read_text())
            rows.append({k:d.get(k) for k in ('name','status','model','target','counts','repeats','seconds','sample_rate_hz','hop_samples')}
                        | {'completed':len(d.get('results',[]))})
        except (OSError,ValueError): pass
    return rows


def read_result(name):
    if not NAME_RE.fullmatch(name or ''): raise ValueError("Invalid experiment name")
    root=(RESULTS_DIR/name).resolve()
    if root.parent!=RESULTS_DIR.resolve(): raise ValueError("Invalid experiment path")
    data=json.loads((root/'experiment.json').read_text())
    model = pi._sidecar(root / "model_step.onnx").get("model") or {}
    data.setdefault("tcn_norm", model.get("tcn_norm"))
    groups=[]
    for count in data['counts']:
        rows=[r for r in data.get('results',[]) if r['channels']==count]
        metrics={}
        for key in ('processing_p50_ms','processing_p99_ms','end_to_end_p99_ms','duty_cycle_percent','process_peak_ram_mib','deadline_miss_rate'):
            values=[r[key] for r in rows if r.get(key) is not None]
            metrics[key]=dict(mean=statistics.mean(values),sd=statistics.stdev(values) if len(values)>1 else None) if values else None
        powers=[r.get('power') or {} for r in rows]
        values=[p['incremental_power_mw']/count for p in powers if 'incremental_power_mw' in p]
        metrics['incremental_power_per_channel_mw']=dict(mean=statistics.mean(values),sd=statistics.stdev(values) if len(values)>1 else None) if values else None
        groups.append(dict(channels=count,completed=len(rows),expected=data['repeats'],metrics=metrics,
                           misses=sum(r.get('deadline_misses',0) for r in rows)))
    return {**data,'groups':groups}


def attach_power(spec):
    from deploy.paper_bench import power_summary
    from experiment.storage import atomic_json
    data=read_result(spec.get('name'))
    item=next((r for r in data['results'] if r['case']==spec.get('case')),None)
    if item is None: raise ValueError("Choose a completed channel/repeat case")
    if data['status']=='running': raise ValueError("Wait until the experiment stops before attaching power")
    setup=str(spec.get('meter') or '').strip()
    if not setup: raise ValueError("Describe the external meter, supply and synchronization")
    active,idle=Path(spec.get('active') or '').expanduser(),Path(spec.get('idle') or '').expanduser()
    if not active.is_file() or not idle.is_file(): raise ValueError("Choose existing active and idle power CSV files")
    power=power_summary(active,idle,item['observation_s'])
    item['power']={**power,'meter_setup':setup}
    item['measurement_record_complete']=bool(item.get('measured_on_device') and not item.get('exploratory') and
                                           (item.get('environment_after') or {}).get('verdict')=='locked')
    root=RESULTS_DIR/data['name']
    case={k:v for k,v in item.items() if k not in ('case','repeat')}
    atomic_json(root/f"{item['case']}.json",case)
    data.pop('groups',None);atomic_json(root/'experiment.json',data)
    return read_result(data['name'])


def meta():
    import numpy as np
    models=[]
    for model in pi.list_models():
        name = model["name"]
        if not NAME_RE.fullmatch(name) or name.endswith("_step") or not model.get("exportable", True):
            continue
        settings=model_settings(name);settings.pop('config',None)
        models.append({**model,**settings})
    testdata=pi.list_testdata()
    for row in testdata:
        try:
            with np.load(pi.TESTDATA_DIR/row['name'],allow_pickle=False) as data:
                row['fs']=float(data['fs']) if 'fs' in data else None
        except (OSError,ValueError,KeyError):
            row['fs']=None
    return dict(models=models,testdata=testdata,eegio=eegio_tab.dataset_status(),results=list_results(),
                defaults=dict(channels=[1,2,8,16],seconds=20,repeats=3,threads=2,cpus='2-3',remote_dir='eeg'),
                host_affinity=hasattr(__import__('os'),'sched_setaffinity'))
