"""Channel-load GUI plans and measurements retain their physical meaning."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

import numpy as np

from gui import channels, pi
from deploy.channel_sweep import channel_counts, case_order, Remote


class ChannelPlanTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        self.artifacts=self.root/'artifacts';self.artifacts.mkdir()
        self.runs=self.root/'runs';(self.runs/'demo').mkdir(parents=True)
        (self.runs/'demo/best.pt').write_bytes(b'fixture')
        self.data=self.root/'testdata';self.data.mkdir()
        np.savez(self.data/'demo.npz',noisy=np.ones((10,500),dtype=np.float32),fs=250)
        self.cfg=dict(protocol='paper',normalization='fixed',normalization_scale=2.,target_fs=250,
                      master_seed=42,artifact='BOTH',valid_history_samples=220,output_delay=0)
        (self.artifacts/'demo_step.onnx').write_bytes(b'fixture')
        (self.artifacts/'demo_step.json').write_text(json.dumps(dict(data_protocol=self.cfg)))
        for owner,name,value in ((pi,'ARTIFACTS_DIR',self.artifacts),(pi,'RUNS_DIR',self.runs),
                                 (pi,'TESTDATA_DIR',self.data),(channels,'RESULTS_DIR',self.root/'results'),
                                 (channels,'INPUT_DIR',self.root/'inputs')):
            self.enterContext(patch.object(owner,name,value))
        self.enterContext(patch.object(pi,'list_models',return_value=[dict(name='demo',exportable=True)]))
        self.enterContext(patch.object(channels.config_io,'_checkpoint_info',side_effect=lambda _:dict(run_config=self.cfg)))
        self.spec=dict(model='demo',target='host',source='testdata',testdata='demo.npz',input_fs=250,
                       seconds=2,repeats=3,threads=2,channels=[1,2,8,16],input_stage='preprocessed')

    def test_one_fixed_model_and_hop_contract_across_all_channel_cases(self):
        steps=channels.build_steps(self.spec)
        self.assertEqual(len(steps),1)
        command=steps[0]['command']
        self.assertEqual(command[command.index('--channels')+1],'1,2,8,16')
        self.assertEqual(command[command.index('--target')+1],'host')
        self.assertEqual(command[command.index('--hop')+1],'4')
        self.assertNotIn('--exploratory',command)
        self.assertEqual(len(case_order([1,2,8,16],3)),12)
        self.assertEqual(case_order([1,2,8,16],2)[4:8],[(2,2),(8,2),(16,2),(1,2)])

    def test_selected_hop_is_forwarded_and_requires_a_complete_steady_state_hop(self):
        command = channels.build_steps({**self.spec, "hop": 7})[0]["command"]
        self.assertEqual(command[command.index("--hop") + 1], "7")
        self.assertIn("--exploratory", command)
        with self.assertRaisesRegex(ValueError, "complete steady-state hops"):
            channels.build_steps({**self.spec, "hop": 256})

    def test_invalid_channel_sets_and_rate_mismatch_are_rejected(self):
        for value in ([],[0],[4],[1,1],[True],['1;echo bad']):
            with self.subTest(value=value),self.assertRaises(ValueError): channel_counts(value)
        with self.assertRaisesRegex(ValueError,'rate'):
            channels.build_steps({**self.spec,'input_fs':256})
        with self.assertRaisesRegex(ValueError,'Input has'):
            channels.build_steps({**self.spec,'seconds':30})

    def test_default_eegio_plan_uses_raw_continuous_samples_at_the_model_rate(self):
        from data.EEGIO import write_synthetic_recording
        from deploy.channel_sweep import eegio_input

        recordings = self.root / "eegio"
        rec = write_synthetic_recording(recordings, "S03", duration_s=6, corrupt=[(0, 1)])
        spec = {k: v for k, v in self.spec.items() if k not in ("source", "testdata", "input_fs")}
        with patch.object(channels.eegio_tab, "find_data_dir", return_value=recordings):
            command = channels.build_steps(spec)[0]["command"]
        self.assertEqual(command[command.index("--eegio_subject") + 1], "S03")
        self.assertEqual(command[command.index("--eegio_electrode") + 1], "Fp1")
        self.assertEqual(command[command.index("--input_stage") + 1], "raw")
        self.assertEqual(float(command[command.index("--input_fs") + 1]), 250)
        native, metadata = eegio_input(str(recordings), "S03", "Fp2", 250)
        self.assertEqual(metadata["start_sample"], 251)
        np.testing.assert_array_equal(native, rec.signal[1, 251:])
        converted, metadata = eegio_input(str(recordings), "S03", "Fp2", 256)
        self.assertEqual(len(converted), (len(native) * 128 + 124) // 125)
        self.assertEqual(metadata["resampling"]["method"], "causal_fir")

    def test_pi_plan_requires_an_address_and_fixed_cpu_affinity(self):
        with self.assertRaisesRegex(ValueError,'CPU'):
            channels.build_steps({**self.spec,'target':'pi'})
        spec={**self.spec,'target':'pi','cpus':'2-3','ssh':'user@raspberrypi.local'}
        command=channels.build_steps(spec)[0]['command']
        self.assertIn('--ssh',command)
        with self.assertRaises(ValueError): channels.build_steps({**spec,'ssh':'host; bad'})
        with self.assertRaises(ValueError): Remote('host','../outside','run')

    def test_legacy_normalization_needs_prior_calibration_and_is_labeled(self):
        self.cfg.update(protocol='exploratory',normalization='epoch',normalization_scale=None)
        with self.assertRaisesRegex(ValueError,'prior scale'): channels.build_steps(self.spec)
        command=channels.build_steps({**self.spec,'sigma':5.})[0]['command']
        self.assertIn('--exploratory',command)
        self.assertEqual(command[command.index('--sigma')+1],'5.0')

    def test_upload_rejects_joint_input_and_reuses_identical_recordings(self):
        def content(array):
            buf=io.BytesIO();np.save(buf,array);return buf.getvalue()
        payload=content(np.arange(100,dtype=np.float32))
        a=channels.save_input(payload);b=channels.save_input(payload)
        self.assertEqual(a,b)
        self.assertEqual(a['samples'],100)
        self.assertEqual(len(list(channels.INPUT_DIR.glob('*.npy'))),1)
        with self.assertRaises(ValueError): channels.save_input(content(np.zeros((2,100))))
        with self.assertRaises(ValueError): channels.save_input(content(np.array([np.nan])))

    def test_partial_results_do_not_fill_missing_repeats_with_zero(self):
        root=channels.RESULTS_DIR/'experiment';root.mkdir(parents=True)
        data=dict(name='experiment',status='stopped',counts=[1,8],repeats=3,results=[
            dict(case='c1_r1',channels=1,repeat=1,processing_p99_ms=10.,deadline_misses=0),
            dict(case='c1_r2',channels=1,repeat=2,processing_p99_ms=12.,deadline_misses=1)])
        (root/'experiment.json').write_text(json.dumps(data))
        result=channels.read_result('experiment')
        self.assertEqual(result['groups'][0]['completed'],2)
        self.assertAlmostEqual(result['groups'][0]['metrics']['processing_p99_ms']['mean'],11.)
        self.assertAlmostEqual(result['groups'][0]['metrics']['processing_p99_ms']['sd'],2**.5)
        self.assertIsNone(result['groups'][1]['metrics']['processing_p99_ms'])
        with self.assertRaises(ValueError): channels.read_result('../experiment')

    def test_calibration_uses_training_pool_not_replay_or_test_epochs(self):
        data=self.root/'epochs';data.mkdir()
        rng=np.random.default_rng(3)
        for name in ('EEG','EOG','EMG'):
            np.save(data/f'{name}_all_epochs.npy',rng.normal(size=(80,512)).astype(np.float32))
        self.cfg.update(normalization='epoch',normalization_scale=None,data_dir=str(data),target_fs=None)
        with contextlib.redirect_stdout(io.StringIO()): first=channels.calibrate('demo')['sigma']
        from data.Dataloader import EEGDenoiseDataset
        values=np.load(data/'EEG_all_epochs.npy')
        indices=EEGDenoiseDataset._split_indices(80,(.7,.15,.15),'test',42)
        values[indices]*=10000;np.save(data/'EEG_all_epochs.npy',values)
        with contextlib.redirect_stdout(io.StringIO()): second=channels.calibrate('demo')['sigma']
        self.assertEqual(first,second)

    def test_metadata_exposes_native_rates_for_input_selection(self):
        candidates = [dict(name="demo)", exportable=True), dict(name="demo", exportable=True)]
        with patch.object(pi, "list_models", return_value=candidates), \
             patch.object(pi,'list_testdata',return_value=[dict(name='demo.npz')]):
            meta=channels.meta()
        self.assertEqual([model["name"] for model in meta["models"]], ["demo"])
        self.assertEqual(meta['models'][0]['sample_rate_hz'],250)
        self.assertEqual(meta['testdata'][0]['fs'],250)

    def test_power_is_imported_from_external_traces_without_promoting_host_results(self):
        root=channels.RESULTS_DIR/'experiment';root.mkdir(parents=True)
        item=dict(case='c2_r1',channels=2,repeat=1,observation_s=1.,measured_on_device=False,
                  exploratory=True,environment_after=dict(verdict='unlocked'))
        data=dict(name='experiment',status='complete',counts=[2],repeats=1,results=[item])
        (root/'experiment.json').write_text(json.dumps(data))
        for name,power in [('active',2500),('idle',500)]:
            (self.root/f'{name}.csv').write_text(f'time_s,power_mw\n0,{power}\n2,{power}\n')
        result=channels.attach_power(dict(name='experiment',case='c2_r1',active=str(self.root/'active.csv'),
                                           idle=str(self.root/'idle.csv'),meter='test external meter'))
        self.assertEqual(result['groups'][0]['metrics']['incremental_power_per_channel_mw']['mean'],1000.)
        self.assertFalse(result['results'][0]['measurement_record_complete'])


class ChannelMeasurementTests(unittest.TestCase):
    def test_replay_keeps_per_channel_state_separate_under_one_hop_deadline(self):
        from deploy import paper_bench
        class Clock:
            now=0.
            def perf(self):return self.now
            def sleep(self,s):self.now+=s
        clock=Clock()
        class Model:
            sample_rate=250
            meta=dict(preprocessing=dict(normalization='fixed',normalization_scale=2.),data_protocol={})
            last_run_s=.0001
            def initial_state(self):return {'sum':0.}
            def step(self,x,state):
                state['sum']+=x;clock.now+=self.last_run_s
                return state['sum'],state
        with patch.object(paper_bench.time,'perf_counter',side_effect=clock.perf),patch.object(paper_bench.time,'sleep',side_effect=clock.sleep):
            output,rows,_=paper_bench.replay(Model(),np.arange(8,dtype=np.float32),channels=2,raw_input=False)
        np.testing.assert_array_equal(output[0],output[1])
        self.assertEqual(len(rows),2)
        self.assertAlmostEqual(rows[0]['processing_ms'],.8)
        self.assertFalse(any(r['deadline_miss'] for r in rows))

    def test_host_diagnostic_is_never_labeled_a_pi_measurement(self):
        from deploy import paper_bench
        from deploy import preflight,raspi_infer
        class Model:
            sample_rate=250;window=1;threads={}
            meta=dict(preprocessing=dict(normalization='fixed',normalization_scale=1.),data_protocol={})
            last_run_s=0.
            def initial_state(self):return {}
            def step(self,x,state):return x,state
        env=dict(board='Test host',verdict='unlocked')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);np.save(root/'input.npy',np.arange(8,dtype=np.float32))
            with patch.object(raspi_infer,'IncrementalDenoiser',return_value=Model()),patch.object(preflight,'check',return_value=env),patch.object(paper_bench.time,'sleep'),contextlib.redirect_stdout(io.StringIO()):
                paper_bench.main(['--model','test.onnx','--input',str(root/'input.npy'),'--input_fs','250',
                                  '--input_stage','preprocessed','--out',str(root/'host'),'--target','host',
                                  '--channels','2','--weights','random','--exploratory'])
            result=json.loads((root/'host.json').read_text())
            self.assertFalse(result['measured_on_device'])
            self.assertFalse(result['measurement_record_complete'])
            self.assertEqual(result['where'],'host')
            self.assertEqual(result['hop_budget_ms'],16.)


class ChannelSweepIntegrationTests(unittest.TestCase):
    def test_real_onnx_host_sweep_saves_each_measured_case(self):
        import torch
        from models.variants import build_model,resolved_arch
        from experiment.config import resolve_config
        from deploy.export_onnx import export_checkpoint
        from deploy import channel_sweep
        torch.set_num_threads(1);torch.manual_seed(2)
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            model=build_model('tcn_se',hidden_channels=8,tcn_norm='channel',se_window=16)
            cfg=resolve_config(dict(profile='paper',hidden_channels=8,normalization_scale=2.,target_fs=250,valid_history_samples=220))
            torch.save(dict(model_state=model.state_dict(),model_config=resolved_arch(model),run_config=cfg,epoch=0),root/'best.pt')
            export_checkpoint(root/'best.pt',out_dir=root,name='tiny',incremental=True)
            np.save(root/'input.npy',np.random.default_rng(0).normal(size=500).astype(np.float32))
            run=channel_sweep.subprocess.run
            with patch.object(channel_sweep.subprocess,'run',side_effect=lambda *args,**kwargs: run(*args,**kwargs,capture_output=True,text=True)):
                channel_sweep.main(['--model',str(root/'tiny_step.onnx'),'--input',str(root/'input.npy'),'--input_fs','250',
                                    '--input_stage','preprocessed','--out_dir',str(root/'results'),'--target','host',
                                    '--channels','1,2','--seconds','1.2','--hop','7','--repeats','1','--threads','1','--weights','random'])
            files=list((root/'results').glob('*/experiment.json'))
            self.assertEqual(len(files),1)
            result=json.loads(files[0].read_text())
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['hop_samples'], 7)
            self.assertEqual([r['channels'] for r in result['results']],[1,2])
            self.assertTrue(all(r['hop_samples'] == 7 and r['hop_budget_ms'] == 28
                                and r['exploratory'] for r in result['results']))
            self.assertTrue(all(r['replica_max_abs_delta']==0 for r in result['results']))
            self.assertTrue(all(not r['measured_on_device'] for r in result['results']))
            self.assertNotEqual(result['model_path'],result['model_source'])
            self.assertEqual(channel_sweep.fingerprint(root/'tiny_step.onnx'),result['model_sha256'])
            np.testing.assert_array_equal(np.load(result['input']),np.load(root/'input.npy')[:294])

    def test_remote_stop_preserves_a_stopped_manifest_and_cleans_up_the_scoped_process(self):
        from deploy import channel_sweep
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            (root/'model.onnx').write_bytes(b'fixture')
            (root/'model.json').write_text('{}')
            np.save(root/'input.npy',np.ones(500,dtype=np.float32))
            with patch.object(channel_sweep,'Remote') as remote:
                remote.return_value.run_case.side_effect=KeyboardInterrupt
                with self.assertRaises(SystemExit) as stopped:
                    channel_sweep.main(['--model',str(root/'model.onnx'),'--input',str(root/'input.npy'),
                                        '--input_fs','250','--out_dir',str(root/'results'),'--seconds','1',
                                        '--target','pi','--ssh','user@pi','--channels','1'])
                self.assertEqual(stopped.exception.code,130)
                remote.return_value.stop.assert_called_once()
            manifest=json.loads(next((root/'results').glob('*/experiment.json')).read_text())
            self.assertEqual(manifest['status'],'stopped')
            self.assertEqual(manifest['results'],[])


class ChannelApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from gui import server
        cls.srv=server.GUIServer('127.0.0.1',0)
        cls.base=f'http://127.0.0.1:{cls.srv.server_address[1]}'
        threading.Thread(target=cls.srv.serve_forever,daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown();cls.srv.server_close()

    def setUp(self):
        from gui import server
        self.server=server
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        self.enterContext(patch.object(server,'RUNS_DIR',self.root))
        self.enterContext(patch.object(channels,'INPUT_DIR',self.root/'inputs'))
        self.enterContext(patch.object(channels,'RESULTS_DIR',self.root/'results'))
        self.enterContext(patch.object(server,'CHANNEL_JOB',server.Job()))
        self.enterContext(patch.object(server,'DEPLOY_JOB',server.DeployJob()))
        self.addCleanup(server.CHANNEL_JOB.stop)
        self.addCleanup(server.DEPLOY_JOB.stop)

    def call(self,path,body=None,raw=None):
        request=urllib.request.Request(self.base+path,method='POST' if body is not None or raw is not None else 'GET',
                                       data=raw if raw is not None else json.dumps(body).encode() if body is not None else None,
                                       headers={'Content-Type':'application/octet-stream' if raw is not None else 'application/json'})
        try:
            with urllib.request.urlopen(request,timeout=10) as response: return response.status,json.loads(response.read())
        except urllib.error.HTTPError as error: return error.code,json.loads(error.read())

    def test_routes_validate_plans_and_upload_recordings(self):
        self.assertEqual(self.call('/api/channels/preview',dict(model=''))[0],400)
        self.assertEqual(self.call('/api/channels/result?name=../outside')[0],400)
        self.assertEqual(self.call('/api/channels/input',raw=b'not an NPY')[0],400)
        stream=io.BytesIO();np.save(stream,np.arange(16,dtype=np.float32))
        code,data=self.call('/api/channels/input',raw=stream.getvalue())
        self.assertEqual(code,200)
        self.assertEqual(data['samples'],16)
        self.assertTrue(Path(data['path']).is_file())

    def test_channel_job_is_separate_stoppable_and_excludes_other_deployment(self):
        slow=[dict(label='test replay',command=[sys.executable,'-u','-c',"import time;print('ready',flush=True);time.sleep(30)"])]
        with patch.object(channels,'build_steps',return_value=slow):
            self.assertEqual(self.call('/api/channels/run',{})[0],200)
            self.assertEqual(self.call('/api/channels/run',{})[0],409)
        self.assertEqual(self.call('/api/pi/run',{})[0],409)
        self.assertFalse(self.server.TRAIN_JOBS.running)
        for _ in range(100):
            status=self.call('/api/channels/status')[1]
            if 'ready' in status['lines']: break
            time.sleep(.01)
        self.assertIn('ready',status['lines'])
        self.assertTrue(self.call('/api/channels/stop',{})[1]['stopped'])
        self.assertFalse(self.call('/api/channels/status')[1]['running'])
        with patch.object(pi,'build_steps',return_value=slow):
            self.assertEqual(self.call('/api/pi/run',{})[0],200)
            self.assertEqual(self.call('/api/channels/run',{})[0],409)


if __name__=='__main__':unittest.main()
