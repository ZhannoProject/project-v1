"""Portable package relocation, native architecture and cross-platform task contracts."""

import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile

from gui import make_windows
from gui.server import Job
from deploy import memory
from deploy.channel_sweep import run_cancellable


ROOT=Path(__file__).resolve().parents[1]


class WindowsPackageTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name).resolve()

    def test_runtime_paths_and_launchers_are_relocatable(self):
        (self.root/'runtime').mkdir()
        make_windows.write_launch_files(self.root,'3.13.9','10.0.14')
        pth=(self.root/'runtime/python313._pth').read_text()
        self.assertIn('../app',pth)
        self.assertIn('site-packages',pth)
        self.assertNotIn(str(ROOT),pth)
        for name in ('Start GUI.cmd','Start in browser.cmd','Check installation.cmd'):
            command=(self.root/name).read_text()
            self.assertIn('"%~dp0runtime',command)
            self.assertIn('-X utf8',command)
        config=json.loads((self.root/'runtime/gui.runtimeconfig.json').read_text())
        self.assertEqual(config['runtimeOptions']['framework']['name'],'Microsoft.WindowsDesktop.App')

    def test_pe_header_distinguishes_arm64_from_x64(self):
        path=self.root/'app.exe'
        header=bytearray(256);header[:2]=b'MZ';struct.pack_into('<I',header,0x3c,64)
        header[64:68]=b'PE\0\0';struct.pack_into('<H',header,68,0xaa64)
        struct.pack_into('<H',header,64+92,2)
        path.write_bytes(header)
        self.assertEqual(make_windows.pe_machine(path),0xaa64)
        self.assertEqual(make_windows.pe_subsystem(path),2)
        struct.pack_into('<H',header,68,0x8664);path.write_bytes(header)
        self.assertEqual(make_windows.pe_machine(path),0x8664)
        path.write_bytes(b'not a PE executable')
        with self.assertRaises(ValueError):make_windows.pe_machine(path)

    def test_runtime_archive_cannot_escape_package_directory(self):
        archive=self.root/'bad.zip'
        for name in ('../outside','C:/outside','/outside','..\\outside'):
            with self.subTest(name=name):
                with zipfile.ZipFile(archive,'w') as stream:stream.writestr(name,'bad')
                with self.assertRaises(ValueError):make_windows.extract_zip(archive,self.root/'target')

    def test_model_and_data_copy_is_explicit_and_skips_active_runs(self):
        source=self.root/'source';source.mkdir()
        for name in make_windows.SOURCE_DIRS:
            (source/name).mkdir();(source/name/'__init__.py').write_text('')
        (source/'packaging/windows').mkdir(parents=True)
        (source/'data/EEGdenoisenet/data').mkdir(parents=True)
        (source/'data/EEGdenoisenet/data/EEG.npy').write_bytes(b'fixture')
        (source/'runs').mkdir()
        for name in ('complete','busy'):
            directory=source/'runs'/name;directory.mkdir();(directory/'best.pt').write_bytes(b'fixture')
        minimal=self.root/'minimal'
        result=make_windows.copy_sources(source,minimal)
        self.assertFalse(result['runs'])
        self.assertFalse((minimal/'data/EEGdenoisenet').exists())
        from experiment.storage import RunLease
        with RunLease(source/'runs/busy'):
            result=make_windows.copy_sources(source,self.root/'full',True,True)
        self.assertEqual(result['runs'],['complete'])
        self.assertEqual(result['active_runs_skipped'],['busy'])
        self.assertTrue((self.root/'full/data/EEGdenoisenet/data/EEG.npy').is_file())
        self.assertFalse((self.root/'full/runs/complete/.run.lock').exists())

    def test_portable_start_uses_bundle_runtime_and_writable_cache(self):
        from gui.windows_start import configure_runtime
        (self.root/'app').mkdir()
        previous=Path.cwd()
        try:
            with patch.dict(os.environ),patch.object(sys,'executable',sys.executable):
                configure_runtime(self.root)
                self.assertEqual(Path.cwd(),(self.root/'app').resolve())
                self.assertEqual(Path(sys.executable),self.root/'runtime/python.exe')
                self.assertEqual(os.environ['DOTNET_ROOT'],str(self.root/'runtime/dotnet'))
                self.assertTrue(os.environ['MPLCONFIGDIR'].startswith(str(self.root)))
        finally:os.chdir(previous)

    def test_missing_project_dependency_is_detected_before_packaging(self):
        source=self.root/'source';source.mkdir()
        (source/'metrics.py').write_text('')
        app=self.root/'app';(app/'trainer').mkdir(parents=True)
        (app/'trainer/training.py').write_text('import metrics\n')
        with self.assertRaisesRegex(ValueError,'metrics'):
            make_windows.validate_project_imports(app,source)
        (app/'metrics.py').write_text('')
        self.assertEqual(make_windows.validate_project_imports(app,source),['metrics'])

    def test_staged_runtime_imports_without_the_checkout_on_search_path(self):
        app=self.root/'app'
        make_windows.copy_sources(ROOT,app)
        self.assertTrue((app/'metrics.py').is_file())
        make_windows.validate_project_imports(app)
        script="""import importlib,pathlib,sys
app=pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0,str(app))
for name in ('metrics','gui.server','trainer.__main__','eval','eegio_eval','deploy.export_onnx','deploy.quantize_int8'):
    module=importlib.import_module(name)
    assert pathlib.Path(module.__file__).resolve().is_relative_to(app),(name,module.__file__)
print('Isolated portable imports passed')
"""
        result=subprocess.run([sys.executable,'-I','-B','-X','utf8','-c',script,str(app)],
                              cwd=self.root,capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_missing_module_message_does_not_prescribe_a_cpp_runtime(self):
        from gui.windows_start import startup_error_message
        error=ModuleNotFoundError("No module named 'metrics'",name='metrics')
        message=startup_error_message(error,'trace')
        self.assertIn('metrics',message)
        self.assertNotIn('VC_redist',message)

    def test_coreclr_adapter_preserves_the_dependency_cache_and_is_repeatable(self):
        path=self.root/'runtime/site-packages/webview/platforms/winforms.py'
        path.parent.mkdir(parents=True)
        original='before=1\nclass OpenFolderDialog:\n    private_api=True\nafter=2\n'
        path.write_text(original)
        cached=self.root/'cached-winforms.py';os.link(path,cached)
        make_windows.patch_pywebview_coreclr(self.root)
        modified=path.read_text()
        self.assertIn('from gui.windows_dialog import OpenFolderDialog',modified)
        self.assertIn('after=2',modified)
        self.assertEqual(cached.read_text(),original)
        make_windows.patch_pywebview_coreclr(self.root)
        self.assertEqual(path.read_text(),modified)

    def test_public_folder_picker_keeps_selection_and_cancel_semantics(self):
        from gui.windows_dialog import OpenFolderDialog
        class Dialog:
            SelectedPath='';SelectedPaths=['one','two'];answer='OK';disposed=False
            def ShowDialog(self,*args):return self.answer
            def Dispose(self):self.disposed=True
        dialog=Dialog()
        forms=SimpleNamespace(FolderBrowserDialog=lambda:dialog,DialogResult=SimpleNamespace(OK='OK'))
        with patch.dict(sys.modules,{'System.Windows.Forms':forms}):
            self.assertEqual(OpenFolderDialog.show(initialDirectory='chosen'),('chosen',))
            self.assertTrue(dialog.disposed)
            self.assertEqual(OpenFolderDialog.show(allow_multiple=True),('one','two'))
            dialog.answer='Cancel'
            self.assertIsNone(OpenFolderDialog.show())

    def test_archive_omits_runtime_logs_and_bytecode_without_removing_local_records(self):
        package=self.root/'portable';(package/'app/__pycache__').mkdir(parents=True)
        (package/'logs').mkdir();(package/'cache').mkdir()
        (package/'app/metrics.py').write_text('')
        (package/'app/__pycache__/metrics.pyc').write_bytes(b'cache')
        (package/'logs/launcher.log').write_text('old error')
        (package/'cache/font.json').write_text('{}')
        archive=make_windows.archive_bundle(package)
        with zipfile.ZipFile(archive) as zipped:
            names=zipped.namelist()
            self.assertIn('portable/app/metrics.py',names)
            self.assertFalse(any('__pycache__' in name or name.endswith('launcher.log') or name.endswith('font.json') for name in names))
        self.assertTrue((package/'logs/launcher.log').is_file())

    def test_windows_ram_units_are_bytes_and_do_not_import_resource(self):
        with patch.object(memory.sys,'platform','win32'),patch.object(memory,'_windows_peak_bytes',return_value=48*2**20):
            self.assertEqual(memory.peak_ram_mib(),48.)

    def test_missing_program_marks_a_job_failed(self):
        job=Job()
        job.start([str(self.root/'missing-program')],ROOT)
        until=time.monotonic()+5
        while job.running and time.monotonic()<until:time.sleep(.01)
        self.assertFalse(job.running)
        self.assertEqual(job.returncode,1)
        self.assertTrue(any('FileNotFoundError' in line for line in job.status(0)['lines']))

    def test_windows_deploy_resolves_git_bash_and_explains_missing_dependency(self):
        from gui import pi
        git=self.root/'Git/bin/bash.exe';git.parent.mkdir(parents=True);git.write_bytes(b'fixture')
        with patch.object(pi.sys,'platform','win32'),patch.dict(os.environ,ProgramW6432=str(self.root)):
            self.assertEqual(pi._bash_executable(),str(git))
        with patch.object(pi.sys,'platform','win32'),patch.dict(os.environ,{},clear=True),patch.object(pi.shutil,'which',return_value=None):
            with self.assertRaisesRegex(ValueError,'Git Bash'):pi._bash_executable()

    def test_stop_file_interrupts_wait_and_reaps_the_local_child(self):
        stop=self.root/'stop';pidfile=self.root/'pid'
        timer=threading.Timer(.4,stop.touch);timer.start()
        try:
            with patch.dict(os.environ,EEG_GUI_STOP_FILE=str(stop)):
                with self.assertRaises(KeyboardInterrupt):
                    run_cancellable([sys.executable,'-c',
                        'import os,sys,time;open(sys.argv[1],"w").write(str(os.getpid()));time.sleep(60)',str(pidfile)])
            if os.name=='posix' and pidfile.exists():
                with self.assertRaises(ProcessLookupError):os.kill(int(pidfile.read_text()),0)
        finally:timer.cancel()


if __name__=='__main__':unittest.main()
