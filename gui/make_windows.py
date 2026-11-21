"""Build a relocatable Windows app (ARM64 or x64), including its own Python and .NET.

python -m gui.make_windows --arch x64 --include-runs --include-data
python -m gui.make_windows --arch both --include-runs --include-data
No Windows binaries are executed when this builder runs on macOS or Linux.
"""

import argparse
import ast
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import stat
import struct
import subprocess
import tempfile
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
#: Everything that differs between the two Windows targets. The application,
#: Python version, .NET version and package versions are the same on both.
ARCHES = {
    'arm64': dict(package='EEGTrainingGUI-Windows-ARM64', label='ARM64', platform='Windows 11 ARM64',
                  runtime_lock='runtime-lock.json', requirements='requirements.lock',
                  uv_platform='aarch64-pc-windows-msvc', zig_target='aarch64-windows-gnu',
                  pe_machine=0xaa64, vcredist='VC_redist.arm64.exe'),
    'x64': dict(package='EEGTrainingGUI-Windows-x64', label='x64', platform='Windows 10/11 x64 (Intel/AMD)',
                runtime_lock='runtime-lock-x64.json', requirements='requirements-x64.lock',
                uv_platform='x86_64-pc-windows-msvc', zig_target='x86_64-windows-gnu',
                pe_machine=0x8664, vcredist='VC_redist.x64.exe'),
}
PACKAGE_NAME = ARCHES['arm64']['package']
SOURCE_DIRS = ('gui','models','trainer','configs','data','utils','experiment','deploy')
SOURCE_FILES = ('eval.py','eegio_eval.py','metrics.py','requirements.txt','README.md','LICENSE')
IGNORED = {'__pycache__','.git','.DS_Store','artifacts','testdata','EEGdenoisenet','EEG-IO','EEG-EyeBlinks'}


def digest(path, algorithm='sha256'):
    value=hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def pe_machine(path):
    with Path(path).open('rb') as stream:
        header=stream.read(64)
        if len(header)!=64 or header[:2]!=b'MZ':raise ValueError(f'Not a Windows executable: {path}')
        stream.seek(struct.unpack_from('<I',header,0x3c)[0])
        signature=stream.read(6)
        if len(signature)!=6 or signature[:4]!=b'PE\0\0':raise ValueError(f'Invalid PE header: {path}')
        return struct.unpack_from('<H',signature,4)[0]


def pe_subsystem(path):
    with Path(path).open('rb') as stream:
        stream.seek(0x3c)
        offset=struct.unpack('<I',stream.read(4))[0]
        stream.seek(offset+92)
        return struct.unpack('<H',stream.read(2))[0]


def extract_zip(path,destination):
    destination=Path(destination)
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            name=item.filename.replace('\\','/')
            if PureWindowsPath(name).drive or name.startswith('/') or '..' in PurePosixPath(name).parts:
                raise ValueError(f'Unsafe archive path: {item.filename}')
            if stat.S_ISLNK(item.external_attr>>16):raise ValueError('Runtime archives may not contain symlinks')
        archive.extractall(destination)


def cached_asset(asset,cache):
    path=cache/asset['filename']
    if not path.is_file() or digest(path)!=asset['sha256']:
        temp=path.with_suffix('.download')
        with urllib.request.urlopen(asset['url'],timeout=60) as response,temp.open('wb') as stream:
            shutil.copyfileobj(response,stream,1024*1024)
        if digest(temp)!=asset['sha256']:
            temp.unlink()
            raise ValueError(f"Runtime checksum mismatch: {asset['filename']}")
        temp.replace(path)
    if asset.get('sha512') and digest(path,'sha512')!=asset['sha512']:
        raise ValueError(f"Publisher checksum mismatch: {asset['filename']}")
    return path


def copy_sources(root,app,include_runs=False,include_data=False):
    ignore=lambda directory,names:[n for n in names if n in IGNORED or n.endswith(('.pyc','.pyo','.dylib','.so'))]
    for name in SOURCE_DIRS:
        shutil.copytree(root/name,app/name,ignore=ignore)
    for name in SOURCE_FILES:
        if (root/name).is_file():shutil.copy2(root/name,app/name)
    shutil.copytree(root/'packaging/windows',app/'packaging/windows')
    if (root/'docs').is_dir():shutil.copytree(root/'docs',app/'docs',ignore=ignore)
    (app/'runs').mkdir(exist_ok=True)
    copied,skipped=[],[]
    if include_runs:
        from experiment.storage import RunLease
        for directory in sorted((root/'runs').iterdir()):
            if not directory.is_dir() or not (directory/'best.pt').is_file():continue
            try:
                with RunLease(directory):
                    target=app/'runs'/directory.name;target.mkdir()
                    for name in ('best.pt','last.pt','history.json','resolved_config.json','loss_fn.json'):
                        if (directory/name).is_file():shutil.copy2(directory/name,target/name)
                copied.append(directory.name)
            except FileExistsError:skipped.append(directory.name)
        if (root/'deploy/artifacts').is_dir():
            shutil.copytree(root/'deploy/artifacts',app/'deploy/artifacts',ignore=ignore)
    if include_data:
        data=root/'data/EEGdenoisenet'
        if data.is_dir():shutil.copytree(data,app/'data/EEGdenoisenet',ignore=ignore)
        if (root/'deploy/testdata').is_dir():shutil.copytree(root/'deploy/testdata',app/'deploy/testdata',ignore=ignore)
    return dict(runs=copied,active_runs_skipped=skipped,training_data_included=include_data)


def write_launch_files(package,python_version,dotnet_version,arch='arm64'):
    spec=ARCHES[arch]
    runtime=package/'runtime'
    minor=''.join(python_version.split('.')[:2])
    (runtime/f'python{minor}._pth').write_text(f'python{minor}.zip\n.\nsite-packages\n../app\nimport site\n',encoding='utf-8')
    (runtime/'gui.runtimeconfig.json').write_text(json.dumps(dict(runtimeOptions=dict(
        tfm='net10.0',framework=dict(name='Microsoft.WindowsDesktop.App',version=dotnet_version),
        rollForward='LatestPatch')),indent=2),encoding='utf-8')
    scripts={
        'Start GUI.cmd':'"%~dp0runtime\\pythonw.exe" -X utf8 -u -m gui.windows_start',
        'Start in browser.cmd':'"%~dp0runtime\\python.exe" -X utf8 -u -m gui.windows_start --browser',
        'Check installation.cmd':'"%~dp0runtime\\python.exe" -X utf8 -u -m gui.windows_start --check\r\npause',
    }
    for name,command in scripts.items():
        content='@echo off\r\ncd /d "%~dp0app"\r\n'+command+'\r\n'
        (package/name).write_bytes(content.encode('ascii'))
    for name in ('logs','cache','dependencies'):(package/name).mkdir(exist_ok=True)
    (package/'README Windows.txt').write_text(
        f'EEG Training GUI — {spec["platform"]} 便携版\n\n'
        '1. 解压整个 ZIP 到可写目录，例如 C:\\EEG-GUI；不要在压缩包内直接运行。\n'
        '2. 双击 EEG Training GUI.exe。无需安装 Python，Python/.NET 随包提供。\n'
        f'3. 若提示缺少 DLL，运行 dependencies\\{spec["vcredist"]} 安装微软运行库。\n'
        '4. 原生窗口需要 Microsoft Edge WebView2；不可用时运行 Start in browser.cmd。\n'
        '5. 双击 Check installation.cmd 可检查训练算子、ONNX、GUI 接口和任务停止。\n\n'
        '数据与记录：app\\configs、app\\runs、app\\data、app\\deploy\\artifacts。\n'
        '请保留整个目录结构；可整体移动。原有记录中的设备条件继续保留原平台信息。\n'
        '运行日志：logs\\launcher.log；各任务日志在 app\\runs 中。\n'
        f'训练和推理默认使用 {spec["label"]} CPU；本包不启用 GPU/NPU 加速。\n'
        f'本包仅适用于 {spec["label"]} 版 Windows（设置 > 系统 > 关于 > 系统类型）。\n'
        'EEG channels 的 Pi 测量使用 Windows OpenSSH（ssh/scp）及已配置的密钥。\n'
        '旧 Deploy 页的 Pi 脚本另需 Git for Windows 的 Git Bash。\n'
        '关闭窗口时会询问是否停止仍在运行的任务。浏览器备用模式结束后，请关闭其控制台。\n\n'
        '仅在 macOS/Linux 构建时，无法代替 Windows 实机启动验证。\n'
        'manifest.json 记录运行库版本、架构检查和迁移的模型目录。\n',encoding='utf-8')


def validate_project_imports(app,source_root=ROOT):
    """Catch missing local modules without importing the developer's checkout."""
    app,source_root=Path(app),Path(source_root)
    local=set(SOURCE_DIRS)|{path.stem for path in source_root.glob('*.py')}
    checked=set()
    missing=[]
    files=list(app.glob('*.py'))
    for name in SOURCE_DIRS:
        files.extend((app/name).rglob('*.py'))
    for path in files:
        tree=ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):names=[item.name for item in node.names]
            elif isinstance(node,ast.ImportFrom) and not node.level and node.module:names=[node.module]
            else:continue
            for name in names:
                if name.split('.')[0] not in local:continue
                target=app.joinpath(*name.split('.'))
                checked.add(name)
                if not target.with_suffix('.py').is_file() and not target.is_dir():
                    missing.append(f'{name} (imported by {path.relative_to(app)})')
    if missing:
        raise ValueError('Missing project modules in portable app: '+', '.join(sorted(set(missing))))
    return sorted(checked)


def patch_pywebview_coreclr(package):
    """Replace a .NET Framework-private reflection helper with a public API.

    pywebview 6.2.1's FileDialogNative type is absent in .NET Core. Keep the
    patch confined to the locked portable runtime, leaving the build cache intact.
    """
    path=Path(package)/'runtime/site-packages/webview/platforms/winforms.py'
    text=path.read_text(encoding='utf-8')
    marker='# EEG_GUI_CORECLR_FOLDER_DIALOG'
    if marker in text:return
    classes=[node for node in ast.parse(text).body if isinstance(node,ast.ClassDef) and node.name=='OpenFolderDialog']
    if len(classes)!=1:raise ValueError('Unsupported pywebview folder picker; review the CoreCLR adapter')
    node=classes[0]
    lines=text.splitlines(keepends=True)
    replacement=marker+'\nfrom gui.windows_dialog import OpenFolderDialog\n'
    updated=''.join(lines[:node.lineno-1])+replacement+''.join(lines[node.end_lineno:])
    # uv may hardlink installed files to its cache: replace, never truncate.
    temporary=path.with_suffix('.eeg.tmp')
    temporary.write_text(updated,encoding='utf-8')
    temporary.replace(path)


def archive_bundle(package):
    """Export program content without a tester's mutable logs or bytecode cache."""
    package=Path(package).resolve()
    target=package.with_suffix('.zip')
    temporary=target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(package.rglob('*')):
            relative=path.relative_to(package)
            if '__pycache__' in relative.parts or path.name=='.DS_Store':continue
            if relative.parts[0] in ('logs','cache') and len(relative.parts)>1:continue
            archive.write(path,str(Path(package.name)/relative))
    temporary.replace(target)
    target.with_suffix('.zip.sha256').write_text(f'{digest(target)}  {target.name}\n',encoding='ascii')
    return target


def validate_bundle(package,arch='arm64'):
    spec=ARCHES[arch]
    required=['EEG Training GUI.exe','runtime/python.exe','runtime/pythonw.exe','runtime/python313.dll',
              'runtime/dotnet/dotnet.exe','runtime/dotnet/shared/Microsoft.WindowsDesktop.App',
              'app/gui/static/index.html','app/gui/windows_start.py','app/gui/windows_dialog.py',
              'app/metrics.py','runtime/gui.runtimeconfig.json']
    for name in required:
        if not (package/name).exists():raise ValueError(f'Missing package content: {name}')
    project_imports=validate_project_imports(package/'app')
    binaries=[package/'EEG Training GUI.exe',package/'runtime/python.exe',package/'runtime/pythonw.exe',
              package/'runtime/python313.dll',package/'runtime/dotnet/dotnet.exe']
    binaries.extend((package/'runtime').rglob('*.pyd'))
    for path in binaries:
        if pe_machine(path)!=spec['pe_machine']:raise ValueError(f"Expected a native {spec['label']} binary: {path}")
    if pe_subsystem(package/'EEG Training GUI.exe')!=2:
        raise ValueError('The launcher must use the Windows GUI subsystem')
    deps={d.metadata['Name']:d.version for d in importlib.metadata.distributions(path=[str(package/'runtime/site-packages')])}
    for name in ('torch','numpy','scipy','onnx','onnxruntime','pywebview','pythonnet'):
        if name not in deps:raise ValueError(f'Missing dependency: {name}')
    return dict(arch=arch,native_binaries_checked=len(binaries),launcher_subsystem='Windows GUI',
                project_imports=project_imports,dependencies=deps,
                pywebview_coreclr_adapter='# EEG_GUI_CORECLR_FOLDER_DIALOG' in (package/'runtime/site-packages/webview/platforms/winforms.py').read_text(encoding='utf-8'),
                windows_device_tested=False)


def build(out,cache,uv,zig,include_runs=False,include_data=False,root=ROOT,offline=False,arch='arm64'):
    spec=ARCHES[arch]
    out,cache,root=Path(out).resolve(),Path(cache).resolve(),Path(root).resolve()
    out.mkdir(parents=True,exist_ok=True);cache.mkdir(parents=True,exist_ok=True)
    final=out/spec['package']
    if final.exists() or final.with_suffix('.zip').exists():
        raise FileExistsError('Output already exists; choose a fresh --out directory to preserve its runs/data')
    lock=json.loads((root/'packaging/windows'/spec['runtime_lock']).read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory(prefix='windows-stage-',dir=out) as temp:
        package=Path(temp)/spec['package'];package.mkdir()
        runtime=package/'runtime';runtime.mkdir()
        for asset in lock['assets']:
            print('[windows] runtime:',asset['name'],flush=True)
            path=cached_asset(asset,cache)
            if asset['name']=='python':extract_zip(path,runtime)
            elif asset['name'] in ('dotnet','desktop'):extract_zip(path,runtime/'dotnet')
            else:
                (package/'dependencies').mkdir(exist_ok=True)
                shutil.copy2(path,package/'dependencies'/asset['filename'])
        print(f"[windows] install the locked {spec['label']} wheels",flush=True)
        subprocess.run([str(uv),'pip','install','--python-platform',spec['uv_platform'],
                        '--python-version',lock['python'],'--target',str(runtime/'site-packages'),
                        '--require-hashes','--no-compile-bytecode','--cache-dir',str(cache.parent/'uv-cache'),
                        *(['--offline'] if offline else []),
                        '-r',str(root/'packaging/windows'/spec['requirements'])],check=True)
        # C++ development headers are not used by the training/ONNX workflows.
        headers=runtime/'site-packages/torch/include'
        if headers.is_dir():shutil.rmtree(headers)
        print('[windows] copy application, configuration and selected records',flush=True)
        content=copy_sources(root,package/'app',include_runs,include_data)
        write_launch_files(package,lock['python'],lock['dotnet'],arch)
        patch_pywebview_coreclr(package)
        print(f"[windows] compile the native {spec['label']} launcher",flush=True)
        subprocess.run([str(zig),'cc','-target',spec['zig_target'],'-municode','-Wl,--subsystem,windows','-O2','-s',
                        str(root/'gui/windows_launcher.c'),'-o',str(package/'EEG Training GUI.exe'),
                        '-lkernel32','-luser32'],check=True,
                       env={**os.environ,'ZIG_GLOBAL_CACHE_DIR':str(cache.parent/'zig-cache')})
        validation=validate_bundle(package,arch)
        manifest=dict(name=spec['package'],arch=arch,built_at=datetime.now().isoformat(timespec='seconds'),
                      platform=spec['platform'],python=lock['python'],dotnet=lock['dotnet'],
                      requirements_sha256=digest(root/'packaging/windows'/spec['requirements']),
                      runtime_assets=lock['assets'],content=content,validation=validation)
        (package/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        package.rename(final)
    print('[windows] compress the portable folder',flush=True)
    archive=archive_bundle(final)
    print(f'[windows] ready: {archive} ({archive.stat().st_size/2**20:.1f} MiB)',flush=True)
    print(f'[windows] {len(content["runs"])} model directories; {validation["native_binaries_checked"]} native {spec["label"]} binaries checked',flush=True)
    return archive


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',default='dist/windows')
    parser.add_argument('--cache',default='.build/windows/downloads')
    parser.add_argument('--uv',default=shutil.which('uv') or str(ROOT/'.build/windows/tools/bin/uv'))
    parser.add_argument('--zig',default=shutil.which('zig') or str(ROOT/'.build/windows/tools/ziglang/zig'))
    parser.add_argument('--include-runs',action='store_true')
    parser.add_argument('--include-data',action='store_true')
    parser.add_argument('--offline',action='store_true',help='use cached wheels and runtimes')
    parser.add_argument('--arch',choices=(*ARCHES,'both'),default='arm64',
                        help='arm64 (Snapdragon / Windows on ARM), x64 (Intel/AMD), or both')
    args=parser.parse_args(argv)
    for arch in (tuple(ARCHES) if args.arch=='both' else (args.arch,)):
        build(args.out,args.cache,args.uv,args.zig,args.include_runs,args.include_data,
              offline=args.offline,arch=arch)


if __name__=='__main__':main()
