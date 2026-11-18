"""Entry point of the relocatable Windows ARM64 distribution."""

import argparse
import ctypes
import os
from pathlib import Path
import sys
import traceback


def startup_error_message(error,detail):
    message='启动失败，详情见 logs\\launcher.log。\n'
    if isinstance(error,ModuleNotFoundError):
        message+=f'程序包缺少模块：{error.name or "未知模块"}。请使用修复后的完整程序包。\n'
    elif isinstance(error,OSError) and getattr(error,'winerror',None)==126:
        message+='Windows 运行库加载失败。可先运行 dependencies\\VC_redist.arm64.exe，随后执行 Check installation.cmd。\n'
    else:
        message+='可运行 Check installation.cmd 检查；原生窗口不可用时，可运行 Start in browser.cmd。\n'
    return message+'\n'+detail[-1800:]


def configure_runtime(bundle):
    bundle = Path(bundle).resolve()
    app, runtime = bundle/'app', bundle/'runtime'
    os.chdir(app)
    # CLI jobs use python.exe, with pipes and CREATE_NO_WINDOW supplied by Job.
    sys.executable = str(runtime/'python.exe')
    os.environ.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1',
                      MPLBACKEND='Agg', MPLCONFIGDIR=str(bundle/'cache/matplotlib'),
                      DOTNET_ROOT=str(runtime/'dotnet'), DOTNET_MULTILEVEL_LOOKUP='0',
                      PYTHONNET_RUNTIME='coreclr',
                      PYTHONNET_CORECLR_RUNTIME_CONFIG=str(runtime/'gui.runtimeconfig.json'))
    return app


def load_desktop_runtime(bundle):
    from pythonnet import load
    bundle=Path(bundle)
    load('coreclr',runtime_config=str(bundle/'runtime/gui.runtimeconfig.json'),
         dotnet_root=str(bundle/'runtime/dotnet'))
    import clr
    # .NET Core resolves these lazily. WinForms' namespace alone does not
    # expose SystemEvents or the Drawing types through pythonnet on ARM64.
    for assembly in ('System.Windows.Forms','Microsoft.Win32.SystemEvents','System.Drawing.Common'):
        clr.AddReference(assembly)


def main(argv=None):
    parser = argparse.ArgumentParser(description='EEG Training GUI for Windows ARM64')
    parser.add_argument('--check', action='store_true', help='run a local installation self-check')
    parser.add_argument('--browser', action='store_true', help='use the default browser if WebView2 is unavailable')
    args = parser.parse_args(argv)
    bundle = Path(__file__).resolve().parents[2]
    log = None
    try:
        (bundle/'logs').mkdir(exist_ok=True)
        log = (bundle/'logs/launcher.log').open('a', encoding='utf-8', buffering=1)
        if sys.stdout is None: sys.stdout = log
        if sys.stderr is None: sys.stderr = log
        configure_runtime(bundle)
        if args.check:
            from gui.windows_smoke import check
            check(bundle)
        elif args.browser:
            from gui.server import serve
            serve(port=0)
        else:
            # Set the runtime explicitly rather than using a machine's .NET
            # Framework or a different architecture's system-wide install.
            load_desktop_runtime(bundle)
            from gui.native import run_native
            run_native()
        return 0
    except Exception as error:
        detail = traceback.format_exc()
        if log: log.write(detail)
        if sys.platform == 'win32':
            ctypes.windll.user32.MessageBoxW(None,
                startup_error_message(error,detail),
                'EEG Training GUI', 0x10)
        else:
            raise
        return 1
    finally:
        if log:
            log.flush()


if __name__ == '__main__':
    raise SystemExit(main())
