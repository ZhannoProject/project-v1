"""
Build a double-clickable macOS app bundle that launches the training GUI.

    python -m gui.make_app                 # → ./EEG Training GUI.app
    python -m gui.make_app --out ~/Applications --open

The bundle is a thin launcher for this project's Python (``sys.executable``,
i.e. the venv you build it from): it runs ``python -m gui --native`` from
the project root, so it always uses the current code.  Paths are stored in
Contents/Resources/launch.cfg, so rebuild (or edit that file) if you move
the project or the venv.  Output goes to ~/Library/Logs/eeg-training-gui.log.

The executable is gui/launcher.c compiled with the system ``cc`` (Xcode
Command Line Tools).  It must be a real binary that keeps running while
Python is its child: macOS attributes Desktop/Documents folder permissions
to the launched app's process, and a shell script that exec'd Python would
hand that identity to Python.app, leaving the venv unreadable without any
prompt.  Without a compiler the shell-script launcher is used as a fallback
and you may have to grant folder access manually.

The icon (.icns) is rendered from gui/static/icon.png with `iconutil`;
without PIL or iconutil the bundle is built without an icon.
"""

from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from gui.config_io import PROJECT_ROOT
from gui.native import APP_NAME, ICON_PNG

BUNDLE_ID = "local.eeg-denoise.training-gui"
LOG_PATH = "$HOME/Library/Logs/eeg-training-gui.log"

LAUNCHER_SH = """#!/bin/bash
# Fallback launcher (no C compiler found) — edit paths here if the project moves.
cd {root} || exit 1
exec {python} -m gui --native >> "{log}" 2>&1
"""
LAUNCHER_C = Path(__file__).resolve().parent / "launcher.c"


def _sh_quote(p: Path | str) -> str:
    return "'" + str(p).replace("'", "'\\''") + "'"


def build_icns(png: Path, out_icns: Path) -> bool:
    """png (1024×1024) → .icns via an iconset + iconutil. Returns False if not possible."""
    if not png.is_file() or shutil.which("iconutil") is None:
        return False
    try:
        from PIL import Image
    except ImportError:
        return False
    src = Image.open(png).convert("RGBA")
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "icon.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                px = size * scale
                name = f"icon_{size}x{size}" + ("@2x" if scale == 2 else "") + ".png"
                src.resize((px, px), Image.LANCZOS).save(iconset / name)
        try:
            subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out_icns)],
                           check=True, capture_output=True)
        except subprocess.CalledProcessError:
            return False
    return out_icns.is_file()


def compile_launcher(out: Path) -> bool:
    """gui/launcher.c → Mach-O at ``out``. Returns False if no compiler is available."""
    cc = shutil.which("cc") or shutil.which("clang")
    if cc is None or not LAUNCHER_C.is_file():
        return False
    r = subprocess.run([cc, "-O2", "-Wall", "-mmacosx-version-min=11.0",
                        "-o", str(out), str(LAUNCHER_C)], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[make_app] warning: compiling launcher.c failed, using the shell fallback:\n{r.stderr}")
        return False
    return True


def build_app(out_dir: Path, python: str | None = None, root: Path | None = None) -> Path:
    python = python or sys.executable
    root = Path(root or PROJECT_ROOT)
    app = out_dir / f"{APP_NAME}.app"
    contents = app / "Contents"
    if app.exists():
        shutil.rmtree(app)
    (contents / "MacOS").mkdir(parents=True)
    (contents / "Resources").mkdir()

    launcher = contents / "MacOS" / "launch"
    (contents / "Resources" / "launch.cfg").write_text(f"root={root}\npython={python}\n")
    native = compile_launcher(launcher)
    if not native:
        launcher.write_text(LAUNCHER_SH.format(root=_sh_quote(root), python=_sh_quote(python), log=LOG_PATH))
        launcher.chmod(0o755)

    has_icon = build_icns(ICON_PNG, contents / "Resources" / "icon.icns")
    info = {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleVersion": "1.0",
        "CFBundleShortVersionString": "1.0",
        "CFBundlePackageType": "APPL",
        "CFBundleExecutable": "launch",
        "LSMinimumSystemVersion": "11.0",
        "NSHighResolutionCapable": True,
        "LSApplicationCategoryType": "public.app-category.developer-tools",
        # The project and its venv usually live under ~/Desktop or ~/Documents;
        # macOS asks for folder access on first launch and shows this text.
        "NSDesktopFolderUsageDescription": "The training GUI reads the project's configs/ and venv.",
        "NSDocumentsFolderUsageDescription": "The training GUI reads the project's configs/ and venv.",
        "NSDownloadsFolderUsageDescription": "The training GUI reads the project's configs/ and venv.",
    }
    if has_icon:
        info["CFBundleIconFile"] = "icon"
    with open(contents / "Info.plist", "wb") as f:
        plistlib.dump(info, f)
    (contents / "PkgInfo").write_text("APPL????")
    _adhoc_sign(app)
    return app


def _adhoc_sign(app: Path) -> None:
    """Ad-hoc signature (no certificate) so macOS privacy prompts attribute cleanly."""
    if shutil.which("codesign") is None:
        return
    subprocess.run(["codesign", "--force", "--deep", "--sign", "-", str(app)],
                   check=False, capture_output=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=f"Build '{APP_NAME}.app'")
    parser.add_argument("--out", default=str(PROJECT_ROOT), help="folder to write the .app into (default: project root)")
    parser.add_argument("--open", action="store_true", help="launch it after building")
    args = parser.parse_args(argv)
    if sys.platform != "darwin":
        raise SystemExit("App bundles are a macOS thing; on other systems run `python -m gui`.")
    try:
        import webview  # noqa: F401
    except ImportError:
        print("[make_app] warning: pywebview is not installed in this interpreter; "
              "the app will fail to open a window until you `python -m pip install pywebview`.")
    app = build_app(Path(args.out).expanduser())
    icon = "with icon" if (app / "Contents" / "Resources" / "icon.icns").is_file() else "no icon"
    kind = "native launcher" if (app / "Contents" / "MacOS" / "launch").read_bytes()[:4] != b"#!/b" else "shell launcher"
    print(f"[make_app] built {app}  ({icon}, {kind})")
    print(f"[make_app] runs: {sys.executable} -m gui --native   (cwd {PROJECT_ROOT})")
    print(f"[make_app] logs: {os.path.expandvars(LOG_PATH.replace('$HOME', '~'))}")
    print("[make_app] Double-click it in Finder, or drag it to the Dock. Rebuild if you move the project or venv.")
    if args.open:
        subprocess.run(["open", str(app)], check=False)


if __name__ == "__main__":
    main()
