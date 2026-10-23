"""
Host the training GUI in a native window instead of a browser tab.

Uses pywebview (a thin wrapper over the platform web view: WKWebView on
macOS).  The page and the JSON API are exactly the ones ``python -m gui
--browser`` serves; only the host differs.  The HTTP server runs on a
background thread bound to 127.0.0.1 with a random free port, and shuts
down when the window is closed.

    pip install pywebview          # optional dependency, see requirements.txt
    python -m gui                  # native window (default when pywebview is present)
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

from gui import server as _server
from gui.server import GUIServer, PROJECT_ROOT

APP_NAME = "EEG Training GUI"
ICON_PNG = Path(__file__).resolve().parent / "static" / "icon.png"


def native_available() -> bool:
    try:
        import webview  # noqa: F401
        return True
    except ImportError:
        return False


def _brand_process_macos() -> None:
    """Dock icon + menu-bar name for the bare Python process (best effort)."""
    if sys.platform != "darwin":
        return
    try:
        from AppKit import NSApplication, NSBundle, NSImage
        info = NSBundle.mainBundle().infoDictionary()
        if info is not None:
            info["CFBundleName"] = APP_NAME          # menu bar shows this instead of "Python"
        if ICON_PNG.is_file():
            image = NSImage.alloc().initWithContentsOfFile_(str(ICON_PNG))
            NSApplication.sharedApplication().setApplicationIconImage_(image)
    except Exception:                                 # cosmetic only; never block launch
        pass


def run_native(host: str = "127.0.0.1", port: int = 0, verbose: bool = False,
               width: int = 1280, height: int = 860) -> None:
    """Serve on a background thread and block in the native window until it closes."""
    try:
        import webview
    except ImportError as e:
        raise SystemExit(
            "pywebview is not installed. Either `python -m pip install pywebview` "
            "or run `python -m gui --browser`."
        ) from e

    httpd = GUIServer(host, port, verbose)
    url = f"http://{host}:{httpd.server_address[1]}/"
    threading.Thread(target=httpd.serve_forever, daemon=True, name="gui-http").start()
    print(f"[gui] {APP_NAME}: serving {url} inside a native window  (project: {PROJECT_ROOT})")

    _brand_process_macos()
    window = webview.create_window(
        "Training settings", url, width=width, height=height, min_size=(900, 600),
        text_select=True,
    )

    # Anything closing the window would kill: a training run, an evaluation, or
    # a deploy run (which may be driving a Raspberry Pi over SSH).
    # TRAIN_JOBS is a registry rather than one job -- there can be several runs
    # going at once -- but it answers ``running`` the same way, so the guard
    # does not need to know the difference.
    jobs = ((_server.TRAIN_JOBS, "training run"), (_server.EVAL_JOB, "evaluation"),
            (_server.DEPLOY_JOB, "deploy run"))

    def on_closing():
        # Returning False keeps the window open.
        busy = [what for job, what in jobs if job.running]
        if busy:
            ok = window.create_confirmation_dialog(
                "Still running",
                f"A {' and a '.join(busy)} is still going. Quit the GUI and stop it?",
            )
            if not ok:
                return False
        return True

    window.events.closing += on_closing

    try:
        # private_mode=False keeps localStorage (last-opened preset) across launches
        webview.start(private_mode=False)
    finally:
        for job, what in jobs:
            if job.running:
                print(f"[gui] stopping the {what} ...")
                job.stop()
        httpd.shutdown()
        httpd.server_close()
