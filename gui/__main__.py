"""
CLI entry point.

    python -m gui                 native window (pywebview) if available, else browser
    python -m gui --browser       serve on http://127.0.0.1:8765/ and open a browser tab
    python -m gui --native        insist on the native window (error if pywebview is missing)
"""

import argparse
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from gui.server import serve
from gui.native import run_native, native_available


def main(argv=None):
    parser = argparse.ArgumentParser(description="Training-settings GUI for the YAML presets")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--native", action="store_true", help="native window (default if pywebview is installed)")
    mode.add_argument("--browser", action="store_true", help="serve for a browser tab instead")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (keep it local)")
    parser.add_argument("--port", type=int, default=None,
                        help="port (browser mode default 8765; native mode picks a free one)")
    parser.add_argument("--no-browser", action="store_true", help="browser mode: do not open a tab")
    parser.add_argument("-v", "--verbose", action="store_true", help="log every HTTP request")
    args = parser.parse_args(argv)

    use_native = args.native or (not args.browser and native_available())
    if use_native:
        run_native(args.host, 0 if args.port is None else args.port, verbose=args.verbose)
    else:
        if not args.browser and not args.native:
            print("[gui] pywebview not installed → browser mode "
                  "(python -m pip install pywebview for a native window)")
        serve(args.host, 8765 if args.port is None else args.port,
              open_browser=not args.no_browser, verbose=args.verbose)


if __name__ == "__main__":
    main()
