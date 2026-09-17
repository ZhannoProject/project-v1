"""CLI entry point: ``python -m gui [--port N] [--host H] [--no-browser] [-v]``."""

import argparse
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from gui.server import serve


def main(argv=None):
    parser = argparse.ArgumentParser(description="Training-settings GUI for the YAML presets")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (keep it local)")
    parser.add_argument("--port", type=int, default=8765, help="port (0 = pick a free one)")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    parser.add_argument("-v", "--verbose", action="store_true", help="log every HTTP request")
    args = parser.parse_args(argv)
    serve(args.host, args.port, open_browser=not args.no_browser, verbose=args.verbose)


if __name__ == "__main__":
    main()
