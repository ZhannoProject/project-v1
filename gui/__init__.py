"""
Training GUI — a local web page that edits the YAML presets in configs/
and launches ``python -m trainer`` from them.

    python -m gui                # opens http://127.0.0.1:8765/ in the browser
    python -m gui --port 9000 --no-browser

Stdlib only (http.server + the pyyaml the project already needs).
    config_io.py   schema of the YAML keys + comment-preserving writer
    server.py      JSON API and the training-job runner
    static/        the single-page front end
"""
