"""Atomic artifact writes and process-scoped ownership of a run directory."""

import json
import os
from pathlib import Path
import tempfile
from uuid import uuid4


SCHEMA_VERSION = 1


def atomic_write(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path, value):
    def convert(item):
        if isinstance(item, Path):
            return str(item)
        raise TypeError(f"Cannot serialize {type(item).__name__} in experiment metadata")
    atomic_write(path, lambda f: f.write(json.dumps(value, indent=2, default=convert).encode("utf-8")))


class RunLease:
    """An OS lock releases even after a crash. Never unlink its lock-file inode."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.stream = None
        self.run_id = uuid4().hex

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        stream = (self.directory / ".run.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if stream.seek(0, 2) == 0:
                    stream.write(b"\0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise FileExistsError(f"Run directory is already active: {self.directory}") from exc
        self.stream = stream
        return self

    def __exit__(self, *_):
        if self.stream is not None:
            # Closing the descriptor releases the OS lock on all supported platforms.
            self.stream.close()
            self.stream = None


def assert_run_available(directory):
    # Do not create directories during read-only GUI previews.
    if Path(directory).exists():
        with RunLease(directory):
            pass


def archive_existing(directory, names, run_id):
    directory = Path(directory)
    present = [directory / n for n in names if (directory / n).is_file()]
    if present:
        archive = directory / "archive" / run_id
        archive.mkdir(parents=True, exist_ok=False)
        for path in present:
            os.replace(path, archive / path.name)
