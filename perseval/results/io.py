"""Atomic local writes and explicit, bounded artifact selection."""
from __future__ import annotations

import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .bundle import MAX_FILE, MAX_FILES, MAX_TOTAL
from .contract import ProtocolError, canonical_path


def atomic_write(path: Path, data: bytes):
    path = Path(path)
    if path.is_symlink():
        raise ProtocolError(f'refusing to overwrite a symlink: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def local_lock(directory: Path):
    path = directory / '.result.lock'
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ProtocolError(f'run is locked; inspect a possibly interrupted export before removing {path}') from exc
    try:
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def selected_artifacts(artifacts):
    """Read only explicit files/directories. Reject all selected symlinks/special files."""
    files, declarations = {}, []
    if artifacts is None:
        return files, declarations
    if not isinstance(artifacts, dict):
        raise ProtocolError('artifacts must map a canonical name to a file or directory')
    for alias, source in sorted(artifacts.items()):
        canonical_path(alias)
        if '/' in alias:
            raise ProtocolError('artifact names must be a single canonical path component')
        source = Path(source).expanduser()
        if source.is_symlink():
            raise ProtocolError(f'artifact symlinks are not allowed: {source}')
        if not source.exists():
            raise ProtocolError(f'artifact does not exist: {source}')
        paths = sorted(source.rglob('*')) if source.is_dir() else [source]
        for path in paths:
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ProtocolError(f'artifact symlink could escape the selection: {path}')
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise ProtocolError(f'artifact is not a regular file: {path}')
            suffix = path.relative_to(source).as_posix() if source.is_dir() else source.name
            name = canonical_path(f'artifacts/{alias}/{suffix}')
            if len(name) > 100:
                raise ProtocolError(f'artifact archive path exceeds 100 bytes: {name}')
            if path.stat().st_size > MAX_FILE:
                raise ProtocolError(f'artifact exceeds 32 MiB: {path}')
            data = path.read_bytes()
            if len(data) > MAX_FILE or sum(map(len, files.values())) + len(data) > MAX_TOTAL or len(files) >= MAX_FILES - 7:
                raise ProtocolError('artifact selection exceeds bundle size limits')
            files[name] = data
            declarations.append({'path':name, 'media_type':'application/octet-stream', 'description':f'Explicitly selected {alias}: {suffix}'})
    return files, declarations
