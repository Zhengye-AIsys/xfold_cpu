"""Source validation for the optional extension; standard library only."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parent
LIBXSMM_REVISION = '0d1f61ed425aae8318fffafeea9834971a33aec7'


def libxsmm_manifest():
    """Read the pinned manifest and reject paths outside the source snapshot."""
    manifest = json.loads((ROOT / 'libxsmm.lock.json').read_text())
    if manifest['revision'] != LIBXSMM_REVISION:
        raise RuntimeError('LIBXSMM revision does not match the build configuration.')
    for name in manifest['files']:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise RuntimeError(f'Invalid LIBXSMM manifest path: {name}')
    return manifest


def verify_libxsmm(source_root):
    """Validate the pinned build inputs without writing into source_root."""
    source_root = Path(source_root).resolve()
    manifest = libxsmm_manifest()
    for name, expected in manifest['files'].items():
        path = source_root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'LIBXSMM source differs from the pinned snapshot: {name}')
    return source_root


def stage_libxsmm(source_root, build_root):
    """Copy only verified build inputs into a fresh workspace, never into source."""
    source = verify_libxsmm(source_root)
    build_root = Path(build_root).resolve()
    if build_root == source or source in build_root.parents or build_root in source.parents:
        raise RuntimeError('LIBXSMM build workspace must be separate from its source tree.')
    build_root.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix='libxsmm-source-', dir=build_root))
    for name, expected in libxsmm_manifest()['files'].items():
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'LIBXSMM source changed while copying: {name}')
    return workspace
