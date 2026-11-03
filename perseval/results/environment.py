"""Capture once at recording time. Export never replaces this snapshot."""
from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def software_provenance():
    try:
        version = importlib.metadata.version('perseval')
    except importlib.metadata.PackageNotFoundError:
        version = '1.0.0'
    root = Path(__file__).resolve().parents[2]
    revision, dirty = None, None
    if (root / '.git').exists():
        try:
            revision = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
            dirty = bool(subprocess.check_output(['git','-C',str(root),'status','--porcelain'], text=True, stderr=subprocess.DEVNULL).strip())
        except (OSError, subprocess.SubprocessError):
            revision, dirty = None, None
    return {'perseval_version':version, 'perseval_git_revision':revision, 'perseval_git_dirty':dirty}


def capture_environment():
    packages = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get('Name')
        if name:
            packages[name] = distribution.version
    # Do not import or initialize a GPU backend solely to take a snapshot.
    torch = sys.modules.get('torch')
    accelerator = None
    details = {'pytorch':packages.get('torch'), 'cuda_version':None, 'cuda_available':None, 'gpus':None}
    if torch is not None:
        torch_version = getattr(torch, '__version__', None)
        details['pytorch'] = str(torch_version) if torch_version is not None else details['pytorch']
        details['cuda_version'] = getattr(getattr(torch, 'version', None), 'cuda', None)
        try:
            details['cuda_available'] = bool(torch.cuda.is_available())
            if details['cuda_available']:
                details['gpus'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                accelerator = ', '.join(details['gpus']) or None
        except (AttributeError, RuntimeError, OSError):
            details['cuda_available'] = None
    return {'schema_version':'1.0', 'python':platform.python_version(),
            'platform':platform.platform(), 'machine':platform.machine() or None,
            'packages':dict(sorted(packages.items())), 'accelerator':accelerator,
            'notes':json.dumps(details, sort_keys=True, separators=(',', ':'))}
