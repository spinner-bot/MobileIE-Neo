"""Standard-library helpers for zrh-owned runtime assets; never import the borrowed env."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[1]
OWNER = Path('/data/Users/zrh')
ASSETS = OWNER/'data/MobileIE-Neo'
ENVIRONMENT = ASSETS/'.venv'
CACHE = ASSETS/'cache'
RECORDS = REPO/'research/4.temporal-consistency/process/asset-migration'


def stamp():
    return datetime.now().astimezone().isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def mounted_data():
    result = subprocess.check_output(['findmnt', '-n', '-o', 'TARGET,SOURCE,FSTYPE', '-T', '/data'], text=True).strip()
    if not result.startswith('/data '):
        raise RuntimeError('/data is not mounted as the project data volume')
    return result


def project_python_processes():
    rows = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry/'cmdline').read_bytes().split(b'\0')
            if not argv or not argv[0]:
                continue
            program = argv[0].decode(errors='replace')
            cwd = (entry/'cwd').resolve()
            if cwd == REPO and (program.startswith('.venv/bin/python') or program.startswith(str(REPO/'.venv/bin/python'))):
                rows.append({'pid': int(entry.name), 'argv': [a.decode(errors='replace') for a in argv if a]})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return rows


def owned_environment():
    env = os.environ.copy()
    for key in ['PYTHONPATH', 'PYTHONHOME', 'PIP_EXTRA_INDEX_URL', 'PIP_TRUSTED_HOST']:
        env.pop(key, None)
    env.update({'VIRTUAL_ENV': str(ENVIRONMENT), 'PYTHONNOUSERSITE': '1',
                'PIP_CONFIG_FILE': '/dev/null', 'PIP_INDEX_URL': 'https://pypi.org/simple',
                'PIP_DISABLE_PIP_VERSION_CHECK': '1', 'PIP_CACHE_DIR': str(ASSETS/'pip-cache'),
                'TMPDIR': str(ASSETS/'tmp'), 'XDG_CACHE_HOME': str(ASSETS/'runtime-cache/xdg'),
                'TORCH_HOME': str(ASSETS/'runtime-cache/torch'), 'MPLCONFIGDIR': str(ASSETS/'runtime-cache/matplotlib'),
                'HF_HOME': str(ASSETS/'runtime-cache/huggingface'), 'CUDA_CACHE_PATH': str(ASSETS/'runtime-cache/cuda'),
                'TRITON_CACHE_DIR': str(ASSETS/'runtime-cache/triton'),
                'TORCH_EXTENSIONS_DIR': str(ASSETS/'runtime-cache/torch-extensions'),
                'TORCHINDUCTOR_CACHE_DIR': str(ASSETS/'runtime-cache/inductor'),
                'NUMBA_CACHE_DIR': str(ASSETS/'runtime-cache/numba'), 'UV_CACHE_DIR': str(ASSETS/'runtime-cache/uv'),
                'PYTHONDONTWRITEBYTECODE': '1'})
    env['PATH'] = str(ENVIRONMENT/'bin')+os.pathsep+env.get('PATH', '')
    for key in ['PIP_CACHE_DIR', 'TMPDIR', 'XDG_CACHE_HOME', 'TORCH_HOME', 'MPLCONFIGDIR', 'HF_HOME',
                'CUDA_CACHE_PATH', 'TRITON_CACHE_DIR', 'TORCH_EXTENSIONS_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'NUMBA_CACHE_DIR', 'UV_CACHE_DIR']:
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def inventory(root):
    root = Path(root)
    files, links = {}, {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in list(dirs)+names:
            path = Path(directory)/name
            relative = str(path.relative_to(root))
            if path.is_symlink():
                target = path.resolve()
                if not target.is_relative_to(OWNER):
                    raise RuntimeError('runtime asset links outside zrh: '+str(path))
                links[relative] = {'target': os.readlink(path), 'resolved': str(target)}
            elif path.is_file():
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(8*1024*1024), b''):
                        digest.update(block)
                files[relative] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
    return {'files': files, 'links': links, 'file_count': len(files), 'bytes': sum(r['bytes'] for r in files.values())}
