"""Installed only into this project's independent venv; contain default runtime caches."""
import os
from pathlib import Path
import sys

_assets = Path('/data/Users/zrh/data/MobileIE-Neo')
if Path(sys.prefix).resolve() == _assets/'.venv':
    _locations = {
        'PIP_CACHE_DIR': 'pip-cache', 'TMPDIR': 'tmp', 'XDG_CACHE_HOME': 'runtime-cache/xdg',
        'TORCH_HOME': 'runtime-cache/torch', 'MPLCONFIGDIR': 'runtime-cache/matplotlib',
        'HF_HOME': 'runtime-cache/huggingface', 'CUDA_CACHE_PATH': 'runtime-cache/cuda',
        'TRITON_CACHE_DIR': 'runtime-cache/triton', 'TORCH_EXTENSIONS_DIR': 'runtime-cache/torch-extensions',
        'TORCHINDUCTOR_CACHE_DIR': 'runtime-cache/inductor', 'NUMBA_CACHE_DIR': 'runtime-cache/numba',
        'UV_CACHE_DIR': 'runtime-cache/uv',
    }
    for _name, _relative in _locations.items():
        _path = _assets/_relative
        _path.mkdir(parents=True, exist_ok=True)
        os.environ[_name] = str(_path)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['PYTHONNOUSERSITE'] = '1'
