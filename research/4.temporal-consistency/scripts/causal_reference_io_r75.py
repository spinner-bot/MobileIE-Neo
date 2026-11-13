"""Owned R75 paths, atomic live metadata, and the frozen training source set."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'research/4.temporal-consistency'
PROCESS = BASE/'process/exploration-r75'
PROTOCOL = BASE/'docs/exploration-r75-causal-reference-protocol.md'
TRAIN_SOURCES = [
    'research/4.temporal-consistency/scripts/'+name for name in [
        'train_causal_reference_r75.py', 'causal_reference_io_r75.py', 'causal_reference_data_r75.py',
        'causal_reference_training_r75.py', 'rebotnet_reference.py', 'mixed_anchor_r40.py',
        'dynamic_training_r34.py', 'native_training_r14.py', 'exploration_r2_data.py', 'train_mixture_r4b.py']
] + ['third_party/rebotnet/archs.py', 'scripts/prebind_shared_gpu.py', 'scripts/shared_gpu_admission.py',
     'scripts/bound_gpu_observer.py', 'scripts/shared_training_schedule.py',
     'scripts/project-python', 'scripts/project_assets.py']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def seed_root(seed):
    return BASE/f'results/exploration-r75-stage001-seed{seed}'


def cache_root(seed, attempt):
    path = ROOT/f'.cache/checkpoints/exploration-r75/seed{seed}/attempt{attempt:03d}'
    if not path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo'):
        raise ValueError('Checkpoint assets are outside zrh project data.')
    return path


def verify_training_sources():
    frozen_path = PROCESS/'source-freeze001.json'
    saved = json.loads(frozen_path.read_text())
    actual = {path: digest(ROOT/path) for path in TRAIN_SOURCES}
    if saved['source_sha256'] != actual or saved['protocol_sha256'] != digest(PROTOCOL):
        raise RuntimeError('Registered training source or protocol changed; do not silently restart.')
    return saved


def stamp():
    return datetime.now().astimezone().isoformat()
