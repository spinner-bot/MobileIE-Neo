"""Owned I1-S assets and small, explicit source scopes; no Torch imported here."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT / 'research/4.temporal-consistency/scripts'))
from causal_reference_io_r75 import append, digest, write


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def identity(pid=None):
    pid = os.getpid() if pid is None else pid
    parts = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=int(parts[19]), state=parts[0])


def owned_cache(name):
    path = ROOT / '.cache/experiments/topic8-instant-video/simple' / name
    assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    return path


def verify_preflight():
    scope = read(BASE / 'process/simple-source-freeze001.json')
    assert all(digest(ROOT / p) == h for p, h in scope['source_sha256'].items())
    vendor = ROOT / 'third_party/instant-video-models'
    for name in ('SOURCE.json', 'SOURCE-SIMPLE001.json'):
        assert all(digest(vendor / p) == v['sha256'] for p, v in read(vendor / name)['files'].items())
    return scope


def verify_training():
    verify_preflight()
    scope = read(BASE / 'process/simple-training-source-freeze001.json')
    assert all(digest(ROOT / p) == h for p, h in scope['source_sha256'].items())
    return scope
