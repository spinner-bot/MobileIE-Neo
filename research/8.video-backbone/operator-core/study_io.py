"""O1 owned paths, metadata and named-source checking; importing needs no Torch."""
from pathlib import Path
import json
import os
import sys
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT / 'research/4.temporal-consistency/scripts'))
from causal_reference_io_r75 import write, append, digest, stamp


def read(path):
    return json.loads(Path(path).read_text())


def verify(name='preflight-source-freeze001.json'):
    frozen = read(BASE / 'process' / name)
    assert all(digest(ROOT / p) == h for p, h in frozen['source_sha256'].items())
    return frozen


def identity():
    values = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=os.getpid(), start_ticks=int(values[19]), state=values[0])


def owned_cache(name):
    path = ROOT / '.cache/experiments/topic8-operator-core' / name
    assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    return path
