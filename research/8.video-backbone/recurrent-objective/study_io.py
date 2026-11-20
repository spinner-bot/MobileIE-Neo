"""Owned paths and immutable source checks for the T2 deployed-recursion control study."""
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
PROCESS = BASE / 'process'
for path in (ROOT / 'scripts', ROOT / 'research/4.temporal-consistency/scripts', BASE.parent, BASE):
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))
from causal_reference_io_r75 import write, append, stamp, digest

ARMS = ('clip_gt', 'rec_gt')
FREEZE = PROCESS / 'training-source-freeze001.json'


def read(path):
    return json.loads(path.read_text())


def verify(freeze_path=FREEZE, protocol_name='training-protocol.md'):
    frozen = read(freeze_path)
    assert frozen['source_sha256'] == {p: digest(ROOT / p) for p in frozen['source_sha256']}
    assert frozen['protocol_sha256'] == digest(BASE / protocol_name)
    return frozen


def owned_cache(name):
    path = ROOT / '.cache/experiments/topic8-recurrent-objective' / name
    assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    return path


def precision():
    import cv2
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def identity(pid):
    try:
        values = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, start_ticks=int(values[19]), state=values[0])
    except FileNotFoundError:
        return dict(pid=pid, already_exited=True)
