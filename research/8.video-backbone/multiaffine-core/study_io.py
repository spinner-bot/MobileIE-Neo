"""Owned M1 spatial-study paths and fixed input/source checks; parent imports no Torch."""
from pathlib import Path
import json
import sys
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
PROCESS = BASE / 'process'
for path in (ROOT / 'scripts', ROOT / 'research/4.temporal-consistency/scripts', BASE.parent, BASE):
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))
from causal_reference_io_r75 import write, append, stamp, digest

ARMS = ('multiaffine', 'original_scratch')
PARAMETERS = dict(multiaffine=5340, original_scratch=4075)


def read(path):
    return json.loads(Path(path).read_text())


def verify():
    frozen = read(PROCESS / 'spatial-source-freeze001.json')
    assert frozen['source_sha256'] == {p: digest(ROOT / p) for p in frozen['source_sha256']}
    return frozen


def owned_cache(name):
    p = ROOT / '.cache/experiments/topic8-multiaffine-core' / name
    assert p.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    return p


def precision():
    import torch
    import cv2
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def models(seed):
    import torch
    from core import MultiaffineCore
    from explore_temporal_controls import MobileIELLENetS
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8930 + 100 * seed)
        first = MultiaffineCore()
        torch.manual_seed(8931 + 100 * seed)
        second = MobileIELLENetS(12)
    result = dict(multiaffine=first, original_scratch=second)
    assert {k: sum(p.numel() for p in m.parameters()) for k, m in result.items()} == PARAMETERS
    return result


def identity(pid):
    try:
        v = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, start_ticks=int(v[19]), state=v[0])
    except FileNotFoundError:
        return dict(pid=pid, already_exited=True)
