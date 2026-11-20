"""Owned paths, common initialization and source locking for topic8 v1."""
from pathlib import Path
import copy
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
LEGACY = ROOT / 'research/4.temporal-consistency/scripts'
for directory in (LEGACY, ROOT / 'scripts'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from causal_reference_io_r75 import write, append, stamp, digest
from motion_transport_r74 import load_spatial
from noise_role_io_r79 import acquire_gpu
from mixed_anchor_r40 import MixedPool
from packed_backbone import PackedBackbone
from causal_pipeline import CausalEnhancer

PROCESS = BASE / 'process/v1'
PROTOCOL = BASE / 'protocol-v1.md'
SOURCE_NAMES = ('packed_backbone.py', 'causal_pipeline.py', 'runtime_io.py',
                'qualify_v1.py', 'train_v1.py', 'evaluate_v1.py')


def owned_cache(name):
    path = ROOT / '.cache/experiments/topic8-v1' / name
    if not path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo'):
        raise ValueError('Runtime assets must be in zrh project data')
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


def fresh_core(seed):
    import torch
    torch.manual_seed(8900 + seed)
    return PackedBackbone()


def make_controls(core):
    result = {}
    for route in ('spatial', 'fixed', 'learned'):
        model = CausalEnhancer(route)
        model.backbone = copy.deepcopy(core)
        result[route] = model
    return result


def verify():
    frozen = json.loads((PROCESS / 'source-freeze001.json').read_text())
    if frozen['source_sha256'] != {p: digest(ROOT / p) for p in frozen['source_sha256']}:
        raise RuntimeError('V1 scientific source changed; do not silently resume')
    if frozen['protocol_sha256'] != digest(PROTOCOL):
        raise RuntimeError('Registered protocol changed')
    return frozen
