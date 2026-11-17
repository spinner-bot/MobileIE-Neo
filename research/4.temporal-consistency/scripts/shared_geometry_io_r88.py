"""R88 ownership, precision and stage-specific immutable source records."""
import json

from causal_reference_io_r75 import ROOT, BASE, digest, write, append, stamp, lines
from noise_role_io_r79 import acquire_gpu, owned_cache

PROCESS = BASE/'process/exploration-r88'
PROTOCOL = BASE/'docs/exploration-r88-shared-stem-causal-core-protocol.md'
STEPS = 2000
COMMON = ['research/4.temporal-consistency/scripts/'+name for name in (
    'shared_geometry_r88.py', 'shared_geometry_data_r88.py', 'shared_geometry_io_r88.py',
    'mixed_anchor_r40.py', 'motion_transport_r74.py', 'budget_causal_r33.py',
    'dynamic_training_r34.py', 'native_training_r14.py', 'train_mixture_r4b.py',
    'exploration_r2_data.py', 'train_exploration_r2.py', 'explore_temporal_controls.py',
    'causal_reference_io_r75.py', 'noise_role_io_r79.py', 'video_metrics.py',
    'aligned_raw_history_r11.py', 'aggregation_controls_r20.py', 'bounded_fst.py',
    'real_data_r4.py', 'train_real_r4.py')]
COMMON += ['scripts/'+name for name in ('prebind_shared_gpu.py', 'shared_gpu_admission.py',
    'bound_gpu_observer.py', 'shared_training_schedule.py', 'project-python', 'project_assets.py')]
COMMON += ['MobileIE/model/lle.py', 'MobileIE/model/utils.py']


def freeze(extra):
    paths = sorted(set(COMMON+['research/4.temporal-consistency/scripts/'+name for name in extra]))
    return dict(at=stamp(), source_sha256={p: digest(ROOT/p) for p in paths},
                protocol_sha256=digest(PROTOCOL))


def verify(value):
    assert value['protocol_sha256'] == digest(PROTOCOL)
    assert value['source_sha256'] == {p: digest(ROOT/p) for p in value['source_sha256']}
    return value


def precision():
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def final_markers():
    values = {}
    verify(json.loads((PROCESS/'training-source-freeze001.json').read_text()))
    for seed in (0, 1):
        marker = json.loads((BASE/f'results/exploration-r88-stage001-seed{seed}/final.json').read_text())
        runtime = json.loads((ROOT/marker['runtime']).read_text())
        assert marker['final_quality_eligible'] and marker['completed_steps'] == STEPS
        assert runtime['completed_normally'] and runtime['stop_reason'] == 'complete'
        assert runtime['completed_steps'] == STEPS
        assert marker['freeze_sha256'] == digest(PROCESS/'training-source-freeze001.json')
        assert (ROOT/marker['path']).resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
        assert digest(ROOT/marker['path']) == marker['sha256']
        values[seed] = marker
    return values
