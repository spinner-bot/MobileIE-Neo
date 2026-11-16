"""Small R79 lifecycle helpers; shared-card policy itself is unchanged."""
import fcntl
import json
from pathlib import Path
import sys

from causal_reference_io_r75 import ROOT, BASE, digest, write, append, stamp

sys.path.insert(0, str(ROOT/'scripts'))
PROCESS = BASE/'process/exploration-r79'
PROTOCOL = BASE/'docs/exploration-r79-causal-noise-role-protocol.md'
SOURCES = ['research/4.temporal-consistency/scripts/'+name for name in (
    'noise_role_data_r79.py', 'noise_role_model_r79.py', 'noise_role_io_r79.py',
    'check_noise_role_r79.py', 'train_noise_role_r79.py', 'multiscale_fst_r16.py',
    'mixed_anchor_r40.py', 'dynamic_training_r34.py', 'native_training_r14.py',
    'train_mixture_r4b.py', 'train_exploration_r2.py', 'explore_temporal_controls.py',
    'exploration_r2_data.py', 'causal_reference_io_r75.py')]
SOURCES += ['scripts/'+name for name in ('prebind_shared_gpu.py', 'shared_gpu_admission.py',
    'bound_gpu_observer.py', 'shared_training_schedule.py', 'project-python', 'project_assets.py')]
SOURCES += ['MobileIE/pretrain/lolv1_best_slim.pkl']


def verify():
    frozen = json.loads((PROCESS/'source-freeze001.json').read_text())
    assert frozen['source_sha256'] == {p: digest(ROOT/p) for p in SOURCES}
    assert frozen['protocol_sha256'] == digest(PROTOCOL)
    return frozen


def acquire_gpu(bound, out):
    import torch
    from prebind_shared_gpu import actual_device
    from shared_gpu_admission import sustained_samples, assess, snapshot
    from shared_training_schedule import SharedTrainingSchedule, ResourcePressure
    assert not torch.cuda.is_initialized()
    samples = sustained_samples()
    selected = next(r for r in assess(samples)['candidates'] if r['uuid'] == bound)
    write(out/'admission.json', dict(samples=samples, selected=selected, CPU_preparation_completed=True))
    if not selected['allowed'] or selected['index'] not in (0, 1, 2, 3):
        raise ResourcePressure('R79 needs headroom in registered physical group0..3.')
    lock = (ROOT/f'.cache/gpu-{selected["index"]}.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise ResourcePressure('Another project task holds this card.') from error
    immediate = snapshot()
    write(out/'immediate-pre-CUDA.json', immediate)
    current = next(r for r in immediate['GPUs'] if r['uuid'] == bound)
    if current['total_mib']-current['used_mib'] < selected['required_free_mib']:
        raise ResourcePressure('Headroom changed before CUDA initialization.')
    torch.cuda.set_per_process_memory_fraction(2048/selected['total_mib'], 0)
    marker = torch.empty(1, device='cuda')
    torch.cuda.synchronize()
    actual = actual_device(bound)
    write(out/'actual-device.json', actual)
    assert actual['matched']
    del marker
    return lock, SharedTrainingSchedule(selected), selected


def owned_cache(name):
    path = ROOT/'.cache'/name
    assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    path.mkdir(parents=True, exist_ok=False)
    return path
