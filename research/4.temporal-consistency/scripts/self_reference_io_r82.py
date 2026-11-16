"""R82 registered lifecycle; no accelerator imports before process binding."""
import json

from causal_reference_io_r75 import ROOT, BASE, digest, write, append, lines, stamp
from noise_role_io_r79 import acquire_gpu, owned_cache

PROCESS = BASE/'process/exploration-r82'
STEPS = 8000
LABELS = ('spatial', 'spatial_recursive_self05', 'direct', 'centered', 'centered_wrong')
PROTOCOLS = [BASE/'docs'/name for name in (
    'exploration-r82-self-reference-prototype-protocol.md',
    'exploration-r82-self-reference-training-protocol.md')]
SOURCES = ['research/4.temporal-consistency/scripts/'+name for name in (
    'self_reference_r82.py', 'self_reference_io_r82.py', 'check_self_reference_cuda_r82.py',
    'train_self_reference_r82.py', 'evaluate_self_reference_r82.py',
    'evaluate_cvvdp_self_reference_r82.py', 'summarize_self_reference_r82.py',
    'render_self_reference_r82.py', 'run_self_reference_r82.py',
    'mixed_anchor_r40.py', 'motion_transport_r74.py', 'dynamic_training_r34.py',
    'budget_causal_r33.py', 'native_training_r14.py', 'train_mixture_r4b.py',
    'exploration_r2_data.py', 'train_exploration_r2.py', 'explore_temporal_controls.py',
    'causal_reference_io_r75.py', 'noise_role_io_r79.py', 'causal_reference_evaluation_r75.py',
    'prepare_cache_r5.py', 'evaluate_cvvdp_references_r66.py', 'cvvdp_adapter_r66.py',
    'render_dense_history_r64.py', 'half_motion_r4.py', 'aggregation_controls_r20.py',
    'aligned_raw_history_r11.py', 'video_metrics.py', 'bounded_fst.py', 'prepare_cache_r4.py',
    'real_data_r4.py', 'train_real_r4.py')]
SOURCES += ['scripts/'+name for name in (
    'prebind_shared_gpu.py', 'shared_gpu_admission.py', 'bound_gpu_observer.py',
    'shared_training_schedule.py', 'project-python', 'project_assets.py')]
SOURCES += ['MobileIE/model/lle.py', 'MobileIE/model/utils.py']


def verify():
    value = json.loads((PROCESS/'source-freeze001.json').read_text())
    assert value['source_sha256'] == {name: digest(ROOT/name) for name in SOURCES}
    assert value['protocol_sha256'] == {str(p.relative_to(ROOT)): digest(p) for p in PROTOCOLS}
    return value


def precision():
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def update(model, optimizer, raw, target):
    import torch
    optimizer.zero_grad(set_to_none=True)
    prediction = model.window(raw).clamp(0, 1)
    loss = (prediction-target).square().mean()
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    assert bool(torch.isfinite(norm))
    optimizer.step()
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
    return dict(clipped_RGB_MSE=float(loss.detach()), preclip_gradient_norm=float(norm))


def final_markers():
    verify()
    values = {}
    for seed in (0, 1):
        root = BASE/f'results/exploration-r82-stage001-seed{seed}'
        marker = json.loads((root/'final.json').read_text())
        runtime = json.loads((ROOT/marker['runtime']).read_text())
        assert marker['completed_normally'] and marker['final_quality_eligible']
        assert marker['completed_steps'] == marker['next_step'] == marker['registered_steps'] == STEPS
        assert runtime['completed_normally'] and runtime['stop_reason'] == 'complete'
        assert runtime['completed_steps'] == STEPS and max(marker['final_reload'].values()) <= 1e-6
        assert marker['freeze_sha256'] == digest(PROCESS/'source-freeze001.json')
        path = ROOT/marker['path']
        assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
        assert digest(path) == marker['sha256']
        values[seed] = marker
    return values
