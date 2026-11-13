"""Frozen reference loaders and frame-streamed original inference for R72."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dynamic_training_r34 import BASE, ROOT, digest
from drv_references_r59 import load_network
from evaluate_spatial_l1_r69 import load_final
from fastdvdnet_adapter_r12 import official_inference, window_indices
from multinex_reference_r55 import prediction
from response_cue_r45 import load_frozen
from stablellve_adapted_r65 import load_adapted
from stablellve_adapter_r15 import input_tensor, output_rgb
from task_fastdvd_r47 import SIGMA

FAMILIES = ['MobileIE_MSE', 'MobileIE_L1', 'Nano', 'StableLLVE_L1', 'FastDVDnet']
SCENES = ['000', '009', '019', '029']
DATA = BASE / 'results/exploration-r72-data001'
PROCESS = BASE / 'process/exploration-r72'


def stamp():
    return datetime.now().astimezone().isoformat()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def frozen_sources():
    names = ['drv_references_r59.py', 'evaluate_spatial_l1_r69.py', 'response_cue_r45.py',
             'train_exploration_r2.py', 'stablellve_adapted_r65.py', 'stablellve_adapter_r15.py',
             'stablellve_task_r65.py', 'fastdvdnet_adapter_r12.py', 'task_fastdvd_r47.py',
             'multinex_reference_r55.py', 'tiny_adapted_reference_r56.py',
             'exploration_r2_data.py', 'explore_state_r2.py', 'video_metrics.py',
             'half_motion_r4.py', 'aligned_raw_history_r11.py', 'gpu_guard.py',
             'training_form_r41.py', 'cvvdp_adapter_r66.py']
    paths = [BASE / 'scripts' / n for n in names]
    paths += list((BASE / 'scripts').glob('*r72.py'))
    paths += list((ROOT / 'MobileIE/model').glob('*.py'))
    for directory in ['stablellve', 'fastdvdnet', 'multinex']:
        paths += list((ROOT / 'third_party' / directory).glob('*.py'))
    paths += [BASE / 'docs/exploration-r72-reds-reference-protocol.md']
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))}


def check_sources(sources):
    assert all(digest(ROOT / name) == value for name, value in sources.items())


def load_models(seed):
    assert seed in [0, 1] and not torch.cuda.is_initialized()
    models, sources = {}, {}
    models['MobileIE_MSE'], marker = load_frozen(seed)
    sources['MobileIE_MSE'] = dict(checkpoint=marker, parameters=4075, project_updates=8000)
    models['MobileIE_L1'], sources['MobileIE_L1'] = load_final(seed)
    models['Nano'], _, sources['Nano'] = load_network(f'nano_seed{seed}')
    models['StableLLVE_L1'], sources['StableLLVE_L1'] = load_adapted(seed, 'spatial_l1')
    models['FastDVDnet'], _, sources['FastDVDnet'] = load_network(f'fastdvd_seed{seed}')
    expected = dict(zip(FAMILIES, [4075, 4075, 696, 4316259, 2479096]))
    for family, model in models.items():
        model.eval().requires_grad_(False)
        assert sum(p.numel() for p in model.parameters()) == expected[family]
        assert all(torch.isfinite(p).all() for p in model.parameters())
        sources[family].update(parameters=expected[family], project_updates=8000,
                               future_wait_max=2 if family == 'FastDVDnet' else 0,
                               pretraining_REDS_overlap='Not fully verified; project adaptation does not use REDS.')
    return models, sources


@torch.inference_mode()
def infer_raw(family, model, low, device):
    assert family in FAMILIES and low.dtype == np.uint8 and low.shape == (64, 720, 1280, 3)
    contexts = window_indices(len(low), 'window') if family == 'FastDVDnet' else None
    routine = official_inference() if contexts is not None else None
    noise = torch.full((1, 1, 720, 1280), SIGMA / 255., device=device) if contexts is not None else None
    frames = []
    for t in range(64):
        if family == 'StableLLVE_L1':
            value = model(input_tensor(low[t:t + 1], device))
            assert torch.isfinite(value).all()
            frame = output_rgb(value)[0]
        else:
            chosen = low[contexts[t]] if contexts is not None else low[t:t + 1]
            inputs = torch.from_numpy(chosen.transpose(0, 3, 1, 2).copy()).to(device=device, dtype=torch.float32) / 255
            if contexts is not None:
                value = routine.temp_denoise(model, inputs.reshape(1, 15, 720, 1280), noise)
            else:
                value = prediction(model, inputs) if family == 'Nano' else model(inputs).clamp(0, 1)
            assert torch.isfinite(value).all()
            frame = value.clamp(0, 1)[0].permute(1, 2, 0).contiguous().cpu().numpy() * 255
        assert frame.shape == (720, 1280, 3) and frame.dtype == np.float32
        assert np.isfinite(frame).all() and frame.min() >= 0 and frame.max() <= 255
        frames.append(frame)
    return np.stack(frames)


def load_data_index():
    runtime = json.loads((DATA / 'runtime.json').read_text())
    assert runtime['completed_normally'] and runtime['conditions'] == 8
    index = json.loads((DATA / 'index.json').read_text())
    assert len(index['scenes']) == 8 and index['source_fps'] == 24
    for row in index['scenes']:
        for name in ['low', 'gt', 'reference_flow', 'reference_mask']:
            assert digest(ROOT / row[name]['path']) == row[name]['sha256']
    return index


def load_arrays(row):
    return {name: np.load(ROOT / row[name]['path'], mmap_mode='r', allow_pickle=False)
            for name in ['low', 'gt', 'reference_flow', 'reference_mask']}
