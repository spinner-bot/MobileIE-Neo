"""Matched original backbone controls; fixed deployed recurrence, zero new weights."""
import copy
import json

import torch

from dynamic_training_r34 import BASE, ROOT, digest
from mixed_anchor_r40 import prediction
from response_cue_r45 import align_quantized
from training_form_r41 import create_model

MODELS = ['plain', 'joint']


def initial_models(seed):
    folder = BASE/f'results/exploration-r44-stage001-seed{seed}'
    runtime = json.loads((folder/'runtime.json').read_text())
    assert runtime['completed_normally'] and runtime['model_updates']['square_mse'] == 4000
    marker = json.loads((folder/'square_mse-step4000.json').read_text())
    assert marker['final_export_gate_passed'] and marker['inherited_updates'] == 4000
    path = ROOT/marker['training_path']
    assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    assert digest(path) == marker['training_sha256']
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['model'] == 'square_mse' and saved['new_step'] == 4000
    assert saved['inherited_updates'] == 4000 and saved['seed'] == seed
    source, _ = create_model(seed)
    source.load_state_dict(saved['state_dict'], strict=True)
    assert sum(p.numel() for p in source.parameters()) == 49978
    assert sum(p.numel() for p in source.parameters() if p.requires_grad) == 47239
    return {'plain': source, 'joint': copy.deepcopy(source)}, marker


def raw_prediction(model, low):
    unclipped = prediction(model, low)
    return unclipped, unclipped.clamp(0, 1)


def rollout(raw, fields):
    """Full history gradient; only motion fields/masks are outside autograd."""
    assert len(fields) == raw.shape[1]-1
    outputs = [raw[:, 0]]
    for t, (flow, mask) in enumerate(fields, 1):
        aligned, valid = align_quantized(outputs[-1], raw[:, t], flow, mask)
        output = .5*raw[:, t]+.5*aligned
        outputs.append(torch.where(valid, output, raw[:, t]))
    return torch.stack(outputs, dim=1)
