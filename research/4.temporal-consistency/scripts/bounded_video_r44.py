"""Common R41 initialization and fixed RMS units for the registered factorial."""
import copy
import json

import torch

from bounded_fst_r43 import convert, export, load_training_anchor, project
from dynamic_training_r34 import BASE, digest
from train_exploration_r2 import new_model

MODELS = ['square_mse', 'square_temporal', 'bounded_mse', 'bounded_temporal']


def make_models(seed):
    gate = BASE/'process/exploration-r43/cpu-gate.json'
    scales_path = BASE/'process/exploration-r43/scale-calibration.json'
    assert json.loads(gate.read_text())['passed']
    calibration = json.loads(scales_path.read_text())
    assert calibration['completed_normally'] and calibration['new_optimizer_updates'] == 0
    row = next(r for r in calibration['seeds'] if r['seed'] == seed)
    scales = tuple(row['statistics'][name]['RMS'] for name in ['head', 'body'])
    original, marker = load_training_anchor(seed)
    assert marker['training_sha256'] == row['training_checkpoint_sha256']
    nets = {name: convert(original, scales) if name.startswith('bounded') else copy.deepcopy(original)
            for name in MODELS}
    return nets, {'source_checkpoint': marker, 'scales': list(scales),
                  'calibration_sha256': digest(scales_path), 'optimizer': 'Common fresh Adam; old moments not restored',
                  'parameters': {name: {'total': sum(p.numel() for p in net.parameters()),
                                      'active': sum(p.numel() for p in net.parameters() if p.requires_grad)}
                                 for name, net in nets.items()}}


def folded(net, model):
    assert not net.training
    result = export(net) if model.startswith('bounded') else net.slim().eval()
    assert sum(p.numel() for p in result.parameters()) == (4077 if model.startswith('bounded') else 4075)
    return result


def project_if_bounded(net, model):
    if model.startswith('bounded'):
        project(net)


def deployment_model(model):
    assert model in MODELS
    original = new_model(torch.device('cpu'))
    return (convert(original) if model.startswith('bounded') else original).eval()


def weight_for(model):
    assert model in MODELS
    return .25 if model.endswith('_temporal') else 0.
