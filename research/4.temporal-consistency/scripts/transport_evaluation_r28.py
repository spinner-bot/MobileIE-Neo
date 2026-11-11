"""Causal finite-feature inference and fixed end-point model loading for R28."""
import json
import torch

from exploration_r2_data import ROOT
from explore_temporal_controls import sha256
from loss_controls_r26 import create_model as r26_model
from train_exploration_r2 import new_model
from transport_fst_r27 import MODES, TransportFST

BASE = ROOT/'research/4.temporal-consistency'
EXTERNAL = 'r26_pre_mse_spatial'


def load_models(mode, device):
    if mode not in ['copy', 'post_transport']:
        raise ValueError('convergence scope permits only copy and post_transport evaluation')
    models, markers = [], []
    for objective in ['charb', 'mse']:
        for seed in [0, 1]:
            folder = (BASE/'results'/f'exploration-r26-seed{seed}'/('plain-'+objective) if mode == 'copy'
                      else BASE/'results'/f'exploration-r28-seed{seed}'/(mode+'-'+objective))
            runtime = json.loads((folder/'runtime.json').read_text())
            assert runtime['final_step'] == 4000 and runtime['new_updates'] == 4000
            marker = json.loads((folder/'checkpoint-step4000.json').read_text())
            assert sha256(ROOT/marker['path']) == marker['sha256']
            saved = torch.load(ROOT/marker['path'], map_location=device, weights_only=False)
            assert saved['step'] == 4000
            net = TransportFST(new_model(device), mode).to(device)
            net.load_state_dict(saved['model'], strict=True)
            models.append((mode, objective, seed, net.eval()))
            markers.append({'mode': mode, 'objective': objective, 'train_seed': seed, **marker,
                            'reused_R26_training': mode == 'copy',
                            'training_config_sha256': sha256(folder/'config.json')})
    if mode == 'copy':
        for seed in [0, 1]:
            folder = BASE/'results'/f'exploration-r26-seed{seed}'/'pre_fixedhalf-mse'
            marker = json.loads((folder/'checkpoint-step4000.json').read_text())
            assert sha256(ROOT/marker['path']) == marker['sha256']
            saved = torch.load(ROOT/marker['path'], map_location=device, weights_only=False)
            assert saved['step'] == 4000
            net = r26_model('pre_fixedhalf', device)
            net.load_state_dict(saved['model'], strict=True)
            models.append((EXTERNAL, 'mse', seed, net.backbone.eval()))
            markers.append({'mode': EXTERNAL, 'objective': 'mse', 'train_seed': seed, **marker,
                            'training_config_sha256': sha256(folder/'config.json')})
    return models, markers


def output_keys(mode, history, reset, has_events, fusion):
    temporal = mode not in ['copy', EXTERNAL]
    model_reset = bool(temporal and history == 'causal' and reset and has_events)
    output_reset = bool(fusion != 'raw' and reset and has_events)
    return model_reset, (model_reset, fusion, output_reset)


@torch.inference_mode()
def infer(net, inputs, data, mode, history, resets):
    if history not in ['causal', 'current']:
        raise ValueError(history)
    spatial = mode in ['copy', EXTERNAL] or history == 'current'
    backbone = net if mode == EXTERNAL else net.backbone
    if not spatial:
        net.reset()
    outputs = []
    for t, current in enumerate(inputs):
        if spatial:
            value = backbone(current[None])[0]
        else:
            if resets is not None and resets[t]:
                net.reset()
            flow = None if t == 0 else data['half_flow'][t-1][None]
            mask = None if t == 0 else data['half_mask'][t-1][None]
            value = net.step(current[None], flow, mask, history='causal')[0]
        outputs.append(value)
    return torch.stack(outputs)
