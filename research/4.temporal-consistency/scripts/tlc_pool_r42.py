"""Minimal fixed-window TLC adapter; known algorithm, no BasicSR dependency.

Source algorithm: megvii-research/TLC local_arch.py at
8e87983d917b387c63fdaa3425eb1a677aa39af6 (MIT-licensed project).
Use valid local means followed by replicated output borders, not input padding.
"""
import copy
import json

import torch
from torch import nn
from torch.nn import functional as F

from dynamic_training_r34 import BASE, ROOT, digest
from train_exploration_r2 import new_model


class FixedLocalAverage(nn.Module):
    def __init__(self, window=192):
        super().__init__()
        if not isinstance(window, int) or window < 1:
            raise ValueError('Expected positive integer window')
        self.window = window

    def forward(self, x):
        if x.ndim != 4 or min(x.shape[-2:]) < 1:
            raise ValueError('Expected nonempty BCHW tensor')
        height, width = x.shape[-2:]
        kh, kw = min(height, self.window), min(width, self.window)
        integral = F.pad(x.cumsum(-1).cumsum(-2), (1, 0, 1, 0))
        local = (integral[..., kh:, kw:] + integral[..., :-kh, :-kw]
                 - integral[..., kh:, :-kw] - integral[..., :-kh, kw:])/(kh*kw)
        dh, dw = height-local.shape[-2], width-local.shape[-1]
        return F.pad(local, (dw//2, (dw+1)//2, dh//2, (dh+1)//2), mode='replicate')


def convert(net):
    if not isinstance(net.att[0], nn.AdaptiveAvgPool2d) or net.att[0].output_size != 1:
        raise ValueError('Expected the original global channel-pooling node')
    local = copy.deepcopy(net)
    local.att[0] = FixedLocalAverage(192)
    local.eval()
    assert sum(p.numel() for p in local.parameters()) == 4075
    assert net.state_dict().keys() == local.state_dict().keys()
    assert all(torch.equal(value, local.state_dict()[name]) for name, value in net.state_dict().items())
    return local


def load_anchor(seed):
    assert seed in [0, 1] and not torch.cuda.is_initialized()
    folder = BASE/f'results/exploration-r41-stage002-seed{seed}'
    runtime = json.loads((folder/'runtime.json').read_text())
    assert runtime['completed_normally'] and runtime['total_updates'] == 8000
    marker = json.loads((folder/'expanded_fp32-step4000.json').read_text())
    assert marker['final_fold_gate_passed'] and marker['final_eval_fold_max_abs_delta'] <= 1e-4
    assert digest(ROOT/marker['path']) == marker['sha256']
    saved = torch.load(ROOT/marker['path'], map_location='cpu', weights_only=False)
    assert saved['model'] == 'expanded_fp32' and saved['seed'] == seed and saved['step'] == 4000
    net = new_model(torch.device('cpu'))
    net.load_state_dict(saved['state_dict'], strict=True)
    return net.eval(), marker
