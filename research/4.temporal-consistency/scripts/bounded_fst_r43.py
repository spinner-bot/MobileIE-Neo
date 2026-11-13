"""Constrained rational FST prototype, a known rational-activation subfamily.

The two extra parameters and division remain in deployment. Only the spatial
MBR convolutions are folded; no temporal dependency or rational term disappears.
"""
import copy
import json

import torch
from torch import nn

from dynamic_training_r34 import BASE, ROOT, digest
from training_form_r41 import create_model
from model.utils import FST


class BoundedFST(FST):
    def __init__(self, source, scale=1.):
        nn.Module.__init__(self)
        if scale <= 0:
            raise ValueError('Expected a fixed positive feature scale')
        self.block1 = source.block1
        self.weight1, self.weight2, self.bias = source.weight1, source.weight2, source.bias
        self.kappa = nn.Parameter(source.weight1.new_zeros(1))
        self.register_buffer('scale', source.weight1.new_tensor(float(scale)))

    def forward(self, image):
        features = self.block1(image)
        original = (self.weight1*features)*(self.weight2*features)
        denominator = 1+self.kappa*(features/self.scale).square()
        return original/denominator+self.bias


def convert(net, scales=(1., 1.)):
    candidate = copy.deepcopy(net)
    for name, scale in zip(['head', 'body'], scales):
        source = getattr(candidate, name)
        assert not isinstance(source, BoundedFST)
        setattr(candidate, name, BoundedFST(source, scale))
    return candidate


@torch.no_grad()
def project(candidate):
    for name in ['head', 'body']:
        getattr(candidate, name).kappa.clamp_(min=0)


def export(candidate):
    assert not candidate.training, 'Only fixed-statistics eval convolutions can be folded'
    slim = candidate.slim()
    for name in ['head', 'body']:
        source = getattr(candidate, name)
        assert isinstance(source, BoundedFST)
        target = BoundedFST(getattr(slim, name), float(source.scale))
        with torch.no_grad():
            target.kappa.copy_(source.kappa)
        setattr(slim, name, target)
    assert sum(p.numel() for p in slim.parameters()) == 4077
    return slim.eval()


def load_training_anchor(seed):
    assert seed in [0, 1]
    folder = BASE/f'results/exploration-r41-stage002-seed{seed}'
    runtime = json.loads((folder/'runtime.json').read_text())
    assert runtime['completed_normally'] and runtime['model_updates']['expanded_fp32'] == 4000
    marker = json.loads((folder/'expanded_fp32-step4000.json').read_text())
    assert marker['final_fold_gate_passed']
    path = ROOT/marker['training_path']
    assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    assert digest(path) == marker['training_sha256']
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['model'] == 'expanded_fp32' and saved['step'] == 4000 and saved['seed'] == seed
    net, _ = create_model(seed)
    net.load_state_dict(saved['state_dict'], strict=True)
    return net, marker
