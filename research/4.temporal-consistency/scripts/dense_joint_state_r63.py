"""R63 controlled question: unchanged dense backbone, active spatial/causal routes.

The entire temporal mechanism is reused from R33; it is not a new method.
Only spatial MBR branches fold at deployment. History and its cost remain.
"""
import copy
import json

import torch
from torch import nn

from budget_causal_r33 import BudgetCausal
from dynamic_training_r34 import BASE, ROOT, digest
from training_form_r41 import create_model


class DenseJointState(BudgetCausal):
    def __init__(self, backbone, temporal=True):
        # Do not call the R33 constructor: it replaces the original dense body.
        nn.Module.__init__(self)
        self.backbone = copy.deepcopy(backbone)
        self.project = nn.Conv2d(12, 8, 1)
        self.mix = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, groups=16),
                                 nn.Conv2d(16, 8, 1), nn.PReLU(8))
        self.field = nn.Conv2d(8, 3, 1)
        self.readout = nn.Conv2d(8, 12, 1)
        nn.init.zeros_(self.field.weight)
        nn.init.zeros_(self.field.bias)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.temporal = temporal
        self._state = None


def make_models(seed):
    marker_path = BASE/f'results/exploration-r44-stage001-seed{seed}/square_mse-step4000.json'
    marker = json.loads(marker_path.read_text())
    assert marker['final_export_gate_passed'] and marker['deployment_parameters'] == 4075
    path = ROOT/marker['training_path']
    assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    assert digest(path) == marker['training_sha256']
    anchor, _ = create_model(seed)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    assert checkpoint['model'] == 'square_mse' and checkpoint['seed'] == seed
    assert checkpoint['new_step'] == 4000 and checkpoint['inherited_updates'] == 4000
    anchor.load_state_dict(checkpoint['state_dict'], strict=True)
    anchor.requires_grad_(True)
    anchor.tail_warm.requires_grad_(False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(6300+seed)
        causal = DenseJointState(anchor)
    spatial = copy.deepcopy(causal)
    spatial.temporal = False
    nets = {'plain': anchor, 'spatial': spatial, 'causal': causal}
    info = {'source_marker': str(marker_path.relative_to(ROOT)),
            'training_checkpoint_sha256': marker['training_sha256'],
            'inherited_updates_each': 8000, 'new_branch_seed': 6300+seed,
            'optimizer_state_inherited': False,
            'parameters': {name: {'total': sum(p.numel() for p in net.parameters()),
                                  'active': sum(p.numel() for p in net.parameters() if p.requires_grad)}
                           for name, net in nets.items()}}
    return nets, info


def prediction(net, clip, history=None):
    if not isinstance(net, DenseJointState):
        return torch.stack([net(frame) for frame in clip.unbind(1)], dim=1)
    state, outputs = None, []
    for frame in clip.unbind(1):
        output, state = net(frame, state, history=history)
        outputs.append(output)
    return torch.stack(outputs, dim=1)


def folded(net):
    if net.training:
        raise ValueError('Export requires eval mode')
    if not isinstance(net, DenseJointState):
        return net.slim().eval()
    result = copy.deepcopy(net)
    result.reset()
    result.backbone = net.backbone.slim().eval()
    dense = result.backbone.body.block1
    if not (isinstance(dense, nn.Conv2d) and dense.groups == 1
            and dense.in_channels == dense.out_channels == 12 and dense.kernel_size == (3, 3)):
        raise ValueError('Original dense spatial body was not preserved')
    return result.eval()
