"""R73 fixed pilot: native empirical moments inside the first quadratic FST.

These are mixture moments, not calibrated noise or exact posterior uncertainty.
rho=0/1 recover the elementary pre/post quadratic-averaging identities.
"""
import copy
import json

import torch
from torch import nn
import torch.nn.functional as F

from budget_causal_r33 import transport
from dynamic_training_r34 import BASE, ROOT, digest
from training_form_r41 import create_model

MODES = ['spatial', 'mean', 'moment']
PARAMETERS = {'spatial': 4562, 'mean': 4550, 'moment': 4562}
PROTOCOL = BASE/'docs/exploration-r73-native-moment-core-protocol.md'


def mix_moments(current, previous, gate):
    mean, second = previous
    mixed = current + gate*(mean-current)
    square = current.square()
    second = square + gate*(second-square)
    return mixed, second, (second-mixed.square()).clamp_min(0)


class NativeMomentCore(nn.Module):
    def __init__(self, backbone, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError('Unknown registered R73 mode')
        self.backbone = copy.deepcopy(backbone)
        self.mode = mode
        self.field = nn.Sequential(nn.Conv2d(24, 24, 3, padding=1, groups=24),
                                   nn.Conv2d(24, 8, 1), nn.PReLU(8), nn.Conv2d(8, 3, 1))
        nn.init.zeros_(self.field[-1].weight)
        nn.init.zeros_(self.field[-1].bias)
        self.rho = None if mode == 'mean' else nn.Parameter(torch.zeros(1, 12, 1, 1))
        self._state = None

    def forward(self, image, state=None, history=None):
        if image.ndim != 4 or image.shape[1] != 3 or min(image.shape[-2:]) < 4:
            raise ValueError('Expected BCHW RGB, minimum side4')
        history = ('current' if self.mode == 'spatial' else 'causal') if history is None else history
        if history not in ['current', 'causal'] or (self.mode == 'spatial' and history != 'current'):
            raise ValueError('Invalid history route')
        net = self.backbone
        current = net.head.block1(image)
        if state is not None:
            if len(state) != 2:
                raise ValueError('Expected a mean/optional-second moment state')
            expected_second = self.mode != 'mean'
            if any(value is not None and (value.shape != current.shape or value.device != current.device
                                         or value.dtype != current.dtype) for value in state):
                raise ValueError('State shape/device/dtype changed; reset sequence')
            if state[0] is None or (state[1] is not None) != expected_second:
                raise ValueError('State channel layout does not match model mode')
        mean = current if history == 'current' or state is None else state[0]
        second = current.square() if history == 'current' or state is None else state[1]
        fields = self.field(torch.cat([F.avg_pool2d(current, 4, 4, ceil_mode=True),
                                       F.avg_pool2d(mean, 4, 4, ceil_mode=True)], dim=1))
        offset = F.interpolate(8*fields[:, :2].tanh(), size=current.shape[-2:],
                               mode='bilinear', align_corners=False)
        gate = F.interpolate(fields[:, 2:3].sigmoid(), size=current.shape[-2:],
                             mode='bilinear', align_corners=False)
        if self.mode == 'mean':
            moved, valid = transport(mean, current, offset)
            updated = current + gate*valid.to(current.dtype)*(moved-current)
            next_state = (updated, None)
        else:
            moved, valid = transport(torch.cat([mean, second], dim=1),
                                     torch.cat([current, current.square()], dim=1), offset)
            updated, second, dispersion = mix_moments(current, moved.chunk(2, dim=1),
                                                       gate*valid.to(current.dtype))
            next_state = (updated, second)
        # Preserve the original two sequential vector multiplies, with a moment
        # correction added in the same output units; spatial MBR alone folds.
        head = (net.head.weight1*updated)*(net.head.weight2*updated)+net.head.bias
        if self.rho is not None:
            head = head + (net.head.weight1*dispersion)*(net.head.weight2*self.rho)
        body = net.body(head)
        channel = net.att(body)
        spatial = net.att1((channel*body).max(dim=1, keepdim=True).values)
        return net.tail((spatial*channel)*body), next_state

    def reset(self):
        self._state = None

    def step(self, image, history=None):
        expected = (image.shape[0], 12, *image.shape[-2:])
        if self._state is not None and any(value is not None and
                (value.shape != expected or value.device != image.device or value.dtype != image.dtype)
                for value in self._state):
            self.reset()
        output, state = self(image, self._state, history=history)
        self._state = tuple(None if value is None else value.detach() for value in state)
        return output

    def state_storage_bytes(self):
        return 0 if self._state is None else sum(value.untyped_storage().nbytes()
                                                for value in self._state if value is not None)


def make_models(seed):
    marker_path = BASE/f'results/exploration-r44-stage001-seed{seed}/square_mse-step4000.json'
    marker = json.loads(marker_path.read_text())
    assert marker['final_export_gate_passed'] and marker['deployment_parameters'] == 4075
    path = ROOT/marker['training_path']
    assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    assert digest(path) == marker['training_sha256']
    anchor, _ = create_model(seed)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    assert checkpoint['new_step'] == checkpoint['inherited_updates'] == 4000
    assert checkpoint['model'] == 'square_mse' and checkpoint['seed'] == seed
    anchor.load_state_dict(checkpoint['state_dict'], strict=True)
    anchor.requires_grad_(True); anchor.tail_warm.requires_grad_(False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7300+seed)
        moment = NativeMomentCore(anchor, 'moment')
    spatial = copy.deepcopy(moment); spatial.mode = 'spatial'
    mean = copy.deepcopy(moment); mean.mode = 'mean'; mean.rho = None
    nets = {'spatial': spatial, 'mean': mean, 'moment': moment}
    info = dict(source_marker=str(marker_path.relative_to(ROOT)),
                source_training_sha256=marker['training_sha256'], inherited_updates=8000,
                optimizer_inherited=False, field_seed=7300+seed,
                source_plain12000=str((BASE/f'results/exploration-r64-stage001-seed{seed}/plain-step4000.json').relative_to(ROOT)),
                deployment_parameters=PARAMETERS)
    return nets, info


def prediction(net, clip, history=None):
    state, outputs = None, []
    for frame in clip.unbind(1):
        output, state = net(frame, state, history=history)
        outputs.append(output)
    return torch.stack(outputs, dim=1)


def folded(net):
    assert not net.training
    result = copy.deepcopy(net); result.reset()
    result.backbone = net.backbone.slim().eval()
    assert isinstance(result.backbone.body.block1, nn.Conv2d)
    assert result.backbone.body.block1.groups == 1
    assert sum(p.numel() for p in result.parameters()) == PARAMETERS[result.mode]
    return result.eval()
