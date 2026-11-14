"""Frozen spatial responses and controlled RGB recursion, not a proven method."""
import json
import math

import torch
from torch import nn
import torch.nn.functional as F

from budget_causal_r33 import transport
from dynamic_training_r34 import BASE, ROOT, digest
from train_exploration_r2 import new_model

CONTROLS = ['fixed', 'scalar', 'rgb', 'response']


def load_frozen(seed):
    assert seed in [0, 1]
    marker_path = BASE/f'results/exploration-r44-stage001-seed{seed}/square_mse-step4000.json'
    marker = json.loads(marker_path.read_text())
    assert marker['final_export_gate_passed']
    path = ROOT/marker['path']
    assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    assert digest(path) == marker['sha256']
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['model'] == 'square_mse' and saved['seed'] == seed
    assert saved['new_step'] == saved['inherited_updates'] == 4000
    net = new_model(torch.device('cpu')).eval()
    net.load_state_dict(saved['state_dict'], strict=True)
    net.requires_grad_(False)
    assert sum(p.numel() for p in net.parameters()) == 4075
    return net, marker


class FrozenResponses(nn.Module):
    """Read two scalar response maps in the unchanged original forward call."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.eval().requires_grad_(False)
        self._captures = []
        self._handles = []
        for fst in [self.backbone.head, self.backbone.body]:
            def capture(module, inputs, value, fst=fst):
                rms = torch.linalg.vector_norm(value, dim=1, keepdim=True)/math.sqrt(value.shape[1])
                self._captures.append(torch.log1p(2*(fst.weight1*fst.weight2).abs()*rms))
            self._handles.append(fst.block1.register_forward_hook(capture))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, image):
        assert not self.backbone.training
        self._captures.clear()
        try:
            raw = self.backbone(image)
            assert len(self._captures) == 2
            cues = torch.cat(self._captures, dim=1)
            return raw, cues
        finally:
            self._captures.clear()


def align_quantized(previous, current, flow, mask):
    """OpenCV 1/32 bilinear coordinates, original bounds, differentiable RGB."""
    b, _, h, w = current.shape
    assert previous.shape == current.shape and flow.shape == (b, 2, h, w)
    assert mask.shape == (b, 1, h, w)
    flow = flow.detach()
    y = torch.arange(h, device=flow.device, dtype=flow.dtype)[None, :, None]
    x = torch.arange(w, device=flow.device, dtype=flow.dtype)[None, None, :]
    sx, sy = x+flow[:, 0], y+flow[:, 1]
    valid = ((sx >= 0) & (sx <= w-1) & (sy >= 0) & (sy <= h-1))[:, None] & mask.bool()
    quantized = torch.stack([torch.round(sx*32)/32-x, torch.round(sy*32)/32-y], dim=1)
    moved, _ = transport(previous, current, quantized)
    return torch.where(valid, moved, current), valid


def gate_features(current, aligned, flow, valid, cues, control):
    assert control in ['rgb', 'response']
    current_mean = current.mean(1, keepdim=True)
    previous_mean = aligned.mean(1, keepdim=True)
    innovation = (current-aligned).abs().mean(1, keepdim=True)
    motion = torch.linalg.vector_norm(flow, dim=1, keepdim=True)/max(current.shape[-2:])
    dx = F.pad((current[..., 1:]-current[..., :-1]).abs().mean(1, keepdim=True), (1, 0, 0, 0), mode='replicate')
    dy = F.pad((current[..., 1:, :]-current[..., :-1, :]).abs().mean(1, keepdim=True), (0, 0, 1, 0), mode='replicate')
    common = [current_mean, previous_mean, innovation, motion, valid.to(current.dtype), .5*(dx+dy)]
    if control == 'rgb':
        extra = [(current-current_mean).abs().mean(1, keepdim=True),
                 (aligned-previous_mean).abs().mean(1, keepdim=True)]
    else:
        assert cues.shape == (current.shape[0], 2, *current.shape[-2:])
        extra = list(cues.split(1, dim=1))
    return torch.cat(common+extra, dim=1)


class RecursiveControl(nn.Module):
    """Functional state: preceding raw is kept by the caller for own-raw flow."""
    def __init__(self, control, seed):
        super().__init__()
        assert control in CONTROLS and seed in [0, 1]
        self.control = control
        if control == 'scalar':
            self.logit = nn.Parameter(torch.zeros(()))
        elif control in ['rgb', 'response']:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(4500+seed)
                self.gate = nn.Sequential(nn.Conv2d(8, 4, 1), nn.ReLU(), nn.Conv2d(4, 1, 1))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
        expected = {'fixed': 0, 'scalar': 1, 'rgb': 41, 'response': 41}[control]
        assert sum(p.numel() for p in self.parameters()) == expected

    def forward(self, current, previous=None, flow=None, mask=None, cues=None):
        shape = (current.shape[0], 1, *current.shape[-2:])
        if previous is None:
            return current, current.new_zeros(shape), torch.zeros(shape, dtype=torch.bool, device=current.device)
        aligned, valid = align_quantized(previous, current, flow, mask)
        if self.control == 'fixed':
            alpha = current.new_full(shape, .5)
        elif self.control == 'scalar':
            alpha = self.logit.sigmoid().expand(shape)
        else:
            features = gate_features(current, aligned, flow, valid, cues, self.control)
            alpha = self.gate(features).sigmoid()
        output = (1-alpha)*current+alpha*aligned
        return torch.where(valid, output, current), alpha, valid
