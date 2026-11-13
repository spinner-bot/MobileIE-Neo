"""R74 fixed RGB-state pilot, not a claimed novel flow estimator or phone result."""
import copy
import json

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from budget_causal_r33 import transport
from dynamic_training_r34 import BASE, ROOT, digest
from half_motion_r4 import half_forward_backward
from train_exploration_r2 import new_model

PROTOCOL = BASE/'docs/exploration-r74-motion-supervision-protocol.md'
MODES = ['task', 'proxy']


def correlation(current, previous, radius):
    if current.shape != previous.shape:
        raise ValueError('Correlation feature shapes differ')
    height, width = current.shape[-2:]
    padded = F.pad(previous, (radius, radius, radius, radius))
    # Fixed offsets, including zero padding; not a full-resolution search.
    return torch.cat([(current*padded[:, :, y:y+height, x:x+width]).mean(1, keepdim=True)
                      for y in range(2*radius+1) for x in range(2*radius+1)], dim=1)


class MotionTransport(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.PReLU(8),
                                     nn.Conv2d(8, 8, 3, padding=1), nn.PReLU(8))
        self.coarse = nn.Sequential(nn.Conv2d(97, 16, 3, padding=1), nn.PReLU(16),
                                    nn.Conv2d(16, 3, 3, padding=1))
        self.fine = nn.Sequential(nn.Conv2d(43, 16, 3, padding=1), nn.PReLU(16),
                                  nn.Conv2d(16, 3, 3, padding=1))
        for head in [self.coarse, self.fine]:
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)

    def forward(self, previous_raw, current_raw):
        if previous_raw.shape != current_raw.shape or current_raw.ndim != 4 or current_raw.shape[1] != 3:
            raise ValueError('Expected matching BCHW RGB raw outputs')
        if min(current_raw.shape[-2:]) < 16:
            raise ValueError('R74 motion encoder needs image sides at least16')
        # RGB intensities are [0,1]; flow remains in native pixel units throughout.
        previous = self.encoder(F.avg_pool2d(previous_raw, 4, 4, ceil_mode=True))
        current = self.encoder(F.avg_pool2d(current_raw, 4, 4, ceil_mode=True))
        pc = F.avg_pool2d(previous, 4, 4, ceil_mode=True)
        cc = F.avg_pool2d(current, 4, 4, ceil_mode=True)
        coarse = self.coarse(torch.cat([cc, pc, correlation(cc, pc, 4)], dim=1))
        flow = F.interpolate(64*coarse[:, :2].tanh(), size=current.shape[-2:],
                             mode='bilinear', align_corners=False)
        confidence = F.interpolate(coarse[:, 2:3], size=current.shape[-2:],
                                   mode='bilinear', align_corners=False)
        moved, _ = transport(previous, current, flow/4)
        fine = self.fine(torch.cat([current, moved, correlation(current, moved, 2), flow], dim=1))
        return flow+8*fine[:, :2].tanh(), confidence+fine[:, 2:3]


def fuse(current_raw, previous_fused, fields, zero_flow=False):
    flow, logit = fields
    size = current_raw.shape[-2:]
    flow = F.interpolate(flow, size=size, mode='bilinear', align_corners=False)
    logit = F.interpolate(logit, size=size, mode='bilinear', align_corners=False)
    if zero_flow:
        flow = torch.zeros_like(flow)
    moved, valid = transport(previous_fused, current_raw, flow)
    coefficient = .5*logit.sigmoid()*valid.to(current_raw.dtype)
    return current_raw+coefficient*(moved-current_raw), valid


def sequence(net, raw, zero_flow=False):
    if raw.ndim != 5 or raw.shape[1] < 2 or raw.shape[2] != 3:
        raise ValueError('Expected BTCHW RGB with at least two frames')
    outputs, fields = [raw[:, 0]], []
    for t in range(1, raw.shape[1]):
        field = net(raw[:, t-1], raw[:, t])
        value, _ = fuse(raw[:, t], outputs[-1], field, zero_flow=zero_flow)
        outputs.append(value); fields.append(field)
    return torch.stack(outputs, dim=1), fields


def make_models(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7400+seed); first = MotionTransport()
    return {'task': first, 'proxy': copy.deepcopy(first)}


def load_spatial(seed):
    marker_path = BASE/f'results/exploration-r44-stage001-seed{seed}/square_mse-step4000.json'
    marker = json.loads(marker_path.read_text())
    assert marker['final_export_gate_passed'] and marker['deployment_parameters'] == 4075
    path = ROOT/marker['path']
    assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo') and digest(path) == marker['sha256']
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    assert checkpoint['seed'] == seed and checkpoint['model'] == 'square_mse'
    net = new_model(torch.device('cpu'))
    net.load_state_dict(checkpoint['state_dict'], strict=True)
    return net.eval().requires_grad_(False), dict(marker=str(marker_path.relative_to(ROOT)),
            checkpoint_sha256=marker['sha256'], frozen=True, project_updates=8000, parameters=4075)


def frozen_raw(net, low):
    assert not net.training and not any(p.requires_grad for p in net.parameters())
    with torch.no_grad():
        return torch.stack([net(frame).clamp(0, 1) for frame in low.unbind(1)], dim=1)


def teacher_proxy(gt):
    """CPU training GT motion proxy; not ground-truth physical flow labels."""
    if gt.device.type != 'cpu':
        raise ValueError('Prepare proxy targets on CPU before CUDA work')
    values = gt.detach().numpy().transpose(0, 1, 3, 4, 2)*255
    flows, masks = [], []
    for sample in values:
        pairs = [half_forward_backward(sample[t-1], sample[t]) for t in range(1, len(sample))]
        flows.append(np.stack([p[0].transpose(2, 0, 1) for p in pairs]))
        masks.append(np.stack([p[1][None].astype(np.float32) for p in pairs]))
    shape = (gt.shape[0], gt.shape[1]-1)
    flow = torch.from_numpy(np.stack(flows).copy()).flatten(0, 1)
    mask = torch.from_numpy(np.stack(masks).copy()).flatten(0, 1)
    size = ((gt.shape[-2]+3)//4, (gt.shape[-1]+3)//4)
    return (F.interpolate(flow, size=size, mode='bilinear', align_corners=False).unflatten(0, shape),
            F.avg_pool2d(mask, 4, 4, ceil_mode=True).unflatten(0, shape))


def proxy_loss(fields, target):
    flow = torch.stack([p[0] for p in fields], dim=1)
    logit = torch.stack([p[1] for p in fields], dim=1)
    desired, coverage = target
    if flow.shape != desired.shape or logit.shape != coverage.shape:
        raise ValueError('Motion supervision shape mismatch')
    motion = ((flow-desired).abs()*coverage).sum()/(2*coverage.sum().clamp_min(1))/4
    confidence = F.binary_cross_entropy_with_logits(logit, coverage)
    return motion+confidence, dict(proxy_native_abs_error=4*float(motion.detach()),
                                  coverage_BCE=float(confidence.detach()),
                                  accepted_proxy_fraction=float(coverage.detach().mean()))
