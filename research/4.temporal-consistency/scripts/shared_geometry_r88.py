"""Frozen restoration, earlier patch descriptors and a fixed causal local solver.

The null-space constraint is on restoration kernel rows, not RGB/image independence.
Generic shared heads, geometric supervision and local Gauss-Newton have prior art.
"""
import copy
import math

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from budget_causal_r33 import transport
from aligned_raw_history_r11 import align_history
from aggregation_controls_r20 import blend
from video_metrics import warp_previous

MODES = ('plain', 'complement')


def normalize(value):
    return F.normalize(value, dim=1, eps=1e-6)


def stem(backbone, image):
    with torch.no_grad():
        return backbone.head.block1[1](backbone.head.block1[0](image))


def capture(backbone, image):
    values = []
    handle = backbone.head.block1[1].register_forward_hook(
        lambda module, inputs, output: values.append(output.detach()))
    try:
        with torch.no_grad():
            raw = backbone(image)
    finally:
        handle.remove()
    assert len(values) == 1
    return raw, values[0]


class Descriptor(nn.Module):
    def __init__(self, initial, basis=None):
        super().__init__()
        self.register_buffer('basis', basis)
        self.weight = nn.Parameter(initial.clone())

    def kernel(self):
        kernel = self.weight if self.basis is None else self.weight @ self.basis.T
        return kernel.reshape(4, 12, 3, 3)

    def forward(self, feature):
        value = F.conv2d(feature, self.kernel(), stride=4, padding=1)
        return normalize(value-value.mean((-2, -1), keepdim=True))

    def folded(self):
        result = copy.deepcopy(self)
        kernel = self.kernel().detach().reshape(4, 108)
        result.basis = None
        result.weight = nn.Parameter(kernel.clone(), requires_grad=False)
        return result.eval()


def make_heads(backbone, seed):
    rows = backbone.head.block1[-1].weight.detach().cpu().double().reshape(12, 108)
    _, singular, vh = torch.linalg.svd(rows, full_matrices=True)
    rank = int((singular > singular.max()*1e-6).sum())
    basis = vh[rank:].T.float().contiguous()
    assert basis.shape[1] > 0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8800+seed)
        coefficients = torch.randn(4, basis.shape[1])/math.sqrt(basis.shape[1])
    effective = coefficients @ basis.T
    return {'plain': Descriptor(effective), 'complement': Descriptor(coefficients, basis)}, rank


def gradient(value):
    padded = F.pad(value, (1, 1, 1, 1), mode='replicate')
    return ((padded[:, :, 1:-1, 2:]-padded[:, :, 1:-1, :-2])*.5,
            (padded[:, :, 2:, 1:-1]-padded[:, :, :-2, 1:-1])*.5)


def local_flow(previous, current):
    """Backward descriptor-grid displacement; fixed 3 levels x 5 updates."""
    assert previous.shape == current.shape and previous.shape[1] == 4
    pyramids = [[previous], [current]]
    for pyramid in pyramids:
        for _ in range(2):
            pyramid.append(normalize(F.avg_pool2d(pyramid[-1], 2, 2, ceil_mode=True)))
    flow = None
    for level in (2, 1, 0):
        p, c = pyramids[0][level], pyramids[1][level]
        height, width = c.shape[-2:]
        if flow is None:
            flow = c.new_zeros((len(c), 2, height, width))
        else:
            flow = (2*F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=False))[:, :, :height, :width]
        gx, gy = gradient(p)
        for _ in range(5):
            moved, valid = transport(p, c, flow)
            dx, _ = transport(gx, torch.zeros_like(gx), flow)
            dy, _ = transport(gy, torch.zeros_like(gy), flow)
            residual = (moved-c)*valid
            dx, dy = dx*valid, dy*valid
            def window(value):
                return F.avg_pool2d(value.sum(1, keepdim=True), 5, 1, padding=2)
            xx, yy, xy = window(dx*dx), window(dy*dy), window(dx*dy)
            bx, by = window(dx*residual), window(dy*residual)
            damping = .001*(xx+yy)+.0001
            xx, yy = xx+damping, yy+damping
            determinant = (xx*yy-xy.square()).clamp_min(1e-12)
            delta = torch.cat([(yy*bx-xy*by)/determinant,
                               (xx*by-xy*bx)/determinant], dim=1).clamp(-1, 1)
            flow = (flow-delta).clamp(-16/(2**level), 16/(2**level))
    return flow


def geometry_loss(descriptors, displacement):
    previous, current = descriptors.chunk(2)
    axis = torch.tensor([8, 12, 16, 20, 24], device=descriptors.device)
    y, x = axis.repeat_interleave(5), axis.repeat(5)
    query = current[:, :, y, x].permute(0, 2, 1)
    candidates = torch.stack([previous[:, :, y+dy, x+dx].permute(0, 2, 1)
                              for dy in range(-4, 5) for dx in range(-4, 5)], dim=2)
    logits = (query[:, :, None]*candidates).sum(-1)/.1
    offsets = torch.tensor([(dx, dy) for dy in range(-4, 5) for dx in range(-4, 5)],
                           dtype=displacement.dtype, device=descriptors.device)
    labels = (1-(offsets[None]-displacement[:, None]).abs()).clamp_min(0).prod(-1)
    assert torch.allclose(labels.sum(-1), torch.ones_like(labels[:, 0]), atol=1e-6)
    ce = -(labels[:, None]*logits.log_softmax(-1)).sum(-1).mean()
    flow = local_flow(previous, current)
    mse = (flow[:, :, 8:-8, 8:-8]-displacement[:, :, None, None]).square().mean()/16
    return ce+mse, dict(CE=float(ce.detach()), flow_MSE_over16=float(mse.detach()))


def update(head, optimizer, features, displacement):
    optimizer.zero_grad(set_to_none=True)
    loss, record = geometry_loss(head(features), displacement)
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in head.parameters())
    norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
    assert bool(torch.isfinite(norm)) and float(norm) > 0
    optimizer.step()
    assert all(bool(torch.isfinite(p).all()) for p in head.parameters())
    return dict(record, loss=float(loss.detach()), preclip_gradient_norm=float(norm))


def native_flow(field, shape):
    quarter = field.detach().cpu().numpy()[0].transpose(1, 2, 0)
    height, width = shape
    x, y = np.meshgrid(np.arange(width, dtype=np.float32)/4,
                       np.arange(height, dtype=np.float32)/4)
    return 4*cv2.remap(quarter, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def correspondence(previous, current, shape):
    backward = native_flow(local_flow(previous, current), shape)
    forward = native_flow(local_flow(current, previous), shape)
    sampled, valid = warp_previous(forward, backward)
    error = np.linalg.norm(backward+sampled, axis=-1)
    mask = valid & (error <= 1+.05*(np.linalg.norm(backward, axis=-1)+np.linalg.norm(sampled, axis=-1)))
    return backward, mask


class CausalCore:
    """Descriptor history is unfused; each RGB route has its own causal history."""
    def __init__(self, routes=('true',)):
        assert set(routes) <= {'true', 'zero'} and len(set(routes)) == len(routes)
        self.routes = routes
        self.previous_descriptor = None
        self.fused = {}
        self.pairs = 0

    def step(self, raw, descriptor):
        assert raw.ndim == 3 and raw.shape[-1] == 3 and len(descriptor) == 1
        if self.previous_descriptor is None:
            result = {name: raw.copy() for name in self.routes}
        else:
            field, mask = correspondence(self.previous_descriptor, descriptor, raw.shape[:2])
            self.pairs += 1
            result = {}
            for name in self.routes:
                aligned, _ = align_history(self.fused[name], raw,
                    field if name == 'true' else np.zeros_like(field), mask)
                result[name] = blend(raw, aligned, .5).clip(0, 255).astype(np.float32)
        self.previous_descriptor = descriptor.detach().clone(memory_format=torch.contiguous_format)
        self.fused = {name: value.copy() for name, value in result.items()}
        return result

    def state_bytes(self):
        if self.previous_descriptor is None:
            return 0
        return self.previous_descriptor.untyped_storage().nbytes()+sum(v.nbytes for v in self.fused.values())
