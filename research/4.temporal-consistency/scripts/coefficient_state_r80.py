"""Conditional linear readout with a causal finite prefix; no backbone weight update."""
import numpy as np
import torch
import torch.nn.functional as F

from half_motion_r4 import half_forward_backward
from video_metrics import warp_previous

WARM_FRAMES = 8


class ConditionalReadout:
    def __init__(self, model):
        self.model = model.eval().requires_grad_(False)
        assert model.channels == 12 and model.body.weight1.numel() == model.body.weight2.numel() == 1
        assert model.tail.weight.shape == (3, 12, 3, 3)

    def parts(self, value):
        net = self.model
        head = net.head(value)
        z = net.body.block1(head)
        q = (net.body.weight1*z)*(net.body.weight2*z)
        u = q+net.body.bias
        att = net.att(u)
        local = net.att1(torch.max(att*u, dim=1, keepdim=True).values)
        gate = local*att
        return dict(q=q, u=u, gate=gate, base=net.tail(gate*u))

    def basis(self, parts):
        assert parts['u'].shape[0] == 1
        weight, padding = self.model.tail.weight, self.model.tail.padding
        scale = F.conv2d(parts['gate']*parts['q'], weight, bias=None, padding=padding)
        grouped = F.conv2d(parts['gate'], weight.permute(1, 0, 2, 3).contiguous().reshape(36, 1, 3, 3),
                           bias=None, padding=padding, groups=12)
        bias = grouped.reshape(1, 12, 3, *scale.shape[-2:])
        return torch.cat([scale[:, None], bias], dim=1)[0].contiguous()

    def predict(self, parts, physical):
        assert physical.shape == (13,) and physical.dtype == torch.float32
        adapted = parts['u']+physical[0]*parts['q']+physical[1:].reshape(1, 12, 1, 1)
        return self.model.tail(parts['gate']*adapted)


def rgb_basis(base):
    assert base.shape[0] == 1
    fields = torch.cat([base[0], torch.ones_like(base[0, :1])], dim=0)
    bank = torch.zeros((12, 3, *base.shape[-2:]), dtype=base.dtype, device=base.device)
    for channel in range(3):
        bank[channel*4:(channel+1)*4, channel] = fields
    return bank


def rgb_predict(base, physical):
    matrix = physical.reshape(3, 4)
    return base+torch.einsum('oc,nchw->nohw', matrix[:, :3], base)+matrix[:, 3].reshape(1, 3, 1, 1)


class PrefixRidge:
    """One frame-mean normal-equation contribution, ridge1, update after prediction."""
    def __init__(self, dimensions):
        assert dimensions in (12, 13)
        self.dimensions = dimensions
        self.count = 0
        self.gram = torch.zeros(dimensions, dimensions, dtype=torch.float64)
        self.rhs = torch.zeros(dimensions, dtype=torch.float64)
        self.normalizer = None
        self.physical = torch.zeros(dimensions, dtype=torch.float32)

    def update(self, basis, base, target):
        if self.count >= WARM_FRAMES:
            raise RuntimeError('Frozen prefix state cannot be updated after frame7.')
        assert basis.device.type == base.device.type == target.device.type == 'cpu'
        values = basis.detach().reshape(self.dimensions, -1).to(torch.float64)
        residual = (target.detach().to(torch.float64)-base.detach().to(torch.float64)).reshape(-1)*255
        assert values.shape[1] == residual.numel()
        if self.normalizer is None:
            self.normalizer = values.square().mean(1).sqrt().clamp_min(1e-6)
        design = values/self.normalizer[:, None]
        self.gram += design@design.T/design.shape[1]
        self.rhs += design@residual/design.shape[1]
        theta = torch.linalg.solve(self.gram+torch.eye(self.dimensions, dtype=torch.float64), self.rhs)
        self.physical = (theta/(255*self.normalizer)).to(torch.float32).contiguous()
        if not torch.isfinite(self.physical).all():
            raise FloatingPointError('Nonfinite prefix coefficients; no scientific auto-retry.')
        self.count += 1
        if self.count == WARM_FRAMES:
            self.gram = self.rhs = self.normalizer = None

    def storage_bytes(self):
        return sum(t.untyped_storage().nbytes() for t in
                   (self.gram, self.rhs, self.normalizer, self.physical) if t is not None)


class PrefixTeacher:
    """Unclamped .5 recursive pseudo-target; flow uses clamped frozen outputs only."""
    def __init__(self):
        self.count = self.flow_pairs = 0
        self.previous_raw = self.previous_target = None

    def step(self, raw):
        if self.count >= WARM_FRAMES:
            raise RuntimeError('Teacher is forbidden after the warm prefix.')
        assert raw.device.type == 'cpu' and raw.shape[0:2] == (1, 3)
        current = raw[0].detach().permute(1, 2, 0).contiguous().numpy()
        clipped = np.ascontiguousarray(np.clip(current, 0, 1)*255)
        if self.previous_raw is None:
            target = current.copy()
        else:
            flow, confidence = half_forward_backward(self.previous_raw, clipped)
            moved, valid = warp_previous(self.previous_target, flow)
            coefficient = np.float32(.5)*(confidence & valid).astype(np.float32)[..., None]
            target = current+coefficient*(moved-current)
            self.flow_pairs += 1
        self.previous_raw, self.previous_target = clipped.copy(), target.copy()
        self.count += 1
        if self.count == WARM_FRAMES:
            self.previous_raw = self.previous_target = None
        return torch.from_numpy(target.transpose(2, 0, 1).copy()[None])

    def storage_bytes(self):
        return sum(value.nbytes for value in (self.previous_raw, self.previous_target) if value is not None)
