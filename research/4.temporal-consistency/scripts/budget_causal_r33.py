"""Untrained budgeted backbone with matched spatial and causal-state routes."""
import copy
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'MobileIE'))
from model.lle import MobileIELLENetS


def transport(previous, current, offset):
    """Continuous bilinear gather in low-resolution pixel units; bounds fallback."""
    if previous.shape != current.shape or offset.shape != (current.shape[0], 2, *current.shape[-2:]):
        raise ValueError('state/offset shape mismatch')
    batch, channels, height, width = current.shape
    y = torch.arange(height, dtype=offset.dtype, device=offset.device)[None, :, None]+offset[:, 1]
    x = torch.arange(width, dtype=offset.dtype, device=offset.device)[None, None, :]+offset[:, 0]
    x0, y0 = x.floor().long(), y.floor().long()
    dx, dy = x-x0.to(x.dtype), y-y0.to(y.dtype)
    valid = ((x >= 0) & (x <= width-1) & (y >= 0) & (y <= height-1))[:, None]
    flat = previous.reshape(batch, channels, height*width)
    def corner(ix, iy):
        indices = (iy.clamp(0, height-1)*width+ix.clamp(0, width-1)).reshape(batch, 1, -1)
        return torch.gather(flat, 2, indices.expand(-1, channels, -1)).reshape_as(current)
    value = (corner(x0, y0)*((1-dx)*(1-dy))[:, None]
             + corner(x0+1, y0)*(dx*(1-dy))[:, None]
             + corner(x0, y0+1)*((1-dx)*dy)[:, None]
             + corner(x0+1, y0+1)*(dx*dy)[:, None])
    return torch.where(valid, value, current), valid


class BudgetCausal(nn.Module):
    """Read R33 protocol: this is a feasibility control, not an established method."""
    def __init__(self, backbone=None, temporal=True):
        super().__init__()
        self.backbone = copy.deepcopy(backbone) if backbone is not None else MobileIELLENetS(12)
        if not isinstance(self.backbone.body.block1, nn.Conv2d) or self.backbone.body.block1.in_channels != 12:
            raise ValueError('R33 expects the original 12-channel slim backbone')
        self.backbone.body.block1 = nn.Sequential(nn.Conv2d(12, 12, 3, padding=1, groups=12), nn.Conv2d(12, 12, 1))
        self.project = nn.Conv2d(12, 8, 1)
        self.mix = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, groups=16), nn.Conv2d(16, 8, 1), nn.PReLU(8))
        self.field = nn.Conv2d(8, 3, 1)
        self.readout = nn.Conv2d(8, 12, 1)
        nn.init.zeros_(self.field.weight)
        nn.init.zeros_(self.field.bias)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.temporal = temporal
        self._state = None

    def forward(self, image, state=None, history=None):
        if image.ndim != 4 or image.shape[1] != 3 or min(image.shape[-2:]) < 4:
            raise ValueError('Expected BCHW RGB with height/width at least4')
        history = ('causal' if self.temporal else 'current') if history is None else history
        if history not in ['causal', 'current'] or (not self.temporal and history == 'causal'):
            raise ValueError('Invalid history route for this control')
        features = self.backbone.head.block1(image)
        current = self.project(F.avg_pool2d(features, 4, 4, ceil_mode=True))
        previous = current if history == 'current' or state is None else state
        if previous.shape != current.shape or previous.device != current.device or previous.dtype != current.dtype:
            raise ValueError('Functional state shape/device/dtype mismatch; reset the sequence')
        fields = self.field(self.mix(torch.cat([current, previous], dim=1)))
        moved, valid = transport(previous, current, 2*fields[:, :2].tanh())
        confidence = fields[:, 2:3].sigmoid()*valid.to(current.dtype)
        next_state = current+confidence*(moved-current)
        correction = F.interpolate(self.readout(next_state), size=features.shape[-2:], mode='bilinear', align_corners=False)
        features = features+correction
        head = (self.backbone.head.weight1*features)*(self.backbone.head.weight2*features)+self.backbone.head.bias
        body = self.backbone.body(head)
        channel = self.backbone.att(body)
        maximum = (channel*body).max(dim=1, keepdim=True).values
        spatial = self.backbone.att1(maximum)
        return self.backbone.tail((spatial*channel)*body), next_state

    def reset(self):
        self._state = None

    def step(self, image, history=None):
        expected = (image.shape[0], 8, (image.shape[-2]+3)//4, (image.shape[-1]+3)//4)
        if self._state is not None and (self._state.shape != expected or self._state.device != image.device or self._state.dtype != image.dtype):
            self.reset()
        prediction, state = self(image, self._state, history=history)
        self._state = state.detach()
        return prediction

    def state_storage_bytes(self):
        return 0 if self._state is None else self._state.untyped_storage().nbytes()
