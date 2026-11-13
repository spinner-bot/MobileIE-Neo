"""R37 prototype: frozen spatial function with full-resolution projected memory."""
import copy

import torch
from torch import nn
import torch.nn.functional as F

from budget_causal_r33 import transport


class PreservedState(nn.Module):
    def __init__(self, backbone, aligned=True):
        super().__init__()
        self.backbone = copy.deepcopy(backbone).eval().requires_grad_(False)
        self.directions = nn.Parameter(torch.randn(3, 12))
        self.field = nn.Sequential(nn.Conv2d(6, 8, 3, padding=1), nn.PReLU(8), nn.Conv2d(8, 3, 1))
        nn.init.zeros_(self.field[-1].weight)
        nn.init.zeros_(self.field[-1].bias)
        self.amplitude = nn.Parameter(torch.zeros(()))
        self.aligned = aligned
        self._state = None

    def basis(self):
        matrix = torch.eye(12, dtype=self.directions.dtype, device=self.directions.device)
        identity = matrix
        for direction in self.directions.unbind(0):
            v = direction/direction.norm().clamp_min(1e-12)
            matrix = matrix@(identity-2*v[:, None]*v[None])
        return matrix[:3]

    def finish(self, features):
        net = self.backbone
        head = (net.head.weight1*features)*(net.head.weight2*features)+net.head.bias
        body = net.body(head)
        channel = net.att(body)
        maximum = (channel*body).max(dim=1, keepdim=True).values
        spatial = net.att1(maximum)
        return net.tail((spatial*channel)*body)

    def forward(self, image, state=None, history='causal'):
        if image.ndim != 4 or image.shape[1] != 3 or min(image.shape[-2:]) < 4:
            raise ValueError('Expected BCHW RGB with dimensions at least four')
        if history == 'current':
            return self.backbone(image), None
        if history != 'causal':
            raise ValueError('Invalid history route')
        features = self.backbone.head.block1(image)
        basis = self.basis()
        current = F.conv2d(features, basis[:, :, None, None])
        if state is None:
            return self.finish(features), current.half()
        if state.shape != current.shape or state.device != current.device or state.dtype != torch.float16:
            raise ValueError('Expected same-shape FP16 projected history on the input device')
        previous = state.float()
        pair = torch.cat([F.avg_pool2d(current, 4, 4, ceil_mode=True),
                          F.avg_pool2d(previous, 4, 4, ceil_mode=True)], dim=1)
        field = self.field(pair)
        offset = F.interpolate(2*field[:, :2].tanh(), size=current.shape[-2:],
                               mode='bilinear', align_corners=False)*4
        if not self.aligned:
            offset = offset*0
        moved, valid = transport(previous, current, offset)
        confidence = F.interpolate(field[:, 2:3].sigmoid(), size=current.shape[-2:],
                                   mode='bilinear', align_corners=False)
        gate = .95*self.amplitude.tanh()*confidence*valid.to(current.dtype)
        innovation = gate*(moved-current)
        updated = features+F.conv2d(innovation, basis.t()[:, :, None, None])
        return self.finish(updated), (current+innovation).half()

    def reset(self):
        self._state = None

    def step(self, image, history='causal'):
        expected = (image.shape[0], 3, *image.shape[-2:])
        if self._state is not None and (self._state.shape != expected or self._state.device != image.device):
            self.reset()
        output, state = self(image, self._state, history=history)
        self._state = None if state is None else state.detach()
        return output

    def state_storage_bytes(self):
        return 0 if self._state is None else self._state.untyped_storage().nbytes()
