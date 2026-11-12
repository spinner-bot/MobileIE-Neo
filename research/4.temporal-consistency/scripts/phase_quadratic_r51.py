"""Finite bilinear quadrature of a quadratic FST; deliberately not alias-free."""
import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

MODES = ['raw', 'quadrature', 'preblur', 'postblur']


def filters(x):
    m = x.new_tensor([1., 6., 1.])/8
    d = x.new_tensor([-1., 0., 1.])/8
    return torch.stack([a[:, None]*b[None, :] for a, b in [(m,m), (m,d), (d,m), (d,d)]])


def components(x):
    channels = x.shape[1]
    weight = filters(x)[:, None].repeat(channels, 1, 1, 1)
    result = F.conv2d(F.pad(x, (1,1,1,1), mode='circular'), weight, groups=channels)
    return result.reshape(x.shape[0], channels, 4, *x.shape[-2:])


def blur(x):
    channels = x.shape[1]
    weight = filters(x)[:1, None].repeat(channels, 1, 1, 1)
    return F.conv2d(F.pad(x, (1,1,1,1), mode='circular'), weight, groups=channels)


def quadratic(x):
    return components(x).square().sum(2)


def direct_quadratic(x):
    padded = F.pad(x, (1,1,1,1), mode='circular')
    high = F.interpolate(padded, scale_factor=2, mode='bilinear', align_corners=False)
    high = high[..., 2:-2, 2:-2]
    return F.avg_pool2d(high.square(), 2, 2)


def translate(value, dx, dy):
    """Periodic DFT shift on odd-size real signals; no interpolation blur."""
    h, w = value.shape[-2:]
    assert h % 2 == 1 and w % 2 == 1
    phase = np.exp(-2j*np.pi*(np.fft.fftfreq(h)[:, None]*dy+np.fft.fftfreq(w)[None, :]*dx))
    return np.fft.ifft2(np.fft.fft2(value, axes=(-2,-1))*phase, axes=(-2,-1)).real


class PhaseFST(nn.Module):
    def __init__(self, source, mode):
        super().__init__()
        assert mode in MODES
        self.block1 = source.block1
        self.weight1, self.weight2, self.bias = source.weight1, source.weight2, source.bias
        self.mode = mode

    def forward(self, image):
        x = self.block1(image)
        if self.mode == 'quadrature':
            v = components(x)
            return ((self.weight1*v)*(self.weight2*v)).sum(2)+self.bias
        if self.mode == 'preblur': x = blur(x)
        y = (self.weight1*x)*(self.weight2*x)
        if self.mode == 'postblur': y = blur(y)
        return y+self.bias


def periodic_variant(original, mode):
    net = copy.deepcopy(original).eval().requires_grad_(False)
    for module in net.modules():
        if isinstance(module, nn.Conv2d): module.padding_mode = 'circular'
    net.head, net.body = PhaseFST(net.head, mode), PhaseFST(net.body, mode)
    assert sum(p.numel() for p in net.parameters()) == 4075
    return net
