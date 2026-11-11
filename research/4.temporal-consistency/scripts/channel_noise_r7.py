"""Train-source covariance calibration, not a guarantee about moving video."""
import itertools

import torch
from torch.nn import functional as F


def channel_linear_rgb(model, u, gate):
    u, gate = u.double(), gate.double()
    mean = u.mean(0, keepdim=True)
    linear = 2*model.body.weight1.double()*model.body.weight2.double()*mean*(u-mean)
    gated = gate.mean(0, keepdim=True)*linear
    return [F.conv2d(gated[:, j:j+1], model.tail.weight[:, j:j+1].double(), padding=1)
            for j in range(u.shape[1])]


def channel_covariance(contributions, output, margin=8):
    def centered(value):
        x = value.double()[..., margin:-margin, margin:-margin]
        return (x-x.mean(0, keepdim=True)).flatten()
    z = torch.stack([centered(v) for v in contributions])
    y = centered(output)
    return z@z.T/z.shape[1], z@y/z.shape[1], y.square().mean()


def predicted_relative_gain(covariance, cross, variance, subset, gamma=.5):
    selected = list(subset)
    block = covariance[selected][:, selected].sum()
    covariance_with_output = cross[selected].sum()
    return (2*gamma*covariance_with_output-2*gamma*gamma*block)/variance


def subsets():
    return list(itertools.combinations(range(12), 3))
