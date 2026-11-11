"""One-parameter body interaction, matched initial function and causal inference."""
import torch
from torch import nn
from torch.nn import functional as F

from explore_products_r3 import sampling


class TrainablePairFST(nn.Module):
    def __init__(self, original, mode):
        super().__init__()
        self.original, self.mode = original, mode
        self.gamma = nn.Parameter(torch.zeros((), device=original.weight1.device))
        self.offsets = None
        self.previous = None
        self.grid = self.mask = None

    def reset(self):
        self.previous = self.grid = self.mask = None
        self.offsets = None

    def product(self, current, past):
        if self.mode == 'input_mean':
            return current+self.gamma*((current+past)/2-current)
        m = self.original
        ordinary = (m.weight1*current)*(m.weight2*current)+m.bias
        if self.mode == 'cross':
            changed = (m.weight1*current)*(m.weight2*past)+m.bias
        elif self.mode == 'mean_square':
            mean = (current+past)/2
            changed = (m.weight1*mean)*(m.weight2*mean)+m.bias
        else:
            raise ValueError(self.mode)
        return ordinary+self.gamma*(changed-ordinary)

    def forward(self, value):
        feature = self.original.block1(value)
        if self.offsets is not None:
            pairs = feature.reshape(-1, 2, *feature.shape[1:])
            outputs = []
            for pair, (dx, dy) in zip(pairs, self.offsets):
                prev, cur = pair[0:1], pair[1:2]
                h, w = cur.shape[-2:]
                x0, x1 = max(8, 8-dx), min(w-8, w-8-dx)
                y0, y1 = max(8, 8-dy), min(h-8, h-8-dy)
                past = cur.clone()
                past[..., y0:y1, x0:x1] = prev[..., y0+dy:y1+dy, x0+dx:x1+dx]
                outputs.extend([self.product(prev, prev), self.product(cur, past)])
            return torch.cat(outputs)
        past = feature
        if self.previous is not None and self.grid is not None:
            warped = F.grid_sample(self.previous, self.grid, align_corners=True, padding_mode='zeros')
            past = torch.where(self.mask, warped, feature)
        self.previous = feature.detach()
        return self.product(feature, past)


def set_alignment(module, flow, mask, device):
    module.grid, module.mask = sampling(flow, mask, device)


class InputIdentity(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.block1 = nn.Identity()
        self.register_buffer('weight1', torch.ones((), device=device))


class InputPairNetwork(nn.Module):
    """Expose the same state controller interface, apply averaging before RGB net."""
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.body = TrainablePairFST(InputIdentity(next(original.parameters()).device), 'input_mean')

    def forward(self, value):
        return self.original(self.body(value))
