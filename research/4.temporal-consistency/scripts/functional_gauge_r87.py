"""A bounded diagonal metric and its degree-two function-preserving parameter map."""
import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from exploration_r2_data import degrade
from video_metrics import warp_previous

TRANSLATIONS = [(-2, -1), (-2, 1), (2, -1), (2, 1)]


class Gauge(nn.Module):
    def __init__(self):
        super().__init__()
        self.v = nn.Parameter(torch.zeros(12))

    def gain(self):
        value = self.v.tanh()
        return (.5*(value-value.mean())).exp()


def fold(backbone, gain):
    result = copy.deepcopy(backbone)
    gain = gain.detach().to(dtype=torch.float32, device='cpu')
    assert gain.shape == (12,) and torch.isfinite(gain).all() and bool((gain > 0).all())
    last, body = result.head.block1[-1], result.body.block1
    assert isinstance(last, nn.Conv2d) and isinstance(body, nn.Conv2d)
    assert last.out_channels == body.in_channels == 12
    with torch.no_grad():
        last.weight.mul_(gain[:, None, None, None]); last.bias.mul_(gain)
        result.head.bias.mul_(gain.square()[None, :, None, None])
        body.weight.div_(gain.square()[None, :, None, None])
    assert sum(p.numel() for p in result.parameters()) == 4075
    return result.eval().requires_grad_(False)


def capture(backbone, image):
    captured = []
    def record(module, inputs, output):
        value = F.avg_pool2d(output, 4, 4)
        captured.append((value-value.mean((-2, -1), keepdim=True)).detach().clone())
    handle = backbone.head.block1.register_forward_hook(record)
    try:
        with torch.no_grad():
            prediction = backbone(image)
    finally:
        handle.remove()
    assert len(captured) == 1 and torch.isfinite(prediction).all()
    return prediction.detach(), captured[0]


def synthetic_pairs(clean, source_index):
    previous, current = [], []
    for j, (dx, dy) in enumerate(TRANSLATIONS):
        field = np.broadcast_to(np.array([4*dx, 4*dy], np.float32), (*clean.shape[:2], 2)).copy()
        shifted, _ = warp_previous(clean, field)
        for side, destination, image in [(0, previous, clean), (1, current, shifted)]:
            rng = np.random.default_rng(np.random.SeedSequence([87, source_index, j, side]))
            destination.append(degrade(image, rng, photons=40, jpeg=False))
    def tensor(values):
        return torch.from_numpy(np.stack(values).transpose(0, 3, 1, 2).copy()/255)
    return tensor(previous), tensor(current)


def queries(previous, current):
    assert previous.shape == current.shape == (4, 12, 32, 32)
    axis = torch.tensor([8, 12, 16, 20, 24])
    y, x = axis.repeat_interleave(5), axis.repeat(5)
    query = current[:, :, y, x].permute(0, 2, 1)
    candidates = torch.stack([previous[:, :, y+dy, x+dx].permute(0, 2, 1)
        for dy in range(-4, 5) for dx in range(-4, 5)], dim=2)
    targets = torch.tensor([(dy+4)*9+dx+4 for dx, dy in TRANSLATIONS]).repeat_interleave(25)
    return dict(query=query.reshape(-1, 12).contiguous(),
                candidates=candidates.reshape(-1, 81, 12).contiguous(), target=targets)


def scores(batch, gain):
    q = F.normalize(batch['query']*gain, dim=-1, eps=1e-6)
    p = F.normalize(batch['candidates']*gain, dim=-1, eps=1e-6)
    return (q[:, None, :]*p).sum(-1)/.1


def concatenate(batches):
    return {key: torch.cat([b[key] for b in batches]) for key in ('query', 'candidates', 'target')}


def metrics(batch, gain):
    with torch.no_grad():
        logits = scores(batch, gain)
        predicted = logits.argmax(-1)
        correct = int((predicted == batch['target']).sum())
        ce = float(F.cross_entropy(logits, batch['target']))
    assert torch.isfinite(logits).all()
    return dict(queries=len(predicted), correct=correct, top1=correct/len(predicted), CE=ce), logits, predicted
