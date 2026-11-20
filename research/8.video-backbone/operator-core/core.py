"""Pretrained shared enhancement/matching core; a candidate, not a novel-method claim."""
import copy
import math
import torch
from torch import nn
import torch.nn.functional as F
from budget_causal_r33 import transport


def correlation(current, previous, radius, temperature):
    """Current-to-previous constant-offset cosine scores with explicit bounds."""
    batch, channels, height, width = current.shape
    current = F.normalize(current, dim=1, eps=1e-6)
    previous = F.normalize(previous, dim=1, eps=1e-6)
    padded = F.pad(previous, (radius, radius, radius, radius))
    y = torch.arange(height, device=current.device)[:, None]
    x = torch.arange(width, device=current.device)[None, :]
    scores, offsets = [], []
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            moved = padded[:, :, radius+dy:radius+dy+height, radius+dx:radius+dx+width]
            valid = (y+dy >= 0) & (y+dy < height) & (x+dx >= 0) & (x+dx < width)
            scores.append((current*moved).sum(1).masked_fill(~valid[None], -1e4))
            offsets.append((dx, dy))
    scores = torch.stack(scores, 1)
    probability = (temperature*scores).softmax(1)
    offsets = current.new_tensor(offsets)
    expected = torch.einsum('bkhw,kc->bchw', probability, offsets)
    return expected, scores.max(1, keepdim=True).values


def mix(current, previous, fields):
    flow, logit = fields
    moved, valid = transport(previous, current, flow)
    alpha = .5*logit.sigmoid()*valid.to(current.dtype)
    return current + alpha*(moved-current)


class SharedCore(nn.Module):
    def __init__(self, original):
        super().__init__()
        self.backbone = copy.deepcopy(original).requires_grad_(True)
        self.descriptor = nn.Sequential(nn.Conv2d(12, 8, 3, padding=1), nn.PReLU(8),
                                        nn.Conv2d(8, 8, 3, padding=1, groups=8))
        self.log_temperature = nn.Parameter(torch.tensor(math.log(10.)))
        self.confidence = nn.Conv2d(12, 1, 1)
        nn.init.zeros_(self.confidence.weight); nn.init.zeros_(self.confidence.bias)
        assert sum(p.numel() for p in self.parameters()) == 5049

    def encode(self, low):
        z = self.backbone.head.block1(low)
        head = (self.backbone.head.weight1*z)*(self.backbone.head.weight2*z)+self.backbone.head.bias
        body = self.backbone.body(head)
        channel = self.backbone.att(body)
        spatial = self.backbone.att1((channel*body).max(1, keepdim=True).values)
        raw = self.backbone.tail((spatial*channel)*body)
        descriptor = self.descriptor(F.avg_pool2d(z, 4, 4, ceil_mode=True))
        return raw, descriptor

    def fields(self, previous, current, native_size):
        temperature = self.log_temperature.exp().clamp(1, 100)
        pc, cc = (F.avg_pool2d(v, 4, 4, ceil_mode=True) for v in (previous, current))
        coarse, coarse_score = correlation(cc, pc, 4, temperature)
        coarse_native = F.interpolate(16*coarse, size=current.shape[-2:], mode='bilinear', align_corners=False)
        moved, _ = transport(previous, current, coarse_native/4)
        residual, fine_score = correlation(current, moved, 2, temperature)
        quarter_flow = coarse_native/4 + residual
        features = torch.cat([(current-moved).abs(), quarter_flow,
            F.interpolate(coarse_score, size=current.shape[-2:], mode='bilinear', align_corners=False), fine_score], 1)
        logit = self.confidence(features)
        return (F.interpolate(4*quarter_flow, size=native_size, mode='bilinear', align_corners=False),
                F.interpolate(logit, size=native_size, mode='bilinear', align_corners=False))

    def batch(self, low):
        batch, frames, _, height, width = low.shape
        raw, desc = self.encode(low.flatten(0, 1))
        raw = raw.unflatten(0, (batch, frames)); desc = desc.unflatten(0, (batch, frames))
        fields = self.fields(desc[:, :-1].flatten(0, 1), desc[:, 1:].flatten(0, 1), (height, width))
        return raw, desc, fields


class Streaming:
    def __init__(self, model):
        self.model = model
        self.previous_descriptor = self.previous_fused = None

    def step(self, low):
        raw, descriptor = self.model.encode(low)
        current = raw.clamp(0, 1)
        if self.previous_descriptor is None:
            result = current
        else:
            fields = self.model.fields(self.previous_descriptor, descriptor, low.shape[-2:])
            result = mix(current, self.previous_fused, fields).clamp(0, 1)
        self.previous_descriptor = descriptor.detach().clone(memory_format=torch.contiguous_format)
        self.previous_fused = result.detach().clone(memory_format=torch.contiguous_format)
        return result

    def state_bytes(self):
        if self.previous_descriptor is None:
            return 0
        return sum(v.untyped_storage().nbytes() for v in (self.previous_descriptor, self.previous_fused))


def models(original, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9020+seed)
        first = SharedCore(original)
    second = copy.deepcopy(first)
    assert all(torch.equal(v, second.state_dict()[k]) for k, v in first.state_dict().items())
    return dict(coordinate=first, response=second)
