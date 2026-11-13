"""Shared deployment-motion fields and full four-frame controller recursion."""
import numpy as np
import torch

from half_motion_r4 import half_forward_backward


def motion_for(raw):
    images = raw.detach().cpu().permute(0, 1, 3, 4, 2).contiguous().numpy()*255
    fields = []
    for t in range(1, raw.shape[1]):
        values = [half_forward_backward(v[t-1], v[t]) for v in images]
        flow = torch.from_numpy(np.stack([v[0] for v in values]).transpose(0, 3, 1, 2).copy())
        mask = torch.from_numpy(np.stack([v[1] for v in values])[:, None].copy()).bool()
        fields.append((flow, mask))
    return fields


def move_fields(fields, device):
    return [(flow.to(device), mask.to(device)) for flow, mask in fields]


def rollout(control, raw, cues, fields):
    outputs = [control(raw[:, 0])[0]]
    for t, (flow, mask) in enumerate(fields, 1):
        outputs.append(control(raw[:, t], outputs[-1], flow, mask, cues[:, t])[0])
    return torch.stack(outputs, dim=1)


def encode(encoder, low):
    values = [encoder(frame) for frame in low.unbind(1)]
    return (torch.stack([v[0] for v in values], dim=1).clamp(0, 1),
            torch.stack([v[1] for v in values], dim=1))
