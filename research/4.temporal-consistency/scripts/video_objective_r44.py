"""Training-only GT-change-compensated temporal MSE; no inference teacher."""
import numpy as np
import torch

from budget_causal_r33 import transport
from half_motion_r4 import half_forward_backward


def prepare_motion(target):
    if target.device.type != 'cpu' or target.ndim != 5 or target.shape[2] != 3:
        raise ValueError('Expected CPU BTCHW training targets in [0,1]')
    batch, frames, channels, height, width = target.shape
    if frames != 4 or min(height, width) <= 16:
        raise ValueError('Expected four-frame windows with a nonempty margin8 interior')
    rgb = target.detach().permute(0, 1, 3, 4, 2).contiguous().numpy()*255
    flows, masks = [], []
    for sample in rgb:
        for previous, current in zip(sample[:-1], sample[1:]):
            flow, mask = half_forward_backward(previous, current)
            flows.append(flow)
            masks.append(mask)
    offset = torch.from_numpy(np.stack(flows)).permute(0, 3, 1, 2).contiguous()
    mask = torch.from_numpy(np.stack(masks))[:, None]
    previous = target[:, :-1].reshape(-1, channels, height, width)
    current = target[:, 1:].reshape_as(previous)
    with torch.no_grad():
        moved, bounds = transport(previous, current, offset)
        mask &= bounds
        mask[..., :8, :] = mask[..., -8:, :] = False
        mask[..., :, :8] = mask[..., :, -8:] = False
        delta = current-moved
    assert torch.isfinite(offset).all() and torch.isfinite(delta).all()
    return {'offset': offset, 'mask': mask, 'target_delta': delta}


def to_device(motion, device):
    return {name: value.to(device) for name, value in motion.items()}


def temporal_mse(prediction, motion):
    batch, frames, channels, height, width = prediction.shape
    previous = prediction[:, :-1].reshape(-1, channels, height, width)
    current = prediction[:, 1:].reshape_as(previous)
    moved, valid = transport(previous, current, motion['offset'])
    mask = motion['mask'] & valid
    error = (current-moved)-motion['target_delta']
    weight = mask.to(error.dtype)
    return (error.square()*weight).sum()/(weight.sum()*channels).clamp_min(1)


def valid_fraction(motion):
    return float(motion['mask'].float().mean())
