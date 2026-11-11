"""Matched conditional-state sampling for feature and RGB controls; motion not free."""
import numpy as np
import torch
from torch.nn import functional as F

from feature_mix_r7 import FeatureMix


def motion_field(flow, mask, device):
    h, w = mask.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    sx, sy = xx+flow[..., 0], yy+flow[..., 1]
    valid = mask & np.isfinite(sx) & np.isfinite(sy) & (sx >= 0) & (sx <= w-1) & (sy >= 0) & (sy <= h-1)
    grid = np.stack([sx*2/(w-1)-1, sy*2/(h-1)-1], -1)
    grid[~np.isfinite(grid)] = 0
    return torch.from_numpy(grid.copy())[None].to(device), torch.from_numpy(valid.copy())[None, None].to(device)


def sample_previous(previous, grid):
    return previous if grid is None else F.grid_sample(previous, grid, mode='bilinear',
                                                      padding_mode='zeros', align_corners=True)


def blend(current, previous, gamma, mask=None):
    delta = previous-current
    if mask is not None:
        delta = torch.where(mask, delta, torch.zeros_like(delta))
    return current+gamma*delta


class AlignedFeatureMix(FeatureMix):
    def __init__(self, backbone, gamma):
        super().__init__(backbone, [6, 7, 8], 'post', gamma)
        self.current_mask = None

    def mix(self, current, previous):
        return super().mix(current, previous) if self.current_mask is None else blend(current, previous, self.gamma, self.current_mask)

    def forward(self, x, grid=None, mask=None, reset=False):
        if reset or (self.state_input_shape is not None and self.state_input_shape != x.shape):
            self.reset()
        if self.state is not None:
            self.state = sample_previous(self.state, grid)
        self.current_mask = mask
        try:
            return super().forward(x)
        finally:
            self.current_mask = None


@torch.inference_mode()
def rgb_stream(frames, fields, gamma, recursive=False, resets=None):
    result, previous = [], None
    for t, current in enumerate(frames):
        current = current[None]
        if previous is None or (resets is not None and resets[t]):
            output = current
        else:
            grid, mask = fields[t-1]
            output = blend(current, sample_previous(previous, grid), gamma, mask)
        result.append(output[0])
        previous = output if recursive else current
    return torch.stack(result)
