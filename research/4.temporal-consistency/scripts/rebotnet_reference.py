"""Explicit self-start/native-tile adaptation of the pinned released ReBotNet XS."""
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
TILE = 384
STRIDE = 256
XS = dict(upscale=1, img_size=[2, 384, 384], window_size=[6, 8, 8],
          depths=[4, 3, 3, 4], indep_reconsts=[9, 10], embed_dims=[8, 12, 20, 32],
          num_heads=[6] * 13, pa_frames=2, deformable_groups=12,
          mlp_dim=64, bottle_depth=4, bottle_dim=32, dropout=0.1, patch_size=1)


def new_model(seed=0):
    torch.manual_seed(seed)
    spec = importlib.util.spec_from_file_location('released_rebotnet_reference', ROOT / 'third_party/rebotnet/archs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.rebotnet(**XS)


def starts(length):
    if length < 1:
        raise ValueError('Spatial dimensions must be positive.')
    end = max(TILE, length) - TILE
    values = list(range(0, end + 1, STRIDE))
    if values[-1] != end:
        values.append(end)
    return values


def tile_layout(height, width):
    return [(y, x) for y in starts(height) for x in starts(width)]


def positive_window(reference):
    positions = torch.arange(TILE, dtype=reference.dtype, device=reference.device)
    one = (1 - (positions - (TILE - 1) / 2).abs() / ((TILE + 1) / 2)).clamp_min(1e-3)
    return (one[:, None] * one[None, :])[None, None]


@torch.no_grad()
def native_pair(model, previous, current):
    if previous.shape != current.shape or current.ndim != 4 or current.shape[1] != 3:
        raise ValueError('Expected matching B,3,H,W RGB tensors.')
    height, width = current.shape[-2:]
    padding = (0, max(0, TILE - width), 0, max(0, TILE - height))
    previous = F.pad(previous, padding, mode='replicate')
    current = F.pad(current, padding, mode='replicate')
    total = torch.zeros_like(current)
    weights = current.new_zeros((1, 1, *current.shape[-2:]))
    window = positive_window(current)
    for y, x in tile_layout(height, width):
        pair = torch.stack((previous[..., y:y+TILE, x:x+TILE],
                            current[..., y:y+TILE, x:x+TILE]), dim=1)
        value = model(pair)
        if value.shape != current[..., y:y+TILE, x:x+TILE].shape:
            raise RuntimeError('Released model returned an incompatible tile.')
        total[..., y:y+TILE, x:x+TILE] += value * window
        weights[..., y:y+TILE, x:x+TILE] += window
    return (total / weights)[..., :height, :width]


def crop_predictions(model, low):
    """Differentiable B,T,3,384,384 self-start; no detach/clamp/GT state."""
    if low.ndim != 5 or low.shape[2:] != (3, TILE, TILE):
        raise ValueError('Training requires native 384 RGB crops.')
    previous = low[:, 0]
    outputs = []
    for current in low.unbind(1):
        previous = model(torch.stack((previous, current), dim=1))
        outputs.append(previous)
    return torch.stack(outputs, dim=1)


def native_predictions(model, frames):
    """Yield complete stitched RGB history; caller clamps only reported outputs."""
    previous = None
    for current in frames:
        previous = native_pair(model, current if previous is None else previous, current)
        yield previous
