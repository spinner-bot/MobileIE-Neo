"""Verified paired real-video inputs, fixed stream and detached residual loss."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from exploration_r2_data import ROOT
from explore_products_r3 import sampling
from explore_state_r2 import forward_backward
from explore_temporal_controls import sha256

MANIFEST = ROOT/'research/4.temporal-consistency/data/exploration-r4/manifest.json'


def load_sequences(split, size=(480, 270), manifest=MANIFEST):
    data = json.loads(Path(manifest).read_text())
    result = {}
    for selected in data['selection']:
        if selected['split'] != split:
            continue
        scene = selected['scene']
        arrays, names = {}, {}
        for side in ['input', 'gt']:
            rows = sorted([r for r in data['files'] if r['scene'] == scene and r['side'] == side],
                          key=lambda r: r['source_path'])
            values, names[side] = [], [Path(r['source_path']).name for r in rows]
            if len(rows) != 16:
                raise ValueError('exact 16-frame protocol required')
            for row in rows:
                path = ROOT/row['path']
                if sha256(path) != row['sha256']:
                    raise ValueError('source hash mismatch: '+str(path))
                with Image.open(path) as image:
                    image = np.asarray(image.convert('RGB'))
                if image.shape[:2] != (1080, 1920):
                    raise ValueError('unexpected native resolution')
                values.append(cv2.resize(image, size, interpolation=cv2.INTER_AREA).astype(np.float32))
            arrays[side] = values
        if names['input'] != names['gt']:
            raise ValueError('frame pairing mismatch')
        result[scene] = arrays
    return result


def reference_fields(targets):
    pairs = [forward_backward(targets[t-1], targets[t]) for t in range(1, len(targets))]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def to_tensor(frames, device):
    array = np.stack(frames).transpose(0, 3, 1, 2).copy()/255.
    return torch.from_numpy(array).to(device)


def prepare_training(sequences, device):
    data = []
    for name, frames in sequences.items():
        flows, masks = reference_fields(frames['gt'])
        grids, valid = [], []
        for flow, mask in zip(flows, masks):
            mask = mask.copy()
            mask[:8] = mask[-8:] = False
            mask[:, :8] = mask[:, -8:] = False
            grid, m = sampling(flow, mask, device)
            grids.append(grid[0]); valid.append(m[0])
        data.append({'scene': name, 'input': to_tensor(frames['input'], device),
                     'target': to_tensor(frames['gt'], device), 'grid': torch.stack(grids), 'mask': torch.stack(valid)})
    return data


def batch(data, seed, step):
    rng = np.random.default_rng(seed*1000000+step)
    low, target, grids, masks, identity = [], [], [], [], []
    for _ in range(2):
        index = int(rng.integers(len(data)))
        row = data[index]
        t = int(rng.integers(1, len(row['input'])))
        low.append(row['input'][t-1:t+1]); target.append(row['target'][t-1:t+1])
        grids.append(row['grid'][t-1]); masks.append(row['mask'][t-1])
        identity.append((row['scene'], t))
    return torch.stack(low), torch.stack(target), torch.stack(grids), torch.stack(masks), identity


def residual_temporal(prediction, target, grid, mask):
    """No gradient through previous error or fixed motion. Returns Charb floor if empty."""
    error = prediction-target
    previous = F.grid_sample(error[:, 0].detach(), grid, align_corners=True, padding_mode='zeros')
    difference = torch.sqrt((error[:, 1]-previous)**2+1e-6)
    valid = mask.expand_as(difference)
    if not bool(valid.any()):
        return prediction.sum()*0+.001
    return difference[valid].mean()
