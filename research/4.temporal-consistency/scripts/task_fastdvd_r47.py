"""Original FastDVDnet, differentiable four-center windows for task adaptation."""
import json

import torch
import torch.nn.functional as F

from dynamic_training_r34 import BASE, ROOT, digest
from fastdvdnet_adapter_r12 import MANIFEST, load_model, official_inference, window_indices

SIGMA = 15.


def initial_model(seed):
    torch.manual_seed(seed)
    model = load_model(torch.device('cpu'))
    assert sum(p.numel() for p in model.parameters()) == 2479096
    assert all(p.requires_grad for p in model.parameters())
    return model


def prediction(model, low, clamp=False):
    """BTCHW -> BTCHW; unchanged model, all target centers in one BN batch."""
    assert low.ndim == 5 and low.shape[2] == 3
    b, t, c, h, w = low.shape
    indices = torch.as_tensor(window_indices(t, 'window'), device=low.device)
    windows = low[:, indices].reshape(b*t, 5*c, h, w)
    bottom, right = (-h) % 4, (-w) % 4
    padded = F.pad(windows, (0, right, 0, bottom), mode='reflect')
    condition = torch.full((b*t, 1, h, w), SIGMA/255,
                           dtype=low.dtype, device=low.device)
    condition = F.pad(condition, (0, right, 0, bottom), mode='reflect')
    output = model(padded, condition)[..., :h, :w]
    if clamp:
        output = output.clamp(0, 1)
    return output.reshape(b, t, c, h, w)


@torch.inference_mode()
def official_centers(model, low):
    routine = official_inference()
    results = []
    for sequence in low:
        h, w = sequence.shape[-2:]
        condition = torch.full((1, 1, h, w), SIGMA/255,
                               dtype=low.dtype, device=low.device)
        frames = []
        for row in window_indices(len(sequence), 'window'):
            window = sequence[row.tolist()].reshape(1, 15, h, w)
            frames.append(routine.temp_denoise(model, window, condition)[0])
        results.append(torch.stack(frames))
    return torch.stack(results)


def bn_counts(model):
    return {name: int(module.num_batches_tracked) for name, module in model.named_modules()
            if isinstance(module, torch.nn.BatchNorm2d)}


def provenance():
    manifest = json.loads(MANIFEST.read_text())
    return {'official_manifest_sha256': digest(MANIFEST),
            'official_checkpoint_sha256': manifest['sha256'],
            'official_commit': manifest['commit'], 'parameters': 2479096,
            'conditioning_sigma_interface': SIGMA, 'future_wait_max': 2,
            'task': 'direct lowlight RGB to paired GT; no MobileIE prefix',
            'pretraining_overlap': 'Official DAVIS training includes three DEV video IDs; not unseen generalization.'}
