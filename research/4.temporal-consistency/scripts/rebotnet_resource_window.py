"""Recrop the first established train-only window for a discarded resource trial."""
import numpy as np
import torch

from exploration_r2_data import degrade
from mixed_anchor_r40 import MixedPool


def prepare():
    pool = MixedPool()
    _, _, identities = pool.batch(0, 0)
    real = {row['scene']: row for row in pool.real}
    lows, targets, records = [], [], []
    for b, identity in enumerate(identities):
        if identity['source'] == 'davis_synthetic':
            lookup = {row['name']: row for row in pool.synthetic.scenes[identity['video']]}
            clean = [pool.synthetic.image(lookup[name]) for name in identity['members']]
            input_frames = None
        else:
            row = real[identity['video']]
            clean = [row['native_gt'][i] for i in identity['indices']]
            input_frames = [row['native_input'][i] for i in identity['indices']]
        height, width = clean[0].shape[:2]
        x, y = identity['xy']
        x, y = min(x, width-384), min(y, height-384)
        if min(x, y) < 0:
            raise ValueError('Resource source is smaller than native384; no resize fallback.')

        def crop(frame):
            patch = frame[y:y+384, x:x+384]
            if identity['flip_horizontal']:
                patch = patch[:, ::-1]
            if identity['flip_vertical']:
                patch = patch[::-1]
            return patch.astype(np.float32).copy()

        target = [crop(frame) for frame in clean]
        low = ([degrade(frame, np.random.default_rng(np.random.SeedSequence([750, 0, b, t])),
                        photons=identity['photons'], jpeg=False) for t, frame in enumerate(target)]
               if input_frames is None else [crop(frame) for frame in input_frames])
        lows.append(np.stack(low))
        targets.append(np.stack(target))
        records.append(dict(**identity, resource_crop_xy=[x,y], original_frame_hw=[height,width],
                            scope='Same first train-only identities; clipped384 coordinate for resource use only.'))
    low = torch.from_numpy(np.stack(lows).transpose(0,1,4,2,3).copy()/255)
    target = torch.from_numpy(np.stack(targets).transpose(0,1,4,2,3).copy()/255)
    assert low.shape == target.shape == (4,4,3,384,384)
    return low, target, records, pool.info
