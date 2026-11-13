"""Uniform native384 four-frame windows from the established disjoint train pool."""
import numpy as np
import torch

from exploration_r2_data import degrade
from mixed_anchor_r40 import MixedPool


def crop(frames, x, y, horizontal, vertical):
    values = np.stack([frame[y:y+384, x:x+384] for frame in frames])
    if horizontal:
        values = values[:, :, ::-1]
    if vertical:
        values = values[:, ::-1]
    if values.shape != (4, 384, 384, 3):
        raise ValueError('Native384 crop unavailable; no resize or pad fallback.')
    return values.astype(np.float32).copy()


class CausalReferencePool(MixedPool):
    def batch(self, seed, step):
        if seed not in (0, 1) or not 0 <= step < 64000:
            raise ValueError('Only registered seed/data-step range is allowed.')
        rng = np.random.default_rng(np.random.SeedSequence([75, seed, step]))
        lows, targets, identities = [], [], []
        for b in range(4):
            if b < 2:
                name = self.synthetic.names[int(rng.integers(len(self.synthetic.names)))]
                records = self.synthetic.scenes[name]
                start = int(rng.integers(len(records)-3))
                selected = records[start:start+4]
                clean = [self.synthetic.image(row) for row in selected]
                raw = None
                identity = dict(source='davis_synthetic', video=name,
                                members=[row['name'] for row in selected])
            else:
                row = self.real[int(rng.integers(len(self.real)))]
                start = int(rng.integers(13))
                clean = row['native_gt'][start:start+4]
                raw = row['native_input'][start:start+4]
                identity = dict(source='olives_real', video=row['scene'],
                                indices=list(range(start,start+4)))
            height, width = clean[0].shape[:2]
            if height < 384 or width < 384 or any(frame.shape != clean[0].shape for frame in clean):
                raise ValueError('Training window native dimensions are incompatible.')
            x = int(rng.integers(width-384+1))
            y = int(rng.integers(height-384+1))
            horizontal, vertical = bool(rng.integers(2)), bool(rng.integers(2))
            target = crop(clean, x, y, horizontal, vertical)
            if raw is None:
                photons = int(rng.choice([20,80]))
                noise_key = [75, seed, step, b, 1]
                noise = np.random.default_rng(np.random.SeedSequence(noise_key))
                low = np.stack([degrade(frame, noise, photons=photons, jpeg=False) for frame in target])
                identity.update(photons=photons, degradation_seed_sequence=noise_key)
            else:
                low = crop(raw, x, y, horizontal, vertical)
            lows.append(low)
            targets.append(target)
            identities.append(dict(**identity, source_start=start, native_hw=[height,width],
                xy=[x,y], flip_horizontal=horizontal, flip_vertical=vertical,
                crop=384, targets=4, resize=False))
        low = torch.from_numpy(np.stack(lows).transpose(0,1,4,2,3).copy()/255)
        target = torch.from_numpy(np.stack(targets).transpose(0,1,4,2,3).copy()/255)
        if low.shape != target.shape or low.dtype != torch.float32 or low.shape != (4,4,3,384,384):
            raise ValueError('Registered native train batch changed.')
        return low, target, identities
