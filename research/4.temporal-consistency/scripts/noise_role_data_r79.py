"""Known-dark synthetic role views; real observations are never decomposed."""
import numpy as np
import torch

from mixed_anchor_r40 import MixedPool


def gain(step):
    if not 0 <= step < 4000:
        raise ValueError('R79 registered data step is 0..3999.')
    return 16.-15.*step/999 if step < 1000 else 1.


def quantize(value):
    return (np.clip(value, 0, 1)*255).astype(np.uint8).astype(np.float32)


def synthetic_views(clean, rng, photons, g):
    dark = (np.clip(clean, 0, 255)/255)**2.2*.35
    observation = rng.poisson(dark*photons)/photons+rng.normal(0, .008, dark.shape)
    ordinary = quantize(observation)
    if g == 1:
        return ordinary, ordinary.copy(), ordinary.copy()
    residual = observation-dark
    return ordinary, quantize(dark+g*residual), quantize(dark+residual/g)


def crop(frames, x, y, horizontal, vertical):
    value = np.stack([frame[y:y+128, x:x+128] for frame in frames])
    if horizontal:
        value = value[:, :, ::-1]
    if vertical:
        value = value[:, ::-1]
    assert value.shape == (4, 128, 128, 3)
    return value.astype(np.float32).copy()


def tensor(values):
    return torch.from_numpy(np.stack(values).transpose(0, 1, 4, 2, 3).copy()/255)


class NoiseRolePool(MixedPool):
    def batch(self, seed, step):
        if seed not in (0, 1):
            raise ValueError('Only two registered seeds.')
        g = gain(step)
        rng = np.random.default_rng(np.random.SeedSequence([79, seed, step]))
        ordinary, current, history, targets, identities = [], [], [], [], []
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
                identity = dict(source='olives_real', video=row['scene'], indices=list(range(start, start+4)))
            height, width = clean[0].shape[:2]
            x, y = int(rng.integers(width-127)), int(rng.integers(height-127))
            horizontal, vertical = bool(rng.integers(2)), bool(rng.integers(2))
            target = crop(clean, x, y, horizontal, vertical)
            if raw is None:
                photons = int(rng.choice([20, 80]))
                key = [79, seed, step, b, 1]
                noise = np.random.default_rng(np.random.SeedSequence(key))
                views = [synthetic_views(frame, noise, photons, g) for frame in target]
                low, cur, past = [np.stack([row[i] for row in views]) for i in range(3)]
                identity.update(photons=photons, noise_seed_sequence=key, gain=g,
                    ordinary_endpoint_fraction=float(((low == 0) | (low == 255)).mean()),
                    current_role_endpoint_fraction=float(((cur == 0) | (cur == 255)).mean()),
                    history_role_endpoint_fraction=float(((past == 0) | (past == 255)).mean()))
            else:
                low = crop(raw, x, y, horizontal, vertical)
                cur, past = low.copy(), low.copy()
                identity.update(gain=1., real_noise_decomposed=False)
            ordinary.append(low); current.append(cur); history.append(past); targets.append(target)
            identities.append(dict(**identity, source_start=start, native_hw=[height, width], xy=[x, y],
                                   flip_horizontal=horizontal, flip_vertical=vertical, crop=128, resize=False))
        values = tuple(tensor(rows) for rows in (ordinary, current, history, targets))
        assert all(value.dtype == torch.float32 and value.shape == (4, 4, 3, 128, 128) for value in values)
        return (*values, identities)
