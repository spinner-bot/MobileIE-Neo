"""Stateless continuous T8 clips from the established, disjoint owned training sources."""
import sys

import numpy as np
import torch

from core import ROOT
sys.path.insert(0, str(ROOT / 'research/4.temporal-consistency/scripts'))
from mixed_anchor_r40 import MixedPool
from exploration_r2_data import degrade


class Pool8:
    def __init__(self):
        self.pool = MixedPool()
        assert len(self.pool.synthetic.names) == 57 and len(self.pool.real) == 12
        assert all(len(v) >= 8 for v in self.pool.synthetic.scenes.values())
        self.info = dict(self.pool.info, temporal_length=8, batch_size=1, crop=128,
                         sampling='SeedSequence([194,seed,step]); even SYN / odd ND',
                         MixedPool_fixed_T4_batch_called=False, new_source_split=False)

    def batch(self, seed, step):
        rng = np.random.default_rng(np.random.SeedSequence([194, seed, step]))
        horizontal, vertical = bool(rng.integers(2)), bool(rng.integers(2))

        def crop(image, x, y):
            value = image[y:y+128, x:x+128]
            if horizontal:
                value = value[:, ::-1]
            if vertical:
                value = value[::-1]
            assert value.shape == (128, 128, 3)
            return value.astype(np.float32)

        if step % 2 == 0:
            pool = self.pool.synthetic
            name = pool.names[int(rng.integers(len(pool.names)))]
            records = pool.scenes[name]
            start = int(rng.integers(len(records) - 7))
            selected = records[start:start+8]
            first = pool.image(selected[0])
            y, x = int(rng.integers(first.shape[0]-127)), int(rng.integers(first.shape[1]-127))
            photons = int(rng.choice([20, 80]))
            clean = [crop(pool.image(r), x, y) for r in selected]
            low = [degrade(v, rng, photons=photons, jpeg=False) for v in clean]
            meta = dict(source='davis_synthetic', video=name, photons=photons,
                        members=[r['name'] for r in selected], input_GT_source_paths=[r['path'] for r in selected])
        else:
            row = self.pool.real[int(rng.integers(len(self.pool.real)))]
            length, height, width, _ = row['native_input'].shape
            assert length == 16 and row['native_gt'].shape == row['native_input'].shape
            start = int(rng.integers(length - 7))
            y, x = int(rng.integers(height-127)), int(rng.integers(width-127))
            low = [crop(v, x, y) for v in row['native_input'][start:start+8]]
            clean = [crop(v, x, y) for v in row['native_gt'][start:start+8]]
            meta = dict(source='olives_real', video=row['scene'],
                        input_source_paths=[r['path'] for r in row['source_files']['input'][start:start+8]],
                        GT_source_paths=[r['path'] for r in row['source_files']['gt'][start:start+8]])
        meta.update(indices=list(range(start, start+8)), xy=[x, y],
                    flip_horizontal=horizontal, flip_vertical=vertical,
                    source_frame_indices_not_sensor_timestamps=True)
        assert len(low) == len(clean) == 8
        values = [torch.from_numpy(np.stack(v).transpose(0, 3, 1, 2).copy()[None] / 255) for v in (low, clean)]
        assert all(v.shape == (1, 8, 3, 128, 128) and v.dtype == torch.float32 and torch.isfinite(v).all() for v in values)
        return *values, meta
