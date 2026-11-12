"""Fixed half-real/half-synthetic windows; retain the strong original backbone."""
import json

import numpy as np
import torch

from dynamic_training_r34 import BASE, ROOT, TrainingPool, digest
from native_training_r14 import load_training
from train_mixture_r4b import EXPANDED


class MixedPool:
    def __init__(self):
        manifest = json.loads(EXPANDED.read_text())
        train_ids = {r['scene'] for r in manifest['selection'] if r['split'] == 'train'}
        other_ids = {r['scene'] for r in manifest['selection'] if r['split'] != 'train'}
        development = {'video_15', 'video_47', 'video_91', 'video_121'}
        assert len(train_ids) == 12 and not train_ids & (other_ids | development)
        files = [r for r in manifest['files'] if r['scene'] in train_ids]
        assert len(files) == 384
        assert all((ROOT/r['path']).resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo') for r in files)
        self.real, info = load_training()
        for row in self.real:
            # Reuse the established verified loader; this experiment uses native crops.
            del row['resized_input'], row['resized_gt']
        self.synthetic = TrainingPool()
        self.info = {**info, 'real_training_ids': sorted(train_ids),
                     'excluded_native_DEV': sorted(development), 'all_real_assets_owned': True,
                     'source_manifest_sha256': digest(EXPANDED),
                     'synthetic_manifest_sha256': digest(BASE/'data/exploration-r32-training/manifest.json'),
                     'synthetic_training_videos': len(self.synthetic.names),
                     'real_tensor_dtype': 'uint8 native inputs/GT; float32 /255 after crops',
                     'timing_scope': 'Consecutive registered frame indices, not independently verified sensor timing.'}

    def batch(self, seed, step):
        synthetic_low, synthetic_gt, synthetic_ids = self.synthetic.batch(seed, step)
        rng = np.random.default_rng(np.random.SeedSequence([40, seed, step]))
        lows, targets, identities = [], [], []
        for _ in range(2):
            row = self.real[int(rng.integers(len(self.real)))]
            start = int(rng.integers(13))
            x, y = int(rng.integers(1793)), int(rng.integers(953))
            horizontal, vertical = bool(rng.integers(2)), bool(rng.integers(2))
            values = []
            for side in ['input', 'gt']:
                patch = row['native_'+side][start:start+4, y:y+128, x:x+128]
                if horizontal:
                    patch = patch[:, :, ::-1]
                if vertical:
                    patch = patch[:, ::-1]
                assert patch.shape == (4, 128, 128, 3)
                values.append(patch.astype(np.float32))
            lows.append(values[0])
            targets.append(values[1])
            identities.append({'source': 'olives_real', 'video': row['scene'],
                               'indices': list(range(start, start+4)), 'xy': [x, y],
                               'flip_horizontal': horizontal, 'flip_vertical': vertical,
                               'input_sources': [r['path'] for r in row['source_files']['input'][start:start+4]],
                               'target_sources': [r['path'] for r in row['source_files']['gt'][start:start+4]]})
        real_low = torch.from_numpy(np.stack(lows).transpose(0, 1, 4, 2, 3).copy()/255)
        real_gt = torch.from_numpy(np.stack(targets).transpose(0, 1, 4, 2, 3).copy()/255)
        return (torch.cat([synthetic_low[:2], real_low]), torch.cat([synthetic_gt[:2], real_gt]),
                [{**r, 'source': 'davis_synthetic'} for r in synthetic_ids[:2]]+identities)


def prediction(net, low):
    return torch.stack([net(frame) for frame in low.unbind(1)], dim=1)
