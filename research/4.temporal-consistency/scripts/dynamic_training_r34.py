"""Disjoint dynamic crop loader and explicit approximate pretrained initialization."""
from collections import OrderedDict
import hashlib
import json

import cv2
import numpy as np
import torch

from budget_causal_r33 import BudgetCausal, ROOT
from exploration_r2_data import degrade
from train_exploration_r2 import new_model

BASE = ROOT/'research/4.temporal-consistency'
MANIFEST = BASE/'data/exploration-r32-training/manifest.json'
DECODE_AUDIT = BASE/'results/exploration-r32-decode/audit.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def models(seed, device):
    """Original pretrained model and identically initialized spatial/causal controls."""
    torch.manual_seed(seed)
    original = new_model(torch.device('cpu'))
    causal = BudgetCausal(original)
    weight = original.body.block1.weight.detach().double()
    dw, pw = causal.backbone.body.block1
    reconstructed = torch.zeros_like(weight)
    with torch.no_grad():
        dw.bias.zero_()
        pw.bias.copy_(original.body.block1.bias)
        for channel in range(12):
            u, s, vh = torch.linalg.svd(weight[:, channel].reshape(12, 9), full_matrices=False)
            left, right = u[:, 0]*s[0].sqrt(), vh[0]*s[0].sqrt()
            pw.weight[:, channel, 0, 0].copy_(left)
            dw.weight[channel, 0].copy_(right.reshape(3, 3))
            reconstructed[:, channel] = (left[:, None]*right[None]).reshape(12, 3, 3)
    spatial = BudgetCausal(original, temporal=False)
    spatial.load_state_dict(causal.state_dict())
    assert all(torch.equal(p, spatial.state_dict()[name]) for name, p in causal.state_dict().items())
    info = {'body_kernel_relative_frobenius_error': ((reconstructed-weight).norm()/weight.norm()).item(),
            'body_initialization': 'per-input-channel best rank1 SVD; approximate, not function preserving',
            'spatial_causal_initial_weights_equal': True,
            'official_weights_sha256': digest(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl')}
    return {name: model.to(device).train() for name, model in
            [('original', original), ('spatial', spatial), ('causal', causal)]}, info


class TrainingPool:
    def __init__(self):
        manifest = json.loads(MANIFEST.read_text())  # No partial-download fallback.
        selection = BASE/'data/exploration-r32-training/selection.json'
        registered = json.loads(selection.read_text())
        audit = json.loads(DECODE_AUDIT.read_text())
        assert audit['passed'] and audit['manifest_sha256'] == digest(MANIFEST)
        assert manifest['selection_sha256'] == digest(selection)
        assert manifest['videos'] == 57 and len(manifest['frames']) == 3428
        assert set(manifest['excluded_development_videos']) == {'bear', 'lucia', 'walking'}
        assert manifest['validation_image_members_read'] == 0
        self.scenes = {name: [] for name in registered['training_videos']}
        for row in manifest['frames']:
            path = ROOT/row['path']
            assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
            self.scenes[row['scene']].append(row)
        for scene, records in self.scenes.items():
            records.sort(key=lambda row: row['name'])
            assert len(records) == registered['frame_counts'][scene] and len(records) >= 4
        self.names = sorted(self.scenes)
        self.cache = OrderedDict()

    def image(self, row):
        path = ROOT/row['path']
        if row['path'] not in self.cache:
            assert digest(path) == row['sha256']
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            assert bgr is not None and bgr.shape[0] == 480 and bgr.shape[2] == 3
            self.cache[row['path']] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.cache.move_to_end(row['path'])
        image = self.cache[row['path']]
        if len(self.cache) > 128:
            self.cache.popitem(last=False)
        return image

    def batch(self, seed, step):
        rng = np.random.default_rng(np.random.SeedSequence([34, seed, step]))
        inputs, targets, identities = [], [], []
        for _ in range(4):
            name = self.names[int(rng.integers(len(self.names)))]
            records = self.scenes[name]
            start = int(rng.integers(len(records)-3))
            selected = records[start:start+4]
            first = self.image(selected[0])
            y, x = int(rng.integers(first.shape[0]-127)), int(rng.integers(first.shape[1]-127))
            horizontal, vertical = bool(rng.integers(2)), bool(rng.integers(2))
            photons = int(rng.choice([20, 80]))
            clean = []
            for row in selected:
                image = self.image(row)
                assert image.shape == first.shape
                patch = image[y:y+128, x:x+128]
                if horizontal:
                    patch = patch[:, ::-1]
                if vertical:
                    patch = patch[::-1]
                clean.append(patch.astype(np.float32))
            targets.append(np.stack(clean))
            inputs.append(np.stack([degrade(frame, rng, photons=photons, jpeg=False) for frame in clean]))
            identities.append({'video': name, 'members': [row['name'] for row in selected],
                               'xy': [x, y], 'flip_horizontal': horizontal, 'flip_vertical': vertical,
                               'photons': photons})
        low = torch.from_numpy(np.stack(inputs).transpose(0, 1, 4, 2, 3).copy()/255)
        target = torch.from_numpy(np.stack(targets).transpose(0, 1, 4, 2, 3).copy()/255)
        return low, target, identities
