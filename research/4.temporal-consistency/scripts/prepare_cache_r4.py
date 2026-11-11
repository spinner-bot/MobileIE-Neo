"""Shared immutable real-video arrays and motion controls for R4 experiments."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from exploration_r2_data import ROOT
from explore_state_r2 import forward_backward
from explore_temporal_controls import inference, save_json, sha256
from real_data_r4 import MANIFEST, load_sequences
from train_exploration_r2 import new_model

CACHE_INDEX = ROOT/'research/4.temporal-consistency/results/exploration-r4-cache/index.json'


def load_cache(split, index=CACHE_INDEX):
    metadata = json.loads(Path(index).read_text())
    result = {}
    for row in metadata['scenes']:
        if row['split'] != split:
            continue
        path = ROOT/row['path']
        if sha256(path) != row['sha256']:
            raise ValueError('cache checksum mismatch')
        with np.load(path, allow_pickle=False) as archive:
            result[row['scene']] = {key: archive[key] for key in archive.files}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cache = ROOT/'.cache/experiments/exploration-r4'
    cache.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    net = new_model(device)
    rows = []
    start = time.perf_counter()
    for split in ['train', 'test']:
        sequences = load_sequences(split)
        for scene, arrays in sequences.items():
            raw, _ = inference(net, arrays['input'], device, 1)
            values = {'low': np.stack(arrays['input']), 'gt': np.stack(arrays['gt']), 'pretrained': np.stack(raw)}
            for label, inputs in [('reference', arrays['gt']), ('low', arrays['input']),
                                  ('gamma', [np.sqrt(frame/255)*255 for frame in arrays['input']]), ('enhanced', raw)]:
                pairs = [forward_backward(inputs[t-1], inputs[t]) for t in range(1, len(inputs))]
                values[label+'_flow'] = np.stack([p[0] for p in pairs])
                values[label+'_mask'] = np.stack([p[1] for p in pairs])
            path = cache/(scene+'.npz')
            np.savez_compressed(path, **values)
            row = {'scene': scene, 'split': split, 'path': str(path.relative_to(ROOT)), 'sha256': sha256(path),
                   'frames': len(raw), 'input_mean': float(values['low'].mean()), 'target_mean': float(values['gt'].mean())}
            rows.append(row)
            print(json.dumps(row), flush=True)
    save_json(out/'index.json', {'scenes': rows, 'source_manifest_sha256': sha256(MANIFEST),
        'pretrained_sha256': sha256(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'), 'size': [480, 270],
        'cudnn_deterministic': True, 'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32,
        'motion_scope': 'reference flow only for evaluation/training supervision; candidates use shared official enhanced-domain motion'})
    save_json(out/'runtime.json', {'wall_seconds': time.perf_counter()-start})


if __name__ == '__main__':
    main()
