"""Frozen checkpoint diagnostics: train/test fit, native resize and brightness."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from exploration_r2_data import ROOT, evaluate
from explore_temporal_controls import inference, save_json, sha256
from real_data_r4 import load_sequences, reference_fields
from train_exploration_r2 import new_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    models, provenance = {'pretrained': new_model(device)}, {}
    choices = [('synthetic_task', 'exploration-r2', 'task')]+[(v, 'exploration-r4', v) for v in ['real_task', 'real_anchor', 'real_temporal']]
    for label, cache, variant in choices:
        for seed in [0, 1]:
            path = ROOT/'.cache/checkpoints'/cache/f'{variant}-seed{seed}'/'last.pth'
            saved = torch.load(path, map_location=device, weights_only=False)
            if saved['step'] != 400:
                raise ValueError('final 400-update checkpoint required')
            net = new_model(device)
            net.load_state_dict(saved['model'])
            models[f'{label}-seed{seed}'] = net
            provenance[f'{label}-seed{seed}'] = {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}
    save_json(out/'config.json', {'models': provenance, 'sizes': [[320, 180], [480, 270]],
        'scope': 'raw outputs; paired deltas at each size, no checkpoint/hyperparameter selection'})
    start = time.perf_counter()
    count = 0
    with (out/'per_sequence.jsonl').open('w') as stream, (out/'per_frame.jsonl').open('w') as trace:
        for size in [(320, 180), (480, 270)]:
            for split in ['train', 'test']:
                for scene, data in load_sequences(split, size).items():
                    flow, mask = reference_fields(data['gt'])
                    for name, model in models.items():
                        output, _ = inference(model, data['input'], device, 1)
                        metrics, frames = evaluate(output, data['gt'], flow, mask)
                        identity = {'size': list(size), 'split': split, 'scene': scene, 'model': name}
                        row = {**identity, **metrics, 'output_rgb_mean': np.mean(np.stack(output), axis=(0, 1, 2)).tolist(),
                               'target_rgb_mean': np.mean(np.stack(data['gt']), axis=(0, 1, 2)).tolist()}
                        stream.write(json.dumps(row, allow_nan=False)+'\n')
                        for frame in frames:
                            trace.write(json.dumps({**identity, **frame}, allow_nan=False)+'\n')
                        count += 1
                    print(json.dumps({'size': size, 'split': split, 'scene': scene, 'rows': count}), flush=True)
    save_json(out/'runtime.json', {'wall_seconds': time.perf_counter()-start, 'rows': count})


if __name__ == '__main__':
    main()
