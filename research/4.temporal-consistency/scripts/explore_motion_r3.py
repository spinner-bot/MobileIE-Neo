"""Actual motion, synthetic low-light degradation; reference-flow is not GT flow."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from exploration_r2_data import ROOT, evaluate
from explore_products_r3 import intervention, pair_blend
from explore_state_r2 import forward_backward, fuse
from explore_temporal_controls import inference, save_json, sha256
from train_exploration_r2 import new_model
from xiph_motion_r3 import corrupt, read_y4m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    device = torch.device('cuda')
    models = {'pretrained': new_model(device)}
    provenance = {}
    for label, cache, variant in [('task', 'exploration-r2', 'task'),
                                  ('T1', 'exploration-r2-lambda10', 'task_temporal')]:
        for seed in [0, 1]:
            path = ROOT/'.cache/checkpoints'/cache/f'{variant}-seed{seed}'/'last.pth'
            saved = torch.load(path, map_location=device, weights_only=False)
            model = new_model(device)
            model.load_state_dict(saved['model'])
            key = f'{label}-seed{seed}'
            models[key] = model
            provenance[key] = {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path), 'steps': saved['step']}
    save_json(out/'config.json', {'models': provenance, 'manifest_sha256': sha256(args.manifest),
        'frames': 24, 'read_rho': [0., .9], 'input_seeds': [0, 1], 'jpeg_quality': 70,
        'temporal_metric': 'reference-Farneback+FB and GT-relative frame difference; motion is estimated, not true',
        'feature_mask_erosion_radius': 8, 'fusion_alpha': .7})
    manifest = json.loads(Path(args.manifest).read_text())
    count = 0
    start = time.perf_counter()
    with (out/'per_sequence.jsonl').open('w') as stream, (out/'per_frame.jsonl').open('w') as trace:
        for clip in manifest['clips']:
            path = ROOT/clip['path']
            if hashlib.sha256(path.read_bytes()).hexdigest() != clip['sha256']:
                raise ValueError('clip hash mismatch')
            gt, _ = read_y4m(path)
            reference_pairs = [forward_backward(gt[t-1], gt[t]) for t in range(1, len(gt))]
            reference_flow, reference_mask = [p[0] for p in reference_pairs], [p[1] for p in reference_pairs]
            for rho in [0., .9]:
                for seed in [0, 1]:
                    low = corrupt(gt, seed, rho)
                    pairs = [forward_backward(low[t-1], low[t]) for t in range(1, len(low))]
                    flow, masks = [p[0] for p in pairs], [p[1] for p in pairs]
                    methods = {}
                    for name, model in models.items():
                        raw, _ = inference(model, low, device, 1)
                        methods[name+'__raw'] = raw
                        methods[name+'__fb'] = fuse(raw, flow, masks)
                    for label, flows, valid in [('low_fb', flow, masks), ('reference_fb', reference_flow, reference_mask)]:
                        valid = [cv2.erode(m.astype(np.uint8), np.ones((17, 17), np.uint8),
                                 borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool) for m in valid]
                        for mode in ['cross', 'mean_square']:
                            methods[label+'__body_'+mode] = intervention(models['pretrained'], low, flows, valid, 'body', mode, device)
                        methods[label+'__input_pair'] = inference(models['pretrained'], pair_blend(low, flows, valid), device, 1)[0]
                        methods[label+'__output_pair'] = pair_blend(methods['pretrained__raw'], flows, valid)
                    for method, output in methods.items():
                        metrics, frames = evaluate(output, gt, reference_flow, reference_mask)
                        identity = {'scene': clip['scene'], 'read_rho': rho, 'input_seed': seed, 'method': method}
                        stream.write(json.dumps({**identity, **metrics,
                            'low_flow_coverage': float(np.mean([m.mean() for m in masks])),
                            'reference_flow_coverage': float(np.mean([m.mean() for m in reference_mask]))}, allow_nan=False)+'\n')
                        for frame in frames:
                            trace.write(json.dumps({**identity, **frame}, allow_nan=False)+'\n')
                        count += 1
                    print(json.dumps({'scene': clip['scene'], 'rho': rho, 'seed': seed, 'rows': count}), flush=True)
    save_json(out/'runtime.json', {'wall_seconds': time.perf_counter()-start, 'sequence_rows': count})


if __name__ == '__main__':
    main()
