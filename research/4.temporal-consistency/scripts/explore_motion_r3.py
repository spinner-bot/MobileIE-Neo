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
from paired_fst_r3 import InputPairNetwork, TrainablePairFST
from train_products_r3 import streaming
from xiph_motion_r3 import corrupt, read_y4m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--include-r3', action='store_true')
    parser.add_argument('--r3-only', action='store_true')
    parser.add_argument('--r3-variants', default='cross,mean_square')
    parser.add_argument('--frozen-fusion', action='store_true')
    args = parser.parse_args()
    if args.r3_only and not args.include_r3:
        parser.error('--r3-only requires --include-r3')
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
    recurrent = {}
    if args.include_r3:
        for variant in args.r3_variants.split(','):
            if variant not in ['cross', 'mean_square', 'input_mean']:
                raise ValueError('unknown R3 variant')
            for seed in [0, 1]:
                path = ROOT/'.cache/checkpoints/exploration-r3'/f'{variant}-seed{seed}'/'last.pth'
                saved = torch.load(path, map_location=device, weights_only=False)
                if saved['step'] != 400:
                    raise ValueError('only final preregistered checkpoint is eligible')
                model = new_model(device)
                if variant == 'input_mean':
                    model = InputPairNetwork(model)
                else:
                    model.body = TrainablePairFST(model.body, variant)
                model.load_state_dict(saved['model'])
                key = f'{variant}-seed{seed}'
                recurrent[key] = model
                provenance[key] = {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path), 'steps': saved['step']}
    save_json(out/'config.json', {'models': provenance, 'manifest_sha256': sha256(args.manifest),
        'frames': 24, 'read_rho': [0., .9], 'input_seeds': [0, 1], 'jpeg_quality': 70,
        'temporal_metric': 'reference-Farneback+FB and GT-relative frame difference; motion is estimated, not true',
        'feature_mask_erosion_radius': 8, 'fusion_alpha': .7})
    baseline_index = {}
    frozen_repeat_max = 0.
    if args.frozen_fusion:
        path = ROOT/'research/4.temporal-consistency/results/exploration-r3-motion-baseline/per_sequence.jsonl'
        for line in path.read_text().splitlines():
            row = json.loads(line)
            baseline_index[(row['scene'], row['read_rho'], row['input_seed'], row['method'])] = row
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
                    input_hash = hashlib.sha256(np.stack(low).tobytes()).hexdigest()
                    pairs = [forward_backward(low[t-1], low[t]) for t in range(1, len(low))]
                    flow, masks = [p[0] for p in pairs], [p[1] for p in pairs]
                    methods = {}
                    for name, model in ([] if args.r3_only else models.items()):
                        raw, _ = inference(model, low, device, 1)
                        methods[name+'__raw'] = raw
                        methods[name+'__fb'] = fuse(raw, flow, masks)
                    feature_masks = [cv2.erode(m.astype(np.uint8), np.ones((17, 17), np.uint8),
                                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool) for m in masks]
                    for name, model in recurrent.items():
                        gamma = float(model.body.gamma.detach())
                        for state in ['history', 'history_off']:
                            with torch.no_grad():
                                model.body.gamma.fill_(gamma if state == 'history' else 0.)
                            raw = streaming(model, low, flow, feature_masks, device)
                            methods[name+'__'+state+'__raw'] = raw
                            methods[name+'__'+state+'__fb'] = fuse(raw, flow, masks)
                        with torch.no_grad():
                            model.body.gamma.fill_(gamma)
                    if args.frozen_fusion:
                        for mode in ['cross', 'mean_square']:
                            raw = intervention(models['pretrained'], low, flow, feature_masks, 'body', mode, device)
                            checked, _ = evaluate(raw, gt, reference_flow, reference_mask)
                            ref = baseline_index[(clip['scene'], rho, seed, 'low_fb__body_'+mode)]
                            frozen_repeat_max = max(frozen_repeat_max, max(abs(checked[k]-ref[k]) for k in checked))
                            methods['pretrained_body_'+mode+'__fb'] = fuse(raw, flow, masks)
                    for label, flows, valid in ([] if args.r3_only else [('low_fb', flow, masks), ('reference_fb', reference_flow, reference_mask)]):
                        valid = [cv2.erode(m.astype(np.uint8), np.ones((17, 17), np.uint8),
                                 borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool) for m in valid]
                        for mode in ['cross', 'mean_square']:
                            methods[label+'__body_'+mode] = intervention(models['pretrained'], low, flows, valid, 'body', mode, device)
                        methods[label+'__input_pair'] = inference(models['pretrained'], pair_blend(low, flows, valid), device, 1)[0]
                        methods[label+'__output_pair'] = pair_blend(methods['pretrained__raw'], flows, valid)
                    for method, output in methods.items():
                        metrics, frames = evaluate(output, gt, reference_flow, reference_mask)
                        identity = {'scene': clip['scene'], 'read_rho': rho, 'input_seed': seed, 'method': method,
                                    'input_sha256': input_hash}
                        stream.write(json.dumps({**identity, **metrics,
                            'low_flow_coverage': float(np.mean([m.mean() for m in masks])),
                            'reference_flow_coverage': float(np.mean([m.mean() for m in reference_mask]))}, allow_nan=False)+'\n')
                        for frame in frames:
                            trace.write(json.dumps({**identity, **frame}, allow_nan=False)+'\n')
                        count += 1
                    print(json.dumps({'scene': clip['scene'], 'rho': rho, 'seed': seed, 'rows': count}), flush=True)
    save_json(out/'runtime.json', {'wall_seconds': time.perf_counter()-start, 'sequence_rows': count,
              'frozen_raw_repeat_max_abs_difference': frozen_repeat_max if args.frozen_fusion else None,
              'frozen_raw_repeat_rows': 16 if args.frozen_fusion else 0})


if __name__ == '__main__':
    main()
