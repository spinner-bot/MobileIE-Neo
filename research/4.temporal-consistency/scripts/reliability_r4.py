"""Train-calibrated traditional discrepancy gates; no ground truth at inference."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from exploration_r2_data import evaluate
from explore_state_r2 import fuse
from explore_temporal_controls import save_json, sha256
from prepare_cache_r4 import CACHE_INDEX, load_cache
from video_metrics import warp_previous

THRESHOLDS = {'photo': [0., 1., 2., 4., 8., 16., 32., float('inf')],
              'census': [0., .05, .1, .2, .4, 1., float('inf')]}


def census(image):
    gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    pad = cv2.copyMakeBorder(gray, 2, 2, 2, 2, cv2.BORDER_REFLECT_101)
    bits = [pad[2+dy:2+dy+h, 2+dx:2+dx+w] >= gray
            for dy in range(-2, 3) for dx in range(-2, 3) if dy or dx]
    return np.stack(bits, -1).astype(np.float32)


def discrepancy(frames, flows, family):
    features = [census(frame) for frame in frames] if family == 'census' else frames
    result = []
    for t, flow in enumerate(flows):
        previous, valid = warp_previous(features[t], flow)
        current = features[t+1]
        if family == 'census':
            error = ((previous > .5) != (current > .5)).mean(-1).astype(np.float32)
            valid[:2] = valid[-2:] = False
            valid[:, :2] = valid[:, -2:] = False
        else:
            error = np.abs(previous-current).mean(-1)
        result.append((error, valid))
    return result


def masks_at_threshold(scores, masks, threshold):
    return [valid & fb & (error < threshold) for (error, valid), fb in zip(scores, masks)]


def quality_feasible(row, reference):
    return (row['psnr_rgb'] >= reference['psnr_rgb']-.05 and row['ssim_rgb'] >= reference['ssim_rgb']-.005
            and row['edge_mae'] <= reference['edge_mae']+.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    train = load_cache('train')
    start = time.perf_counter()
    selected, trials = [], []
    for motion in ['low', 'enhanced']:
        for family, thresholds in THRESHOLDS.items():
            groups = {tau: [] for tau in thresholds}
            for scene, data in train.items():
                raw = data['pretrained']
                ref, _ = evaluate(raw, data['gt'], data['reference_flow'], data['reference_mask'])
                scores = discrepancy(raw, data[motion+'_flow'], family)
                for tau in thresholds:
                    masks = masks_at_threshold(scores, data[motion+'_mask'], tau)
                    output = fuse(raw, data[motion+'_flow'], masks)
                    metrics, _ = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                    row = {'scene': scene, 'motion': motion, 'family': family, 'threshold': None if np.isinf(tau) else tau,
                           'feasible': quality_feasible(metrics, ref), 'coverage': float(np.mean([m.mean() for m in masks])), **metrics}
                    groups[tau].append(row); trials.append(row)
            feasible = [tau for tau in thresholds if all(r['feasible'] for r in groups[tau])]
            if not feasible:
                raise ValueError('raw fallback must be feasible')
            winner = min(feasible, key=lambda tau: (float(np.mean([r['temporal_error'] for r in groups[tau]])), tau))
            selected.append({'motion': motion, 'family': family, 'threshold': None if np.isinf(winner) else winner,
                             'train_temporal_error': float(np.mean([r['temporal_error'] for r in groups[winner]]))})
    save_json(out/'calibration_trials.json', trials)
    save_json(out/'selected.json', {'configurations': selected, 'cache_index_sha256': sha256(CACHE_INDEX),
        'quality_constraints': {'psnr_max_drop': .05, 'ssim_max_drop': .005, 'edge_max_increase': .05},
        'scope': 'train-only selection, all four training clips constrained; no validation threshold tuning'})
    rows = []
    for domain in ['test', 'legacy_synthetic']:
        for scene, data in load_cache(domain).items():
            raw = data['pretrained']
            for config in selected:
                motion, family = config['motion'], config['family']
                tau = float('inf') if config['threshold'] is None else config['threshold']
                scores = discrepancy(raw, data[motion+'_flow'], family)
                masks = masks_at_threshold(scores, data[motion+'_mask'], tau)
                output = fuse(raw, data[motion+'_flow'], masks)
                metrics, _ = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                rows.append({'domain': domain, 'scene': scene, 'method': motion+'__'+family,
                    'threshold': config['threshold'], 'coverage': float(np.mean([m.mean() for m in masks])), **metrics})
    save_json(out/'validation.json', rows)
    save_json(out/'runtime.json', {'wall_seconds': time.perf_counter()-start, 'calibration_rows': len(trials), 'validation_rows': len(rows)})
    print(json.dumps({'selected': selected, 'validation_rows': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
