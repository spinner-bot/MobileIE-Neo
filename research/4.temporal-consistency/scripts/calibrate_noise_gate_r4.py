"""Heuristic Haar noise-scale normalized discrepancy, calibrated only on train."""
import argparse
from pathlib import Path
import time

import cv2
import numpy as np

from exploration_r2_data import evaluate
from explore_state_r2 import fuse
from explore_temporal_controls import save_json, sha256
from prepare_cache_r4 import CACHE_INDEX, load_cache
from reliability_r4 import discrepancy, masks_at_threshold, quality_feasible


def haar_noise_scale(image):
    x = image.astype(np.float32)
    h, w = x.shape[:2]
    if min(h, w) < 2:
        raise ValueError('at least 2x2 image required')
    x = x[:h-h%2, :w-w%2]
    hh = (x[::2, ::2]-x[::2, 1::2]-x[1::2, ::2]+x[1::2, 1::2])/2
    return float(np.median(np.abs(hh))/.67448975)


def normalized_scores(data):
    frames = data['pretrained']
    sigma = [haar_noise_scale(frame) for frame in frames]
    scores = discrepancy(frames, data['enhanced_flow'], 'photo')
    normalized = [(error/max(1., np.hypot(sigma[t], sigma[t+1])), valid)
                  for t, (error, valid) in enumerate(scores)]
    return normalized, sigma


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    start = time.perf_counter()
    thresholds = [0., .25, .5, 1., 2., 4., 8., float('inf')]
    groups = {tau: [] for tau in thresholds}
    scales, trials = [], []
    for scene, data in load_cache('train').items():
        scores, sigma = normalized_scores(data)
        scales.append({'split': 'train', 'scene': scene, 'sigma': sigma})
        reference, _ = evaluate(data['pretrained'], data['gt'], data['reference_flow'], data['reference_mask'])
        for tau in thresholds:
            masks = masks_at_threshold(scores, data['enhanced_mask'], tau)
            output = fuse(data['pretrained'], data['enhanced_flow'], masks)
            metrics, _ = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
            row = {'scene': scene, 'threshold': None if np.isinf(tau) else tau,
                   'feasible': quality_feasible(metrics, reference), **metrics}
            groups[tau].append(row); trials.append(row)
    feasible = [tau for tau in thresholds if all(r['feasible'] for r in groups[tau])]
    if not feasible:
        raise ValueError('no-history fallback must be feasible')
    selected = min(feasible, key=lambda tau: (float(np.mean([r['temporal_error'] for r in groups[tau]])), tau))
    save_json(out/'selected.json', {'threshold': None if np.isinf(selected) else selected,
        'threshold_null_means': 'infinity', 'cache_index_sha256': sha256(CACHE_INDEX),
        'scope': 'heuristic output noise scale; not calibrated physical variance; train-only selection'})
    save_json(out/'calibration_trials.json', trials)
    rows = []
    for domain in ['test', 'legacy_synthetic']:
        for scene, data in load_cache(domain).items():
            scores, sigma = normalized_scores(data)
            scales.append({'split': domain, 'scene': scene, 'sigma': sigma})
            masks = masks_at_threshold(scores, data['enhanced_mask'], selected)
            output = fuse(data['pretrained'], data['enhanced_flow'], masks)
            metrics, _ = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
            rows.append({'domain': domain, 'scene': scene, 'method': 'noise_normalized_photo',
                         'coverage': float(np.mean([m.mean() for m in masks])), **metrics})
    save_json(out/'noise_scales.json', scales)
    save_json(out/'validation.json', rows)
    save_json(out/'runtime.json', {'wall_seconds': time.perf_counter()-start, 'calibration_rows': len(trials), 'validation_rows': len(rows)})


if __name__ == '__main__':
    main()
