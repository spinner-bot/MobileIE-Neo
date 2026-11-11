"""Closed-form train-only calibration to distinguish simple remapping from research gain."""
import argparse
from pathlib import Path
import time

import cv2
import numpy as np

from exploration_r2_data import evaluate
from explore_temporal_controls import save_json, sha256
from prepare_cache_r4 import CACHE_INDEX, load_cache


def fit_mappings(pairs, regularization=1e-3):
    xtx, xty = np.zeros((4, 4)), np.zeros((4, 3))
    count = 0
    for prediction, target in pairs:
        x, y = prediction.astype(np.float64).reshape(-1, 3)/255, target.astype(np.float64).reshape(-1, 3)/255
        design = np.concatenate([x, np.ones((len(x), 1))], 1)
        xtx += design.T@design
        xty += design.T@y
        count += len(x)
    if not count:
        raise ValueError('training pairs required')
    penalty = regularization*count
    identity = np.concatenate([np.eye(3), np.zeros((1, 3))], 0)
    affine = np.linalg.solve(xtx+penalty*np.eye(4), xty+penalty*identity)
    gains = np.maximum((np.diag(xty[:3])+penalty)/(np.diag(xtx[:3, :3])+penalty), 0)
    gain = max(float((np.trace(xty[:3])+penalty)/(np.trace(xtx[:3, :3])+penalty)), 0.)
    return {'scalar_gain': gain, 'channel_gain': gains.tolist(), 'rgb_affine': affine.tolist(), 'training_pixels': count}


def transform(image, kind, coefficients):
    x = image.astype(np.float64)/255
    if kind == 'scalar_gain':
        result = x*coefficients[kind]
    elif kind == 'channel_gain':
        result = x*np.asarray(coefficients[kind])
    elif kind == 'rgb_affine':
        matrix = np.asarray(coefficients[kind])
        result = x@matrix[:3]+matrix[3]
    else:
        raise ValueError(kind)
    return (np.clip(result, 0, 1)*255).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    start = time.perf_counter()
    training = load_cache('train')
    coefficients = fit_mappings((prediction, gt) for data in training.values()
                                for prediction, gt in zip(data['pretrained'], data['gt']))
    save_json(out/'coefficients.json', {**coefficients, 'ridge_per_pixel': 1e-3,
        'identity_centered': True, 'fit_split': 'train only', 'cache_index_sha256': sha256(CACHE_INDEX)})
    rows = []
    for domain in ['test', 'legacy_synthetic']:
        for scene, data in load_cache(domain).items():
            for kind in ['scalar_gain', 'channel_gain', 'rgb_affine']:
                output = [transform(frame, kind, coefficients) for frame in data['pretrained']]
                metrics, _ = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                rows.append({'domain': domain, 'scene': scene, 'method': kind, **metrics})
    save_json(out/'validation.json', rows)
    save_json(out/'runtime.json', {'wall_seconds': time.perf_counter()-start, 'rows': len(rows)})


if __name__ == '__main__':
    main()
