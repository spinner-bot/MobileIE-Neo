"""Train-only scalar history weight, same quality constraints as local gates."""
import argparse
from pathlib import Path
import time

import cv2
import numpy as np

from exploration_r2_data import evaluate
from explore_state_r2 import fuse
from explore_temporal_controls import save_json, sha256
from prepare_cache_r4 import CACHE_INDEX, load_cache
from reliability_r4 import quality_feasible


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    start = time.perf_counter()
    training = load_cache('train')
    alphas = [0., .05, .1, .2, .3, .5, .7, .9]
    trials, selected = [], []
    for motion in ['low', 'enhanced']:
        groups = {alpha: [] for alpha in alphas}
        for scene, data in training.items():
            raw = data['pretrained']
            reference, _ = evaluate(raw, data['gt'], data['reference_flow'], data['reference_mask'])
            for alpha in alphas:
                output = fuse(raw, data[motion+'_flow'], data[motion+'_mask'], alpha=alpha)
                metrics, _ = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                row = {'scene': scene, 'motion': motion, 'alpha': alpha,
                       'feasible': quality_feasible(metrics, reference), **metrics}
                groups[alpha].append(row); trials.append(row)
        feasible = [a for a in alphas if all(r['feasible'] for r in groups[a])]
        if not feasible:
            raise ValueError('zero-alpha fallback must be feasible')
        winner = min(feasible, key=lambda a: (float(np.mean([r['temporal_error'] for r in groups[a]])), a))
        selected.append({'motion': motion, 'alpha': winner})
    save_json(out/'calibration_trials.json', trials)
    save_json(out/'selected.json', {'configurations': selected, 'cache_index_sha256': sha256(CACHE_INDEX),
        'selection': 'all train clips satisfy same three quality constraints, minimize mean temporal error; no test tuning'})
    rows = []
    for domain in ['test', 'legacy_synthetic']:
        for scene, data in load_cache(domain).items():
            for config in selected:
                motion, alpha = config['motion'], config['alpha']
                output = fuse(data['pretrained'], data[motion+'_flow'], data[motion+'_mask'], alpha=alpha)
                metrics, _ = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                rows.append({'domain': domain, 'scene': scene, 'method': 'global_alpha_'+motion, 'alpha': alpha, **metrics})
    save_json(out/'validation.json', rows)
    save_json(out/'runtime.json', {'wall_seconds': time.perf_counter()-start, 'calibration_rows': len(trials), 'validation_rows': len(rows)})


if __name__ == '__main__':
    main()
