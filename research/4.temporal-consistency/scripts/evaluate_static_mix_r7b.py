"""Realized frozen filtering on hashed train-source static draws, not real videos."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256
from feature_mix_r7 import FeatureMix, SPECS
from frozen_mix_utils_r7 import predict, provenance
from noise_moments_r6 import rgb_statistics
from train_exploration_r2 import new_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    previous = ROOT/'research/4.temporal-consistency/results/exploration-r7-channel'
    calibration = json.loads((previous/'per_condition.json').read_text())
    old_raw = {(r['scene'], r['crop'], r['photons'], r['jpeg']): r['raw_diagnostic_quality'] for r in calibration}
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    base = new_model(device)
    models = {'pretrained': base, **{name: FeatureMix(base, **spec).to(device).eval() for name, spec in SPECS.items()}}
    save_json(out/'config.json', {**provenance(), 'cache_index_sha256': sha256(previous/'cache_index.json'),
        'specs': SPECS, 'raw_batch': 1, 'scope': 'same train-only IID-noise draws, 2-frame artificial static streams; no heldout or natural video claim'})
    start, count, repeat, first = time.perf_counter(), 0, [], []
    with torch.inference_mode():
        for meta in json.loads((previous/'cache_index.json').read_text())['conditions']:
            path = ROOT/meta['path']
            if sha256(path) != meta['sha256']:
                raise ValueError('input cache changed')
            with np.load(path, allow_pickle=False) as archive:
                inputs = torch.from_numpy(archive['low']).to(device)
                target = torch.from_numpy(archive['target'].transpose(2, 0, 1).copy()).to(device)
            identity = {k: meta[k] for k in ['scene', 'crop', 'photons', 'jpeg']}
            raw = predict(base, inputs)
            raw_metrics = rgb_statistics(raw.clamp(0, 1), target)
            reference = old_raw[(meta['scene'], meta['crop'], meta['photons'], meta['jpeg'])]
            repeat.append({**identity, 'batch1_minus_batch8': {k: v-reference[k] for k, v in raw_metrics.items()}})
            methods = {}
            for method, model in models.items():
                prediction = raw if method == 'pretrained' else predict(model, inputs)
                difference = float((prediction[0]-raw[0]).abs().max())
                if difference > 2e-6:
                    raise AssertionError('first-frame function changed')
                value = prediction.clamp(0, 1)
                metrics = rgb_statistics(value, target)
                adjacent = (value[1:]-value[:-1])[..., 8:-8, 8:-8]
                row = {**identity, 'method': method, **metrics, 'adjacent_static_mae_255': float(adjacent.abs().mean()*255),
                    'first_frame_max_abs_difference': difference,
                    'state_storage_bytes_128': model.state_storage_bytes() if hasattr(model, 'state_storage_bytes') else 0}
                with (out/'per_condition.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                methods[method] = metrics
                first.append(difference)
                count += 1
            for key in methods['body_pre_selected_g1']:
                if methods['body_pre_selected_g1'][key] != methods['body_post_selected_g1'][key]:
                    raise AssertionError('full replacement pre/post should agree at fixed weights')
            print(json.dumps({**identity, 'rows': count}), flush=True)
    if count != 432:
        raise ValueError('static control budget mismatch')
    save_json(out/'batch_repeat_audit.json', repeat)
    save_json(out/'runtime.json', {'wall_seconds': time.perf_counter()-start, 'rows': count,
        'max_first_frame_difference': max(first), 'full_replacement_pre_post_all_metrics_exact': True})


if __name__ == '__main__':
    main()
