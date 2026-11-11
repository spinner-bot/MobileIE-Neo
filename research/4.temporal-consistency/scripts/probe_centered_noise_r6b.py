"""Mean-preserving oracle controls on exactly the previously hashed R6 draws."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from causal_fst_r5 import CausalFST
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256
from noise_moments_r6 import quadratic, rgb_statistics
from probe_noise_moments_r6 import collect
from train_exploration_r2 import new_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    previous = ROOT/'research/4.temporal-consistency/results/exploration-r6-moments'
    config = json.loads((previous/'config.json').read_text())
    old_rows = [json.loads(line) for line in (previous/'rgb_per_condition.jsonl').read_text().splitlines()]
    old_raw = {(r['scene'], r['photons'], r['jpeg'], r['model'], r['clipped']): r for r in old_rows if r['intervention'] == 'raw'}
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    models = {}
    for name, info in config['models'].items():
        path = ROOT/info['path']
        if sha256(path) != info['sha256']:
            raise ValueError('checkpoint changed')
        base = new_model(device)
        if name == 'official':
            net = base
        else:
            variant = 'spatial_post' if name.endswith('spatial_post') else 'plain'
            net = CausalFST(base, variant).to(device).eval()
            net.load_state_dict(torch.load(path, map_location=device, weights_only=False)['model'])
        models[name] = net
    save_json(out/'config.json', {'r6_config_sha256': sha256(previous/'config.json'),
        'r6_cache_index_sha256': sha256(previous/'cache_index.json'),
        'protocol_sha256': sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r6b-protocol.md'),
        'git_revision_at_startup': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256_at_startup': {str(p.relative_to(ROOT)): sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'cudnn_deterministic': True, 'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32,
        'interventions': ['centered_head', 'centered_body'], 'oracle': 'independent 64 calibration draws'})
    start, rgb_count, layer_count = time.perf_counter(), 0, 0
    repeats = []
    with torch.inference_mode():
        for meta in json.loads((previous/'cache_index.json').read_text())['conditions']:
            path = ROOT/meta['path']
            if sha256(path) != meta['sha256']:
                raise ValueError('input cache changed')
            with np.load(path, allow_pickle=False) as archive:
                calibration = torch.from_numpy(archive['calibration']).to(device)
                evaluation = torch.from_numpy(archive['evaluation']).to(device)
                target = torch.from_numpy(archive['target'].transpose(2, 0, 1).copy()).to(device)
            identity = {key: meta[key] for key in ['scene', 'photons', 'jpeg']}
            for name, net in models.items():
                _, features = collect(net, calibration, capture=True)
                statistics = {k: {'mean': v.double().mean(0).float(), 'variance': v.double().var(0, unbiased=False).float()}
                              for k, v in features.items()}
                del features
                raw, ef = collect(net, evaluation, capture=True)
                for clipped in [False, True]:
                    current = rgb_statistics(raw.clamp(0, 1) if clipped else raw, target)
                    reference = old_raw[(*identity.values(), name, clipped)]
                    delta = {k: abs(v-reference[k]) for k, v in current.items()}
                    if max(delta.values()) > 1e-6:
                        raise AssertionError('R6 raw repeat changed')
                    repeats.append({**identity, 'model': name, 'clipped': clipped, 'max_abs_by_metric': delta})
                backbone = getattr(net, 'backbone', net)
                for layer in ['head', 'body']:
                    module = getattr(backbone, layer)
                    original = quadratic(module, ef[layer]).double()[..., 8:-8, 8:-8]
                    changed = quadratic(module, ef[layer], 'centered', statistics[layer]).double()[..., 8:-8, 8:-8]
                    old_variance = original.var(0, unbiased=False).mean()
                    new_variance = changed.var(0, unbiased=False).mean()
                    mean_shift = (changed.mean(0)-original.mean(0)).square().mean()
                    row = {**identity, 'model': name, 'layer': layer,
                        'local_original_variance': float(old_variance), 'local_changed_variance': float(new_variance),
                        'local_variance_ratio': float(new_variance/old_variance.clamp_min(1e-30)),
                        'local_mean_shift_mse': float(mean_shift),
                        'mean_shift_mse_over_original_variance': float(mean_shift/old_variance.clamp_min(1e-30))}
                    with (out/'layer_per_condition.jsonl').open('a') as stream:
                        stream.write(json.dumps(row, allow_nan=False)+'\n')
                    layer_count += 1
                    pred, _ = collect(net, evaluation, 'centered_'+layer, statistics)
                    for clipped in [False, True]:
                        metrics = rgb_statistics(pred.clamp(0, 1) if clipped else pred, target)
                        with (out/'rgb_per_condition.jsonl').open('a') as stream:
                            stream.write(json.dumps({**identity, 'model': name, 'intervention': 'centered_'+layer,
                                'clipped': clipped, **metrics}, allow_nan=False)+'\n')
                        rgb_count += 1
                del ef, raw
                print(json.dumps({**identity, 'model': name, 'rgb_rows': rgb_count}), flush=True)
    if (rgb_count, layer_count, len(repeats)) != (288, 144, 144):
        raise ValueError('fixed diagnosis budget mismatch')
    save_json(out/'raw_repeat_audit.json', repeats)
    save_json(out/'runtime.json', {'wall_seconds': time.perf_counter()-start, 'rgb_rows': rgb_count, 'layer_rows': layer_count})


if __name__ == '__main__':
    main()
