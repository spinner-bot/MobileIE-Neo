"""Train-only static channel calibration for evaluating a stronger causal reference."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from channel_noise_r7 import channel_covariance, channel_linear_rgb, predicted_relative_gain, subsets
from exploration_r2_data import MANIFEST, ROOT, degrade, load_images
from explore_temporal_controls import save_json, sha256
from noise_moments_r6 import forward_probe, rgb_statistics
from rgb_noise_components_r6c import body_components
from train_exploration_r2 import new_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cache = ROOT/'.cache/experiments/exploration-r7-channel'
    cache.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    net = new_model(device)
    save_json(out/'config.json', {'split': 'train', 'images': 'Kodak01-06 from pinned R2 manifest',
        'crops_yx': [[64, 64], [192, 320]], 'patch': 128, 'draws': 64,
        'photons': [20, 80], 'jpeg': [False, True], 'read_sigma': .008, 'jpeg_quality': 70,
        'subset_size': 3, 'primary_gamma': .5, 'additional_gamma': 1., 'ranking': 'mean relative variance proxy gain over all48; ties lexicographic',
        'manifest_sha256': sha256(MANIFEST), 'weights_sha256': sha256(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'),
        'protocol_sha256': sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r7-protocol.md'),
        'git_revision_at_startup': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256_at_startup': {str(p.relative_to(ROOT)): sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'cudnn_deterministic': True, 'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32,
        'scope': 'independent static-noise, fixed-response proxy; no motion/quality or trained-method guarantee'})
    start, rows, cache_rows = time.perf_counter(), [], []
    with torch.inference_mode():
        for index, (scene, image) in enumerate(load_images('train').items()):
            for crop, (y, x) in enumerate([(64, 64), (192, 320)]):
                target = image[y:y+128, x:x+128]
                for photons in [20, 80]:
                    for jpeg in [False, True]:
                        seed = 1107000+index*10000+crop*1000+photons*10+int(jpeg)
                        rng = np.random.default_rng(seed)
                        low = np.stack([degrade(target, rng, photons, jpeg) for _ in range(64)]).transpose(0, 3, 1, 2).copy()/255.
                        path = cache/f'{scene}_crop{crop}_p{photons}_jpeg{int(jpeg)}.npz'
                        np.savez_compressed(path, low=low, target=target/255.)
                        identity = {'scene': scene, 'crop': crop, 'photons': photons, 'jpeg': jpeg}
                        cache_rows.append({**identity, 'seed': seed, 'path': str(path.relative_to(ROOT)), 'sha256': sha256(path),
                            'low_array_sha256': hashlib.sha256(low.tobytes()).hexdigest()})
                        inputs = torch.from_numpy(low).to(device)
                        pred, body, gates = [], [], []
                        for offset in range(0, 64, 8):
                            output, values = forward_probe(net, inputs[offset:offset+8])
                            pred.append(output); body.append(values['body']); gates.append(values['gate'])
                        pred, body, gates = torch.cat(pred), torch.cat(body), torch.cat(gates)
                        channel = channel_linear_rgb(net, body, gates)
                        _, check = body_components(net, body, gates, pred)
                        difference = float((sum(channel)-check['feature_linear_mean_gate']).abs().max())
                        if difference > 1e-9:
                            raise AssertionError('channel sum differs from exact body linear component')
                        cov, cross, variance = channel_covariance(channel, pred.clamp(0, 1))
                        target_tensor = torch.from_numpy((target/255.).transpose(2, 0, 1).copy()).to(device)
                        row = {**identity, 'channel_covariance': cov.cpu().tolist(), 'covariance_with_actual_rgb': cross.cpu().tolist(),
                               'actual_clipped_rgb_variance': float(variance), 'channel_sum_max_abs_error': difference,
                               'raw_diagnostic_quality': rgb_statistics(pred.clamp(0, 1), target_tensor)}
                        rows.append(row)
                        print(json.dumps({**identity, 'completed_conditions': len(rows)}), flush=True)
    if len(rows) != 48:
        raise ValueError('fixed calibration budget mismatch')
    ranking = []
    for subset in subsets():
        effects = []
        for row in rows:
            cov, cross = np.array(row['channel_covariance']), np.array(row['covariance_with_actual_rgb'])
            variance = row['actual_clipped_rgb_variance']
            effects.append({**{k: row[k] for k in ['scene', 'crop', 'photons', 'jpeg']},
                'gamma05_predicted_relative_variance_gain': float(predicted_relative_gain(cov, cross, variance, subset, .5)),
                'gamma1_predicted_relative_variance_gain': float(predicted_relative_gain(cov, cross, variance, subset, 1.))})
        gains = [r['gamma05_predicted_relative_variance_gain'] for r in effects]
        ranking.append({'channels_zero_based': list(subset), 'mean_gamma05_gain': float(np.mean(gains)),
            'worst_gamma05_gain': min(gains), 'positive_conditions': sum(v > 0 for v in gains),
            'mean_gamma1_gain': float(np.mean([r['gamma1_predicted_relative_variance_gain'] for r in effects])),
            'per_condition': effects})
    ranking.sort(key=lambda r: (-r['mean_gamma05_gain'], r['channels_zero_based']))
    for rank, row in enumerate(ranking, 1):
        row['rank'] = rank
    reference_subsets = {'first3': [0, 1, 2], 'uniform': [0, 4, 8],
                         'random109': sorted(np.random.default_rng(109).choice(12, 3, replace=False).tolist())}
    save_json(out/'per_condition.json', rows)
    save_json(out/'cache_index.json', {'conditions': cache_rows, 'static_conditions': 48, 'draw_images': 3072, 'split': 'train'})
    save_json(out/'all_subsets.json', ranking)
    save_json(out/'selection.json', {'selected': {k: v for k, v in ranking[0].items() if k != 'per_condition'},
        'references': {name: next({k: v for k, v in r.items() if k != 'per_condition'} for r in ranking if r['channels_zero_based'] == indices)
                       for name, indices in reference_subsets.items()},
        'scope': 'train-source independent-noise/fixed-response proxy; validate actual causal video behavior separately'})
    save_json(out/'runtime.json', {'wall_seconds': time.perf_counter()-start, 'conditions': len(rows), 'subsets': len(ranking),
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    main()
