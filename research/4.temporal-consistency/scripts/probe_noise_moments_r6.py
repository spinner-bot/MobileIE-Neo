"""Independent-calibration FST oracle diagnostics; static repeats, not video results."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from causal_fst_r5 import CausalFST
from exploration_r2_data import MANIFEST, ROOT, degrade, load_images
from explore_temporal_controls import save_json, sha256
from noise_moments_r6 import central_statistics, forward_probe, rgb_statistics
from train_exploration_r2 import new_model

INTERVENTIONS = ['raw', 'debias_head', 'debias_body', 'linearize_head', 'linearize_body']


@torch.inference_mode()
def collect(model, samples, intervention='raw', statistics=None, capture=False):
    outputs, features = [], {'head': [], 'body': []}
    for offset in range(0, len(samples), 8):
        pred, values = forward_probe(model, samples[offset:offset+8], intervention, statistics)
        outputs.append(pred)
        if capture:
            for key in features:
                features[key].append(values[key])
    return torch.cat(outputs), {k: torch.cat(v) for k, v in features.items()} if capture else {}


def channel_rows(features, clean, calibration, module):
    rows = []
    a = float((module.weight1.double()*module.weight2.double()).detach())
    for label, samples in [('all64', features), ('first32', features[:32]), ('last32', features[32:])]:
        s = samples[..., 8:-8, 8:-8]
        stat = central_statistics(s)
        for channel in range(s.shape[1]):
            get = lambda key: stat[key][channel]
            total = float(get('total').mean())
            row = {'draw_subset': label, 'channel': channel, 'a': a,
                'linear_energy': float(get('linear').mean()), 'quadratic_energy': float(get('quadratic').mean()),
                'cross_energy': float(get('cross').mean()), 'total_square_variance': total,
                'actual_fst_variance_ideal_fp64': a*a*total,
                'formula_max_abs_over_max_direct': float(get('closure').abs().max()/get('total').abs().max().clamp_min(1e-12)),
                'jensen_bias_mean': a*float(get('variance').mean()),
                'jensen_bias_rms': abs(a)*float(get('variance').square().mean().sqrt()),
                'feature_mean_energy': float(get('mean').square().mean()),
                'feature_noise_variance': float(get('variance').mean()),
                'mean_shift_from_noiseless_feature_mse': float((get('mean')-clean[channel, 8:-8, 8:-8].double()).square().mean()),
                'calibration_to_eval_mean_mse': float((get('mean')-calibration['mean'][channel, 8:-8, 8:-8].double()).square().mean())}
            for name in ['linear', 'quadratic', 'cross']:
                row[name+'_ratio'] = row[name+'_energy']/total if total > 1e-12 else None
            if row['formula_max_abs_over_max_direct'] > 1e-8:
                raise AssertionError('empirical moment identity failed')
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cache = ROOT/'.cache/experiments/exploration-r6'
    cache.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(731)
    device = torch.device('cuda')
    models, checkpoints = {}, {}
    for name, scope, variant in [('official', None, None), ('r5_plain', 'exploration-r5', 'plain'),
                                  ('r5_spatial_post', 'exploration-r5c', 'spatial_post')]:
        base = new_model(device)
        if scope:
            path = ROOT/'.cache/checkpoints'/scope/f'{variant}-seed0'/'step1000.pth'
            net = CausalFST(base, variant).to(device).eval()
            net.load_state_dict(torch.load(path, map_location=device, weights_only=False)['model'])
        else:
            path, net = ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl', base
        models[name] = net
        checkpoints[name] = {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path),
                             'parameters': sum(p.numel() for p in net.parameters())}
    save_json(out/'config.json', {'draws_calibration': 64, 'draws_evaluation': 64, 'batch': 8,
        'patch': [128, 128], 'source_crop_yx': [192, 320], 'metric_margin': 8,
        'photons': [20, 80], 'read_sigma': .008, 'jpeg': [False, True], 'jpeg_quality': 70,
        'models': checkpoints, 'manifest_sha256': sha256(MANIFEST),
        'interventions': INTERVENTIONS, 'feature_moments': 'population central moments over evaluation draws, FP64',
        'oracle_statistics': 'independent calibration draws; mean and variance cast to FP32 for interventions',
        'git_revision_at_startup': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256_at_startup': {str(p.relative_to(ROOT)): sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'protocol_sha256': sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r6-protocol.md'),
        'cudnn_deterministic': True, 'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32,
        'scope': 'static synthetic conditional noise diagnosis, oracle not a deployable video method'})
    start, cache_rows = time.perf_counter(), []
    feature_count, rgb_count, reference_count = 0, 0, 0
    with torch.inference_mode():
        for index, (scene, image) in enumerate(load_images('exploratory_validation').items()):
            target = image[192:320, 320:448]
            clean = (target/255.)**2.2*.35
            for photons in [20, 80]:
                for jpeg in [False, True]:
                    arrays = {'target': target/255., 'clean_low': clean}
                    seeds = {'calibration': 806000+index*10000+photons*10+int(jpeg),
                             'evaluation': 906000+index*10000+photons*10+int(jpeg)}
                    for split, seed in seeds.items():
                        rng = np.random.default_rng(seed)
                        arrays[split] = np.stack([degrade(target, rng, photons, jpeg) for _ in range(64)]).transpose(0, 3, 1, 2).copy()/255.
                    path = cache/f'{scene}_p{photons}_jpeg{int(jpeg)}.npz'
                    np.savez_compressed(path, **arrays)
                    identity = {'scene': scene, 'photons': photons, 'jpeg': jpeg}
                    cache_rows.append({**identity, 'seeds': seeds, 'path': str(path.relative_to(ROOT)), 'sha256': sha256(path),
                        'array_sha256': {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in arrays.items()}})
                    calibration = torch.from_numpy(arrays['calibration']).to(device)
                    evaluation = torch.from_numpy(arrays['evaluation']).to(device)
                    target_tensor = torch.from_numpy(arrays['target'].transpose(2, 0, 1).copy()).to(device)
                    clean_tensor = torch.from_numpy(clean.transpose(2, 0, 1).copy())[None].to(device)
                    for name, net in models.items():
                        _, cf = collect(net, calibration, capture=True)
                        statistics = {k: {'mean': v.double().mean(0).float(),
                                          'variance': v.double().var(0, unbiased=False).float()} for k, v in cf.items()}
                        del cf
                        raw, ef = collect(net, evaluation, capture=True)
                        clean_output, clean_features = forward_probe(net, clean_tensor)
                        base = getattr(net, 'backbone', net)
                        for layer in ['head', 'body']:
                            for row in channel_rows(ef[layer], clean_features[layer][0], statistics[layer], getattr(base, layer)):
                                with (out/'feature_per_channel.jsonl').open('a') as stream:
                                    stream.write(json.dumps({**identity, 'model': name, 'layer': layer, **row}, allow_nan=False)+'\n')
                                feature_count += 1
                        del ef
                        for intervention in INTERVENTIONS:
                            pred = raw if intervention == 'raw' else collect(net, evaluation, intervention, statistics)[0]
                            for clipped in [False, True]:
                                value = pred.clamp(0, 1) if clipped else pred
                                metrics = rgb_statistics(value, target_tensor)
                                if metrics['closure_abs'] > 1e-10:
                                    raise AssertionError('RGB MSE decomposition failed')
                                with (out/'rgb_per_condition.jsonl').open('a') as stream:
                                    stream.write(json.dumps({**identity, 'model': name, 'intervention': intervention,
                                        'clipped': clipped, **metrics}, allow_nan=False)+'\n')
                                rgb_count += 1
                        mean_input_output, _ = forward_probe(net, evaluation.mean(0, keepdim=True))
                        references = {'mean_clipped_output': raw.clamp(0, 1).mean(0, keepdim=True),
                                      'model_mean_input': mean_input_output.clamp(0, 1),
                                      'model_noiseless_low': clean_output.clamp(0, 1)}
                        for label, pred in references.items():
                            metrics = rgb_statistics(pred, target_tensor)
                            with (out/'oracle_references.jsonl').open('a') as stream:
                                stream.write(json.dumps({**identity, 'model': name, 'reference': label,
                                    'psnr_rgb': metrics['mean_output_psnr'], 'mse_normalized': metrics['mse_normalized'],
                                    'scope': 'multi-observation or known noiseless input, not single-frame deployable'}, allow_nan=False)+'\n')
                            reference_count += 1
                        print(json.dumps({**identity, 'model': name, 'feature_rows': feature_count,
                                          'rgb_rows': rgb_count}), flush=True)
    save_json(out/'cache_index.json', {'conditions': cache_rows, 'draw_images': 3072,
                                     'scope': '24 fixed static conditions; source image repeats are correlated'})
    if (feature_count, rgb_count, reference_count) != (5184, 720, 216):
        raise ValueError('fixed diagnosis budget mismatch')
    save_json(out/'runtime.json', {'wall_seconds': time.perf_counter()-start,
        'feature_rows': feature_count, 'rgb_rows': rgb_count, 'oracle_reference_rows': reference_count,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    main()
