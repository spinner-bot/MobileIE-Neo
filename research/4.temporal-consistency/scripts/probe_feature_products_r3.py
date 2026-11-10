"""Paired actual-feature moments; static diagnostic, not a video model result."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from exploration_r2_data import ROOT, load_images
from explore_temporal_controls import save_json, sha256
from train_exploration_r2 import new_model


def observations(target, seed, shared_read, jpeg, count=64):
    rng = np.random.default_rng(seed)
    clean = (target / 255.)**2.2*.35
    common = rng.normal(0, .008, clean.shape)
    result = []
    for _ in range(count):
        read = common if shared_read else rng.normal(0, .008, clean.shape)
        low = rng.poisson(clean*40)/40 + read
        low = (np.clip(low, 0, 1)*255).astype(np.uint8)
        if jpeg:
            ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(low, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                raise RuntimeError('JPEG encoding failed')
            low = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        result.append(low.astype(np.float32)/255)
    return clean, result


def moments(clean, samples):
    """Inputs [C,H,W] and [2*pairs,C,H,W], report each channel separately.

    Centering is over draws at each fixed spatial location. Shared fixed-pattern
    noise consequently contributes to mean bias, not draw covariance. Report
    uncentered pair-error products separately so it is not hidden.
    """
    clean, samples = clean.double(), samples.double()
    a, b = samples[::2], samples[1::2]
    error = samples-clean
    mean = error.mean(0)
    variance = ((error-mean)**2).mean(0)
    ea, eb = a-clean, b-clean
    centered_cov = ((ea-ea.mean(0))*(eb-eb.mean(0))).mean(0)
    ref = clean**2
    products = {'self': a*a, 'cross': a*b, 'mean_square': ((a+b)/2)**2}
    rows = []
    for channel in range(clean.shape[0]):
        row = {'channel': channel, 'feature_clean_energy': float((clean[channel]**2).mean()),
               'feature_bias_mse': float((mean[channel]**2).mean()),
               'feature_variance': float(variance[channel].mean()),
               'pair_centered_covariance': float(centered_cov[channel].mean()),
               'pair_error_product': float((ea[:, channel]*eb[:, channel]).mean())}
        for name, value in products.items():
            difference = value[:, channel]-ref[channel]
            row[name+'_mse'] = float((difference**2).mean())
            row[name+'_bias_mse'] = float((difference.mean(0)**2).mean())
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    device = torch.device('cuda')
    net = new_model(device)
    capture = {}
    handles = [getattr(net, name).block1.register_forward_hook(
        lambda module, inp, value, key=name: capture.__setitem__(key, value.detach().cpu()))
        for name in ['head', 'body']]
    images = {**load_images('train'), **load_images('exploratory_validation')}
    start = time.perf_counter()
    save_json(out/'config.json', {'draws': 64, 'pairs': 32, 'patch': 128, 'margin': 8,
        'photons': 40, 'read_sigma': .008, 'jpeg_quality': 70,
        'weights_sha256': sha256(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'),
        'clean_reference': 'unquantized noiseless low-light input',
        'fixed_pattern_interpretation': 'shared across draws; captured in mean bias, not centered covariance'})
    rows = []
    try:
        with torch.inference_mode():
            for index, (name, image) in enumerate(images.items()):
                patch = image[192:320, 320:448]
                for shared_read in [False, True]:
                    for jpeg in [False, True]:
                        clean, samples = observations(patch, 3000+index, shared_read, jpeg)
                        net(torch.from_numpy(clean.transpose(2, 0, 1).copy())[None].to(device))
                        refs = {key: value[0, :, 8:-8, 8:-8].clone() for key, value in capture.items()}
                        features = {key: [] for key in refs}
                        for offset in range(0, len(samples), 8):
                            value = np.stack(samples[offset:offset+8]).transpose(0, 3, 1, 2).copy()
                            net(torch.from_numpy(value).to(device))
                            for key in refs:
                                features[key].append(capture[key][:, :, 8:-8, 8:-8].clone())
                        for key in refs:
                            values = torch.cat(features[key])
                            for row in moments(refs[key], values):
                                rows.append({'scene': name, 'split': 'train' if index < 6 else 'exploratory_validation',
                                             'shared_read': shared_read, 'jpeg': jpeg, 'layer': key, **row})
                        print(json.dumps({'scene': name, 'shared_read': shared_read, 'jpeg': jpeg}), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    save_json(out/'per_channel.json', rows)
    save_json(out/'runtime.json', {'wall_seconds': time.perf_counter()-start, 'rows': len(rows)})


if __name__ == '__main__':
    main()
