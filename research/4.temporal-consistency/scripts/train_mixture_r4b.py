"""Matched-shape real-diversity and mixed-domain controls, no architecture change."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from exploration_r2_data import ROOT, degrade, load_images
from explore_temporal_controls import save_json, sha256
from prepare_cache_r4 import CACHE_INDEX, load_cache
from real_data_r4 import MANIFEST, batch, load_sequences, prepare_training
from train_exploration_r2 import new_model
from train_real_r4 import validation

EXPANDED = ROOT/'research/4.temporal-consistency/data/exploration-r4b/manifest.json'


def synthetic_pair(images, seed, step):
    rng = np.random.default_rng(910000000+seed*1000000+step)
    names = list(images)
    name = names[int(rng.integers(len(names)))]
    image = images[name]
    x = int(rng.integers(4, image.shape[1]-320-3))
    y = int(rng.integers(4, image.shape[0]-180-3))
    dx, dy = [int(v) for v in rng.integers(-4, 5, size=2)]
    brightness = float(rng.choice([.6, .8, 1.]))
    photons, jpeg = int(rng.choice([20, 40, 80])), bool(rng.integers(2))
    target = [image[y:y+180, x:x+320]*brightness,
              image[y+dy:y+dy+180, x+dx:x+dx+320]*brightness]
    low = [degrade(frame, rng, photons, jpeg) for frame in target]
    tensor = lambda frames: torch.from_numpy(np.stack(frames).transpose(0, 3, 1, 2).copy()/255.)
    return tensor(low), tensor(target), {'source': 'synthetic', 'scene': name, 'step': step,
        'photons': photons, 'jpeg': jpeg, 'brightness': brightness, 'offset': [dx, dy]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--variants', required=True)
    args = parser.parse_args()
    variants = args.variants.split(',')
    if any(v not in ['real12_only', 'real12_mix', 'real4_mix'] for v in variants):
        parser.error('known variants required')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in Path(__file__).parent.glob('*.py')}
    all_training = prepare_training(load_sequences('train', (320, 180), EXPANDED), device)
    original = json.loads(MANIFEST.read_text())
    names = [r['scene'] for r in original['selection'] if r['split'] == 'train']
    base_training = [next(r for r in all_training if r['scene'] == name) for name in names]
    if len(all_training) != 12 or len(base_training) != 4:
        raise ValueError('fixed 4/12 training scopes required')
    synthetic = load_images('train')
    cache = {domain: load_cache(domain) for domain in ['test', 'legacy_synthetic']}
    for variant in variants:
        training = base_training if variant.startswith('real4_') else all_training
        for seed in [0, 1]:
            folder = out/f'{variant}-seed{seed}'
            folder.mkdir()
            checkpoint = ROOT/'.cache/checkpoints/exploration-r4b'/f'{variant}-seed{seed}'/'last.pth'
            checkpoint.parent.mkdir(parents=True, exist_ok=False)
            torch.manual_seed(seed)
            net = new_model(device).train()
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, weight_decay=0)
            config = {'variant': variant, 'seed': seed, 'steps': 400, 'batch_pairs': 2, 'size': [320, 180],
                'lr': 1e-5, 'clip': 1., 'real_scenes': [r['scene'] for r in training],
                'synthetic_pair_per_step': int(variant.endswith('_mix')),
                'expanded_manifest_sha256': sha256(EXPANDED), 'base_manifest_sha256': sha256(MANIFEST),
                'evaluation_cache_index_sha256': sha256(CACHE_INDEX),
                'pretrained_sha256': sha256(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'),
                'git_revision_at_startup': revision, 'source_sha256_at_startup': sources,
                'cudnn_deterministic': True, 'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32}
            save_json(folder/'config.json', config)
            start = time.perf_counter()
            for step in range(400):
                low, target, _, _, identity = batch(training, seed, step)
                samples = [{'source': 'real', 'scene': n, 'frame_pair': t} for n, t in identity]
                if variant.endswith('_mix'):
                    sx, sy, samples[1] = synthetic_pair(synthetic, seed, step)
                    low[1], target[1] = sx.to(device), sy.to(device)
                pred = net(low.flatten(0, 1)).reshape_as(target)
                loss = torch.sqrt((pred-target)**2+1e-6).mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('nonfinite mixture loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.))
                optimizer.step()
                row = {'step': step+1, 'task': float(loss.detach()), 'gradient_norm': norm, 'sample_identity': samples}
                with (folder/'training.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                if (step+1)%50 == 0:
                    torch.save({'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                                'step': step+1, 'config': config}, checkpoint)
                    print(json.dumps({'variant': variant, 'seed': seed, **row}), flush=True)
            save_json(folder/'validation.json', validation(net, cache, device))
            save_json(folder/'runtime.json', {'wall_seconds': time.perf_counter()-start,
                'checkpoint': str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256': sha256(checkpoint),
                'parameters': sum(p.numel() for p in net.parameters())})


if __name__ == '__main__':
    main()
