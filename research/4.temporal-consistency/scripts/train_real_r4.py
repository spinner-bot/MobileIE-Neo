"""Fixed-budget real-domain controls without extra architecture parameters."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import torch

from exploration_r2_data import ROOT, evaluate
from explore_state_r2 import fuse
from explore_temporal_controls import inference, save_json, sha256
from prepare_cache_r4 import CACHE_INDEX, load_cache
from real_data_r4 import MANIFEST, batch, load_sequences, prepare_training, residual_temporal
from train_exploration_r2 import new_model


def validation(net, cache):
    rows = []
    net.eval()
    for domain, scenes in cache.items():
        for scene, data in scenes.items():
            raw, _ = inference(net, data['low'], 'cuda', 1)
            for fusion, output in [('raw', raw), ('enhanced_fb', fuse(raw, data['enhanced_flow'], data['enhanced_mask']))]:
                metrics, _ = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                rows.append({'domain': domain, 'scene': scene, 'fusion': fusion, **metrics})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--variants', default='real_task,real_anchor,real_temporal')
    args = parser.parse_args()
    variants = args.variants.split(',')
    if any(v not in ['real_task', 'real_anchor', 'real_temporal'] for v in variants):
        parser.error('unknown variant')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    source = {str(p.relative_to(ROOT)): sha256(p) for p in Path(__file__).parent.glob('*.py')}
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    training = prepare_training(load_sequences('train', (320, 180)), device)
    cache = {domain: load_cache(domain) for domain in ['test', 'legacy_synthetic']}
    teacher = new_model(device)
    for variant in variants:
        for seed in [0, 1]:
            folder = out/f'{variant}-seed{seed}'
            folder.mkdir()
            checkpoint = ROOT/'.cache/checkpoints/exploration-r4'/f'{variant}-seed{seed}'/'last.pth'
            checkpoint.parent.mkdir(parents=True, exist_ok=False)
            torch.manual_seed(seed)
            net = new_model(device).train()
            optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, weight_decay=0)
            config = {'variant': variant, 'seed': seed, 'steps': 400, 'batch_pairs': 2,
                'size': [320, 180], 'optimizer': 'Adam', 'lr': 1e-5, 'clip': 1.,
                'anchor_lambda': .1 if variant == 'real_anchor' else 0.,
                'temporal_lambda': .1 if variant == 'real_temporal' else 0.,
                'previous_temporal_branch_detached': True, 'training_manifest_sha256': sha256(MANIFEST),
                'evaluation_cache_index_sha256': sha256(CACHE_INDEX),
                'pretrained_sha256': sha256(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'),
                'source_sha256_at_startup': source, 'git_revision_at_startup': revision,
                'cudnn_deterministic': True, 'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32}
            save_json(folder/'config.json', config)
            start = time.perf_counter()
            for step in range(400):
                low, target, grid, mask, identity = batch(training, seed, step)
                prediction = net(low.flatten(0, 1)).reshape_as(target)
                task = torch.sqrt((prediction-target)**2+1e-6).mean()
                temporal = residual_temporal(prediction, target, grid, mask)
                anchor = prediction.sum()*0+.001
                if variant == 'real_anchor':
                    with torch.no_grad():
                        reference = teacher(low.flatten(0, 1)).reshape_as(target)
                    anchor = torch.sqrt((prediction-reference)**2+1e-6).mean()
                loss = task+config['anchor_lambda']*anchor+config['temporal_lambda']*temporal
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('nonfinite training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.))
                optimizer.step()
                row = {'step': step+1, 'task': float(task.detach()), 'temporal': float(temporal.detach()),
                    'anchor': float(anchor.detach()), 'loss': float(loss.detach()), 'gradient_norm': norm,
                    'sample_identity': identity}
                with (folder/'training.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                if (step+1)%50 == 0:
                    torch.save({'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                                'step': step+1, 'config': config}, checkpoint)
                    print(json.dumps({'variant': variant, 'seed': seed, **row}), flush=True)
            save_json(folder/'validation.json', validation(net, cache))
            save_json(folder/'runtime.json', {'wall_seconds': time.perf_counter()-start,
                'checkpoint': str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256': sha256(checkpoint),
                'parameters': sum(p.numel() for p in net.parameters())})


if __name__ == '__main__':
    main()
