"""Matched 400-step training of ordinary and causal body interaction candidates."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from exploration_r2_data import MANIFEST, ROOT, evaluate, load_images, make_sequence
from explore_state_r2 import forward_backward, fuse
from explore_temporal_controls import inference, save_json, sha256
from paired_fst_r3 import TrainablePairFST, set_alignment
from train_exploration_r2 import new_model, training_batch


def streaming(net, frames, flows, masks, device):
    net.body.reset()
    result = []
    with torch.inference_mode():
        for t, frame in enumerate(frames):
            if t:
                set_alignment(net.body, flows[t-1], masks[t-1], device)
            value = torch.from_numpy((frame/255).transpose(2, 0, 1).copy())[None].to(device)
            pred = net(value).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()*255
            result.append(pred.astype(np.float32))
    net.body.reset()
    return result


def validate(net, variant, images, device):
    rows = []
    net.eval()
    names = list(images)
    saved_gamma = float(net.body.gamma.detach()) if variant != 'task' else None
    for index, name in enumerate(names):
        for case in ['pan', 'light_step', 'occlusion', 'cut']:
            for seed in [0, 1]:
                gt, low, true_flow, true_mask = make_sequence(images[name], images[names[(index+1)%len(names)]], case, seed)
                pairs = [forward_backward(low[t-1], low[t]) for t in range(1, len(low))]
                flows, full_mask = [p[0] for p in pairs], [p[1] for p in pairs]
                masks = [cv2.erode(m.astype(np.uint8), np.ones((17, 17), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool) for m in full_mask]
                states = ['ordinary'] if variant == 'task' else ['history', 'history_off']
                for state in states:
                    if state == 'ordinary':
                        raw, _ = inference(net, low, device, 1)
                    else:
                        with torch.no_grad():
                            net.body.gamma.fill_(saved_gamma if state == 'history' else 0.)
                        raw = streaming(net, low, flows, masks, device)
                    for fusion, output in [('raw', raw), ('fb', fuse(raw, flows, full_mask))]:
                        metrics, _ = evaluate(output, gt, true_flow, true_mask)
                        rows.append({'scene': name, 'case': case, 'input_seed': seed,
                                     'state': state, 'fusion': fusion, **metrics})
    if saved_gamma is not None:
        with torch.no_grad():
            net.body.gamma.fill_(saved_gamma)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, default=400)
    args = parser.parse_args()
    variants = args.variants.split(',')
    if args.steps < 1 or any(v not in ['task', 'cross', 'mean_square'] for v in variants):
        parser.error('positive step budget and known variants required')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in Path(__file__).parent.glob('*.py')}
    train, validation = load_images('train'), load_images('exploratory_validation')
    for variant in variants:
        for seed in [0, 1]:
            folder = out/f'{variant}-seed{seed}'
            folder.mkdir()
            cache = ROOT/'.cache/checkpoints/exploration-r3'/f'{variant}-seed{seed}'
            cache.mkdir(parents=True, exist_ok=False)
            torch.manual_seed(seed)
            net = new_model(device)
            if variant != 'task':
                net.body = TrainablePairFST(net.body, variant)
            groups = [{'params': [p for n, p in net.named_parameters() if not n.endswith('.gamma')], 'lr': 1e-5}]
            if variant != 'task':
                groups.append({'params': [net.body.gamma], 'lr': 1e-3})
            optimizer = torch.optim.Adam(groups, weight_decay=0)
            config = {'variant': variant, 'seed': seed, 'steps': args.steps, 'pairs': 4, 'patch': 128,
                      'network_lr': 1e-5, 'gamma_lr': 1e-3 if variant != 'task' else None,
                      'temporal_lambda': 0., 'grad_clip': 1., 'git_revision_at_startup': revision,
                      'source_sha256_at_startup': sources, 'manifest_sha256': sha256(MANIFEST),
                      'pretrained_sha256': sha256(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'),
                      'cudnn_deterministic': True, 'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
                      'torch_version': torch.__version__}
            save_json(folder/'config.json', config)
            start = time.perf_counter()
            net.train()
            for step in range(args.steps):
                low, target, offsets = training_batch(train, seed, step)
                low, target = low.to(device), target.to(device)
                if variant != 'task':
                    net.body.offsets = offsets
                prediction = net(low.flatten(0, 1)).reshape_as(target)
                loss = torch.sqrt((prediction-target)**2+1e-6).mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('nonfinite task loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.))
                optimizer.step()
                if variant != 'task':
                    with torch.no_grad():
                        net.body.gamma.clamp_(0, 1)
                row = {'step': step+1, 'task': float(loss.detach()), 'gradient_norm': norm,
                       'gamma': float(net.body.gamma.detach()) if variant != 'task' else None}
                with (folder/'training.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                if (step+1)%50 == 0 or step+1 == args.steps:
                    torch.save({'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                                'step': step+1, 'config': config}, cache/'last.pth')
                    print(json.dumps({'variant': variant, 'seed': seed, **row}), flush=True)
            if variant != 'task':
                net.body.reset()
            save_json(folder/'validation.json', validate(net, variant, validation, device))
            save_json(folder/'runtime.json', {'wall_seconds': time.perf_counter()-start,
                'checkpoint': str((cache/'last.pth').relative_to(ROOT)), 'checkpoint_sha256': sha256(cache/'last.pth'),
                'parameters': sum(p.numel() for p in net.parameters())})


if __name__ == '__main__':
    main()
