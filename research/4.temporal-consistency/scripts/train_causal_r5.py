"""Matched real12+synthetic task training for finite causal FST controls."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import torch

from causal_fst_r5 import CausalFST, MODES
from exploration_r2_data import ROOT, load_images
from explore_temporal_controls import save_json, sha256
from real_data_r4 import batch, load_sequences, prepare_training
from train_exploration_r2 import new_model
from train_mixture_r4b import EXPANDED, mixture_loss, synthetic_pair
from synthetic_temporal_r5b import synthetic_residual_temporal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--synthetic-temporal-weight', type=float, default=0.)
    parser.add_argument('--checkpoint-root', default='.cache/checkpoints/exploration-r5')
    parser.add_argument('--protocol', default='research/4.temporal-consistency/docs/exploration-r5-protocol.md')
    args = parser.parse_args()
    variants = args.variants.split(',')
    if any(v not in MODES for v in variants) or len(set(variants)) != len(variants):
        parser.error('unique known modes required')
    if args.synthetic_temporal_weight < 0:
        parser.error('nonnegative temporal coefficient required')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in Path(__file__).parent.glob('*.py')}
    data = prepare_training(load_sequences('train', (320, 180), EXPANDED), device)
    synthetic = load_images('train')
    for variant in variants:
        for seed in [0, 1]:
            folder = out/f'{variant}-seed{seed}'
            folder.mkdir()
            check_dir = ROOT/args.checkpoint_root/f'{variant}-seed{seed}'
            check_dir.mkdir(parents=True, exist_ok=False)
            torch.manual_seed(seed)
            net = CausalFST(new_model(device), variant).to(device).train()
            backbone = list(net.backbone.parameters())
            extra = net.branch_parameters()
            groups = [{'params': backbone, 'lr': 1e-5}]
            if extra:
                groups.append({'params': extra, 'lr': 1e-3 if variant == 'tsm_learned' else 1e-5})
            optimizer = torch.optim.Adam(groups, weight_decay=0)
            config = {'variant': variant, 'seed': seed, 'steps': 1000, 'batch_pairs': 2,
                'size': [320, 180], 'lr_backbone': 1e-5, 'lr_extra': groups[-1]['lr'] if extra else None,
                'clip_backbone': 1., 'clip_extra': 1., 'loss': 'task Charbonnier epsilon=.001',
                'history_gradient': 'within pair, raw pre-correction state; inference detached',
                'real_scenes': [r['scene'] for r in data], 'synthetic_pair_per_step': 1,
                'expanded_manifest_sha256': sha256(EXPANDED),
                'pretrained_sha256': sha256(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'),
                'protocol': args.protocol, 'protocol_sha256': sha256(ROOT/args.protocol),
                'synthetic_temporal_weight': args.synthetic_temporal_weight,
                'temporal_protocol_sha256': sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r5b-protocol.md') if args.synthetic_temporal_weight else None,
                'git_revision_at_startup': revision, 'source_sha256_at_startup': sources,
                'parameters': sum(p.numel() for p in net.parameters()),
                'cudnn_deterministic': True, 'cudnn_benchmark': False,
                'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32,
                'matmul_tf32_allowed': torch.backends.cuda.matmul.allow_tf32,
                'torch_version': torch.__version__, 'resume_supported': False}
            save_json(folder/'config.json', config)
            start = time.perf_counter()
            for step in range(1000):
                low, target, _, _, identity = batch(data, seed, step)
                sx, sy, meta = synthetic_pair(synthetic, seed, step)
                low[1], target[1] = sx.to(device), sy.to(device)
                pred = net.forward_pair(low)
                task = mixture_loss(pred, target)
                temporal = synthetic_residual_temporal(pred[1], target[1], meta['offset']) if args.synthetic_temporal_weight else None
                loss = task+args.synthetic_temporal_weight*temporal if temporal is not None else task
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('nonfinite training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(backbone, 1.))
                extra_norm = float(torch.nn.utils.clip_grad_norm_(extra, 1.)) if extra else None
                if not torch.isfinite(torch.tensor(norm)) or (extra_norm is not None and not torch.isfinite(torch.tensor(extra_norm))):
                    raise FloatingPointError('nonfinite gradients')
                optimizer.step()
                net.project()
                samples = [{'source': 'real', 'scene': identity[0][0], 'frame_pair': identity[0][1]}, meta]
                row = {'step': step+1, 'task': float(task.detach()), 'total': float(loss.detach()),
                       'synthetic_temporal': float(temporal.detach()) if temporal is not None else None,
                       'gradient_norm_backbone': norm,
                       'gradient_norm_extra': extra_norm, 'sample_identity': samples}
                if hasattr(net, 'gamma'):
                    row['gamma'] = float(net.gamma.detach())
                if hasattr(net, 'branch'):
                    row['branch_l2'] = float(net.branch.weight.detach().norm())
                with (folder/'training.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                if (step+1)%100 == 0:
                    path = check_dir/f'step{step+1:04d}.pth'
                    torch.save({'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                                'step': step+1, 'config': config}, path)
                    save_json(folder/f'checkpoint-step{step+1:04d}.json',
                              {'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
                    print(json.dumps({'variant': variant, 'seed': seed, **row}), flush=True)
                    if variant == 'plain' and step+1 == 400 and not args.synthetic_temporal_weight:
                        previous = ROOT/'.cache/checkpoints/exploration-r4b'/f'real12_mix-seed{seed}'/'last.pth'
                        old = torch.load(previous, map_location=device, weights_only=False)['model']
                        deltas = {k: float((v-old[k]).abs().max()) for k, v in net.backbone.state_dict().items()}
                        save_json(folder/'r4b-repeat-audit.json', {'previous_sha256': sha256(previous),
                                  'max_abs_by_tensor': deltas, 'exact_match': all(v == 0 for v in deltas.values())})
            save_json(folder/'runtime.json', {'wall_seconds': time.perf_counter()-start,
                'final_checkpoint': str((check_dir/'step1000.pth').relative_to(ROOT)),
                'final_checkpoint_sha256': sha256(check_dir/'step1000.pth'),
                'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()})
            del optimizer, net
    print('all fixed-budget runs completed', flush=True)


if __name__ == '__main__':
    main()
