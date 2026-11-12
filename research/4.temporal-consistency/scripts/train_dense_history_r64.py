"""Fixed matched dense-backbone continuation; no quality selection or grid."""
import argparse
from datetime import datetime
import json
import os
import subprocess
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import cv2
import torch

from dense_joint_state_r63 import folded, make_models, prediction
from dynamic_training_r34 import BASE, ROOT, digest
from mixed_anchor_r40 import MixedPool
from training_form_r41 import configure_precision, cuda_headroom

MODELS = ['plain', 'spatial', 'causal']


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def checkpoint(net, optimizer, name, seed, step, low, cache, out):
    net.eval()
    train_path = cache/f'{name}-training-step{step:04d}.pt'
    deploy_path = cache/f'{name}-deploy-step{step:04d}.pt'
    torch.save({'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
                'model': name, 'seed': seed, 'new_step': step, 'inherited_updates': 8000}, train_path)
    slim = folded(net).to(low.device).eval()
    expected = 4075 if name == 'plain' else 4618
    assert sum(p.numel() for p in slim.parameters()) == expected
    error = None
    if step == 4000:
        with torch.no_grad():
            original, deployed = prediction(net, low), prediction(slim, low)
            assert torch.isfinite(original).all() and torch.isfinite(deployed).all()
            error = float((original-deployed).abs().max())
    torch.save({'state_dict': slim.state_dict(), 'model': name, 'seed': seed,
                'new_step': step, 'inherited_updates': 8000}, deploy_path)
    marker = {'model': name, 'seed': seed, 'new_step': step, 'inherited_updates': 8000,
              'path': str(deploy_path.relative_to(ROOT)), 'sha256': digest(deploy_path),
              'training_path': str(train_path.relative_to(ROOT)), 'training_sha256': digest(train_path),
              'deployment_parameters': expected, 'final_fold_delta': error,
              'final_fold_passed': None if error is None else error <= 1e-4}
    write(out/f'{name}-step{step:04d}.json', marker)
    net.train()
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    gates = [BASE/f'process/exploration-r64/{device}-gate001.json' for device in ['cpu', 'cuda']]
    assert all(json.loads(p.read_text())['passed'] for p in gates)
    guard = BASE/'process/exploration-r64/cuda-gate001-guard.json'
    assert json.loads(guard.read_text())['returncode'] == 0
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    configure_precision()
    assert not torch.cuda.is_initialized()
    out = BASE/f'results/exploration-r64-stage001-seed{args.seed}'
    out.mkdir(exist_ok=False)
    started, complete, updates, final = time.perf_counter(), False, {n: 0 for n in MODELS}, {}
    try:
        nets, initialization = make_models(args.seed)
        pool = MixedPool()
        first_low, first_gt, first_context = pool.batch(args.seed, 8000)
        admission = cuda_headroom()
        write(out/'pre_cuda_guard.json', admission)
        assert admission['allowed']
        nets = {name: net.cuda().train() for name, net in nets.items()}
        optimizers = {name: torch.optim.Adam([p for p in net.parameters() if p.requires_grad],
            lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0) for name, net in nets.items()}
        cache = ROOT/f'.cache/checkpoints/exploration-r64/stage001/seed{args.seed}'
        assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
        cache.mkdir(parents=True, exist_ok=False)
        names = ['train_dense_history_r64.py', 'dense_joint_state_r63.py', 'budget_causal_r33.py',
                 'mixed_anchor_r40.py', 'dynamic_training_r34.py', 'training_form_r41.py', 'gpu_guard.py']
        write(out/'config.json', {'created_at': datetime.now().astimezone().isoformat(),
            'seed': args.seed, 'models': MODELS, 'new_steps_each': 4000, 'inherited_updates_each': 8000,
            'data_steps': [8000, 11999], 'initialization': initialization, 'data': pool.info,
            'batch': 4, 'frames': 4, 'crop': 128, 'loss': 'Identical clipped raw MSE, all four frames',
            'fresh_Adam_lr': 1e-4, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0, 'global_clip': 1,
            'BN_calls_per_update': 4, 'history_detached_in_window': False, 'FP32': True, 'TF32': False,
            'selection': 'Final new4000 only; old R63 stress gate still failed',
            'protocol_sha256': digest(BASE/'docs/exploration-r64-dense-history-training-protocol.md'),
            'gates': {str(p.relative_to(ROOT)): digest(p) for p in gates},
            'source_sha256': {n: digest(BASE/'scripts'/n) for n in names},
            'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
        with (out/'training.jsonl').open('x') as trace, (out/'windows.jsonl').open('x') as windows:
            for step in range(4000):
                low, gt, context = ((first_low, first_gt, first_context) if step == 0
                                   else pool.batch(args.seed, 8000+step))
                windows.write(json.dumps({'new_step': step+1, 'data_step': 8000+step, 'windows': context})+'\n')
                low, gt = low.cuda(), gt.cuda()
                initial = {name: prediction(net, low) for name, net in nets.items()} if step == 0 else None
                if initial is not None:
                    differences = {name: float((raw-initial['plain']).detach().abs().max())
                                   for name, raw in initial.items()}
                    write(out/'initial-real-window.json', {'raw_max_abs_delta': differences,
                          'passed': all(v <= 1e-5 for v in differences.values()), 'BN_calls_each': 4,
                          'context': context})
                    assert all(v <= 1e-5 for v in differences.values())
                last = []
                for name, net in nets.items():
                    optimizer = optimizers[name]
                    optimizer.zero_grad(set_to_none=True)
                    raw = initial[name] if initial is not None else prediction(net, low)
                    assert torch.isfinite(raw).all()
                    error = (raw.clamp(0, 1)-gt).square()
                    loss = error.mean()
                    assert torch.isfinite(loss)
                    loss.backward()
                    active = [p for p in net.parameters() if p.requires_grad]
                    norm = float(torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True))
                    optimizer.step()
                    updates[name] += 1
                    row = {'new_step': step+1, 'data_step': 8000+step, 'model': name,
                           'clipped_MSE': float(loss.detach()), 'synthetic_MSE': float(error[:2].detach().mean()),
                           'real_MSE': float(error[2:].detach().mean()), 'gradient_norm_before_clip': norm,
                           'unclipped_MSE': float((raw.detach()-gt).square().mean()),
                           'raw_outside_0_1_fraction': float(((raw.detach() < 0) | (raw.detach() > 1)).float().mean())}
                    trace.write(json.dumps(row, allow_nan=False)+'\n')
                    last.append(row)
                    if (step+1) % 1000 == 0:
                        marker = checkpoint(net, optimizer, name, args.seed, step+1, low, cache, out)
                        if step == 3999:
                            final[name] = marker
                            assert marker['final_fold_passed'], marker
                if (step+1) % 100 == 0:
                    trace.flush()
                    windows.flush()
                    print(json.dumps({'new_step': step+1, 'models': last,
                                      'wall_seconds': time.perf_counter()-started}), flush=True)
        assert updates == {name: 4000 for name in MODELS}
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'model_updates': updates,
              'total_new_updates': sum(updates.values()), 'final_markers': final,
              'seconds': time.perf_counter()-started, 'ended_at': datetime.now().astimezone().isoformat(),
              'scope': 'Fixed algorithm budget, no wall cutoff, no quality selection from partial loss.'})


if __name__ == '__main__':
    main()
