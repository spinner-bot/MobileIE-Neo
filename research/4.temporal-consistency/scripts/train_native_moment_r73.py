"""Fixed R73 root-moment continuation; final4000 only, no grid or selection."""
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

from native_moment_core_r73 import folded, make_models, prediction, MODES, PARAMETERS, PROTOCOL
from dynamic_training_r34 import BASE, ROOT, digest
from mixed_anchor_r40 import MixedPool
from training_form_r41 import configure_precision, cuda_headroom

MODELS = MODES


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def checkpoint(net, optimizer, name, seed, step, low, cache, out):
    net.eval()
    train_path = cache/f'{name}-training-step{step:04d}.pt'
    deploy_path = cache/f'{name}-deploy-step{step:04d}.pt'
    torch.save({'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
                'model': name, 'seed': seed, 'new_step': step, 'inherited_updates': 8000}, train_path)
    slim = folded(net).to(low.device).eval()
    expected = PARAMETERS[name]
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
    parser.add_argument('--admission-attempt', type=int, default=1)
    args = parser.parse_args()
    qualified = json.loads((BASE/'process/exploration-r73/cuda-qualification.json').read_text())
    gates = [BASE/'process/exploration-r73/cpu-gate001/runtime.json', ROOT/qualified['gate_runtime']]
    assert all(json.loads(p.read_text())['passed'] for p in gates)
    guard = ROOT/qualified['guard']
    assert json.loads(guard.read_text())['returncode'] == 0
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    configure_precision()
    assert not torch.cuda.is_initialized()
    nets, initialization = make_models(args.seed)
    pool = MixedPool()
    first_low, first_gt, first_context = pool.batch(args.seed, 8000)
    admission = cuda_headroom()
    write(BASE/f'process/exploration-r73/seed{args.seed}-internal-admission{args.admission_attempt:03d}.json', admission)
    if not admission['allowed']:
        raise SystemExit(75)
    out = BASE/f'results/exploration-r73-stage001-seed{args.seed}'
    out.mkdir(exist_ok=False)
    started, complete, updates, final = time.perf_counter(), False, {n: 0 for n in MODELS}, {}
    try:
        write(out/'pre_cuda_guard.json', admission)
        nets = {name: net.cuda().train() for name, net in nets.items()}
        optimizers = {name: torch.optim.Adam([p for p in net.parameters() if p.requires_grad],
            lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0) for name, net in nets.items()}
        cache = ROOT/f'.cache/checkpoints/exploration-r73/stage001/seed{args.seed}'
        assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
        cache.mkdir(parents=True, exist_ok=False)
        names = ['train_native_moment_r73.py', 'native_moment_core_r73.py', 'check_native_moment_r73.py', 'budget_causal_r33.py',
                 'mixed_anchor_r40.py', 'dynamic_training_r34.py', 'training_form_r41.py', 'gpu_guard.py']
        write(out/'config.json', {'created_at': datetime.now().astimezone().isoformat(),
            'seed': args.seed, 'models': MODELS, 'new_steps_each': 4000, 'inherited_updates_each': 8000,
            'data_steps': [8000, 11999], 'initialization': initialization, 'data': pool.info,
            'batch': 4, 'frames': 4, 'crop': 128, 'loss': 'Identical clipped raw MSE, all four frames',
            'fresh_Adam_lr': 1e-4, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0, 'global_clip': 1,
            'BN_calls_per_update': 4, 'history_detached_in_window': False, 'FP32': True, 'TF32': False,
            'selection': 'Final new4000 only; discarded qualification is not inherited',
            'protocol_sha256': digest(PROTOCOL),
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
                    differences = dict(mean_minus_moment_max=float((initial['mean']-initial['moment']).detach().abs().max()),
                          first_frame_moment_minus_spatial_max=float((initial['moment'][:, 0]-initial['spatial'][:, 0]).detach().abs().max()),
                          actual_moment_minus_spatial_max=float((initial['moment']-initial['spatial']).detach().abs().max()))
                    passed = differences['mean_minus_moment_max'] <= 1e-5 and differences['first_frame_moment_minus_spatial_max'] <= 1e-5
                    write(out/'initial-real-window.json', dict(raw_max_abs_delta=differences, passed=passed,
                          BN_calls_each=4, context=context, actual_history_initialization_is_not_function_preserving=True))
                    assert passed
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
                           'raw_outside_0_1_fraction': float(((raw.detach() < 0) | (raw.detach() > 1)).float().mean()),
                           'rho': None if net.rho is None else net.rho.detach().flatten().tolist()}
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
