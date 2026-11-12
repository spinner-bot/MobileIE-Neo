"""Registered fixed 2x2 continuation: four models, common data, fresh Adam."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import cv2
import torch

from bounded_video_r44 import MODELS, folded, make_models, project_if_bounded, weight_for
from dynamic_training_r34 import BASE, ROOT, digest
from mixed_anchor_r40 import MixedPool, prediction
from training_form_r41 import configure_precision, cuda_headroom
from video_objective_r44 import prepare_motion, temporal_mse, to_device, valid_fraction


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def checkpoint(net, optimizer, model, seed, step, cache, out, low):
    net.eval()
    train_path, path = cache/f'{model}-training-step{step:04d}.pt', cache/f'{model}-deploy-step{step:04d}.pt'
    torch.save({'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(), 'model': model,
                'seed': seed, 'new_step': step, 'inherited_updates': 4000}, train_path)
    slim = folded(net, model).to(low.device).eval()
    delta = None
    if step == 4000:
        with torch.no_grad():
            delta = float((prediction(net, low)-prediction(slim, low)).abs().max())
    train_sha = digest(train_path)
    torch.save({'state_dict': slim.state_dict(), 'model': model, 'seed': seed, 'new_step': step,
                'inherited_updates': 4000, 'training_sha256': train_sha}, path)
    marker = {'model': model, 'seed': seed, 'new_step': step, 'inherited_updates': 4000,
              'path': str(path.relative_to(ROOT)), 'sha256': digest(path),
              'training_path': str(train_path.relative_to(ROOT)), 'training_sha256': train_sha,
              'deployment_parameters': sum(p.numel() for p in slim.parameters()),
              'final_eval_export_max_abs_delta': delta,
              'final_export_gate_passed': None if delta is None else delta <= 1e-4}
    write(out/f'{model}-step{step:04d}.json', marker)
    net.train()
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    parser.add_argument('--stage', choices=['stage001'], required=True)
    args = parser.parse_args()
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    gates = [BASE/f'process/exploration-r44/{name}-gate.json' for name in ['cpu', 'cuda']]
    assert all(json.loads(p.read_text())['passed'] for p in gates)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    configure_precision()
    assert not torch.cuda.is_initialized()
    out = BASE/f'results/exploration-r44-{args.stage}-seed{args.seed}'
    out.mkdir(parents=True, exist_ok=False)
    pool = MixedPool()
    nets, info = make_models(args.seed)
    initial_low, initial_gt, initial_context = pool.batch(args.seed, 4000)
    initial_motion = prepare_motion(initial_gt)
    admission = cuda_headroom()
    write(out/'pre_cuda_guard.json', admission)
    assert admission['allowed'], 'Headroom changed during full CPU source/first-window preparation'
    nets = {name: net.cuda().train() for name, net in nets.items()}
    optimizers = {name: torch.optim.Adam([p for p in net.parameters() if p.requires_grad],
                   lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0) for name, net in nets.items()}
    cache = ROOT/f'.cache/checkpoints/exploration-r44/{args.stage}/seed{args.seed}'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True, exist_ok=False)
    names = ['train_bounded_video_r44.py', 'bounded_video_r44.py', 'bounded_fst_r43.py',
             'video_objective_r44.py', 'budget_causal_r33.py', 'half_motion_r4.py', 'video_metrics.py',
             'mixed_anchor_r40.py', 'training_form_r41.py', 'dynamic_training_r34.py', 'gpu_guard.py']
    write(out/'config.json', {'seed': args.seed, 'stage': args.stage, 'models': MODELS,
          'new_steps_each': 4000, 'inherited_updates_each': 4000, 'data_steps': [4000, 7999],
          'batch': 4, 'frames': 4, 'patch': 128, 'data': pool.info, 'initialization': info,
          'optimizer': 'Adam, fresh for every model', 'lr': 1e-4, 'betas': [.9, .999], 'eps': 1e-8,
          'weight_decay': 0, 'global_clip': 1, 'temporal_weights': {n: weight_for(n) for n in MODELS},
          'BN_calls_per_update': 4, 'selection': 'final new4000 only', 'dtype': 'float32',
          'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32,
          'matmul_tf32_allowed': torch.backends.cuda.matmul.allow_tf32,
          'gates': {str(p.relative_to(ROOT)): digest(p) for p in gates},
          'protocol_sha256': digest(BASE/'docs/exploration-r44-bounded-video-training-protocol.md'),
          'source_sha256': {name: digest(Path(__file__).parent/name) for name in names},
          'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'scope': 'Two factors, shared training-only GT motion, no inherited gate/BN/optimizer states. '
                   'Rational denominator stays at inference; quality and novelty unproven.'})
    started, complete = time.perf_counter(), False
    updates, final_markers = {name: 0 for name in MODELS}, {}
    try:
        with (out/'training.jsonl').open('x') as trace, (out/'windows.jsonl').open('x') as windows:
            for step in range(4000):
                data_step = 4000+step
                if step == 0:
                    low, gt, context, motion = initial_low, initial_gt, initial_context, initial_motion
                else:
                    low, gt, context = pool.batch(args.seed, data_step)
                    motion = prepare_motion(gt)
                validity = valid_fraction(motion)
                windows.write(json.dumps({'new_step': step+1, 'data_step': data_step,
                    'windows': context, 'training_GT_motion_valid_fraction': validity})+'\n')
                low, gt, motion = low.cuda(), gt.cuda(), to_device(motion, 'cuda')
                initial = None
                if step == 0:
                    initial = {name: prediction(net, low) for name, net in nets.items()}
                    deltas = {name: float((v-initial['square_mse']).detach().abs().max()) for name, v in initial.items()}
                    write(out/'initial-real-window.json', {'max_abs_differences': deltas,
                          'passed': all(v <= 1e-5 for v in deltas.values()),
                          'scope': 'Shared first new window before any optimizer update, with no extra BN calls.'})
                    assert all(v <= 1e-5 for v in deltas.values()), deltas
                last = []
                for name, net in nets.items():
                    optimizer = optimizers[name]
                    optimizer.zero_grad(set_to_none=True)
                    output = initial[name] if initial is not None else prediction(net, low)
                    error = (output-gt).square()
                    task, temporal = error.mean(), temporal_mse(output, motion)
                    loss = task+weight_for(name)*temporal
                    assert torch.isfinite(loss) and torch.isfinite(temporal)
                    loss.backward()
                    active = [p for p in net.parameters() if p.requires_grad]
                    gradients = ({layer: float(getattr(net, layer).kappa.grad) for layer in ['head', 'body']}
                                 if name.startswith('bounded') else None)
                    norm = float(torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True))
                    optimizer.step()
                    project_if_bounded(net, name)
                    updates[name] += 1
                    row = {'new_step': step+1, 'data_step': data_step, 'model': name,
                           'MSE': float(task.detach()), 'time_MSE': float(temporal.detach()), 'loss': float(loss.detach()),
                           'synthetic_MSE': float(error[:2].detach().mean()), 'real_MSE': float(error[2:].detach().mean()),
                           'gradient_norm_before_clip': norm, 'kappa_gradients_before_clip': gradients,
                           'kappa': ({layer: float(getattr(net, layer).kappa.detach()) for layer in ['head', 'body']}
                                     if name.startswith('bounded') else None), 'motion_valid_fraction': validity}
                    trace.write(json.dumps(row, allow_nan=False)+'\n')
                    last.append(row)
                    if (step+1) % 1000 == 0:
                        marker = checkpoint(net, optimizer, name, args.seed, step+1, cache, out, low)
                        if step == 3999:
                            final_markers[name] = marker
                            assert marker['final_export_gate_passed'], marker
                if (step+1) % 100 == 0:
                    trace.flush()
                    windows.flush()
                    print(json.dumps({'new_step': step+1, 'models': last,
                                      'wall_seconds': time.perf_counter()-started}), flush=True)
        complete = True
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'model_updates': updates,
              'total_new_updates': sum(updates.values()), 'final_deployment_gates': final_markers,
              'wall_seconds': time.perf_counter()-started,
              'scope': 'Shared R41 inherited4000 plus fixed4000 new per model; no quality or paper claim from loss.'})


if __name__ == '__main__':
    main()
