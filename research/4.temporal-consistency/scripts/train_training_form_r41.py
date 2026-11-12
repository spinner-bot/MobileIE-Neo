"""Matched strict-FP32 slim/MBR training, with fixed windows and deployment gates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import cv2
import torch

from dynamic_training_r34 import BASE, ROOT, digest
from mixed_anchor_r40 import MixedPool, prediction
from train_exploration_r2 import new_model
from training_form_r41 import configure_precision, create_model, cuda_headroom


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def checkpoint(net, optimizer, model, seed, step, cache, out, last_input):
    net.eval()
    training_path = cache/f'{model}-training-step{step:04d}.pt'
    deployment_path = cache/f'{model}-deploy-step{step:04d}.pt'
    torch.save({'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(), 'model': model,
                'seed': seed, 'step': step}, training_path)
    folded = net.slim().cuda().eval() if model == 'expanded_fp32' else net
    assert sum(p.numel() for p in folded.parameters()) == 4075
    delta = None
    if step == 4000:
        with torch.no_grad():
            delta = (float((prediction(net, last_input)-prediction(folded, last_input)).abs().max())
                     if model == 'expanded_fp32' else 0.)
    training_sha = digest(training_path)
    torch.save({'state_dict': folded.state_dict(), 'model': model, 'seed': seed,
                'step': step, 'training_sha256': training_sha}, deployment_path)
    marker = {'path': str(deployment_path.relative_to(ROOT)), 'sha256': digest(deployment_path),
              'training_path': str(training_path.relative_to(ROOT)), 'training_sha256': training_sha,
              'model': model, 'seed': seed, 'step': step, 'deployment_parameters': 4075,
              'final_eval_fold_max_abs_delta': delta,
              'final_fold_gate_passed': None if delta is None else delta <= 1e-4}
    write(out/f'{model}-step{step:04d}.json', marker)
    net.train()
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    parser.add_argument('--stage', choices=['stage002'], required=True)
    args = parser.parse_args()
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    gates = [BASE/'process/exploration-r41'/args.stage/name for name in ['cpu-gate.json', 'cuda-gate.json']]
    assert all(json.loads(path.read_text())['passed'] for path in gates)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    configure_precision()
    out = BASE/f'results/exploration-r41-{args.stage}-seed{args.seed}'
    out.mkdir(parents=True, exist_ok=False)
    pool = MixedPool()
    expanded, info = create_model(args.seed)
    slim = new_model(torch.device('cpu'))
    admission = cuda_headroom()
    write(out/'pre_cuda_guard.json', admission)
    assert admission['allowed'], 'Headroom changed during complete CPU data preparation'
    nets = {'slim_fp32': slim.cuda().train(), 'expanded_fp32': expanded.cuda().train()}
    optimizers = {name: torch.optim.Adam([p for p in net.parameters() if p.requires_grad],
                   lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0) for name, net in nets.items()}
    cache = ROOT/f'.cache/checkpoints/exploration-r41/{args.stage}/seed{args.seed}'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True, exist_ok=False)
    sources = ['train_training_form_r41.py', 'training_form_r41.py', 'mixed_anchor_r40.py',
               'native_training_r14.py', 'dynamic_training_r34.py', 'exploration_r2_data.py', 'gpu_guard.py']
    write(out/'config.json', {'seed': args.seed, 'stage': args.stage, 'models': list(nets), 'steps_per_model': 4000,
          'batch': 4, 'frames': 4, 'patch': 128, 'data': pool.info, 'initialization': info,
          'parameters': {name: {'total': sum(p.numel() for p in net.parameters()),
                                'active': sum(p.numel() for p in net.parameters() if p.requires_grad)}
                         for name, net in nets.items()},
          'deployment_parameters': 4075, 'objective': 'MSE all four frames', 'optimizer': 'Adam',
          'lr': 1e-4, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0, 'clip': 1,
          'dtype': 'float32', 'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32,
          'matmul_tf32_allowed': torch.backends.cuda.matmul.allow_tf32,
          'BN_train_calls_per_update': 4, 'warmup_used': False, 'selection': 'final4000 folded only',
          'gates': {str(p.relative_to(ROOT)): digest(p) for p in gates},
          'protocol_sha256': digest(BASE/'docs/exploration-r41-training-form-protocol.md'),
          'source_sha256': {name: digest(Path(__file__).parent/name) for name in sources},
          'upstream_source_sha256': {name: digest(ROOT/'MobileIE'/name) for name in ['model/lle.py', 'model/utils.py']},
          'R40_old_precision_config_sha256': digest(BASE/f'results/exploration-r40-seed{args.seed}/config.json'),
          'torch_version': torch.__version__,
          'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
    started, complete, updates, final_markers = time.perf_counter(), False, {name: 0 for name in nets}, {}
    try:
        with (out/'training.jsonl').open('w') as trace, (out/'windows.jsonl').open('w') as contexts:
            for step in range(4000):
                low, target, identities = pool.batch(args.seed, step)
                contexts.write(json.dumps({'step': step+1, 'windows': identities})+'\n')
                low, target = low.cuda(), target.cuda()
                initial = None
                if step == 0:
                    initial = {name: prediction(net, low) for name, net in nets.items()}
                    delta = float((initial['slim_fp32']-initial['expanded_fp32']).detach().abs().max())
                    write(out/'initial-real-window.json', {'max_abs_difference': delta, 'passed': delta <= 1e-5,
                          'scope': 'Formal first shared window before any optimizer update; no extra BN forward calls.'})
                    assert delta <= 1e-5, delta
                last = []
                for name, net in nets.items():
                    optimizer = optimizers[name]
                    optimizer.zero_grad(set_to_none=True)
                    output = initial[name] if initial is not None else prediction(net, low)
                    error = (output-target).square()
                    loss = error.mean()
                    assert torch.isfinite(loss)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],
                                                         1., error_if_nonfinite=True)
                    optimizer.step()
                    updates[name] += 1
                    row = {'step': step+1, 'model': name, 'mse': float(loss.detach()),
                           'synthetic_mse': float(error[:2].detach().mean()),
                           'real_mse': float(error[2:].detach().mean()), 'gradient_norm_before_clip': float(norm)}
                    trace.write(json.dumps(row, allow_nan=False)+'\n')
                    last.append(row)
                    if (step+1) % 1000 == 0:
                        marker = checkpoint(net, optimizer, name, args.seed, step+1, cache, out, low)
                        if step == 3999:
                            final_markers[name] = marker
                            assert marker['final_fold_gate_passed'], marker
                if (step+1) % 100 == 0:
                    trace.flush()
                    contexts.flush()
                    print(json.dumps({'step': step+1, 'losses': last,
                                      'wall_seconds': time.perf_counter()-started}), flush=True)
        complete = True
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'model_updates': updates,
              'total_updates': sum(updates.values()), 'final_deployment_gates': final_markers,
              'wall_seconds': time.perf_counter()-started,
              'scope': 'Two matched FP32 training forms; quality is unproven until complete folded evaluation.'})


if __name__ == '__main__':
    main()
