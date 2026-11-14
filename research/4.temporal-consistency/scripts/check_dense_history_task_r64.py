"""Actual fixed training-window qualification; never changes R63 stress results."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import cv2
import torch

from check_dense_joint_state_r63 import gradient_norm, norms
from dense_joint_state_r63 import folded, make_models, prediction
from dynamic_training_r34 import BASE, digest
from mixed_anchor_r40 import MixedPool
from training_form_r41 import configure_precision, cuda_headroom


def export_check(net, low):
    net.eval()
    deploy = folded(net).to(low.device).eval()
    with torch.no_grad():
        raw, converted = prediction(net, low), prediction(deploy, low)
        assert torch.isfinite(raw).all() and torch.isfinite(converted).all()
        error = float((raw-converted).abs().max())
    return {'raw_max_abs_delta': error, 'passed': error <= 1e-4,
            'parameters': sum(p.numel() for p in deploy.parameters()),
            'raw_range': [float(raw.min()), float(raw.max())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    args = parser.parse_args()
    out = BASE/f'process/exploration-r64/{args.device}-gate001.json'
    assert not out.exists(), 'Keep every failed qualification; no automatic retry'
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    configure_precision()
    assert not torch.cuda.is_initialized()
    started = time.perf_counter()
    record = {'created_at': datetime.now().astimezone().isoformat(), 'passed': False,
              'device': args.device, 'formal_updates': 0, 'discarded_SGD_updates': 0, 'seeds': [],
              'protocol_sha256': digest(BASE/'docs/exploration-r64-dense-history-training-protocol.md'),
              'source_sha256': {name: digest(Path(__file__).parent/name) for name in
                  ['check_dense_history_task_r64.py', 'dense_joint_state_r63.py', 'budget_causal_r33.py',
                   'mixed_anchor_r40.py', 'training_form_r41.py', 'dynamic_training_r34.py']},
              'scope': 'Actual task windows only, not an arbitrary-input fold proof. R63 failures retained.'}
    try:
        pool = MixedPool()
        data = {seed: pool.batch(seed, 8000) for seed in [0, 1]}
        initialized = {seed: make_models(seed) for seed in [0, 1]}
        record['data'] = pool.info
        if args.device == 'cuda':
            assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
            assert json.loads((out.parent/'cpu-gate001.json').read_text())['passed']
            record['pre_cuda_admission'] = cuda_headroom()
            assert record['pre_cuda_admission']['allowed']
        device = torch.device(args.device)
        for seed in [0, 1]:
            nets, info = initialized[seed]
            nets = {name: net.to(device) for name, net in nets.items()}
            low, gt, context = data[seed]
            low, gt = low.to(device), gt.to(device)
            row = {'seed': seed, 'initialization': info, 'windows': context,
                   'initial_checks': [], 'models': {}}
            record['seeds'].append(row)
            for training in [True, False]:
                with torch.no_grad():
                    initial = {name: prediction(net.train(training), low) for name, net in nets.items()}
                for name, output in initial.items():
                    error = float((output-initial['plain']).abs().max())
                    row['initial_checks'].append({'model': name, 'training': training, 'raw_max_delta': error})
                    assert error <= 1e-5
            for name, net in nets.items():
                model_row = {'before_updates_export': export_check(net, low), 'updates': []}
                row['models'][name] = model_row
                assert model_row['before_updates_export']['passed']
                active = [p for p in net.parameters() if p.requires_grad]
                optimizer = torch.optim.SGD(active, lr=1e-6)
                for step in range(2):
                    net.train()
                    optimizer.zero_grad(set_to_none=True)
                    raw = prediction(net, low)
                    loss = (raw.clamp(0, 1)-gt).square().mean()
                    assert torch.isfinite(raw).all() and torch.isfinite(loss)
                    loss.backward()
                    norm = gradient_norm(net)
                    backbone = net if name == 'plain' else net.backbone
                    key_norms = {key: float(backbone.get_parameter(key).grad.abs().sum()) for key in
                                 ['head.weight1', 'body.weight1', 'tail.conv_out.weight']}
                    assert all(value > 0 for value in key_norms.values())
                    branch_norms = None if name == 'plain' else norms(net)
                    if name != 'plain':
                        assert branch_norms['readout'] > 0
                    torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True)
                    optimizer.step()
                    record['discarded_SGD_updates'] += 1
                    model_row['updates'].append({'step': step+1, 'clipped_MSE': float(loss.detach()),
                        'gradient_norm_before_clip': norm, 'backbone_key_gradients': key_norms,
                        'branch_gradient_norms': branch_norms})
                net.train().zero_grad(set_to_none=True)
                last = prediction(net, low)
                (last.clamp(0, 1)-gt).square().mean().backward()
                model_row['after_two_updates_gradient_norms'] = (
                    {'plain': gradient_norm(net)} if name == 'plain' else norms(net))
                model_row['after_updates_export'] = export_check(net, low)
                assert model_row['after_updates_export']['passed']
                expected = 4075 if name == 'plain' else 4618
                assert model_row['after_updates_export']['parameters'] == expected
                if name != 'plain':
                    differentiable = low.detach().clone().requires_grad_()
                    last = prediction(net.eval(), differentiable)
                    gradient, = torch.autograd.grad(last[:, -1].clamp(0, 1).square().mean(), differentiable)
                    past = float(gradient[:, :-1].abs().sum())
                    model_row['past_input_gradient_abs_sum'] = past
                    assert torch.isfinite(gradient).all()
                    assert past > 0 if name == 'causal' else past == 0
            del nets
        assert record['discarded_SGD_updates'] == 12
        record['passed'] = True
    except Exception:
        record['failure'] = traceback.format_exc()
        raise
    finally:
        record.update(wall_seconds=time.perf_counter()-started,
                      cuda_initialized=torch.cuda.is_initialized(),
                      ended_at=datetime.now().astimezone().isoformat())
        out.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'passed': record['passed'], 'device': args.device,
                          'wall_seconds': record['wall_seconds'],
                          'discarded_updates': record['discarded_SGD_updates']}), flush=True)


if __name__ == '__main__':
    main()
