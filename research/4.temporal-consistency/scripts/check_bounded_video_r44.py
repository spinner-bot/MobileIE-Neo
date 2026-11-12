"""Gate the complete factorial's objective, initialization, gradients and export."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
import traceback

import cv2
import torch

from bounded_video_r44 import MODELS, folded, make_models, project_if_bounded, weight_for
from check_video_objective_r44 import objective_gate
from dynamic_training_r34 import BASE, digest
from mixed_anchor_r40 import MixedPool, prediction
from training_form_r41 import configure_precision, cuda_headroom
from video_objective_r44 import prepare_motion, temporal_mse, to_device, valid_fraction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    args = parser.parse_args()
    target = BASE/f'process/exploration-r44/{args.device}-gate.json'
    assert not target.exists(), 'Retain failed gates; no automatic retry'
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    configure_precision()
    started = time.perf_counter()
    record = {'created_at': datetime.now().astimezone().isoformat(), 'device': args.device,
              'passed': False, 'seeds': [], 'discarded_optimizer_updates': 0, 'formal_updates': 0,
              'source_sha256': {name: digest(Path(__file__).parent/name) for name in
                  ['check_bounded_video_r44.py', 'check_video_objective_r44.py', 'video_objective_r44.py',
                   'bounded_video_r44.py', 'bounded_fst_r43.py', 'budget_causal_r33.py', 'mixed_anchor_r40.py']},
              'scope': 'Four initial/backward checks; only bounded_temporal receives two discarded SGD updates per seed.'}
    try:
        assert not torch.cuda.is_initialized()
        pool = MixedPool()
        data = {seed: pool.batch(seed, 4000) for seed in [0, 1]}
        motions = {seed: prepare_motion(data[seed][1]) for seed in [0, 1]}
        initialized = {seed: make_models(seed) for seed in [0, 1]}
        record['data'] = pool.info
        if args.device == 'cuda':
            assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
            record['pre_cuda_admission'] = cuda_headroom()
            assert record['pre_cuda_admission']['allowed']
        device = torch.device(args.device)
        record['objective'] = objective_gate(device)
        for seed in [0, 1]:
            nets, info = initialized[seed]
            nets = {name: net.to(device) for name, net in nets.items()}
            low, gt, context = data[seed]
            low, gt, motion = low.to(device), gt.to(device), to_device(motions[seed], device)
            row = {'seed': seed, 'initialization': info, 'windows': context,
                   'valid_fraction': valid_fraction(motion), 'initial_checks': [], 'backward_checks': [],
                   'discarded_updates': [], 'export_checks': []}
            record['seeds'].append(row)
            for training in [True, False]:
                values = {}
                with torch.no_grad():
                    for name in MODELS:
                        values[name] = prediction(nets[name].train(training), low)
                    for name in MODELS:
                        delta = float((values[name]-values['square_mse']).abs().max())
                        row['initial_checks'].append({'model': name, 'train': training, 'max_abs_delta': delta})
                        assert delta <= 1e-5, row['initial_checks'][-1]
            for name in MODELS:
                net = nets[name].train()
                net.zero_grad(set_to_none=True)
                output = prediction(net, low)
                task, temporal = (output-gt).square().mean(), temporal_mse(output, motion)
                loss = task+weight_for(name)*temporal
                assert torch.isfinite(loss)
                loss.backward()
                active = [p for p in net.parameters() if p.requires_grad]
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active)
                gradients = None
                if name.startswith('bounded'):
                    gradients = {layer: float(getattr(net, layer).kappa.grad) for layer in ['head', 'body']}
                    assert all(value != 0 for value in gradients.values())
                row['backward_checks'].append({'model': name, 'MSE': float(task.detach()),
                    'time_MSE': float(temporal.detach()), 'kappa_gradients': gradients})
            name = 'bounded_temporal'
            net = nets[name]
            active = [p for p in net.parameters() if p.requires_grad]
            optimizer = torch.optim.SGD(active, lr=1e-6)
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                output = prediction(net.train(), low)
                loss = (output-gt).square().mean()+.25*temporal_mse(output, motion)
                assert torch.isfinite(loss)
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True))
                optimizer.step()
                project_if_bounded(net, name)
                record['discarded_optimizer_updates'] += 1
                row['discarded_updates'].append({'model': name, 'step': step+1,
                    'loss': float(loss.detach()), 'gradient_norm_before_clip': norm,
                    'kappa': {layer: float(getattr(net, layer).kappa.detach()) for layer in ['head', 'body']}})
            for name in MODELS:
                net = nets[name].eval()
                slim = folded(net, name).to(device)
                with torch.no_grad():
                    delta = float((prediction(net, low)-prediction(slim, low)).abs().max())
                row['export_checks'].append({'model': name, 'max_abs_delta': delta,
                                            'parameters': sum(p.numel() for p in slim.parameters())})
                assert delta <= 1e-4, row['export_checks'][-1]
            del nets
        record['passed'] = True
    except Exception:
        record['failure'] = traceback.format_exc()
        raise
    finally:
        record.update(wall_seconds=time.perf_counter()-started, cuda_initialized=torch.cuda.is_initialized())
        target.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'passed': record['passed'], 'wall_seconds': record['wall_seconds'],
                          'discarded_optimizer_updates': record['discarded_optimizer_updates']}), flush=True)


if __name__ == '__main__':
    main()
