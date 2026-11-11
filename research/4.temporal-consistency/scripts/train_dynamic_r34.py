"""Fixed shared-window training of original, spatial and causal controls."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import cv2
import torch

from dynamic_training_r34 import BASE, MANIFEST, ROOT, TrainingPool, digest, models


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prediction(net, name, low):
    if name == 'original':
        return net(low.flatten(0, 1)).reshape_as(low)
    if name == 'spatial':
        return net(low.flatten(0, 1))[0].reshape_as(low)
    state, outputs = None, []
    for frame in low.unbind(1):
        output, state = net(frame, state)
        outputs.append(output)
    return torch.stack(outputs, dim=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    gate = BASE/'process/exploration-r33/cuda-backward.json'
    assert json.loads(gate.read_text())['passed'], 'CUDA backward must pass before training'
    pool = TrainingPool()  # Requires full manifest and source-decode audit.
    out = BASE/f'results/exploration-r34-seed{args.seed}'
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    nets, initialization = models(args.seed, device)
    optimizers = {name: torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0) for name, net in nets.items()}
    checkpoints = ROOT/'.cache/checkpoints/exploration-r34'/f'seed{args.seed}'
    checkpoints.mkdir(parents=True, exist_ok=False)
    files = ['budget_causal_r33.py', 'dynamic_training_r34.py', 'train_dynamic_r34.py', 'exploration_r2_data.py']
    config = {'seed': args.seed, 'models': list(nets), 'steps_per_model': 4000, 'batch': 4, 'frames': 4, 'patch': 128,
              'objective': 'MSE over all four supervised frames', 'optimizer': 'Adam', 'lr': 1e-4,
              'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0, 'global_norm_clip': 1,
              'dtype': 'float32', 'device': str(device), 'deterministic_algorithms': True,
              'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32,
              'matmul_tf32_allowed': torch.backends.cuda.matmul.allow_tf32,
              'torch_version': torch.__version__, 'source_manifest_sha256': digest(MANIFEST),
              'protocol_sha256': digest(BASE/'docs/exploration-r34-training-protocol.md'),
              'cuda_gate_sha256': digest(gate), 'initialization': initialization,
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'source_sha256': {name: digest(Path(__file__).parent/name) for name in files},
              'parameters': {name: sum(p.numel() for p in net.parameters()) for name, net in nets.items()},
              'checkpoint_selection': 'final4000 only', 'resume_supported': False}
    write(out/'config.json', config)
    updates, started, completed = {name: 0 for name in nets}, time.perf_counter(), False
    try:
        with (out/'windows.jsonl').open('w') as windows, (out/'training.jsonl').open('w') as trace:
            for step in range(4000):
                low, target, identities = pool.batch(args.seed, step)
                low, target = low.to(device), target.to(device)
                windows.write(json.dumps({'step': step+1, 'windows': identities})+'\n')
                last = []
                for name, net in nets.items():
                    optimizer = optimizers[name]
                    optimizer.zero_grad(set_to_none=True)
                    estimate = prediction(net, name, low)
                    loss = (estimate-target).square().mean()
                    assert bool(torch.isfinite(loss)), f'{name}: non-finite loss'
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    updates[name] += 1
                    row = {'step': step+1, 'model': name, 'mse': loss.item(), 'gradient_norm_before_clip': norm.item()}
                    trace.write(json.dumps(row, allow_nan=False)+'\n')
                    last.append(row)
                    if (step+1) % 1000 == 0:
                        path = checkpoints/f'{name}-step{step+1:04d}.pt'
                        assert not path.exists()
                        torch.save({'state_dict': net.state_dict(), 'step': step+1, 'model': name,
                                    'seed': args.seed, 'optimizer': optimizer.state_dict()}, path)
                        write(out/f'{name}-step{step+1:04d}.json', {'path': str(path.relative_to(ROOT)),
                              'sha256': digest(path), 'step': step+1, 'model': name, 'seed': args.seed})
                if (step+1) % 100 == 0:
                    trace.flush()
                    windows.flush()
                    print(json.dumps({'step': step+1, 'losses': last, 'wall_seconds': time.perf_counter()-started}), flush=True)
        completed = True
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', {'completed_normally': completed, 'model_updates': updates,
                                  'total_updates': sum(updates.values()), 'wall_seconds': time.perf_counter()-started,
                                  'scope': 'Matched-budget development training, not final benchmark or established method.'})


if __name__ == '__main__':
    main()
