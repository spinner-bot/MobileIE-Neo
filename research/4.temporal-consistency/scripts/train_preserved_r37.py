"""Fixed-budget temporal-only training; the strong spatial anchor stays frozen."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import cv2
import torch

from dynamic_training_r34 import BASE, MANIFEST, ROOT, TrainingPool, digest
from training_preserved_r37 import models, prediction


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    gates = [BASE/f'process/exploration-r37/{name}-gate.json' for name in ['cpu', 'cuda']]
    assert all(json.loads(path.read_text())['passed'] for path in gates)
    pool = TrainingPool()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    nets, marker = models(args.seed, torch.device('cuda:0'))
    optimizer = {name: torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-3)
                 for name, net in nets.items()}
    out = BASE/f'results/exploration-r37-seed{args.seed}'
    checkpoints = ROOT/f'.cache/checkpoints/exploration-r37/seed{args.seed}'
    out.mkdir(parents=True, exist_ok=False)
    checkpoints.mkdir(parents=True, exist_ok=False)
    sources = ['train_preserved_r37.py', 'training_preserved_r37.py', 'preserved_state_r37.py',
               'budget_causal_r33.py', 'dynamic_training_r34.py', 'exploration_r2_data.py']
    write(out/'config.json', {'seed': args.seed, 'models': list(nets), 'steps_per_model': 4000,
          'batch': 4, 'frames': 4, 'patch': 128, 'anchor': marker, 'anchor_frozen': True,
          'loss': 'MSE over all four frames', 'optimizer': 'Adam', 'lr': 1e-3,
          'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0, 'clip_gradient_norm': 1,
          'state_dtype': 'float16', 'compute_dtype': 'float32', 'selection': 'final4000 only',
          'source_manifest_sha256': digest(MANIFEST),
          'source_sha256': {name: digest(Path(__file__).parent/name) for name in sources},
          'protocol_sha256': digest(BASE/'docs/exploration-r37-preserved-state-protocol.md'),
          'gates': {str(path.relative_to(ROOT)): digest(path) for path in gates},
          'parameters': {name: {'total': sum(p.numel() for p in net.parameters()),
                                'trainable': sum(p.numel() for p in net.parameters() if p.requires_grad)}
                         for name, net in nets.items()},
          'torch_version': torch.__version__, 'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32,
          'matmul_tf32_allowed': torch.backends.cuda.matmul.allow_tf32,
          'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
    started, updates, complete = time.perf_counter(), {name: 0 for name in nets}, False
    try:
        with (out/'training.jsonl').open('w') as trace, (out/'windows.jsonl').open('w') as contexts:
            for step in range(4000):
                low, target, identities = pool.batch(args.seed, step)
                low, target = low.cuda(), target.cuda()
                contexts.write(json.dumps({'step': step+1, 'windows': identities})+'\n')
                reference = None
                if step == 0:
                    with torch.no_grad():
                        anchor = nets['learned_alignment'].backbone
                        reference = torch.stack([anchor(frame) for frame in low.unbind(1)], dim=1)
                last = []
                for name, net in nets.items():
                    optimizer[name].zero_grad(set_to_none=True)
                    estimate = prediction(net, low)
                    if reference is not None:
                        assert float((estimate-reference).abs().max()) <= 1e-6
                    loss = (estimate-target).square().mean()
                    assert bool(torch.isfinite(loss))
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],
                                                         1., error_if_nonfinite=True)
                    assert all(p.grad is None for p in net.backbone.parameters())
                    optimizer[name].step()
                    updates[name] += 1
                    row = {'step': step+1, 'model': name, 'mse': float(loss.detach()),
                           'gradient_norm_before_clip': float(norm), 'amplitude': float(net.amplitude.detach())}
                    trace.write(json.dumps(row, allow_nan=False)+'\n')
                    last.append(row)
                    if (step+1) % 1000 == 0:
                        path = checkpoints/f'{name}-step{step+1:04d}.pt'
                        torch.save({'state_dict': net.state_dict(), 'optimizer': optimizer[name].state_dict(),
                                    'model': name, 'seed': args.seed, 'step': step+1}, path)
                        write(out/f'{name}-step{step+1:04d}.json', {'path': str(path.relative_to(ROOT)),
                              'sha256': digest(path), 'model': name, 'seed': args.seed, 'step': step+1})
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
                                  'total_updates': sum(updates.values()),
                                  'wall_seconds': time.perf_counter()-started,
                                  'scope': 'Fixed-budget temporal training, no DEV selection, not a quality result.'})


if __name__ == '__main__':
    main()
