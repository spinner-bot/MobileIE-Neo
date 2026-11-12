"""One fixed mixed-data baseline per seed, retaining the original architecture."""
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


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    gate = BASE/'process/exploration-r40/cpu-data-gate.json'
    assert json.loads(gate.read_text())['passed']
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pool = MixedPool()
    torch.manual_seed(args.seed)
    net = new_model(torch.device('cpu')).cuda().train()
    assert sum(p.numel() for p in net.parameters()) == 4075
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0)
    out = BASE/f'results/exploration-r40-seed{args.seed}'
    checkpoints = ROOT/f'.cache/checkpoints/exploration-r40/seed{args.seed}'
    assert checkpoints.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    out.mkdir(parents=True, exist_ok=False)
    checkpoints.mkdir(parents=True, exist_ok=False)
    files = ['train_mixed_anchor_r40.py', 'mixed_anchor_r40.py', 'dynamic_training_r34.py',
             'native_training_r14.py', 'exploration_r2_data.py', 'train_exploration_r2.py']
    control = BASE/f'results/exploration-r34-seed{args.seed}'
    write(out/'config.json', {'seed': args.seed, 'model': 'mixed_original', 'parameters': 4075,
          'steps': 4000, 'batch': 4, 'frames': 4, 'patch': 128,
          'domain_windows_per_batch': {'synthetic': 2, 'real': 2},
          'scheduling': 'Four B4 forwards, original R34 schedule; synthetic keeps R34 first two windows',
          'objective': 'MSE equally over all pixels, frames and batch elements',
          'optimizer': 'Adam', 'lr': 1e-4, 'betas': [.9, .999], 'eps': 1e-8,
          'weight_decay': 0, 'global_norm_clip': 1, 'dtype': 'float32', 'data': pool.info,
          'pretrained_sha256': digest(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'),
          'pure_control_training_sha256': digest(control/'config.json'),
          'pure_control_final_marker': json.loads((control/'original-step4000.json').read_text()),
          'gate_sha256': digest(gate), 'protocol_sha256': digest(BASE/'docs/exploration-r40-mixed-anchor-protocol.md'),
          'source_sha256': {name: digest(Path(__file__).parent/name) for name in files},
          'torch_version': torch.__version__, 'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32,
          'matmul_tf32_allowed': torch.backends.cuda.matmul.allow_tf32,
          'selection': 'final4000 only', 'resume_supported': False,
          'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
    started, updates, complete = time.perf_counter(), 0, False
    try:
        with (out/'training.jsonl').open('w') as trace, (out/'windows.jsonl').open('w') as contexts:
            for step in range(4000):
                low, target, identities = pool.batch(args.seed, step)
                contexts.write(json.dumps({'step': step+1, 'windows': identities})+'\n')
                low, target = low.cuda(), target.cuda()
                optimizer.zero_grad(set_to_none=True)
                output = prediction(net, low)
                error = (output-target).square()
                loss = error.mean()
                assert torch.isfinite(loss), 'Nonfinite real mixed window; retain and stop'
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                row = {'step': step+1, 'mse': float(loss.detach()),
                       'synthetic_mse': float(error[:2].detach().mean()),
                       'real_mse': float(error[2:].detach().mean()), 'gradient_norm_before_clip': float(norm)}
                trace.write(json.dumps(row, allow_nan=False)+'\n')
                if (step+1) % 1000 == 0:
                    path = checkpoints/f'step{step+1:04d}.pt'
                    torch.save({'state_dict': net.state_dict(), 'optimizer': optimizer.state_dict(),
                                'step': step+1, 'seed': args.seed, 'model': 'mixed_original'}, path)
                    write(out/f'checkpoint-step{step+1:04d}.json', {'path': str(path.relative_to(ROOT)),
                          'sha256': digest(path), 'step': step+1, 'seed': args.seed, 'model': 'mixed_original'})
                if (step+1) % 100 == 0:
                    trace.flush()
                    contexts.flush()
                    print(json.dumps({**row, 'wall_seconds': time.perf_counter()-started}), flush=True)
            complete = True
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'new_updates': updates,
              'final_step': updates, 'wall_seconds': time.perf_counter()-started,
              'scope': 'Fixed mixed-data training only; no quality or novelty conclusion.'})


if __name__ == '__main__':
    main()
