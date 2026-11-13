"""Fixed 4000-update scalar/RGB/internal-response controls on a frozen spatial net."""
import argparse
from datetime import datetime
import json
import os
import subprocess
import time
import traceback

import cv2
import torch

from dynamic_training_r34 import BASE, ROOT, digest
from mixed_anchor_r40 import MixedPool
from response_cue_r45 import FrozenResponses, RecursiveControl, load_frozen
from response_sequence_r45 import encode, motion_for, move_fields, rollout
from training_form_r41 import configure_precision, cuda_headroom

LEARNED = ['scalar', 'rgb', 'response']


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def checkpoint(control, optimizer, name, seed, step, cache, out, marker):
    path = cache/f'{name}-step{step:04d}.pt'
    torch.save({'state_dict': control.state_dict(), 'optimizer': optimizer.state_dict(),
                'control': name, 'seed': seed, 'new_step': step,
                'base_checkpoint_sha256': marker['sha256']}, path)
    record = {'control': name, 'seed': seed, 'new_step': step,
              'path': str(path.relative_to(ROOT)), 'sha256': digest(path),
              'base_checkpoint_sha256': marker['sha256'],
              'controller_parameters': sum(p.numel() for p in control.parameters()),
              'pipeline_parameters': 4075+sum(p.numel() for p in control.parameters()),
              'scope': 'No BN/folding in controller. Same state used for eval; spatial predictor frozen.'}
    write(out/f'{name}-step{step:04d}.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    gates = [BASE/'process/exploration-r45/cpu-attempt002/gate.json',
             BASE/'process/exploration-r45/cuda-attempt001/gate.json']
    assert all(json.loads(p.read_text())['passed'] for p in gates)
    assert json.loads((BASE/'process/exploration-r45/cuda-attempt001-guard.json').read_text())['returncode'] == 0
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True;torch.backends.cudnn.benchmark = False
    configure_precision()
    assert not torch.cuda.is_initialized()
    out = BASE/f'results/exploration-r45-stage001-seed{args.seed}'
    out.mkdir(parents=True, exist_ok=False)
    started, complete = time.perf_counter(), False
    updates, final = {n: 0 for n in LEARNED}, {}
    try:
        pool = MixedPool()
        initial_low, initial_gt, initial_context = pool.batch(args.seed, 8000)
        backbone, marker = load_frozen(args.seed)
        before = {n: t.clone() for n, t in backbone.state_dict().items()}
        admission = cuda_headroom()
        write(out/'pre_cuda_guard.json', admission)
        assert admission['allowed'], 'GPU became busy during CPU data preparation'
        encoder = FrozenResponses(backbone.cuda().eval())
        controls = {n: RecursiveControl(n, args.seed).cuda().train() for n in LEARNED}
        optimizers = {n: torch.optim.Adam(c.parameters(), lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=0)
                      for n, c in controls.items()}
        cache = ROOT/f'.cache/checkpoints/exploration-r45/stage001/seed{args.seed}'
        assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
        cache.mkdir(parents=True, exist_ok=False)
        names = ['train_response_recursion_r45.py', 'response_cue_r45.py', 'response_sequence_r45.py',
                 'budget_causal_r33.py', 'mixed_anchor_r40.py', 'half_motion_r4.py', 'video_metrics.py', 'gpu_guard.py']
        write(out/'config.json', {'started_at': datetime.now().astimezone().isoformat(),
              'seed': args.seed, 'controls': LEARNED, 'new_updates_each': 4000,
              'base_checkpoint': marker, 'frozen_base_inherited_updates': 8000,
              'data_steps': [8000, 11999], 'batch': 4, 'frames': 4, 'patch': 128, 'data': pool.info,
              'optimizer': 'fresh Adam per controller', 'lr': 1e-3, 'betas': [.9, .999], 'eps': 1e-8,
              'weight_decay': 0, 'global_clip': 1, 'objective': 'Full RGB MSE, all four frames equally weighted',
              'motion': 'Own clipped raw half-FB, shared across controllers; no GT motion',
              'parameter_counts': {n: {'controller': sum(p.numel() for p in c.parameters()),
                                      'pipeline': 4075+sum(p.numel() for p in c.parameters())} for n, c in controls.items()},
              'precision': 'FP32; TF32 off; deterministic', 'selection': 'final4000 only',
              'gates_sha256': {str(p.relative_to(ROOT)): digest(p) for p in gates},
              'protocol_sha256': digest(BASE/'docs/exploration-r45-controlled-recursion-training-protocol.md'),
              'source_sha256': {n: digest(BASE/'scripts'/n) for n in names},
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'scope': 'Known learned recursion controls. Internal cue equals feature-energy rescaling, not new uncertainty theory. No formal quality from training loss.'})
        with (out/'training.jsonl').open('x') as trace, (out/'windows.jsonl').open('x') as windows:
            for step in range(4000):
                data_step = 8000+step
                if step == 0:
                    low, gt, context = initial_low, initial_gt, initial_context
                else:
                    low, gt, context = pool.batch(args.seed, data_step)
                low, gt = low.cuda(), gt.cuda()
                raw, cues = encode(encoder, low)
                assert bool(torch.isfinite(raw).all()) and bool(torch.isfinite(cues).all())
                cpu_fields = motion_for(raw)
                fields = move_fields(cpu_fields, 'cuda')
                validity = sum(float(mask.float().mean()) for _, mask in cpu_fields)/len(cpu_fields)
                if step == 0:
                    with torch.no_grad():
                        fixed = rollout(RecursiveControl('fixed', args.seed).cuda(), raw, cues, fields)
                        deltas = {n: float((rollout(c, raw, cues, fields)-fixed).abs().max()) for n, c in controls.items()}
                    assert all(v <= 1e-5 for v in deltas.values())
                    write(out/'initial-real-window.json', {'deltas_to_fixed05': deltas, 'data_step': data_step,
                          'context': context, 'own_raw_motion_valid_fraction': validity})
                windows.write(json.dumps({'new_step': step+1, 'data_step': data_step,
                    'context': context, 'own_raw_motion_valid_fraction': validity})+'\n')
                losses = {}
                for name, control in controls.items():
                    optimizer = optimizers[name]
                    optimizer.zero_grad(set_to_none=True)
                    output = rollout(control, raw, cues, fields)
                    loss = (output-gt).square().mean()
                    assert bool(torch.isfinite(loss))
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(control.parameters(), 1, error_if_nonfinite=True)
                    optimizer.step()
                    assert all(bool(torch.isfinite(p).all()) for p in control.parameters())
                    updates[name] = step+1
                    row = {'control': name, 'seed': args.seed, 'new_step': step+1,
                           'mse': float(loss.detach()), 'gradient_norm': float(norm),
                           'scalar_alpha': float(control.logit.sigmoid().detach()) if name == 'scalar' else None}
                    trace.write(json.dumps(row, allow_nan=False)+'\n')
                    losses[name] = row['mse']
                if (step+1) % 1000 == 0:
                    for name, control in controls.items():
                        final[name] = checkpoint(control, optimizers[name], name, args.seed, step+1, cache, out, marker)
                if (step+1) % 100 == 0:
                    trace.flush();windows.flush()
                    print(json.dumps({'seed': args.seed, 'new_step': step+1, 'mse': losses,
                                      'wall_seconds': time.perf_counter()-started}), flush=True)
        assert updates == {n: 4000 for n in LEARNED}
        assert all(torch.equal(before[n], t.cpu()) for n, t in backbone.state_dict().items())
        assert all(not p.requires_grad and p.grad is None for p in backbone.parameters())
        complete = True
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'model_updates': updates,
              'total_new_controller_updates': sum(updates.values()), 'frozen_base_unchanged': complete,
              'final_checkpoints': final, 'wall_seconds': time.perf_counter()-started,
              'ended_at': datetime.now().astimezone().isoformat(),
              'scope': 'Training closure only. No quality/cost/independent cue contribution conclusion.'})


if __name__ == '__main__':
    main()
