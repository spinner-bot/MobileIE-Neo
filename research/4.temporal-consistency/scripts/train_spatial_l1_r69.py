"""Fixed original-MBR spatial L1 reference, one admitted GPU per seed."""
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
import torch.nn.functional as F

from check_spatial_l1_r69 import write
from dynamic_training_r34 import BASE, ROOT, digest
from mixed_anchor_r40 import MixedPool, prediction
from spatial_l1_r69 import initial_model, verify_old_reference, check_one
from training_form_r41 import configure_precision, cuda_headroom


def optimizer_for(model):
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                            lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0)


def checkpoint(model, optimizer, seed, step, low, cache, out):
    model.eval()
    train_path, path = cache/f'training-step{step:04d}.pt', cache/f'deploy-step{step:04d}.pt'
    torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                    seed=seed, step=step, data_step=step-1, task='R69 MobileIE spatial L1'), train_path)
    slim = model.slim().to(low.device).eval()
    delta = None
    if step == 8000:
        with torch.inference_mode():
            delta = float((prediction(model, low)-prediction(slim, low)).abs().max())
    assert sum(p.numel() for p in slim.parameters()) == 4075
    torch.save(dict(state_dict=slim.state_dict(), seed=seed, step=step, data_step=step-1,
                    task='R69 MobileIE spatial L1', training_sha256=digest(train_path)), path)
    marker = dict(seed=seed, step=step, parameters=4075, path=str(path.relative_to(ROOT)),
                  sha256=digest(path), training_path=str(train_path.relative_to(ROOT)),
                  training_sha256=digest(train_path), final_quality_eligible=step == 8000,
                  final_eval_fold_max_abs_delta=delta,
                  final_fold_gate_passed=None if delta is None else delta <= 1e-4)
    write(out/f'step{step:04d}.json', marker)
    model.train()
    return marker


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    configure_precision(); assert not torch.cuda.is_initialized()
    cpu = BASE/'process/exploration-r69/cpu-gate001'
    assert json.loads((cpu/'gate.json').read_text())['passed']
    assert json.loads((cpu/'runtime.json').read_text())['completed_normally']
    cpu_config = json.loads((cpu/'config.json').read_text())
    assert all(digest(BASE/'scripts'/name) == sha for name, sha in cpu_config['source_sha256'].items())
    out = BASE/f'results/exploration-r69-stage001-seed{args.seed}'; out.mkdir(exist_ok=False)
    cache = ROOT/f'.cache/checkpoints/exploration-r69/stage001/seed{args.seed}'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    start, complete, updates, markers = time.perf_counter(), False, 0, {}
    try:
        reference = verify_old_reference(args.seed)
        write(out/'MSE-reference.json', reference)
        pool = MixedPool(); first = pool.batch(args.seed, 0)
        model, info = initial_model(args.seed)
        admission = cuda_headroom(); write(out/'pre_cuda_guard.json', admission); assert admission['allowed']
        gate = check_one(model.cuda(), first[0].cuda(), first[1].cuda(), 'L1')
        write(out/'cuda-gate.json', gate); del model
        model, _ = initial_model(args.seed); model = model.cuda().train(); optimizer = optimizer_for(model)
        cache.mkdir(parents=True, exist_ok=False); torch.cuda.reset_peak_memory_stats()
        names = ['train_spatial_l1_r69.py', 'spatial_l1_r69.py', 'check_spatial_l1_r69.py',
                 'training_form_r41.py', 'mixed_anchor_r40.py', 'dynamic_training_r34.py',
                 'native_training_r14.py', 'exploration_r2_data.py', 'train_exploration_r2.py', 'gpu_guard.py']
        frozen = {name: digest(BASE/'scripts'/name) for name in names}
        windows_path = BASE/f'results/exploration-r65-stage001-seed{args.seed}/windows.jsonl'
        write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), seed=args.seed,
              steps=8000, batch=4, frames=4, crop=128, data_steps=[0, 7999], data=pool.info,
              initialization=info, training_parameters=49978, active_parameters=47239, deployment_parameters=4075,
              objective='unclipped four-frame mean L1; no temporal/auxiliary term',
              optimizer='fresh Adam initially and after4000', lr=1e-4, betas=[.9, .999], eps=1e-8,
              weight_decay=0, clip=1, BN_calls_per_update=4, FP32=True, TF32=False,
              discarded_CUDA_SGD=1, CPU_gate_sha256=digest(cpu/'gate.json'), source_sha256=frozen,
              windows_reference_sha256=digest(windows_path), selection='final8000 only',
              protocol_sha256=digest(BASE/'docs/exploration-r69-matched-spatial-L1-protocol.md'),
              revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              scope='Necessary objective-matched spatial reference, not a novel video method or phone/final-test claim.'))
        with (out/'training.jsonl').open('x') as trace, (out/'windows.jsonl').open('x') as windows, windows_path.open() as expected:
            for step in range(8000):
                if step == 4000: optimizer = optimizer_for(model)
                low, target, identities = first if step == 0 else pool.batch(args.seed, step)
                old = json.loads(next(expected)); assert old['data_step'] == step and old['windows'] == identities
                windows.write(json.dumps(dict(step=step+1, data_step=step, windows=identities))+'\n')
                low, target = low.cuda(), target.cuda(); optimizer.zero_grad(set_to_none=True)
                output = prediction(model, low); loss = F.l1_loss(output, target)
                assert torch.isfinite(output).all() and torch.isfinite(loss)
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                           1., error_if_nonfinite=True))
                optimizer.step(); updates += 1
                error = (output-target).detach()
                row = dict(step=updates, data_step=step, L1=float(loss.detach()), MSE=float(error.square().mean()),
                           synthetic_MAE=float(error[:2].abs().mean()), real_MAE=float(error[2:].abs().mean()),
                           gradient_norm_before_clip=norm, Adam_reset=step in [0, 4000])
                trace.write(json.dumps(row, allow_nan=False)+'\n')
                del output, error, loss
                if updates % 100 == 0:
                    trace.flush(); windows.flush(); print(json.dumps({**row, 'seconds': time.perf_counter()-start}), flush=True)
                if updates % 1000 == 0:
                    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
                    marker = checkpoint(model, optimizer, args.seed, updates, low, cache, out)
                    markers[str(updates)] = marker
                    if updates == 8000: assert marker['final_fold_gate_passed']
            assert next(expected, None) is None
        assert updates == 8000 and all(digest(BASE/'scripts'/name) == sha for name, sha in frozen.items())
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, seed=args.seed, updates=updates,
              checkpoints=markers, seconds=time.perf_counter()-start,
              peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
