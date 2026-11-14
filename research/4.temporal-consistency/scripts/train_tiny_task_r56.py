"""One published Nano adapted to the fixed MixedPool8000 target/update protocol."""
import argparse
import json
import os
import subprocess
import time
import traceback
from datetime import datetime

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import cv2
import torch

from dynamic_training_r34 import BASE, ROOT, digest
from mixed_anchor_r40 import MixedPool, prediction
from tiny_task_r56 import initial_model, check_one, bn_counts
from training_form_r41 import configure_precision, cuda_headroom


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def optimizer_for(model):
    return torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    configure_precision(); assert not torch.cuda.is_initialized()
    gate = BASE/'process/exploration-r56/cpu-gate001/gate.json'
    assert json.loads(gate.read_text())['passed']
    out = BASE/f'results/exploration-r56-stage001-seed{args.seed}'; out.mkdir(exist_ok=False)
    start, complete, updates, markers = time.perf_counter(), False, 0, {}
    try:
        pool = MixedPool(); first = pool.batch(args.seed, 0)
        admission = cuda_headroom(); write(out/'pre_cuda_guard.json', admission); assert admission['allowed']
        model, _ = initial_model(); model = model.cuda()
        cuda_gate = check_one(model, first[0].cuda(), first[1].cuda())
        write(out/'cuda-gate.json', cuda_gate); del model
        # Reload every original parameter and BN buffer; inherit no discarded update.
        model, source = initial_model(); model = model.cuda().train()
        initial_counts = bn_counts(model); optimizer = optimizer_for(model)
        torch.cuda.reset_peak_memory_stats()
        cache = ROOT/f'.cache/checkpoints/exploration-r56/stage001/seed{args.seed}'
        assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
        cache.mkdir(parents=True, exist_ok=False)
        names = ['train_tiny_task_r56.py', 'tiny_task_r56.py', 'check_tiny_task_r56.py', 'multinex_reference_r55.py',
                 'mixed_anchor_r40.py', 'dynamic_training_r34.py', 'native_training_r14.py', 'training_form_r41.py', 'gpu_guard.py']
        write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), seed=args.seed,
              steps=8000, data_steps=[0, 7999], batch=4, target_frames=4, crop=128, parameters=696,
              initialization=source, data=pool.info, objective='unclipped MSE four original targets',
              BN='Four sequential B4 calls per update; original BN kept', BN_initial_counts=initial_counts,
              optimizer='fresh Adam initially and after4000', lr=1e-4, betas=[.9, .999], eps=1e-8,
              weight_decay=0, clip=1, FP32=True, TF32=False, selection='final8000 only, no hyperparameter or seed search',
              CPU_gate_sha256=digest(gate), CUDA_gate=cuda_gate,
              protocol_sha256=digest(BASE/'docs/exploration-r56-tiny-task-adaptation-protocol.md'),
              source_sha256={n: digest(BASE/'scripts'/n) for n in names},
              upstream_source_sha256=digest(ROOT/'third_party/multinex/basicsr/models/archs/MultinexNano_arch.py'),
              revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              scope='Fixed budget published spatial reference; unequal parameterization/pretraining/compute, no new-method or full-convergence claim.'))
        with (out/'training.jsonl').open('x') as trace, (out/'windows.jsonl').open('x') as windows:
            for step in range(8000):
                if step == 4000: optimizer = optimizer_for(model)
                low, target, identities = first if step == 0 else pool.batch(args.seed, step)
                windows.write(json.dumps(dict(step=step+1, data_step=step, windows=identities))+'\n')
                low, target = low.cuda(), target.cuda()
                optimizer.zero_grad(set_to_none=True)
                error = (prediction(model, low)-target).square(); loss = error.mean(); assert torch.isfinite(loss)
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
                optimizer.step(); updates += 1
                counts = bn_counts(model)
                assert all(counts[n]-value == updates*4 for n, value in initial_counts.items())
                row = dict(step=updates, data_step=step, MSE=float(loss.detach()),
                           synthetic_MSE=float(error[:2].detach().mean()), real_MSE=float(error[2:].detach().mean()),
                           gradient_norm_before_clip=norm, Adam_reset=step in [0, 4000], BN_counts=counts)
                trace.write(json.dumps(row, allow_nan=False)+'\n')
                if updates % 100 == 0:
                    trace.flush(); windows.flush()
                    print(json.dumps({**row, 'seconds': time.perf_counter()-start}), flush=True)
                if updates % 1000 == 0:
                    path = cache/f'nano-step{updates:04d}.pt'
                    torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(), seed=args.seed,
                                    step=updates, data_step=step, task='R56 MixedPool lowlight Nano',
                                    optimizer_phase=1 if step < 4000 else 2), path)
                    marker = dict(path=str(path.relative_to(ROOT)), sha256=digest(path), step=updates, seed=args.seed,
                                  parameters=696, BN_counts=counts, Adam_phase=1 if step < 4000 else 2,
                                  loss_last_window=row, final_quality_eligible=updates == 8000)
                    write(out/f'nano-step{updates:04d}.json', marker); markers[str(updates)] = marker
        assert updates == 8000 and markers['8000']['final_quality_eligible']; complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, steps=updates, checkpoints=markers,
              seconds=time.perf_counter()-start, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()
              if torch.cuda.is_initialized() else None, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    main()
