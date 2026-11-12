"""Fixed8000 updates per published-loss arm, one seed per admitted GPU."""
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

from check_stablellve_task_r65 import write
from dynamic_training_r34 import BASE, ROOT, digest
from mixed_anchor_r40 import MixedPool
from stablellve_adapter_r15 import MANIFEST
from stablellve_task_r65 import ARMS, PARAMETERS, check_one, initial_model, loss_terms, prepare_offsets
from training_form_r41 import configure_precision, cuda_headroom


def optimizer_for(model):
    return torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    configure_precision(); assert not torch.cuda.is_initialized()
    gate = BASE/'process/exploration-r65/cpu-gate001/gate.json'
    assert json.loads(gate.read_text())['passed']
    out = BASE/f'results/exploration-r65-stage001-seed{args.seed}'; out.mkdir(exist_ok=False)
    cache = ROOT/f'.cache/checkpoints/exploration-r65/stage001/seed{args.seed}'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    start, complete = time.perf_counter(), False
    updates = {arm: 0 for arm in ARMS}; markers = {arm: {} for arm in ARMS}
    try:
        pool = MixedPool(); first = pool.batch(args.seed, 0); first_offset = prepare_offsets(first[1])
        admission = cuda_headroom(); write(out/'pre_cuda_guard.json', admission); assert admission['allowed']
        cuda_gates = {}
        for arm in ARMS:
            model = initial_model().cuda()
            cuda_gates[arm] = check_one(model, first[0].cuda(), first[1].cuda(), first_offset.cuda(), arm)
            del model
        write(out/'cuda-gate.json', dict(passed=all(r['passed'] for r in cuda_gates.values()),
                                       arms=cuda_gates, discarded_SGD_updates=2))
        models = {arm: initial_model().cuda().train() for arm in ARMS}
        optimizers = {arm: optimizer_for(model) for arm, model in models.items()}
        cache.mkdir(parents=True, exist_ok=False)
        torch.cuda.reset_peak_memory_stats()
        names = ['train_stablellve_task_r65.py', 'check_stablellve_task_r65.py', 'stablellve_task_r65.py',
                 'stablellve_adapter_r15.py', 'mixed_anchor_r40.py', 'dynamic_training_r34.py',
                 'native_training_r14.py', 'half_motion_r4.py', 'training_form_r41.py', 'gpu_guard.py']
        frozen = {name: digest(BASE/'scripts'/name) for name in names}
        write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), seed=args.seed, arms=ARMS,
              steps_per_arm=8000, total_formal_updates=16000, data_steps=[0, 7999], batch=4, frames=4, crop=128,
              parameters_per_arm=PARAMETERS, initialization=json.loads(MANIFEST.read_text()), data=pool.info,
              objective='unclipped L1 or L1+20*mean absolute F(W(x))-W(F(x)), first3 source frames',
              flow='half-FB training GT after crop/flip; field only, no next noisy input/teacher target/mask',
              warp='official endpoint grid, reflection, align_corners=False; original CUDA nondeterminism retained',
              channel_convention='training RGB converted to/from official BGR at every network call',
              optimizer='fresh Adam initially and after4000', lr=1e-4, betas=[.9, .999], eps=1e-8,
              weight_decay=0, clip=1, FP32=True, TF32=False, cudnn_deterministic=True,
              deterministic_algorithms_warn_only=True, selection='final8000 only; no hyperparameter/seed search',
              CPU_gate_sha256=digest(gate), discarded_cuda_gate_updates=2,
              protocol_sha256=digest(BASE/'docs/exploration-r65-stablellve-task-adaptation-protocol.md'),
              source_sha256=frozen, upstream_model_sha256=digest(ROOT/'third_party/stablellve/model.py'),
              revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              scope='Published network/loss fixed-budget adaptation; changed data/flow/crop/initialization, no full-paper reproduction/new method/phone cost.'))
        with (out/'training.jsonl').open('x') as trace, (out/'windows.jsonl').open('x') as windows:
            for step in range(8000):
                if step == 4000: optimizers = {arm: optimizer_for(model) for arm, model in models.items()}
                low, target, identities = first if step == 0 else pool.batch(args.seed, step)
                offset = first_offset if step == 0 else prepare_offsets(target)
                windows.write(json.dumps(dict(step=step+1, data_step=step, windows=identities,
                    flow_abs_max=float(offset.abs().max()), flow_mean_absolute=float(offset.abs().mean())))+'\n')
                low, target, offset = low.cuda(), target.cuda(), offset.cuda()
                for arm in ARMS:
                    model, optimizer = models[arm], optimizers[arm]
                    optimizer.zero_grad(set_to_none=True)
                    loss, terms = loss_terms(model, low, target, offset, arm); assert torch.isfinite(loss)
                    loss.backward()
                    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
                    optimizer.step(); updates[arm] += 1
                    error = (terms['prediction']-target).detach()
                    row = dict(arm=arm, step=updates[arm], data_step=step, total=float(loss.detach()),
                               spatial_L1=float(terms['spatial'].detach()), consistency_L1=float(terms['consistency'].detach()),
                               synthetic_MAE=float(error[:2].abs().mean()), real_MAE=float(error[2:].abs().mean()),
                               unclipped_MSE=float(error.square().mean()), gradient_norm_before_clip=norm,
                               Adam_reset=step in [0, 4000])
                    trace.write(json.dumps(row, allow_nan=False)+'\n')
                    del loss, terms, error
                    if updates[arm] % 100 == 0:
                        trace.flush(); windows.flush()
                        print(json.dumps({**row, 'seconds': time.perf_counter()-start}), flush=True)
                    if updates[arm] % 1000 == 0:
                        path = cache/f'{arm}-step{updates[arm]:04d}.pt'
                        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
                        torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(), seed=args.seed,
                            arm=arm, step=updates[arm], data_step=step, task='R65 StableLLVE MixedPool',
                            optimizer_phase=1 if step < 4000 else 2), path)
                        marker = dict(path=str(path.relative_to(ROOT)), sha256=digest(path), seed=args.seed, arm=arm,
                            step=updates[arm], parameters=PARAMETERS, Adam_phase=1 if step < 4000 else 2,
                            loss_last_window=row, final_quality_eligible=updates[arm] == 8000)
                        write(out/f'{arm}-step{updates[arm]:04d}.json', marker); markers[arm][str(updates[arm])] = marker
        assert all(v == 8000 for v in updates.values())
        assert all(digest(BASE/'scripts'/name) == sha for name, sha in frozen.items()); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, seed=args.seed, updates=updates, checkpoints=markers,
              seconds=time.perf_counter()-start, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()
              if torch.cuda.is_initialized() else None, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
