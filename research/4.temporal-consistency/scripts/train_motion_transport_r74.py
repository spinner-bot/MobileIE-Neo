"""Two fixed motion-learning conditions; frozen R44 spatial network, final4000 only."""
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

from motion_transport_r74 import (BASE, ROOT, PROTOCOL, MODES, MotionTransport,
        make_models, load_spatial, frozen_raw, teacher_proxy, sequence, proxy_loss)
from check_motion_transport_r74 import write
from dynamic_training_r34 import digest
from mixed_anchor_r40 import MixedPool
from training_form_r41 import configure_precision, cuda_headroom


def checkpoint(net, optimizer, name, seed, step, raw, spatial_source, cache, out):
    path = cache/f'{name}-step{step:04d}.pt'
    torch.save(dict(state_dict={k: v.detach().cpu() for k, v in net.state_dict().items()},
                    optimizer=optimizer.state_dict(), model=name, seed=seed, new_step=step,
                    spatial_source=spatial_source, spatial_updates=8000, motion_updates=step), path)
    delta = None
    if step == 4000:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        clone = MotionTransport().to(raw.device).eval()
        clone.load_state_dict(saved['state_dict'], strict=True)
        with torch.no_grad():
            expected, _ = sequence(net.eval(), raw); actual, _ = sequence(clone, raw)
            assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
            delta = float((expected-actual).abs().max())
        net.train()
    marker = dict(model=name, seed=seed, new_step=step, spatial_updates=8000,
            motion_updates=step, path=str(path.relative_to(ROOT)), sha256=digest(path),
            motion_parameters=sum(p.numel() for p in net.parameters()),
            total_parameters_including_frozen_spatial=4075+sum(p.numel() for p in net.parameters()),
            spatial_source=spatial_source, final_reload_max_delta=delta,
            final_reload_passed=None if delta is None else delta <= 1e-6)
    write(out/f'{name}-step{step:04d}.json', marker)
    return marker


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    parser.add_argument('--admission-attempt', type=int, default=1); args = parser.parse_args()
    cpu_gate = BASE/'process/exploration-r74/cpu-gate001/runtime.json'
    assert json.loads(cpu_gate.read_text())['passed']
    qualified = json.loads((BASE/'process/exploration-r74/cuda-qualification.json').read_text())
    gate = ROOT/qualified['gate_runtime']
    assert qualified['passed'] and digest(gate) == qualified['gate_sha256']
    cuda_gate = json.loads(gate.read_text()); assert cuda_gate['passed'] and cuda_gate['discarded_motion_updates'] == 8
    assert json.loads((ROOT/qualified['guard']).read_text())['returncode'] == 0
    cv2.setNumThreads(1); torch.set_num_threads(2); configure_precision()
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False; assert not torch.cuda.is_initialized()
    spatial, spatial_source = load_spatial(args.seed); nets = make_models(args.seed); pool = MixedPool()
    first_low, first_gt, first_context = pool.batch(args.seed, 8000)
    first_proxy = teacher_proxy(first_gt)
    admission = cuda_headroom()
    write(BASE/f'process/exploration-r74/seed{args.seed}-internal-admission{args.admission_attempt:03d}.json', admission)
    if not admission['allowed']: raise SystemExit(75)
    out = BASE/f'results/exploration-r74-stage001-seed{args.seed}'; out.mkdir(exist_ok=False)
    began, complete, updates, final = time.perf_counter(), False, {name: 0 for name in MODES}, {}
    try:
        spatial = spatial.cuda().eval(); nets = {name: net.cuda().train() for name, net in nets.items()}
        optimizers = {name: torch.optim.Adam(net.parameters(), lr=1e-4, betas=(.9, .999),
                      eps=1e-8, weight_decay=0) for name, net in nets.items()}
        cache = ROOT/f'.cache/checkpoints/exploration-r74/stage001/seed{args.seed}'
        assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
        cache.mkdir(parents=True, exist_ok=False)
        names = ['motion_transport_r74.py', 'train_motion_transport_r74.py', 'mixed_anchor_r40.py',
                 'half_motion_r4.py', 'budget_causal_r33.py', 'training_form_r41.py']
        write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), seed=args.seed, models=MODES,
              new_motion_steps_each=4000, spatial_updates=8000, spatial_always_frozen_eval=True,
              spatial_source=spatial_source, motion_initial_seed=7400+args.seed, first_data_step=8000, last_data_step=11999,
              batch=4, frames=4, crop=128, lr=1e-4, betas=[.9, .999], eps=1e-8, weight_decay=0, clip=1,
              losses=dict(task='clipped output all-frame MSE', proxy='covered native L1/4 plus unit coverage BCE'),
              same_initial_motion_weights=True, shared_frozen_raw_computation=True,
              expected_frozen_enhancement_batch_calls=16000, expected_frozen_enhancement_centers=64000,
              expected_motion_batch_pair_calls=24000, expected_motion_individual_pair_predictions=96000,
              training_FLOPs_not_matched=True, data=pool.info, pre_CUDA_admission=admission,
              protocol_sha256=digest(PROTOCOL), source_sha256={n: digest(BASE/'scripts'/n) for n in names},
              revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              selection='Only final4000; no inherited qualification updates, wall cutoff or scientific retry.'))
        with (out/'training.jsonl').open('x') as trace, (out/'windows.jsonl').open('x') as windows:
            for step in range(4000):
                if step == 0:
                    low, gt, context, target = first_low, first_gt, first_context, first_proxy
                else:
                    low, gt, context = pool.batch(args.seed, 8000+step); target = teacher_proxy(gt)
                windows.write(json.dumps(dict(new_step=step+1, data_step=8000+step, windows=context))+'\n')
                low, gt = low.cuda(), gt.cuda(); target = tuple(value.cuda() for value in target)
                raw = frozen_raw(spatial, low)
                last = []
                for name, net in nets.items():
                    optimizer = optimizers[name]; optimizer.zero_grad(set_to_none=True)
                    prediction, fields = sequence(net, raw)
                    reconstruction = (prediction.clamp(0, 1)-gt).square().mean()
                    proxy, proxy_stats = proxy_loss(fields, target)
                    loss = reconstruction if name == 'task' else proxy
                    assert torch.isfinite(loss) and torch.isfinite(prediction).all()
                    loss.backward()
                    norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True))
                    assert all(p.grad is None for p in spatial.parameters())
                    optimizer.step(); updates[name] += 1
                    row = dict(new_step=step+1, data_step=8000+step, model=name, loss=float(loss.detach()),
                          clipped_output_MSE=float(reconstruction.detach()), gradient_norm_before_clip=norm,
                          native_flow_abs_mean=float(torch.stack([p[0].detach().abs().mean() for p in fields]).mean()),
                          history_coefficient_mean=float(.5*torch.stack([p[1].detach().sigmoid().mean() for p in fields]).mean()),
                          **proxy_stats)
                    trace.write(json.dumps(row, allow_nan=False)+'\n'); last.append(row)
                    if (step+1) % 1000 == 0:
                        marker = checkpoint(net, optimizer, name, args.seed, step+1, raw, spatial_source, cache, out)
                        if step == 3999:
                            final[name] = marker; assert marker['final_reload_passed']
                if (step+1) % 100 == 0:
                    trace.flush(); windows.flush()
                    print(json.dumps(dict(new_step=step+1, models=last, seconds=time.perf_counter()-began)), flush=True)
        assert updates == {name: 4000 for name in MODES}; complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, motion_model_updates=updates,
              total_new_motion_updates=sum(updates.values()), spatial_model_updates=0, final_markers=final,
              seconds=time.perf_counter()-began, ended=datetime.now().astimezone().isoformat(),
              scope='Fixed formal motion budget; frozen spatial network stays at8000. No quality or phone claim.'))


if __name__ == '__main__': main()
