"""Four discarded R74 updates to measure shared-GPU training resource demand."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from shared_gpu_admission import ROOT, assess, snapshot, sustained_samples, write
sys.path.insert(0, str(ROOT/'research/4.temporal-consistency/scripts'))
import cv2
import torch
from motion_transport_r74 import BASE, make_models, load_spatial, frozen_raw, teacher_proxy, sequence, proxy_loss
from mixed_anchor_r40 import MixedPool
from training_form_r41 import configure_precision


def main():
    out = BASE/'process/exploration-r74/shared-gpu-probe001'
    out.mkdir(exist_ok=False)
    start, complete, updates, measurements, observations = time.perf_counter(), False, 0, [], []
    lock = None
    frozen = json.loads((BASE/'process/exploration-r74/finalize002/config.json').read_text())['source_sha256']
    def unchanged():
        return all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest() == sha for n, sha in frozen.items())
    try:
        assert unchanged()
        torch.set_num_threads(2); cv2.setNumThreads(1); configure_precision()
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        assert not torch.cuda.is_initialized()
        pool = MixedPool(); spatial, source = load_spatial(0); nets = make_models(0)
        low, gt, context = pool.batch(0, 8000); target = teacher_proxy(gt)
        write(out/'config.json', dict(seed=0, data_step=8000, context=context, spatial_source=source,
            preparation_seconds=time.perf_counter()-start, allocator_cap_mib=2048,
            expected_total_increment_mib=4096, target_compute_duty_fraction=.15,
            expected_discarded_motion_updates=4, expected_frozen_spatial_centers=32,
            formal_updates=0, DEV_rows=0, weights_inherited=False,
            scope='First-window resource trial, not formal training, quality or phone timing.'))
        print('CPU preparation complete; sampling shared GPU headroom', flush=True)
        samples = sustained_samples(); admission = assess(samples)
        write(out/'admission.json', dict(samples=samples, **admission))
        selected = admission['selected']
        if selected is None:
            raise RuntimeError('No measured capacity; no CUDA work launched')
        gpu = selected['index']
        lock = (ROOT/f'.cache/gpu-{gpu}.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = snapshot(); observations.append(current)
        row = next(r for r in current['GPUs'] if r['uuid'] == selected['uuid'])
        if row['total_mib']-row['used_mib'] < selected['required_free_mib']:
            raise RuntimeError('Headroom changed before CUDA initialization')
        os.environ['CUDA_VISIBLE_DEVICES'] = selected['uuid']
        torch.cuda.set_per_process_memory_fraction(2048/row['total_mib'], device=0)
        torch.cuda.reset_peak_memory_stats()
        spatial = spatial.cuda(); nets = {n: net.cuda().train() for n, net in nets.items()}
        optimizers = {n: torch.optim.Adam(net.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8)
                      for n, net in nets.items()}
        low, gt = low.cuda(), gt.cuda(); target = tuple(value.cuda() for value in target)
        for step in range(2):
            torch.cuda.synchronize(); began = time.perf_counter()
            raw = frozen_raw(spatial, low); diagnostics = []
            for name, net in nets.items():
                optimizer = optimizers[name]; optimizer.zero_grad(set_to_none=True)
                output, fields = sequence(net, raw)
                reconstruction = (output.clamp(0, 1)-gt).square().mean()
                proxy, _ = proxy_loss(fields, target)
                loss = reconstruction if name == 'task' else proxy
                assert torch.isfinite(loss) and torch.isfinite(output).all()
                loss.backward()
                norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True))
                assert all(p.grad is None for p in spatial.parameters())
                optimizer.step(); updates += 1
                diagnostics.append(dict(model=name, loss=float(loss.detach()), gradient_norm=norm))
            torch.cuda.synchronize(); elapsed = time.perf_counter()-began
            measurements.append(dict(batch_iteration=step+1, synchronized_work_seconds=elapsed,
                                     cooperative_pause_seconds=elapsed*(1/.15-1), diagnostics=diagnostics))
            observations.append(snapshot())
            row = next(r for r in observations[-1]['GPUs'] if r['uuid'] == selected['uuid'])
            if row['total_mib']-row['used_mib'] < selected['incumbent_growth_reserve_mib']+1024:
                raise RuntimeError('Incumbent growth reserve exhausted; yielding GPU')
            time.sleep(elapsed*(1/.15-1))
        assert updates == 4 and unchanged()
        write(out/'result.json', dict(completed=True, selected_gpu=selected,
            measured_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
            measured_peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
            measurements=measurements, observations=observations,
            frozen42_sources_unchanged=True,
            limitations='128-crop training only; context/external allocations excluded from allocator cap. '
                        'No incumbent throughput measurement; no zero-slowdown guarantee. '
                        'Full-resolution perceptual metric requires a separate estimate.'))
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, discarded_motion_updates=updates,
            formal_updates=0, DEV_rows=0, CUDA_initialized=torch.cuda.is_initialized(),
            seconds=time.perf_counter()-start, frozen42_sources_unchanged=unchanged()))
        # Keep the project-local lock until this process exits and releases CUDA.
        if lock is not None:
            globals()['_held_gpu_lock'] = lock


if __name__ == '__main__':
    main()
