"""Three independent heads, one frozen R44 forward per window, fixed 8K/seed."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
from prebind_shared_gpu import require_initial_binding
from shared_training_schedule import ResourcePressure
from self_reference_io_r82 import BASE, PROCESS, STEPS, digest, write, append, stamp, verify, acquire_gpu, owned_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(0, 1), required=True)
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    bound = require_initial_binding()
    root = BASE/f'results/exploration-r82-stage001-seed{args.seed}'
    seed_lock = (ROOT/f'.cache/self-reference-r82-seed{args.seed}.lock').open('a')
    fcntl.flock(seed_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert args.attempt > 0 and not (root/'final.json').exists()
    out = root/f'attempt{args.attempt:03d}'; out.mkdir(parents=True, exist_ok=False)
    import cv2
    import torch
    from mixed_anchor_r40 import MixedPool
    from motion_transport_r74 import load_spatial, frozen_raw
    from self_reference_r82 import MODES, make_models
    from self_reference_io_r82 import precision, update
    cv2.setNumThreads(1); precision()
    began, complete, steps, inherited, reason = time.perf_counter(), False, 0, 0, 'error'
    gpu_lock = schedule = optimizers = cache = latest = None
    backbone_batches = backbone_images = 0
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    frozen_sha = digest(PROCESS/'source-freeze001.json')

    def save(kind):
        nonlocal latest
        path = cache/f'{kind}-step{steps:04d}.pt'
        assert not path.exists()
        saved = dict(experiment='R82', seed=args.seed, completed_steps=steps, next_step=steps,
            next_data_step=8000+steps, registered_steps=STEPS, freeze_sha256=frozen_sha,
            backbone=source, models={name: net.state_dict() for name, net in nets.items()},
            optimizers={name: opt.state_dict() for name, opt in optimizers.items()},
            cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        temporary = path.with_suffix('.pt.tmp')
        torch.save(saved, temporary); os.replace(temporary, path)
        latest = dict(at=stamp(), seed=args.seed, attempt=args.attempt, kind=kind,
            completed_steps=steps, next_step=steps, next_data_step=8000+steps, registered_steps=STEPS,
            path=str(path.relative_to(ROOT)), sha256=digest(path), bytes=path.stat().st_size,
            freeze_sha256=frozen_sha, backbone=source, new_parameters_each=304,
            total_parameters_each=4379, completed_optimizer_updates=3*steps, final_quality_eligible=False)
        write(out/(path.stem+'.json'), latest)
        return latest

    try:
        frozen = verify()
        cpu_config = json.loads((PROCESS/'cpu-gate001/config.json').read_text())
        assert cpu_config['source_sha256'] == {p: digest(ROOT/p) for p in cpu_config['source_sha256']}
        assert json.loads((PROCESS/'cpu-gate001/gate.json').read_text())['passed']
        cuda = json.loads((PROCESS/'cuda-gate001/gate.json').read_text())
        assert cuda['passed'] and cuda['discarded_updates'] == 12 and cuda['formal_updates'] == 0
        assert cuda['frozen_sources_sha256'] == frozen_sha
        saved = None
        if args.resume:
            marker_path = args.resume.resolve()
            assert marker_path.is_relative_to(root) and marker_path.parent != out
            marker = json.loads(marker_path.read_text())
            prior = json.loads((marker_path.parent/'runtime.json').read_text())
            assert prior['stop_reason'] == 'resource' and not prior['completed_normally']
            assert marker['kind'] == 'resource' and marker['seed'] == args.seed
            assert marker['freeze_sha256'] == frozen_sha and prior['completed_steps'] == marker['completed_steps']
            path = ROOT/marker['path']
            assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo') and digest(path) == marker['sha256']
            saved = torch.load(path, map_location='cpu', weights_only=True)
            inherited = steps = saved['completed_steps']
            assert saved['experiment'] == 'R82' and saved['seed'] == args.seed
            assert saved['freeze_sha256'] == frozen_sha and saved['registered_steps'] == STEPS
            assert steps == saved['next_step'] == marker['completed_steps'] and 0 <= steps < STEPS
            assert saved['next_data_step'] == 8000+steps
        elif args.attempt != 1:
            prior = [json.loads(p.read_text()) for p in root.glob('attempt*/runtime.json')]
            assert prior and all(r['stop_reason'] == 'resource' and r['completed_steps'] == 0 for r in prior)
        pool = MixedPool(); first = pool.batch(args.seed, 8000+steps)
        backbone, source = load_spatial(args.seed)
        nets = make_models(args.seed)
        assert not torch.cuda.is_initialized()
        write(out/'config.json', dict(at=stamp(), pid=os.getpid(), seed=args.seed, attempt=args.attempt,
            revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            frozen=frozen, freeze_sha256=frozen_sha, data=pool.info, backbone=source,
            inherited_steps=inherited, registered_steps=STEPS, first_window=first[2],
            batch=4, frames=4, crop=128, initial_seed=8200+args.seed, modes=MODES,
            optimizer='Adam', lr=.001, betas=[.9,.999], eps=1e-8, weight_decay=0,
            loss='clipped all-four-frame RGB MSE, equal weights', gradient_clip=1, TF32=False,
            scope='Frozen backbone; independent same-init heads. No DEV/checkpoint selection or handset claim.'))
        gpu_lock, schedule, selected = acquire_gpu(bound, out)
        backbone = backbone.cuda()
        nets = {name: net.cuda().train() for name, net in nets.items()}
        optimizers = {name: torch.optim.Adam(net.parameters(), lr=1e-3, betas=(.9,.999),
            eps=1e-8, weight_decay=0) for name, net in nets.items()}
        if saved:
            assert saved['backbone'] == source
            for name, net in nets.items():
                net.load_state_dict(saved['models'][name], strict=True)
                optimizers[name].load_state_dict(saved['optimizers'][name])
            torch.set_rng_state(saved['cpu_rng']); torch.cuda.set_rng_state(saved['cuda_rng'])
            del saved
        cache = owned_cache(f'checkpoints/exploration-r82/seed{args.seed}/attempt{args.attempt:03d}')
        torch.cuda.reset_peak_memory_stats()
        for step in range(steps, STEPS):
            schedule.check_memory()
            if stop[0]:
                raise ResourcePressure('Cooperative stop at a completed joint-update boundary.')
            low, target, identities = first if step == inherited else pool.batch(args.seed, 8000+step)
            low, target = low.cuda(), target.cuda()
            started = time.perf_counter()
            raw = frozen_raw(backbone, low)
            backbone_batches += 4; backbone_images += 16
            rows = [dict(model=name, **update(net, optimizers[name], raw, target)) for name, net in nets.items()]
            torch.cuda.synchronize(); work = time.perf_counter()-started
            steps = step+1
            if steps % 1000 == 0:
                save('regular')
                append(out/'checkpoint-windows.jsonl', dict(completed_steps=steps, data_step=8000+step, identities=identities))
            if steps == inherited+1 or steps % 100 == 0:
                row = dict(at=stamp(), completed_steps=steps, models=rows,
                    backbone_batches=backbone_batches, backbone_image_frames=backbone_images,
                    H_calls={name: net.head_calls for name, net in nets.items()},
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(), resource=schedule.summary())
                append(out/'training.jsonl', row)
                write(out/'progress.json', dict(**row, seed=args.seed, pid=os.getpid(), attempt=args.attempt,
                    registered_steps=STEPS, completed_normally=False, latest_checkpoint=latest))
                print(json.dumps({k:v for k,v in row.items() if k != 'resource'}), flush=True)
            if steps < STEPS:
                schedule.yield_after_work(work)
            else:
                schedule.work_seconds += work
        saved = torch.load(ROOT/latest['path'], map_location='cpu', weights_only=True)
        errors = {}
        for name, net in nets.items():
            clone = make_models(args.seed)[name].cuda().eval()
            clone.load_state_dict(saved['models'][name], strict=True)
            with torch.no_grad():
                errors[name] = float((net.eval().window(raw)-clone.window(raw)).abs().max())
            del clone
        assert max(errors.values()) <= 1e-6
        verify()
        write(out/'final-reload.json', dict(passed=True, errors=errors, new_DEV_pixels=0))
        latest = dict(latest, final_quality_eligible=True, final_reload=errors)
        write(out/'regular-step8000.json', latest)
        complete, reason = True, 'complete'
    except ResourcePressure:
        reason = 'resource'
        (out/'resource-exit.txt').write_text(traceback.format_exc())
        if optimizers is not None and cache is not None:
            save('resource')
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', dict(ended=stamp(), pid=os.getpid(), seed=args.seed, attempt=args.attempt,
            completed_normally=complete, stop_reason=reason, completed_steps=steps, inherited_steps=inherited,
            new_optimizer_updates=(steps-inherited)*3, registered_steps=STEPS, latest_checkpoint=latest,
            backbone_batches=backbone_batches, backbone_image_frames=backbone_images,
            H_calls={name:net.head_calls for name,net in nets.items()} if optimizers is not None else {},
            seconds=time.perf_counter()-began, resource=schedule.summary() if schedule else None,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None))
        if complete:
            write(root/'final.json', dict(**latest, completed_normally=True, runtime=str((out/'runtime.json').relative_to(ROOT))))
        globals()['_held_seed_lock'], globals()['_held_gpu_lock'] = seed_lock, gpu_lock
    if not complete:
        raise SystemExit(75)


if __name__ == '__main__':
    main()
