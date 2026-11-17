"""Two independently optimized descriptor heads, shared frozen stem, fixed 2K/seed."""
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
from shared_geometry_io_r88 import BASE, PROCESS, STEPS, verify, precision, digest, write, append, stamp, acquire_gpu, owned_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(0, 1), required=True)
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    bound = require_initial_binding()
    seed_lock = (ROOT/f'.cache/shared-geometry-r88-seed{args.seed}.lock').open('a')
    fcntl.flock(seed_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root = BASE/f'results/exploration-r88-stage001-seed{args.seed}'
    assert args.attempt > 0 and not (root/'final.json').exists()
    out = root/f'attempt{args.attempt:03d}'; out.mkdir(parents=True, exist_ok=False)
    import cv2
    import torch
    from mixed_anchor_r40 import MixedPool
    from motion_transport_r74 import load_spatial
    from shared_geometry_r88 import make_heads, stem, update, initialize_backend
    from shared_geometry_data_r88 import batch
    precision(); cv2.setNumThreads(1)
    began, complete, steps, inherited, reason = time.perf_counter(), False, 0, 0, 'error'
    gpu_lock = schedule = optimizers = cache = latest = None
    stem_batches = stem_images = 0
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    frozen_path = PROCESS/'training-source-freeze002.json'
    frozen_sha = digest(frozen_path)

    def save(kind):
        nonlocal latest
        path = cache/f'{kind}-step{steps:04d}.pt'
        assert not path.exists()
        value = dict(experiment='R88', seed=args.seed, completed_steps=steps, next_step=steps,
            next_data_step=8000+steps, registered_steps=STEPS, freeze_sha256=frozen_sha,
            backbone=source, models={name: head.state_dict() for name, head in heads.items()},
            folded={name: head.folded().state_dict() for name, head in heads.items()},
            optimizers={name: opt.state_dict() for name, opt in optimizers.items()},
            cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        temporary = path.with_suffix('.pt.tmp')
        torch.save(value, temporary); os.replace(temporary, path)
        latest = dict(at=stamp(), seed=args.seed, attempt=args.attempt, kind=kind,
            completed_steps=steps, next_step=steps, next_data_step=8000+steps, registered_steps=STEPS,
            path=str(path.relative_to(ROOT)), sha256=digest(path), bytes=path.stat().st_size,
            freeze_sha256=frozen_sha, backbone=source, deployed_parameters_each=4507,
            completed_optimizer_updates=2*steps, final_quality_eligible=False)
        write(out/(path.stem+'.json'), latest)

    try:
        frozen = verify(json.loads(frozen_path.read_text()))
        for device in ('cpu', 'cuda'):
            config = json.loads((PROCESS/f'{device}-gate002/config.json').read_text())
            verify(config['frozen'])
            gate = json.loads((PROCESS/f'{device}-gate002/gate.json').read_text())
            assert gate['passed'] and gate['discarded_updates'] == 8 and gate['formal_updates'] == 0
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
            assert saved['experiment'] == 'R88' and saved['seed'] == args.seed
            assert saved['freeze_sha256'] == frozen_sha and saved['registered_steps'] == STEPS
            assert steps == saved['next_step'] == marker['completed_steps'] and 0 <= steps < STEPS
            assert saved['next_data_step'] == 8000+steps
        elif args.attempt != 1:
            priors = [json.loads(p.read_text()) for p in root.glob('attempt*/runtime.json')]
            assert priors and all(r['stop_reason'] == 'resource' and r['completed_steps'] == 0 for r in priors)
        pool = MixedPool(); first = batch(pool, args.seed, steps)
        backbone, source = load_spatial(args.seed)
        heads, rank = make_heads(backbone, args.seed)
        assert not torch.cuda.is_initialized()
        write(out/'config.json', dict(at=stamp(), pid=os.getpid(), seed=args.seed, attempt=args.attempt,
            revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            frozen=frozen, freeze_sha256=frozen_sha, pool=pool.info, backbone=source, rank=rank,
            first_batch=first[2], registered_steps=STEPS, inherited_steps=inherited,
            trainable_degrees={name: sum(p.numel() for p in head.parameters()) for name, head in heads.items()},
            registered_optimizer='Adam lr.001 betas(.9,.999) eps1e-8 wd0 clip1',
            no_DEV_selection=True, scientific_retry=False))
        gpu_lock, schedule, selected = acquire_gpu(bound, out)
        backbone = backbone.cuda()
        heads = initialize_backend(heads, torch.device('cuda'))
        for head in heads.values():
            head.train()
        optimizers = {name: torch.optim.Adam(head.parameters(), lr=1e-3, betas=(.9, .999), eps=1e-8,
                                            weight_decay=0) for name, head in heads.items()}
        if saved:
            assert saved['backbone'] == source
            for name, head in heads.items():
                head.load_state_dict(saved['models'][name], strict=True)
                optimizers[name].load_state_dict(saved['optimizers'][name])
            torch.set_rng_state(saved['cpu_rng']); torch.cuda.set_rng_state(saved['cuda_rng'])
            del saved
        cache = owned_cache(f'checkpoints/exploration-r88/seed{args.seed}/attempt{args.attempt:03d}')
        torch.cuda.reset_peak_memory_stats()
        for step in range(steps, STEPS):
            schedule.check_memory()
            if stop[0]:
                raise ResourcePressure('Cooperative stop at completed paired-head update boundary.')
            inputs, target, identity = first if step == inherited else batch(pool, args.seed, step)
            inputs, target = inputs.cuda(), target.cuda()
            started = time.perf_counter()
            features = stem(backbone, inputs); stem_batches += 1; stem_images += 8
            rows = {name: update(head, optimizers[name], features, target) for name, head in heads.items()}
            assert all(p.grad is None for p in backbone.parameters())
            torch.cuda.synchronize(); work = time.perf_counter()-started
            steps = step+1
            if steps % 500 == 0:
                save('regular')
                append(out/'checkpoint-windows.jsonl', dict(completed_steps=steps, data=identity))
            if steps == inherited+1 or steps % 100 == 0:
                row = dict(at=stamp(), completed_steps=steps, models=rows, stem_batch_calls=stem_batches,
                    stem_image_frames=stem_images, descriptor_image_forwards=2*stem_images,
                    individual_directional_train_fields=stem_images,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(), resource=schedule.summary())
                append(out/'training.jsonl', row)
                write(out/'progress.json', dict(**row, seed=args.seed, pid=os.getpid(), attempt=args.attempt,
                    registered_steps=STEPS, completed_normally=False, latest_checkpoint=latest))
                print(json.dumps({k: v for k, v in row.items() if k != 'resource'}), flush=True)
            if steps < STEPS:
                schedule.yield_after_work(work)
            else:
                schedule.work_seconds += work
        saved = torch.load(ROOT/latest['path'], map_location='cpu', weights_only=True)
        errors = {}
        for name, head in heads.items():
            clone = head.folded()
            clone.load_state_dict(saved['folded'][name], strict=True)
            with torch.no_grad():
                errors[name] = float((head.eval()(features)-clone(features)).abs().max())
        assert max(errors.values()) <= 1e-5
        verify(frozen)
        write(out/'final-reload.json', dict(passed=True, errors=errors, new_DEV_pixels=0))
        latest = dict(latest, final_quality_eligible=True, final_reload=errors)
        write(out/'regular-step2000.json', latest)
        complete, reason = True, 'complete'
    except ResourcePressure:
        reason = 'resource'; (out/'resource-exit.txt').write_text(traceback.format_exc())
        if optimizers is not None and cache is not None:
            save('resource')
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(ended=stamp(), pid=os.getpid(), seed=args.seed, attempt=args.attempt,
            completed_normally=complete, stop_reason=reason, completed_steps=steps, inherited_steps=inherited,
            new_optimizer_updates=(steps-inherited)*2, registered_steps=STEPS, latest_checkpoint=latest,
            stem_batch_calls=stem_batches, stem_image_frames=stem_images, descriptor_image_forwards=2*stem_images,
            individual_directional_train_fields=stem_images, seconds=time.perf_counter()-began,
            resource=schedule.summary() if schedule else None,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None))
        if complete:
            write(root/'final.json', dict(**latest, completed_normally=True, runtime=str((out/'runtime.json').relative_to(ROOT))))
        globals()['_held_seed_lock'], globals()['_held_gpu_lock'] = seed_lock, gpu_lock
    if not complete:
        raise SystemExit(75)


if __name__ == '__main__':
    main()
