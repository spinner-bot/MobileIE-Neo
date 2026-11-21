"""Source-fixed I1-S learning; bind the physical UUID before importing Torch."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import argparse
import copy
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from prebind_shared_gpu import actual_device, require_initial_binding
BOUND = require_initial_binding()

import cv2
import numpy as np
import torch
from simple_io import BASE, append, digest, identity, owned_cache, read, stamp, verify_training, write
from simple_core import make_models, objective, sequence
from pool8 import Pool8
from simple_gpu import acquire
from shared_training_schedule import ResourcePressure
from motion_transport_r74 import load_spatial


def available_kib():
    return next(int(r.split()[1]) for r in Path('/proc/meminfo').read_text().splitlines()
                if r.startswith('MemAvailable:'))


def main(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cv2.setNumThreads(1)
    os.nice(5)
    frozen = verify_training()
    assert read(BASE / 'process/simple-preflight001/qualification.json')['passed']
    assert not torch.cuda.is_initialized()
    if args.qualify:
        assert args.seed == 0
    else:
        proofs = sorted((BASE / 'process').glob('simple-cuda-qualification*/qualification.json'))
        assert proofs and read(proofs[-1])['passed']
        assert read(proofs[-1].parent / 'config.json')['source_freeze'] == frozen
    label = (f'simple-cuda-qualification{args.attempt:03d}' if args.qualify
             else f'simple-train-seed{args.seed}-attempt{args.attempt:03d}')
    out, assets = BASE / 'process' / label, owned_cache(label)
    out.mkdir(parents=True, exist_ok=False)
    assets.mkdir(parents=True, exist_ok=False)
    limit = 2 if args.qualify else 80000
    began, complete, paused = time.perf_counter(), False, False
    counts = dict(common_steps=0, optimizer_updates=0, student_images=0)
    lock = schedule = optimizers = latest = inherited = None
    models = {}
    setup_images = 0
    try:
        if available_kib() < 8 * 1024**2:
            raise ResourcePressure('Host preparation reserve below8GiB')
        torch.manual_seed(9300 + args.seed)
        original, provenance = load_spatial(args.seed)
        base_reference = {n: p.detach().clone() for n, p in original.named_parameters()}
        models = make_models(original)
        setup_images = sum(m.frame_calls for m in models.values())
        if args.resume:
            assert args.resume.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
            inherited = torch.load(args.resume, map_location='cpu', weights_only=False)
            assert inherited['resource_pause'] and inherited['source_freeze'] == frozen
            assert (inherited['seed'], inherited['qualification']) == (args.seed, args.qualify)
            counts = copy.deepcopy(inherited['counts'])
            assert counts['optimizer_updates'] == 2 * counts['common_steps']
            assert counts['student_images'] == 16 * counts['common_steps']
            for level, model in models.items():
                model.load_state_dict(inherited['models'][level], strict=True)
            latest = dict(path=str(args.resume.relative_to(ROOT)), sha256=digest(args.resume),
                          counts=copy.deepcopy(counts), inherited=True)
        pool = Pool8()
        prepared = counts['common_steps']
        first = pool.batch(args.seed, prepared) if prepared < limit else None
        write(out / 'config.json', dict(at=stamp(), process=identity(), source_freeze=frozen,
              seed=args.seed, qualification=args.qualify, common_step_budget=limit,
              inherited_counts=copy.deepcopy(counts), data=pool.info, F0_source=provenance,
              temporal_length=8, batch_size=1, formal_steps_per_level=80000,
              parameters=dict(all=4141, output=4078), first_window=None if first is None else first[2],
              CPU_synthetic_materialization_F0_images=setup_images,
              qualified_weight_or_Adam_inheritance=False, wall_clock_cutoff=False,
              scientific_retry=False, CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG']))
        lock, schedule = acquire(BOUND, out, args.qualify)
        for model in models.values():
            model.to('cuda').train()
        optimizers = {k: torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-3)
                      for k, m in models.items()}
        if inherited is not None:
            for level, optimizer in optimizers.items():
                optimizer.load_state_dict(inherited['optimizers'][level])
            torch.set_rng_state(inherited['torch_rng'])
            torch.cuda.set_rng_state_all(inherited['cuda_rng'])
            random.setstate(inherited['python_rng'])
            np.random.set_state(inherited['numpy_rng'])
        torch.cuda.reset_peak_memory_stats()

        def checkpoint(resource_pause=False):
            nonlocal latest
            path = assets / f'step{counts["common_steps"]:05d}{"-resource" if resource_pause else ""}.pt'
            temporary = path.with_suffix('.tmp')
            torch.save(dict(seed=args.seed, qualification=args.qualify, resource_pause=resource_pause,
                       source_freeze=frozen, counts=copy.deepcopy(counts),
                       models={k: m.state_dict() for k, m in models.items()},
                       optimizers={k: o.state_dict() for k, o in optimizers.items()},
                       torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                       python_rng=random.getstate(), numpy_rng=np.random.get_state()), temporary)
            temporary.replace(path)
            latest = dict(path=str(path.relative_to(ROOT)), sha256=digest(path), counts=copy.deepcopy(counts),
                          resource_pause=resource_pause, seed=args.seed, qualification=args.qualify)
            write(out / (path.stem + '.json'), latest)

        for step in range(counts['common_steps'], limit):
            if available_kib() < 4 * 1024**2:
                raise ResourcePressure('Host running reserve below4GiB')
            low, target, metadata = first if step == prepared else pool.batch(args.seed, step)
            schedule.check_memory()
            low, target = low.to('cuda'), target.to('cuda')
            lr = 1e-3 if step < 40000 else (1e-4 if step < 60000 else 1e-5)
            torch.cuda.synchronize()
            work_began, details = time.perf_counter(), {}
            for level, model in models.items():
                optimizer = optimizers[level]
                optimizer.param_groups[0]['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                outputs = sequence(model, low)
                counts['student_images'] += 8
                assert torch.isfinite(outputs).all()
                loss = objective(outputs, target)
                assert torch.isfinite(loss)
                loss.backward()
                trainable = [p for p in model.parameters() if p.requires_grad]
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
                assert all(p.grad is None for p in model.original_parameters().values())
                optimizer.step()
                counts['optimizer_updates'] += 1
                assert all(torch.isfinite(p).all() for p in trainable)
                details[level] = float(loss.detach())
                del outputs, loss
            torch.cuda.synchronize()
            work = time.perf_counter() - work_began
            counts['common_steps'] = step + 1
            append(assets / 'windows.jsonl', dict(common_step=step+1, window=metadata))
            if args.qualify or (step + 1) % 1000 == 0:
                append(out / 'training.jsonl', dict(at=stamp(), lr=lr, last_batch_unified_loss=details,
                       last_synchronized_student_seconds=work, elapsed_seconds=time.perf_counter()-began, **counts))
                print(dict(at=stamp(), seed=args.seed, elapsed_seconds=time.perf_counter()-began, **counts), flush=True)
            if not args.qualify and (step + 1) % 10000 == 0 and step + 1 < limit:
                checkpoint()
            if step + 1 < limit:
                schedule.after_work(work)
            else:
                schedule.work += work
        assert counts == dict(common_steps=limit, optimizer_updates=2*limit, student_images=16*limit)
        assert all(torch.equal(p.detach().cpu(), base_reference[n])
                   for m in models.values() for n, p in m.original_parameters().items())
        verify_training()
        device = actual_device(BOUND)
        own = [p for row in device['sample']['GPUs'] for p in row['processes']
               if int(p.split(',')[1].strip()) == os.getpid()]
        assert device['matched'] and len(own) == 1
        if args.qualify:
            assert int(own[0].split(',')[2].strip()) <= 4096
            assert torch.cuda.max_memory_allocated() <= 2 * 1024**3
            write(out / 'qualification.json', dict(at=stamp(), passed=True, all_updates_discarded=True,
                  no_qualification_weights_exported=True, counts=counts, actual_UUID_matched=True,
                  own_final_NVML_mib=int(own[0].split(',')[2].strip()),
                  maximum_allocated_bytes=torch.cuda.max_memory_allocated()))
        else:
            for level, model in models.items():
                path = assets / (level + '-final.pt')
                torch.save(model.state_dict(), path)
                exported = torch.load(path, map_location='cpu', weights_only=True)
                assert all(torch.isfinite(v).all() for v in exported.values())
                write(out / (level + '-final.json'), dict(at=stamp(), seed=args.seed, level=level,
                      unique_final=True, qualification=False, updates=80000,
                      registered_parameters=4141 if level == 'all' else 4078,
                      all_reloaded_finite=True, path=str(path.relative_to(ROOT)), sha256=digest(path)))
        windows = assets / 'windows.jsonl'
        write(out / 'window-log.json', dict(path=str(windows.relative_to(ROOT)), sha256=digest(windows),
              bytes=windows.stat().st_size, actual_completed_windows_this_attempt=limit-prepared,
              all_eight_indices_recorded=True))
        complete = True
    except ResourcePressure as error:
        paused = True
        if optimizers is not None and counts['common_steps'] > (0 if inherited is None else inherited['counts']['common_steps']):
            checkpoint(True)
        write(out / 'resource-pause.json', dict(at=stamp(), reason=str(error), resume=latest))
        raise SystemExit(75)
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        actual = None
        if torch.cuda.is_initialized():
            try:
                actual = actual_device(BOUND)
            except Exception as error:
                actual = dict(observation_error=str(error))
        write(out / 'runtime.json', dict(at=stamp(), process=identity(), seed=args.seed, qualification=args.qualify,
              completed_normally=complete, resource_pause=paused, counts=counts, latest_checkpoint=latest,
              CPU_synthetic_materialization_F0_images=setup_images,
              seconds=time.perf_counter()-began, final_actual_device=actual,
              CUDA_initialized=torch.cuda.is_initialized(), GPU=None if schedule is None else schedule.summary(),
              maximum_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
              maximum_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0))
        if lock:
            lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(0, 1), required=True)
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--qualify', action='store_true')
    parser.add_argument('--resume', type=Path)
    main(parser.parse_args())
