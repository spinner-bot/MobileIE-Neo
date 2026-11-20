"""Fixed 8K spatial GT recovery; all CUDA qualification updates discarded."""
import argparse
import os
from pathlib import Path
import sys
import time
import math
import copy
import traceback
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--seed', type=int, choices=(0, 1), required=True)
parser.add_argument('--qualify', action='store_true')
parser.add_argument('--attempt', type=int, required=True)
parser.add_argument('--resume', type=Path)
ARGS = parser.parse_args()
from prebind_shared_gpu import require_initial_binding
BOUND = require_initial_binding()
os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '2'

import torch
from study_io import BASE, ROOT, PROCESS, ARMS, PARAMETERS, read, write, append, digest, stamp, verify, precision, models, owned_cache, identity
from mixed_anchor_r40 import MixedPool
from shared_group_v1 import acquire_gpu
from shared_training_schedule import ResourcePressure


def available_kib():
    return next(int(r.split()[1]) for r in Path('/proc/meminfo').read_text().splitlines() if r.startswith('MemAvailable:'))


def main(args):
    precision()
    os.nice(5)
    frozen = verify()
    assert not torch.cuda.is_initialized()
    assert read(PROCESS / 'cpu-preflight002/qualification.json')['passed']
    if not args.qualify:
        qualified = sorted(PROCESS.glob('cuda-spatial-qualification*/runtime.json'))
        assert qualified and read(qualified[-1])['completed_normally']
        assert read(qualified[-1].parent / 'qualification.json')['passed']
        assert read(qualified[-1].parent / 'config.json')['source_freeze'] == frozen
    else:
        assert args.seed == 0
    limit = 2 if args.qualify else 8000
    label = f'cuda-spatial-qualification{args.attempt:03d}' if args.qualify else f'spatial-seed{args.seed}-attempt{args.attempt:03d}'
    out, assets = PROCESS / label, owned_cache(label)
    out.mkdir(parents=True, exist_ok=False)
    assets.mkdir(parents=True, exist_ok=False)
    started, complete, paused = time.perf_counter(), False, False
    counts = dict(common_steps=0, optimizer_updates=0, student_images=0)
    lock = schedule = latest = optimizers = None
    inherited = None
    try:
        if available_kib() < 8 * 1024 * 1024:
            raise ResourcePressure('Host available memory below8GiB preparation reserve')
        pool, students = MixedPool(), models(args.seed)
        if args.resume:
            assert args.resume.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
            inherited = torch.load(args.resume, map_location='cpu', weights_only=False)
            assert inherited['resource_pause'] and inherited['source_freeze'] == frozen
            assert (inherited['seed'], inherited['qualification']) == (args.seed, args.qualify)
            counts = copy.deepcopy(inherited['counts'])
            assert 0 <= counts['common_steps'] <= limit
            for arm in ARMS:
                students[arm].load_state_dict(inherited['models'][arm], strict=True)
            latest = dict(path=str(args.resume), sha256=digest(args.resume), inherited=True, counts=copy.deepcopy(counts))
        prepared = min(counts['common_steps'], limit - 1)
        first = pool.batch(args.seed, prepared)
        write(out / 'config.json', dict(at=stamp(), process=identity(os.getpid()), seed=args.seed,
            qualification=args.qualify, source_freeze=frozen, device='cuda', data=pool.info,
            initial_sources='Independent random models; no official/F0/student qualification weights',
            parameters=PARAMETERS, inherited_counts=copy.deepcopy(counts), common_step_budget=limit,
            first_prepared_data_step=prepared, first_windows=first[2], supervised_target='GT, raw student MSE',
            resume=None if args.resume is None else dict(path=str(args.resume), sha256=digest(args.resume)),
            expected_totals=dict(common_steps=limit, optimizer_updates=2*limit, student_images=32*limit),
            wall_clock_cutoff=False, science_failure_automatic_retry=False, teacher_images=0))
        lock, schedule, _ = acquire_gpu(BOUND, out)
        for student in students.values():
            student.to('cuda').train().requires_grad_(True)
        optimizers = {arm: torch.optim.Adam(student.parameters(), lr=1e-3, betas=(.9, .999), eps=1e-8, weight_decay=0)
                      for arm, student in students.items()}
        if inherited is not None:
            for arm in ARMS:
                optimizers[arm].load_state_dict(inherited['optimizers'][arm])
            torch.set_rng_state(inherited['torch_rng'])
            torch.cuda.set_rng_state_all(inherited['cuda_rng'])
        torch.cuda.reset_peak_memory_stats()

        def checkpoint(resource_pause=False):
            nonlocal latest
            path = assets / f'step{counts["common_steps"]:05d}{"-resource" if resource_pause else ""}.pt'
            payload = dict(seed=args.seed, qualification=args.qualify, resource_pause=resource_pause,
                source_freeze=frozen, counts=copy.deepcopy(counts),
                models={k: m.state_dict() for k, m in students.items()},
                optimizers={k: o.state_dict() for k, o in optimizers.items()},
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
            temporary = path.with_suffix('.tmp')
            torch.save(payload, temporary); temporary.replace(path)
            latest = dict(path=str(path.relative_to(ROOT)), sha256=digest(path), seed=args.seed,
                qualification=args.qualify, counts=copy.deepcopy(counts), resource_pause=resource_pause)
            write(out / (path.stem + '.json'), latest)
            return latest

        for step in range(counts['common_steps'], limit):
            if available_kib() < 4 * 1024 * 1024:
                raise ResourcePressure('Host available memory below4GiB reserve')
            schedule.check_memory()
            low, gt, windows = first if step == prepared else pool.batch(args.seed, step)
            low, gt = low.flatten(0, 1).to('cuda'), gt.flatten(0, 1).to('cuda')
            lr = 1e-5 + .5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * step / 7999))
            torch.cuda.synchronize(); began = time.perf_counter(); statistics = {}
            for arm in ARMS:
                student, optimizer = students[arm], optimizers[arm]
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                prediction = student(low)
                loss = (prediction - gt).square().mean()
                assert torch.isfinite(loss)
                loss.backward()
                parameters = tuple(student.parameters())
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
                norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1))
                assert math.isfinite(norm)
                optimizer.step()
                assert all(torch.isfinite(p).all() for p in parameters)
                statistics[arm] = dict(loss=float(loss.detach()), gradient_norm_before_clip=norm)
            torch.cuda.synchronize(); work = time.perf_counter() - began
            counts.update(common_steps=step+1, optimizer_updates=2*(step+1), student_images=32*(step+1))
            append(out / 'training.jsonl', dict(at=stamp(), common_step=step+1, data_step=step,
                learning_rate=lr, controls=statistics, synchronized_model_work_seconds=work, **counts))
            append(out / 'windows.jsonl', dict(common_step=step+1, data_step=step, windows=windows))
            if step+1 in (2000, 4000, 6000):
                checkpoint()
            if (step+1) % 100 == 0 or args.qualify:
                print(dict(seed=args.seed, **counts), flush=True)
            if step+1 < limit:
                schedule.yield_after_work(work)
            else:
                schedule.work_seconds += work
        assert counts == dict(common_steps=limit, optimizer_updates=2*limit, student_images=32*limit)
        for arm, student in students.items():
            path = assets / (arm + '-final.pt')
            torch.save(student.state_dict(), path)
            exported = torch.load(path, map_location='cpu', weights_only=True)
            assert all(torch.isfinite(value).all() for value in exported.values())
            write(out / (arm + '-final.json'), dict(at=stamp(), seed=args.seed, arm=arm,
                qualification=args.qualify, unique_final=True, updates=limit, parameters=PARAMETERS[arm],
                path=str(path.relative_to(ROOT)), sha256=digest(path), all_reloaded_finite=True))
        verify()
        if args.qualify:
            write(out / 'qualification.json', dict(at=stamp(), passed=True, all_updates_discarded=True,
                supervised_target='GT', no_formal_weight_inheritance=True, **counts))
        complete = True
    except ResourcePressure as error:
        paused = True
        if optimizers is not None and counts['common_steps'] > (0 if inherited is None else inherited['counts']['common_steps']):
            latest = checkpoint(resource_pause=True)
        write(out / 'resource-pause.json', dict(at=stamp(), reason=str(error), resume=latest))
        raise SystemExit(75)
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        final_device, own_final_used_mib = None, None
        if torch.cuda.is_initialized():
            try:
                from prebind_shared_gpu import actual_device
                final_device = actual_device(BOUND)
                own = [p for g in final_device['sample']['GPUs'] for p in g['processes']
                       if int(p.split(',')[1].strip()) == os.getpid()]
                if len(own) == 1 and own[0].split(',')[2].strip().isdigit():
                    own_final_used_mib = int(own[0].split(',')[2].strip())
            except Exception as error:
                final_device = dict(observation_error=str(error))
        write(out / 'runtime.json', dict(at=stamp(), process=identity(os.getpid()), seed=args.seed,
            qualification=args.qualify, completed_normally=complete, resource_pause=paused, counts=counts,
            latest_checkpoint=latest, seconds=time.perf_counter()-started,
            GPU=None if schedule is None else schedule.summary(), CUDA_initialized=torch.cuda.is_initialized(),
            final_actual_device=final_device, own_final_used_mib=own_final_used_mib,
            own_NVML_scope='End-of-run process memory, not lifetime peak; allocator lifetime peaks recorded separately',
            max_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
            max_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0))
        if lock:
            lock.close()


if __name__ == '__main__':
    main(ARGS)
