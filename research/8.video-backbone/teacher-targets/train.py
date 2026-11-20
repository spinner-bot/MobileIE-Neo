"""Fixed-budget T1 controls; qualifications and formal runs never share weights."""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--seed', type=int, choices=(0, 1), required=True)
parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
parser.add_argument('--qualify', action='store_true')
parser.add_argument('--attempt', type=int, required=True)
parser.add_argument('--resume', type=Path)
ARGS = parser.parse_args()
if ARGS.device == 'cuda':
    from prebind_shared_gpu import require_initial_binding
    BOUND = require_initial_binding()
else:
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert ARGS.qualify, 'Formal CPU execution needs a separate explicit backend amendment'
    BOUND = None
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'

import copy
import torch
from study_io import ROOT, BASE, PROCESS, ARMS, read, write, append, stamp, digest, verify, owned_cache, precision
from mixed_anchor_r40 import MixedPool
from motion_transport_r74 import load_spatial
from shared_group_v1 import acquire_gpu
from shared_training_schedule import ResourcePressure
from targets import teacher_targets, objective


def available_kib():
    return next(int(row.split()[1]) for row in Path('/proc/meminfo').read_text().splitlines()
                if row.startswith('MemAvailable:'))


def main(args):
    precision()
    os.nice(5)
    frozen = verify()
    assert available_kib() >= 8 * 1024 * 1024
    assert not torch.cuda.is_initialized()
    if args.qualify:
        assert args.seed == 0
    else:
        for backend in ('cpu', 'cuda'):
            attempts = sorted(PROCESS.glob(f'{backend}-qualification*'))
            assert attempts and read(attempts[-1] / 'qualification.json')['passed']
            assert read(attempts[-1] / 'runtime.json')['completed_normally']
            assert read(attempts[-1] / 'config.json')['source_freeze'] == frozen
    limit = 2 if args.qualify else 4000
    label = f'{args.device}-qualification{args.attempt:03d}' if args.qualify else f'train-seed{args.seed}-attempt{args.attempt:03d}'
    out = PROCESS / label
    out.mkdir(parents=True, exist_ok=False)
    assets = owned_cache(label)
    assets.mkdir(parents=True, exist_ok=False)
    counts = dict(common_steps=0, optimizer_updates=0, teacher_images=0, student_images=0, DIS_pairs=0)
    started = time.perf_counter()
    complete = paused = False
    teacher = models = optimizers = None
    lock = schedule = latest = None
    inherited = None
    try:
        torch.manual_seed(19000 + args.seed)
        pool = MixedPool()
        teacher, provenance = load_spatial(args.seed)
        models = {arm: copy.deepcopy(teacher).train().requires_grad_(True) for arm in ARMS}
        assert all(sum(p.numel() for p in model.parameters()) == 4075 for model in models.values())
        initial_same = all(torch.equal(value, teacher.state_dict()[key])
                           for model in models.values() for key, value in model.state_dict().items())
        assert initial_same
        if args.resume:
            inherited = torch.load(args.resume, map_location='cpu', weights_only=False)
            assert inherited['resource_pause'] and inherited['source_freeze'] == frozen
            assert (inherited['seed'], inherited['device'], inherited['qualification']) == (args.seed, args.device, args.qualify)
            counts = copy.deepcopy(inherited['counts'])
            assert 0 <= counts['common_steps'] <= limit
            for arm in ARMS:
                models[arm].load_state_dict(inherited['models'][arm], strict=True)
            resume_path = ROOT / args.resume
            assert resume_path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
            recorded_path = str(resume_path.relative_to(ROOT)) if resume_path.is_relative_to(ROOT) else str(resume_path)
            latest = dict(path=recorded_path, sha256=digest(args.resume),
                          seed=args.seed, qualification=args.qualify, counts=copy.deepcopy(counts),
                          resource_pause=True, inherited_checkpoint=True)
        prepared_step = 8000 + min(counts['common_steps'], limit - 1)
        first = pool.batch(args.seed, prepared_step)
        write(out / 'config.json', dict(at=stamp(), pid=os.getpid(), seed=args.seed,
              qualification=args.qualify, device=args.device, source_freeze=frozen,
              teacher=provenance, data=pool.info, first_prepared_data_step=prepared_step,
              first_windows=first[2], common_step_budget=limit, per_control_budget=limit,
              inherited_counts=copy.deepcopy(counts), same_original_initial_weights=initial_same,
              resume=None if args.resume is None else dict(path=str(args.resume), sha256=digest(args.resume)),
              optimizer='Independent fresh Adam1e-4 (.9,.999),eps1e-8,wd0,globalclip1; resource resume retains all states',
              expected_totals=dict(common_steps=limit, optimizer_updates=3*limit,
                  teacher_images=16*limit, student_images=48*limit, DIS_pairs=12*limit),
              Torch_threads=2, OpenCV_threads=1, wall_clock_cutoff=False,
              scientific_failure_automatic_retry=False))
        if args.device == 'cuda':
            lock, schedule, _ = acquire_gpu(BOUND, out)
        device = torch.device(args.device)
        teacher.to(device)
        for model in models.values():
            model.to(device)
        optimizers = {arm: torch.optim.Adam(model.parameters(), lr=1e-4,
                      betas=(.9, .999), eps=1e-8, weight_decay=0) for arm, model in models.items()}
        if inherited is not None:
            for arm in ARMS:
                optimizers[arm].load_state_dict(inherited['optimizers'][arm])
            torch.set_rng_state(inherited['torch_rng'])
            if args.device == 'cuda':
                torch.cuda.set_rng_state_all(inherited['cuda_rng'])
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        def checkpoint(resource_pause=False):
            nonlocal latest
            path = assets / f'step{counts["common_steps"]:04d}{"-resource" if resource_pause else ""}.pt'
            payload = dict(seed=args.seed, device=args.device, qualification=args.qualify,
                resource_pause=resource_pause, source_freeze=frozen, counts=copy.deepcopy(counts),
                models={arm: model.state_dict() for arm, model in models.items()},
                optimizers={arm: opt.state_dict() for arm, opt in optimizers.items()},
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if args.device == 'cuda' else None)
            temporary = path.with_suffix('.tmp')
            torch.save(payload, temporary)
            temporary.replace(path)
            latest = dict(path=str(path.relative_to(ROOT)), sha256=digest(path), seed=args.seed,
                          qualification=args.qualify, counts=copy.deepcopy(counts), resource_pause=resource_pause)
            write(out / (path.stem + '.json'), latest)
            return latest

        for step in range(counts['common_steps'], limit):
            if available_kib() < 4 * 1024 * 1024:
                raise ResourcePressure('Host available memory below4GiB reserve')
            if schedule:
                schedule.check_memory()
            low, gt, identities = first if 8000 + step == prepared_step else pool.batch(args.seed, 8000 + step)
            low, gt = low.to(device), gt.to(device)
            if args.device == 'cuda':
                torch.cuda.synchronize()
            began = time.perf_counter()
            spatial, temporal = teacher_targets(teacher, low)
            statistics = {}
            for arm in ARMS:
                model, opt = models[arm], optimizers[arm]
                opt.zero_grad(set_to_none=True)
                prediction = model(low.flatten(0, 1)).unflatten(0, (4, 4))
                assert prediction.shape == gt.shape
                loss = objective(arm, prediction, gt, spatial, temporal)
                assert torch.isfinite(loss)
                loss.backward()
                active = list(model.parameters())
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active)
                norm = float(torch.nn.utils.clip_grad_norm_(active, 1))
                opt.step()
                assert all(torch.isfinite(p).all() for p in active)
                statistics[arm] = dict(loss=float(loss.detach()), gradient_norm=norm)
            assert all(p.grad is None for p in teacher.parameters())
            if args.device == 'cuda':
                torch.cuda.synchronize()
            work = time.perf_counter() - began
            counts['common_steps'] = step + 1
            counts['optimizer_updates'] += 3
            counts['teacher_images'] += 16
            counts['student_images'] += 48
            counts['DIS_pairs'] += 12
            append(out / 'training.jsonl', dict(at=stamp(), common_step=step+1,
                   data_step=8000+step, controls=statistics, synchronized_pipeline_work_seconds=work,
                   **{key: counts[key] for key in counts if key != 'common_steps'}))
            append(out / 'windows.jsonl', dict(common_step=step+1, data_step=8000+step, windows=identities))
            if (step + 1) % 1000 == 0:
                checkpoint()
            if (step + 1) % 100 == 0 or args.qualify:
                print(dict(common_step=step+1, seed=args.seed, **counts), flush=True)
            if schedule:
                if step + 1 < limit:
                    schedule.yield_after_work(work)
                else:
                    schedule.work_seconds += work
        assert counts == dict(common_steps=limit, optimizer_updates=3*limit,
                              teacher_images=16*limit, student_images=48*limit, DIS_pairs=12*limit)
        for arm, model in models.items():
            path = assets / f'{arm}-final.pt'
            torch.save(model.state_dict(), path)
            reloaded = torch.load(path, map_location='cpu', weights_only=True)
            assert all(torch.isfinite(value).all() for value in reloaded.values())
            write(out / f'{arm}-final.json', dict(at=stamp(), seed=args.seed, arm=arm,
                  path=str(path.relative_to(ROOT)), sha256=digest(path), parameters=4075,
                  updates=limit, qualification=args.qualify, unique_final=True, all_reloaded_finite=True))
        verify()
        if args.qualify:
            write(out / 'qualification.json', dict(at=stamp(), passed=True, device=args.device,
                  all_updates_discarded=True, same_original_initial_weights=initial_same,
                  original_parameters=4075, all_final_reloads_finite=True, **counts))
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
        write(out / 'runtime.json', dict(at=stamp(), pid=os.getpid(), seed=args.seed,
              qualification=args.qualify, device=args.device, completed_normally=complete,
              resource_pause=paused, counts=counts, latest_checkpoint=latest,
              seconds=time.perf_counter()-started, GPU=None if schedule is None else schedule.summary(),
              CUDA_initialized=torch.cuda.is_initialized(),
              max_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
              max_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0))
        if lock:
            lock.close()


if __name__ == '__main__':
    main(ARGS)
