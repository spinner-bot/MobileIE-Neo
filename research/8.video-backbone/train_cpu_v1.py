"""CPU continuation of frozen V1 math, windows and complete algorithm budgets."""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
sys.path.insert(0, str(ROOT / 'scripts'))

import torch
from runtime_io import (PROCESS, owned_cache, precision, verify, write, append, stamp, digest,
                        MixedPool, fresh_core, make_controls, load_spatial)
from shared_training_schedule import ResourcePressure

PHASES = ('warm', 'spatial', 'fixed', 'learned')
BUDGET = dict(warm=2000, spatial=4000, fixed=4000, learned=4000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', choices=(0, 1), type=int, required=True)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    precision()
    os.nice(5)
    frozen = verify()
    execution = __import__('json').loads((PROCESS / 'cpu-execution-freeze001.json').read_text())
    assert execution['source_sha256'] == {p: digest(ROOT / p) for p in execution['source_sha256']}
    def host_headroom():
        available = next(int(row.split()[1]) for row in Path('/proc/meminfo').read_text().splitlines() if row.startswith('MemAvailable:'))
        if available < 4 * 1024 * 1024:
            raise ResourcePressure('CPU host available memory below4GiB reserve')
        return available
    assert host_headroom() >= 8 * 1024 * 1024, 'Prepare the approximately3GiB owned pool only with host memory headroom'
    for device in ('cpu', 'cuda'):
        assert __import__('json').loads((PROCESS / f'{device}-qualification001/qualification.json').read_text())['passed']
    out = PROCESS / f'train-seed{args.seed}-attempt{args.attempt:03d}'
    out.mkdir(parents=True, exist_ok=False)
    cache = owned_cache(out.name)
    cache.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    complete = paused = False
    lock = schedule = optimizer = None
    phase = 'warm'
    counts = dict(updates={p: 0 for p in PHASES}, teacher_images=0, backbone_images=0,
                  DIS_pairs=0, gate_images=0)
    latest = None
    try:
        pool = MixedPool()
        low0, gt0, ids0 = pool.batch(args.seed, 0)
        teacher, provenance = load_spatial(args.seed)
        teacher.eval().requires_grad_(False)
        core = fresh_core(args.seed)
        models = make_controls(core)
        inherited = None
        if args.resume:
            inherited = torch.load(args.resume, map_location='cpu', weights_only=False)
            assert inherited['resource_pause'] and inherited['seed'] == args.seed
            assert inherited['source_freeze'] == frozen
            if 'CPU_execution_freeze' in inherited:
                assert inherited['CPU_execution_freeze'] == execution
            core.load_state_dict(inherited['core'], strict=True)
            for name, model in models.items():
                model.load_state_dict(inherited['models'][name], strict=True)
            counts = inherited['counts']
            assert all(0 <= counts['updates'][p] <= BUDGET[p] for p in PHASES)
            phase = inherited['phase']
        write(out / 'config.json', dict(at=stamp(), pid=os.getpid(), seed=args.seed,
              attempt=args.attempt, source_freeze=frozen, data=pool.info, teacher=provenance,
              first_window=ids0, budgets=BUDGET, data_steps=dict(warm=[0, 1999], adaptation=[2000, 5999]),
              resume=None if args.resume is None else dict(path=str(args.resume), sha256=digest(args.resume)),
              inherited_counts=counts, optimizer='Fresh Adam for each phase; resource resume retains exact state',
              precision='CPU FP32 / TF32 off', CPU_execution_freeze=execution,
              training_backend='cpu', Torch_threads=torch.get_num_threads(), OpenBLAS_threads=2,
              nice_priority=os.getpriority(os.PRIO_PROCESS, 0),
              host_available_kib=host_headroom(), unique_final=True, wall_clock_cutoff=False))

        def checkpoint(resource_pause=False):
            nonlocal latest
            path = cache / f'{phase}-step{counts["updates"][phase]:04d}{"-resource" if resource_pause else ""}.pt'
            state = dict(seed=args.seed, phase=phase, counts=counts,
                         source_freeze=frozen, resource_pause=resource_pause,
                         core=core.state_dict(), models={k: v.state_dict() for k, v in models.items()},
                         optimizer=None if optimizer is None else optimizer.state_dict(),
                         torch_rng=torch.get_rng_state(), cuda_rng=None,
                         training_backend='cpu', CPU_execution_freeze=execution)
            temporary = path.with_suffix('.tmp')
            torch.save(state, temporary)
            temporary.replace(path)
            latest = dict(path=str(path.relative_to(ROOT)), sha256=digest(path), seed=args.seed,
                          phase=phase, counts=__import__('copy').deepcopy(counts), resource_pause=resource_pause)
            write(out / f'{phase}-step{counts["updates"][phase]:04d}{"-resource" if resource_pause else ""}.json', latest)
            return latest

        for phase in PHASES:
            if inherited is not None and PHASES.index(phase) < PHASES.index(inherited['phase']):
                # Earlier controls are already final. In particular do not copy
                # the warm weights over adapted controls on resource resume.
                continue
            model = core if phase == 'warm' else models[phase]
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3 if phase == 'warm' else 1e-4,
                                         betas=(.9, .999), eps=1e-8, weight_decay=0)
            if inherited is not None and phase == inherited['phase']:
                optimizer.load_state_dict(inherited['optimizer'])
                torch.set_rng_state(inherited['torch_rng'])
                # CPU numerical operators have no CUDA RNG; window sampling is step-seeded.
            for step in range(counts['updates'][phase], BUDGET[phase]):
                host_headroom()
                data_step = step if phase == 'warm' else 2000 + step
                low, gt, identities = (low0, gt0, ids0) if data_step == 0 else pool.batch(args.seed, data_step)
                began = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                if phase == 'warm':
                    with torch.no_grad():
                        target = teacher(low.flatten(0, 1)).clamp(0, 1)
                    output = core(low.flatten(0, 1))
                    loss = (output - target).square().mean()
                    image_count, pair_count = 16, 0
                else:
                    output, preview, _, details = model.window(low)
                    loss = (output - gt).square().mean()
                    if phase != 'spatial':
                        loss = .5 * loss + .5 * (preview - gt).square().mean()
                    image_count = sum(v['backbone_images'] for v in details)
                    pair_count = sum(v['DIS_pairs'] for v in details)
                assert torch.isfinite(loss), f'{phase}/{step}: nonfinite objective'
                loss.backward()
                active = list(model.parameters())
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active)
                norm = float(torch.nn.utils.clip_grad_norm_(active, 1))
                optimizer.step()
                seconds = time.perf_counter() - began
                counts['updates'][phase] = step + 1
                counts['teacher_images'] += 16 if phase == 'warm' else 0
                counts['backbone_images'] += image_count
                counts['DIS_pairs'] += pair_count
                counts['gate_images'] += 12 if phase == 'learned' else 0
                append(out / 'training.jsonl', dict(phase=phase, step=step + 1, data_step=data_step,
                       loss=float(loss.detach()), gradient_norm=norm, synchronized_work_seconds=seconds,
                       backbone_images=image_count, DIS_pairs=pair_count))
                append(out / 'windows.jsonl', dict(phase=phase, step=step + 1, data_step=data_step, windows=identities))
                if (step + 1) % 1000 == 0:
                    checkpoint()
                    print(f'{phase} {step + 1}/{BUDGET[phase]} seed{args.seed}', flush=True)
            if phase == 'warm':
                for target in models.values():
                    target.backbone.load_state_dict(core.state_dict(), strict=True)
            final = cache / f'{phase}-final.pt'
            torch.save(model.state_dict(), final)
            saved = torch.load(final, map_location='cpu', weights_only=True)
            assert all(torch.isfinite(v).all() for v in saved.values())
            write(out / f'{phase}-final.json', dict(at=stamp(), seed=args.seed, phase=phase,
                  path=str(final.relative_to(ROOT)), sha256=digest(final),
                  parameters=sum(p.numel() for p in model.parameters()), updates=BUDGET[phase],
                  all_tensors_reloaded_finite=True, training_backend='cpu', unique_final=True))
        assert counts == dict(updates=BUDGET, teacher_images=32000, backbone_images=320000,
                              DIS_pairs=96000, gate_images=48000)
        verify()
        assert execution['source_sha256'] == {p: digest(ROOT / p) for p in execution['source_sha256']}
        complete = True
    except ResourcePressure as error:
        paused = True
        if optimizer is not None:
            latest = checkpoint(resource_pause=True)
        write(out / 'resource-pause.json', dict(at=stamp(), reason=str(error), resume=latest))
        raise SystemExit(75)
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out / 'runtime.json', dict(at=stamp(), pid=os.getpid(), seed=args.seed,
              completed_normally=complete, resource_pause=paused, counts=counts, latest_checkpoint=latest,
              seconds=time.perf_counter() - started, GPU=None, training_backend='cpu',
              CUDA_initialized=torch.cuda.is_initialized(),
              CPU_execution_freeze=execution, CUDA_max_allocated_bytes=0, CUDA_max_reserved_bytes=0))
        if lock:
            lock.close()


if __name__ == '__main__':
    main()
