"""Fixed three-arm R79 budget; recover resources only at complete joint updates."""
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
from noise_role_io_r79 import BASE, PROCESS, digest, write, append, stamp, verify, acquire_gpu, owned_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(0, 1), required=True)
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    assert args.attempt > 0
    bound = require_initial_binding()
    root = BASE/f'results/exploration-r79-stage001-seed{args.seed}'
    seed_lock = (ROOT/f'.cache/noise-role-r79-seed{args.seed}.lock').open('a')
    fcntl.flock(seed_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (root/'final.json').exists()
    out = root/f'attempt{args.attempt:03d}'
    out.mkdir(parents=True, exist_ok=False)
    import cv2
    import torch
    from noise_role_data_r79 import NoiseRolePool, gain
    from noise_role_model_r79 import MODES, STEPS, models, sequence, update, optimizer_for, precision
    cv2.setNumThreads(1); precision()
    began = time.perf_counter()
    complete, steps, inherited, reason = False, 0, 0, 'error'
    gpu_lock, schedule, optimizers, cache, latest = None, None, None, None, None
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    frozen_sha = digest(PROCESS/'source-freeze001.json')

    def save(kind):
        nonlocal latest
        path = cache/f'{kind}-step{steps:04d}.pt'
        assert not path.exists()
        saved = dict(experiment='R79', seed=args.seed, completed_steps=steps, next_data_step=steps,
            registered_steps=STEPS, freeze_sha256=frozen_sha,
            models={name: net.state_dict() for name, net in nets.items()},
            optimizers={name: opt.state_dict() for name, opt in optimizers.items()},
            cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
        temporary = path.with_suffix('.pt.tmp')
        torch.save(saved, temporary); os.replace(temporary, path)
        latest = dict(at=stamp(), seed=args.seed, attempt=args.attempt, completed_steps=steps,
            next_data_step=steps, registered_steps=STEPS, kind=kind, path=str(path.relative_to(ROOT)),
            sha256=digest(path), bytes=path.stat().st_size, freeze_sha256=frozen_sha,
            parameters_each=sum(p.numel() for p in nets['spatial'].parameters()),
            completed_optimizer_updates=steps*3, final_quality_eligible=False)
        write(out/(path.stem+'.json'), latest)
        return latest

    try:
        frozen = verify()
        gates = {}
        for device in ('cpu', 'cuda'):
            path = PROCESS/f'{device}-gate001/gate.json'
            gate = json.loads(path.read_text())
            assert gate['passed'] and gate['discarded_updates'] == 6 and gate['formal_updates'] == 0
            assert gate['frozen_sources_sha256'] == frozen_sha
            gates[device] = digest(path)
        saved = None
        if args.resume:
            marker_path = args.resume.resolve()
            assert marker_path.is_relative_to(root) and marker_path.parent != out
            marker = json.loads(marker_path.read_text())
            prior = json.loads((marker_path.parent/'runtime.json').read_text())
            assert prior['stop_reason'] == 'resource' and not prior['completed_normally']
            assert marker['freeze_sha256'] == frozen_sha and marker['seed'] == args.seed
            assert marker['kind'] == 'resource' and prior['completed_steps'] == marker['completed_steps']
            checkpoint = ROOT/marker['path']
            assert checkpoint.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
            assert digest(checkpoint) == marker['sha256']
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            assert saved['experiment'] == 'R79' and saved['seed'] == args.seed
            assert saved['freeze_sha256'] == frozen_sha and saved['registered_steps'] == STEPS
            inherited = steps = saved['completed_steps']
            assert steps == saved['next_data_step'] == marker['completed_steps'] and 0 <= steps < STEPS
        elif args.attempt != 1:
            prior = [json.loads(p.read_text()) for p in root.glob('attempt*/runtime.json')]
            assert prior and all(r['stop_reason'] == 'resource' and r['completed_steps'] == 0 for r in prior)
        pool = NoiseRolePool(); first = pool.batch(args.seed, steps)
        nets = models(args.seed)
        assert not torch.cuda.is_initialized()
        write(out/'config.json', dict(at=stamp(), pid=os.getpid(), seed=args.seed, attempt=args.attempt,
            git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            frozen_sources=frozen, freeze_sha256=frozen_sha, gates=gates, data=pool.info,
            inherited_steps=inherited, registered_steps_each=STEPS, models=MODES, initial_seed=7900+args.seed,
            first_window=first[4], batch=4, frames=4, crop=128, all_frames_unclamped_Charbonnier=True,
            lr=1e-3, optimizer='Adam', gradient_clip=1, TF32=False,
            schedule='existing 25% cooperative work duty; no scientific wall-clock cutoff',
            scope='Known curriculum learning-problem control, not a new architecture/method or handset result.'))
        gpu_lock, schedule, selected = acquire_gpu(bound, out)
        nets = {name: net.cuda().train() for name, net in nets.items()}
        optimizers = {name: optimizer_for(net) for name, net in nets.items()}
        if saved:
            for name, net in nets.items():
                net.load_state_dict(saved['models'][name], strict=True)
                optimizers[name].load_state_dict(saved['optimizers'][name])
            torch.set_rng_state(saved['cpu_rng']); torch.cuda.set_rng_state(saved['cuda_rng'])
            del saved
        cache = owned_cache(f'checkpoints/exploration-r79/seed{args.seed}/attempt{args.attempt:03d}')
        torch.cuda.reset_peak_memory_stats()
        for step in range(steps, STEPS):
            schedule.check_memory()
            if stop[0]:
                raise ResourcePressure('Requested cooperative stop at completed joint-update boundary.')
            ordinary, cur, past, target, identities = first if step == inherited else pool.batch(args.seed, step)
            ordinary, cur, past, target = [value.cuda() for value in (ordinary, cur, past, target)]
            started = time.perf_counter(); rows = []
            for name, net in nets.items():
                row = update(net, optimizers[name], ordinary, target,
                    cur if name == 'curriculum' else None, past if name == 'curriculum' else None)
                rows.append(dict(model=name, **row))
            torch.cuda.synchronize()
            work_seconds = time.perf_counter()-started
            steps = step+1
            if steps == 1 or steps % 1000 == 0:
                save('regular')
                append(out/'checkpoint-windows.jsonl', dict(completed_steps=steps, identities=identities))
            if steps == inherited+1 or steps % 100 == 0:
                row = dict(at=stamp(), completed_steps=steps, gain=gain(step), models=rows,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(), resource=schedule.summary())
                append(out/'training.jsonl', row)
                write(out/'progress.json', dict(**row, seed=args.seed, pid=os.getpid(), attempt=args.attempt,
                    registered_steps=STEPS, completed_normally=False, latest_checkpoint=latest))
                print(json.dumps({k: v for k, v in row.items() if k != 'resource'}), flush=True)
            if steps < STEPS:
                schedule.yield_after_work(work_seconds)
            else:
                schedule.work_seconds += work_seconds
        # Same-device reload of the final joint checkpoint on the fixed training window, no DEV selection.
        saved = torch.load(ROOT/latest['path'], map_location='cpu', weights_only=True)
        errors = {}
        for name, net in nets.items():
            clone = models(args.seed)[name].cuda().eval()
            clone.load_state_dict(saved['models'][name], strict=True)
            with torch.no_grad():
                errors[name] = float((sequence(net.eval(), ordinary)-sequence(clone, ordinary)).abs().max())
            del clone
        assert max(errors.values()) <= 1e-6
        write(out/'final-reload.json', dict(passed=True, errors=errors, new_DEV_pixels=0))
        latest = dict(latest, final_quality_eligible=True, final_reload=errors)
        write(out/'regular-step4000.json', latest)
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
            new_joint_steps=steps-inherited, new_optimizer_updates=(steps-inherited)*3, registered_steps=4000,
            latest_checkpoint=latest, seconds=time.perf_counter()-began,
            resource=schedule.summary() if schedule else None,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None))
        if complete:
            write(root/'final.json', dict(**latest, completed_normally=True,
                runtime=str((out/'runtime.json').relative_to(ROOT))))
        globals()['_held_seed_lock'], globals()['_held_gpu_lock'] = seed_lock, gpu_lock
    if not complete:
        raise SystemExit(75)


if __name__ == '__main__':
    main()
