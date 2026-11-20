"""Single M1 spatial chain; capacity admission, qualification, two seeds, fixed evaluation."""
import os
import subprocess
import sys
import time
import traceback
from study_io import ROOT, BASE, PROCESS, read, write, append, digest, stamp, verify, owned_cache, identity
from shared_gpu_admission import sustained_samples, assess


def main():
    frozen = verify()
    out = PROCESS / 'spatial-chain001'
    out.mkdir(parents=True, exist_ok=False)
    logs = owned_cache('spatial-chain001')
    logs.mkdir(parents=True, exist_ok=False)
    python = str(ROOT / 'scripts/project-python')
    started, complete = time.perf_counter(), False
    state = dict(at=stamp(), process=identity(os.getpid()), source_freeze=frozen, stage='initial', child=None)

    def run_child(command, label, cpu=False):
        env = os.environ.copy()
        if cpu:
            env['CUDA_VISIBLE_DEVICES'] = ''
        with (logs / (label + '.log')).open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            state.update(at=stamp(), stage=label, child=identity(child.pid))
            write(out / 'state.json', state)
            append(out / 'events.jsonl', dict(at=stamp(), action='start', label=label, child=identity(child.pid), command=command))
            code = child.wait()
        append(out / 'events.jsonl', dict(at=stamp(), action='exit', label=label, code=code))
        return code

    def training(seed, qualification):
        prefix = 'cuda-spatial-qualification' if qualification else f'spatial-seed{seed}-attempt'
        resume = None
        for attempt in range(1, 1000000):
            label = f'{prefix}{attempt:03d}'
            samples = sustained_samples()
            assessed = assess(samples)
            eligible = sorted((r for r in assessed['candidates'] if r['allowed'] and r['index'] in (4, 5, 6, 7)),
                              key=lambda r: (r['mean_utilization_percent'], -r['minimum_free_mib']))
            write(out / (label + '-selection.json'), dict(at=stamp(), samples=samples, assessment=assessed,
                selected=None if not eligible else eligible[0], scope='Read-only headroom, not a reservation'))
            if not eligible:
                state.update(at=stamp(), stage='resource_wait', pending=label, child=None)
                write(out / 'state.json', state)
                time.sleep(60)
                continue
            command = [python, str(BASE / 'train_spatial.py'), '--seed', str(seed), '--attempt', str(attempt)]
            if qualification:
                command += ['--qualify']
            if resume:
                command += ['--resume', str(resume)]
            wrapper = [python, str(ROOT / 'scripts/prebind_shared_gpu.py'), '--gpu', str(eligible[0]['index']),
                '--record', str(out / (label + '-guard.json')), '--', *command]
            code = run_child(wrapper, label)
            if code == 75:
                runtime_path = PROCESS / label / 'runtime.json'
                if runtime_path.exists():
                    runtime = read(runtime_path)
                    assert runtime['resource_pause'] and not runtime['completed_normally']
                    marker = read(runtime_path.parent / 'resource-pause.json')['resume']
                    if marker is not None:
                        resume = ROOT / marker['path']
                        assert resume.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo') and digest(resume) == marker['sha256']
                time.sleep(60)
                continue
            if code != 0:
                raise RuntimeError(f'{label} exited {code}; scientific/implementation failure is not retried')
            runtime = read(PROCESS / label / 'runtime.json')
            expected = 2 if qualification else 8000
            assert runtime['completed_normally'] and runtime['counts'] == dict(common_steps=expected,
                optimizer_updates=2*expected, student_images=32*expected)
            if qualification:
                proof = read(PROCESS / label / 'qualification.json')
                assert proof['passed'] and proof['all_updates_discarded']
                assert runtime['max_reserved_bytes'] <= 2048 * 1024 * 1024
            return
        raise RuntimeError('Resource wait counter exhausted; not a science wall-clock limit')

    write(out / 'config.json', dict(at=stamp(), process=identity(os.getpid()), source_freeze=frozen,
        formal_optimizer_updates=32000, GPU_group=[4, 5, 6, 7], wall_clock_cutoff=False,
        scientific_failure_automatic_retry=False, no_video_training=True))
    try:
        training(0, True)
        for seed in (0, 1):
            training(seed, False)
            code = run_child([python, str(BASE / 'evaluate_spatial.py'), '--seed', str(seed)], f'held-spatial-seed{seed}', cpu=True)
            if code != 0:
                raise RuntimeError(f'Spatial quality seed{seed} exited {code}; do not duplicate inference')
        code = run_child([python, str(BASE / 'summarize_spatial.py')], 'summarize-spatial', cpu=True)
        assert code == 0
        state.update(at=stamp(), stage='completed', child=None)
        write(out / 'state.json', state)
        complete = True
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out / 'runtime.json', dict(at=stamp(), process=identity(os.getpid()), completed_normally=complete,
            seconds=time.perf_counter()-started, state=state))


if __name__ == '__main__':
    main()
