"""O1 execution amendment001 chain: fresh resource admission, qualification, two seeds, full held screen."""
import os
import sys
import time
import subprocess
import itertools
import traceback
from pathlib import Path
from study_io import BASE, ROOT, read, write, append, digest, stamp, identity, verify, owned_cache
sys.path.insert(0, str(ROOT / 'scripts'))
from shared_gpu_admission import sustained_samples, assess


def child_identity(pid):
    value = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=int(value[19]), state=value[0])


def main():
    frozen = verify('training-source-freeze001.json')
    execution = verify('cuda-execution-freeze001.json')
    out = BASE / 'process/chain002'; out.mkdir(parents=True, exist_ok=False)
    logs = owned_cache('chain002'); logs.mkdir(parents=True, exist_ok=False)
    python = str(ROOT / 'scripts/project-python')
    began, complete = time.perf_counter(), False
    state = dict(at=stamp(), process=identity(), source_freeze=frozen, stage='initial', child=None)

    def run_child(command, label, cpu=False):
        env = os.environ.copy()
        if cpu: env['CUDA_VISIBLE_DEVICES'] = ''
        with (logs / (label+'.log')).open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            observed = child_identity(child.pid)
            state.update(at=stamp(), stage=label, child=observed)
            write(out / 'state.json', state)
            append(out / 'events.jsonl', dict(at=stamp(), action='start', label=label, child=observed, command=command))
            code = child.wait()
        append(out / 'events.jsonl', dict(at=stamp(), action='exit', label=label, code=code))
        return code

    def training(seed, qualification):
        prefix = 'cuda-qualification' if qualification else f'train-seed{seed}-attempt'
        resume = None
        for attempt in itertools.count(2 if qualification else 1):
            label = f'{prefix}{attempt:03d}'
            samples = sustained_samples(); assessment = assess(samples)
            eligible = sorted((r for r in assessment['candidates'] if r['allowed'] and r['index'] in (4, 5, 6, 7)),
                              key=lambda r: (r['mean_utilization_percent'], -r['minimum_free_mib']))
            write(out / (label+'-selection.json'), dict(at=stamp(), samples=samples, assessment=assessment,
                selected=None if not eligible else eligible[0], scope='Read-only headroom selection, not reservation'))
            if not eligible:
                state.update(at=stamp(), stage='resource_wait', pending=label, child=None)
                write(out / 'state.json', state); time.sleep(60); continue
            command = [python, str(BASE / 'cuda_environment001.py'), '--', python, str(BASE / 'train.py'), '--seed', str(seed), '--attempt', str(attempt)]
            if qualification: command += ['--qualify']
            if resume: command += ['--resume', str(resume)]
            wrapped = [python, str(ROOT / 'scripts/prebind_shared_gpu.py'), '--gpu', str(eligible[0]['index']),
                '--record', str(out / (label+'-guard.json')), '--', *command]
            code = run_child(wrapped, label)
            if code == 75:
                runtime = BASE / 'process' / label / 'runtime.json'
                if runtime.exists():
                    result = read(runtime); assert result['resource_pause'] and not result['completed_normally']
                    marker = read(runtime.parent / 'resource-pause.json')['resume']
                    if marker is not None:
                        resume = ROOT / marker['path']
                        assert resume.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo') and digest(resume) == marker['sha256']
                time.sleep(60); continue
            if code != 0: raise RuntimeError(f'{label} exit {code}; scientific failure is not retried')
            result = read(BASE / 'process' / label / 'runtime.json')
            limit = 2 if qualification else 4000
            assert result['completed_normally'] and result['counts'] == dict(common_steps=limit,
                optimizer_updates=2*limit, student_images=32*limit, teacher_images=16*limit, teacher_FB_pairs=12*limit)
            if qualification:
                proof = read(BASE / 'process' / label / 'qualification.json')
                assert proof['passed'] and proof['all_updates_discarded'] and result['final_actual_device']['matched']
                assert result['own_final_NVML_mib'] <= 4096 and result['maximum_allocated_bytes'] <= 2*1024**3
            return

    write(out / 'config.json', dict(at=stamp(), process=identity(), source_freeze=frozen, execution_freeze=execution, GPU_group=[4, 5, 6, 7],
        formal_optimizer_updates=16000, formal_student_images=256000, new_DEV=False, independent_final=False,
        wall_clock_cutoff=False, automatic_scientific_retry=False))
    try:
        training(0, True)
        for seed in (0, 1):
            training(seed, False)
            code = run_child([python, str(BASE / 'evaluate.py'), '--seed', str(seed)], f'held-seed{seed}', cpu=True)
            if code != 0: raise RuntimeError(f'held seed{seed} exit {code}; do not duplicate completed inference')
        code = run_child([python, str(BASE / 'summarize.py')], 'summarize', cpu=True)
        assert code == 0
        state.update(at=stamp(), stage='completed', child=None); write(out / 'state.json', state)
        complete = True
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out / 'runtime.json', dict(at=stamp(), process=identity(), completed_normally=complete,
            seconds=time.perf_counter()-began, state=state))


if __name__ == '__main__':
    main()
