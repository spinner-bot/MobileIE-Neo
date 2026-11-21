"""Qualify I1-S, then run two independent exact-budget seeds on physical GPUs1/2."""
import argparse
import itertools
import os
from pathlib import Path
import subprocess
import time
import traceback

from simple_io import BASE, ROOT, append, digest, identity, owned_cache, read, stamp, verify_training, write


def main(seed):
    frozen = verify_training()
    label = 'simple-chain001' if seed is None else f'simple-seed{seed}-chain001'
    out, logs = BASE / 'process' / label, owned_cache(label)
    out.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=False)
    python = str(ROOT / 'scripts/project-python')
    began, complete = time.perf_counter(), False
    state = dict(at=stamp(), process=identity(), stage='initial', children={})

    def start(command, name):
        env = os.environ.copy()
        env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        with (logs / (name + '.log')).open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
        state['children'][name] = identity(child.pid)
        state.update(at=stamp(), stage=name)
        write(out / 'state.json', state)
        append(out / 'events.jsonl', dict(at=stamp(), action='start', label=name,
               child=state['children'][name], command=command))
        return child

    def wait(child, name):
        code = child.wait()
        state['children'][name]['exit_code'] = code
        state.update(at=stamp())
        write(out / 'state.json', state)
        append(out / 'events.jsonl', dict(at=stamp(), action='exit', label=name, code=code))
        return code

    def training(model_seed, qualification):
        prefix = 'simple-cuda-qualification' if qualification else f'simple-train-seed{model_seed}-attempt'
        gpu = 1 if qualification or model_seed == 0 else 2
        resume = None
        for attempt in itertools.count(1):
            name = f'{prefix}{attempt:03d}'
            command = [python, str(BASE / 'train_simple.py'), '--seed', str(model_seed), '--attempt', str(attempt)]
            if qualification:
                command.append('--qualify')
            if resume:
                command += ['--resume', str(resume)]
            wrapped = [python, str(ROOT / 'scripts/prebind_shared_gpu.py'), '--gpu', str(gpu),
                       '--record', str(out / (name + '-guard.json')), '--', *command]
            code = wait(start(wrapped, name), name)
            if code == 75:
                runtime = BASE / 'process' / name / 'runtime.json'
                if runtime.exists():
                    result = read(runtime)
                    assert result['resource_pause'] and not result['completed_normally']
                    marker = read(runtime.parent / 'resource-pause.json')['resume']
                    if marker is not None:
                        resume = ROOT / marker['path']
                        assert resume.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
                        assert digest(resume) == marker['sha256']
                state.update(at=stamp(), stage='resource_wait', pending=name, desired_gpu=gpu)
                write(out / 'state.json', state)
                time.sleep(60)
                continue
            if code != 0:
                raise RuntimeError(f'{name} exit{code}; non-resource failure is retained, not retried')
            runtime = read(BASE / 'process' / name / 'runtime.json')
            limit = 2 if qualification else 80000
            assert runtime['completed_normally'] and runtime['counts'] == dict(
                common_steps=limit, optimizer_updates=2*limit, student_images=16*limit)
            if qualification:
                proof = read(BASE / 'process' / name / 'qualification.json')
                assert proof['passed'] and proof['all_updates_discarded'] and proof['no_qualification_weights_exported']
                assert runtime['final_actual_device']['matched'] and proof['own_final_NVML_mib'] <= 4096
                assert runtime['maximum_allocated_bytes'] <= 2*1024**3
            return

    write(out / 'config.json', dict(at=stamp(), process=identity(), source_freeze=frozen,
          model_seed=seed, GPU_group=[0, 1, 2, 3], preferred_physical_GPU=dict(seed0=1, seed1=2),
          maximum_parallel_numeric_workers=2, formal_updates=320000 if seed is None else 160000,
          formal_student_images=2560000 if seed is None else 1280000,
          unique_final_only=True, wall_clock_cutoff=False, automatic_scientific_retry=False,
          evaluation_not_run_by_this_training_driver=True))
    try:
        if seed is None:
            training(0, True)
            children = {s: start([python, str(BASE / 'run_simple.py'), '--seed', str(s)], f'seed{s}-driver') for s in (0, 1)}
            codes = {s: wait(c, f'seed{s}-driver') for s, c in children.items()}
            assert codes == {0: 0, 1: 0}, f'Retained exits{codes}; do not kill a live sibling'
        else:
            training(seed, False)
        verify_training()
        state.update(at=stamp(), stage='training_completed_pending_evaluation')
        write(out / 'state.json', state)
        complete = True
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out / 'runtime.json', dict(at=stamp(), process=identity(), completed_normally=complete,
              seconds=time.perf_counter()-began, state=state, evaluation_completed=False, goal_complete=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(0, 1))
    main(parser.parse_args().seed)
