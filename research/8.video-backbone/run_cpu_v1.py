"""Finish unchanged V1 budgets on CPU, then retain the original full DEV chain."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
from run_v1 import ROOT, BASE, PROCESS, read, identity, resource_resume, write, append, stamp, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chain', type=int, required=True)
    parser.add_argument('--post-chain', type=int, required=True)
    args = parser.parse_args()
    out = PROCESS / f'cpu-chain{args.chain:03d}'
    out.mkdir(parents=True, exist_ok=False)
    assets = ROOT / '.cache/experiments/topic8-v1' / out.name
    assert assets.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    assets.mkdir(parents=True, exist_ok=False)
    execution = read(PROCESS / 'cpu-execution-freeze001.json')
    frozen = read(PROCESS / 'source-freeze001.json')
    def verify():
        for source in (execution, frozen):
            assert source['source_sha256'] == {p: digest(ROOT / p) for p in source['source_sha256']}
    verify()
    state = dict(at=stamp(), coordinator=identity(os.getpid()), status='starting', stage='training')
    write(out / 'config.json', dict(at=stamp(), coordinator=state['coordinator'],
          CPU_execution_freeze=execution, canonical_science_freeze=frozen,
          post_chain=args.post_chain, Torch_threads_per_worker=2, OpenCV_threads_per_worker=1,
          training_nice=5, wall_clock_cutoff=False, automatic_scientific_retry=False,
          statement='Backend amendment only; same 28000 algorithm updates and final-only selection.'))
    started = time.perf_counter()
    complete = False
    quality_workers = []
    def publish(**values):
        state.update(at=stamp(), **values)
        write(out / 'state.json', state)
    def launch(stage, command):
        verify()
        path = assets / f'{stage}.stdout.txt'
        with path.open('xb') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
        append(out / 'launches.jsonl', dict(at=stamp(), stage=stage, child=identity(child.pid),
               command=command, stdout=str(path.relative_to(ROOT))))
        publish(stage=stage, status='running', child=identity(child.pid), stdout=str(path.relative_to(ROOT)))
        return child
    def wait(stage, child):
        while True:
            try:
                code = child.wait(timeout=5)
                append(out / 'exits.jsonl', dict(at=stamp(), stage=stage, pid=child.pid, returncode=code))
                publish(stage=stage, status='child_terminal', child=identity(child.pid), returncode=code)
                return code
            except subprocess.TimeoutExpired:
                publish(stage=stage, status='running', child=identity(child.pid))
    python = str(ROOT / 'scripts/project-python')
    try:
        assert read(PROCESS / 'cpu-qualification001/qualification.json')['passed']
        assert read(PROCESS / 'cuda-qualification001/qualification.json')['passed']
        for seed in (0, 1):
            while True:
                attempts = sorted(PROCESS.glob(f'train-seed{seed}-attempt*'))
                resume = None
                if attempts:
                    terminal = attempts[-1] / 'runtime.json'
                    assert terminal.exists(), 'Inspect original live worker; never duplicate an unfinished attempt'
                    previous = read(terminal)
                    if previous['completed_normally']:
                        break
                    assert previous['resource_pause'], 'No scientific failure automatic retry'
                    resume, record = resource_resume(attempts, previous)
                    if record is not None:
                        append(out / 'resource-continuations.jsonl', dict(at=stamp(), seed=seed, **record))
                number = 1 if not attempts else int(attempts[-1].name.split('attempt')[1]) + 1
                stage = f'CPU-train-seed{seed}-attempt{number:03d}'
                command = [python, str(BASE / 'train_cpu_v1.py'), '--seed', str(seed), '--attempt', str(number)]
                if resume is not None:
                    command += ['--resume', str(resume)]
                code = wait(stage, launch(stage, command))
                assert code in (0, 75), 'Original CPU worker failure requires review'
                if code == 75:
                    publish(status='waiting_for_host_memory')
                    time.sleep(30)
            quality = BASE / f'results/v1-quality-seed{seed}'
            if quality.exists():
                assert read(quality / 'runtime.json')['completed_normally']
            else:
                stage = f'CPU-quality-seed{seed}'
                child = launch(stage, ['/usr/bin/nice', '-n', '5', python,
                                       str(BASE / 'evaluate_v1.py'), '--seed', str(seed)])
                quality_workers.append((stage, child))
        for stage, child in quality_workers:
            assert wait(stage, child) == 0
        for seed in (0, 1):
            assert read(BASE / f'results/v1-quality-seed{seed}/runtime.json')['completed_normally']
        assert not (PROCESS / f'chain{args.post_chain:03d}').exists()
        stage = f'original-post-training-chain{args.post_chain:03d}'
        command = [python, str(BASE / 'run_v1.py'), '--chain', str(args.post_chain)]
        assert wait(stage, launch(stage, command)) == 0
        complete = True
        publish(stage='registered_development_chain', status='complete', actual_contact_review_pending=True)
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc())
        publish(status='stopped_for_original_terminal_review')
        raise
    finally:
        write(out / 'runtime.json', dict(at=stamp(), coordinator=identity(os.getpid()),
              completed_normally=complete, seconds=time.perf_counter() - started, final_state=state,
              CPU_quality_children=[dict(stage=stage, **identity(child.pid), returncode=child.poll())
                                    for stage, child in quality_workers],
              paper_objective_achieved=False))


if __name__ == '__main__':
    main()
