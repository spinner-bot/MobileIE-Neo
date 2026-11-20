"""One registered T2 chain, resource-only continuation, no scientific retries."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time
import traceback

from study_io import ROOT, BASE, PROCESS, read, write, append, stamp, digest, verify, owned_cache, identity
from shared_gpu_admission import sustained_samples, assess


def resource_resume(previous, seed, qualification):
    assert previous['resource_pause'] and not previous['completed_normally']
    marker = previous['latest_checkpoint']
    if marker is None:
        assert not any(previous['counts'].values()), 'Never discard uncheckpointed updates'
        return None
    assert marker['resource_pause'] and marker['counts'] == previous['counts']
    assert marker['seed'] == seed and marker['qualification'] == qualification
    path = ROOT / marker['path']
    assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    assert digest(path) == marker['sha256']
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chain', type=int, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    chain_lock = owned_cache('coordinator.lock').open('a')
    fcntl.flock(chain_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen = verify()
    out = PROCESS / f'chain{args.chain:03d}'
    out.mkdir(parents=True, exist_ok=False)
    assets = owned_cache(out.name)
    assets.mkdir(parents=True, exist_ok=False)
    state = dict(at=stamp(), coordinator=identity(os.getpid()), status='starting',
                 stage='qualification', resource_polls=0, guard_attempts=0)
    write(out / 'config.json', dict(at=stamp(), coordinator=state['coordinator'], source_freeze=frozen,
          revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          wall_clock_deadline=False, allowed_GPU_group=[4, 5, 6, 7], maximum_own_GPU_workers=1,
          scientific_failure_automatic_retry=False, asset_directory=str(assets.relative_to(ROOT))))
    started, complete = time.perf_counter(), False
    cpu_workers = []

    def publish(**values):
        state.update(at=stamp(), **values)
        write(out / 'state.json', state)

    def launch(stage, command):
        verify()
        log = assets / f'{stage}.stdout.txt'
        with log.open('xb') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
        entry = dict(at=stamp(), stage=stage, command=command, child=identity(child.pid),
                     stdout=str(log.relative_to(ROOT)))
        append(out / 'launches.jsonl', entry)
        publish(stage=stage, status='running', child=entry['child'], stdout=entry['stdout'])
        return child

    def wait(child, stage, guard=None):
        while True:
            try:
                code = child.wait(timeout=30)
                append(out / 'exits.jsonl', dict(at=stamp(), stage=stage, pid=child.pid, returncode=code))
                publish(stage=stage, status='child_terminal', child=identity(child.pid), returncode=code)
                return code
            except subprocess.TimeoutExpired:
                details = dict(stage=stage, status='running', child=identity(child.pid))
                if guard is not None and guard.exists():
                    record = read(guard)
                    if record.get('child_pid'):
                        details['worker'] = identity(record['child_pid'])
                publish(**details)

    def gpu(stage, command):
        while True:
            verify()
            publish(stage=stage, status='sampling_resources', child=None, worker=None)
            samples = sustained_samples()
            group = [r for r in assess(samples)['candidates'] if r['index'] in (4, 5, 6, 7)]
            state['resource_polls'] += 1
            path = assets / f'resource-poll{state["resource_polls"]:05d}.json'
            write(path, dict(at=stamp(), samples=samples, group=group))
            append(out / 'resource-history.jsonl', dict(at=stamp(), stage=stage,
                   poll=state['resource_polls'], raw_samples=str(path.relative_to(ROOT)), group=group))
            eligible = sorted((r for r in group if r['allowed']),
                              key=lambda r: (r['mean_utilization_percent'], -r['minimum_free_mib']))
            if not eligible:
                publish(status='waiting_for_headroom', last_resource_poll=str(path.relative_to(ROOT)))
                time.sleep(60)
                continue
            state['guard_attempts'] += 1
            ordinal = state['guard_attempts']
            guard = assets / f'guard{ordinal:05d}.json'
            wrapper = [python, str(ROOT / 'scripts/prebind_shared_gpu.py'),
                       '--gpu', str(eligible[0]['index']), '--record', str(guard), '--', *command]
            child = launch(f'{stage}-guard{ordinal:05d}', wrapper)
            code = wait(child, stage, guard)
            record = read(guard)
            if code == 75 and not record.get('child_pid'):
                publish(status='waiting_after_prechild_refusal', last_guard=str(guard.relative_to(ROOT)))
                time.sleep(60)
                continue
            return code

    python = str(ROOT / 'scripts/project-python')

    def training(seed, backend, qualification):
        prefix = f'{backend}-qualification' if qualification else f'train-seed{seed}-attempt'
        while True:
            attempts = sorted(PROCESS.glob(prefix + '*'))
            resume = None
            if attempts:
                terminal = attempts[-1] / 'runtime.json'
                assert terminal.exists(), f'Inspect original child, never duplicate {attempts[-1]}'
                previous = read(terminal)
                assert read(attempts[-1] / 'config.json')['source_freeze'] == frozen
                assert (previous['seed'], previous['device'], previous['qualification']) == (seed, backend, qualification)
                if previous['completed_normally']:
                    limit = 2 if qualification else 4000
                    assert previous['counts'] == dict(common_steps=limit, optimizer_updates=2*limit,
                          student_images=32*limit, DIS_pairs=12*limit)
                    if qualification:
                        proof = read(attempts[-1] / 'qualification.json')
                        assert proof['passed'] and proof['all_updates_discarded']
                    return
                resume = resource_resume(previous, seed, qualification)
                append(out / 'resource-continuations.jsonl', dict(at=stamp(), seed=seed,
                       qualification=qualification, previous_attempt=str(attempts[-1].relative_to(ROOT)),
                       previous_counts=previous['counts'], resume=None if resume is None else str(resume)))
            number = 1 if not attempts else int(attempts[-1].name[len(prefix):]) + 1
            stage = f'{prefix}{number:03d}'
            command = [python, str(BASE / 'train.py'), '--seed', str(seed), '--device', backend,
                       '--attempt', str(number)]
            if qualification:
                command += ['--qualify']
            if resume is not None:
                command += ['--resume', str(resume)]
            code = gpu(stage, command) if backend == 'cuda' else wait(launch(stage, command), stage)
            assert code in (0, 75), 'Original scientific failure; inspect original records before any new run'

    try:
        training(0, 'cpu', True)
        training(0, 'cuda', True)
        for seed in (0, 1):
            training(seed, 'cuda', False)
            quality = BASE / f'results/quality-seed{seed}'
            if quality.exists():
                assert read(quality / 'runtime.json')['completed_normally']
            else:
                child = launch(f'CPU-quality-seed{seed}', [python, str(BASE / 'evaluate.py'), '--seed', str(seed)])
                cpu_workers.append((seed, child))
        for seed, child in cpu_workers:
            assert wait(child, f'CPU-quality-seed{seed}') == 0
        for seed in (0, 1):
            runtime = read(BASE / f'results/quality-seed{seed}/runtime.json')
            assert runtime['completed_normally'] and (runtime['quality_rows'], runtime['frame_rows']) == (60, 2688)
        while True:
            attempts = sorted((BASE / 'results').glob('perceptual-attempt*'))
            number, resume = 1, None
            if attempts:
                previous = read(attempts[-1] / 'runtime.json')
                if previous['completed_normally']:
                    assert (previous['quality_rows'], previous['comparison_frames']) == (120, 5376)
                    break
                assert previous['resource_pause'], 'Do not repeat a scientific perceptual failure'
                number = int(attempts[-1].name.split('attempt')[1]) + 1
                resume = attempts[-1]
            command = [python, str(BASE / 'perceptual.py'), '--attempt', str(number)]
            if resume is not None:
                command += ['--resume-from', str(resume)]
            assert gpu(f'perceptual-attempt{number:03d}', command) in (0, 75)
        for stage, script, terminal in (
                ('summary', 'summarize.py', BASE / 'report/joint/audit.json'),
                ('contacts', 'render.py', BASE / 'report/contacts/render.json')):
            if not terminal.exists():
                assert wait(launch(stage, [python, str(BASE / script)]), stage) == 0
            assert read(terminal)['completed']
        verify()
        complete = True
        publish(stage='registered_development_chain', status='complete',
                actual_contact_review_pending=True, paper_objective_achieved=False)
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc())
        publish(status='stopped_for_original_terminal_review')
        raise
    finally:
        write(out / 'runtime.json', dict(at=stamp(), coordinator=identity(os.getpid()),
              completed_normally=complete, seconds=time.perf_counter()-started, final_state=state,
              CPU_quality_children=[dict(seed=seed, **identity(child.pid), observed_returncode=child.poll())
                                    for seed, child in cpu_workers], paper_objective_achieved=False))
        chain_lock.close()


if __name__ == '__main__':
    main()
