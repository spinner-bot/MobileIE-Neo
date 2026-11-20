"""Post-training execution on GPU group4-7; original science remains frozen."""
from datetime import datetime
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PROCESS = BASE / 'process/v1'
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'research/4.temporal-consistency/scripts'))
from causal_reference_io_r75 import write, append, stamp, digest
from shared_gpu_admission import sustained_samples, assess


def read(path):
    return json.loads(path.read_text())


def identity(pid):
    try:
        value = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, start_ticks=int(value[19]), state=value[0])
    except FileNotFoundError:
        return dict(pid=pid, already_exited=True)


def resource_resume(attempts, previous):
    """Retain the upstream checkpoint when a resumed child did no new work."""
    assert previous['resource_pause'] and not previous['completed_normally']
    marker = previous['latest_checkpoint']
    forwarded = False
    if marker is None and any(previous['counts']['updates'].values()):
        config = read(attempts[-1] / 'config.json')
        assert config['seed'] == previous['seed']
        assert config['inherited_counts'] == previous['counts'], 'Uncheckpointed new work must not be discarded'
        assert previous['GPU'] is None and previous['CUDA_max_allocated_bytes'] == 0
        upstream = config['resume']
        assert upstream is not None, 'Nonzero inherited state needs an actual upstream checkpoint'
        upstream_path = Path(upstream['path']).resolve()
        for older in reversed(attempts[:-1]):
            candidate = read(older / 'runtime.json').get('latest_checkpoint')
            if (candidate is not None and (ROOT / candidate['path']).resolve() == upstream_path
                    and candidate['sha256'] == upstream['sha256']
                    and candidate['counts'] == previous['counts']):
                marker = candidate
                forwarded = True
                break
        assert marker is not None, 'Upstream checkpoint needs an original matching saved-count record'
    if marker is None:
        assert not any(previous['counts']['updates'].values())
        return None, None
    assert marker['resource_pause'] and marker['seed'] == previous['seed']
    assert marker['counts'] == previous['counts']
    path = ROOT / marker['path']
    assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    assert digest(path) == marker['sha256']
    return path, dict(checkpoint=marker, forwarded_without_new_update=forwarded,
                      previous_attempt=str(attempts[-1].relative_to(ROOT)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chain', type=int, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Coordinator itself is CPU only'
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    out = PROCESS / f'chain{args.chain:03d}'
    out.mkdir(parents=True, exist_ok=False)
    assets = ROOT / '.cache/experiments/topic8-v1' / out.name
    assert assets.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    assets.mkdir(parents=True, exist_ok=False)
    freezes = {name: read(PROCESS / name) for name in
               ('source-freeze001.json', 'perceptual-source-freeze001.json')}
    def verify():
        for frozen in freezes.values():
            assert frozen['source_sha256'] == {p: digest(ROOT / p) for p in frozen['source_sha256']}
        assert freezes['source-freeze001.json']['protocol_sha256'] == digest(BASE / 'protocol-v1.md')
    route_freeze = read(PROCESS / 'gpu-group-execution-freeze001.json')
    for source, expected in route_freeze['source_sha256'].items():
        assert digest(ROOT / source) == expected
    for seed in (0, 1):
        latest = sorted(PROCESS.glob(f'train-seed{seed}-attempt*/runtime.json'))[-1]
        trained = read(latest)
        assert trained['completed_normally']
        assert trained['counts']['updates'] == dict(warm=2000, spatial=4000, fixed=4000, learned=4000)
        assert read(BASE / f'results/v1-quality-seed{seed}/runtime.json')['completed_normally']
    verify()
    state = dict(at=stamp(), coordinator=identity(os.getpid()), status='starting',
                 stage='qualification', resource_polls=0, guard_attempts=0)
    write(out / 'config.json', dict(at=stamp(), coordinator=state['coordinator'],
          orchestrator_sha256=digest(Path(__file__)), science_freezes=freezes,
          revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          wall_clock_deadline=False, allowed_GPU_group=[4, 5, 6, 7], maximum_own_GPU_workers=1,
          science_failure_automatic_retry=False, resource_wait_seconds=30,
          asset_directory=str(assets.relative_to(ROOT))))
    started = time.perf_counter()
    complete = False
    cpu_workers = []

    def publish(**values):
        state.update(at=stamp(), **values)
        write(out / 'state.json', state)

    def launch(stage, command):
        verify()
        log = assets / f'{stage}.stdout.txt'
        with log.open('xb') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(),
                                     stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        entry = dict(at=stamp(), stage=stage, command=command, child=identity(child.pid),
                     stdout=str(log.relative_to(ROOT)))
        append(out / 'launches.jsonl', entry)
        publish(stage=stage, status='running', child=entry['child'], stdout=entry['stdout'])
        return child

    def wait(child, stage, guard=None):
        while True:
            try:
                code = child.wait(timeout=5)
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
            assessment = assess(samples)
            group = [row for row in assessment['candidates'] if row['index'] in (4, 5, 6, 7)]
            state['resource_polls'] += 1
            path = assets / f'resource-poll{state["resource_polls"]:05d}.json'
            write(path, dict(at=stamp(), samples=samples, group=group))
            append(out / 'resource-history.jsonl', dict(at=stamp(), stage=stage,
                   poll=state['resource_polls'], raw_samples=str(path.relative_to(ROOT)), group=group))
            eligible = sorted((row for row in group if row['allowed']),
                              key=lambda row: (row['mean_utilization_percent'], -row['minimum_free_mib']))
            if not eligible:
                publish(status='waiting_for_headroom', last_resource_poll=str(path.relative_to(ROOT)))
                time.sleep(30)
                continue
            state['guard_attempts'] += 1
            ordinal = state['guard_attempts']
            guard = assets / f'guard{ordinal:05d}.json'
            selected = eligible[0]
            wrapper = [str(ROOT / 'scripts/project-python'), str(ROOT / 'scripts/prebind_shared_gpu.py'),
                       '--gpu', str(selected['index']), '--record', str(guard), '--', *command]
            child = launch(f'{stage}-guard{ordinal:05d}', wrapper)
            code = wait(child, stage, guard)
            record = read(guard)
            if code == 75 and not record.get('child_pid'):
                publish(status='waiting_after_prechild_refusal', last_guard=str(guard.relative_to(ROOT)))
                time.sleep(30)
                continue
            return code, record

    python = str(ROOT / 'scripts/project-python')
    try:
        assert read(PROCESS / 'cpu-qualification001/qualification.json')['passed']
        cuda = PROCESS / 'cuda-qualification001'
        if not (cuda / 'qualification.json').exists():
            if cuda.exists():
                raise RuntimeError('An existing CUDA child needs its original terminal reviewed; do not repeat qualification')
            code, _ = gpu('cuda-qualification001', [python, str(BASE / 'qualify_v1.py'), '--device', 'cuda', '--attempt', '1'])
            if code != 0:
                raise RuntimeError(f'Original CUDA qualification child exit={code}; review original records, no automatic retry')
        assert read(cuda / 'qualification.json')['passed']
        assert read(cuda / 'runtime.json')['completed_normally']
        for seed in (0, 1):
            while True:
                attempts = sorted(PROCESS.glob(f'train-seed{seed}-attempt*'))
                resume = None
                if attempts:
                    terminal = attempts[-1] / 'runtime.json'
                    if not terminal.exists():
                        raise RuntimeError(f'Inspect the original training handle before restarting {attempts[-1]}')
                    previous = read(terminal)
                    if previous['completed_normally']:
                        break
                    if not previous['resource_pause']:
                        raise RuntimeError(f'Original training failure is not resource-only: {terminal}')
                    resume, resume_record = resource_resume(attempts, previous)
                    if resume_record is not None:
                        append(out / 'resource-continuations.jsonl', dict(at=stamp(), seed=seed, **resume_record))
                number = 1 if not attempts else int(attempts[-1].name.split('attempt')[1]) + 1
                command = [python, str(BASE / 'train_v1.py'), '--seed', str(seed), '--attempt', str(number)]
                if resume is not None:
                    command += ['--resume', str(resume)]
                code, _ = gpu(f'train-seed{seed}-attempt{number:03d}', command)
                if code not in (0, 75):
                    raise RuntimeError(f'Original seed{seed} training exit={code}; no automatic scientific retry')
            quality = BASE / f'results/v1-quality-seed{seed}'
            if quality.exists():
                assert read(quality / 'runtime.json')['completed_normally']
            else:
                child = launch(f'CPU-quality-seed{seed}', [python, str(BASE / 'evaluate_v1.py'), '--seed', str(seed)])
                cpu_workers.append((seed, child))
        for seed, child in cpu_workers:
            assert wait(child, f'CPU-quality-seed{seed}') == 0
        for seed in (0, 1):
            runtime = read(BASE / f'results/v1-quality-seed{seed}/runtime.json')
            assert runtime['completed_normally'] and runtime['quality_rows'] == 110
        while True:
            attempts = sorted((BASE / 'results').glob('v1-perceptual-attempt*'))
            number = 1
            resume = None
            if attempts:
                previous = read(attempts[-1] / 'runtime.json')
                if previous['completed_normally']:
                    break
                assert previous['resource_pause'], 'Do not automatically repeat a perceptual scientific failure'
                number = int(attempts[-1].name.split('attempt')[1]) + 1
                if previous['quality_rows']:
                    resume = attempts[-1]
            command = [python, str(BASE / 'perceptual_v1.py'), '--attempt', str(number)]
            if resume is not None:
                command += ['--resume-from', str(resume)]
            code, _ = gpu(f'perceptual-attempt{number:03d}', command)
            assert code in (0, 75), 'Original perceptual failure; no automatic scientific retry'
        for stage, script, terminal in (
            ('summary', 'summarize_v1.py', BASE / 'report/v1-joint/audit.json'),
            ('contacts', 'render_v1.py', BASE / 'report/v1-contacts/render.json')):
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
        # A CPU quality job already launched may still be valid. Never kill it
        # because a different training/metric stage stopped.
        write(out / 'runtime.json', dict(at=stamp(), coordinator=identity(os.getpid()),
              completed_normally=complete, seconds=time.perf_counter() - started,
              final_state=state, CPU_quality_children=[dict(seed=seed, **identity(child.pid),
                  observed_returncode=child.poll()) for seed, child in cpu_workers],
              paper_objective_achieved=False))


if __name__ == '__main__':
    main()
