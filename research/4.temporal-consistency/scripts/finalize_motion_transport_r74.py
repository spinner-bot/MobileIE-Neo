"""Frozen R74 qualification, sequential guarded training and complete evidence chain."""
import argparse
import ast
from datetime import datetime
import json
import subprocess
import time
import traceback

from motion_transport_r74 import BASE, ROOT, PROTOCOL, MODES
from check_motion_transport_r74 import write
from dynamic_training_r34 import digest
from finalize_preserved_r37 import process_identity
from gpu_guard import sample


def stamp(): return datetime.now().astimezone().isoformat()


def scientific_sources():
    """Freeze reachable local script imports, model sources and protocol, without executing imports."""
    pending = list((BASE/'scripts').glob('*r74.py')); paths = set()
    while pending:
        path = pending.pop()
        if path in paths: continue
        paths.add(path)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) else
                       [item.name for item in node.names] if isinstance(node, ast.Import) else [])
            for name in modules:
                if name:
                    candidate = BASE/'scripts'/(name.split('.')[0]+'.py')
                    if candidate.is_file(): pending.append(candidate)
    paths.update((ROOT/'MobileIE/model').glob('*.py')); paths.add(PROTOCOL)
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--qualification-start', type=int, default=1); args = parser.parse_args()
    process = BASE/'process/exploration-r74'; cpu_path = process/'cpu-gate001/runtime.json'
    cpu = json.loads(cpu_path.read_text())
    assert cpu['passed'] and cpu['discarded_motion_updates'] == 8 and not cpu['CUDA_initialized']
    out = process/f'finalize{args.attempt:03d}'; out.mkdir(exist_ok=False)
    frozen = scientific_sources()
    write(out/'config.json', dict(at=stamp(), source_sha256=frozen, expected_formal_motion_updates=16000,
          new_spatial_updates=0, GPU_seeds_sequential=True, no_wall_cutoff=True,
          no_automatic_scientific_retry=True, CPU_qualification_sha256=digest(cpu_path),
          scope='One registered supervision/transport pilot, final4000 only. Not a novelty, quality, phone or final-test claim.'))
    started, complete, cost_executed = time.perf_counter(), False, False
    children, streams, render, render_stream = [], [], None, None
    quality_passed = None

    def check(): assert all(digest(ROOT/n) == value for n, value in frozen.items())
    def command(script, extra=()): return ['./scripts/project-python', str((BASE/'scripts'/script).relative_to(ROOT)), *extra]
    def state(value):
        write(out/'state.json', dict(at=stamp(), stage=value))
        print(json.dumps(dict(at=stamp(), stage=value)), flush=True)
    def run(script, log, extra=()):
        check()
        with (out/log).open('x') as stream:
            subprocess.run(command(script, extra), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)

    def guarded(script, tag, extra=()):
        attempt = 0
        while True:
            check(); chosen = None
            for gpu in [4, 5, 6, 7]:
                samples = [sample(gpu)]; time.sleep(1); samples.append(sample(gpu))
                if all(not r['compute_processes'] and r['memory_mib'] <= 64 and r['utilization_percent'] <= 5 for r in samples):
                    chosen = gpu; break
            if chosen is None:
                write(out/f'{tag}-resource-wait.json', dict(at=stamp(), resource_only=True, stage='Waiting for fresh idle same-group GPU samples'))
                time.sleep(30); continue
            attempt += 1; record = out/f'{tag}-guard{attempt:03d}.json'; log = out/f'{tag}-attempt{attempt:03d}.log'
            admission_number = args.qualification_start+attempt-1 if tag == 'qualification' else attempt
            cmd = command('gpu_guard.py', ['--gpu', str(chosen), '--record', str(record.relative_to(ROOT)),
                          '--', *command(script, [*extra, '--admission-attempt', str(admission_number)])])
            with log.open('x') as stream:
                child = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                time.sleep(.2)
                write(out/f'{tag}-launch{attempt:03d}.json', dict(at=stamp(), guard_pid=child.pid,
                      identity=process_identity(child.pid), gpu=chosen, command=cmd, preguard_samples=samples))
                code = child.wait()
            if not record.exists():
                if code != 0 and 'BlockingIOError' in log.read_text() and 'Resource temporarily unavailable' in log.read_text():
                    write(out/f'{tag}-lock-race{attempt:03d}.json', dict(at=stamp(), resource_only=True, command_not_admitted=True))
                    time.sleep(5); continue
                raise RuntimeError(f'{tag} failed before guard audit; no retry')
            audit = json.loads(record.read_text())
            if not audit['allowed']:
                time.sleep(5); continue
            if code == 75:
                if tag == 'qualification':
                    directory = process/f'cuda-gate{admission_number:03d}'
                    denied = json.loads((directory/'internal-admission.json').read_text())
                    terminal = json.loads((directory/'runtime.json').read_text())
                    assert terminal['discarded_motion_updates'] == 0 and not terminal['CUDA_initialized']
                else:
                    internal = 'metric' if tag == 'perceptual' else tag
                    denied = json.loads((process/f'{internal}-internal-admission{attempt:03d}.json').read_text())
                assert not denied['allowed']
                write(out/f'{tag}-resource-reject{attempt:03d}.json', dict(at=stamp(), resource_only=True, returncode=75, no_CUDA_or_updates=True))
                time.sleep(5); continue
            assert code == 0, f'{tag} admitted scientific failure preserved; no automatic retry'
            return record, admission_number

    try:
        state('Complete source chain frozen; separately admitted actual-window CUDA qualification next')
        guard, attempt = guarded('check_motion_transport_r74.py', 'qualification', ['--device', 'cuda'])
        gate = process/f'cuda-gate{attempt:03d}/runtime.json'
        assert json.loads(gate.read_text())['passed']
        write(process/'cuda-qualification.json', dict(passed=True, at=stamp(), gate_runtime=str(gate.relative_to(ROOT)),
              gate_sha256=digest(gate), guard=str(guard.relative_to(ROOT)), guard_sha256=digest(guard),
              discarded_updates_not_inherited=True))
        for seed in [0, 1]:
            state(f'Freshly initialized formal seed{seed} waits new external/internal GPU admission')
            guarded('train_motion_transport_r74.py', f'seed{seed}', ['--seed', str(seed)])
            runtime = json.loads((BASE/f'results/exploration-r74-stage001-seed{seed}/runtime.json').read_text())
            assert runtime['completed_normally'] and runtime['motion_model_updates'] == {n: 4000 for n in MODES}
            for name in MODES:
                marker = json.loads((BASE/f'results/exploration-r74-stage001-seed{seed}/{name}-step4000.json').read_text())
                assert marker['final_reload_passed']
        state('Both formal budgets normal and GPUs released; original two CPU evaluations actually starting')
        for seed in [0, 1]:
            stream = (out/f'quality-seed{seed}.log').open('x'); streams.append(stream)
            cmd = command('evaluate_motion_transport_r74.py', ['--seed', str(seed)])
            child = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT); children.append(child)
            time.sleep(.2)
            write(out/f'quality-seed{seed}-launch.json', dict(at=stamp(), pid=child.pid, identity=process_identity(child.pid), command=cmd))
        assert [child.wait() for child in children] == [0, 0]
        run('summarize_motion_transport_r74.py', 'quality-summary.log')
        render_stream = (out/'render.log').open('x'); cmd = command('render_motion_transport_r74.py')
        render = subprocess.Popen(cmd, cwd=ROOT, stdout=render_stream, stderr=subprocess.STDOUT)
        time.sleep(.2)
        write(out/'render-launch.json', dict(at=stamp(), pid=render.pid, identity=process_identity(render.pid), command=cmd))
        state('Full CPU quality normal; all fixed contacts and freshly admitted80-row perception next')
        guarded('evaluate_cvvdp_motion_transport_r74.py', 'perceptual')
        run('summarize_motion_transport_r74.py', 'joint-summary.log', ['--perceptual'])
        assert render.wait() == 0
        quality_passed = json.loads((BASE/'report/exploration-r74-joint/audit.json').read_text())['all_gates_pass']
        if quality_passed:
            state('All16 joint gates passed; GPU released and renderer finished, one fixed CPU cost experiment next')
            cost_executed = True; run('benchmark_motion_transport_r74.py', 'cost.log')
        else:
            state('One or more full joint gates failed; fixed CPU cost experiment skipped as registered')
        check(); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        for stream in streams: stream.close()
        if render_stream: render_stream.close()
        write(out/'runtime.json', dict(completed_normally=complete, seconds=time.perf_counter()-started, ended=stamp(),
              quality_gates_passed=quality_passed, cost_executed=cost_executed, actual_visual_review_still_required=True,
              scope='Coordinator terminal only; not method success, phone cost or independent final.'))


if __name__ == '__main__': main()
