"""Immutable R73 qualification, sequential GPU seeds, and complete evidence chain."""
from datetime import datetime
import argparse
import json
import subprocess
import time
import traceback

from native_moment_core_r73 import BASE, ROOT, PROTOCOL, MODES
from check_native_moment_r73 import write
from dynamic_training_r34 import digest
from finalize_preserved_r37 import process_identity
from gpu_guard import sample


def stamp(): return datetime.now().astimezone().isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--qualification-start', type=int, default=1)
    args = parser.parse_args()
    process = BASE/'process/exploration-r73'
    cpu = json.loads((process/'cpu-gate001/runtime.json').read_text())
    assert cpu['passed'] and cpu['discarded_SGD_updates'] == 12 and not cpu['CUDA_initialized']
    out = process/f'finalize{args.attempt:03d}'; out.mkdir(exist_ok=False)
    names = ['mixed_anchor_r40.py', 'dynamic_training_r34.py', 'native_training_r14.py', 'training_form_r41.py',
             'budget_causal_r33.py', 'gpu_guard.py', 'prepare_cache_r5.py', 'exploration_r2_data.py',
             'video_metrics.py', 'half_motion_r4.py', 'aligned_raw_history_r11.py', 'aggregation_controls_r20.py',
             'render_dense_history_r64.py', 'cvvdp_adapter_r66.py', 'evaluate_cvvdp_references_r66.py',
             'train_exploration_r2.py', 'train_mixture_r4b.py']
    paths = list((BASE/'scripts').glob('*r73.py')) + [BASE/'scripts'/n for n in names]
    paths += list((ROOT/'MobileIE/model').glob('*.py')) + [PROTOCOL]
    frozen = {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))}
    write(out/'config.json', dict(started=stamp(), source_sha256=frozen, expected_formal_updates=24000,
          GPU_seeds_sequential=True, no_wall_cutoff=True, no_automatic_scientific_retry=True,
          CPU_qualification_sha256=digest(process/'cpu-gate001/runtime.json'),
          scope='One registered root mechanism; final4000 only. No new method success, phone or independent final.'))
    started, complete = time.perf_counter(), False
    children, streams, render, render_stream = [], [], None, None

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
                    assert terminal['discarded_SGD_updates'] == 0 and not terminal['CUDA_initialized']
                else:
                    internal = 'metric' if tag == 'perceptual' else tag
                    denied = json.loads((process/f'{internal}-internal-admission{attempt:03d}.json').read_text())
                assert not denied['allowed']
                write(out/f'{tag}-resource-reject{attempt:03d}.json', dict(at=stamp(), resource_only=True, returncode=75, no_CUDA_or_updates=True))
                time.sleep(5); continue
            assert code == 0, f'{tag} admitted scientific failure preserved; no automatic retry'
            return record, admission_number

    try:
        state('Full source chain frozen; separately admitted CUDA actual-window qualification next')
        guard, attempt = guarded('check_native_moment_r73.py', 'qualification', ['--device', 'cuda'])
        gate = process/f'cuda-gate{attempt:03d}/runtime.json'
        assert json.loads(gate.read_text())['passed']
        write(process/'cuda-qualification.json', dict(passed=True, at=stamp(), gate_runtime=str(gate.relative_to(ROOT)),
              gate_sha256=digest(gate), guard=str(guard.relative_to(ROOT)), guard_sha256=digest(guard),
              discarded_updates_not_inherited=True))
        for seed in [0, 1]:
            state(f'Qualification normal; formal seed{seed} waits new external/internal GPU admission and fresh initialization')
            guarded('train_native_moment_r73.py', f'seed{seed}', ['--seed', str(seed)])
            runtime = json.loads((BASE/f'results/exploration-r73-stage001-seed{seed}/runtime.json').read_text())
            assert runtime['completed_normally'] and runtime['model_updates'] == {n: 4000 for n in MODES}
            for name in MODES:
                marker = json.loads((BASE/f'results/exploration-r73-stage001-seed{seed}/{name}-step4000.json').read_text())
                assert marker['final_fold_passed']
            stream = (out/f'quality-seed{seed}.log').open('x'); streams.append(stream)
            cmd = command('evaluate_native_moment_r73.py', ['--seed', str(seed)])
            child = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT); children.append(child)
            time.sleep(.2)
            write(out/f'quality-seed{seed}-launch.json', dict(at=stamp(), pid=child.pid, identity=process_identity(child.pid), command=cmd))
            state(f'GPU seed{seed} normal and released; original CPU quality child actually started')
        assert [c.wait() for c in children] == [0, 0]
        run('summarize_native_moment_r73.py', 'quality-summary.log')
        render_stream = (out/'render.log').open('x')
        cmd = command('render_native_moment_r73.py')
        render = subprocess.Popen(cmd, cwd=ROOT, stdout=render_stream, stderr=subprocess.STDOUT)
        write(out/'render-launch.json', dict(at=stamp(), pid=render.pid, command=cmd))
        state('All final CPU quality and old reference caches normal; fixed contacts and freshly admitted240-row perception next')
        guarded('evaluate_cvvdp_native_moment_r73.py', 'perceptual')
        run('summarize_native_moment_r73.py', 'joint-summary.log', ['--perceptual'])
        assert render.wait() == 0
        check(); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        for stream in streams: stream.close()
        if render_stream: render_stream.close()
        write(out/'runtime.json', dict(completed_normally=complete, seconds=time.perf_counter()-started,
              ended=stamp(), actual_visual_review_still_required=True,
              scope='Coordinator terminal only; not a quality acceptance or new method/phone/final-test claim.'))


if __name__ == '__main__': main()
