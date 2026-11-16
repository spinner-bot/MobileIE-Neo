"""One registered R82 chain; resource-only continuation, never scientific auto-retry."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
from self_reference_io_r82 import BASE, PROCESS, write, append, stamp, verify, owned_cache
from shared_gpu_admission import snapshot, assess


def choose(prefer=0):
    candidates = [r for r in assess([snapshot()])['candidates'] if r['index'] in (0, 1, 2, 3) and r['allowed']]
    available = []
    for row in candidates:
        with (ROOT/f'.cache/gpu-{row["index"]}.lock').open('a') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                available.append(row)
            except BlockingIOError:
                pass
    return min(available, key=lambda r: (bool(r['maximum_process_count']),
        r['index'] != prefer, r['mean_utilization_percent'], -r['minimum_free_mib'])) if available else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-seed', type=int, choices=(0, 1))
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    worker = args.train_seed
    name = 'pipeline001' if worker is None else f'seed{worker}-worker001'
    lock = (ROOT/f'.cache/self-reference-r82-{name}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = PROCESS/name; out.mkdir(exist_ok=False)
    logs = owned_cache('experiments/exploration-r82/'+name)
    frozen = verify()
    write(out/'config.json', dict(at=stamp(), pid=os.getpid(), frozen=frozen,
        scope='Fixed original pipeline; scientific failure preserved, only resource-only training continuation.'))
    began, complete, children = time.perf_counter(), False, []

    def launch(label, command):
        path = logs/f'{label}.log'
        with path.open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        fields = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()
        record = dict(at=stamp(), name=label, command=command, pid=child.pid,
            start_ticks=int(fields[19]), log=str(path.relative_to(ROOT)))
        append(out/'children.jsonl', record); children.append((child, record))
        return child, record

    def finish(pair, allowed_codes=(0,)):
        child, record = pair; code = child.wait()
        append(out/'child-terminals.jsonl', dict(**record, ended=stamp(), returncode=code))
        assert code in allowed_codes, f'Original {record["name"]} exited {code}; preserve, no scientific retry.'
        return code

    def bound(label, script, extra=(), prefer=0):
        trial = 0
        while True:
            chosen = choose(prefer)
            if chosen is None:
                write(out/'resource-wait.json', dict(at=stamp(), reason='No unlocked card with headroom in0..3'))
                time.sleep(15); continue
            trial += 1; external = out/f'{label}-external{trial:03d}.json'
            pair = launch(f'{label}-guard{trial:03d}', ['./scripts/project-python', 'scripts/prebind_shared_gpu.py',
                '--gpu', str(chosen['index']), '--record', str(external.relative_to(ROOT)), '--',
                './scripts/project-python', 'research/4.temporal-consistency/scripts/'+script, *extra])
            code = finish(pair, (0, 75))
            evidence = json.loads(external.read_text())
            if code == 75 and not evidence['allowed']:
                time.sleep(15); continue  # No scientific child existed.
            return code, evidence

    try:
        if worker is not None:
            attempt, resume = 0, None
            while True:
                attempt += 1
                extra = ['--seed', str(worker), '--attempt', str(attempt)]
                if resume:
                    extra += ['--resume', str(resume.relative_to(ROOT))]
                code, external = bound(f'train-attempt{attempt:03d}', 'train_self_reference_r82.py', extra, worker)
                if code == 0:
                    break
                run = BASE/f'results/exploration-r82-stage001-seed{worker}/attempt{attempt:03d}'
                runtime = json.loads((run/'runtime.json').read_text())
                assert runtime['stop_reason'] == 'resource' and not runtime['completed_normally']
                if runtime['latest_checkpoint']:
                    marker = runtime['latest_checkpoint']
                    assert marker['kind'] == 'resource'
                    resume = run/f'resource-step{marker["completed_steps"]:04d}.json'
                else:
                    assert runtime['completed_steps'] == 0
                time.sleep(15)
        else:
            assert not (PROCESS/'cuda-gate001').exists(), 'Inspect existing qualification; do not duplicate it.'
            code, _ = bound('cuda-qualification', 'check_self_reference_cuda_r82.py')
            assert code == 0
            gate = json.loads((PROCESS/'cuda-gate001/gate.json').read_text())
            assert gate['passed'] and gate['discarded_updates'] == 12
            pairlist = [launch(f'train-seed{seed}', ['./scripts/project-python',
                'research/4.temporal-consistency/scripts/run_self_reference_r82.py', '--train-seed', str(seed)]) for seed in (0, 1)]
            for pair in pairlist:
                finish(pair)
            pairlist = [launch(f'quality-seed{seed}', ['./scripts/project-python',
                'research/4.temporal-consistency/scripts/evaluate_self_reference_r82.py', '--seed', str(seed)]) for seed in (0, 1)]
            for pair in pairlist:
                finish(pair)
            code, _ = bound('perceptual', 'evaluate_cvvdp_self_reference_r82.py')
            assert code == 0
            for label, script in (('joint', 'summarize_self_reference_r82.py'), ('contacts', 'render_self_reference_r82.py')):
                finish(launch(label, ['./scripts/project-python', 'research/4.temporal-consistency/scripts/'+script]))
        verify(); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(ended=stamp(), pid=os.getpid(), completed_normally=complete,
            seconds=time.perf_counter()-began, actual_visual_review_pending=complete and worker is None,
            children_still_running=[r for c,r in children if c.poll() is None],
            scope='No wall deadline, interim DEV, automatic method/phone claim or final pixel unsealing.'))
        globals()['_held_lock'] = lock


if __name__ == '__main__':
    main()
