"""One frozen coefficient-state experiment; no network training or scientific auto-retry."""
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
from coefficient_io_r80 import BASE, PROCESS, write, append, stamp, verify, owned_cache
from shared_gpu_admission import assess, snapshot


def main():
    lock = (ROOT/'.cache/coefficient-r80-finalize.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen = verify(); out = PROCESS/'finalize001'; out.mkdir(exist_ok=False)
    logs = owned_cache('experiments/exploration-r80/finalize001')
    write(out/'config.json', dict(at=stamp(), pid=os.getpid(), frozen_sources=frozen,
        scope='Two original CPU jobs, then complete new perceptual matrix and evidence. No scientific auto-retry or final-test unsealing.'))
    started, complete, active = time.perf_counter(), False, []
    cpu = os.environ.copy(); cpu['CUDA_VISIBLE_DEVICES'] = ''

    def launch(name, command, env=None):
        path = logs/f'{name}.log'
        with path.open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        ticks = int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
        record = dict(at=stamp(), name=name, command=command, pid=child.pid, start_ticks=ticks,
                      log=str(path.relative_to(ROOT)))
        append(out/'children.jsonl', record); active.append((child, record)); return child, record

    def terminal(child, record):
        code = child.wait()
        append(out/'child-terminals.jsonl', dict(**record, ended=stamp(), returncode=code))
        assert code == 0, f'{record["name"]} returned {code}; preserve and investigate without rerunning science.'

    try:
        children = [launch(f'quality-seed{seed}', ['./scripts/project-python',
            'research/4.temporal-consistency/scripts/evaluate_coefficient_state_r80.py', '--seed', str(seed)], cpu)
            for seed in (0, 1)]
        for child, record in children:
            terminal(child, record)
        verify(); attempt = 0
        while True:
            candidates = [r for r in assess([snapshot()])['candidates'] if r['index'] in (0, 1, 2, 3) and r['allowed']]
            if not candidates:
                time.sleep(15); continue
            selected = min(candidates, key=lambda r: (r['mean_utilization_percent'], -r['minimum_free_mib']))
            attempt += 1; external = out/f'metric-external{attempt:03d}.json'
            child, record = launch(f'metric-guard{attempt:03d}', ['./scripts/project-python',
                'scripts/prebind_shared_gpu.py', '--gpu', str(selected['index']), '--record', str(external.relative_to(ROOT)),
                '--', './scripts/project-python', 'research/4.temporal-consistency/scripts/evaluate_cvvdp_coefficient_r80.py'])
            code = child.wait()
            append(out/'child-terminals.jsonl', dict(**record, ended=stamp(), returncode=code))
            if code == 75 and not json.loads(external.read_text())['allowed']:
                time.sleep(15); continue  # No metric child or score existed; resource-only wait.
            assert code == 0, 'An actual metric child failed; no scientific auto-retry.'
            break
        for name, script in (('joint', 'summarize_coefficient_state_r80.py'), ('contacts', 'render_coefficient_state_r80.py')):
            child, record = launch(name, ['./scripts/project-python', 'research/4.temporal-consistency/scripts/'+script], cpu)
            terminal(child, record)
        verify(); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(ended=stamp(), completed_normally=complete, pid=os.getpid(),
            seconds=time.perf_counter()-started, actual_visual_review_pending=complete,
            children_still_running=[record for child, record in active if child.poll() is None],
            scope='Complete fixed experiment execution, not method/phone/independent-final success.'))
        globals()['_held_finalize_lock'] = lock


if __name__ == '__main__':
    main()
