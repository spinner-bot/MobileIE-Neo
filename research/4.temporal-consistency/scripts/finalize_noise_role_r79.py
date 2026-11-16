"""Observe the original trainers, then one complete frozen final evidence chain."""
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
from noise_role_io_r79 import BASE, PROCESS, write, append, stamp, verify, owned_cache
from noise_role_evidence_r79 import verify_evaluation
from shared_gpu_admission import snapshot, assess


def main():
    lock = (ROOT/'.cache/noise-role-r79-finalize.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    out = PROCESS/'finalize001'; out.mkdir(exist_ok=False)
    logs = owned_cache('experiments/exploration-r79/finalize001')
    verify(); frozen = verify_evaluation()
    write(out/'config.json', dict(at=stamp(), pid=os.getpid(), evaluation_freeze=frozen,
        scope='Original guards only; no scientific auto-retry, wall cutoff, interim DEV or automatic final unsealing.'))
    began, complete, active = time.perf_counter(), False, []

    def launch(name, command, env=None):
        path = logs/f'{name}.log'
        with path.open('x') as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        ticks = int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19])
        record = dict(at=stamp(), name=name, command=command, pid=child.pid, start_ticks=ticks,
                      log=str(path.relative_to(ROOT)))
        append(out/'children.jsonl', record)
        active.append((child, record))
        return child, record

    def finish(child, record):
        code = child.wait()
        append(out/'child-terminals.jsonl', dict(**record, ended=stamp(), returncode=code))
        assert code == 0, f'{record["name"]} exited {code}; preserve its original output.'

    cpu_env = os.environ.copy(); cpu_env['CUDA_VISIBLE_DEVICES'] = ''
    try:
        while True:
            done = []
            for seed in (0, 1):
                launch_record = json.loads((PROCESS/f'seed{seed}-launch001.json').read_text())
                path = PROCESS/f'seed{seed}-external001.json'
                guard = json.loads(path.read_text()) if path.exists() else {}
                if 'returncode' in guard:
                    assert guard['returncode'] == 0, f'Original seed{seed} guard did not complete: {guard["returncode"]}'
                    done.append(seed)
                else:
                    stat = Path(f'/proc/{launch_record["pid"]}/stat')
                    assert stat.exists(), 'Original guard absent without terminal record; investigate, do not restart.'
                    fields = stat.read_text().rsplit(')', 1)[1].split()
                    assert int(fields[19]) == launch_record['start_ticks'] and fields[0] != 'Z'
            if len(done) == 2:
                break
            write(out/'waiting.json', dict(at=stamp(), completed_original_guards=done,
                scope='Training incomplete; no DEV evaluation yet.'))
            time.sleep(15)
        verify(); verify_evaluation()
        children = [launch(f'quality-seed{seed}', ['./scripts/project-python',
            'research/4.temporal-consistency/scripts/evaluate_noise_role_r79.py', '--seed', str(seed)], cpu_env)
                    for seed in (0, 1)]
        for child, record in children:
            finish(child, record)
        # A failed external admission has no metric child; only this resource-only wait can repeat.
        attempt = 0
        while True:
            candidates = [r for r in assess([snapshot()])['candidates'] if r['index'] in (0, 1, 2, 3) and r['allowed']]
            if not candidates:
                time.sleep(15); continue
            chosen = min(candidates, key=lambda r: (r['mean_utilization_percent'], -r['minimum_free_mib']))
            attempt += 1
            external = out/f'metric-external{attempt:03d}.json'
            child, record = launch(f'metric-guard{attempt:03d}', ['./scripts/project-python',
                'scripts/prebind_shared_gpu.py', '--gpu', str(chosen['index']), '--record', str(external.relative_to(ROOT)),
                '--', './scripts/project-python', 'research/4.temporal-consistency/scripts/evaluate_cvvdp_noise_role_r79.py'])
            code = child.wait()
            append(out/'child-terminals.jsonl', dict(**record, ended=stamp(), returncode=code))
            if code == 75 and not json.loads(external.read_text())['allowed']:
                time.sleep(15); continue
            assert code == 0, 'Actual metric child failed; no scientific auto-retry.'
            break
        for name, script in (('joint', 'summarize_noise_role_r79.py'), ('contacts', 'render_noise_role_r79.py')):
            child, record = launch(name, ['./scripts/project-python', 'research/4.temporal-consistency/scripts/'+script], cpu_env)
            finish(child, record)
        verify(); verify_evaluation(); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', dict(ended=stamp(), pid=os.getpid(), completed_normally=complete,
            seconds=time.perf_counter()-began, actual_visual_review_pending=complete,
            children_still_running=[record for child, record in active if child.poll() is None],
            scope='Evidence generation only; no automatic method/phone/final claim. Original child states preserved.'))
        globals()['_held_finalize_lock'] = lock


if __name__ == '__main__':
    main()
