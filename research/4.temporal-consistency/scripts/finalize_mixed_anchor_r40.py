"""Wait for identified complete R40 trainings, then run the frozen CPU evaluation."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time
import traceback

from dynamic_training_r34 import BASE, ROOT, digest
from finalize_preserved_r37 import process_identity


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def read_stable(path):
    # The existing guard writes its terminal JSON in place, so tolerate that write.
    for attempt in range(5):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 4:
                raise
            time.sleep(.1)


def main():
    folder = BASE/'process/exploration-r40/finalize001'
    folder.mkdir(parents=True, exist_ok=False)
    names = ['finalize_mixed_anchor_r40.py', 'evaluate_mixed_anchor_r40.py', 'summarize_mixed_anchor_r40.py',
             'half_motion_r4.py', 'aligned_raw_history_r11.py', 'exploration_r2_data.py', 'video_metrics.py',
             'finalize_preserved_r37.py']
    frozen = {name: digest(Path(__file__).parent/name) for name in names}
    dependencies = []
    for seed in [0, 1]:
        launch = read_stable(BASE/f'process/exploration-r40/seed{seed}-launch.json')
        guard = read_stable(BASE/f'process/exploration-r40/seed{seed}-guard.json')
        assert guard['allowed']
        identity = process_identity(launch['guard_pid'])
        assert 'returncode' in guard or (identity and 'gpu_guard.py' in identity['commandline'])
        dependencies.append({'seed': seed, 'pid': launch['guard_pid'], 'identity': identity})
    write(folder/'config.json', {'created_at': datetime.now().astimezone().isoformat(),
          'dependencies': dependencies, 'source_sha256': frozen, 'no_wall_cutoff': True,
          'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'scope': 'Both fixed4000 trainings must complete before final CPU quality; no automatic retry or checkpoint selection.'})
    started, evaluated, complete = time.perf_counter(), {}, False
    try:
        for dep in dependencies:
            path = BASE/f"process/exploration-r40/seed{dep['seed']}-guard.json"
            while True:
                guard = read_stable(path)
                if 'returncode' in guard:
                    assert guard['returncode'] == 0
                    runtime = read_stable(BASE/f"results/exploration-r40-seed{dep['seed']}/runtime.json")
                    assert runtime['completed_normally'] and runtime['final_step'] == runtime['new_updates'] == 4000
                    break
                assert process_identity(dep['pid']) == dep['identity'], 'Recorded training guard no longer matches'
                time.sleep(2)
        assert all(digest(Path(__file__).parent/name) == value for name, value in frozen.items())
        children, streams = {}, {}
        try:
            for seed in [0, 1]:
                command = ['./scripts/project-python', str((Path(__file__).parent/'evaluate_mixed_anchor_r40.py').relative_to(ROOT)),
                           '--seed', str(seed)]
                streams[seed] = (folder/f'evaluate-seed{seed}.log').open('x')
                children[seed] = subprocess.Popen(command, cwd=ROOT, stdout=streams[seed], stderr=subprocess.STDOUT)
                write(folder/f'evaluate-seed{seed}-launch.json', {'pid': children[seed].pid, 'command': command,
                      'started_at': datetime.now().astimezone().isoformat()})
            for seed, child in children.items():
                evaluated[seed] = child.wait()
        finally:
            for stream in streams.values():
                stream.close()
        assert set(evaluated) == {0, 1} and all(value == 0 for value in evaluated.values()), evaluated
        with (folder/'summarize.log').open('x') as stream:
            subprocess.run(['./scripts/project-python', str((Path(__file__).parent/'summarize_mixed_anchor_r40.py').relative_to(ROOT))],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        complete = True
    except Exception:
        (folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(folder/'runtime.json', {'completed_normally': complete, 'evaluation_returncodes': evaluated,
              'wall_seconds': time.perf_counter()-started, 'ended_at': datetime.now().astimezone().isoformat(),
              'scope': 'Quality closure only; a mixed-data gain is not automatically temporal innovation or a new method.'})


if __name__ == '__main__':
    main()
