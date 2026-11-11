"""Wait for the two identified guarded trainings, then evaluate both final seeds."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time
import traceback

from dynamic_training_r34 import BASE, ROOT, digest


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def process_identity(pid):
    proc = Path('/proc')/str(pid)
    try:
        start = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]
        command = proc.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode()
        return {'start_ticks': start, 'commandline': command}
    except FileNotFoundError:
        return None


def main():
    folder = BASE/'process/exploration-r37/finalize001'
    folder.mkdir(parents=True, exist_ok=False)
    sources = ['finalize_preserved_r37.py', 'evaluate_preserved_r37.py', 'summarize_preserved_r37.py',
               'preserved_state_r37.py', 'training_preserved_r37.py', 'budget_causal_r33.py',
               'exploration_r2_data.py', 'video_metrics.py']
    frozen = {name: digest(Path(__file__).parent/name) for name in sources}
    dependencies = []
    for seed in [0, 1]:
        launch = json.loads((BASE/f'process/exploration-r37/seed{seed}-launch.json').read_text())
        guard = json.loads((BASE/f'process/exploration-r37/seed{seed}-guard.json').read_text())
        assert guard['allowed']
        identity = process_identity(launch['guard_pid'])
        assert 'returncode' in guard or (identity and 'gpu_guard.py' in identity['commandline'])
        dependencies.append({'seed': seed, 'pid': launch['guard_pid'], 'identity': identity})
    write(folder/'config.json', {'created_at': datetime.now().astimezone().isoformat(),
          'dependencies': dependencies, 'source_sha256': frozen, 'no_wall_cutoff': True,
          'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'scope': 'Complete both training budgets before CPU final4000 evaluation; failure is retained without automatic retry or new model selection.'})
    started, complete, evaluated = time.perf_counter(), False, {}
    try:
        for dep in dependencies:
            guard_path = BASE/f"process/exploration-r37/seed{dep['seed']}-guard.json"
            while True:
                guard = json.loads(guard_path.read_text())
                if 'returncode' in guard:
                    assert guard['returncode'] == 0
                    runtime = json.loads((BASE/f"results/exploration-r37-seed{dep['seed']}/runtime.json").read_text())
                    assert runtime['completed_normally'] and runtime['total_updates'] == 8000
                    assert set(runtime['model_updates'].values()) == {4000}
                    break
                assert process_identity(dep['pid']) == dep['identity'], 'Dependency no longer matches its recorded live process'
                time.sleep(2)
        assert all(digest(Path(__file__).parent/name) == value for name, value in frozen.items())
        children, streams = {}, {}
        try:
            for seed in [0, 1]:
                streams[seed] = (folder/f'evaluate-seed{seed}.log').open('x')
                command = ['./scripts/project-python', str((Path(__file__).parent/'evaluate_preserved_r37.py').relative_to(ROOT)),
                           '--seed', str(seed)]
                children[seed] = subprocess.Popen(command, cwd=ROOT, stdout=streams[seed], stderr=subprocess.STDOUT)
                write(folder/f'evaluate-seed{seed}-launch.json', {'pid': children[seed].pid, 'command': command,
                      'started_at': datetime.now().astimezone().isoformat()})
            for seed, child in children.items():
                evaluated[seed] = child.wait()
        finally:
            for stream in streams.values():
                stream.close()
        assert set(evaluated) == {0, 1} and all(code == 0 for code in evaluated.values()), evaluated
        with (folder/'summarize.log').open('x') as stream:
            subprocess.run(['./scripts/project-python', str((Path(__file__).parent/'summarize_preserved_r37.py').relative_to(ROOT))],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        complete = True
    except Exception:
        (folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(folder/'runtime.json', {'completed_normally': complete, 'evaluation_returncodes': evaluated,
              'wall_seconds': time.perf_counter()-started, 'ended_at': datetime.now().astimezone().isoformat(),
              'scope': 'Quality closure only; no phone/cost or novelty conclusion is automatic.'})


if __name__ == '__main__':
    main()
