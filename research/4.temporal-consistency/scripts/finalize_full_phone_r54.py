"""Observe identified CPU references, then render all prescribed phone artifacts."""
import json
import subprocess
import time
import traceback
from datetime import datetime

from dynamic_training_r34 import BASE, ROOT, digest
from finalize_bounded_video_r44 import read_stable
from finalize_preserved_r37 import process_identity


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    process = BASE/'process/exploration-r54'; out = process/'finalize001'; out.mkdir(exist_ok=False)
    dependencies = []
    for seed in [0, 1]:
        launch = read_stable(process/f'seed{seed}-launch.json')
        identity = process_identity(launch['pid'])
        terminal = BASE/f'results/exploration-r54-full-phone-seed{seed}/runtime.json'
        assert terminal.exists() or (identity and 'evaluate_full_phone_r54.py' in identity['commandline'])
        dependencies.append(dict(seed=seed, pid=launch['pid'], identity=identity))
    names = ['evaluate_full_phone_r54.py', 'render_full_phone_r54.py', 'finalize_full_phone_r54.py',
             'response_cue_r45.py', 'half_motion_r4.py', 'aligned_raw_history_r11.py', 'evaluate_phone_references_r38.py']
    frozen = {name: digest(BASE/'scripts'/name) for name in names}
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), dependencies=dependencies,
          source_sha256=frozen, no_wall_cutoff=True, no_automatic_retry=True,
          scope='Final fixed references only; visual review remains separate.'))
    start, complete = time.perf_counter(), False
    try:
        for dep in dependencies:
            terminal = BASE/f"results/exploration-r54-full-phone-seed{dep['seed']}/runtime.json"
            while not terminal.exists():
                if process_identity(dep['pid']) != dep['identity']:
                    time.sleep(1)
                    assert terminal.exists(), 'Identified CPU job disappeared without runtime; no retry.'
                    break
                time.sleep(5)
            runtime = read_stable(terminal)
            assert runtime['completed_normally'] and runtime['rows'] == 9 and runtime['neural_frames'] == 952
        assert all(digest(BASE/'scripts'/name) == sha for name, sha in frozen.items())
        with (out/'render.log').open('x') as stream:
            subprocess.run(['./scripts/project-python', str(BASE/'scripts/render_full_phone_r54.py')],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, seconds=time.perf_counter()-start,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    main()
