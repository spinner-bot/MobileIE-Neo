"""Finish only the identified fixed Night evaluation, then complete tables/visuals."""
import json
import subprocess
import time
import traceback
from datetime import datetime

from dynamic_training_r34 import BASE, ROOT, digest
from fetch_wenzhou_samples_r52 import write
from finalize_bounded_video_r44 import read_stable
from finalize_preserved_r37 import process_identity


def main():
    folder = BASE/'process/exploration-r60'
    out = folder/'finalize001'; out.mkdir(exist_ok=False)
    launch = read_stable(folder/'evaluate001-launch.json')
    identity = process_identity(launch['pid'])
    terminal = BASE/'results/exploration-r60-wenzhou-reference-attempt001/runtime.json'
    names = ['finalize_wenzhou_references_r60.py', 'summarize_wenzhou_r60.py', 'render_wenzhou_r60.py',
             'wenzhou_references_r60.py', 'evaluate_wenzhou_r60.py']
    frozen = {name: digest(BASE/'scripts'/name) for name in names}
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), pid=launch['pid'], identity=identity,
          source_sha256=frozen, no_wall_cutoff=True, no_retry=True,
          scope='Wait for one fixed CPU evaluation; complete-only tables/all fixed figures/previews, no new CNN/SGD/GPU.'))
    start, complete = time.perf_counter(), False
    try:
        assert terminal.exists() or (identity and 'evaluate_wenzhou_r60.py' in identity['commandline'])
        while not terminal.exists():
            if process_identity(launch['pid']) != identity:
                time.sleep(1)
                assert terminal.exists(), 'Evaluation disappeared without terminal; no automatic retry.'
                break
            time.sleep(5)
        status = read_stable(terminal)
        assert status['completed_normally'] and status['sequences'] == 96 and status['CNN_center_outputs'] == 6144
        assert all(digest(BASE/'scripts'/name) == sha for name, sha in frozen.items())
        for name in ['summarize_wenzhou_r60.py', 'render_wenzhou_r60.py']:
            with (out/(name+'.log')).open('x') as stream:
                subprocess.run(['./scripts/project-python', str(BASE/'scripts'/name)], cwd=ROOT,
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
        assert read_stable(BASE/'report/exploration-r60-wenzhou/summary-audit.json')['completed_normally']
        assert read_stable(folder/'render001/runtime.json')['completed_normally']
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, seconds=time.perf_counter()-start,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
