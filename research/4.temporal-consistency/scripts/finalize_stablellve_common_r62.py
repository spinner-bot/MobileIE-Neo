"""Wait for the identified common-CPU reference, then summarize once; no retry."""
import subprocess
import time
import traceback
from datetime import datetime

from dynamic_training_r34 import BASE, ROOT, digest
from fetch_wenzhou_samples_r52 import write
from finalize_bounded_video_r44 import read_stable
from finalize_preserved_r37 import process_identity


def main():
    process = BASE/'process/exploration-r62'
    out = process/'finalize001'; out.mkdir(exist_ok=False)
    launch = read_stable(process/'evaluate001-launch.json')
    identity = process_identity(launch['pid'])
    terminal = BASE/'results/exploration-r62-stablellve-common-attempt001/runtime.json'
    names = ['evaluate_stablellve_common_r62.py', 'summarize_stablellve_common_r62.py', 'finalize_stablellve_common_r62.py']
    frozen = {name:digest(BASE/'scripts'/name) for name in names}
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), pid=launch['pid'], identity=identity,
          source_sha256=frozen, no_wall_cutoff=True, no_retry=True,
          scope='One actual common-CPU clean-reference evaluation, then complete-only summary; no new neural/SGD/GPU.'))
    start, complete = time.perf_counter(), False
    try:
        assert terminal.exists() or (identity and 'evaluate_stablellve_common_r62.py' in identity['commandline'])
        while not terminal.exists():
            if process_identity(launch['pid']) != identity:
                time.sleep(1); assert terminal.exists(), 'Evaluation disappeared without terminal; no retry.'
                break
            time.sleep(5)
        status = read_stable(terminal)
        assert status['completed_normally'] and status['sequences'] == 20 and status['CNN_center_outputs'] == 448
        assert all(digest(BASE/'scripts'/name) == sha for name,sha in frozen.items())
        with (out/'summary.log').open('x') as stream:
            subprocess.run(['./scripts/project-python', str(BASE/'scripts/summarize_stablellve_common_r62.py')],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        assert read_stable(BASE/'report/exploration-r62-stablellve-common/audit.json')['completed_normally']
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, seconds=time.perf_counter()-start,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
