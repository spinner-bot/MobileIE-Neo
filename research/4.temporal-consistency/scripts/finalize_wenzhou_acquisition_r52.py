"""Wait for the already running author backup download, then list headers only."""
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
    process = BASE/'process/external-night-wenzhou'
    out = process/'finalize001'
    out.mkdir(exist_ok=False)
    launch = read_stable(process/'acquisition001-launch.json')
    identity = process_identity(launch['pid'])
    terminal = process/'acquisition001/runtime.json'
    assert terminal.exists() or (identity and 'fetch_wenzhou_samples_r52.py' in identity['commandline'])
    names = ['fetch_wenzhou_samples_r52.py', 'list_wenzhou_samples_r52.py', 'finalize_wenzhou_acquisition_r52.py']
    frozen = {name: digest(BASE/'scripts'/name) for name in names}
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), pid=launch['pid'],
          identity=identity, source_sha256=frozen, no_wall_cutoff=True, no_download_retry=True,
          scope='Wait for current download only; header catalog, no extraction/image or video decode/CNN.'))
    start, complete = time.perf_counter(), False
    try:
        while not terminal.exists():
            if process_identity(launch['pid']) != identity:
                time.sleep(1)
                assert terminal.exists(), 'Downloader disappeared without terminal metadata; no automatic retry.'
                break
            time.sleep(5)
        assert read_stable(terminal)['completed_normally']
        assert all(digest(BASE/'scripts'/name) == sha for name, sha in frozen.items())
        with (out/'catalog.log').open('x') as stream:
            subprocess.run(['./scripts/project-python', str(BASE/'scripts/list_wenzhou_samples_r52.py')],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        assert read_stable(process/'catalog001/runtime.json')['completed_normally']
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, seconds=time.perf_counter()-start,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    main()
