"""Wait for the identified one-shot recovery; list only its complete archive."""
import json
import subprocess
import time
import traceback
from datetime import datetime

from dynamic_training_r34 import BASE, ROOT, digest
from finalize_bounded_video_r44 import read_stable
from finalize_preserved_r37 import process_identity
from fetch_wenzhou_samples_r52 import write


def main():
    process = BASE/'process/external-night-wenzhou'
    out = process/'finalize002'; out.mkdir(exist_ok=False)
    launch = read_stable(process/'acquisition003-launch.json')
    terminal = process/'acquisition003/runtime.json'
    manifest = process/'acquisition003/manifest.json'
    catalog = process/'catalog002'
    identity = process_identity(launch['pid'])
    sources = ['resume_wenzhou_samples_r52.py', 'list_wenzhou_samples_r52.py',
               'finalize_wenzhou_recovery_r52.py']
    frozen = {name: digest(BASE/'scripts'/name) for name in sources}
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), pid=launch['pid'],
          identity=identity, source_sha256=frozen, no_wall_cutoff=True, no_download_retry=True,
          manifest=str(manifest.relative_to(ROOT)), catalog=str(catalog.relative_to(ROOT)),
          scope='Wait for actual recovered complete archive; headers only, no member extraction/image decode/CNN.'))
    start, complete = time.perf_counter(), False
    try:
        assert terminal.exists() or (identity and 'resume_wenzhou_samples_r52.py' in identity['commandline'])
        while not terminal.exists():
            if process_identity(launch['pid']) != identity:
                time.sleep(1)
                assert terminal.exists(), 'Recovery disappeared without terminal; no automatic retry.'
                break
            time.sleep(5)
        assert read_stable(terminal)['completed_normally']
        assert read_stable(manifest)['download_complete']
        assert all(digest(BASE/'scripts'/name) == sha for name, sha in frozen.items())
        command = ['./scripts/project-python', str(BASE/'scripts/list_wenzhou_samples_r52.py'),
                   '--manifest', str(manifest), '--out', str(catalog)]
        with (out/'catalog.log').open('x') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        assert read_stable(catalog/'runtime.json')['completed_normally']
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, seconds=time.perf_counter()-start,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    main()
