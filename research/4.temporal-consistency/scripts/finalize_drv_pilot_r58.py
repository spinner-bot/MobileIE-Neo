"""Follow the actual fixed acquisition, then only its preregistered RAW rendering."""
from datetime import datetime
import json
import subprocess
import time
import traceback

from dynamic_training_r34 import BASE, ROOT, digest
from finalize_bounded_video_r44 import read_stable
from finalize_preserved_r37 import process_identity


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    out = BASE/'process/external-drv/raw-finalize001'; out.mkdir(exist_ok=False)
    launch = read_stable(BASE/'process/external-drv/raw-pilot001-launch.json')
    identity = process_identity(launch['pid'])
    assert identity and 'fetch_drv_pilot_r58.py' in identity['commandline']
    assert str(identity['start_ticks']) == launch['proc_start_ticks']
    files = [BASE/'scripts'/n for n in ['fetch_drv_pilot_r58.py', 'render_drv_pilot_r58.py',
             'drv_render_r58.py', 'finalize_drv_pilot_r58.py']]
    files += [BASE/'data/external-drv/raw-pilot-plan-r58.json',
              BASE/'docs/exploration-r58-drv-static-raw-pilot-protocol.md',
              BASE/'process/external-drv/render-gate001/gate.json']
    frozen = {str(p.relative_to(ROOT)):digest(p) for p in files}
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(),
          dependency_pid=launch['pid'], dependency_identity=identity, source_sha256=frozen,
          no_wall_cutoff=True, no_automatic_retry=True, new_CNN=False, GPU=False,
          scope='Wait for existing complete36 acquisition, then fixed render only; metadata failure stops.'))
    start, complete, code = time.perf_counter(), False, None
    try:
        terminal = BASE/'process/external-drv/raw-acquisition001/runtime.json'
        while not terminal.exists():
            now = process_identity(launch['pid'])
            assert now and now['start_ticks'] == identity['start_ticks'], 'Original acquisition ended without runtime'
            time.sleep(3)
        run = read_stable(terminal)
        assert run['completed_normally'] and run['complete_members'] == 36
        assert all(digest(ROOT/n) == sha for n,sha in frozen.items())
        command = ['./scripts/project-python', str(BASE/'scripts/render_drv_pilot_r58.py')]
        with (out/'render.log').open('xb') as stream:
            child = subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            write(out/'render-launch.json',dict(at=datetime.now().astimezone().isoformat(),pid=child.pid,command=command))
            code = child.wait()
        assert code == 0
        rendered = read_stable(BASE/'results/exploration-r58-drv-render-attempt001/runtime.json')
        assert rendered['completed_normally'] and rendered['rendered_scenes'] == 4
        assert all(digest(ROOT/n) == sha for n,sha in frozen.items()); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json',dict(completed_normally=complete,render_returncode=code,
              seconds=time.perf_counter()-start,ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
