"""Observe the identified fixed reference job, then build all registered artifacts."""
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
    process = BASE/'process/exploration-r55'; out = process/'finalize001'; out.mkdir(exist_ok=False)
    launch = read_stable(process/'evaluate001-launch.json')
    identity = process_identity(launch['pid'])
    terminal = BASE/'results/exploration-r55-stage001/runtime.json'
    assert terminal.exists() or (identity and 'evaluate_multinex_reference_r55.py' in identity['commandline'])
    names = ['multinex_reference_r55.py', 'evaluate_multinex_reference_r55.py', 'summarize_multinex_reference_r55.py',
             'render_multinex_phone_r55.py', 'finalize_multinex_reference_r55.py', 'half_motion_r4.py',
             'aligned_raw_history_r11.py', 'video_metrics.py', 'exploration_r2_data.py', 'prepare_cache_r5.py']
    frozen = {str((BASE/'scripts'/n).relative_to(ROOT)): digest(BASE/'scripts'/n) for n in names}
    for name in ['basicsr/models/archs/MultinexNano_arch.py', 'Options/Multinex-Nano_LOL-v1.yaml', 'UPSTREAM.json']:
        path = ROOT/'third_party/multinex'/name; frozen[str(path.relative_to(ROOT))] = digest(path)
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), pid=launch['pid'], identity=identity,
          source_sha256=frozen, no_wall_cutoff=True, no_automatic_retry=True,
          scope='All fixed rows and previews regardless of winner; visual review remains separate.'))
    start, complete = time.perf_counter(), False
    try:
        while not terminal.exists():
            if process_identity(launch['pid']) != identity:
                time.sleep(1)
                assert terminal.exists(), 'Identified evaluation disappeared without terminal record; no retry.'
                break
            time.sleep(5)
        runtime = read_stable(terminal)
        assert runtime['completed_normally'] and runtime['quality_rows'] == 20 and runtime['phone_rows'] == 6
        assert runtime['neural_sequences'] == 13 and runtime['neural_frames'] == 1400
        assert all(digest(ROOT/name) == sha for name, sha in frozen.items())
        for name in ['summarize_multinex_reference_r55.py', 'render_multinex_phone_r55.py']:
            with (out/(name+'.log')).open('x') as stream:
                subprocess.run(['./scripts/project-python', str(BASE/'scripts'/name)], cwd=ROOT,
                               stdout=stream, stderr=subprocess.STDOUT, check=True)
        assert all(digest(ROOT/name) == sha for name, sha in frozen.items())
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, seconds=time.perf_counter()-start,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    main()
