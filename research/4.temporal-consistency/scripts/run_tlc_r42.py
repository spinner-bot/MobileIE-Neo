"""Run both registered CPU TLC evaluations once, then close the complete table."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time
import traceback

from dynamic_training_r34 import BASE, ROOT, digest


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    folder = BASE/'process/exploration-r42/stage001'
    folder.mkdir(parents=True, exist_ok=False)
    parent = Path(__file__).parent
    names = ['run_tlc_r42.py', 'evaluate_tlc_r42.py', 'summarize_tlc_r42.py', 'tlc_pool_r42.py',
             'half_motion_r4.py', 'aligned_raw_history_r11.py', 'aggregation_controls_r20.py',
             'exploration_r2_data.py', 'video_metrics.py', 'prepare_cache_r5.py', 'train_exploration_r2.py']
    frozen = {name: digest(parent/name) for name in names}
    gate = BASE/'process/exploration-r42/cpu-gate.json'
    checked = json.loads(gate.read_text())
    assert checked['passed'] and not checked['cuda_initialized']
    assert checked['source_sha256']['tlc_pool_r42.py'] == frozen['tlc_pool_r42.py']
    write(folder/'config.json', {'created_at': datetime.now().astimezone().isoformat(),
          'source_sha256': frozen, 'gate_sha256': digest(gate), 'new_training_updates': 0,
          'planned_new_quality_rows': 60, 'planned_new_neural_sequences': 20, 'no_wall_cutoff': True,
          'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'scope': 'Known fixed192 TLC on both R41 expanded final4000 seeds; no automatic retry or seed selection.'})
    started, complete, returncodes = time.perf_counter(), False, {}
    try:
        children, streams = {}, {}
        try:
            for seed in [0, 1]:
                command = ['./scripts/project-python', str((parent/'evaluate_tlc_r42.py').relative_to(ROOT)),
                           '--seed', str(seed)]
                streams[seed] = (folder/f'seed{seed}-evaluation.log').open('x')
                children[seed] = subprocess.Popen(command, cwd=ROOT, stdout=streams[seed], stderr=subprocess.STDOUT)
                write(folder/f'seed{seed}-launch.json', {'pid': children[seed].pid, 'command': command,
                      'started_at': datetime.now().astimezone().isoformat()})
            for seed, child in children.items():
                returncodes[seed] = child.wait()
        finally:
            for stream in streams.values():
                stream.close()
        assert set(returncodes) == {0, 1} and all(code == 0 for code in returncodes.values()), returncodes
        assert all(digest(parent/name) == value for name, value in frozen.items()), 'Frozen source changed'
        with (folder/'summarize.log').open('x') as stream:
            subprocess.run(['./scripts/project-python', str((parent/'summarize_tlc_r42.py').relative_to(ROOT))],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        complete = True
    except Exception:
        (folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(folder/'runtime.json', {'completed_normally': complete, 'evaluation_returncodes': returncodes,
              'wall_seconds': time.perf_counter()-started, 'ended_at': datetime.now().astimezone().isoformat(),
              'scope': 'Complete CPU quality stage only; no automatic novelty or phone-cost conclusion.'})


if __name__ == '__main__':
    main()
