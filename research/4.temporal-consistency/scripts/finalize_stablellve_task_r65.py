"""Follow the actual admitted jobs to completion; evaluate every final arm once."""
import json
import subprocess
import time
import traceback
from datetime import datetime

from check_stablellve_task_r65 import write
from dynamic_training_r34 import BASE, ROOT, digest
from finalize_bounded_video_r44 import read_stable
from finalize_preserved_r37 import process_identity
from finalize_task_fastdvd_r47 import wait_training
from stablellve_task_r65 import ARMS


def main():
    process = BASE/'process/exploration-r65'; out = process/'finalize001'; out.mkdir(exist_ok=False)
    names = ['stablellve_task_r65.py', 'check_stablellve_task_r65.py', 'train_stablellve_task_r65.py',
             'stablellve_adapted_r65.py', 'evaluate_stablellve_task_r65.py', 'summarize_stablellve_task_r65.py',
             'finalize_stablellve_task_r65.py', 'stablellve_adapter_r15.py', 'mixed_anchor_r40.py',
             'dynamic_training_r34.py', 'native_training_r14.py', 'training_form_r41.py', 'gpu_guard.py',
             'half_motion_r4.py', 'aligned_raw_history_r11.py', 'drv_references_r59.py', 'prepare_cache_r5.py',
             'exploration_r2_data.py', 'video_metrics.py', 'finalize_task_fastdvd_r47.py']
    frozen = {str((BASE/'scripts'/name).relative_to(ROOT)): digest(BASE/'scripts'/name) for name in names}
    for path in [ROOT/'third_party/stablellve/model.py', BASE/'docs/exploration-r65-stablellve-task-adaptation-protocol.md']:
        frozen[str(path.relative_to(ROOT))] = digest(path)
    dependencies = []
    for seed in [0, 1]:
        launch = read_stable(process/f'seed{seed}-launch.json'); guard = read_stable(process/f'seed{seed}-guard.json')
        identity = process_identity(launch['guard_pid'])
        assert guard['allowed'] and ('returncode' in guard or (identity and 'gpu_guard.py' in identity['commandline']))
        dependencies.append(dict(seed=seed, pid=launch['guard_pid'], identity=identity))
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), dependencies=dependencies,
          source_sha256=frozen, no_wall_cutoff=True, no_automatic_retry=True,
          scope='All four final8000 checkpoints only; common paired CPU evaluation and complete summary, no additional training.'))
    start, complete, evaluated = time.perf_counter(), False, {}
    children, streams = {}, {}
    try:
        for dep in dependencies:
            wait_training(process/f'seed{dep["seed"]}-guard.json', dep['pid'], dep['identity'])
            run = BASE/f'results/exploration-r65-stage001-seed{dep["seed"]}'
            runtime = read_stable(run/'runtime.json')
            assert runtime['completed_normally'] and runtime['updates'] == {arm: 8000 for arm in ARMS}
            for arm in ARMS:
                marker = read_stable(run/f'{arm}-step8000.json')
                assert marker['step'] == 8000 and marker['final_quality_eligible']
                assert digest(ROOT/marker['path']) == marker['sha256']
        assert all(digest(ROOT/path) == sha for path, sha in frozen.items())
        for seed in [0, 1]:
            command = ['./scripts/project-python', str((BASE/'scripts/evaluate_stablellve_task_r65.py').relative_to(ROOT)),
                       '--seed', str(seed)]
            streams[seed] = (out/f'evaluate-seed{seed}.log').open('x')
            children[seed] = subprocess.Popen(command, cwd=ROOT, stdout=streams[seed], stderr=subprocess.STDOUT)
            write(out/f'evaluate-seed{seed}-launch.json', dict(at=datetime.now().astimezone().isoformat(),
                  pid=children[seed].pid, command=command))
        write(out/'state.json', dict(at=datetime.now().astimezone().isoformat(), stage='Both normal final8000; final CPU evaluations actually launched.'))
        for seed, child in children.items(): evaluated[seed] = child.wait()
        assert len(evaluated) == 2 and all(code == 0 for code in evaluated.values()), evaluated
        assert all(digest(ROOT/path) == sha for path, sha in frozen.items())
        with (out/'summarize.log').open('x') as stream:
            subprocess.run(['./scripts/project-python', str((BASE/'scripts/summarize_stablellve_task_r65.py').relative_to(ROOT))],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        for stream in streams.values(): stream.close()
        write(out/'runtime.json', dict(completed_normally=complete, evaluation_returncodes=evaluated,
              seconds=time.perf_counter()-start, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
