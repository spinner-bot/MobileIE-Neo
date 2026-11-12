"""Wait for both actual trainings, evaluate every final seed, render all artifacts."""
import json
import subprocess
import time
import traceback
from datetime import datetime

from dynamic_training_r34 import BASE, ROOT, digest
from finalize_bounded_video_r44 import read_stable
from finalize_preserved_r37 import process_identity
from finalize_task_fastdvd_r47 import wait_training


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    process = BASE/'process/exploration-r56'; out = process/'finalize001'; out.mkdir(exist_ok=False)
    names = ['tiny_task_r56.py', 'train_tiny_task_r56.py', 'check_tiny_task_r56.py', 'tiny_adapted_reference_r56.py',
             'evaluate_tiny_reference_r56.py', 'summarize_tiny_task_r56.py', 'render_tiny_phone_r56.py',
             'finalize_tiny_task_r56.py', 'multinex_reference_r55.py', 'mixed_anchor_r40.py', 'dynamic_training_r34.py',
             'native_training_r14.py', 'half_motion_r4.py', 'aligned_raw_history_r11.py', 'prepare_cache_r5.py',
             'exploration_r2_data.py', 'video_metrics.py', 'finalize_task_fastdvd_r47.py', 'training_form_r41.py', 'gpu_guard.py']
    frozen = {str((BASE/'scripts'/n).relative_to(ROOT)): digest(BASE/'scripts'/n) for n in names}
    for n in ['basicsr/models/archs/MultinexNano_arch.py', 'Options/Multinex-Nano_LOL-v1.yaml', 'UPSTREAM.json']:
        p = ROOT/'third_party/multinex'/n; frozen[str(p.relative_to(ROOT))] = digest(p)
    dependencies = []
    for seed in [0, 1]:
        launch = read_stable(process/f'seed{seed}-launch.json'); guard = read_stable(process/f'seed{seed}-guard.json')
        identity = process_identity(launch['guard_pid'])
        assert guard['allowed'] and ('returncode' in guard or (identity and 'gpu_guard.py' in identity['commandline']))
        dependencies.append(dict(seed=seed, pid=launch['guard_pid'], identity=identity))
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), dependencies=dependencies,
          source_sha256=frozen, protocol_sha256=digest(BASE/'docs/exploration-r56-tiny-task-adaptation-protocol.md'),
          no_wall_cutoff=True, no_automatic_retry=True, selection='both final8000, all fixed paired and phone frames'))
    start, complete, evaluated = time.perf_counter(), False, {}
    streams, children = {}, {}
    try:
        for dep in dependencies:
            wait_training(process/f"seed{dep['seed']}-guard.json", dep['pid'], dep['identity'])
            run = BASE/f"results/exploration-r56-stage001-seed{dep['seed']}"
            runtime, marker = read_stable(run/'runtime.json'), read_stable(run/'nano-step8000.json')
            assert runtime['completed_normally'] and runtime['steps'] == marker['step'] == 8000
            assert marker['final_quality_eligible'] and digest(ROOT/marker['path']) == marker['sha256']
        assert all(digest(ROOT/n) == sha for n, sha in frozen.items())
        for seed in [0, 1]:
            command = ['./scripts/project-python', str(BASE/'scripts/evaluate_tiny_reference_r56.py'), '--seed', str(seed)]
            streams[seed] = (out/f'evaluate-seed{seed}.log').open('x')
            children[seed] = subprocess.Popen(command, cwd=ROOT, stdout=streams[seed], stderr=subprocess.STDOUT)
            write(out/f'evaluate-seed{seed}-launch.json', dict(at=datetime.now().astimezone().isoformat(), pid=children[seed].pid, command=command))
        write(out/'state.json', dict(at=datetime.now().astimezone().isoformat(), stage='both normal8000; final CPU paired and phone evaluations launched'))
        for seed, child in children.items(): evaluated[seed] = child.wait()
        assert len(evaluated) == 2 and all(code == 0 for code in evaluated.values())
        assert all(digest(ROOT/n) == sha for n, sha in frozen.items())
        commands = [('summarize', ['./scripts/project-python', str(BASE/'scripts/summarize_tiny_task_r56.py')])]
        commands += [(f'render-seed{seed}', ['./scripts/project-python', str(BASE/'scripts/render_tiny_phone_r56.py'), '--seed', str(seed)]) for seed in [0, 1]]
        for label, command in commands:
            with (out/f'{label}.log').open('x') as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        assert all(digest(ROOT/n) == sha for n, sha in frozen.items()); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        for stream in streams.values(): stream.close()
        write(out/'runtime.json', dict(completed_normally=complete, evaluation_returncodes=evaluated,
              seconds=time.perf_counter()-start, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    main()
