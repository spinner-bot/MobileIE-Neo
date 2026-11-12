"""Wait for the two admitted fixed-budget trainings; run all final CPU references."""
import json
import subprocess
import time
import traceback
from datetime import datetime

from dynamic_training_r34 import BASE, ROOT, digest
from finalize_bounded_video_r44 import read_stable
from finalize_preserved_r37 import process_identity


def stamp():
    return datetime.now().astimezone().isoformat()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    process = BASE/'process/exploration-r53'
    folder = process/'finalize001'
    folder.mkdir(exist_ok=False)
    names = ['source_context_r53.py', 'check_source_context_r53.py', 'train_source_context_r53.py',
             'evaluate_source_context_r53.py', 'summarize_source_context_r53.py', 'finalize_source_context_r53.py',
             'task_fastdvd_r47.py', 'mixed_anchor_r40.py', 'dynamic_training_r34.py', 'native_training_r14.py',
             'fastdvdnet_adapter_r12.py', 'training_form_r41.py', 'gpu_guard.py',
             'exploration_r2_data.py', 'video_metrics.py', 'prepare_cache_r5.py']
    frozen = {name: digest(BASE/'scripts'/name) for name in names}
    dependencies = []
    for seed in [0, 1]:
        launch = read_stable(process/f'seed{seed}-launch.json')
        guard = read_stable(process/f'seed{seed}-guard.json')
        assert guard['allowed']
        identity = process_identity(launch['guard_pid'])
        assert 'returncode' in guard or (identity and 'gpu_guard.py' in identity['commandline'])
        dependencies.append(dict(seed=seed, pid=launch['guard_pid'], identity=identity))
    write(folder/'config.json', dict(started=stamp(), dependencies=dependencies, source_sha256=frozen,
          protocol_sha256=digest(BASE/'docs/exploration-r53-source-context-protocol.md'),
          no_wall_cutoff=True, no_automatic_retry=True, selection='final8000 both seeds, both information modes'))
    start, complete, evaluated = time.perf_counter(), False, {}
    children, streams = {}, {}
    try:
        for dep in dependencies:
            while True:
                guard = read_stable(process/f"seed{dep['seed']}-guard.json")
                if 'returncode' in guard:
                    assert guard['returncode'] == 0, guard
                    break
                if process_identity(dep['pid']) != dep['identity']:
                    time.sleep(1)
                    assert read_stable(process/f"seed{dep['seed']}-guard.json").get('returncode') == 0
                    break
                time.sleep(5)
            run = BASE/f"results/exploration-r53-stage001-seed{dep['seed']}"
            runtime, marker = read_stable(run/'runtime.json'), read_stable(run/'fastdvd-step8000.json')
            assert runtime['completed_normally'] and runtime['steps'] == marker['step'] == 8000
            assert marker['final_quality_eligible'] and digest(ROOT/marker['path']) == marker['sha256']
        assert all(digest(BASE/'scripts'/name) == sha for name, sha in frozen.items())
        for seed in [0, 1]:
            command = ['./scripts/project-python', str(BASE/'scripts/evaluate_source_context_r53.py'), '--seed', str(seed)]
            streams[seed] = (folder/f'evaluate-seed{seed}.log').open('x')
            children[seed] = subprocess.Popen(command, cwd=ROOT, stdout=streams[seed], stderr=subprocess.STDOUT)
            write(folder/f'evaluate-seed{seed}-launch.json', dict(at=stamp(), pid=children[seed].pid, command=command))
        write(folder/'state.json', dict(at=stamp(), stage='both normal8000; final CPU evaluations launched'))
        for seed, child in children.items():
            evaluated[seed] = child.wait()
        assert all(code == 0 for code in evaluated.values()) and len(evaluated) == 2
        assert all(digest(BASE/'scripts'/name) == sha for name, sha in frozen.items())
        with (folder/'summarize.log').open('x') as stream:
            subprocess.run(['./scripts/project-python', str(BASE/'scripts/summarize_source_context_r53.py')],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        complete = True
    except BaseException:
        (folder/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        for stream in streams.values():
            stream.close()
        write(folder/'runtime.json', dict(completed_normally=complete, evaluation_returncodes=evaluated,
              seconds=time.perf_counter()-start, ended=stamp()))


if __name__ == '__main__':
    main()
