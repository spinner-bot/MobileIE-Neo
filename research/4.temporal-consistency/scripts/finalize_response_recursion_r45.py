"""Wait for the identified R45 fixed budgets, preserving each coordinator attempt."""
import argparse
from datetime import datetime
import json
import subprocess
import time
import traceback

from dynamic_training_r34 import BASE, ROOT, digest
from finalize_bounded_video_r44 import read_stable
from finalize_preserved_r37 import process_identity
from train_response_recursion_r45 import LEARNED


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args()
    folder = BASE/f'process/exploration-r45/stage001/finalize{args.attempt:03d}'
    folder.mkdir(parents=True, exist_ok=False)
    names = ['finalize_response_recursion_r45.py', 'train_response_recursion_r45.py',
             'evaluate_response_recursion_r45.py', 'summarize_response_recursion_r45.py',
             'response_cue_r45.py', 'response_sequence_r45.py', 'budget_causal_r33.py',
             'half_motion_r4.py', 'aligned_raw_history_r11.py', 'aggregation_controls_r20.py',
             'exploration_r2_data.py', 'video_metrics.py', 'prepare_cache_r5.py',
             'dynamic_training_r34.py', 'finalize_preserved_r37.py', 'finalize_bounded_video_r44.py']
    frozen = {n: digest(BASE/'scripts'/n) for n in names}
    dependencies = []
    for seed in [0, 1]:
        launch = read_stable(BASE/f'process/exploration-r45/stage001/seed{seed}-launch.json')
        guard = read_stable(BASE/f'process/exploration-r45/stage001/seed{seed}-guard.json')
        assert guard['allowed']
        identity = process_identity(launch['guard_pid'])
        assert 'returncode' in guard or (identity and 'gpu_guard.py' in identity['commandline'])
        dependencies.append({'seed': seed, 'pid': launch['guard_pid'], 'identity': identity})
    write(folder/'config.json', {'started_at': datetime.now().astimezone().isoformat(),
          'dependencies': dependencies, 'source_sha256': frozen,
          'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'scope': 'Both identified trainings complete 12000 new controller updates each, frozen base unchanged, before 180 final CPU rows. No auto retry/checkpoint selection or wall cutoff.'})
    started, complete, evaluated = time.perf_counter(), False, {}
    try:
        for dep in dependencies:
            path = BASE/f"process/exploration-r45/stage001/seed{dep['seed']}-guard.json"
            while True:
                guard = read_stable(path)
                if 'returncode' in guard:
                    assert guard['returncode'] == 0
                    runtime = read_stable(BASE/f"results/exploration-r45-stage001-seed{dep['seed']}/runtime.json")
                    assert runtime['completed_normally'] and runtime['frozen_base_unchanged']
                    assert runtime['total_new_controller_updates'] == 12000
                    assert runtime['model_updates'] == {n: 4000 for n in LEARNED}
                    assert set(runtime['final_checkpoints']) == set(LEARNED)
                    assert all(v['new_step'] == 4000 for v in runtime['final_checkpoints'].values())
                    break
                assert process_identity(dep['pid']) == dep['identity'], 'Identified guard process no longer matches'
                time.sleep(2)
        assert all(digest(BASE/'scripts'/n) == value for n, value in frozen.items())
        children, streams = {}, {}
        try:
            for seed in [0, 1]:
                command = ['./scripts/project-python', 'research/4.temporal-consistency/scripts/evaluate_response_recursion_r45.py',
                           '--seed', str(seed)]
                streams[seed] = (folder/f'evaluate-seed{seed}.log').open('x')
                child = subprocess.Popen(command, cwd=ROOT, stdout=streams[seed], stderr=subprocess.STDOUT)
                children[seed] = child
                write(folder/f'evaluate-seed{seed}-launch.json', {'pid': child.pid, 'command': command,
                      'started_at': datetime.now().astimezone().isoformat()})
            for seed, child in children.items():
                evaluated[seed] = child.wait()
        finally:
            for stream in streams.values():
                stream.close()
        assert set(evaluated) == {0, 1} and all(v == 0 for v in evaluated.values()), evaluated
        with (folder/'summarize.log').open('x') as stream:
            subprocess.run(['./scripts/project-python', 'research/4.temporal-consistency/scripts/summarize_response_recursion_r45.py'],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        complete = True
    except Exception:
        (folder/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        write(folder/'runtime.json', {'completed_normally': complete, 'evaluation_returncodes': evaluated,
              'wall_seconds': time.perf_counter()-started, 'ended_at': datetime.now().astimezone().isoformat(),
              'scope': 'Quality closure only. Independent cue screen, alpha dominance and costs remain distinct.'})


if __name__ == '__main__':
    main()
