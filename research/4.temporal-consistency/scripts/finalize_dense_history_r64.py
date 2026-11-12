"""Follow only the registered live jobs, then final CPU evaluation and summary."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from dynamic_training_r34 import BASE, ROOT, digest
from train_dense_history_r64 import MODELS, write


def stamp():
    return datetime.now().astimezone().isoformat()


def verify_guard(job):
    marker = ROOT/job['guard_record']
    if marker.exists():
        value = json.loads(marker.read_text())
        if not value['allowed']:
            raise RuntimeError(f"GPU admission denied for seed{job['seed']}; no automatic relaunch")
        if 'returncode' in value:
            assert value['returncode'] == 0, value
            return True
    command = (Path('/proc')/str(job['guard_pid'])/'cmdline').read_bytes().split(b'\0')
    decoded = [v.decode() for v in command if v]
    assert str(BASE/'scripts/gpu_guard.py') in decoded
    assert str(marker) in decoded, 'Registered guard identity missing or reused; never auto restart'
    return False


def main():
    out = BASE/'process/exploration-r64/finalize001'
    out.mkdir(exist_ok=False)
    launch = BASE/'process/exploration-r64/launch001.json'
    jobs = json.loads(launch.read_text())['jobs']
    assert sorted(job['seed'] for job in jobs) == [0, 1]
    names = ['finalize_dense_history_r64.py', 'train_dense_history_r64.py',
             'evaluate_dense_history_r64.py', 'summarize_dense_history_r64.py',
             'dense_joint_state_r63.py', 'budget_causal_r33.py', 'mixed_anchor_r40.py',
             'dynamic_training_r34.py', 'training_form_r41.py', 'exploration_r2_data.py',
             'prepare_cache_r5.py', 'half_motion_r4.py', 'video_metrics.py',
             'aligned_raw_history_r11.py', 'aggregation_controls_r20.py', 'gpu_guard.py']
    hashes = {str((BASE/'scripts'/n).relative_to(ROOT)): digest(BASE/'scripts'/n) for n in names}
    hashes.update({str(p.relative_to(ROOT)): digest(p) for p in
                   [ROOT/'MobileIE/model/lle.py', ROOT/'MobileIE/model/utils.py']})
    write(out/'config.json', {'created_at': stamp(), 'launch_sha256': digest(launch), 'jobs': jobs,
        'source_sha256': hashes, 'pid': os.getpid(), 'no_wall_cutoff': True,
        'scope': 'No auto restart, no retraining, no checkpoint selection, no final pixels or paper mutation.'})
    started, complete, phase = time.perf_counter(), False, 'wait-training'
    try:
        with (out/'wait-training.jsonl').open('x') as stream:
            while True:
                states = {job['seed']: verify_guard(job) for job in jobs}
                stream.write(json.dumps({'at': stamp(), 'guard_return0': states})+'\n')
                stream.flush()
                if all(states.values()):
                    break
                time.sleep(15)
        for job in jobs:
            run = BASE/f"results/exploration-r64-stage001-seed{job['seed']}"
            runtime = json.loads((run/'runtime.json').read_text())
            assert runtime['completed_normally'] and runtime['model_updates'] == {n: 4000 for n in MODELS}
            assert runtime['total_new_updates'] == 12000
        assert all(digest(ROOT/p) == sha for p, sha in hashes.items()), 'Frozen dependency changed'
        phase = 'CPU-evaluation'
        evaluations = []
        for job in jobs:
            seed = job['seed']
            command = [str(ROOT/'scripts/project-python'), str(BASE/'scripts/evaluate_dense_history_r64.py'),
                       '--seed', str(seed)]
            log = out/f'evaluate-seed{seed}.log'
            with log.open('x') as stream:
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            evaluations.append((seed, process))
        write(out/'evaluation-launches.json', {'at': stamp(), 'jobs': [
            {'seed': seed, 'pid': process.pid} for seed, process in evaluations]})
        codes = {seed: process.wait() for seed, process in evaluations}
        write(out/'evaluation-returns.json', {'at': stamp(), 'returncodes': codes})
        assert all(code == 0 for code in codes.values()), codes
        assert all(digest(ROOT/p) == sha for p, sha in hashes.items()), 'Frozen dependency changed'
        phase = 'summary'
        with (out/'summary.log').open('x') as stream:
            result = subprocess.run([str(ROOT/'scripts/project-python'),
                str(BASE/'scripts/summarize_dense_history_r64.py')], cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT, check=False)
        assert result.returncode == 0
        audit = json.loads((BASE/'report/exploration-r64-dense-history/audit.json').read_text())
        assert audit['processing_complete'] and audit['new_quality_rows'] == 140
        complete = True
        phase = 'complete'
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'last_phase': phase,
            'seconds': time.perf_counter()-started, 'ended_at': stamp(),
            'scope': 'Terminal only. No automatic retries or success claim from partial traces.'})


if __name__ == '__main__':
    main()
