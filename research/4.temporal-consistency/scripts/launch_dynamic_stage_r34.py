"""Wait for the registered download, then run gated fixed-budget R34 stages."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from dynamic_training_r34 import BASE, MANIFEST, ROOT, digest
from gpu_guard import sample

PYTHON = str(ROOT/'scripts/project-python')
SCRIPTS = BASE/'scripts'


def now():
    return datetime.datetime.now().astimezone().isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-pid', type=int, required=True)
    parser.add_argument('--stage-id', type=int, required=True)
    args = parser.parse_args()
    assert args.stage_id >= 1
    out = BASE/f'process/exploration-r34/stage{args.stage_id:03d}'
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    critical = ['budget_causal_r33.py', 'dynamic_training_r34.py', 'train_dynamic_r34.py',
                'verify_dynamic_training_r34.py', 'check_dynamic_training_r34.py',
                'check_budget_causal_cuda_r33.py', 'evaluate_dynamic_r34.py']
    hashes = {name: digest(SCRIPTS/name) for name in critical}
    config = {'started': now(), 'download_pid': args.download_pid, 'script_sha256': hashes,
              'protocol_sha256': digest(BASE/'docs/exploration-r34-training-protocol.md'),
              'gpu_group': [0, 1, 2, 3], 'gpu_preference': [3, 2, 1, 0], 'concurrent_training_jobs': 1,
              'retry_scope': 'wait for idle only; failed actual commands are never silently restarted',
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')

    def event(stage, **extra):
        row = {'time': now(), 'stage': stage, **extra}
        with (out/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)

    def unchanged():
        assert all(digest(SCRIPTS/name) == value for name, value in hashes.items()), 'Source changed while queued; inspect before relaunch'
        assert digest(BASE/'docs/exploration-r34-training-protocol.md') == config['protocol_sha256']

    def command(label, filename, *arguments):
        unchanged()
        event(label, status='starting')
        with (out/(label+'.log')).open('w') as log:
            subprocess.run([PYTHON, str(SCRIPTS/filename), *arguments], cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        event(label, status='finished')

    def guarded(label, filename, *arguments):
        attempts = 0
        while True:
            unchanged()
            snapshots = [sample(index) for index in [3, 2, 1, 0]]
            idle = [s for s in snapshots if not s['compute_processes'] and s['memory_mib'] <= 64 and s['utilization_percent'] <= 5]
            if not idle:
                event(label, status='waiting_for_idle', samples=snapshots)
                time.sleep(60)
                continue
            attempts += 1
            record = out/f'{label}-guard{attempts:03d}.json'
            with (out/f'{label}-attempt{attempts:03d}.log').open('w') as log:
                result = subprocess.run([PYTHON, str(SCRIPTS/'gpu_guard.py'), '--gpu', str(idle[0]['gpu']),
                        '--record', str(record), '--', PYTHON, str(SCRIPTS/filename), *arguments],
                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            audit = json.loads(record.read_text())
            if not audit['allowed']:
                event(label, status='admission_changed_waiting', record=str(record.relative_to(ROOT)))
                time.sleep(60)
                continue
            assert result.returncode == 0, f'{label} actual command failed; inspect {record}'
            event(label, status='finished', record=str(record.relative_to(ROOT)))
            return

    complete = False
    try:
        proc = Path(f'/proc/{args.download_pid}/cmdline')
        if proc.exists():
            assert 'fetch_davis_training.py' in proc.read_bytes().decode()
            event('download', status='waiting_for_existing_registered_process')
            while proc.exists():
                time.sleep(10)
        assert MANIFEST.is_file(), 'Download ended without final manifest; no training launched'
        unchanged()
        event('download', status='terminal_manifest_present')
        decode = BASE/'results/exploration-r32-decode/audit.json'
        if decode.exists():
            audit = json.loads(decode.read_text())
            assert audit['passed'] and audit['manifest_sha256'] == digest(MANIFEST)
            assert audit['frames'] == 3428 and audit['videos'] == 57
            event('decode', status='reuse_completed_manifest_matched_audit', audit_sha256=digest(decode))
        else:
            command('decode', 'verify_dynamic_training_r34.py')
        interface = BASE/'report/exploration-r33-prototype/training-interface-registered-data.json'
        if interface.exists():
            audit = json.loads(interface.read_text())
            assert audit['initial_spatial_causal_output_equal'] and audit['optimizer_updates'] == 0
            assert audit['batch_shape'] == [4, 4, 3, 128, 128]
            required = {'budget_causal_r33.py', 'dynamic_training_r34.py', 'train_dynamic_r34.py', 'check_dynamic_training_r34.py'}
            assert set(audit['source_sha256']) == required
            assert all(digest(SCRIPTS/name) == value for name, value in audit['source_sha256'].items())
            event('data-interface', status='reuse_completed_source_matched_audit', audit_sha256=digest(interface))
        else:
            command('data-interface', 'check_dynamic_training_r34.py', '--registered-data')
        guarded('cuda-backward', 'check_budget_causal_cuda_r33.py')
        for seed in [0, 1]:
            guarded(f'train-seed{seed}', 'train_dynamic_r34.py', '--seed', str(seed))
        for seed in [0, 1]:
            command(f'evaluate-seed{seed}', 'evaluate_dynamic_r34.py', '--seed', str(seed))
        complete = True
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        event('stage', status='failed_no_automatic_restart')
        raise
    finally:
        (out/'runtime.json').write_text(json.dumps({'completed_normally': complete, 'ended': now()}, indent=2)+'\n')


if __name__ == '__main__':
    main()
