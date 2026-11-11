"""Resource provenance and final artifact totals; unfinished jobs fail the audit."""
import ast
import datetime
import json
from pathlib import Path

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json
from summarize_r3 import BASE


def main():
    process = BASE/'process/exploration-r4'
    jobs, events = [], []
    for path in sorted(process.glob('gpu*.json')):
        data = json.loads(path.read_text())
        if data['allowed']:
            assert data['returncode'] == 0, path
            assert all(not s['compute_processes'] and s['memory_mib'] <= 64 and s['utilization_percent'] <= 5 for s in data['samples'])
            end = datetime.datetime.fromisoformat(data['ended']).timestamp()
            events.extend([(end-data['process_wall_seconds'], 1), (end, -1)])
        jobs.append({'record': path.name, 'gpu': data['samples'][0]['gpu'], 'allowed': data['allowed'],
                     'seconds': data.get('process_wall_seconds', 0.), 'returncode': data.get('returncode')})
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    assert peak <= 4
    used = {r['gpu'] for r in jobs if r['allowed']}
    assert used <= {4, 5, 6, 7}
    summary = {'jobs': jobs, 'allowed_jobs': sum(r['allowed'] for r in jobs), 'denied_jobs': sum(not r['allowed'] for r in jobs),
        'peak_project_gpu_jobs': peak, 'total_gpu_process_seconds': sum(r['seconds'] for r in jobs),
        'per_gpu_process_seconds': {str(g): sum(r['seconds'] for r in jobs if r['gpu'] == g) for g in sorted(used)},
        'scope': 'includes CPU preparation/validation and startup; not active CUDA time. Overlap inferred from end timestamp and monotonic duration.'}
    save_json(process/'resource-accounting.json', summary)
    a = json.loads((BASE/'data/exploration-r4/manifest.json').read_text())
    b = json.loads((BASE/'data/exploration-r4b/manifest.json').read_text())
    old_test = {(r['source_path'], r['sha256']) for r in a['files'] if r['split'] == 'test'}
    new_test = {(r['source_path'], r['sha256']) for r in b['files'] if r['split'] == 'test'}
    assert old_test == new_test
    train = {r['sha256'] for r in b['files'] if r['split'] == 'train'}
    assert not train & {digest for _, digest in new_test}
    files = list((BASE/'scripts').glob('*.py'))+list((BASE/'tests').glob('*.py'))
    for path in files:
        ast.parse(path.read_text(), filename=str(path))
    evidence = json.loads((BASE/'report/exploration-r4/audit.json').read_text())
    assert len(evidence['training_runs']) == 14 and evidence['updates'] == 5600
    assert evidence['evaluation_rows'] == 770 and evidence['calibration_rows'] == 248
    counts = {'reliability': 120, 'alpha': 64, 'noise-gate': 32, 'half': 32}
    for name, expected in counts.items():
        assert len(json.loads((BASE/f'results/exploration-r4-{name}/calibration_trials.json').read_text())) == expected
    timed = 0
    for name in ['pipeline-cost', 'half-cost']:
        values = json.loads((BASE/f'results/exploration-r4-{name}.json').read_text())
        for records in values['per_frame'].values():
            assert len(records) == 45
            assert all(r['total_ms'] >= r['model_transfer_sync_ms'] >= 0 for r in records)
            timed += len(records)
    save_json(process/'final-accounting.json', {'neural_training_runs': 14, 'updates': 5600,
        'evaluation_rows': 770, 'calibration_rows': 248, 'scale_fit_diagnostic_rows': 180,
        'total_method_sequence_rows': 1198, 'evaluation_input_conditions': 14,
        'real_test_clips': 6, 'new_real_test_clips': 5, 'legacy_synthetic_conditions': 8, 'legacy_source_clips': 2,
        'real_training_clips_max': 12, 'paired_source_files': len(b['files']), 'source_bytes': b['integrity']['bytes'],
        'test_files_unchanged': True, 'train_test_file_hash_overlap': 0, 'parsed_python_files': len(files),
        'calibration_counts_verified': counts, 'timed_pipeline_frame_samples': timed,
        'note': 'repeated controls/conditions included; pixels, seeds, related clips and rows are not independent videos'})
    print(json.dumps({'jobs': len(jobs), 'allowed': summary['allowed_jobs'], 'peak_jobs': peak,
        'gpu_process_seconds': summary['total_gpu_process_seconds'], 'python_files': len(files), 'method_sequence_rows': 1198}, indent=2))


if __name__ == '__main__':
    main()
