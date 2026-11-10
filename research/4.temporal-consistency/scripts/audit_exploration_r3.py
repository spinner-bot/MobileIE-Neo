"""Final resource/row accounting, no GPU work and no silent unfinished jobs."""
import ast
import datetime
import json
from pathlib import Path

from explore_temporal_controls import save_json
from summarize_r3 import BASE


def main():
    process = BASE/'process/exploration-r3'
    jobs, events = [], []
    for path in sorted(process.glob('gpu*.json')):
        row = json.loads(path.read_text())
        if row['allowed']:
            assert row['returncode'] == 0, path
            assert all(not s['compute_processes'] and s['memory_mib'] <= 64 and s['utilization_percent'] <= 5 for s in row['samples'])
            end = datetime.datetime.fromisoformat(row['ended']).timestamp()
            start = end-row['process_wall_seconds']
            events.extend([(start, 1), (end, -1)])
        jobs.append({'record': path.name, 'gpu': row['samples'][0]['gpu'], 'allowed': row['allowed'],
                     'process_wall_seconds': row.get('process_wall_seconds', 0.), 'returncode': row.get('returncode')})
    active = peak = 0
    for _, delta in sorted(events):
        active += delta
        peak = max(peak, active)
    used = {j['gpu'] for j in jobs if j['allowed']}
    assert used <= {0, 1, 2, 3} and peak <= 4
    resource = {'jobs': jobs, 'allowed_jobs': sum(j['allowed'] for j in jobs),
                'denied_jobs': sum(not j['allowed'] for j in jobs), 'peak_concurrent_project_jobs': peak,
                'per_gpu_process_seconds': {str(g): sum(j['process_wall_seconds'] for j in jobs if j['gpu'] == g) for g in sorted(used)},
                'total_gpu_process_seconds': sum(j['process_wall_seconds'] for j in jobs),
                'scope': 'child process elapsed, including CPU work/startup; not active CUDA kernel time. Overlap inferred from end timestamp and monotonic duration.'}
    save_json(process/'resource-accounting.json', resource)
    files = list((BASE/'scripts').glob('*.py'))+list((BASE/'tests').glob('*.py'))
    for path in files:
        ast.parse(path.read_text(), filename=str(path))
    counts = {'reset_rows': 720, 'frozen_product_rows': 720, 'trained_validation_rows': 1344,
              'actual_motion_synthetic_rows': 352, 'real_paired_rows': 84,
              'real_flow_diagnostic_rows': 14, 'real_enhanced_motion_rows': 84}
    save_json(process/'final-accounting.json', {'row_counts': counts, 'total_method_sequence_rows': sum(counts.values()),
        'repeated_method_input_model_identities': 288+48+192+4+22,
        'new_method_input_model_identities': sum(counts.values())-(288+48+192+4+22),
        'unique_video_input_conditions': 48+8+2, 'feature_channel_statistics_rows': 1152,
        'training_runs': 8, 'updates': 3200, 'parsed_python_files': len(files),
        'note': 'identities include deterministic repeat audits; 192 R2-task repeats use different batch execution, audit numerical differences separately; no claim rows are independent videos'})
    print(json.dumps({'allowed_jobs': resource['allowed_jobs'], 'denied_jobs': resource['denied_jobs'],
        'peak_jobs': peak, 'process_seconds': resource['total_gpu_process_seconds'], 'parsed_files': len(files),
        'total_rows': sum(counts.values())}, indent=2))


if __name__ == '__main__':
    main()
