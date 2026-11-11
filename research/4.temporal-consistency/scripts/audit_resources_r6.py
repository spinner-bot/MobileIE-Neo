"""Verify one real resource refusal and three completed one-card R6 diagnoses."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    paths = sorted((ROOT/'research/4.temporal-consistency/process/exploration-r6').glob('gpu-*.json'))
    if len(paths) != 4:
        raise ValueError('expected one refusal and three diagnoses')
    denied, jobs, checked = [], [], {}
    for path in paths:
        r = json.loads(path.read_text())
        if len(r['samples']) != 2:
            raise ValueError('two idle preflights required')
        revision = r['repository_revision_at_launch']
        for name, expected in r['script_sha256_at_launch'].items():
            key = (revision, name)
            if key not in checked:
                checked[key] = hashlib.sha256(subprocess.check_output(['git', 'show', revision+':'+name], cwd=ROOT)).hexdigest()
            if checked[key] != expected:
                raise ValueError('launch source not committed: '+name)
        row = {'record': str(path.relative_to(ROOT)), 'allowed': r['allowed'], 'gpu': r['samples'][0]['gpu'],
               'revision': revision, 'process_wall_seconds': r.get('process_wall_seconds')}
        if r['allowed']:
            if r.get('returncode') != 0 or row['gpu'] != 2:
                raise ValueError('unexpected run status/GPU')
            if any(s['compute_processes'] or s['memory_mib'] > 64 or s['utilization_percent'] > 5 for s in r['samples']):
                raise ValueError('occupied card admitted')
            jobs.append(row)
        else:
            if 'returncode' in r:
                raise ValueError('rejected command executed')
            denied.append(row)
    if len(denied) != 1 or len(jobs) != 3:
        raise ValueError('budget mismatch')
    save_json(args.out, {'completed': jobs, 'resource_refusals': denied,
        'source_hashes_verified': len(checked), 'source_mismatches': [],
        'used_gpu': 2, 'peak_own_concurrent_gpu_jobs': 1,
        'process_gpu_seconds': sum(r['process_wall_seconds'] for r in jobs),
        'scope': 'process duration includes CPU/data work; no kernel-time or global-reservation claim'})


if __name__ == '__main__':
    main()
