"""Audit all R5-family GPU launches, shared-group concurrency and source provenance."""
import argparse
from datetime import datetime
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
    root = ROOT/'research/4.temporal-consistency/process'
    paths = sorted(path for name in ['exploration-r5', 'exploration-r5b', 'exploration-r5c']
                   for path in (root/name).glob('gpu-*.json'))
    if len(paths) != 20:
        raise ValueError('expected 20 GPU launches')
    jobs, events, source_cache, mismatches = [], [], {}, []
    for path in paths:
        r = json.loads(path.read_text())
        if not r['allowed'] or len(r['samples']) != 2 or 'ended' not in r:
            raise ValueError('incomplete or rejected job '+str(path))
        gpu = r['samples'][0]['gpu']
        if not 0 <= gpu <= 3 or any(s['compute_processes'] or s['memory_mib'] > 64 or s['utilization_percent'] > 5 for s in r['samples']):
            raise ValueError('resource policy failure')
        expected_failure = path.name in ['gpu-shift-numerics.json', 'gpu-shift-numerics-full.json']
        if r['returncode'] != int(expected_failure):
            raise ValueError('unexpected exit status '+str(path))
        revision = r['repository_revision_at_launch']
        for filename, expected in r['script_sha256_at_launch'].items():
            key = (revision, filename)
            if key not in source_cache:
                content = subprocess.check_output(['git', 'show', revision+':'+filename], cwd=ROOT)
                source_cache[key] = hashlib.sha256(content).hexdigest()
            if source_cache[key] != expected:
                mismatches.append({'job': str(path.relative_to(ROOT)), 'source': filename, 'revision': revision})
        start = datetime.fromisoformat(r['samples'][-1]['time']).timestamp()
        end = datetime.fromisoformat(r['ended']).timestamp()
        events += [(start, 1, gpu, str(path)), (end, -1, gpu, str(path))]
        jobs.append({'record': str(path.relative_to(ROOT)), 'gpu': gpu, 'returncode': r['returncode'],
            'process_wall_seconds': r['process_wall_seconds'], 'revision': revision,
            'expected_numerical_check_failure': expected_failure})
    active, peak = {}, 0
    for _, direction, gpu, job in sorted(events):
        if direction == 1:
            if gpu in active:
                raise ValueError('overlapping jobs on one GPU')
            active[gpu] = job
            peak = max(peak, len(active))
        else:
            del active[gpu]
    if mismatches or peak > 4:
        raise ValueError({'source_mismatches': mismatches, 'peak': peak})
    save_json(args.out, {'jobs': jobs, 'all_preflights_idle': True, 'used_physical_gpus': [0, 1, 2, 3],
        'peak_simultaneous_jobs': peak, 'source_hashes_checked_against_launch_commit': len(source_cache),
        'source_mismatches': mismatches, 'expected_failed_numerical_checks': 2,
        'total_process_gpu_seconds': sum(j['process_wall_seconds'] for j in jobs),
        'scope': 'process wall time includes CPU/data/metrics and is not GPU kernel time; preflight/local locks do not guarantee global exclusive reservation'})


if __name__ == '__main__':
    main()
