"""Capacity-based resource sampling; no CUDA initialization or global changes."""
from datetime import datetime
import json
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def snapshot():
    rows = subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    processes = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
        '--format=csv,noheader,nounits'], text=True).splitlines()
    result = []
    for row in rows.splitlines():
        index, uuid, total, used, util = [s.strip() for s in row.split(',')]
        result.append(dict(index=int(index), uuid=uuid, total_mib=int(total), used_mib=int(used),
            utilization_percent=int(util), processes=[p for p in processes if uuid in p]))
    return dict(at=datetime.now().astimezone().isoformat(), GPUs=result)


def assess(samples, expected_increment_mib=4096):
    candidates = []
    for first in samples[0]['GPUs']:
        index, uuid = first['index'], first['uuid']
        rows = [next(r for r in s['GPUs'] if r['uuid'] == uuid) for s in samples]
        used = [r['used_mib'] for r in rows]
        util = [r['utilization_percent'] for r in rows]
        reserve = max(8192, 2*(max(used)-min(used)))
        required_free = expected_increment_mib+reserve+1024
        mean = statistics.mean(util)
        busy_fraction = sum(v >= 90 for v in util)/len(util)
        allowed = (first['total_mib']-max(used) >= required_free
                   and mean <= 60 and busy_fraction <= .25)
        candidates.append(dict(index=index, uuid=uuid, allowed=allowed,
            mean_utilization_percent=mean, busy_sample_fraction=busy_fraction,
            max_used_mib=max(used), minimum_free_mib=first['total_mib']-max(used),
            total_mib=first['total_mib'], incumbent_growth_reserve_mib=reserve,
            expected_own_increment_mib=expected_increment_mib, required_free_mib=required_free))
    eligible = sorted((r for r in candidates if r['allowed']),
                      key=lambda r: (r['mean_utilization_percent'], -r['minimum_free_mib']))
    return dict(candidates=candidates, selected=eligible[0] if eligible else None,
                scope='Observed headroom, not a reservation or incumbent performance guarantee.')


def sustained_samples():
    samples = []
    for index in range(11):
        if index:
            time.sleep(2)
        samples.append(snapshot())
    return samples


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
