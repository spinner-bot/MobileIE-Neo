"""Reusable shared-GPU/source audit with explicit expected process outcomes."""
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
    p.add_argument('--process-dir',required=True)
    p.add_argument('--additional-process-dir',action='append',default=[])
    p.add_argument('--completed',type=int,required=True)
    p.add_argument('--denied',type=int,default=0)
    p.add_argument('--failed',type=int,default=0)
    p.add_argument('--out',required=True)
    args = p.parse_args()
    jobs,failed,denied,hashes,events = [],[],[],{},[]
    directories=[args.process_dir,*args.additional_process_dir]
    paths=sorted({path for directory in directories for path in (ROOT/directory).glob('gpu-*.json')})
    for path in paths:
        r = json.loads(path.read_text())
        if len(r['samples'])!=2:
            raise ValueError('two preflight samples required')
        revision = r['repository_revision_at_launch']
        for name,expected in r['script_sha256_at_launch'].items():
            key = (revision,name)
            if key not in hashes:
                hashes[key] = hashlib.sha256(subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)).hexdigest()
            if hashes[key]!=expected:
                raise ValueError('source does not match launch Git')
        gpu = r['samples'][0]['gpu']
        row = {'record':str(path.relative_to(ROOT)),'gpu':gpu,'revision':revision,'allowed':r['allowed']}
        if not r['allowed']:
            if 'returncode' in r:
                raise ValueError('rejected child executed')
            denied.append(row)
            continue
        if not isinstance(r.get('returncode'),int) or 'ended' not in r:
            raise ValueError('incomplete job')
        if any(s['compute_processes'] or s['memory_mib']>64 or s['utilization_percent']>5 for s in r['samples']):
            raise ValueError('occupied GPU admitted')
        row['process_wall_seconds'] = r['process_wall_seconds']
        row['returncode'] = r['returncode']
        (jobs if r['returncode']==0 else failed).append(row)
        events.extend([(datetime.fromisoformat(r['samples'][-1]['time']).timestamp(),1,gpu),
                       (datetime.fromisoformat(r['ended']).timestamp(),-1,gpu)])
    if len(jobs)!=args.completed or len(denied)!=args.denied or len(failed)!=args.failed:
        raise ValueError('job budget mismatch')
    used = sorted({r['gpu'] for r in jobs+failed})
    if len({g//4 for g in used})>1:
        raise ValueError('multiple GPU groups used')
    active,peak = set(),0
    for _,direction,gpu in sorted(events):
        if direction==1:
            if gpu in active:
                raise ValueError('same GPU concurrent project jobs')
            active.add(gpu); peak=max(peak,len(active))
        else:
            active.remove(gpu)
    if peak>4:
        raise ValueError('concurrency limit exceeded')
    save_json(args.out,{'process_directories':directories,'completed':jobs,'failed':failed,'resource_refusals':denied,'used_physical_gpus':used,
        'peak_own_concurrent_jobs':peak,'source_hashes_verified':len(hashes),'source_mismatches':[],
        'process_gpu_seconds':sum(r['process_wall_seconds'] for r in jobs+failed),
        'scope':'includes CPU/data/metrics; not GPU kernel time; preflight and local locks are not a global reservation'})


if __name__=='__main__':
    main()
