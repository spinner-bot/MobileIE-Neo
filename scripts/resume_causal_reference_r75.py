"""Explicit resource-only recovery from the latest committed checkpoint; no scientific retry."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research/4.temporal-consistency/scripts'))
from causal_reference_io_r75 import PROCESS, seed_root, digest, stamp, write, verify_training_sources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=[0,1],required=True)
    parser.add_argument('--gpu',type=int,choices=[0,1,2,3],required=True)
    parser.add_argument('--recovery',type=int,required=True)
    args=parser.parse_args()
    verify_training_sources()
    root=seed_root(args.seed)
    if (root/'final.json').exists():
        raise SystemExit('Final training exists; do not recover or repeat it.')
    # The lock proves no same-seed child is currently alive, independent of stale status.
    with (ROOT/f'.cache/rebotnet-r75-seed{args.seed}.lock').open('a') as check:
        fcntl.flock(check,fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempts=sorted(root.glob('attempt[0-9][0-9][0-9]'))
    if not attempts:
        raise SystemExit('No original attempt; use original launch.')
    previous=attempts[-1]
    runtime=json.loads((previous/'runtime.json').read_text())
    if runtime['stop_reason']!='resource' or runtime['completed_normally']:
        raise SystemExit('Only an authoritative terminal resource exit can be resumed.')
    latest=runtime['latest_checkpoint']
    marker_path=None
    if latest is not None:
        marker_path=previous/f'{latest["kind"]}-step{latest["completed_steps"]:06d}.json'
        marker=json.loads(marker_path.read_text())
        if marker!=latest or marker['completed_steps']!=runtime['completed_steps']:
            raise SystemExit('Recovery must use the exact terminal latest committed state, not an older checkpoint.')
        if digest(ROOT/marker['path'])!=marker['sha256']:
            raise SystemExit('Latest checkpoint payload changed.')
    elif runtime['completed_steps']!=0:
        raise SystemExit('Committed updates exist without a recovery checkpoint.')
    attempt=int(previous.name[-3:])+1
    name=f'recovery{args.recovery:03d}-seed{args.seed}'
    launch=PROCESS/f'{name}-launch.json'
    parent=PROCESS/f'{name}-parent.json'
    log=PROCESS/f'{name}.log'
    if any(p.exists() for p in (launch,parent,log)):
        raise SystemExit('Recovery identifier exists; inspect its real handle before another action.')
    command=[str(ROOT/'scripts/project-python'),str(ROOT/'scripts/prebind_shared_gpu.py'),
             '--gpu',str(args.gpu),'--record',str(parent),'--',str(ROOT/'scripts/project-python'),
             str(ROOT/'research/4.temporal-consistency/scripts/train_causal_reference_r75.py'),
             '--seed',str(args.seed),'--attempt',str(attempt)]
    if marker_path is not None:
        command+=['--resume',str(marker_path)]
    with log.open('x') as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(at=stamp(),parent_pid=child.pid,
                parent_start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21],
                previous_terminal=str((previous/'runtime.json').relative_to(ROOT)),
                authoritative_committed_steps=runtime['completed_steps'],resume=latest,
                attempt=attempt,command=command,parent_record=str(parent.relative_to(ROOT)),
                scope='One explicitly requested resource recovery admission; no automatic retries or older-state replay.')
    write(launch,record)
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
