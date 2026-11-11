"""Wait without a deadline, then delegate the final idle check to gpu_guard.

No CUDA context or project lock is held while another user's job owns the GPU.
This is advisory polling, not a server-wide reservation system.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from gpu_guard import sample


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--record-prefix',required=True)
    parser.add_argument('--poll-seconds',type=float,default=30)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args();command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command or args.poll_seconds<10:parser.error('command required; polling interval must be >=10 seconds')
    prefix=Path(args.record_prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    observations=prefix.with_name(prefix.name+'-waiting.jsonl')
    if observations.exists():raise FileExistsError(observations)
    attempt=0
    while True:
        status=sample(args.gpu)
        with observations.open('a') as stream:stream.write(json.dumps(status)+'\n')
        idle=not status['compute_processes'] and status['memory_mib']<=64 and status['utilization_percent']<=5
        print(json.dumps({'gpu':args.gpu,'idle':idle,'time':status['time'],'guard_attempts':attempt}),flush=True)
        if idle:
            attempt+=1;record=prefix.with_name(prefix.name+f'-attempt{attempt:03d}.json')
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('gpu_guard.py')),
                '--gpu',str(args.gpu),'--record',str(record),'--',*command])
            audit=json.loads(record.read_text())
            if audit['allowed']:
                # A launched job failure is terminal; never repeat a partially executed experiment.
                raise SystemExit(result.returncode)
        time.sleep(args.poll_seconds)


if __name__=='__main__':main()
