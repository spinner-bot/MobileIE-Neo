"""Wait without a CUDA context, using any idle GPU in one explicitly bounded group."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from gpu_guard import sample


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', required=True)
    p.add_argument('--record-prefix', required=True)
    p.add_argument('--poll-seconds', type=float, default=10)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    gpus = [int(v) for v in args.gpus.split(',')]
    if not 1 <= len(gpus) <= 4 or len(set(gpus)) != len(gpus) or min(gpus) < 0 or len({g//4 for g in gpus}) != 1:
        p.error('one unique group of at most four GPU indices required')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.poll_seconds < 10:
        p.error('command required; polling interval >=10 seconds')
    prefix = Path(args.record_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    observations = prefix.with_name(prefix.name+'-waiting.jsonl')
    if observations.exists():
        raise FileExistsError(observations)
    attempt = 0
    while True:
        samples = [sample(g) for g in gpus]
        idle = [r['gpu'] for r in samples if not r['compute_processes'] and r['memory_mib'] <= 64 and r['utilization_percent'] <= 5]
        with observations.open('a') as stream:
            stream.write(json.dumps({'samples': samples, 'idle_candidates': idle})+'\n')
        print(json.dumps({'idle_candidates': idle, 'guard_attempts': attempt, 'time': samples[-1]['time']}), flush=True)
        for gpu in idle:
            attempt += 1
            record = prefix.with_name(prefix.name+f'-attempt{attempt:03d}.json')
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('gpu_guard.py')),
                                     '--gpu', str(gpu), '--record', str(record), '--', *command])
            if record.exists():
                audit = json.loads(record.read_text())
                if audit['allowed']:
                    # Never rerun a partially executed experiment after a child failure.
                    raise SystemExit(result.returncode)
            else:
                # Another project job can win the local GPU lock after our sample.
                with prefix.with_name(prefix.name+'-lock-races.jsonl').open('a') as stream:
                    stream.write(json.dumps({'attempt': attempt, 'gpu': gpu, 'returncode': result.returncode,
                                             'child_not_admitted': True, 'reason': 'guard produced no admission record'})+'\n')
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
