"""Resume an existing paused metric only after admitting its actual physical GPU."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from shared_gpu_admission import ROOT, assess, snapshot, sustained_samples, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--start-ticks', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    args = parser.parse_args()
    out = ROOT/'research/4.temporal-consistency/process/exploration-r74/shared-perception001/actual-device-supervision'
    out.mkdir(exist_ok=False)

    def identity():
        p = Path(f'/proc/{args.pid}/stat')
        if not p.exists():
            return None
        fields = p.read_text().split()
        return fields if fields[21] == args.start_ticks and fields[2] != 'Z' else None

    current = identity()
    assert current and current[2] in ['T', 't']
    lock = (ROOT/f'.cache/gpu-{args.gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    samples = sustained_samples(); admission = assess(samples)
    selected = next(r for r in admission['candidates'] if r['index'] == args.gpu)
    write(out/'admission.json', dict(at=datetime.now().astimezone().isoformat(), samples=samples,
        actual_device=selected, allowance_includes_already_allocated_own_context=True,
        scope='Supplemental actual-device admission; original mistaken selection preserved. Same process, same scores.'))
    assert selected['allowed'], 'Actual GPU not admitted; existing metric stays paused, no retry or score overwrite.'
    row = next(r for r in snapshot()['GPUs'] if r['index'] == args.gpu)
    own = [p for p in row['processes'] if int(p.split(',')[1].strip()) == args.pid]
    assert len(own) == 1 and identity() is not None
    assert row['total_mib']-row['used_mib'] >= selected['required_free_mib']
    os.kill(args.pid, signal.SIGCONT)
    write(out/'resume.json', dict(at=datetime.now().astimezone().isoformat(), pid=args.pid,
        start_ticks=args.start_ticks, actual_gpu=args.gpu, actual_gpu_uuid=selected['uuid'],
        same_original_process_resumed=True, no_recomputed_scores=True,
        scope='Original allocator cap and pacing remain; this supervisor observes actual device every2seconds.'))
    observations = 0; action = None
    with (out/'observations.jsonl').open('x') as stream:
        while identity() is not None:
            sample = snapshot(); row = next(r for r in sample['GPUs'] if r['index'] == args.gpu)
            stream.write(json.dumps(sample)+'\n'); stream.flush(); observations += 1
            if row['total_mib']-row['used_mib'] < selected['incumbent_growth_reserve_mib']+1024:
                # Release our context before jeopardizing incumbent memory growth.
                if identity() is not None:
                    os.kill(args.pid, signal.SIGTERM)
                    action = 'Terminated only owned metric to release actual-device memory; no automatic scoring retry.'
                break
            time.sleep(2)
    write(out/'runtime.json', dict(ended=datetime.now().astimezone().isoformat(), samples=observations,
        original_process_terminal=identity() is None, protective_action=action,
        scope='Resource observation only, not proof of zero incumbent slowdown or metric success.'))


if __name__ == '__main__':
    main()
