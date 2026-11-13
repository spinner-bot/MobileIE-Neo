"""Select a physical GPU before child imports; child must recheck and lock after CPU preparation."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import time

from shared_gpu_admission import ROOT, assess, snapshot, sustained_samples, write


def require_initial_binding():
    expected = os.environ.get('MOBILEIE_SHARED_GPU_UUID')
    initial = dict(row.split(b'=', 1) for row in Path('/proc/self/environ').read_bytes().split(b'\0') if b'=' in row)
    if not expected or initial.get(b'CUDA_VISIBLE_DEVICES') != expected.encode():
        raise RuntimeError('GPU UUID must be bound in the child initial environment via prebind_shared_gpu.py')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != expected:
        raise RuntimeError('Visible device changed after process startup')
    return expected


def actual_device(expected):
    sample = snapshot()
    own = [r for r in sample['GPUs'] if any(int(p.split(',')[1].strip()) == os.getpid() for p in r['processes'])]
    return dict(expected_uuid=expected, matched=len(own) == 1 and own[0]['uuid'] == expected,
                actual_own_GPU_indices=[r['index'] for r in own], sample=sample)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--record', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Command required after --')
    path = Path(args.record); assert path.resolve().is_relative_to('/data/Users/zrh') and not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = sustained_samples(); assessment = assess(samples)
    selected = assessment['selected'] if args.gpu is None else next(r for r in assessment['candidates'] if r['index'] == args.gpu)
    allowed = selected is not None and selected['allowed']
    record = dict(at=datetime.now().astimezone().isoformat(), samples=samples, selected=selected,
        allowed=allowed, command=command,
        scope='Process-start binding only; child must fresh-check actual bound device and acquire project lock before CUDA.')
    write(path, record)
    if not allowed:
        raise SystemExit(75)
    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = selected['uuid']
    env['MOBILEIE_SHARED_GPU_UUID'] = selected['uuid']
    began = time.perf_counter(); child = subprocess.Popen(command, cwd=ROOT, env=env)
    record['child_pid'] = child.pid; write(path, record)
    code = child.wait()
    record.update(returncode=code, child_wall_seconds=time.perf_counter()-began,
                  ended=datetime.now().astimezone().isoformat())
    write(path, record)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
