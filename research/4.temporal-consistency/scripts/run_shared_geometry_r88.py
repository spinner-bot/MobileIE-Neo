"""One non-retrying end-to-end chain: train, full DEV, official JOD, fixed summary/contacts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from shared_geometry_io_r88 import ROOT, BASE, PROCESS, verify, write, stamp, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert args.name.replace('-', '').isalnum()
    for device in ('cpu', 'cuda'):
        gate = json.loads((PROCESS/f'{device}-gate002/gate.json').read_text())
        assert gate['passed'] and gate['discarded_updates'] == 8
        verify(json.loads((PROCESS/f'{device}-gate002/config.json').read_text())['frozen'])
    frozen = {p.name: digest(p) for p in PROCESS.glob('*source-freeze00*.json')}
    verify(json.loads((PROCESS/'training-source-freeze002.json').read_text()))
    verify(json.loads((PROCESS/'evaluation-source-freeze001.json').read_text()))
    out = PROCESS/args.name; out.mkdir(exist_ok=False)
    log = []; started = time.perf_counter(); complete = False
    python = str(ROOT/'scripts/project-python')
    scripts = BASE/'scripts'
    write(out/'config.json', dict(at=stamp(), pid=os.getpid(), train=args.train,
        stage_freezes=frozen, coordinator_sha256=digest(__file__), scientific_retry=False))

    def batch(jobs):
        active = []
        for name, command in jobs:
            stdout = (out/(name+'.stdout.log')).open('w')
            child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=subprocess.STDOUT)
            value = dict(name=name, pid=child.pid, command=command, began=stamp())
            log.append(value); active.append((child, stdout, value))
            print(json.dumps(dict(event='launched', **value)), flush=True)
        write(out/'children.json', log)
        for child, stdout, value in active:
            code = child.wait(); stdout.close()
            value.update(returncode=code, ended=stamp()); write(out/'children.json', log)
            print(json.dumps(dict(event='finished', **value)), flush=True)
        assert all(value['returncode'] == 0 for _, _, value in active), 'No automatic scientific retry.'

    def gpu(name, index, script, extra=()):
        return (name, [python, str(ROOT/'scripts/prebind_shared_gpu.py'), '--gpu', str(index),
            '--record', str(out/(name+'-binding.json')), '--', python, str(scripts/script), *extra])

    try:
        if args.train:
            batch([gpu(f'train-seed{seed}', seed, 'train_shared_geometry_r88.py',
                       ['--seed', str(seed), '--attempt', '1']) for seed in (0, 1)])
        batch([(f'quality-seed{seed}', [python, str(scripts/'evaluate_shared_geometry_r88.py'), '--seed', str(seed)])
               for seed in (0, 1)])
        batch([gpu('metric', 0, 'evaluate_cvvdp_shared_geometry_r88.py')])
        batch([('joint', [python, str(scripts/'summarize_shared_geometry_r88.py')])])
        batch([('contacts', [python, str(scripts/'render_shared_geometry_r88.py')])])
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(ended=stamp(), pid=os.getpid(), completed_normally=complete,
            seconds=time.perf_counter()-started, children=log, actual_contact_review=False,
            cost_or_phone_measured=False, independent_final=False, new_method_established=False))


if __name__ == '__main__':
    main()
