"""Run both complete CPU evaluations after normal fixed calibration; no retries."""
import json
import subprocess
import time
import traceback
from datetime import datetime

from dynamic_training_r34 import BASE, ROOT, digest


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    calibration = json.loads((BASE/'results/exploration-r50-stage001-calibration/runtime.json').read_text())
    guard = json.loads((BASE/'process/exploration-r50/calibration001-guard.json').read_text())
    assert calibration['completed_normally'] and calibration['models_completed'] == 6 and guard['returncode'] == 0
    out = BASE/'process/exploration-r50/finalize001'; out.mkdir(exist_ok=False)
    names = ['finalize_bn_calibration_r50.py', 'evaluate_bn_calibration_r50.py',
             'summarize_bn_calibration_r50.py', 'bn_calibration_r50.py', 'exploration_r2_data.py',
             'video_metrics.py', 'prepare_cache_r5.py', 'half_motion_r4.py',
             'aligned_raw_history_r11.py', 'aggregation_controls_r20.py', 'train_exploration_r2.py']
    paths = [BASE/'scripts'/n for n in names]+[ROOT/'MobileIE/model/lle.py', ROOT/'MobileIE/model/utils.py',
                  BASE/'docs/exploration-r50-bn-calibration-protocol.md']
    frozen = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    write(out/'sources.json', frozen)
    start, complete, returncodes, processes, logs = time.perf_counter(), False, {}, {}, []
    try:
        for seed in [0, 1]:
            cmd = ['./scripts/project-python', str(BASE/'scripts/evaluate_bn_calibration_r50.py'), '--seed', str(seed)]
            log = (out/f'evaluate-seed{seed}.log').open('xb'); logs.append(log)
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            processes[seed] = proc
            write(out/f'evaluate-seed{seed}-launch.json', {'at': datetime.now().astimezone().isoformat(),
                  'pid': proc.pid, 'command': cmd})
        while any(p.poll() is None for p in processes.values()):
            time.sleep(3)
        returncodes = {seed: proc.returncode for seed, proc in processes.items()}
        assert all(v == 0 for v in returncodes.values()), returncodes
        assert all(digest(ROOT/name) == expected for name, expected in frozen.items())
        with (out/'summary.log').open('xb') as stream:
            subprocess.run(['./scripts/project-python', str(BASE/'scripts/summarize_bn_calibration_r50.py')],
                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        for log in logs: log.close()
        write(out/'runtime.json', {'completed_normally': complete, 'evaluation_returncodes': returncodes,
              'seconds': time.perf_counter()-start, 'ended': datetime.now().astimezone().isoformat()})


if __name__ == '__main__':
    main()
