"""Wait for the live fixed stage, then summarize and measure final-model cost."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time
import traceback

from dynamic_training_r34 import BASE, ROOT, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coordinator-pid', type=int, required=True)
    args = parser.parse_args()
    out = BASE/'process/exploration-r34/finalize001'
    out.mkdir(parents=True, exist_ok=False)
    scripts = ['summarize_dynamic_r34.py', 'benchmark_dynamic_r34.py', 'summarize_runtime_r34.py']
    hashes = {name: digest(BASE/'scripts'/name) for name in scripts}
    (out/'config.json').write_text(json.dumps({'started': datetime.now().astimezone().isoformat(),
          'coordinator_pid': args.coordinator_pid, 'dependency_stage': 'stage003', 'script_sha256': hashes,
          'scope': 'No training restart or final-test pixels; only completed DEV quality and final-weight CPU cost.'}, indent=2)+'\n')
    complete = False
    try:
        proc = Path(f'/proc/{args.coordinator_pid}/cmdline')
        if proc.exists():
            assert 'launch_dynamic_stage_r34.py' in proc.read_bytes().decode()
            print('Waiting for verified live R34 coordinator', flush=True)
            while proc.exists():
                time.sleep(10)
        runtime = json.loads((BASE/'process/exploration-r34/stage003/runtime.json').read_text())
        assert runtime['completed_normally'], 'Parent stage failed; preserve failure, do not benchmark partial weights'
        for name in scripts:
            assert digest(BASE/'scripts'/name) == hashes[name], 'Queued postprocessing source changed'
            with (out/(name.removesuffix('.py')+'.log')).open('w') as log:
                subprocess.run([str(ROOT/'scripts/project-python'), str(BASE/'scripts'/name)],
                               cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(json.dumps({'completed': name, 'time': datetime.now().astimezone().isoformat()}), flush=True)
        complete = True
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        (out/'runtime.json').write_text(json.dumps({'completed_normally': complete,
            'ended': datetime.now().astimezone().isoformat()}, indent=2)+'\n')


if __name__ == '__main__':
    main()
