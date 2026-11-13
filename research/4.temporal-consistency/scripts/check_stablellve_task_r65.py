"""Two discarded CPU updates on original training windows; no DEV/GPU pixels."""
import json
import time
import traceback
from datetime import datetime

import cv2
import torch

from dynamic_training_r34 import BASE, digest
from mixed_anchor_r40 import MixedPool
from stablellve_task_r65 import ARMS, check_one, initial_model, prepare_offsets
from training_form_r41 import configure_precision


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True); configure_precision()
    assert not torch.cuda.is_initialized()
    out = BASE/'process/exploration-r65/cpu-gate001'; out.mkdir(parents=True, exist_ok=False)
    start, complete, results = time.perf_counter(), False, {}
    try:
        pool = MixedPool(); low, target, windows = pool.batch(0, 0)
        with (BASE/'results/exploration-r56-stage001-seed0/windows.jsonl').open() as stream:
            assert json.loads(stream.readline())['windows'] == windows
        offset = prepare_offsets(target)
        for arm in ARMS:
            model = initial_model()
            results[arm] = check_one(model, low, target, offset, arm)
            del model
        assert all(row['passed'] for row in results.values())
        write(out/'gate.json', dict(at=datetime.now().astimezone().isoformat(), passed=True, arms=results,
              data=pool.info, original_first_window=windows, first_window_identity_matches_R56=True,
              offsets_shape=list(offset.shape), offset_abs_max=float(offset.abs().max()),
              offset_mean_absolute=float(offset.abs().mean()), discarded_SGD_updates=2,
              protocol_sha256=digest(BASE/'docs/exploration-r65-stablellve-task-adaptation-protocol.md'),
              source_sha256={name: digest(BASE/'scripts'/name) for name in
                  ['stablellve_task_r65.py', 'check_stablellve_task_r65.py', 'stablellve_adapter_r15.py',
                   'mixed_anchor_r40.py', 'dynamic_training_r34.py', 'half_motion_r4.py']},
              cuda_initialized=False, scope='Only first original training window; not quality or full-convergence evidence.'))
        assert not torch.cuda.is_initialized(); complete = True
        print(json.dumps(dict(passed=True, discarded_SGD_updates=2, results=results)), flush=True)
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, seconds=time.perf_counter()-start,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
