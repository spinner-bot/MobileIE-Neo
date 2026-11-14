"""One discarded CPU update on the unchanged registered first training targets."""
import json
import time
import traceback
from datetime import datetime

import cv2
import torch

from dynamic_training_r34 import BASE, digest
from mixed_anchor_r40 import MixedPool
from tiny_task_r56 import check_one, initial_model
from training_form_r41 import configure_precision


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    configure_precision(); assert not torch.cuda.is_initialized()
    out = BASE/'process/exploration-r56/cpu-gate001'; out.mkdir(parents=True, exist_ok=False)
    start, complete = time.perf_counter(), False
    try:
        pool = MixedPool(); low, target, identities = pool.batch(0, 0)
        with (BASE/'results/exploration-r53-stage001-seed0/windows.jsonl').open() as stream:
            assert json.loads(stream.readline())['windows'] == identities
        old = json.loads((BASE/'results/exploration-r53-stage001-seed0/config.json').read_text())
        for name in ['mixed_anchor_r40.py', 'dynamic_training_r34.py']:
            assert old['source_sha256'][name] == digest(BASE/'scripts'/name)
        model, source = initial_model()
        result = check_one(model, low, target)
        write(out/'gate.json', dict(at=datetime.now().astimezone().isoformat(), **result,
              initialization=source, data=pool.info, original_targets=identities, original_target_identities_match=True,
              device='cpu', cuda_initialized=False,
              protocol_sha256=digest(BASE/'docs/exploration-r56-tiny-task-adaptation-protocol.md'),
              source_sha256={n: digest(BASE/'scripts'/n) for n in ['tiny_task_r56.py', 'check_tiny_task_r56.py',
                             'multinex_reference_r55.py', 'mixed_anchor_r40.py', 'dynamic_training_r34.py']},
              scope='One discarded update only, no DEV or final pixels; all original BN buffers will be reloaded before formal training.'))
        assert not torch.cuda.is_initialized(); complete = True
        print(json.dumps(dict(passed=True, branch_gradient_L2=result['branch_gradient_L2'], BN_after=result['BN_after'])), flush=True)
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, seconds=time.perf_counter()-start,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    main()
