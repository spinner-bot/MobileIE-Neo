"""CPU-only qualification for the existing metric at the new source timing."""
import time
import traceback
import cv2
import torch

from reds_metric_r72 import qualify24
from reds_references_r72 import PROCESS, stamp, write


def main():
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    out = PROCESS / 'metric-cpu-gate001'; out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); complete = False
    try:
        write(out / 'gate.json', qualify24('cpu'))
        assert not torch.cuda.is_initialized(); complete = True
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out / 'runtime.json', dict(completed_normally=complete, new_enhancement_CNN=0,
              new_SGD=0, CUDA_initialized=torch.cuda.is_initialized(), ended=stamp(),
              seconds=time.perf_counter() - started))


if __name__ == '__main__':
    main()
