"""CPU-only artificial video interface qualification before any real metric score."""
import json
import time
import traceback
from datetime import datetime

import torch

from cvvdp_adapter_r66 import BASE, qualify


def main():
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    out = BASE/'process/exploration-r66/cpu-gate001'; out.mkdir(parents=True, exist_ok=False)
    start, complete = time.perf_counter(), False
    try:
        result = qualify('cpu'); assert not torch.cuda.is_initialized()
        (out/'gate.json').write_text(json.dumps(dict(at=datetime.now().astimezone().isoformat(), **result), indent=2)+'\n')
        print(json.dumps(dict(passed=True, scores=result['scores'], metric_info=result['source']['metric_info'])), flush=True)
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        (out/'runtime.json').write_text(json.dumps(dict(completed_normally=complete,
              seconds=time.perf_counter()-start, cuda_initialized=torch.cuda.is_initialized(),
              ended=datetime.now().astimezone().isoformat()), indent=2)+'\n')


if __name__ == '__main__': main()
