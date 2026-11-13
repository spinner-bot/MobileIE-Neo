"""One-byte CUDA placement regression after the real CPU metric constructor; zero scores."""
import argparse
from datetime import datetime
import fcntl
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from prebind_shared_gpu import require_initial_binding, actual_device
from shared_gpu_admission import ROOT, assess, sustained_samples, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True); args = parser.parse_args()
    bound = require_initial_binding()
    out = Path(args.out); assert out.resolve().is_relative_to('/data/Users/zrh')
    out.mkdir(exist_ok=False)
    import torch
    sys.path.insert(0, str(ROOT/'research/4.temporal-consistency/scripts'))
    from cvvdp_adapter_r66 import load_metric
    torch.set_num_threads(2); started=time.perf_counter(); complete=False
    try:
        assert not torch.cuda.is_initialized()
        metric, source = load_metric('cpu')
        assert metric.device.type == 'cpu' and not torch.cuda.is_initialized()
        samples = sustained_samples(); assessment = assess(samples)
        selected = next(r for r in assessment['candidates'] if r['uuid'] == bound)
        write(out/'admission.json', dict(samples=samples, selected=selected,
              real_CPU_metric_prepared=True, package_version=source['package_version']))
        if not selected['allowed']:
            raise SystemExit(75)
        with (ROOT/f'.cache/gpu-{selected["index"]}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            torch.cuda.set_per_process_memory_fraction(256/selected['total_mib'], device=0)
            value = torch.empty(1, dtype=torch.uint8, device='cuda'); torch.cuda.synchronize()
            actual = actual_device(bound); write(out/'actual-device.json', actual)
            assert actual['matched'] and value.numel() == value.element_size() == 1
            write(out/'result.json', dict(passed=True, expected_physical_GPU=selected['index'],
                actual_physical_GPU=actual['actual_own_GPU_indices'], UUID=bound,
                CPU_metric_preparation_included=True, CUDA_bytes_requested=1,
                metric_predictions=0, new_enhancement_CNN=0, new_motion_CNN=0, SGD=0))
            complete=True
            globals()['_held_lock'] = lock
    finally:
        write(out/'runtime.json', dict(ended=datetime.now().astimezone().isoformat(),
            completed_normally=complete, seconds=time.perf_counter()-started,
            metric_predictions=0, SGD=0, scope='Physical device binding regression only, no numerical qualification or method evidence.'))


if __name__ == '__main__':
    main()
