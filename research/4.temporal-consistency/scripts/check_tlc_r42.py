"""Independent direct-mean and unchanged-weight gates for the fixed TLC adapter."""
from datetime import datetime
import json
from pathlib import Path
import time
import traceback

import cv2
import torch
from torch.nn import functional as F

from dynamic_training_r34 import BASE, digest
from tlc_pool_r42 import FixedLocalAverage, convert, load_anchor


def direct_average(x, window):
    height, width = x.shape[-2:]
    kh, kw = min(height, window), min(width, window)
    value = F.avg_pool2d(x, (kh, kw), stride=1)
    dh, dw = height-value.shape[-2], width-value.shape[-1]
    return F.pad(value, (dw//2, (dw+1)//2, dh//2, (dh+1)//2), mode='replicate')


def main():
    folder = BASE/'process/exploration-r42'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder/'cpu-gate.json'
    assert not target.exists(), 'Retain prior gate; no automatic retry'
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(4200)
    started = time.perf_counter()
    record = {'created_at': datetime.now().astimezone().isoformat(), 'passed': False,
              'pool_checks': [], 'network_checks': [], 'torch_threads': 2, 'opencv_threads': 1,
              'source_sha256': {n: digest(Path(__file__).parent/n) for n in
                               ['check_tlc_r42.py', 'tlc_pool_r42.py']},
              'scope': 'Algorithm correctness and state_dict invariance only; no quality, training or speed ranking.'}
    try:
        assert not torch.cuda.is_initialized()
        with torch.inference_mode():
            for shape, window in [((2, 3, 27, 31), 3), ((1, 4, 28, 32), 4),
                                  ((1, 2, 201, 203), 192), ((2, 3, 31, 37), 192)]:
                x = torch.randn(shape)
                value, reference = FixedLocalAverage(window)(x), direct_average(x, window)
                delta = float((value-reference).abs().max())
                row = {'shape': list(shape), 'window': window, 'max_abs_difference': delta}
                record['pool_checks'].append(row)
                assert value.shape == x.shape and delta <= 1e-5, row
            x = torch.full((1, 2, 512, 854), .75)
            delta = float((FixedLocalAverage()(x)-.75).abs().max())
            record['pool_checks'].append({'constant_shape': list(x.shape), 'window': 192,
                                          'value': .75, 'max_abs_difference': delta})
            assert delta <= 1e-5
            for seed in [0, 1]:
                net, marker = load_anchor(seed)
                local = convert(net)
                for shape in [(2, 3, 33, 37), (1, 3, 128, 128)]:
                    x = .2*torch.rand(shape)
                    delta = float((net(x)-local(x)).abs().max())
                    row = {'seed': seed, 'shape': list(shape), 'input_range': [0., .2],
                           'max_abs_difference': delta, 'deployment_parameters': 4075,
                           'checkpoint_sha256': marker['sha256']}
                    record['network_checks'].append(row)
                    assert delta <= 1e-4, row
        assert not torch.cuda.is_initialized()
        record['passed'] = True
    except Exception:
        record['failure'] = traceback.format_exc()
        raise
    finally:
        record.update(wall_seconds=time.perf_counter()-started, cuda_initialized=torch.cuda.is_initialized())
        target.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
