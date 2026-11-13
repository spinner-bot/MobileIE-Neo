"""Small analytic and actual-model gate; no SGD and no development pixels."""
import json
import time
from datetime import datetime

import cv2
import torch
from torch import nn

from bn_calibration_r50 import calibrate, snapshot, source_model, verify_change
from dynamic_training_r34 import BASE
from mixed_anchor_r40 import MixedPool, prediction


def main():
    torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    out = BASE/'process/exploration-r50/cpu-gate001'
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    toy = nn.BatchNorm2d(2).requires_grad_(False).eval()
    inputs = [torch.arange(48, dtype=torch.float32).reshape(2, 2, 3, 4)/20,
              torch.arange(48, dtype=torch.float32).reshape(2, 2, 3, 4)/10+1]
    torch.optim.swa_utils.update_bn(inputs, toy)
    mean = torch.stack([x.mean((0, 2, 3)) for x in inputs]).mean(0)
    var = torch.stack([x.var((0, 2, 3), unbiased=True) for x in inputs]).mean(0)
    mean_delta = float((toy.running_mean-mean).abs().max())
    var_delta = float((toy.running_var-var).abs().max())
    assert mean_delta <= 1e-6 and var_delta <= 1e-6 and int(toy.num_batches_tracked) == 2
    assert toy.momentum == .1 and not toy.training
    pool = MixedPool()
    batches = [pool.batch(50, i) for i in range(2)]
    net, marker = source_model(0, 'square_mse')
    before = calibrate(net, (batch[0] for batch in batches), 'cpu')
    change = verify_change(net, before, 8)
    with torch.no_grad():
        slim = net.slim().eval()
        delta = float((prediction(net, batches[0][0])-prediction(slim, batches[0][0])).abs().max())
    assert delta <= 1e-4 and sum(p.numel() for p in slim.parameters()) == 4075
    assert not torch.cuda.is_initialized()
    result = {'at': datetime.now().astimezone().isoformat(), 'passed': True,
              'analytic_mean_delta': mean_delta, 'analytic_unbiased_batch_variance_delta': var_delta,
              'actual_model': 'square_mse', 'seed': 0, 'discarded_calibration_windows': 2,
              'discarded_SGD_updates': 0, 'actual_model_check': change, 'fold_max_abs_delta': delta,
              'source': marker, 'windows': [batch[2] for batch in batches], 'data': pool.info,
              'seconds': time.perf_counter()-start, 'cuda_initialized': False}
    (out/'gate.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ['passed', 'seconds', 'fold_max_abs_delta']}))


if __name__ == '__main__':
    main()
