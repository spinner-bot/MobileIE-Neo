"""Same executed-Conv MAC convention as R47; analytic priors are excluded."""
import copy
import json
from datetime import datetime

import torch

from dynamic_training_r34 import BASE, digest
from multinex_reference_r55 import load_reference, prediction


def main():
    assert not torch.cuda.is_initialized()
    out = BASE/'results/exploration-r55-conv-budget001'; out.mkdir(exist_ok=False)
    model, source = load_reference()
    net = copy.deepcopy(model).to('meta').eval()
    rows = []
    for h, w in [(288, 512), (360, 640), (480, 854), (1080, 1920)]:
        calls, handles = [], []
        def count(module, inputs, output):
            mac = output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
            calls.append(dict(input_shape=list(inputs[0].shape), output_shape=list(output.shape), MAC=int(mac)))
        try:
            for module in net.modules():
                if isinstance(module, torch.nn.Conv2d): handles.append(module.register_forward_hook(count))
            value = prediction(net, torch.empty(1, 3, h, w, device='meta'))
            assert list(value.shape) == [1, 3, h, w]
            rows.append(dict(height=h, width=w, parameters=source['parameters'], executed_Conv_calls=len(calls),
                             Conv_MACs=sum(r['MAC'] for r in calls), calls=calls))
        finally:
            for handle in handles: handle.remove()
    path = out/'cost.json'
    path.write_text(json.dumps(dict(at=datetime.now().astimezone().isoformat(), source=source, rows=rows,
          cuda_initialized=False, real_CNN_images=0, device='meta',
          source_sha256=digest(BASE/'scripts/count_multinex_convs_r55.py'),
          excluded='bias adds, analytic min/max/sqrt/ratios, pooling, normalization, activation, pointwise, memory, flow, warp and IO',
          scope='Executed Conv MAC shape arithmetic only. Not complete FLOPs, measured latency, memory or phone performance.'), indent=2)+'\n')
    print(json.dumps([dict(height=r['height'], width=r['width'], parameters=r['parameters'], Conv_MACs=r['Conv_MACs']) for r in rows]))


if __name__ == '__main__':
    main()
