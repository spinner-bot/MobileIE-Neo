"""Steady-state GPU interaction cost including warp, excluding flow/grid creation."""
import argparse
import statistics

import numpy as np
import torch

from explore_temporal_controls import save_json
from paired_fst_r3 import TrainablePairFST, set_alignment
from train_exploration_r2 import new_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    from pathlib import Path
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    device = torch.device('cuda')
    models = {name: new_model(device) for name in ['ordinary', 'cross', 'mean_square']}
    flow = np.zeros((400, 600, 2), np.float32)
    flow[..., 0] = 2
    mask = np.ones((400, 600), bool)
    mask[:, -2:] = False
    for name in ['cross', 'mean_square']:
        models[name].body = TrainablePairFST(models[name].body, name)
        with torch.no_grad():
            models[name].body.gamma.fill_(.3)
        set_alignment(models[name].body, flow, mask, device)
    value = torch.rand(1, 3, 400, 600, device=device)
    records = {name: {'latencies_ms': [], 'peak_incremental_bytes': []} for name in models}
    with torch.inference_mode():
        names = list(models)
        for repeat in range(3):
            for name in names[repeat:]+names[:repeat]:
                model = models[name]
                for _ in range(50):
                    model(value)
                torch.cuda.synchronize()
                initial = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                events = []
                for _ in range(200):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    model(value)
                    end.record()
                    events.append((begin, end))
                torch.cuda.synchronize()
                records[name]['latencies_ms'].extend(a.elapsed_time(b) for a, b in events)
                records[name]['peak_incremental_bytes'].append(torch.cuda.max_memory_allocated()-initial)
    for name, row in records.items():
        row['median_ms'] = statistics.median(row['latencies_ms'])
        row['parameters'] = sum(p.numel() for p in models[name].parameters())
        row['persistent_temporal_buffers_bytes'] = 0 if name == 'ordinary' else sum(
            v.numel()*v.element_size() for v in [models[name].body.previous, models[name].body.grid, models[name].body.mask])
    save_json(args.out, {'shape': [1, 3, 400, 600], 'backend': 'PyTorch eager', 'input_dtype': 'float32',
        'conv_tf32_allowed': torch.backends.cudnn.allow_tf32, 'gpu': torch.cuda.get_device_name(),
        'torch': torch.__version__, 'cudnn': torch.backends.cudnn.version(),
        'warmup_per_block': 50, 'timed_per_block': 200, 'blocks': 3,
        'scope': 'GPU feature warp + interaction; excludes CPU flow/grid creation/transfer; not mobile/end-to-end',
        'models': records})


if __name__ == '__main__':
    main()
