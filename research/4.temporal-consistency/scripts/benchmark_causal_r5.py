"""Eager batch-one model cost including coarse arithmetic and actual state copies."""
import argparse
import gc
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from causal_fst_r5 import CausalFST, MODES
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256
from train_exploration_r2 import new_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--variants', default=','.join(MODES))
    p.add_argument('--direct-fixed-shift', action='store_true')
    args = p.parse_args()
    modes = args.variants.split(',')
    if any(m not in MODES for m in modes):
        p.error('known modes required')
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(32)
    device = torch.device('cuda')
    inputs = [torch.rand(1, 3, 400, 600, device=device)*.1 for _ in range(2)]
    rows = []
    start = time.perf_counter()
    with torch.inference_mode():
        for block in range(3):
            order = np.random.default_rng(32+block).permutation(modes).tolist()
            for mode in order:
                net = CausalFST(new_model(device), mode).to(device).eval()
                net.direct_fixed_shift = args.direct_fixed_shift
                scope = 'exploration-r5c' if mode == 'spatial_post' else 'exploration-r5'
                check = ROOT/'.cache/checkpoints'/scope/f'{mode}-seed0'/'step1000.pth'
                provenance = None
                if check.exists():
                    net.load_state_dict(torch.load(check, map_location=device, weights_only=False)['model'])
                    provenance = {'path': str(check.relative_to(ROOT)), 'sha256': sha256(check)}
                net.reset()
                for i in range(50):
                    net(inputs[i%2])
                torch.cuda.synchronize()
                live = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                pairs = []
                wall = time.perf_counter()
                for i in range(200):
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    net(inputs[i%2])
                    b.record()
                    pairs.append((a, b))
                torch.cuda.synchronize()
                wall_ms = (time.perf_counter()-wall)*1000/200
                times = [a.elapsed_time(b) for a, b in pairs]
                row = {'block': block, 'mode': mode, 'parameters': sum(p.numel() for p in net.parameters()),
                    'direct_fixed_shift': args.direct_fixed_shift if mode == 'tsm_fixed' else False,
                    'checkpoint': provenance, 'cuda_event_ms': times, 'median_ms': float(np.median(times)),
                    'mean_ms': float(np.mean(times)), 'p95_ms': float(np.percentile(times, 95)),
                    'synchronized_wall_ms_per_call': wall_ms, 'state_storage_bytes': net.state_storage_bytes(),
                    'live_allocated_after_warmup_bytes': live, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_above_warmup_bytes': torch.cuda.max_memory_allocated()-live}
                rows.append(row)
                print({k: row[k] for k in ['block', 'mode', 'median_ms', 'state_storage_bytes']}, flush=True)
                del net, pairs
                gc.collect()
                torch.cuda.empty_cache()
    save_json(args.out, {'rows': rows, 'shape': [1, 3, 400, 600], 'warmup': 50, 'timed': 200, 'blocks': 3,
        'scope': 'GPU model only incl pooling/upsampling/branch/state clone; no decode/transfer/CPU flow; eager FP32',
        'inputs': 'alternating two seeded random low-range frames, common across modes',
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cudnn_deterministic': True,
        'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32,
        'matmul_tf32_allowed': torch.backends.cuda.matmul.allow_tf32,
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'wall_seconds': time.perf_counter()-start})


if __name__ == '__main__':
    main()
