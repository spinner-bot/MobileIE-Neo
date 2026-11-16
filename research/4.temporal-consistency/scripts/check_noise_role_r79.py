"""One real-training-window qualification, then discard every model/optimizer state."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
from prebind_shared_gpu import require_initial_binding
from noise_role_io_r79 import PROCESS, digest, write, stamp, verify, acquire_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    bound = require_initial_binding() if args.device == 'cuda' else None
    if args.device == 'cpu' and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit('CPU qualification needs an initially empty CUDA mask.')
    out = args.out.resolve()
    assert out.is_relative_to(PROCESS) and not out.exists()
    out.mkdir(parents=True)
    import cv2
    import numpy as np
    import torch
    from exploration_r2_data import degrade
    from train_exploration_r2 import new_model
    from noise_role_data_r79 import NoiseRolePool, synthetic_views, gain
    from noise_role_model_r79 import MODES, models, inputs, sequence, update, optimizer_for, precision
    cv2.setNumThreads(1); precision()
    began = time.perf_counter()
    passed, completed, schedule, gpu_lock = False, 0, None, None
    try:
        verify()
        if bound:
            assert json.loads((PROCESS/'cpu-gate001/gate.json').read_text())['passed']
        pool = NoiseRolePool()
        ordinary, cur, past, target, identities = pool.batch(0, 0)
        assert torch.equal(ordinary[2:], cur[2:]) and torch.equal(ordinary[2:], past[2:])
        actual, previous = inputs(ordinary, cur, past)
        assert torch.equal(actual[:, 0], ordinary[:, 0]) and torch.equal(previous[:, 0], ordinary[:, 0])
        assert gain(0) == 16 and gain(999) == gain(1000) == gain(3999) == 1
        clean = target[0, 0].permute(1, 2, 0).numpy()*255
        a = np.random.default_rng(79001); b = np.random.default_rng(79001)
        legacy = degrade(clean, a, photons=20, jpeg=False)
        views = synthetic_views(clean, b, 20, 1.)
        assert all(np.array_equal(legacy, view) for view in views)
        nets = models(0)
        first = nets['spatial'].state_dict()
        assert all(all(torch.equal(first[k], value) for k, value in net.state_dict().items()) for net in nets.values())
        original = new_model(torch.device('cpu')).eval()
        with torch.no_grad():
            expected = original(ordinary.flatten(0, 1))
            errors = {name: float((sequence(net.eval(), ordinary).flatten(0, 1)-expected).abs().max())
                      for name, net in nets.items()}
        assert max(errors.values()) <= 2e-5
        del expected, original
        if bound:
            gpu_lock, schedule, selected = acquire_gpu(bound, out)
        nets = {name: net.to(args.device).train() for name, net in nets.items()}
        values = [v.to(args.device) for v in (ordinary, cur, past, target)]
        ordinary, cur, past, target = values
        if bound:
            torch.cuda.reset_peak_memory_stats()
        stats = []
        for name, net in nets.items():
            optimizer = optimizer_for(net)
            for _ in range(2):
                started = time.perf_counter()
                if schedule:
                    schedule.check_memory()
                row = update(net, optimizer, ordinary, target,
                             cur if name == 'curriculum' else None, past if name == 'curriculum' else None)
                if bound:
                    torch.cuda.synchronize()
                completed += 1
                stats.append(dict(model=name, **row))
                if schedule:
                    schedule.yield_after_work(time.perf_counter()-started)
            del optimizer
        checks = {}
        for name, net in nets.items():
            net.eval()
            changed = ordinary.clone(); changed[:, 3] = 1-changed[:, 3]
            with torch.no_grad():
                causal = sequence(net, ordinary)
                altered = sequence(net, changed)
                current = sequence(net, ordinary, replace_history=True)
            delta = float((causal-current).abs().max())
            prefix = float((causal[:, :3]-altered[:, :3]).abs().max())
            assert prefix <= 1e-6
            assert delta == 0 if name == 'spatial' else delta > 1e-8
            checks[name] = dict(future_change_prefix_max_delta=prefix, history_replacement_max_delta=delta,
                               parameters=sum(p.numel() for p in net.parameters()))
        observed_own = None
        if bound:
            from shared_gpu_admission import snapshot
            sample = snapshot(); write(out/'post-work-load.json', sample)
            own = [int(p.split(',')[2]) for row in sample['GPUs'] for p in row['processes']
                   if int(p.split(',')[1]) == os.getpid()]
            assert len(own) == 1 and own[0] <= 4096
            observed_own = own[0]
            assert torch.cuda.max_memory_reserved() <= 2048*1048576
        else:
            assert not torch.cuda.is_initialized()
        passed = True
        write(out/'gate.json', dict(at=stamp(), passed=True, device=args.device, seed=0,
            frozen_sources_sha256=digest(PROCESS/'source-freeze001.json'), data=pool.info,
            window=identities, initial_function_errors=errors, endpoint_g1_legacy_pixels_equal=True,
            real_role_pixels_unchanged=True, first_frame_ordinary_fallback=True, checks=checks,
            updates=stats, discarded_updates=completed, formal_updates=0, observed_own_process_mib=observed_own,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if bound else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if bound else None,
            resource=schedule.summary() if schedule else None,
            scope='Real train window qualification only; all states discarded. No DEV, quality, novelty or speed claim.'))
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', dict(ended=stamp(), passed=passed, discarded_updates=completed,
            formal_updates=0, pid=os.getpid(), seconds=time.perf_counter()-began,
            cuda_initialized=torch.cuda.is_initialized()))
        globals()['_held_gpu_lock'] = gpu_lock


if __name__ == '__main__':
    main()
