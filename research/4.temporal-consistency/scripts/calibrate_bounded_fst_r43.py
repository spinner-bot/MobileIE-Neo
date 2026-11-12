"""Fixed training-window RMS units, with no parameter update or DEV selection."""
from datetime import datetime
import json
from pathlib import Path
import time
import traceback

import cv2
import torch

from bounded_fst_r43 import convert, load_training_anchor
from dynamic_training_r34 import BASE, digest
from mixed_anchor_r40 import MixedPool, prediction


def main():
    folder = BASE/'process/exploration-r43'
    target = folder/'scale-calibration.json'
    assert not target.exists(), 'No automatic retry or overwrite'
    assert json.loads((folder/'cpu-gate.json').read_text())['passed']
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    record = {'created_at': datetime.now().astimezone().isoformat(), 'completed_normally': False,
              'new_optimizer_updates': 0, 'window_steps': list(range(4000, 4008)), 'seeds': [],
              'source_sha256': {name: digest(Path(__file__).parent/name) for name in
                               ['calibrate_bounded_fst_r43.py', 'bounded_fst_r43.py', 'mixed_anchor_r40.py']},
              'scope': 'Fixed training-input RMS units, all BN/gradient states discarded; no DEV, quality or optimizer replay.'}
    try:
        assert not torch.cuda.is_initialized()
        pool = MixedPool()
        record['data'] = pool.info
        for seed in [0, 1]:
            net, marker = load_training_anchor(seed)
            net.train()
            stats = {name: {'sum_squares': 0., 'count': 0, 'max_abs': 0.} for name in ['head', 'body']}
            def hook(name):
                def collect(module, args, output):
                    value = output.detach()
                    stats[name]['sum_squares'] += float(value.double().square().sum())
                    stats[name]['count'] += value.numel()
                    stats[name]['max_abs'] = max(stats[name]['max_abs'], float(value.abs().max()))
                return collect
            handles = [getattr(net, name).block1.register_forward_hook(hook(name)) for name in stats]
            windows = []
            try:
                with torch.no_grad():
                    for step in range(4000, 4008):
                        low, _, context = pool.batch(seed, step)
                        prediction(net, low)
                        windows.append({'step': step, 'windows': context})
            finally:
                for handle in handles:
                    handle.remove()
            for values in stats.values():
                assert values['count'] == 8*4*4*12*128*128
                values['RMS'] = (values['sum_squares']/values['count'])**.5
                assert values['RMS'] > 0 and torch.isfinite(torch.tensor(values['RMS']))
            row = {'seed': seed, 'training_checkpoint_sha256': marker['training_sha256'],
                   'statistics': stats, 'contexts': windows, 'same_window_gradient_checks': []}
            record['seeds'].append(row)
            del net
            low, gt, _ = pool.batch(seed, 4000)
            for label, scales in [('unit', (1., 1.)), ('fixed_training_RMS', tuple(stats[n]['RMS'] for n in ['head', 'body']))]:
                original, _ = load_training_anchor(seed)
                candidate = convert(original, scales).train()
                loss = (prediction(candidate, low)-gt).square().mean()
                assert torch.isfinite(loss)
                loss.backward()
                active = [p for p in candidate.parameters() if p.requires_grad]
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active)
                norm = float(torch.stack([p.grad.norm() for p in active]).norm())
                gradients = {name: float(getattr(candidate, name).kappa.grad) for name in ['head', 'body']}
                fraction = sum(g*g for g in gradients.values())**.5/max(norm, 1e-30)
                row['same_window_gradient_checks'].append({'scale_rule': label, 'scales': list(scales),
                    'MSE': float(loss.detach()), 'kappa_gradients': gradients, 'all_gradient_norm': norm,
                    'kappa_gradient_norm_fraction': fraction, 'optimizer_updates': 0})
                del candidate, original
            assert abs(row['same_window_gradient_checks'][0]['MSE']-row['same_window_gradient_checks'][1]['MSE']) <= 1e-7
        assert not torch.cuda.is_initialized()
        record['completed_normally'] = True
    except Exception:
        record['failure'] = traceback.format_exc()
        raise
    finally:
        record.update(wall_seconds=time.perf_counter()-started, cuda_initialized=torch.cuda.is_initialized())
        target.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'completed_normally': record['completed_normally'],
                          'wall_seconds': record['wall_seconds']}), flush=True)


if __name__ == '__main__':
    main()
