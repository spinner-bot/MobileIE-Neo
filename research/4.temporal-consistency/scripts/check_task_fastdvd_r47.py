"""Finite interface/gradient checks, not a quality selection experiment."""
import argparse
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import cv2
import torch

from dynamic_training_r34 import BASE, ROOT, digest
from mixed_anchor_r40 import MixedPool
from task_fastdvd_r47 import initial_model, official_centers, prediction, provenance
from training_form_r41 import configure_precision, cuda_headroom


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True, choices=['cpu', 'cuda'])
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    assert out.is_relative_to(BASE/'process/exploration-r47')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    configure_precision()
    assert not torch.cuda.is_initialized()
    write(out/'config.json', {'at': datetime.now().astimezone().isoformat(),
         'device': args.device, 'discarded_updates_registered': 2,
         'protocol_sha256': digest(BASE/'docs/exploration-r47-task-adapted-video-protocol.md'),
         'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in
             [Path(__file__), BASE/'scripts/task_fastdvd_r47.py', ROOT/'third_party/fastdvdnet/models.py']},
         'initialization': provenance(), 'scope': 'Training data only; no DEV or quality claim.'})
    start, complete, updates, parity = time.perf_counter(), False, 0, []
    try:
        model = initial_model(0)
        pool = MixedPool()
        low, gt, context = pool.batch(0, 0)
        write(out/'training-window.json', {'data': pool.info, 'windows': context})
        print('Owned MixedPool and first registered training window loaded.', flush=True)
        if args.device == 'cuda':
            admission = cuda_headroom()
            write(out/'pre_cuda_guard.json', admission)
            assert admission['allowed']
            model, low, gt = model.cuda(), low.cuda(), gt.cuda()
            torch.cuda.reset_peak_memory_stats()
        model.eval()
        cases = [low] if args.device == 'cuda' else [
            torch.rand(2, 4, 3, 32, 36), torch.rand(2, 4, 3, 33, 37)]
        with torch.inference_mode():
            for value in cases:
                expected = official_centers(model, value)
                actual = prediction(model, value, clamp=True)
                delta = float((actual-expected).abs().max())
                tolerance = 1e-4 if args.device == 'cuda' else 1e-5
                parity.append({'shape': list(value.shape), 'max_abs_delta': delta,
                               'tolerance': tolerance, 'passed': delta <= tolerance})
        assert all(r['passed'] for r in parity), parity
        model.train()
        parameters = list(model.parameters())
        first, last = parameters[0], parameters[-1]
        first_before, last_before = first.detach().clone(), last.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        steps = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = prediction(model, low)
            loss = (output-gt).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
            first_gradient, last_gradient = float(first.grad.norm()), float(last.grad.norm())
            assert first_gradient > 0 and last_gradient > 0
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
            optimizer.step(); updates += 1
            steps.append({'step': step+1, 'MSE': float(loss.detach()),
                          'first_gradient': first_gradient, 'last_gradient': last_gradient,
                          'norm_before_clip': norm})
        changed = {'first': float((first-first_before).detach().abs().max()),
                   'last': float((last-last_before).detach().abs().max())}
        assert all(v > 0 for v in changed.values())
        write(out/'gate.json', {'passed': True, 'parity': parity, 'discarded_updates': updates,
             'steps': steps, 'parameter_deltas': changed,
             'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
             'inherited_into_formal_training': False, 'DEV_pixels': 0})
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'device': args.device,
            'discarded_updates': updates, 'parity': parity, 'seconds': time.perf_counter()-start,
            'ended': datetime.now().astimezone().isoformat()})


if __name__ == '__main__':
    main()
