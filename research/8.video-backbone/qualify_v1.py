"""Finite real-window qualification; all eight updates are discarded."""
import argparse
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from prebind_shared_gpu import require_initial_binding

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
parser.add_argument('--attempt', type=int, default=1)
args = parser.parse_args()
if args.device == 'cpu':
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    bound = None
else:
    bound = require_initial_binding()

import torch
from torch import nn
from runtime_io import (PROCESS, owned_cache, precision, verify, write, append, stamp,
                        MixedPool, fresh_core, make_controls, load_spatial, acquire_gpu)
from causal_pipeline import CausalEnhancer


def main():
    precision()
    frozen = verify()
    out = PROCESS / f'{args.device}-qualification{args.attempt:03d}'
    out.mkdir(parents=True, exist_ok=False)
    cache = owned_cache(out.name)
    cache.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    complete = False
    counts = dict(discarded_updates=0, backbone_images=0, teacher_images=0, DIS_pairs=0)
    lock = schedule = None
    try:
        pool = MixedPool()
        low, gt, identities = pool.batch(0, 0)
        core = fresh_core(0)
        teacher, provenance = load_spatial(0)
        teacher.eval().requires_grad_(False)
        write(out / 'config.json', dict(at=stamp(), pid=os.getpid(), device=args.device,
              source_freeze=frozen, data=pool.info, teacher=provenance, windows=identities,
              expected_discarded_updates=8, inherited_by_formal_training=False))
        if bound:
            lock, schedule, _ = acquire_gpu(bound, out)
        device = torch.device(args.device)
        core, teacher, low, gt = core.to(device), teacher.to(device), low.to(device), gt.to(device)
        torch.manual_seed(89000)
        artificial = torch.rand(1, 3, 127, 133, device=device)
        observed = []
        def hook(module, inputs, output):
            observed.append(output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1])
        handles = [module.register_forward_hook(hook) for module in core.modules() if isinstance(module, nn.Conv2d)]
        with torch.no_grad():
            toy = core(artificial)
        counts['backbone_images'] += 1
        for handle in handles:
            handle.remove()
        assert torch.equal(toy, artificial)
        assert sum(p.numel() for p in core.parameters()) == 16180
        assert sum(observed) == core.convolution_macs(127, 133)
        write(out / 'artificial.json', dict(identity_difference=0., output_shape=list(toy.shape),
              parameters=16180, observed_convolution_MAC=sum(observed), passed=True,
              full_model_shift_equivariance_claim=False))

        def finish_update(model, optimizer, loss, label):
            assert torch.isfinite(loss)
            loss.backward()
            active = list(model.parameters())
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active)
            norm = float(torch.nn.utils.clip_grad_norm_(active, 1))
            optimizer.step()
            counts['discarded_updates'] += 1
            append(out / 'updates.jsonl', dict(label=label, loss=float(loss.detach()), gradient_norm=norm,
                   all_gradients_present_and_finite=True, discarded_update=counts['discarded_updates']))

        optimizer = torch.optim.Adam(core.parameters(), lr=1e-3)
        for step in range(2):
            began = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                target = teacher(low.flatten(0, 1)).clamp(0, 1)
            value = core(low.flatten(0, 1))
            counts['teacher_images'] += 16
            counts['backbone_images'] += 16
            finish_update(core, optimizer, (value - target).square().mean(), 'warm')
            if schedule:
                torch.cuda.synchronize()
                schedule.yield_after_work(time.perf_counter() - began)
        models = {k: v.to(device) for k, v in make_controls(core).items()}

        def window(model, value):
            result = model.window(value)
            for details in result[-1]:
                counts['backbone_images'] += details['backbone_images']
                counts['DIS_pairs'] += details['DIS_pairs']
            return result

        with torch.no_grad():
            a, _, _, _ = window(models['fixed'], low)
            b, _, _, _ = window(models['learned'], low)
        initial_delta = float((a - b).abs().max())
        assert initial_delta <= 1e-6
        temporal_gradients = {}
        for route, model in models.items():
            expected = 16365 if route == 'learned' else 16180
            assert sum(p.numel() for p in model.parameters()) == expected
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            for step in range(2):
                began = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                inputs = low.detach().clone().requires_grad_(True)
                value, preview, state, details = window(model, inputs)
                assert torch.equal(value[:, 0], preview[:, 0])
                history_gradient = torch.autograd.grad(value[:, -1].sum(), inputs, retain_graph=True)[0][:, :-1]
                temporal_gradients[f'{route}-{step}'] = float(history_gradient.abs().sum())
                assert temporal_gradients[f'{route}-{step}'] == 0 if route == 'spatial' else temporal_gradients[f'{route}-{step}'] > 0
                loss = (value - gt).square().mean()
                if route != 'spatial':
                    loss = .5 * loss + .5 * (preview - gt).square().mean()
                finish_update(model, optimizer, loss, route)
                if schedule:
                    torch.cuda.synchronize()
                    schedule.yield_after_work(time.perf_counter() - began)
            path = cache / f'{route}.pt'
            torch.save(model.state_dict(), path)
            saved = torch.load(path, map_location=device, weights_only=True)
            assert all(torch.isfinite(v).all() for v in saved.values())
            restored = CausalEnhancer(route).to(device)
            restored.load_state_dict(saved, strict=True)
            with torch.no_grad():
                expected_output = window(model.eval(), low)[0]
                actual = window(restored.eval(), low)[0]
            reload_delta = float((expected_output - actual).abs().max())
            assert reload_delta <= 1e-6
            write(out / f'{route}.json', dict(parameters=expected, reload_difference=reload_delta,
                  first_frame_preview_difference=0., saved_all_finite=True, passed=True))
        assert counts['discarded_updates'] == 8
        verify()
        write(out / 'qualification.json', dict(passed=True, fixed_learned_initial_difference=initial_delta,
              temporal_input_gradient_L1=temporal_gradients, **counts,
              scope='Finite engineering qualification on one train window, not DEV quality or innovation.'))
        complete = True
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out / 'runtime.json', dict(at=stamp(), pid=os.getpid(), completed_normally=complete,
              seconds=time.perf_counter() - started, **counts,
              CUDA_initialized=torch.cuda.is_initialized(),
              GPU=None if schedule is None else schedule.summary()))
        if lock:
            lock.close()


if __name__ == '__main__':
    main()
