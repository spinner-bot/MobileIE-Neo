"""Bounded actual-window qualification; discarded updates are never checkpoints."""
import argparse
import json
import time
import traceback

import cv2
import torch

from native_moment_core_r73 import (BASE, ROOT, MODES, PARAMETERS, PROTOCOL,
                                    folded, make_models, mix_moments, prediction)
from budget_causal_r33 import transport
from mixed_anchor_r40 import MixedPool
from training_form_r41 import configure_precision, cuda_headroom
from dynamic_training_r34 import digest


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def export_check(net, low):
    net.eval()
    with torch.no_grad():
        original = prediction(net, low)
        slim = folded(net).to(low.device)
        deployed = prediction(slim, low)
    delta = float((original-deployed).abs().max())
    assert torch.isfinite(original).all() and torch.isfinite(deployed).all()
    return dict(max_abs_delta=delta, passed=delta <= 1e-4,
                parameters=sum(p.numel() for p in slim.parameters()))


def artificial_gate():
    value = torch.full((1, 12, 8, 10), 2.)
    zero = torch.zeros_like(value)
    mean, second, dispersion = mix_moments(value, (zero, zero), .5)
    assert torch.equal(mean, torch.ones_like(value))
    assert torch.equal(second, value) and torch.equal(dispersion, torch.ones_like(value))
    # Transport the noncentral second moment, rather than treating independent
    # random variables or sensor noise as already identified.
    m = torch.arange(10.).reshape(1, 1, 1, 10).expand_as(value)
    pair = torch.cat([m, m.square()], dim=1)
    offset = torch.zeros((1, 2, 8, 10)); offset[:, 0] = .5
    moved, valid = transport(pair, pair, offset)
    a, b = moved.chunk(2, dim=1)
    variance_error = float(((b-a.square())[:, :, :, :-1]-.25).abs().max())
    assert variance_error <= 1e-6 and valid[:, :, :, :-1].all() and not valid[:, :, :, -1].any()
    moved, valid = transport(pair, torch.cat([value, value.square()], dim=1), offset+100)
    a, b, v = mix_moments(value, moved.chunk(2, dim=1), valid.float()*.5)
    assert torch.equal(a, value) and torch.equal(b, value.square()) and not v.any()
    return dict(passed=True, analytic_half_pixel_mixture_variance_max_error=variance_error,
                invalid_support_returns_current=True, calibrated_sensor_or_posterior=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--admission-attempt', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(2); cv2.setNumThreads(1); configure_precision()
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    assert not torch.cuda.is_initialized()
    out = BASE/f'process/exploration-r73/{args.device}-gate{args.admission_attempt:03d}'; out.mkdir(parents=True, exist_ok=False)
    start, complete, updates, records = time.perf_counter(), False, 0, []
    try:
        write(out/'artificial.json', artificial_gate())
        pool = MixedPool()
        prepared = {seed: (make_models(seed), pool.batch(seed, 8000)) for seed in [0, 1]}
        source_names = ['native_moment_core_r73.py', 'check_native_moment_r73.py', 'mixed_anchor_r40.py',
                        'training_form_r41.py', 'budget_causal_r33.py', 'dynamic_training_r34.py']
        write(out/'config.json', dict(device=args.device, seeds=[0, 1], discarded_updates_each=2,
              expected_discarded_updates=12, data_step=8000, source_sha256={n: digest(BASE/'scripts'/n) for n in source_names},
              protocol_sha256=digest(PROTOCOL), data=pool.info,
              scope='Training-source window qualification only. No DEV, formal training, quality ranking or novelty proof.'))
        if args.device == 'cuda':
            cpu = json.loads((BASE/'process/exploration-r73/cpu-gate001/runtime.json').read_text())
            assert cpu['passed']
            admission = cuda_headroom(); write(out/'internal-admission.json', admission)
            if not admission['allowed']:
                raise SystemExit(75)
        for seed, ((nets, info), (low, gt, context)) in prepared.items():
            low, gt = low.to(args.device), gt.to(args.device)
            nets = {name: net.to(args.device).eval() for name, net in nets.items()}
            pointers = [set(p.data_ptr() for p in net.parameters()) for net in nets.values()]
            assert all(not pointers[i] & pointers[j] for i in range(3) for j in range(i))
            with torch.no_grad():
                current = prediction(nets['moment'], low, 'current')
                original = torch.stack([nets['moment'].backbone(frame) for frame in low.unbind(1)], dim=1)
                causal = prediction(nets['moment'], low)
                mean_only = prediction(nets['mean'], low)
            initial = dict(current_minus_original_max=float((current-original).abs().max()),
                           mean_minus_moment_max=float((causal-mean_only).abs().max()),
                           first_frame_causal_minus_current_max=float((causal[:, 0]-current[:, 0]).abs().max()),
                           actual_history_minus_current_max=float((causal-current).abs().max()))
            assert all(initial[k] <= 1e-5 for k in initial if k != 'actual_history_minus_current_max')
            assert initial['actual_history_minus_current_max'] > 0
            write(out/f'seed{seed}-initial.json', dict(initial=initial, initialization=info, context=context))
            for name, net in nets.items():
                before = export_check(net, low); assert before['passed'] and before['parameters'] == PARAMETERS[name]
                optimizer = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=1e-4)
                losses, norms = [], []
                for step in range(2):
                    net.train(); optimizer.zero_grad(set_to_none=True)
                    output = prediction(net, low); loss = (output.clamp(0, 1)-gt).square().mean()
                    assert torch.isfinite(loss)
                    loss.backward()
                    active = [p for p in net.parameters() if p.requires_grad]
                    norm = float(torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True))
                    if step == 0:
                        gradient = dict(field_last=float(net.field[-1].weight.grad.abs().sum()),
                                        rho=None if net.rho is None else float(net.rho.grad.abs().sum()))
                        assert gradient['field_last'] > 0
                        if name == 'moment': assert gradient['rho'] > 0
                    optimizer.step(); updates += 1; losses.append(float(loss.detach())); norms.append(norm)
                after = export_check(net, low); assert after['passed']
                differentiable = low.detach().clone().requires_grad_()
                last = prediction(net.eval(), differentiable)
                grad, = torch.autograd.grad(last[:, -1].clamp(0, 1).square().mean(), differentiable)
                past_grad = float(grad[:, :-1].abs().sum())
                assert torch.isfinite(grad).all()
                assert past_grad == 0 if name == 'spatial' else past_grad > 0
                with torch.no_grad():
                    own_current = prediction(net, low, 'current')
                    own_causal = prediction(net, low)
                    first = float((own_current[:, 0]-own_causal[:, 0]).abs().max()); assert first <= 1e-5
                    net.reset()
                    for frame in low.unbind(1): net.step(frame)
                    state_bytes = net.state_storage_bytes()
                    expected = (12 if name == 'mean' else 24)*low.shape[0]*low.shape[-2]*low.shape[-1]*4
                    assert state_bytes == expected
                    net.reset(); assert net.state_storage_bytes() == 0
                record = dict(seed=seed, mode=name, initial_export=before, after_two_discarded_updates_export=after,
                              losses=losses, gradient_norms=norms, first_update_gradients=gradient,
                              last_frame_gradient_to_past_inputs=past_grad, first_frame_own_current_delta=first,
                              state_bytes_actual_window=state_bytes,
                              state_bytes_1080p=(12 if name == 'mean' else 24)*1080*1920*4,
                              scope='Functional qualification, no final quality result.')
                write(out/f'seed{seed}-{name}.json', record); records.append(record)
                print(json.dumps(dict(seed=seed, mode=name, passed=True, discarded_updates=updates)), flush=True)
        assert updates == 12 and len(records) == 6
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(passed=complete, completed_normally=complete, device=args.device,
              discarded_SGD_updates=updates, formal_SGD_updates=0, DEV_conditions=0,
              CUDA_initialized=torch.cuda.is_initialized(), seconds=time.perf_counter()-start,
              ended=time.strftime('%Y-%m-%dT%H:%M:%S%z'), completed_controls=len(records)))


if __name__ == '__main__':
    main()
