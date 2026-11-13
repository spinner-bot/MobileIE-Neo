"""R74 geometry, compact state and eight discarded actual-training updates."""
import argparse
import copy
from datetime import datetime
import io
import json
import os
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import cv2
import torch

from motion_transport_r74 import (BASE, ROOT, PROTOCOL, MODES, MotionTransport, RGBStream,
        fuse, sequence, make_models, load_spatial, frozen_raw, teacher_proxy, proxy_loss)
from mixed_anchor_r40 import MixedPool
from training_form_r41 import configure_precision, cuda_headroom
from dynamic_training_r34 import digest


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def geometry():
    previous = torch.arange(80.).reshape(1, 1, 8, 10).repeat(1, 3, 1, 1)/100
    current = torch.zeros_like(previous)
    flow = torch.zeros(1, 2, 2, 3); flow[:, 0] = 1
    logit = torch.zeros(1, 1, 2, 3)
    value, valid = fuse(current, previous, (flow, logit))
    expected = torch.zeros_like(previous); expected[..., :-1] = .25*previous[..., 1:]
    delta = float((value-expected).abs().max())
    assert delta <= 1e-6 and valid[..., :-1].all() and not valid[..., -1].any()
    outside, valid = fuse(current, previous, (flow+1000, logit))
    assert not valid.any() and torch.equal(outside, current)
    net = make_models(0)['task'].eval()
    raw = torch.full((1, 3, 3, 33, 47), .4)
    with torch.no_grad():
        output, fields = sequence(net, raw)
        zero, _ = sequence(net, raw, zero_flow=True)
    assert torch.equal(output, raw) and torch.equal(output, zero)
    assert all(p[0].shape == (1, 2, 9, 12) and p[1].shape == (1, 1, 9, 12) for p in fields)
    assert all(p[0].abs().max() == 0 and p[1].abs().max() == 0 for p in fields)
    return dict(known_positive_x_shift_max_delta=delta, out_of_bounds_current_fallback=True,
                odd33x47_and_constant_passed=True, initial_flow_and_logit_zero=True,
                motion_parameters=sum(p.numel() for p in net.parameters()), new_CNN_scope='Artificial motion only; no source pixels.')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--admission-attempt', type=int, default=1); args = parser.parse_args()
    torch.set_num_threads(2); cv2.setNumThreads(1); configure_precision()
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False; assert not torch.cuda.is_initialized()
    out = BASE/f'process/exploration-r74/{args.device}-gate{args.admission_attempt:03d}'
    out.mkdir(parents=True, exist_ok=False)
    began, passed, updates, records = time.perf_counter(), False, 0, []
    try:
        artificial = geometry(); assert artificial['motion_parameters']+4075 < 50000
        write(out/'artificial.json', artificial)
        pool = MixedPool(); prepared = {}
        for seed in [0, 1]:
            spatial, source = load_spatial(seed); low, gt, context = pool.batch(seed, 8000)
            prepared[seed] = (spatial, source, make_models(seed), low, gt, context, teacher_proxy(gt))
        write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), device=args.device,
              data_step=8000, discarded_updates_expected=8, protocol_sha256=digest(PROTOCOL), data=pool.info,
              source_sha256={n: digest(BASE/'scripts'/n) for n in ['motion_transport_r74.py', 'check_motion_transport_r74.py',
                    'mixed_anchor_r40.py', 'half_motion_r4.py', 'budget_causal_r33.py', 'training_form_r41.py']},
              scope='Two actual training-source windows, not DEV; eight discarded motion updates, no formal training or quality claim.'))
        if args.device == 'cuda':
            assert json.loads((BASE/'process/exploration-r74/cpu-gate001/runtime.json').read_text())['passed']
            admission = cuda_headroom(); write(out/'internal-admission.json', admission)
            if not admission['allowed']: raise SystemExit(75)
        for seed, (spatial, source, nets, low, gt, context, target) in prepared.items():
            spatial = spatial.to(args.device).eval(); low, gt = low.to(args.device), gt.to(args.device)
            target = tuple(t.to(args.device) for t in target)
            nets = {name: net.to(args.device) for name, net in nets.items()}
            assert all(torch.equal(a, b) for a, b in zip(nets['task'].parameters(), nets['proxy'].parameters()))
            assert not {p.data_ptr() for p in nets['task'].parameters()} & {p.data_ptr() for p in nets['proxy'].parameters()}
            raw = frozen_raw(spatial, low); assert raw.min() >= 0 and raw.max() <= 1
            write(out/f'seed{seed}-initial.json', dict(spatial=source, context=context,
                  initial_motion_parameters_equal=True, independent_motion_parameters=True,
                  spatial_eval_and_frozen=True, proxy_is_estimated_GT_motion_not_true_flow=True))
            for name in MODES:
                net = nets[name].train(); optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
                losses, gradients = [], None
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    output, fields = sequence(net, raw)
                    loss = ((output.clamp(0, 1)-gt).square().mean() if name == 'task'
                            else proxy_loss(fields, target)[0])
                    assert torch.isfinite(loss); loss.backward()
                    norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True))
                    if step == 0:
                        gradients = dict(fine_flow_weight_L1=float(net.fine[-1].weight.grad[:2].abs().sum()),
                              fine_confidence_weight_L1=float(net.fine[-1].weight.grad[2:].abs().sum()))
                        assert all(value > 0 for value in gradients.values())
                    assert all(p.grad is None for p in spatial.parameters())
                    optimizer.step(); updates += 1
                    losses.append(dict(loss=float(loss.detach()), gradient_norm_before_clip=norm))
                net.eval()
                with torch.no_grad():
                    expected, _ = sequence(net, raw)
                    stream = RGBStream(net); control = RGBStream(net, zero_flow=True); values = []
                    for frame in raw.unbind(1):
                        value, field = stream.step(frame); control.step(frame, fields=field); values.append(value)
                    actual = torch.stack(values, dim=1)
                    streaming_delta = float((actual-expected).abs().max()); assert streaming_delta <= 1e-6
                    assert torch.equal(actual[:, 0], raw[:, 0]) and actual.min() >= -1e-6 and actual.max() <= 1+1e-6
                    state_bytes = stream.state_storage_bytes()
                    assert state_bytes == 6*raw.shape[0]*raw.shape[-2]*raw.shape[-1]*4
                    # Deliberately change dimensions: the old buffers must reset.
                    changed = raw[:, 0, :, :-1, :-1]; reset_value, reset_field = stream.step(changed)
                    assert torch.equal(reset_value, changed) and reset_field is None
                    stream.reset(); assert stream.state_storage_bytes() == 0
                    buffer = io.BytesIO(); torch.save(net.state_dict(), buffer); buffer.seek(0)
                    restored = copy.deepcopy(net)
                    restored.load_state_dict(torch.load(buffer, map_location=args.device, weights_only=True), strict=True)
                    roundtrip, _ = sequence(restored, raw)
                    reload_delta = float((roundtrip-expected).abs().max()); assert reload_delta <= 1e-6
                differentiable = raw.detach().clone().requires_grad_()
                prediction, _ = sequence(net, differentiable)
                gradient, = torch.autograd.grad(prediction[:, -1].square().mean(), differentiable)
                history_gradient = float(gradient[:, :-1].abs().sum())
                assert torch.isfinite(gradient).all() and history_gradient > 0
                record = dict(seed=seed, mode=name, losses=losses, initial_head_gradients=gradients,
                      motion_parameters=sum(p.numel() for p in net.parameters()), frozen_spatial_parameters=4075,
                      streaming_max_delta=streaming_delta, save_reload_max_delta=reload_delta,
                      actual_state_bytes=state_bytes, state_bytes1080p_FP32=49766400,
                      first_frame_exact_current=True, changed_size_resets=True,
                      last_output_gradient_to_previous_raw=history_gradient)
                records.append(record); write(out/f'seed{seed}-{name}.json', record)
            del spatial, nets, low, gt, raw
        assert updates == 8; passed = True
        write(out/'gate.json', dict(passed=True, modes=records, artificial=artificial,
              formal_SGD_updates=0, DEV_quality_rows=0, qualification_updates_inherited=False))
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(passed=passed, discarded_motion_updates=updates, formal_updates=0,
              DEV_rows=0, CUDA_initialized=torch.cuda.is_initialized(), seconds=time.perf_counter()-began,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
