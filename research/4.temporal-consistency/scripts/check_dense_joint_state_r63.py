"""One CPU qualification, no images, optimizer updates, or CUDA initialization."""
import copy
import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import traceback

import torch

from dense_joint_state_r63 import folded, make_models, prediction
from dynamic_training_r34 import BASE, digest


def delta(a, b):
    return float((a.detach()-b.detach()).abs().max())


def gradient_norm(module):
    values = [p.grad for p in module.parameters() if p.requires_grad]
    assert values and all(v is not None and torch.isfinite(v).all() for v in values)
    return float(sum(v.detach().square().double().sum() for v in values).sqrt())


def norms(net):
    return {name: gradient_norm(getattr(net, name))
            for name in ['backbone', 'project', 'mix', 'field', 'readout']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', choices=['002'], required=True)
    args = parser.parse_args()
    target = BASE/f'process/exploration-r63/cpu-gate{args.attempt}.json'
    assert not target.exists(), 'Retain failed qualifications; never overwrite or auto retry'
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    started = time.perf_counter()
    sources = ['check_dense_joint_state_r63.py', 'dense_joint_state_r63.py',
               'budget_causal_r33.py', 'training_form_r41.py']
    record = {'created_at': datetime.now().astimezone().isoformat(), 'passed': False,
              'formal_optimizer_updates': 0, 'discarded_optimizer_updates': 0,
              'real_image_observations': 0, 'seeds': [],
              'source_sha256': {s: digest(Path(__file__).parent/s) for s in sources},
              'protocol_sha256': digest(BASE/'docs/exploration-r63-dense-joint-state-qualification-protocol.md'),
              'scope': 'Structural qualification only. Perturbed readout is discarded, not learned or selected.'}
    try:
        for seed in [0, 1]:
            initialized, info = make_models(seed)
            item = {'seed': seed, 'initialization': info, 'cases': []}
            record['seeds'].append(item)
            for case, shape in enumerate([(1, 4, 3, 17, 19), (2, 4, 3, 16, 20)]):
                # Keep every shape's BN buffers at the common inherited state.
                nets = {name: copy.deepcopy(net) for name, net in initialized.items()}
                torch.manual_seed(63600+10*seed+case)
                clip, gt = torch.rand(shape), torch.rand(shape)
                row = {'shape': list(shape), 'initial_deltas': [], 'initial_gradients': {},
                       'activated_gradients': {}, 'history_gradients': {}, 'export': {}}
                item['cases'].append(row)
                # Original pooled BN has one value/channel at batch1 in train mode.
                for training in ([False] if shape[0] == 1 else [True, False]):
                    with torch.no_grad():
                        values = {name: prediction(net.train(training), clip) for name, net in nets.items()}
                    for name, value in values.items():
                        error = delta(value, values['plain'])
                        row['initial_deltas'].append({'training': training, 'model': name, 'max_abs': error})
                        assert error <= 1e-5
                for name in ['spatial', 'causal']:
                    net = nets[name].train(shape[0] > 1)
                    net.zero_grad(set_to_none=True)
                    (prediction(net, clip)-gt).square().mean().backward()
                    grad = norms(net)
                    row['initial_gradients'][name] = grad
                    assert grad['backbone'] > 0 and grad['readout'] > 0
                    assert all(grad[key] == 0 for key in ['project', 'mix', 'field'])
                    with torch.no_grad():
                        # The same fixed perturbation in both controls, no optimizer.
                        generator = torch.Generator().manual_seed(63900+seed)
                        net.readout.weight.copy_(1e-3*torch.randn(net.readout.weight.shape, generator=generator))
                    net.zero_grad(set_to_none=True)
                    (prediction(net, clip)-gt).square().mean().backward()
                    row['activated_gradients'][name] = norms(net)
                    assert all(row['activated_gradients'][name][key] > 0
                               for key in ['backbone', 'project', 'field', 'readout'])
                    # The field is zero at init, so mix gradients remain blocked until it changes.
                    net.eval().zero_grad(set_to_none=True)
                    differentiable = clip.clone().requires_grad_()
                    output = prediction(net, differentiable)
                    grad_input, = torch.autograd.grad(output[:, -1].square().mean(), differentiable)
                    past = float(grad_input[:, :-1].abs().sum())
                    row['history_gradients'][name] = {'past_input_abs_sum': past,
                                                      'current_input_abs_sum': float(grad_input[:, -1].abs().sum())}
                    assert torch.isfinite(grad_input).all()
                    assert past > 0 if name == 'causal' else past == 0
                with torch.no_grad():
                    causal = nets['causal'].eval()
                    full = prediction(causal, clip)
                    current = prediction(causal, clip, history='current')
                    future_changed = clip.clone()
                    future_changed[:, 2:] = 1-future_changed[:, 2:]
                    changed = prediction(causal, future_changed)
                    row['first_frame_current_delta'] = delta(full[:, 0], current[:, 0])
                    row['changed_future_prefix_delta'] = delta(full[:, :2], changed[:, :2])
                    row['activated_history_difference'] = delta(full[:, 1:], current[:, 1:])
                    assert row['first_frame_current_delta'] == row['changed_future_prefix_delta'] == 0
                    assert row['activated_history_difference'] > 0
                    for name, net in nets.items():
                        net.eval()
                        deploy = folded(net)
                        error = delta(prediction(net, clip), prediction(deploy, clip))
                        count = sum(p.numel() for p in deploy.parameters())
                        row['export'][name] = {'max_abs_delta': error, 'parameters': count}
                        assert error <= 1e-4
                        assert count == (4075 if name == 'plain' else 4618)
                        if name != 'plain':
                            deploy.reset()
                            for frame in clip.unbind(1):
                                deploy.step(frame)
                            expected = shape[0]*8*((shape[-2]+3)//4)*((shape[-1]+3)//4)*4
                            observed = deploy.state_storage_bytes()
                            assert observed == expected and not deploy._state.requires_grad
                            resized = torch.rand(shape[0], 3, 12, 16)
                            reset_output = deploy.step(resized)
                            fresh_output, _ = deploy(resized)
                            reset_error = delta(reset_output, fresh_output)
                            assert reset_error == 0
                            row['export'][name].update(state_storage_bytes=observed,
                                expected_state_storage_bytes=expected, resized_reset_delta=reset_error)
        record['persistent_FP32_state_bytes_batch1'] = {
            f'{height}x{width}': 8*((height+3)//4)*((width+3)//4)*4
            for height, width in [(720, 1280), (1080, 1920)]}
        record['state_scope'] = 'Persistent state only, excludes activations, temporary gather storage and runtime latency.'
        record['passed'] = True
    except Exception:
        record['failure'] = traceback.format_exc()
        raise
    finally:
        record.update(wall_seconds=time.perf_counter()-started,
                      cuda_initialized=torch.cuda.is_initialized(),
                      finished_at=datetime.now().astimezone().isoformat())
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'passed': record['passed'], 'wall_seconds': record['wall_seconds'],
                          'result': str(target)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
