"""Small CPU feasibility checks; no training or enhancement-quality claim."""
import copy
import json
from pathlib import Path

import torch
from torch import nn

from budget_causal_r33 import BudgetCausal, ROOT, transport


def run():
    torch.set_num_threads(2)
    torch.manual_seed(33)
    assert not torch.cuda.is_initialized()
    net = BudgetCausal().double().eval()
    spatial = copy.deepcopy(net)
    spatial.temporal = False
    assert sum(p.numel() for p in net.parameters()) == 3586
    frames = torch.rand(4, 1, 3, 13, 17, dtype=torch.float64)
    with torch.no_grad():
        initial, state = net(frames[0])
        assert torch.equal(initial, net.backbone(frames[0]))
        assert torch.equal(initial, spatial(frames[0])[0])
        assert state.shape == (1, 8, 4, 5)
        # Explicit activation is essential: the actual readout is zero at init.
        net.readout.weight.normal_(0, .05)
        net.readout.bias.fill_(.01)
        spatial.load_state_dict(net.state_dict())
        first = net(frames[0])[0]
        assert torch.equal(first, net(frames[0], history='current')[0])
        assert torch.equal(first, spatial(frames[0])[0])
        net.reset()
        sequence = torch.stack([net.step(frame) for frame in frames])
        assert not net._state.requires_grad
        stored = net.state_storage_bytes()
        assert stored == state.numel()*state.element_size()
        net.reset()
        changed = frames.clone()
        changed[3] = 1-changed[3]
        prefix = torch.stack([net.step(frame) for frame in changed])
        assert torch.equal(sequence[:3], prefix[:3])
        net.reset()
        reset_output = net.step(frames[2])
        assert torch.equal(reset_output, net(frames[2])[0])
        different = torch.rand(2, 3, 15, 21, dtype=torch.float64)
        assert torch.equal(net.step(different), net(different)[0])
        assert net._state.shape == (2, 8, 4, 6)
        _, past = net(frames[0])
        causal = net(frames[1], past)[0]
        current = net(frames[1], past, history='current')[0]
        history_delta = (causal-current).abs().max().item()
        assert history_delta > 0
        assert torch.equal(current, spatial(frames[1], past)[0])
    # Functional unroll must preserve a path back to the oldest input.
    inputs = frames[:3].clone().requires_grad_()
    state = None
    for frame in inputs:
        prediction, state = net(frame, state)
    prediction.square().mean().backward()
    past_gradient = inputs.grad[0].abs().sum().item()
    field_gradient = net.field.weight.grad[:2].abs().sum().item()
    assert past_gradient > 0 and field_gradient > 0
    assert torch.isfinite(inputs.grad).all() and torch.isfinite(net.field.weight.grad).all()
    # A known fractional x shift and an out-of-bounds fallback.
    source = torch.arange(12, dtype=torch.float64).reshape(1, 1, 3, 4).requires_grad_()
    fallback = torch.full_like(source, -9)
    offset = torch.zeros(1, 2, 3, 4, dtype=torch.float64)
    offset[:, 0] = .5
    offset.requires_grad_()
    warped, valid = transport(source, fallback, offset)
    assert torch.equal(warped[..., :-1], source[..., :-1]+.5)
    assert torch.equal(warped[..., -1], fallback[..., -1])
    assert valid.sum().item() == 9
    warped.sum().backward()
    assert offset.grad[:, 0].sum().item() == 9
    assert source.grad.abs().sum().item() == 9
    # Meta execution counts Conv MACs only; excludes gather, pooling and interpolation.
    meta = BudgetCausal().to('meta').eval()
    macs = []
    handles = []
    for module in meta.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(lambda layer, args, out: macs.append(
                out.numel()*(layer.in_channels//layer.groups)*layer.kernel_size[0]*layer.kernel_size[1])))
    with torch.no_grad():
        _, meta_state = meta(torch.empty(1, 3, 1080, 1920, device='meta'))
    for handle in handles:
        handle.remove()
    assert not torch.cuda.is_initialized()
    return {'parameters_spatial_and_causal': 3586, 'output_shape': list(initial.shape),
            'tested_small_state_shape': [1, 8, 4, 5], 'tested_small_state_storage_bytes_float64': stored,
            'activated_readout_history_max_delta': history_delta,
            'oldest_input_gradient_l1': past_gradient, 'offset_field_gradient_l1': field_gradient,
            'causality_reset_shape_change_current_control': 'passed',
            'fractional_transport_bounds_and_gradients': 'passed',
            'conv_MACs_1080p': sum(macs), 'state_1080p_shape': list(meta_state.shape),
            'state_1080p_bytes_float32': meta_state.numel()*4, 'cuda_initialized': False,
            'scope': 'Untrained CPU float64 functional checks plus meta Conv count; no GPU, quality, speed, peak-memory or novelty evidence. Readout was explicitly activated for gradient/history checks.'}


if __name__ == '__main__':
    result = run()
    out = ROOT/'research/4.temporal-consistency/report/exploration-r33-prototype'
    out.mkdir(parents=True, exist_ok=False)
    (out/'checks.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
