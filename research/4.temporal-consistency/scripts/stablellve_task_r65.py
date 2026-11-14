"""Published StableLLVE L1/commutator form on the fixed project training windows.

Official warp geometry is retained, including endpoint grid with align_corners=False.
RGB is converted to the official BGR network convention at every neural call.
"""
import numpy as np
import torch
import torch.nn.functional as F

from half_motion_r4 import half_forward_backward
from stablellve_adapter_r15 import load_model

ARMS = ('spatial_l1', 'commute20')
PARAMETERS = 4316259


def initial_model():
    model = load_model(torch.device('cpu')).requires_grad_(True)
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    return model


def forward_rgb(model, images):
    return model(images[:, [2, 1, 0]])[:, [2, 1, 0]]


def official_warp(image, flow):
    assert flow.shape == (len(image), 2, *image.shape[-2:])
    height, width = image.shape[-2:]
    horizontal = torch.linspace(-1, 1, width, device=image.device, dtype=image.dtype)
    vertical = torch.linspace(-1, 1, height, device=image.device, dtype=image.dtype)
    grid_y, grid_x = torch.meshgrid(vertical, horizontal, indexing='ij')
    grid = torch.stack((grid_x, grid_y), -1)[None]
    scale = image.new_tensor([(width-1)/2, (height-1)/2]).view(1, 1, 1, 2)
    grid = grid+flow.permute(0, 2, 3, 1)/scale
    return F.grid_sample(image, grid, mode='bilinear', padding_mode='reflection', align_corners=False)


def prepare_offsets(target):
    assert target.device.type == 'cpu' and target.shape[1:3] == (4, 3)
    values = target.permute(0, 1, 3, 4, 2).contiguous().numpy()*255
    fields = []
    for sample in values:
        fields.append(np.stack([half_forward_backward(previous, current)[0]
                                for previous, current in zip(sample[:-1], sample[1:])]))
    offset = torch.from_numpy(np.stack(fields)).permute(0, 1, 4, 2, 3).contiguous()
    assert offset.shape == (len(target), 3, 2, *target.shape[-2:]) and torch.isfinite(offset).all()
    return offset


def loss_terms(model, low, target, offset, arm):
    assert arm in ARMS and low.shape == target.shape
    prediction = torch.stack([forward_rgb(model, frame) for frame in low.unbind(1)], 1)
    spatial = F.l1_loss(prediction, target)
    if arm == 'spatial_l1':
        return spatial, dict(spatial=spatial, consistency=spatial.new_zeros(()), prediction=prediction)
    transformed = torch.stack([forward_rgb(model, official_warp(low[:, t], offset[:, t]))
                               for t in range(3)], 1)
    moved = torch.stack([official_warp(prediction[:, t], offset[:, t]) for t in range(3)], 1)
    consistency = F.l1_loss(transformed, moved)
    return spatial+20*consistency, dict(spatial=spatial, consistency=consistency,
                                      prediction=prediction, transformed=transformed, moved=moved)


def check_one(model, low, target, offset, arm):
    model.train()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    loss, terms = loss_terms(model, low, target, offset, arm)
    assert terms['prediction'].shape == target.shape and torch.isfinite(terms['prediction']).all()
    assert torch.isfinite(loss) and float(terms['spatial'].detach()) > 0
    paths = {}
    if arm == 'commute20':
        gradients = torch.autograd.grad(terms['consistency'], (terms['transformed'], terms['moved']), retain_graph=True)
        for name, gradient in zip(('F_W_input', 'W_F_input'), gradients):
            assert torch.isfinite(gradient).all()
            paths[name] = float(gradient.double().square().sum().sqrt())
        assert min(paths.values()) > 0
    loss.backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing and all(torch.isfinite(p.grad).all() for p in model.parameters())
    gradient = {name: float(p.grad.detach().double().square().sum().sqrt())
                for name, p in model.named_parameters() if name in ('inc.double_conv.0.weight', 'outc.conv.weight')}
    assert len(gradient) == 2 and min(gradient.values()) > 0
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
    optimizer.step()
    change = {name: float((p-before[name]).detach().abs().max())
              for name, p in model.named_parameters() if name in gradient}
    assert min(change.values()) > 0 and all(torch.isfinite(p).all() for p in model.parameters())
    ramp = torch.linspace(0, 1, low.shape[-1], device=low.device)[None, None, None]
    ramp = ramp.expand(1, 3, *low.shape[-2:])
    zero = torch.zeros((1, 2, *low.shape[-2:]), device=low.device)
    zero_flow_error = float((official_warp(ramp, zero)-ramp).abs().max())
    assert zero_flow_error > 0  # Preserved upstream endpoint/default alignment mismatch.
    return dict(passed=True, arm=arm, parameters=PARAMETERS, total=float(loss.detach()),
                spatial_L1=float(terms['spatial'].detach()), consistency_L1=float(terms['consistency'].detach()),
                consistency_path_gradient_L2=paths, endpoint_gradient_L2=gradient,
                endpoint_max_parameter_change=change, gradient_norm_before_clip=norm,
                original_warp_zero_flow_ramp_max_error=zero_flow_error,
                discarded_SGD_updates=1, inherited_into_formal_training=False)
