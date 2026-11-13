"""Independent analytic/gradient gates for the training-only temporal target."""
import torch
from torch.nn import functional as F

from budget_causal_r33 import transport
from video_objective_r44 import temporal_mse


def objective_gate(device):
    generator = torch.Generator().manual_seed(4400)
    previous = torch.rand((2, 3, 32, 36), generator=generator)
    current = torch.rand((2, 3, 32, 36), generator=generator)
    offset = .8*torch.rand((2, 2, 32, 36), generator=generator)-.4
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(36), indexing='ij')
    grid = torch.stack([2*(xx+offset[:, 0])/35-1, 2*(yy+offset[:, 1])/31-1], dim=-1)
    expected = F.grid_sample(previous, grid, padding_mode='border', align_corners=True)
    moved, valid = transport(previous.to(device), current.to(device), offset.to(device))
    delta = float((moved-expected.to(device)).abs().masked_select(valid.expand_as(moved)).max())
    assert delta <= 1e-5, delta
    base = torch.rand((2, 3, 32, 36), generator=generator).to(device)
    gt = torch.stack([torch.roll(base, 2*t, dims=-1)+.05*t for t in range(4)], dim=1)
    flow = torch.zeros((6, 2, 32, 36), device=device)
    flow[:, 0] = -2
    prev, cur = gt[:, :-1].reshape(6, 3, 32, 36), gt[:, 1:].reshape(6, 3, 32, 36)
    warped, mask = transport(prev, cur, flow)
    mask[..., :8, :] = mask[..., -8:, :] = False
    mask[..., :, :8] = mask[..., :, -8:] = False
    motion = {'offset': flow, 'mask': mask, 'target_delta': cur-warped}
    zero = float(temporal_mse(gt, motion))
    bias = torch.tensor([.05, -.03, .1], device=device).reshape(1, 1, 3, 1, 1)
    constant = float(temporal_mse(gt+bias, motion))
    assert max(zero, constant) <= 1e-10, (zero, constant)
    pred = gt.clone()
    pred[:, 0] += .1*torch.rand((2, 3, 32, 36), generator=generator).to(device)
    pred.requires_grad_()
    changed = temporal_mse(pred, motion)
    changed.backward()
    norms = [float(pred.grad[:, t].norm()) for t in [0, 1]]
    assert float(changed.detach()) > 0 and all(n > 0 for n in norms)
    assert torch.isfinite(pred.grad).all()
    empty = {**motion, 'mask': torch.zeros_like(mask)}
    blank = gt.clone().requires_grad_()
    invalid = temporal_mse(blank, empty)
    invalid.backward()
    assert float(invalid.detach()) == 0 and torch.isfinite(blank.grad).all()
    assert float(blank.grad.norm()) == 0
    return {'device': str(device), 'continuous_warp_max_abs_delta': delta,
            'GT_prediction_time_MSE': zero, 'constant_residual_time_MSE': constant,
            'perturbed_frame_time_MSE': float(changed.detach()), 'previous_current_gradient_norms': norms,
            'all_invalid_time_MSE': float(invalid.detach()),
            'scope': 'Known-motion synthetic target with genuine brightness change; no data quality or motion-truth claim.'}
