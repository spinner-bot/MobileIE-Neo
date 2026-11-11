"""Finite aligned body-FST operator controls; bilinear operations are known prior art."""
import torch
from torch import nn

MODES = ('copy', 'post_transport', 'post_after_warp', 'pre_square', 'cross')


def gather_features(previous, current, flow, mask):
    """OpenCV-style 1/32-grid bilinear gather, with differentiable feature values.

    Flow is fixed current-to-previous, never differentiated. Bounds are tested on
    unquantized coordinates, then combined with the supplied acceptance mask.
    Replicated borders agree with the RGB reference; rejected pixels use current.
    """
    if previous.shape != current.shape or current.ndim != 4:
        raise ValueError('matching NCHW features required')
    n, channels, height, width = current.shape
    flow = torch.as_tensor(flow, dtype=current.dtype, device=current.device).detach()
    mask = torch.as_tensor(mask, dtype=torch.bool, device=current.device)
    if flow.shape != (n, height, width, 2) or mask.shape != (n, height, width):
        raise ValueError('batched native-grid backward flow and mask required')
    if not bool(torch.isfinite(flow).all()):
        raise ValueError('nonfinite flow')
    y, x = torch.meshgrid(torch.arange(height, device=current.device, dtype=current.dtype),
                          torch.arange(width, device=current.device, dtype=current.dtype), indexing='ij')
    sx, sy = x+flow[..., 0], y+flow[..., 1]
    valid = mask & (sx >= 0) & (sx <= width-1) & (sy >= 0) & (sy <= height-1)
    qx, qy = torch.round(sx*32).long(), torch.round(sy*32).long()
    ix, iy = torch.div(qx, 32, rounding_mode='floor'), torch.div(qy, 32, rounding_mode='floor')
    fx, fy = (qx-32*ix).to(current.dtype)/32, (qy-32*iy).to(current.dtype)/32
    flat = previous.flatten(2)
    def corner(dx, dy):
        index = ((iy+dy).clamp(0, height-1)*width+(ix+dx).clamp(0, width-1)).flatten(1)
        return torch.gather(flat, 2, index[:, None].expand(-1, channels, -1)).reshape_as(current)
    fx, fy = fx[:, None], fy[:, None]
    warped = corner(0, 0)*((1-fx)*(1-fy))+corner(1, 0)*(fx*(1-fy))
    warped = warped+corner(0, 1)*((1-fx)*fy)+corner(1, 1)*(fx*fy)
    return torch.where(valid[:, None], warped, current), valid


class TransportFST(nn.Module):
    """Same 4075 parameters; cache one unfused current feature, no recurrent feedback."""
    def __init__(self, backbone, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.backbone, self.mode = backbone, mode
        self.state = None

    def features(self, image):
        return self.backbone.body.block1(self.backbone.head(image))

    def ordinary(self, u):
        body = self.backbone.body
        return (body.weight1*u)*(body.weight2*u)+body.bias

    def finish(self, body):
        gate = self.backbone.att(body)
        maximum = torch.max(gate*body, dim=1, keepdim=True)[0]
        return self.backbone.tail(torch.mul(self.backbone.att1(maximum), gate)*body)

    def combine(self, current, previous=None, flow=None, mask=None):
        if self.mode == 'copy' or previous is None:
            return self.ordinary(current)
        if self.mode == 'post_transport':
            past, _ = gather_features(previous, self.ordinary(current), flow, mask)
            return .5*(self.ordinary(current)+past)
        aligned, _ = gather_features(previous, current, flow, mask)
        if self.mode == 'post_after_warp':
            return .5*(self.ordinary(current)+self.ordinary(aligned))
        if self.mode == 'pre_square':
            return self.ordinary(.5*(current+aligned))
        body = self.backbone.body
        return (body.weight1*current)*(body.weight2*aligned)+body.bias

    def transform(self, current, previous=None, flow=None, mask=None):
        u = self.features(current)
        past = None if previous is None or self.mode == 'copy' else self.features(previous)
        if past is not None and self.mode == 'post_transport':
            past = self.ordinary(past)
        return self.finish(self.combine(u, past, flow, mask))

    def reset(self):
        self.state = None

    def step(self, current, flow=None, mask=None, history='causal'):
        if history not in ('current', 'causal'):
            raise ValueError('registered history required')
        u = self.features(current)
        previous = self.state
        if history == 'current' or previous is None or previous.shape != u.shape:
            previous = None
        result = self.finish(self.combine(u, previous, flow, mask))
        self.state = None if self.mode == 'copy' else (self.ordinary(u) if self.mode == 'post_transport' else u).detach().clone()
        return result

    def state_storage_bytes(self):
        return 0 if self.state is None else self.state.untyped_storage().nbytes()
