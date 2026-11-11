"""Finite one-frame causal controls around the official slim body FST."""
import torch
from torch import nn

MODES = ('plain', 'spatial_bilinear', 'temporal_bilinear', 'temporal_post',
         'tsm_learned', 'tsm_fixed', 'spatial_post')


def coarse_pool(x):
    """Ceil-sized 4x mean pooling with edge replication, no scatter backward."""
    h, w = x.shape[-2:]
    if h < 1 or w < 1:
        raise ValueError('nonempty spatial dimensions required')
    ph, pw = (-h) % 4, (-w) % 4
    if pw:
        x = torch.cat([x, x[..., -1:].expand(*x.shape[:-1], pw)], -1)
    if ph:
        x = torch.cat([x, x[..., -1:, :].expand(*x.shape[:-2], ph, x.shape[-1])], -2)
    return x.reshape(*x.shape[:-2], (h+ph)//4, 4, (w+pw)//4, 4).mean((-3, -1))


def coarse_up(x, size):
    """Factor-four bilinear align_corners=False, then remove pooling padding."""
    left = torch.cat([x[..., :1], x[..., :-1]], -1)
    right = torch.cat([x[..., 1:], x[..., -1:]], -1)
    x = torch.stack([.375*left+.625*x, .125*left+.875*x,
                     .875*x+.125*right, .625*x+.375*right], -1).flatten(-2)
    above = torch.cat([x[..., :1, :], x[..., :-1, :]], -2)
    below = torch.cat([x[..., 1:, :], x[..., -1:, :]], -2)
    x = torch.stack([.375*above+.625*x, .125*above+.875*x,
                     .875*x+.125*below, .625*x+.375*below], -2).flatten(-3, -2)
    return x[..., :size[0], :size[1]]


class CausalFST(nn.Module):
    def __init__(self, backbone, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.backbone, self.mode = backbone, mode
        self.history_mode = 'causal'
        self.state = None
        self.state_input_shape = None
        if mode in ('spatial_bilinear', 'temporal_bilinear', 'temporal_post', 'spatial_post'):
            self.branch = nn.Conv2d(12, 12, 3, padding=1, groups=12, bias=False)
            nn.init.zeros_(self.branch.weight)
        if mode == 'tsm_learned':
            self.gamma = nn.Parameter(torch.zeros(()))

    def reset(self, history_mode='causal'):
        if history_mode not in ('causal', 'current', 'off'):
            raise ValueError(history_mode)
        self.state, self.history_mode = None, history_mode
        self.state_input_shape = None

    def _features(self, x):
        return self.backbone.body.block1(self.backbone.head(x))

    def _ordinary(self, u):
        b = self.backbone.body
        return (b.weight1*u)*(b.weight2*u)+b.bias

    def _state(self, u):
        if self.mode.startswith('tsm_'):
            return u[:, :3]
        return coarse_pool(self._ordinary(u) if self.mode.endswith('_post') else u)

    def _body(self, u, current, previous):
        if self.mode == 'plain':
            return self._ordinary(u)
        if self.history_mode == 'off':
            return self._ordinary(u)
        if self.history_mode == 'current' or self.mode.startswith('spatial_'):
            previous = current
        if self.mode.startswith('tsm_'):
            gamma = self.gamma if self.mode == 'tsm_learned' else 1.
            shifted = u[:, :3]+gamma*(previous-u[:, :3])
            return self._ordinary(torch.cat([shifted, u[:, 3:]], 1))
        delta = coarse_up(self.branch(previous), u.shape[-2:])
        if self.mode.endswith('_post'):
            return self._ordinary(u)+delta
        b = self.backbone.body
        return (b.weight1*u)*(b.weight2*(u+delta))+b.bias

    def _finish(self, body):
        att = self.backbone.att(body)
        maximum = torch.max(att*body, dim=1, keepdim=True)[0]
        att1 = self.backbone.att1(maximum)
        return self.backbone.tail(torch.mul(att1, att)*body)

    def forward_pair(self, x):
        """[B,2,C,H,W], differentiable through the first frame's raw state."""
        if x.ndim != 5 or x.shape[1] != 2:
            raise ValueError('two-frame paired input required')
        b, _, _, h, w = x.shape
        u = self._features(x.flatten(0, 1))
        if self.mode == 'plain':
            return self._finish(self._ordinary(u)).reshape(b, 2, 3, h, w)
        current = self._state(u)
        paired = current.reshape(b, 2, *current.shape[1:])
        previous = torch.stack([paired[:, 0], paired[:, 0]], 1).flatten(0, 1)
        return self._finish(self._body(u, current, previous)).reshape(b, 2, 3, h, w)

    def forward(self, x):
        """A stream step. Always reset at sequence boundaries; no future frames."""
        u = self._features(x)
        if self.mode == 'plain':
            return self._finish(self._ordinary(u))
        current = self._state(u)
        previous = self.state
        if previous is None or previous.shape != current.shape or self.state_input_shape != x.shape:
            previous = current
        result = self._finish(self._body(u, current, previous))
        # clone is essential: a 3-channel view would retain all 12 channels.
        self.state = None if self.mode.startswith('spatial_') else current.detach().clone()
        self.state_input_shape = x.shape
        return result

    def branch_parameters(self):
        return [p for name, p in self.named_parameters() if not name.startswith('backbone.')]

    @torch.no_grad()
    def project(self):
        if self.mode == 'tsm_learned':
            self.gamma.clamp_(0., 1.)

    def state_storage_bytes(self):
        return 0 if self.state is None else self.state.untyped_storage().nbytes()
