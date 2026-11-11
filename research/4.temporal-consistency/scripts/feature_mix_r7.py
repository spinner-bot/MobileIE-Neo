"""Frozen two-frame selected-channel controls before/after the body square."""
import torch
from torch import nn

SPECS = {f'body_{placement}_{group}_g05': {'placement': placement, 'channels': channels, 'gamma': .5}
         for placement in ['pre', 'post'] for group, channels in
         [('first3', [0, 1, 2]), ('selected', [6, 7, 8]), ('uniform', [0, 4, 8])]}
SPECS.update({f'body_{p}_selected_g1': {'placement': p, 'channels': [6, 7, 8], 'gamma': 1.} for p in ['pre', 'post']})


class FeatureMix(nn.Module):
    def __init__(self, backbone, channels, placement, gamma):
        super().__init__()
        channels = tuple(channels)
        if placement not in ('pre', 'post') or not 0 <= gamma <= 1 or len(channels) != 3 or tuple(sorted(set(channels))) != channels or not 0 <= channels[0] <= channels[-1] < 12:
            raise ValueError('three sorted distinct channels, pre/post, gamma in [0,1] required')
        self.backbone, self.channels, self.placement, self.gamma = backbone, channels, placement, float(gamma)
        self.contiguous_group = channels == tuple(range(channels[0], channels[0]+3))
        self.register_buffer('channel_index', torch.tensor(channels, dtype=torch.long))
        self.state, self.state_input_shape = None, None

    def reset(self):
        self.state, self.state_input_shape = None, None

    def select(self, x):
        if self.contiguous_group:
            return x[:, self.channels[0]:self.channels[-1]+1]
        return x.index_select(1, self.channel_index)

    def replace(self, x, value):
        if not self.contiguous_group:
            return x.index_copy(1, self.channel_index, value)
        start, stop = self.channels[0], self.channels[-1]+1
        parts = ([x[:, :start]] if start else [])+[value]+([x[:, stop:]] if stop < 12 else [])
        return torch.cat(parts, 1)

    def ordinary(self, u):
        b = self.backbone.body
        return (b.weight1*u)*(b.weight2*u)+b.bias

    def mix(self, current, previous):
        return previous if self.gamma == 1 else current if self.gamma == 0 else current+self.gamma*(previous-current)

    def forward(self, x):
        net = self.backbone
        u = net.body.block1(net.head(x))
        original = u if self.placement == 'pre' else self.ordinary(u)
        current = self.select(original)
        previous = self.state
        if previous is None or self.state_input_shape != x.shape:
            previous = current
        mixed = self.mix(current, previous)
        body = self.replace(original, mixed)
        if self.placement == 'pre':
            body = self.ordinary(body)
        att = net.att(body)
        att1 = net.att1(torch.max(att*body, dim=1, keepdim=True)[0])
        output = net.tail(torch.mul(att1, att)*body)
        self.state = current.detach().clone()
        self.state_input_shape = x.shape
        return output

    def state_storage_bytes(self):
        return 0 if self.state is None else self.state.untyped_storage().nbytes()
