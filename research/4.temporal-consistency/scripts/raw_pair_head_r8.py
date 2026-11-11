"""Zero-initialized raw-pair head references, retaining the original nonlinear trunk."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MODES = ('spatial_copy','spatial_detail','temporal_concat','temporal_delta','temporal_delta_aug')


class RawPairHead(nn.Module):
    def __init__(self, backbone, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.backbone, self.mode = backbone, mode
        self.auxiliary = nn.Conv2d(3,12,5,padding=2,bias=False)
        nn.init.zeros_(self.auxiliary.weight)
        self.state = None
        self.history_mode = 'causal'

    def reset(self, history_mode='causal'):
        if history_mode not in ('causal','current'):
            raise ValueError(history_mode)
        self.state, self.history_mode = None, history_mode

    def signal(self, current, previous):
        if self.mode=='spatial_copy':
            return current
        if self.mode=='spatial_detail':
            return current-F.avg_pool2d(current,3,stride=1,padding=1,count_include_pad=False)
        if self.mode=='temporal_concat':
            return previous
        return previous-current

    def transform(self, current, previous):
        net = self.backbone
        h = net.head.block1[0](current)+self.auxiliary(self.signal(current,previous))
        h = net.head.block1[2](net.head.block1[1](h))
        h = (net.head.weight1*h)*(net.head.weight2*h)+net.head.bias
        body = net.body(h)
        att = net.att(body)
        att1 = net.att1(torch.max(att*body,dim=1,keepdim=True)[0])
        return net.tail(torch.mul(att1,att)*body)

    def forward_pair(self, current, second_history=None):
        if current.ndim!=5 or current.shape[1]!=2:
            raise ValueError('B,2,C,H,W pairs required')
        previous = torch.stack([current[:,0],current[:,0] if second_history is None else second_history],1)
        if self.history_mode=='current':
            previous = current
        output = self.transform(current.flatten(0,1),previous.flatten(0,1))
        return output.reshape_as(current)

    def forward(self, current):
        previous = self.state
        if previous is None or previous.shape!=current.shape or self.history_mode=='current':
            previous = current
        output = self.transform(current,previous)
        self.state = None if self.mode.startswith('spatial_') else current.detach().clone()
        return output

    def state_storage_bytes(self):
        return 0 if self.state is None else self.state.untyped_storage().nbytes()


def augmented_history(low, seed, step):
    """Modify only the additional second-frame history, never the task inputs/GT."""
    rng = np.random.default_rng(1208000+seed*1000000+step)
    history = low[:,0].clone()
    actions = []
    if low.shape[0]!=2:
        raise ValueError('fixed two-pair protocol required')
    for j in range(2):
        draw = float(rng.random())
        action = 'actual_previous'
        if draw<.1:
            history[j] = low[j,1]
            action = 'current'
        elif draw<.2:
            history[j] = low[1-j,0]
            action = 'other_pair_previous'
        actions.append({'pair':j,'draw':draw,'history':action})
    return history, actions
