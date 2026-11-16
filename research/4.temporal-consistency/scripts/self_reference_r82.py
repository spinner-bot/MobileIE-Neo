"""Small conditional correction; self identity is not proof of useful history."""
import copy

import torch
from torch import nn

MODES = ('spatial', 'direct', 'centered')


class SelfReferenceCorrection(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError('Unknown prototype mode')
        self.mode = mode
        self.head = nn.Sequential(nn.Conv2d(6, 8, 1), nn.PReLU(8),
            nn.Conv2d(8, 8, 5, padding=2, groups=8), nn.PReLU(8),
            nn.Conv2d(8, 3, 1, bias=False))
        nn.init.zeros_(self.head[-1].weight)
        self.head_calls = 0
        self.reset()

    def _h(self, current, previous):
        self.head_calls += 1
        return self.head(torch.cat([current, previous], dim=1))

    def forward(self, current, previous=None, history=None):
        if current.ndim != 4 or current.shape[1] != 3:
            raise ValueError('Expected BCHW frozen RGB output')
        history = ('current' if self.mode == 'spatial' else 'causal') if history is None else history
        if history not in ('current', 'causal') or (self.mode == 'spatial' and history != 'current'):
            raise ValueError('Invalid history route')
        if previous is None:
            return current  # Equal startup for all three controls.
        if history == 'current':
            previous = current
        if previous.shape != current.shape or previous.dtype != current.dtype or previous.device != current.device:
            raise ValueError('Matching current/history RGB required')
        residual = self._h(current, previous)
        if self.mode == 'centered':
            residual = residual-self._h(current, current)
        return current+residual

    def window(self, values, history=None):
        if values.ndim != 5 or values.shape[2] != 3:
            raise ValueError('Expected BTCHW frozen RGB clip')
        previous, outputs = None, []
        for current in values.unbind(1):
            output = self(current, previous, history)
            outputs.append(output)
            previous = output.clamp(0, 1)  # BPTT retained within this window.
        return torch.stack(outputs, dim=1)

    def reset(self):
        self._state = None
        self._started = False
        self._shape = None

    def step(self, current, history=None):
        identity = (current.shape, current.dtype, current.device)
        if self._shape is not None and self._shape != identity:
            self.reset()
        previous = current if self.mode == 'spatial' and self._started else self._state
        output = self(current, previous, history)
        self._state = None if self.mode == 'spatial' else output.clamp(0, 1).detach().clone()
        self._started, self._shape = True, identity
        return output

    def state_bytes(self):
        return 0 if self._state is None else self._state.untyped_storage().nbytes()


def make_models(seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8200+seed)
        first = SelfReferenceCorrection('spatial')
    models = {mode: copy.deepcopy(first) for mode in MODES}
    for mode, model in models.items():
        model.mode = mode
    return models
