"""Two source-faithful published Simple learned levels; original F0 remains frozen."""
import copy
import sys

import torch
from torch import nn
import yaml

from core import ROOT, VENDOR, objective, sequence
sys.path.insert(0, str(VENDOR))
from instant_video_models.hooks import add_hook_modules


class Simple(nn.Module):
    def __init__(self, original, level):
        super().__init__()
        assert level in ('all', 'output')
        self.level = level
        self.net = copy.deepcopy(original).eval()
        self.original_names = tuple(n for n, _ in self.net.named_parameters())
        for p in self.net.parameters():
            p.requires_grad_(False)
        config = yaml.safe_load((VENDOR / 'configs/stabilizers/hdrnet/simple_learned.yml').read_text())
        hooks = config['stabilizers'] if level == 'all' else [config['stabilizers'][-1]]
        self.stabilizers = add_hook_modules(self.net, hooks)
        assert len(self.stabilizers) == (7 if level == 'all' else 1)
        self.frame_calls = 0

    def forward(self, x):
        self.frame_calls += x.shape[0]
        return self.net(x)

    def reset(self):
        for module in self.stabilizers.values():
            module.reset()

    def enable(self, enabled=True):
        for module in self.stabilizers.values():
            module.enable() if enabled else module.disable()

    def materialize(self):
        with torch.no_grad():
            self(torch.zeros(1, 3, 24, 32))
        self.reset()
        assert not any(torch.nn.parameter.is_lazy(p) for p in self.parameters())
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) == (66 if self.level == 'all' else 3)

    def original_parameters(self):
        return {n: self.net.get_parameter(n) for n in self.original_names}

    def state_inventory(self, height=1080, width=1920):
        rows = []
        for name, module in self.stabilizers.items():
            value = module.memory
            assert value is not None and value.dtype == torch.float32
            is_global = value.shape[-2:] == (1, 1)
            elements = value.shape[1] * (1 if is_global else height * width)
            rows.append(dict(name=name, fixture_shape=list(value.shape),
                             fixture_bytes=value.numel() * value.element_size(),
                             analytic_1080p_bytes=4 * elements,
                             analytic_nonfirst_EMA_scalar_ops=3 * elements))
        return dict(tensors=rows, analytic_1080p_persistent_bytes=sum(r['analytic_1080p_bytes'] for r in rows),
                    analytic_nonfirst_EMA_scalar_ops=sum(r['analytic_nonfirst_EMA_scalar_ops'] for r in rows),
                    registered_parameters=sum(p.numel() for p in self.parameters()),
                    original_frozen_parameters=4075, added_trainable_parameters=66 if self.level == 'all' else 3,
                    analytic_original_and_total_conv_MAC=3828 * height * width + 144,
                    scope='Retained state and scalar arithmetic only; sigmoid, copies and temporary tensors excluded; not speed/peak RAM')


def make_models(original):
    models = {level: Simple(original, level) for level in ('all', 'output')}
    for model in models.values():
        model.materialize()
    return models
