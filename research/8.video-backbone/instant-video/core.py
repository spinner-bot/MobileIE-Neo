"""Pinned IVM controlled stabilizers on every original MobileIE Conv2d and output.

This is a task adapter around unchanged upstream code, not a new method.
"""
import copy
from pathlib import Path
import sys

import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / 'third_party/instant-video-models'
sys.path.insert(0, str(VENDOR))
from instant_video_models.hooks import add_hook_modules
from instant_video_models.losses.rmse import RMSE
from instant_video_models.metrics.unified_loss import UnifiedLoss


class Adapted(nn.Module):
    def __init__(self, original, spatial):
        super().__init__()
        self.net = copy.deepcopy(original).eval()
        self.original_names = tuple(n for n, _ in self.net.named_parameters())
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.targets = [n for n, m in self.net.named_modules() if isinstance(m, nn.Conv2d)] + ['']
        name = 'controlled_spatial.yml' if spatial else 'controlled.yml'
        config = yaml.safe_load((VENDOR / 'configs/stabilizers/hdrnet' / name).read_text())
        self.controllers = add_hook_modules(self.net, config['controllers'])
        self.stabilizers = add_hook_modules(self.net, config['stabilizers'])
        assert len(self.controllers) == 1 and len(self.stabilizers) == 7
        self.spatial = spatial
        self.frame_calls = 0

    def forward(self, x):
        self.frame_calls += x.shape[0]
        return self.net(x)

    def reset(self):
        for module in (*self.controllers.values(), *self.stabilizers.values()):
            module.reset()

    def enable(self, enabled=True):
        for module in (*self.controllers.values(), *self.stabilizers.values()):
            module.enable() if enabled else module.disable()

    def materialize(self, shape=(1, 3, 24, 32)):
        # Head parameters are lazy until the second frame; finish before Adam creation.
        with torch.no_grad():
            self(torch.zeros(shape))
            self(torch.zeros(shape))
        assert not any(torch.nn.parameter.is_lazy(p) for p in self.parameters())
        self.reset()

    def original_parameters(self):
        return {n: self.net.get_parameter(n) for n in self.original_names}

    def parameters_and_cost(self, height=1080, width=1920):
        # All base spatial convolutions are native resolution, except att.1 at1x1.
        # No actual high-resolution forward, timing, peak RAM or non-convolution cost.
        base = 3828 * height * width + 144
        controller = next(iter(self.controllers.values()))
        backbone = sum(m.weight.numel() for m in controller.conv_layers) * height * width
        heads = {}
        for name, module in self.stabilizers.items():
            area = 1 if name.startswith('att.1-') else height * width
            heads[name] = sum(w.numel() for w in module.conv_weights) * area
        total = base + backbone + sum(heads.values())
        return dict(parameters=sum(p.numel() for p in self.parameters()),
                    original_frozen=sum(p.numel() for p in self.original_parameters().values()),
                    trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    resolution=[height, width], original_conv_MAC=base,
                    controller_backbone_conv_MAC=backbone, controller_heads_conv_MAC=heads,
                    total_conv_MAC=total, conv_MAC_ratio_to_original=total / base,
                    scope='Analytic convolution only; excludes nonlinear/memory/state costs, not measured speed')

    def state_inventory(self):
        records = []
        seen = set()
        for name, module in {**self.controllers, **self.stabilizers}.items():
            for field in ('memory', 'output', 'memory_stabilized', 'memory_unstabilized'):
                value = getattr(module, field, None)
                if value is None:
                    continue
                key = value.untyped_storage().data_ptr()
                records.append(dict(module=name, field=field, shape=list(value.shape),
                                    bytes=value.numel() * value.element_size(),
                                    newly_counted_storage=key not in seen))
                seen.add(key)
        return dict(tensors=records, persistent_tensor_bytes=sum(
            r['bytes'] for r in records if r['newly_counted_storage']),
            scope='Retained state tensors only, not peak memory or temporary unfold/activations')


def objective(outputs, target):
    loss = UnifiedLoss(RMSE(epsilon=1e-6), stability_weight=.4, mask_nan=False)
    for t in range(outputs.shape[1]):
        loss.update(outputs[:, t], target[:, t])
    return loss.compute()


def sequence(model, frames):
    model.reset()
    return torch.stack([model(frames[:, t]) for t in range(frames.shape[1])], dim=1)
