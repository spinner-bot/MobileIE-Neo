"""Known training-only BN buffer recalibration; all learned weights remain frozen."""
import json

import torch
from torch import nn

from dynamic_training_r34 import BASE, ROOT, digest
from training_form_r41 import create_model

MODELS = ['square_mse', 'plain', 'joint']
WINDOWS = 128


def source_model(seed, name):
    assert seed in [0, 1] and name in MODELS
    family = 'r44' if name == 'square_mse' else 'r48'
    folder = BASE/f'results/exploration-{family}-stage001-seed{seed}'
    runtime = json.loads((folder/'runtime.json').read_text())
    assert runtime['completed_normally'] and runtime['model_updates'][name] == 4000
    marker = json.loads((folder/f'{name}-step4000.json').read_text())
    path = ROOT/marker['training_path']
    assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    assert digest(path) == marker['training_sha256']
    saved = torch.load(path, map_location='cpu', weights_only=False)
    assert saved['seed'] == seed and saved['model'] == name and saved['new_step'] == 4000
    assert saved['inherited_updates'] == (4000 if family == 'r44' else 8000)
    net, _ = create_model(seed)
    net.load_state_dict(saved['state_dict'], strict=True)
    return net.requires_grad_(False).eval(), marker


class SequentialFrames(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, low):
        assert low.shape[1:] == (4, 3, 128, 128) and low.shape[0] == 4
        # Keep each original B4 call; do not silently flatten to B16.
        for frame in low.unbind(1):
            output = self.model(frame)
            assert torch.isfinite(output).all()


def snapshot(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def verify_change(net, before, calls):
    changed = []
    for name, value in net.state_dict().items():
        if not torch.equal(value.detach().cpu(), before[name]):
            assert name.endswith(('.running_mean', '.running_var', '.num_batches_tracked')), name
            changed.append(name)
    for name, value in net.named_parameters():
        assert not value.requires_grad and value.grad is None
        assert torch.equal(value.detach().cpu(), before[name]), name
    counts = {}
    for name, module in net.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            expected = 0 if name.startswith('tail_warm.') else calls
            counts[name] = int(module.num_batches_tracked)
            assert counts[name] == expected
            assert module.momentum == .1
            assert torch.isfinite(module.running_mean).all()
            assert torch.isfinite(module.running_var).all() and (module.running_var >= 0).all()
    assert changed
    return {'learned_parameters_unchanged': True, 'changed_buffers': changed,
            'BN_counts': counts, 'gradients_absent': True}


@torch.no_grad()
def calibrate(net, loader, device):
    before = snapshot(net)
    wrapper = SequentialFrames(net).eval()
    torch.optim.swa_utils.update_bn(loader, wrapper, device=device)
    assert not wrapper.training and not net.training
    return before
