"""Preserve the pretrained function while restoring the upstream MBR branches."""
import os
import time

import torch
from torch import nn

from dynamic_training_r34 import ROOT, digest
from gpu_guard import sample
from train_exploration_r2 import new_model
from model.lle import MobileIELLENet
from model.utils import MBRConv1, MBRConv3, MBRConv5

LAYERS = ['head.block1.0', 'head.block1.2', 'body.block1', 'att.1', 'att1.0', 'tail']


def create_model(seed):
    torch.manual_seed(seed)
    slim = new_model(torch.device('cpu'))
    torch.manual_seed(4100+seed)
    full = MobileIELLENet(12, rep_scale=4)
    with torch.no_grad():
        full_state = full.state_dict()
        for name, value in slim.state_dict().items():
            if name in full_state:
                assert full_state[name].shape == value.shape
                full_state[name].copy_(value)
        for name in LAYERS:
            branch = full.get_submodule(name)
            source = slim.get_submodule(name)
            assert isinstance(branch, (MBRConv1, MBRConv3, MBRConv5)) and isinstance(source, nn.Conv2d)
            outputs = source.out_channels
            assert branch.conv.weight.shape[1:] == source.weight.shape[1:]
            branch.conv.weight[:outputs].copy_(source.weight)
            branch.conv.bias[:outputs].copy_(source.bias)
            branch.conv_out.weight.zero_()
            branch.conv_out.bias.zero_()
            index = torch.arange(outputs)
            branch.conv_out.weight[index, index, 0, 0] = 1
    full.tail_warm.requires_grad_(False)
    info = {'pretrained_sha256': digest(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'),
            'rep_scale': 4, 'branch_seed': 4100+seed, 'expanded_layers': LAYERS,
            'training_parameters_total': sum(p.numel() for p in full.parameters()),
            'training_parameters_active': sum(p.numel() for p in full.parameters() if p.requires_grad),
            'inactive_warm_parameters': sum(p.numel() for p in full.tail_warm.parameters()),
            'deployment_parameters': sum(p.numel() for p in slim.parameters()),
            'initialization': 'Copy each slim kernel/bias into first outputs of the largest raw branch; identity compression, zero other readout columns; copy FST/PReLU.',
            'warmup': False, 'scope': 'Original MBR parameterization control, not a reproduction of the complete official training protocol.'}
    return full, info


def cuda_headroom():
    assert not torch.cuda.is_initialized(), 'Headroom must be checked before initializing CUDA'
    physical = int(os.environ['CUDA_VISIBLE_DEVICES'])
    assert physical in [4, 5, 6, 7], 'R41 resource amendment fixes the NUMA1 GPU group'
    samples = [sample(physical)]
    time.sleep(1)
    samples.append(sample(physical))
    allowed = all(not r['compute_processes'] and r['memory_mib'] <= 64
                  and r['utilization_percent'] <= 5 for r in samples)
    return {'allowed': allowed, 'samples': samples,
            'scope': 'Fresh pre-CUDA check after full CPU data preparation; external project GPU lock remains held.'}
