"""Load the fixed strong anchor and initialize paired R37 temporal controls."""
import copy
import json

import torch

from dynamic_training_r34 import BASE, ROOT, digest
from preserved_state_r37 import PreservedState
from train_exploration_r2 import new_model


def models(seed, device):
    folder = BASE/f'results/exploration-r34-seed{seed}'
    runtime = json.loads((folder/'runtime.json').read_text())
    assert runtime['completed_normally'] and runtime['model_updates']['original'] == 4000
    marker = json.loads((folder/'original-step4000.json').read_text())
    assert digest(ROOT/marker['path']) == marker['sha256']
    saved = torch.load(ROOT/marker['path'], map_location='cpu', weights_only=False)
    assert saved['seed'] == seed and saved['step'] == 4000 and saved['model'] == 'original'
    backbone = new_model(torch.device('cpu'))
    backbone.load_state_dict(saved['state_dict'], strict=True)
    torch.manual_seed(3700+seed)
    aligned = PreservedState(backbone, aligned=True)
    unaligned = copy.deepcopy(aligned)
    unaligned.aligned = False
    return {'learned_alignment': aligned.to(device), 'zero_alignment': unaligned.to(device)}, marker


def prediction(net, low):
    state, outputs = None, []
    for frame in low.unbind(1):
        output, state = net(frame, state)
        outputs.append(output)
    return torch.stack(outputs, dim=1)
