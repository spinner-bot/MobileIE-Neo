"""Final8000 only; preserve the original 696-parameter architecture and BN."""
import json

import torch

from dynamic_training_r34 import BASE, ROOT, digest
from multinex_reference_r55 import load_reference


def load_adapted(seed):
    assert seed in [0, 1]
    run = BASE/f'results/exploration-r56-stage001-seed{seed}'
    runtime = json.loads((run/'runtime.json').read_text())
    marker = json.loads((run/'nano-step8000.json').read_text())
    assert runtime['completed_normally'] and runtime['steps'] == marker['step'] == 8000
    assert marker['seed'] == seed and marker['final_quality_eligible'] and marker['parameters'] == 696
    assert digest(ROOT/marker['path']) == marker['sha256']
    checkpoint = torch.load(ROOT/marker['path'], map_location='cpu', weights_only=False)
    assert checkpoint['step'] == 8000 and checkpoint['data_step'] == 7999 and checkpoint['seed'] == seed
    assert checkpoint['task'] == 'R56 MixedPool lowlight Nano'
    model, source = load_reference()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    source.update(adapted_checkpoint=marker, adaptation_config_sha256=digest(run/'config.json'),
                  train_seed=seed, adaptation_updates=8000,
                  training='R56 fixed MixedPool8000 adaptation, original BN kept, no full-convergence or new-method claim.')
    return model, source
