"""Load only a normal final8000 checkpoint, keeping original StableLLVE inference."""
import json

import torch

from dynamic_training_r34 import BASE, ROOT, digest
from stablellve_adapter_r15 import MANIFEST
from stablellve_task_r65 import ARMS, PARAMETERS, initial_model


def load_adapted(seed, arm):
    assert seed in (0, 1) and arm in ARMS
    run = BASE/f'results/exploration-r65-stage001-seed{seed}'
    runtime = json.loads((run/'runtime.json').read_text())
    marker = json.loads((run/f'{arm}-step8000.json').read_text())
    assert runtime['completed_normally'] and all(v == 8000 for v in runtime['updates'].values())
    assert marker['seed'] == seed and marker['arm'] == arm and marker['step'] == 8000
    assert marker['final_quality_eligible'] and marker['parameters'] == PARAMETERS
    assert digest(ROOT/marker['path']) == marker['sha256']
    saved = torch.load(ROOT/marker['path'], map_location='cpu', weights_only=False)
    assert saved['task'] == 'R65 StableLLVE MixedPool' and saved['seed'] == seed and saved['arm'] == arm
    assert saved['step'] == 8000 and saved['data_step'] == 7999
    model = initial_model(); model.load_state_dict(saved['state_dict'], strict=True)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())
    model.eval().requires_grad_(False)
    return model, dict(seed=seed, arm=arm, parameters=PARAMETERS, project_updates=8000,
                       original_manifest=json.loads(MANIFEST.read_text()), adapted_checkpoint=marker,
                       training_config_sha256=digest(run/'config.json'), future_wait_max=0,
                       scope='Fixed-budget official network/loss adaptation; not full-paper reproduction/new method.')
