"""Shared batch-one stream execution for frozen selected-channel comparisons."""
from pathlib import Path
import subprocess

import torch

from exploration_r2_data import ROOT
from explore_temporal_controls import sha256


@torch.inference_mode()
def predict(model, frames, resets=None):
    if hasattr(model, 'reset'):
        model.reset()
    outputs = []
    for i, frame in enumerate(frames):
        if resets is not None and resets[i] and hasattr(model, 'reset'):
            model.reset()
        outputs.append(model(frame.unsqueeze(0))[0])
    return torch.stack(outputs)


def provenance():
    return {'git_revision_at_startup': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_sha256_at_startup': {str(p.relative_to(ROOT)): sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'weights_sha256': sha256(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'),
        'protocol_sha256': sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r7b-protocol.md'),
        'cudnn_deterministic': True, 'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32}
