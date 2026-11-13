"""Independently train unchanged XS with duplicated current observations and no data history."""
import json

import torch

from causal_reference_io_r75 import BASE
from causal_reference_training_r75 import optimizer_for,finish_update,STEPS


def spatial_predictions(model,low):
    if low.ndim!=5 or low.shape[1:]!=(4,3,384,384):
        raise ValueError('Require the registered B,T4,RGB,native384 training video.')
    return torch.stack([model(torch.stack((current,current),dim=1))
                        for current in low.unbind(1)],dim=1)


def backward_video(model,low,target):
    output=spatial_predictions(model,low)
    if output.shape!=target.shape:
        raise ValueError('Output and target must have exactly matching video shapes.')
    loss=((output-target).square()+1e-6).sqrt().mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Non-finite spatial control loss; no scientific auto-retry.')
    (loss/4).backward()
    return float(loss.detach())


def numerical_investment_status():
    primary=BASE/'report/exploration-r75-joint/audit.json'
    control=BASE/'report/exploration-r76-joint/audit.json'
    if not primary.exists() or not control.exists():
        return 'pending_complete_original_and_self4_results'
    a,b=json.loads(primary.read_text()),json.loads(control.read_text())
    assert a['completed'] and a['full_gates_evaluated'] and a['all_quality_rows']==80
    assert b['completed'] and b['scores_complete'] and b['all_conditions']==100
    ag=json.loads((primary.parent/'gates.json').read_text())
    bg=json.loads((control.parent/'attribution_support.json').read_text())
    assert len(ag)==8 and len(bg)==4
    outcomes=[r['passed'] for r in ag]+[r['support_passed'] for r in bg]
    assert all(type(value) is bool for value in outcomes)
    if not all(outcomes):
        return 'stop_additional_128K_training_investment'
    return 'numeric_pass_still_requires_actual_visual_review_and_new_execution_resource_qualification'
