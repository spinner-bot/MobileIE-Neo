"""One matched spatial objective control; no temporal or architectural search."""
import json

import torch
import torch.nn.functional as F

from dynamic_training_r34 import BASE, ROOT, digest
from mixed_anchor_r40 import prediction
from training_form_r41 import create_model


def initial_model(seed):
    model, info = create_model(seed)
    assert sum(p.numel() for p in model.parameters()) == 49978
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 47239
    return model, info


def verify_old_reference(seed):
    """Check settings/window identities, without rerunning MSE or tensor replay."""
    first = BASE/f'results/exploration-r41-stage002-seed{seed}'
    second = BASE/f'results/exploration-r44-stage001-seed{seed}'
    published = BASE/f'results/exploration-r65-stage001-seed{seed}'
    sources = {}
    def read(path):
        sources[str(path.relative_to(ROOT))] = digest(path)
        return json.loads(path.read_text())
    c1, c2 = read(first/'config.json'), read(second/'config.json')
    r1, r2 = read(first/'runtime.json'), read(second/'runtime.json')
    assert r1['completed_normally'] and r2['completed_normally']
    assert r1['model_updates']['expanded_fp32'] == r2['model_updates']['square_mse'] == 4000
    assert c1['objective'] == 'MSE all four frames'
    assert c2['temporal_weights']['square_mse'] == 0
    for config in (c1, c2):
        assert (config['seed'], config['batch'], config['frames'], config['patch']) == (seed, 4, 4, 128)
        assert (config['lr'], config['eps'], config['weight_decay']) == (1e-4, 1e-8, 0)
        assert config['betas'] == [.9, .999]
        for name in ['training_form_r41.py', 'mixed_anchor_r40.py', 'dynamic_training_r34.py']:
            assert digest(BASE/'scripts'/name) == config['source_sha256'][name]
    marker = read(second/'square_mse-step4000.json')
    assert marker['inherited_updates'] == 4000 and marker['final_export_gate_passed']
    assert digest(ROOT/marker['path']) == marker['sha256']
    assert marker['deployment_parameters'] == 4075
    paths = [first/'windows.jsonl', second/'windows.jsonl', published/'windows.jsonl']
    for path in paths: sources[str(path.relative_to(ROOT))] = digest(path)
    old = []
    for path in paths[:2]:
        rows = [json.loads(line) for line in path.open()]
        assert len(rows) == 4000
        old.extend(row['windows'] for row in rows)
    compared = 0
    with paths[2].open() as stream:
        for step, line in enumerate(stream):
            row = json.loads(line)
            assert step < 8000 and row['data_step'] == step and row['windows'] == old[step]
            compared += 1
    assert compared == 8000
    return dict(passed=True, seed=seed, MSE_reference=marker, source_sha256=sources,
                matched_window_identities=compared, numerical_replay=False,
                scope='Same recorded sampling identities and objective/optimizer setting; not GPU bit identity or isolated architecture/capacity causality.')


def check_one(model, low, target, objective):
    assert objective in ('L1', 'MSE')
    model.train()
    names = ['head.block1.0.conv_out.weight', 'tail.conv_out.weight', 'head.weight1', 'body.weight1']
    parameters = dict(model.named_parameters())
    before = {name: parameters[name].detach().clone() for name in names}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    output = prediction(model, low)
    l1, mse = F.l1_loss(output, target), F.mse_loss(output, target)
    loss = l1 if objective == 'L1' else mse
    assert output.shape == target.shape and torch.isfinite(output).all() and torch.isfinite(loss)
    loss.backward()
    active = [p for p in model.parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active)
    gradients = {name: float(parameters[name].grad.detach().double().norm()) for name in names}
    assert min(gradients.values()) > 0
    norm = float(torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True))
    optimizer.step()
    changes = {name: float((parameters[name]-before[name]).detach().abs().max()) for name in names}
    assert min(changes.values()) > 0
    model.eval()
    folded = model.slim().to(low.device).eval()
    assert sum(p.numel() for p in folded.parameters()) == 4075
    with torch.inference_mode():
        delta = float((prediction(model, low)-prediction(folded, low)).abs().max())
    assert delta <= 1e-4
    return dict(passed=True, objective=objective, L1=float(l1.detach()), MSE=float(mse.detach()),
                gradient_norm_before_clip=norm, selected_gradient_L2=gradients,
                selected_parameter_max_changes=changes, actual_window_eval_fold_max_abs=delta,
                discarded_SGD_updates=1, inherited_into_formal_training=False)
