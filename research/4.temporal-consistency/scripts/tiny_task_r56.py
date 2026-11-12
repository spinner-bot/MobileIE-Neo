"""Original 696-parameter reference, four temporal B4 calls, discarded gate."""
import torch

from mixed_anchor_r40 import prediction
from multinex_reference_r55 import load_reference


def initial_model():
    model, info = load_reference()
    model.requires_grad_(True)
    assert sum(p.numel() for p in model.parameters()) == 696
    return model, info


def bn_counts(model):
    return {name: int(module.num_batches_tracked) for name, module in model.named_modules()
            if isinstance(module, torch.nn.BatchNorm2d)}


def branch(name):
    if name.startswith('illum_') or name.startswith('head_luma'): return 'luminance'
    if name.startswith('chroma_') or name.startswith('head_chroma'): return 'chrominance'
    raise ValueError(name)


def check_one(model, low, target):
    model.eval()
    with torch.inference_mode():
        value = prediction(model, low)
        assert value.shape == target.shape and torch.isfinite(value).all()
        evaluation_MSE = float((value-target).square().mean())
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    counts_before = bn_counts(model)
    model.train(); optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    value = prediction(model, low)
    loss = (value-target).square().mean(); assert torch.isfinite(loss)
    loss.backward()
    gradient = {'luminance': 0., 'chrominance': 0.}; missing = []
    for name, p in model.named_parameters():
        if p.grad is None:
            missing.append(name); continue
        assert torch.isfinite(p.grad).all()
        gradient[branch(name)] += float(p.grad.detach().double().square().sum())
    assert min(gradient.values()) > 0
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
    optimizer.step()
    change = {'luminance': 0., 'chrominance': 0.}
    for name, p in model.named_parameters():
        change[branch(name)] = max(change[branch(name)], float((p-before[name]).detach().abs().max()))
    assert min(change.values()) > 0
    counts_after = bn_counts(model)
    assert counts_before and all(counts_after[name]-count == 4 for name, count in counts_before.items())
    return dict(passed=True, initial_eval_MSE=evaluation_MSE, training_MSE=float(loss.detach()),
                branch_gradient_L2={k: v**.5 for k, v in gradient.items()}, branch_max_parameter_change=change,
                missing_gradient_names=missing, norm_before_clip=norm, BN_before=counts_before, BN_after=counts_after,
                discarded_SGD_updates=1, inherited_into_formal_training=False)
