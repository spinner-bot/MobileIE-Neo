"""Fixed coordinate control and covariance-weighted state-response supervision."""
import numpy as np
import torch
import torch.nn.functional as F
from half_motion_r4 import half_forward_backward
from response_cue_r45 import align_quantized
from budget_causal_r33 import transport


def prepare_teacher(raw, seed):
    batch, frames, _, height, width = raw.shape
    displayed = raw.detach().clamp(0, 1).cpu().numpy().transpose(0, 1, 3, 4, 2)*255
    pairs = [half_forward_backward(clip[t-1], clip[t]) for clip in displayed for t in range(1, frames)]
    flow = torch.from_numpy(np.stack([p[0].transpose(2, 0, 1) for p in pairs]).copy())
    mask = torch.from_numpy(np.stack([p[1][None] for p in pairs]).copy())
    rng = np.random.default_rng(np.random.SeedSequence([902, seed]))
    probes = []
    for size in (5, 17):
        value = torch.from_numpy(rng.standard_normal((batch*(frames-1), 3, height, width)).astype(np.float32))
        value = F.avg_pool2d(F.pad(value, (size//2,)*4, mode='replicate'), size, 1)
        value = value-value.mean((-2, -1), keepdim=True)
        value = value/value.square().mean((-2, -1), keepdim=True).sqrt().clamp_min(1e-6)
        probes.append(value)
    responses = [teacher_linear(value, flow, mask) for value in probes]
    return dict(flow=flow, mask=mask, probes=probes, responses=responses)


def teacher_linear(state, flow, mask):
    moved, valid = align_quantized(state, torch.zeros_like(state), flow, mask)
    return .5*valid.to(state.dtype)*moved


def student_linear(state, flow, logit):
    moved, valid = transport(state, torch.zeros_like(state), flow)
    return .5*logit.sigmoid()*valid.to(state.dtype)*moved


def losses(arm, raw, target, fields, teacher):
    flow, logit = fields
    space = (raw-target).square().mean()
    if arm == 'coordinate':
        mask = teacher['mask'].to(flow.dtype)
        coordinate = ((flow-teacher['flow']).abs()*mask).sum()/(32*mask.sum().clamp_min(1))
        confidence = F.binary_cross_entropy_with_logits(logit, mask)
        auxiliary = coordinate+confidence
    elif arm == 'response':
        parts = [(student_linear(value, flow, logit)-wanted).square().mean()/wanted.square().mean().clamp_min(1e-4)
                 for value, wanted in zip(teacher['probes'], teacher['responses'])]
        auxiliary = torch.stack(parts).mean()
    else:
        raise ValueError(arm)
    total = 100*space+auxiliary
    return total, dict(space_MSE=float(space.detach()), auxiliary=float(auxiliary.detach()), total=float(total.detach()))


def optimizer(model):
    original = list(model.backbone.parameters())
    original_ids = {id(p) for p in original}
    new = [p for p in model.parameters() if id(p) not in original_ids]
    return torch.optim.Adam([dict(params=original, lr=1e-5), dict(params=new, lr=1e-3)],
                            betas=(.9, .999), eps=1e-8, weight_decay=0)
