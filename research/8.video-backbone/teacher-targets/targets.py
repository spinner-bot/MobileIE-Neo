"""Fixed teacher targets; neither target reads GT or changes student inference."""
import numpy as np
import torch

from study_io import ARMS
from fixed_dis_r86 import FixedDIS, RecursivePipeline


def teacher_targets(teacher, low):
    assert low.shape == (4, 4, 3, 128, 128)
    assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())
    with torch.no_grad():
        raw = teacher(low.flatten(0, 1)).clamp(0, 1).unflatten(0, (4, 4))
    images = raw.permute(0, 1, 3, 4, 2).contiguous().cpu().numpy() * 255
    output = np.empty_like(images)
    pairs = 0
    for b in range(4):
        motion = FixedDIS()
        stream = RecursivePipeline(motion)
        for t, frame in enumerate(images[b]):
            output[b, t], _ = stream.step(frame)
        assert np.array_equal(images[b, 0], output[b, 0]) and motion.pairs == 3
        pairs += motion.pairs
    assert output.dtype == np.float32 and np.isfinite(output).all() and pairs == 12
    post = torch.from_numpy(output.transpose(0, 1, 4, 2, 3).copy() / 255).to(low)
    assert not raw.requires_grad and not post.requires_grad
    return raw, post


def objective(arm, prediction, gt, spatial, temporal):
    assert arm in ARMS
    reconstruction = (prediction - gt).square().mean()
    if arm == 'gt':
        return reconstruction
    target = spatial if arm == 'spatial' else temporal
    return .5 * reconstruction + .5 * (prediction - target).square().mean()
