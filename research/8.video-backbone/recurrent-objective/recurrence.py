"""Native DIS fields with fixed-field differentiable original RGB recurrence."""
import numpy as np
import torch
from study_io import BASE
from fixed_dis_r86 import FixedDIS, RecursivePipeline
from state_operator_r83 import StateOperator


def apply(current, state, operator):
    """Fields and acceptance detached; state and current retain RGB gradients."""
    channels, height, width = current.shape
    assert channels == 3 and (height, width) == (operator.h, operator.w)
    index = torch.from_numpy(operator.index.reshape(-1, 4).copy()).to(current.device)
    weights = torch.from_numpy(operator.weights.reshape(-1, 4).astype(np.float32)).to(current)
    valid = torch.from_numpy(operator.valid.copy()).to(current)
    moved = (state.reshape(3, -1)[:, index] * weights[None]).sum(-1).reshape_as(current)
    return current * (1 - .5 * valid[None]) + moved


def recursive(raw, reference=False):
    assert raw.ndim == 5 and raw.shape[2] == 3 and raw.dtype == torch.float32
    assert torch.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1
    batch, length = raw.shape[:2]
    images = raw.detach().permute(0, 1, 3, 4, 2).contiguous().cpu().numpy() * 255
    outputs, fields, acceptances = [], [], []
    pairs, maximum_difference = 0, 0.
    for b in range(batch):
        motion, values, own_fields = FixedDIS(), [raw[b, 0]], []
        for t in range(1, length):
            flow, mask = motion(images[b, t - 1], images[b, t])
            operator = StateOperator(flow, mask)
            values.append(apply(raw[b, t], values[-1], operator))
            own_fields.append((flow, mask))
            acceptances.append(dict(window=b, t=t, valid_fraction=float(operator.valid.mean())))
        assert motion.pairs == length - 1
        pairs += motion.pairs
        output = torch.stack(values)
        assert torch.equal(output[0], raw[b, 0])
        if reference:
            iterator = iter(own_fields)
            native = RecursivePipeline(lambda previous, current: next(iterator))
            expected = np.stack([native.step(frame)[0] for frame in images[b]]) / 255
            actual = output.detach().permute(0, 2, 3, 1).cpu().numpy()
            maximum_difference = max(maximum_difference, float(np.max(np.abs(actual - expected))))
        outputs.append(output)
        fields.append(own_fields)
    result = torch.stack(outputs)
    assert torch.isfinite(result).all() and result.min() >= 0 and result.max() <= 1 + 2e-6
    assert result.shape == raw.shape
    return result, dict(DIS_pairs=pairs, acceptances=acceptances,
                       fixed_field_native_max_abs_difference=maximum_difference if reference else None,
                       fields_and_masks_stop_gradient=True, RGB_history_detached=False)
