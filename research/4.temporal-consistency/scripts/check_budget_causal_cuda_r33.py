"""Guarded CUDA FP32 backward smoke check; synthetic inputs, no quality training."""
import json
import os
from pathlib import Path
import traceback

import torch

from budget_causal_r33 import BudgetCausal, ROOT


def main():
    out = ROOT/'research/4.temporal-consistency/process/exploration-r33/cuda-backward.json'
    assert not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(33)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    result = {'device': 'cuda:0', 'dtype': 'float32', 'deterministic_algorithms': True,
              'scope': 'Synthetic-input backward smoke check only, no optimizer updates or enhancement-quality measurement.'}
    try:
        net = BudgetCausal().cuda().train()
        with torch.no_grad():
            net.readout.weight.normal_(0, .05)
        frames = torch.rand(3, 2, 3, 128, 128, device='cuda', requires_grad=True)
        state = None
        for frame in frames:
            prediction, state = net(frame, state)
        loss = prediction.square().mean()
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(frames.grad).all()
        assert frames.grad[0].abs().sum().item() > 0
        assert net.field.weight.grad[:2].abs().sum().item() > 0
        result.update(passed=True, oldest_input_gradient_l1=frames.grad[0].abs().sum().item(),
                      offset_field_gradient_l1=net.field.weight.grad[:2].abs().sum().item(),
                      synthetic_loss=loss.item())
    except Exception:
        result.update(passed=False, error=traceback.format_exc())
        raise
    finally:
        out.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
