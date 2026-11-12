"""Complete owned-data and mixed-window CPU backward prerequisites."""
import json
from pathlib import Path

import cv2
import torch

from dynamic_training_r34 import BASE, digest
from mixed_anchor_r40 import MixedPool, prediction
from train_exploration_r2 import new_model


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    pool = MixedPool()
    low, target, identities = pool.batch(0, 0)
    assert low.shape == target.shape == (4, 4, 3, 128, 128)
    assert low.dtype == target.dtype == torch.float32
    assert all(bool(torch.isfinite(value).all()) and float(value.min()) >= 0 and float(value.max()) <= 1
               for value in [low, target])
    control_low, control_gt, _ = pool.synthetic.batch(0, 0)
    assert torch.equal(low[:2], control_low[:2]) and torch.equal(target[:2], control_gt[:2])
    assert [r['source'] for r in identities] == ['davis_synthetic']*2+['olives_real']*2
    net = new_model(torch.device('cpu'))
    output = prediction(net, low)
    loss = (output-target).square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in net.parameters())
    out = BASE/'process/exploration-r40/cpu-data-gate.json'
    assert not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {'passed': True, 'data': pool.info, 'shape': list(low.shape),
              'synthetic_first_two_equal_R34': True, 'real_windows': identities[2:],
              'cpu_backward_finite': True, 'discarded_gate_mse': float(loss.detach()),
              'parameters': sum(p.numel() for p in net.parameters()), 'optimizer_updates': 0,
              'source_sha256': {name: digest(Path(__file__).parent/name) for name in
                                ['mixed_anchor_r40.py', 'check_mixed_anchor_r40.py', 'native_training_r14.py', 'dynamic_training_r34.py']},
              'scope': 'Complete owned training-source/shape/backward gate, no trained model or quality result.'}
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'passed': True, 'verified_real_files': pool.info['verified_files'], 'optimizer_updates': 0}))


if __name__ == '__main__':
    main()
