"""Actual initialization/state and real training-window backward gates."""
import argparse
import json
from pathlib import Path

import cv2
import torch

from dynamic_training_r34 import ROOT, TrainingPool, digest
from training_preserved_r37 import models, prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    assert not out.exists()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    nets, marker = models(0, torch.device(args.device))
    net = nets['learned_alignment']
    shape_checks = []
    for h, w in [(32, 36), (33, 37)]:
        image = torch.rand(2, 3, h, w, device=args.device)
        past = torch.randn(2, 3, h, w, device=args.device).half()
        with torch.no_grad():
            reference = net.backbone(image)
            first, state = net(image)
            actual, _ = net(image, past)
            current, _ = net(image, past, history='current')
        deltas = {name: float((value-reference).abs().max()) for name, value in
                  [('first', first), ('arbitrary_history_zero_amplitude', actual), ('current', current)]}
        assert max(deltas.values()) <= 1e-6
        assert state.shape == (2, 3, h, w) and state.dtype == torch.float16
        assert state.untyped_storage().nbytes() == 2*3*h*w*2
        shape_checks.append({'shape': [2, 3, h, w], 'max_abs_deltas': deltas,
                             'state_storage_bytes': state.untyped_storage().nbytes()})
    basis = net.basis().detach()
    error = float((basis@basis.t()-torch.eye(3, device=args.device)).abs().max())
    assert error <= 1e-6
    pool = TrainingPool()
    low, target, identities = pool.batch(0, 0)
    low, target = low.to(args.device), target.to(args.device)
    backward = []
    for name, model in nets.items():
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
        rounds = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = prediction(model, low)
            loss = (output-target).square().mean()
            assert bool(torch.isfinite(loss))
            loss.backward()
            assert all(p.grad is None for p in model.backbone.parameters())
            gradients = {key: float(p.grad.norm()) for key, p in model.named_parameters() if p.grad is not None}
            assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            assert gradients['amplitude'] > 0
            if step == 1:
                assert gradients['directions'] > 0 and gradients['field.2.weight'] > 0
            optimizer.step()
            rounds.append({'smoke_update': step+1, 'mse': float(loss), 'gradient_norms': gradients})
        # Current is an explicit immutable-anchor branch after temporal updates.
        with torch.no_grad():
            value, _ = model(low[:, 0], history='current')
            delta = float((value-model.backbone(low[:, 0])).abs().max())
        assert delta == 0
        backward.append({'model': name, 'rounds': rounds, 'current_after_smoke_max_abs_delta': delta})
    result = {'passed': True, 'device': args.device, 'shape_checks': shape_checks,
              'orthogonality_max_abs_error': error, 'real_window_backward': backward,
              'windows': identities, 'anchor': marker,
              'parameters': {name: {'total': sum(p.numel() for p in model.parameters()),
                                    'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad)}
                             for name, model in nets.items()},
              'source_sha256': {name: digest(Path(__file__).parent/name) for name in
                                ['preserved_state_r37.py', 'training_preserved_r37.py', 'check_preserved_r37.py']},
              'scope': 'Functional/backward gates only, two discarded SGD smoke steps. Fresh models required for real training; no quality/latency or optimizer replay claim.'}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'passed': True, 'device': args.device, 'parameters': result['parameters']}))


if __name__ == '__main__':
    main()
