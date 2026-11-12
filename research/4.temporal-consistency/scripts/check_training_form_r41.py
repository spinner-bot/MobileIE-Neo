"""Bounded initialization, branch trainability and eval/fold checks for R41."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import traceback

import cv2
import torch

from dynamic_training_r34 import BASE, digest
from mixed_anchor_r40 import MixedPool, prediction
from train_exploration_r2 import new_model
from training_form_r41 import LAYERS, configure_precision, create_model, cuda_headroom


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    assert not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    configure_precision()
    result = {'passed': False, 'device': args.device, 'created_at': datetime.now().astimezone().isoformat(),
              'source_sha256': {name: digest(Path(__file__).parent/name) for name in
                                ['training_form_r41.py', 'check_training_form_r41.py', 'mixed_anchor_r40.py']},
              'protocol_sha256': digest(BASE/'docs/exploration-r41-training-form-protocol.md'),
              'scope': 'Functional gates with two discarded SGD updates; fresh initialization required for training. No quality or optimizer replay claim.'}
    try:
        pool = MixedPool()
        low, target, identities = pool.batch(0, 0)
        net, info = create_model(0)
        reference = new_model(torch.device('cpu'))
        result['data'] = pool.info
        result['parameters'] = info
        if args.device == 'cuda':
            result['pre_cuda_admission'] = cuda_headroom()
            assert result['pre_cuda_admission']['allowed']
        net, reference = net.to(args.device), reference.to(args.device)
        shapes = []
        for h, w in [(32, 36), (33, 37)]:
            image = torch.rand(2, 3, h, w, device=args.device)
            with torch.no_grad():
                expected = reference(image)
                net.train()
                train_delta = float((net(image)-expected).abs().max())
                net.eval()
                eval_delta = float((net(image)-expected).abs().max())
            row = {'shape': list(image.shape), 'train_max_abs_delta': train_delta, 'eval_max_abs_delta': eval_delta}
            shapes.append(row)
            result['initial_shapes'] = shapes
            assert max(train_delta, eval_delta) <= 1e-5, row
        low, target = low.to(args.device), target.to(args.device)
        active = [p for p in net.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(active, lr=1e-4)
        rounds = []
        for step in range(2):
            net.train()
            optimizer.zero_grad(set_to_none=True)
            values = prediction(net, low)
            loss = (values-target).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in active)
            assert all(p.grad is None for p in net.tail_warm.parameters())
            readouts = {}
            unlocked = {}
            for name in LAYERS:
                module = net.get_submodule(name)
                gradient = float(module.conv_out.weight.grad[:, module.out_channels:].norm())
                assert gradient > 0, name
                readouts[name] = gradient
                if step == 1:
                    extra = float(module.conv.weight.grad[module.out_channels:].norm())
                    other = float(module.conv1.weight.grad.norm()) if hasattr(module, 'conv1') else float(module.conv_bn[0].weight.grad.norm())
                    assert extra > 0 and other > 0, (name, extra, other)
                    unlocked[name] = {'extra_raw_channels': extra, 'additional_conv_or_BN': other}
            norm = torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True)
            optimizer.step()
            rounds.append({'discarded_step': step+1, 'mse': float(loss.detach()),
                           'gradient_norm_before_clip': float(norm), 'other_readout_gradient_norms': readouts,
                           'unlocked_branch_gradient_norms': unlocked})
        result['discarded_backward'] = rounds
        net.eval()
        folded = net.slim().to(args.device).eval()
        assert sum(p.numel() for p in folded.parameters()) == 4075
        with torch.no_grad():
            full_value = prediction(net, low)
            folded_value = prediction(folded, low)
            delta = float((full_value-folded_value).abs().max())
        result['real_window_eval_fold_max_abs_delta'] = delta
        assert delta <= 1e-4, delta
        result['real_windows'] = identities
        result['passed'] = True
        print(json.dumps({'passed': True, 'device': args.device, 'parameters': info,
                          'eval_fold_delta': delta}), flush=True)
    except Exception:
        result['failure'] = traceback.format_exc()
        raise
    finally:
        out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
