"""CPU-only bounded-FST feasibility; no formal training or quality evaluation."""
import copy
from datetime import datetime
import json
from pathlib import Path
import time
import traceback

import cv2
import torch
from torch import nn

from bounded_fst_r43 import BoundedFST, convert, export, load_training_anchor, project
from dynamic_training_r34 import BASE, digest
from mixed_anchor_r40 import MixedPool, prediction
from model.utils import FST


def scalar_gate():
    source = FST(nn.Identity(), 1).double()
    with torch.no_grad():
        source.weight1.fill_(-.03)
        source.weight2.fill_(.1)
        source.bias.fill_(.04)
    candidate = BoundedFST(source).double()
    with torch.no_grad():
        candidate.kappa.fill_(.1)
    critical = 1/(.3**.5)
    u = torch.tensor([-12., -critical, -2., -.2, 0., .2, 2., critical, 12.],
                     dtype=torch.float64).reshape(1, 1, 1, -1).requires_grad_()
    result = candidate(u)
    actual, kappa_grad = torch.autograd.grad(result.sum(), [u, candidate.kappa])
    a, kappa = -.003, .1
    analytic = 2*a*u/(1+kappa*u.square()).square()
    analytic_kappa = (-a*u.pow(4)/(1+kappa*u.square()).square()).sum()
    bound = 9*abs(a)/(8*(3*kappa)**.5)
    record = {'kappa': kappa, 'scale': 1., 'dtype': 'float64', 'analytic_slope_bound': bound,
              'input_gradient_max_abs_delta': float((actual-analytic).abs().max().detach()),
              'kappa_gradient_abs_delta': float((kappa_grad-analytic_kappa).abs().max().detach()),
              'attained_slope_bound_abs_delta': abs(float(actual.abs().max())-bound)}
    with torch.no_grad():
        candidate.kappa.zero_()
        record['zero_kappa_original_max_abs_delta'] = float((candidate(u)-source(u)).abs().max())
    assert max(record[key] for key in record if key.endswith('delta')) <= 1e-10, record
    return record


def main():
    folder = BASE/'process/exploration-r43'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder/'cpu-gate.json'
    assert not target.exists(), 'Retain prior gate; no automatic retry'
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(4300)
    started = time.perf_counter()
    record = {'created_at': datetime.now().astimezone().isoformat(), 'passed': False,
              'seeds': [], 'discarded_SGD_updates': 0, 'new_formal_updates': 0,
              'source_sha256': {n: digest(Path(__file__).parent/n) for n in
                  ['check_bounded_fst_r43.py', 'bounded_fst_r43.py', 'training_form_r41.py', 'mixed_anchor_r40.py']},
              'scope': 'One prototype, fixed scale1; analytic/function/gradient/export feasibility only. '
                       'Known rational activation, no quality, novelty or zero-inference-cost claim.'}
    try:
        assert not torch.cuda.is_initialized()
        record['scalar_check'] = scalar_gate()
        pool = MixedPool()
        record['training_data'] = pool.info
        for seed in [0, 1]:
            net, marker = load_training_anchor(seed)
            low, gt, windows = pool.batch(seed, 0)
            row = {'seed': seed, 'training_checkpoint_sha256': marker['training_sha256'],
                   'initial_checks': [], 'windows': windows, 'discarded_updates': []}
            record['seeds'].append(row)
            generator = torch.Generator().manual_seed(4300+seed)
            toy = .2*torch.rand((2, 3, 33, 37), generator=generator)
            for mode in [True, False]:
                reference, candidate = copy.deepcopy(net).train(mode), convert(net).train(mode)
                for name, value in [('toy', toy), ('real_first_window', low)]:
                    with torch.no_grad():
                        expected = prediction(reference, value) if name.startswith('real') else reference(value)
                        actual = prediction(candidate, value) if name.startswith('real') else candidate(value)
                        delta = float((actual-expected).abs().max())
                    entry = {'input': name, 'shape': list(value.shape), 'train': mode, 'max_abs_delta': delta}
                    row['initial_checks'].append(entry)
                    assert delta <= 1e-5, entry
                del reference, candidate
            candidate = convert(net)
            total = sum(p.numel() for p in candidate.parameters())
            active = [p for p in candidate.parameters() if p.requires_grad]
            row['parameters'] = {'training_total': total, 'training_active': sum(p.numel() for p in active),
                                 'deployment': 4077}
            assert total == 49980 and sum(p.numel() for p in active) == 47241
            optimizer = torch.optim.SGD(active, lr=1e-6)
            for step in range(2):
                candidate.train()
                optimizer.zero_grad(set_to_none=True)
                loss = (prediction(candidate, low)-gt).square().mean()
                assert torch.isfinite(loss)
                loss.backward()
                gradients = {name: float(getattr(candidate, name).kappa.grad) for name in ['head', 'body']}
                assert all(torch.isfinite(torch.tensor(g)) and g != 0 for g in gradients.values()), gradients
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active)
                assert all(p.grad is None for p in candidate.tail_warm.parameters())
                norm = float(torch.nn.utils.clip_grad_norm_(active, 1., error_if_nonfinite=True))
                optimizer.step()
                project(candidate)
                record['discarded_SGD_updates'] += 1
                row['discarded_updates'].append({'step': step+1, 'MSE': float(loss.detach()),
                    'kappa_gradient_before_clip': gradients, 'gradient_norm_before_clip': norm,
                    'projected_kappa': {name: float(getattr(candidate, name).kappa.detach()) for name in ['head', 'body']}})
            candidate.eval()
            row['fold_checks'] = []
            for label in ['after_discarded_updates', 'positive_kappa_point1']:
                checked = copy.deepcopy(candidate)
                if label.startswith('positive'):
                    with torch.no_grad():
                        for name in ['head', 'body']:
                            getattr(checked, name).kappa.fill_(.1)
                folded = export(checked)
                with torch.no_grad():
                    delta = float((prediction(checked, low)-prediction(folded, low)).abs().max())
                entry = {'case': label, 'max_abs_delta': delta, 'deployment_parameters': 4077,
                         'kappa': {name: float(getattr(folded, name).kappa.detach()) for name in ['head', 'body']}}
                row['fold_checks'].append(entry)
                assert delta <= 1e-4, entry
                assert all(torch.equal(getattr(checked, name).kappa, getattr(folded, name).kappa)
                           and torch.equal(getattr(checked, name).scale, getattr(folded, name).scale)
                           for name in ['head', 'body'])
            del candidate, net
        assert not torch.cuda.is_initialized()
        record['passed'] = True
    except Exception:
        record['failure'] = traceback.format_exc()
        raise
    finally:
        record.update(wall_seconds=time.perf_counter()-started, cuda_initialized=torch.cuda.is_initialized())
        target.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        print(json.dumps({'passed': record['passed'], 'wall_seconds': record['wall_seconds'],
                          'discarded_SGD_updates': record['discarded_SGD_updates']}), flush=True)


if __name__ == '__main__':
    main()
