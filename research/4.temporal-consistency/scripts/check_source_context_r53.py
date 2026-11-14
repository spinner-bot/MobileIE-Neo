"""Actual source-window parity and one discarded gradient update per device gate."""
import json
import time
from datetime import datetime

import cv2
import torch

from dynamic_training_r34 import BASE, digest
from source_context_r53 import SourceContextPool, prediction
from task_fastdvd_r47 import initial_model, official_centers
from training_form_r41 import configure_precision


def check_one(model, source8, target):
    model.eval()
    with torch.inference_mode():
        expected = official_centers(model, source8)[:,2:6]
        actual = prediction(model, source8, clamp=True)
        delta = float((actual-expected).abs().max())
    assert delta <= 1e-5
    model.train()
    parameters = list(model.parameters())
    first, last = parameters[0], parameters[-1]
    before = [first.detach().clone(), last.detach().clone()]
    optimizer = torch.optim.SGD(parameters, lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    loss = (prediction(model, source8)-target).square().mean(); assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    gradients = [float(first.grad.norm()), float(last.grad.norm())]
    assert min(gradients) > 0
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
    optimizer.step()
    changes = [float((p-v).detach().abs().max()) for p,v in zip([first,last],before)]
    assert min(changes) > 0
    return {'passed': True, 'source8_official_target_centers_max_abs_delta': delta,
            'MSE': float(loss.detach()), 'first_last_gradient': gradients,
            'first_last_parameter_change': changes, 'norm_before_clip': norm,
            'discarded_SGD_updates': 1, 'inherited_into_formal_training': False}


def main():
    torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True); configure_precision()
    assert not torch.cuda.is_initialized()
    out = BASE/'process/exploration-r53/cpu-gate001'; out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    pool = SourceContextPool()
    source8, target, identities, context, low = pool.batch(0,0)
    old = json.loads((BASE/'results/exploration-r47-stage001-seed0/config.json').read_text())
    assert old['source_sha256']['mixed_anchor_r40.py'] == digest(BASE/'scripts/mixed_anchor_r40.py')
    assert old['source_sha256']['dynamic_training_r34.py'] == digest(BASE/'scripts/dynamic_training_r34.py')
    with (BASE/'results/exploration-r47-stage001-seed0/windows.jsonl').open() as stream:
        assert json.loads(stream.readline())['windows'] == identities
    assert torch.equal(source8[:,2:6], low) and target.shape == low.shape
    result = check_one(initial_model(0), source8, target)
    result.update(at=datetime.now().astimezone().isoformat(), source_context=context,
                  original_targets=identities, old_target_context_matches=True, device='cpu',
                  seconds=time.perf_counter()-start, cuda_initialized=torch.cuda.is_initialized(),
                  packed_windows_bytes_per_step_estimate=len(json.dumps({'windows':identities,'source_context':context}).encode()))
    assert not result['cuda_initialized']
    (out/'gate.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['passed','seconds','source8_official_target_centers_max_abs_delta',
                                           'packed_windows_bytes_per_step_estimate']}))


if __name__ == '__main__': main()
