"""One guarded CUDA backward integration check; no data pool or optimizer updates."""
import argparse
import json
import os
import traceback

import torch

from dynamic_training_r34 import BASE, digest, models
from train_dynamic_r34 import prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args()
    out = BASE/f'process/exploration-r34/cuda-interface-attempt{args.attempt:03d}.json'
    assert not out.exists() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    nets, initialization = models(0, torch.device('cuda:0'))
    torch.manual_seed(34)
    low = torch.rand(1, 4, 3, 64, 64, device='cuda')*.04
    target = torch.rand_like(low)
    result = {'passed': False, 'initialization': initialization, 'dtype': 'float32', 'models': [],
              'source_sha256': {name: digest(BASE/'scripts'/name) for name in
                               ['budget_causal_r33.py', 'dynamic_training_r34.py', 'train_dynamic_r34.py']},
              'scope': 'Preliminary synthetic-input deterministic CUDA backward check; zero optimizer updates, no training-pool pixels or quality evidence. Registered R33 gate still required.'}
    try:
        for name, net in nets.items():
            result['active_model'] = name
            estimate = prediction(net, name, low)
            loss = (estimate-target).square().mean()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
            assert torch.isfinite(loss) and norm.item() > 0
            result['models'].append({'model': name, 'synthetic_mse': loss.item(), 'gradient_norm': norm.item()})
        result['passed'] = True
    except Exception:
        result['error'] = traceback.format_exc()
        raise
    finally:
        out.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
