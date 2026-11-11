"""One shared-window CPU integration check, without optimizer updates."""
import argparse
import json

import cv2
import numpy as np
import torch

from dynamic_training_r34 import BASE, TrainingPool, models
from exploration_r2_data import degrade
from train_dynamic_r34 import prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registered-data', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    mode = 'registered-data' if args.registered_data else 'synthetic'
    out = BASE/f'report/exploration-r33-prototype/training-interface-{mode}.json'
    assert not out.exists()
    identities = None
    if args.registered_data:
        pool = TrainingPool()
        low, target, identities = pool.batch(0, 0)
        duplicate = pool.batch(0, 0)
        assert torch.equal(low, duplicate[0]) and torch.equal(target, duplicate[1]) and identities == duplicate[2]
    else:
        rng = np.random.default_rng(34)
        # Independent noise on a slowly moving smooth artificial RGB pattern.
        y, x = np.mgrid[:128, :128]
        clean = np.stack([np.stack([((x+t)%128)/127, y/127, (x+y)/254], axis=-1)*255 for t in range(4)]).astype(np.float32)
        target = torch.from_numpy(np.stack([clean]*4).transpose(0, 1, 4, 2, 3).copy()/255)
        low = torch.from_numpy(np.stack([[degrade(frame, rng, photons=20, jpeg=False) for frame in clean]
                                        for _ in range(4)]).transpose(0, 1, 4, 2, 3).copy()/255)
    nets, initialization = models(0, torch.device('cpu'))
    rows = []
    initial = {}
    for name, net in nets.items():
        output = prediction(net, name, low)
        loss = (output-target).square().mean()
        assert torch.isfinite(loss)
        if name in ['spatial', 'causal']:
            initial[name] = output.detach()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        assert norm.item() > 0
        rows.append({'model': name, 'initial_mse': loss.item(), 'gradient_norm_before_clip': norm.item()})
    assert torch.equal(initial['spatial'], initial['causal'])
    value = {'mode': mode, 'batch_shape': list(low.shape), 'initialization': initialization, 'models': rows,
             'initial_spatial_causal_output_equal': True, 'identities': identities, 'cuda_initialized': torch.cuda.is_initialized(),
             'optimizer_updates': 0, 'scope': 'CPU integration on one window only; not trained quality or final benchmark.'}
    out.write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
