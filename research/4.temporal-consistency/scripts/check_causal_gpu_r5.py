"""Record actual CUDA initial-function/gradient and one-update sanity checks."""
import argparse
import copy

import torch

from causal_fst_r5 import CausalFST, MODES
from explore_temporal_controls import save_json
from train_exploration_r2 import new_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(13)
    device = torch.device('cuda')
    base = new_model(device)
    x = torch.rand(2, 2, 3, 180, 320, device=device)*.1
    target = torch.rand_like(x)*.7
    ref = CausalFST(copy.deepcopy(base), 'plain').to(device)
    y = ref.forward_pair(x)
    torch.sqrt((y-target)**2+1e-6).mean().backward()
    rows = []
    for mode in MODES:
        net = CausalFST(copy.deepcopy(base), mode).to(device)
        pred = net.forward_pair(x)
        loss = torch.sqrt((pred-target)**2+1e-6).mean()
        loss.backward()
        delta = float((pred-y).abs().max().detach())
        gd = max(float((p.grad-q.grad).abs().max()) for p, q in zip(net.backbone.parameters(), ref.backbone.parameters()))
        eg = sum(float(p.grad.abs().sum()) for p in net.branch_parameters())
        if mode != 'tsm_fixed':
            if delta > 2e-6 or gd > 2e-5:
                raise AssertionError((mode, delta, gd))
            if net.branch_parameters() and not eg > 0:
                raise AssertionError('branch cannot learn: '+mode)
        opt = torch.optim.Adam(net.parameters(), lr=1e-5)
        opt.step()
        with torch.no_grad():
            net.reset()
            net(x[:, 0])
            output = net(x[:, 1])
            if not bool(torch.isfinite(output).all()):
                raise AssertionError('nonfinite updated output')
        rows.append({'mode': mode, 'initial_output_max_abs_difference': delta,
                     'initial_backbone_gradient_max_abs_difference': gd,
                     'extra_gradient_l1': eg, 'one_update_finite': True,
                     'state_storage_bytes_batch2_180x320': net.state_storage_bytes()})
        del net, opt
    save_json(args.out, {'rows': rows, 'gpu': torch.cuda.get_device_name(),
        'torch': torch.__version__, 'cudnn_deterministic': True,
        'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32})


if __name__ == '__main__':
    main()
