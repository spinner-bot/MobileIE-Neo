"""Meaningful operator, reference-transport, gradient and streaming gates before training."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
import torch

from continuation_r14 import create_model
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256
from transport_fst_r27 import MODES, TransportFST, gather_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    torch.manual_seed(270011)
    config = {'device': str(device), 'torch': torch.__version__, 'opencv': cv2.__version__,
              'deterministic_algorithms': True, 'cublas_workspace': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
              'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), Path(__file__).with_name('transport_fst_r27.py')]},
              'protocol_sha256': sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r27-protocol.md')}
    save_json(out/'config.json', config)
    rng = np.random.default_rng(270012)
    warps = []
    for channels in [3, 12]:
        prior = rng.random((2, 13, 17, channels), dtype=np.float32)*255
        current = rng.random(prior.shape, dtype=np.float32)*255
        for kind in ['integer', 'fractional', 'random', 'rejected']:
            flow = np.zeros((2, 13, 17, 2), np.float32)
            flow[:] = [1, -1] if kind == 'integer' else [.375, -.625]
            if kind == 'random':
                flow[:] = rng.uniform(-3, 3, flow.shape)
            mask = rng.random((2, 13, 17)) > .15
            if kind == 'rejected':
                mask[:] = False
            tensor = lambda v: torch.from_numpy(v.transpose(0, 3, 1, 2).copy()).to(device)
            actual, valid = gather_features(tensor(prior), tensor(current), flow, mask)
            expected = []
            y, x = np.mgrid[:13, :17].astype(np.float32)
            for b in range(2):
                sx, sy = x+flow[b, ..., 0], y+flow[b, ..., 1]
                accepted = (sx >= 0) & (sx <= 16) & (sy >= 0) & (sy <= 12) & mask[b]
                assert np.array_equal(valid[b].cpu().numpy(), accepted)
                value = cv2.remap(prior[b], sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                expected.append(np.where(accepted[..., None], value, current[b]))
            maximum = float((actual-tensor(np.stack(expected))).abs().max())
            assert maximum <= 1e-3
            warps.append({'channels': channels, 'kind': kind, 'max_abs_gray': maximum})
    tiny = torch.randn(1, 2, 3, 4, dtype=torch.float64, device=device, requires_grad=True)
    tiny_current = torch.randn_like(tiny, requires_grad=True)
    flow = torch.zeros(1, 3, 4, 2, dtype=torch.float64, device=device)
    flow[..., 0], flow[..., 1] = .375, -.625
    mask = torch.ones(1, 3, 4, dtype=torch.bool, device=device)
    gradcheck = torch.autograd.gradcheck(lambda p, c: gather_features(p, c, flow, mask)[0], (tiny, tiny_current))
    assert gradcheck
    original = create_model('plain', device).eval()
    u = torch.randn(1, 12, 13, 17, dtype=torch.float64, device=device)*10
    v = torch.randn_like(u)*10
    flow = torch.zeros(1, 13, 17, 2, device=device)
    flow[..., 0], flow[..., 1] = .375, -.625
    mask = torch.ones(1, 13, 17, dtype=torch.bool, device=device)
    net = TransportFST(copy.deepcopy(original.backbone).double(), 'copy')
    aligned, valid = gather_features(v, u, flow, mask)
    second, _ = gather_features(v.square(), u.square(), flow, mask)
    body = net.backbone.body
    a = body.weight1*body.weight2
    values = {}
    for mode in MODES:
        net.mode = mode
        values[mode] = net.combine(u, net.ordinary(v) if mode == 'post_transport' else v, flow, mask)
    gap = a/4*(u-aligned).square()
    equalities = [('post_after_warp', 'pre_square', gap), ('pre_square', 'cross', gap),
                  ('post_transport', 'post_after_warp', a/2*(second-aligned.square()))]
    identities = []
    accepted = valid[:, None].expand_as(u)
    for left, right, correction in equalities:
        residual = (values[left]-values[right]-correction)[accepted]
        scale = max(float(values[left].detach().abs().max()), float(values[right].detach().abs().max()), 1e-30)
        error = float(residual.detach().abs().max())/scale
        assert error <= 1e-12
        identities.append({'left': left, 'right': right, 'scaled_max_abs': error})
    current = torch.rand(1, 3, 13, 17, device=device)
    previous = torch.rand_like(current)
    controls = []
    with torch.no_grad():
        expected = original(current)
    for mode in MODES:
        model = TransportFST(copy.deepcopy(original.backbone), mode).to(device).eval()
        assert sum(p.numel() for p in model.parameters()) == 4075
        with torch.no_grad():
            model.reset()
            first = model.step(current)
            initial = float((first-expected).abs().max())
            replaced = float((model.step(current, history='current')-expected).abs().max())
            fallback = float((model.transform(current, previous, flow, torch.zeros_like(mask))-expected).abs().max())
            model.reset()
            model.step(previous)
            streamed = model.step(current, flow, mask)
            direct = model.transform(current, previous, flow, mask)
            stream_error = float((streamed-direct).abs().max())
            state_bytes = model.state_storage_bytes()
            assert state_bytes == (0 if mode == 'copy' else 12*13*17*4)
            model.reset()
            assert model.state_storage_bytes() == 0
        assert max(initial, replaced, fallback, stream_error) <= 2e-5
        p = previous.detach().clone().requires_grad_()
        c = current.detach().clone().requires_grad_()
        loss = model.transform(c, p, flow, mask).square().mean()
        loss.backward()
        past_grad = None if p.grad is None else float(p.grad.abs().sum())
        assert float(c.grad.abs().sum()) > 0
        assert past_grad is None if mode == 'copy' else past_grad is not None and past_grad > 0
        with torch.no_grad():
            influence = float((model.transform(current, previous, flow, mask)-model.transform(current, previous*.5, flow, mask)).abs().max())
        assert influence == 0 if mode == 'copy' else influence > 0
        controls.append({'mode': mode, 'parameters': 4075, 'first_frame_error': initial, 'current_limit_error': replaced,
                         'all_rejected_error': fallback, 'stream_direct_error': stream_error, 'state_bytes': state_bytes,
                         'past_gradient_l1': past_grad, 'past_perturbation_max_output': influence})
    save_json(out/'audit.json', {'warp_reference_checks': warps, 'feature_gradcheck': gradcheck, 'FP64_identities': identities,
                               'controls': controls, 'passed': True, 'scope': 'Implementation gates only; no training, quality evaluation or novelty evidence.'})
    print(json.dumps({'passed': True, 'device': str(device), 'controls': len(controls)}), flush=True)


if __name__ == '__main__':
    main()
