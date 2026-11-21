"""Finite CPU source/gradient/state qualification, with no real video or formal training."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import json
from pathlib import Path
import sys
import time
import traceback

import torch

from core import Adapted, ROOT, VENDOR, objective, sequence
sys.path.insert(0, str(ROOT / 'research/4.temporal-consistency/scripts'))
from causal_reference_io_r75 import digest, write
from motion_transport_r74 import load_spatial
from instant_video_models.stabilizers.controlled_conv_stabilizer import _spatiotemporal_fusion

BASE = Path(__file__).resolve().parent
OUT = BASE / 'process/preflight001'


def verify():
    own = json.loads((BASE / 'process/preflight-source-freeze001.json').read_text())
    assert all(digest(ROOT / p) == h for p, h in own['source_sha256'].items())
    vendor = json.loads((VENDOR / 'SOURCE.json').read_text())
    assert all(digest(VENDOR / p) == v['sha256'] for p, v in vendor['files'].items())
    return own


def fusion_fixture():
    rows = []
    for k in (1, 3):
        q = torch.zeros(1, 64, 8, 10)
        z = torch.full((1, 3, 8, 10), 3.)
        memory = torch.full_like(z, 2.)
        weight = torch.zeros(3 * k * k, 64, 3, 3)
        bias = torch.full((3 * k * k,), -4.)
        actual = _spatiotemporal_fusion(q, z, weight, bias, memory, k)
        ep = torch.exp(torch.tensor(-4.))
        denominator = 1 + k * k * ep
        expected_interior = (3 + 2 * k * k * ep) / denominator
        expected_corner = (3 + 2 * (1 if k == 1 else 4) * ep) / denominator
        interior = float((actual[:, :, 3, 4] - expected_interior).abs().max())
        corner = float((actual[:, :, 0, 0] - expected_corner).abs().max())
        assert max(interior, corner) <= 1e-6
        rows.append(dict(kernel=k, interior_max_difference=interior,
                         corner_max_difference=corner,
                         fixed_zero_head_current_weight=float(1 / denominator),
                         actual_random_initialized_controller_weight_not_claimed=True))
    return rows


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    os.nice(5)
    frozen = verify()
    OUT.mkdir(parents=True, exist_ok=False)
    began, complete = time.perf_counter(), False
    counts = dict(original_F0_images=0, adapted_backbone_images=0, discarded_optimizer_updates=0,
                  new_real_video_frames=0, new_flow_calls=0, GPU_used=False)
    try:
        torch.manual_seed(9300)
        fixture = fusion_fixture()
        original, provenance = load_spatial(0)
        original.eval()
        frames = torch.rand(1, 4, 3, 24, 32)
        target = torch.rand_like(frames)
        with torch.no_grad():
            raw = torch.stack([original(frames[:, t]) for t in range(4)], dim=1)
        counts['original_F0_images'] += 4
        variants = {}
        for spatial in (False, True):
            torch.manual_seed(9300)
            model = Adapted(original, spatial)
            model.materialize()
            before = {n: p.detach().clone() for n, p in model.original_parameters().items()}
            with torch.no_grad():
                initial = sequence(model, frames)
                first = float((initial[:, 0] - raw[:, 0]).abs().max())
                replay = float((sequence(model, frames) - initial).abs().max())
                changed = frames.clone()
                changed[:, 2:] = 1 - changed[:, 2:]
                prefix = float((sequence(model, changed)[:, :2] - initial[:, :2]).abs().max())
                model.enable(False)
                disabled = float((sequence(model, frames) - raw).abs().max())
                model.enable()
            assert (first, replay, prefix, disabled) == (0., 0., 0., 0.)
            trainable = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable, lr=1e-4)
            start = [p.detach().clone() for p in trainable]
            updates = []
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                outputs = sequence(model, frames)
                loss = objective(outputs, target)
                assert torch.isfinite(loss)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable)
                nonzero = sum(bool((p.grad != 0).any()) for p in trainable)
                assert nonzero > 0
                assert all(p.grad is None for p in model.original_parameters().values())
                optimizer.step()
                assert all(torch.isfinite(p).all() for p in model.parameters())
                counts['discarded_optimizer_updates'] += 1
                updates.append(dict(loss=float(loss.detach()), nonzero_gradient_parameter_tensors=nonzero))
            assert all(torch.equal(before[n], p) for n, p in model.original_parameters().items())
            changed_parameters = sum(not torch.equal(a, b) for a, b in zip(start, trainable))
            assert changed_parameters > 0
            with torch.no_grad():
                sequence(model, frames)
            state = model.state_inventory()
            pixel_bytes = sum(r['bytes'] for r in state['tensors']
                              if r['newly_counted_storage'] and r['shape'][-2:] == [24, 32])
            fixed_bytes = state['persistent_tensor_bytes'] - pixel_bytes
            assert pixel_bytes % (24 * 32) == 0
            state['analytic_1080p_retained_bytes'] = pixel_bytes // (24 * 32) * 1080 * 1920 + fixed_bytes
            state['analytic_bytes_per_image_pixel'] = pixel_bytes // (24 * 32)
            cost = model.parameters_and_cost()
            variants['spatial' if spatial else 'controlled'] = dict(
                passed=True, first_frame_max_difference=first, reset_replay_max_difference=replay,
                future_change_prefix_max_difference=prefix, disabled_max_difference=disabled,
                original_frozen_parameters_unchanged=True, changed_trainable_parameter_tensors=changed_parameters,
                updates=updates, counts=dict(adapted_backbone_images=model.frame_calls,
                                            discarded_optimizer_updates=2),
                costs=cost, state=state)
            counts['adapted_backbone_images'] += model.frame_calls
            model.reset()
            del model, optimizer, trainable, start, before
        verify()
        assert counts['discarded_optimizer_updates'] == 4
        write(OUT / 'qualification.json', dict(passed=True, source_freeze=frozen,
              original_F0_provenance=provenance, fusion_fixture=fixture, variants=variants,
              all_qualification_weights_discarded=True, counts=counts,
              actual_video_quality_or_phone_speed_claimed=False, formal_training_started=False))
        complete = True
    except BaseException:
        (OUT / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        active = locals().get('model')
        if active is not None:
            counts['adapted_backbone_images'] += active.frame_calls
        write(OUT / 'runtime.json', dict(completed_normally=complete,
              seconds=time.perf_counter() - began, process_pid=os.getpid(), **counts))


if __name__ == '__main__':
    main()
