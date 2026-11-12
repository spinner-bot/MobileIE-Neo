"""CPU-only implementation gates, training sources only, twelve discarded SGD steps."""
import argparse
from datetime import datetime
import json
import subprocess
import time
import traceback

import cv2
import numpy as np
import torch

from aggregation_controls_r20 import blend
from aligned_raw_history_r11 import align_history
from dynamic_training_r34 import BASE, ROOT, digest
from half_motion_r4 import half_forward_backward
from mixed_anchor_r40 import MixedPool, prediction
from response_cue_r45 import CONTROLS, FrozenResponses, RecursiveControl, align_quantized, load_frozen


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def motion_for(raw):
    images = raw.detach().permute(0, 1, 3, 4, 2).contiguous().numpy()*255
    fields = []
    for t in range(1, raw.shape[1]):
        values = [half_forward_backward(v[t-1], v[t]) for v in images]
        flow = torch.from_numpy(np.stack([v[0] for v in values]).transpose(0, 3, 1, 2).copy())
        mask = torch.from_numpy(np.stack([v[1] for v in values])[:, None].copy()).bool()
        fields.append((flow, mask))
    return fields


def rollout(control, raw, cues, fields):
    outputs = [control(raw[:, 0])[0]]
    for t, (flow, mask) in enumerate(fields, 1):
        outputs.append(control(raw[:, t], outputs[-1], flow, mask, cues[:, t])[0])
    return torch.stack(outputs, dim=1)


def numpy_reference(raw, fields):
    images = raw.detach().permute(0, 1, 3, 4, 2).contiguous().numpy()*255
    outputs = []
    for b, values in enumerate(images):
        current = [values[0]]
        for t, (flow, mask) in enumerate(fields, 1):
            moved, _ = align_history(current[-1], values[t], flow[b].permute(1, 2, 0).numpy(), mask[b, 0].numpy())
            current.append(blend(values[t], moved, .5))
        outputs.append(np.stack(current))
    return torch.from_numpy(np.stack(outputs).transpose(0, 1, 4, 2, 3).copy()/255)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args()
    out = BASE/f'process/exploration-r45/cpu-attempt{args.attempt:03d}'
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    names = ['check_response_cue_r45.py', 'response_cue_r45.py', 'budget_causal_r33.py',
             'half_motion_r4.py', 'video_metrics.py', 'mixed_anchor_r40.py']
    write(out/'config.json', {'started_at': datetime.now().astimezone().isoformat(),
          'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'protocol_sha256': digest(BASE/'docs/exploration-r45-response-cue-prototype-protocol.md'),
          'source_sha256': {n: digest(BASE/'scripts'/n) for n in names},
          'scope': 'CPU implementation only; MixedPool training step8000, no DEV quality or formal training.'})
    started, complete, updates = time.perf_counter(), False, 0
    try:
        rng = np.random.default_rng(45)
        previous = torch.from_numpy(rng.random((2, 3, 33, 37), dtype=np.float32))
        current = torch.from_numpy(rng.random((2, 3, 33, 37), dtype=np.float32))
        mask = torch.from_numpy(rng.random((2, 1, 33, 37)) > .15)
        errors = []
        for dx, dy in [(0, 0), (1, -1), (.28125, -.71875), (.015625, -.015625)]:
            field = torch.empty((2, 2, 33, 37));field[:, 0] = dx;field[:, 1] = dy
            actual, valid = align_quantized(previous, current, field, mask)
            for b in range(2):
                expected, accepted = align_history(previous[b].permute(1, 2, 0).numpy(),
                    current[b].permute(1, 2, 0).numpy(), field[b].permute(1, 2, 0).numpy(), mask[b, 0].numpy())
                delta = float((actual[b]-torch.from_numpy(expected).permute(2, 0, 1)).abs().max())
                assert delta <= 1e-5 and np.array_equal(valid[b, 0].numpy(), accepted)
                errors.append({'offset': [dx, dy], 'batch': b, 'max_abs_delta': delta})
        zero_flow = torch.zeros(2, 2, 33, 37)
        for name in CONTROLS:
            control = RecursiveControl(name, 0)
            cues = torch.zeros(2, 2, 33, 37)
            rejected = control(current, previous, zero_flow, torch.zeros_like(mask), cues)[0]
            assert torch.equal(rejected, current) and torch.equal(control(current)[0], current)
            left = control(current, torch.zeros_like(previous), zero_flow, torch.ones_like(mask), cues)[0]
            right = control(current, torch.ones_like(previous), zero_flow, torch.ones_like(mask), cues)[0]
            assert float((left-right).abs().max().detach()) > 0
        print(json.dumps({'phase': 'toy_gates_passed', 'discarded_updates': updates}), flush=True)
        pool = MixedPool()
        print(json.dumps({'phase': 'training_pool_loaded', 'discarded_updates': updates}), flush=True)
        seeds = []
        for seed in [0, 1]:
            low, gt, context = pool.batch(seed, 8000)
            backbone, marker = load_frozen(seed)
            before = {n: t.clone() for n, t in backbone.state_dict().items()}
            with torch.no_grad():
                original = prediction(backbone, low)
            encoder = FrozenResponses(backbone)
            values = [encoder(frame) for frame in low.unbind(1)]
            raw = torch.stack([v[0] for v in values], dim=1)
            cues = torch.stack([v[1] for v in values], dim=1)
            parity = float((raw-original).abs().max())
            assert parity <= 1e-6 and bool(torch.isfinite(cues).all())
            raw = raw.clamp(0, 1)
            fields = motion_for(raw)
            reference = numpy_reference(raw, fields)
            initial, gradients = {}, {}
            controls = {n: RecursiveControl(n, seed) for n in CONTROLS}
            for name, control in controls.items():
                output = rollout(control, raw, cues, fields)
                delta = float((output-reference).abs().max().detach())
                assert delta <= 1e-5 and torch.equal(output[:, 0], raw[:, 0])
                initial[name] = delta
                if name == 'fixed':
                    continue
                optimizer = torch.optim.SGD(control.parameters(), lr=.01)
                norms = []
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    loss = (rollout(control, raw, cues, fields)-gt).square().mean()
                    assert bool(torch.isfinite(loss))
                    loss.backward()
                    entries = {n: float(p.grad.norm()) for n, p in control.named_parameters()}
                    assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in control.parameters())
                    assert sum(entries.values()) > 0
                    if name in ['rgb', 'response'] and step == 1:
                        assert entries['gate.0.weight'] > 0, 'First layer did not unlock'
                    norms.append(entries)
                    optimizer.step();updates += 1
                gradients[name] = norms
            assert all(torch.equal(before[n], t) for n, t in backbone.state_dict().items())
            assert all(not p.requires_grad and p.grad is None for p in backbone.parameters())
            seeds.append({'seed': seed, 'checkpoint_sha256': marker['sha256'], 'context': context,
                          'raw_parity': parity, 'initial_recursive_deltas': initial,
                          'discarded_gradient_norms': gradients, 'frozen_backbone_unchanged': True,
                          'control_parameters': {n: sum(p.numel() for p in c.parameters()) for n, c in controls.items()}})
            print(json.dumps({'phase': 'seed_gate_passed', 'seed': seed, 'discarded_updates': updates}), flush=True)
        assert updates == 12 and not torch.cuda.is_initialized()
        write(out/'gate.json', {'passed': True, 'warp_cases': errors, 'seeds': seeds,
              'discarded_SGD_updates': updates, 'formal_updates': 0, 'new_DEV_quality_rows': 0,
              'scope': 'Implementation only. No effectiveness, response-cue contribution or calibrated-variance claim.'})
        complete = True
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'discarded_SGD_updates': updates,
              'cuda_initialized': torch.cuda.is_initialized(), 'wall_seconds': time.perf_counter()-started,
              'ended_at': datetime.now().astimezone().isoformat()})
    print(json.dumps({'passed': True, 'discarded_updates': updates}), flush=True)


if __name__ == '__main__':
    main()
