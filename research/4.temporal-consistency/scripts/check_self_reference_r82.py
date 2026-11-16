"""One real-window CPU qualification; discard twelve tiny updates, no DEV."""
import os
if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise SystemExit('Initially empty CUDA mask required before Torch imports.')

import json
import time
import traceback

import cv2
import torch

from causal_reference_io_r75 import ROOT, BASE, write, append, stamp, digest
from mixed_anchor_r40 import MixedPool
from motion_transport_r74 import load_spatial
from self_reference_r82 import MODES, make_models


def main():
    out = BASE/'process/exploration-r82/cpu-gate001'; out.mkdir(parents=True, exist_ok=False)
    protocol = BASE/'docs/exploration-r82-self-reference-prototype-protocol.md'
    sources = ['research/4.temporal-consistency/scripts/'+name for name in (
        'self_reference_r82.py', 'check_self_reference_r82.py', 'mixed_anchor_r40.py',
        'motion_transport_r74.py', 'dynamic_training_r34.py', 'budget_causal_r33.py',
        'native_training_r14.py', 'train_mixture_r4b.py', 'exploration_r2_data.py',
        'train_exploration_r2.py', 'explore_temporal_controls.py', 'causal_reference_io_r75.py')]
    sources += ['MobileIE/model/lle.py', 'MobileIE/model/utils.py', 'scripts/project-python', 'scripts/project_assets.py']
    frozen = {name: digest(ROOT/name) for name in sources}; protocol_sha = digest(protocol)
    write(out/'config.json', dict(at=stamp(), pid=os.getpid(), source_sha256=frozen,
        protocol_sha256=protocol_sha, expected_discarded_updates=12, expected_backbone_image_frames=32,
        data_selection=dict(seed=0, step=8000, B=4, T=4, crop=128),
        scope='One real MixedPool training window, function/gradient/state qualification only.'))
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    began, complete, updates, backbone_batches, backbone_frames = time.perf_counter(), False, 0, 0, 0
    records, total_h_calls = [], 0
    try:
        pool = MixedPool(); low, target, identities = pool.batch(0, 8000)
        write(out/'data.json', dict(at=stamp(), pool=pool.info, identities=identities,
            input_shape=list(low.shape), target_shape=list(target.shape), new_DEV_pixels=0))
        for seed in (0, 1):
            backbone, marker = load_spatial(seed)
            original = {name: value.clone() for name, value in backbone.state_dict().items()}
            raw = []
            with torch.no_grad():
                for t in range(4):
                    raw.append(backbone(low[:, t]).clamp(0, 1))
                    backbone_batches += 1; backbone_frames += len(low)
            raw = torch.stack(raw, dim=1)
            models = make_models(seed)
            assert all(sum(p.numel() for p in model.parameters()) == 304 for model in models.values())
            assert all(torch.equal(value, models['spatial'].state_dict()[name])
                       for model in models.values() for name, value in model.state_dict().items())
            for mode, model in models.items():
                model.train()
                initial = model.window(raw)
                initial_delta = float((initial-raw).abs().max()); assert initial_delta <= 1e-6
                before = {name: value.clone() for name, value in model.state_dict().items()}
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
                step_records = []
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model.window(raw).clamp(0, 1)
                    loss = (prediction-target).square().mean()
                    assert bool(torch.isfinite(loss)); loss.backward()
                    norms = {}
                    for name, parameter in model.named_parameters():
                        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                        norms[name] = float(parameter.grad.norm())
                    assert norms['head.4.weight'] > 0
                    total_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.))
                    optimizer.step(); updates += 1
                    row = dict(seed=seed, mode=mode, step=step+1, loss=float(loss.detach()),
                        gradient_norms=norms, preclip_gradient_norm=total_norm)
                    step_records.append(row); append(out/'discarded-updates.jsonl', row)
                delta = max(float((before[name]-value).abs().max()) for name, value in model.state_dict().items())
                assert delta > 0
                model.eval()
                with torch.no_grad():
                    same = model.window(raw, history='current')
                    self_delta = float((same-raw).abs().max())
                    if mode == 'centered':
                        assert self_delta <= 1e-6
                    # Compare the existing final training-window output to the
                    # streaming route without introducing another window call.
                    model.reset()
                    streamed = torch.stack([model.step(frame) for frame in raw.unbind(1)], dim=1)
                    state_bytes = model.state_bytes(); model.reset()
                    assert model.state_bytes() == 0
                # Direct pair sensitivity after the two discarded updates.
                current = raw[:, 1].detach()
                previous = raw[:, 0].detach().clone().requires_grad_(True)
                output = model(current, previous)
                probe = output[:, :, 32:96, 32:96].square().mean()
                gradient = torch.autograd.grad(probe, previous, allow_unused=True)[0]
                past_gradient = 0. if gradient is None else float(gradient.abs().sum())
                assert bool(torch.isfinite(output).all())
                assert past_gradient == 0 if mode == 'spatial' else past_gradient > 0
                expected_bytes = 0 if mode == 'spatial' else 4*3*128*128*4
                assert state_bytes == expected_bytes
                records.append(dict(seed=seed, mode=mode, checkpoint=marker, parameters=304,
                    total_frozen_plus_new_parameters=4379, initial_max_delta=initial_delta,
                    final_current_replacement_max_delta=self_delta, final_parameter_max_change=delta,
                    final_past_gradient_L1=past_gradient, actual_stream_batch_state_bytes=state_bytes,
                    state_bytes_per_image=state_bytes//4, final_readout_gradient_norm=step_records[-1]['gradient_norms']['head.4.weight'],
                    actual_H_calls=model.head_calls, streamed_finite=bool(torch.isfinite(streamed).all())))
                assert bool(torch.isfinite(streamed).all())
                total_h_calls += model.head_calls
            assert all(torch.equal(value, backbone.state_dict()[name]) for name, value in original.items())
            assert not backbone.training and not any(p.requires_grad for p in backbone.parameters())
        assert updates == 12 and backbone_batches == 8 and backbone_frames == 32
        assert frozen == {name:digest(ROOT/name) for name in sources} and protocol_sha == digest(protocol)
        assert not torch.cuda.is_initialized()
        write(out/'gate.json', dict(at=stamp(), passed=True, records=records,
            discarded_updates=updates, backbone_batch_calls=backbone_batches, backbone_image_frames=backbone_frames,
            actual_H_calls=total_h_calls, formal_updates=0, new_DEV_pixels=0, CUDA_initialized=False,
            scope='Function/self-reference/past-gradient/stream-state feasibility only. All learned states discarded; no quality, novelty or phone result.'))
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(ended=stamp(), completed_normally=complete, pid=os.getpid(),
            seconds=time.perf_counter()-began, discarded_updates=updates, backbone_batches=backbone_batches,
            backbone_image_frames=backbone_frames, completed_controls=len(records), actual_H_calls=total_h_calls,
            new_GPU=False, CUDA_initialized=torch.cuda.is_initialized(), formal_updates=0))


if __name__ == '__main__':
    main()
