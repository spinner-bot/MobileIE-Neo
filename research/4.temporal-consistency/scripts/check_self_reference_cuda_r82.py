"""One fixed CUDA qualification after real CPU preparation; discard twelve SGD updates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'scripts'))
from prebind_shared_gpu import require_initial_binding
BOUND = require_initial_binding()

import json
import time
import traceback

import cv2
import torch

from causal_reference_io_r75 import ROOT, BASE, write, append, stamp, digest
from mixed_anchor_r40 import MixedPool
from motion_transport_r74 import load_spatial
from self_reference_r82 import MODES, make_models
from self_reference_io_r82 import PROCESS, verify, acquire_gpu, precision


def main():
    out = PROCESS/'cuda-gate001'; out.mkdir(parents=True, exist_ok=False)
    frozen = verify(); frozen_sha = digest(PROCESS/'source-freeze001.json')
    cpu = json.loads((PROCESS/'cpu-gate001/gate.json').read_text())
    assert cpu['passed'] and cpu['discarded_updates'] == 12
    write(out/'config.json', dict(at=stamp(), pid=os.getpid(), frozen_sources=frozen,
        frozen_sources_sha256=frozen_sha, expected_discarded_updates=12, expected_backbone_image_frames=32,
        data_selection=dict(seed=0, step=8000, B=4, T=4, crop=128),
        scope='One real MixedPool training window, function/gradient/state qualification only.'))
    precision(); cv2.setNumThreads(1)
    began, complete, updates, backbone_batches, backbone_frames = time.perf_counter(), False, 0, 0, 0
    records, total_h_calls, schedule, gpu_lock = [], 0, None, None
    try:
        pool = MixedPool(); low, target, identities = pool.batch(0, 8000)
        write(out/'data.json', dict(at=stamp(), pool=pool.info, identities=identities,
            input_shape=list(low.shape), target_shape=list(target.shape), new_DEV_pixels=0))
        prepared = {seed: load_spatial(seed) for seed in (0, 1)}
        models_cpu = {seed: make_models(seed) for seed in (0, 1)}
        gpu_lock, schedule, selected = acquire_gpu(BOUND, out)
        low, target = low.cuda(), target.cuda()
        torch.cuda.reset_peak_memory_stats()
        for seed in (0, 1):
            backbone, marker = prepared[seed]
            backbone = backbone.cuda()
            original = {name: value.clone() for name, value in backbone.state_dict().items()}
            raw = []
            with torch.no_grad():
                for t in range(4):
                    raw.append(backbone(low[:, t]).clamp(0, 1))
                    backbone_batches += 1; backbone_frames += len(low)
            raw = torch.stack(raw, dim=1)
            models = {name: model.cuda() for name, model in models_cpu[seed].items()}
            assert all(sum(p.numel() for p in model.parameters()) == 304 for model in models.values())
            assert all(torch.equal(value, models['spatial'].state_dict()[name])
                       for model in models.values() for name, value in model.state_dict().items())
            for mode, model in models.items():
                model.train()
                initial = model.window(raw)
                initial_delta = float((initial-raw).detach().abs().max()); assert initial_delta <= 1e-6
                before = {name: value.clone() for name, value in model.state_dict().items()}
                optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
                step_records = []
                for step in range(2):
                    schedule.check_memory(); started = time.perf_counter()
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
                    torch.cuda.synchronize(); schedule.yield_after_work(time.perf_counter()-started)
                delta = max(float((before[name]-value).abs().max()) for name, value in model.state_dict().items())
                assert delta > 0
                model.eval()
                with torch.no_grad():
                    same = model.window(raw, history='current')
                    self_delta = float((same-raw).abs().max())
                    if mode == 'centered':
                        assert self_delta <= 1e-6
                    functional = model.window(raw)
                    model.reset()
                    streamed = torch.stack([model.step(frame) for frame in raw.unbind(1)], dim=1)
                    streaming_delta = float((functional-streamed).abs().max())
                    assert streaming_delta <= 1e-6
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
                    streaming_max_delta=streaming_delta,
                    final_past_gradient_L1=past_gradient, actual_stream_batch_state_bytes=state_bytes,
                    state_bytes_per_image=state_bytes//4, final_readout_gradient_norm=step_records[-1]['gradient_norms']['head.4.weight'],
                    actual_H_calls=model.head_calls, streamed_finite=bool(torch.isfinite(streamed).all())))
                assert bool(torch.isfinite(streamed).all())
                total_h_calls += model.head_calls
            assert all(torch.equal(value, backbone.state_dict()[name]) for name, value in original.items())
            assert not backbone.training and not any(p.requires_grad for p in backbone.parameters())
        assert updates == 12 and backbone_batches == 8 and backbone_frames == 32
        verify()
        from shared_gpu_admission import snapshot
        sample = snapshot(); write(out/'post-work-load.json', sample)
        own = [int(p.split(',')[2]) for r in sample['GPUs'] for p in r['processes'] if int(p.split(',')[1]) == os.getpid()]
        assert len(own) == 1 and own[0] <= 4096
        assert torch.cuda.max_memory_reserved() <= 2048*1048576
        write(out/'gate.json', dict(at=stamp(), passed=True, records=records,
            frozen_sources_sha256=frozen_sha, observed_own_process_mib=own[0],
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), resource=schedule.summary(),
            discarded_updates=updates, backbone_batch_calls=backbone_batches, backbone_image_frames=backbone_frames,
            actual_H_calls=total_h_calls, formal_updates=0, new_DEV_pixels=0, CUDA_initialized=True,
            scope='Function/self-reference/past-gradient/stream-state feasibility only. All learned states discarded; no quality, novelty or phone result.'))
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(ended=stamp(), completed_normally=complete, pid=os.getpid(),
            seconds=time.perf_counter()-began, discarded_updates=updates, backbone_batches=backbone_batches,
            backbone_image_frames=backbone_frames, completed_controls=len(records), actual_H_calls=total_h_calls,
            new_GPU=True, CUDA_initialized=torch.cuda.is_initialized(), formal_updates=0,
            resource=schedule.summary() if schedule else None))
        globals()['_held_gpu_lock'] = gpu_lock


if __name__ == '__main__':
    main()
