"""Fixed synthetic field and actual training-window qualification; all updates discarded."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
from shared_geometry_io_r88 import PROCESS, freeze, verify, precision, write, append, stamp, acquire_gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    args = parser.parse_args()
    if args.device == 'cpu':
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        bound = None
    else:
        from prebind_shared_gpu import require_initial_binding
        bound = require_initial_binding()
        prior = json.loads((PROCESS/'cpu-gate001/gate.json').read_text())
        assert prior['passed'] and prior['discarded_updates'] == 8
        verify(json.loads((PROCESS/'cpu-gate001/config.json').read_text())['frozen'])
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from mixed_anchor_r40 import MixedPool
    from motion_transport_r74 import load_spatial
    from shared_geometry_data_r88 import batch
    from shared_geometry_r88 import make_heads, capture, stem, normalize, local_flow, transport, update, CausalCore
    precision(); cv2.setNumThreads(1)
    out = PROCESS/f'{args.device}-gate001'; out.mkdir(parents=True, exist_ok=False)
    frozen = freeze(['check_shared_geometry_r88.py'])
    write(out/'config.json', dict(at=stamp(), pid=os.getpid(), frozen=frozen,
        expected_discarded_updates=8, expected_full_F0_frames=16, expected_stem_only_frames=16,
        artificial_seed=8800, artificial_smoothing='avg5 stride1 pad2, then channel L2',
        artificial_shift=[1., -.5], data_steps=[8000, 8000], device=args.device))
    began, complete, updates, full_frames, stem_frames = time.perf_counter(), False, 0, 0, 0
    lock = schedule = None
    records = []
    try:
        pool = MixedPool()
        prepared = []
        for seed in (0, 1):
            inputs, target, identity = batch(pool, seed, 0)
            backbone, marker = load_spatial(seed)
            heads, rank = make_heads(backbone, seed)
            prepared.append((inputs, target, identity, backbone, marker, heads, rank))
        if bound:
            lock, schedule, selected = acquire_gpu(bound, out)
        device = torch.device(args.device)
        generator = torch.Generator().manual_seed(8800)
        previous = normalize(F.avg_pool2d(torch.randn(1, 4, 32, 40, generator=generator), 5, 1, 2)).to(device)
        truth = previous.new_tensor([1., -.5])[None, :, None, None].expand(1, 2, 32, 40)
        current, _ = transport(previous, previous, truth)
        current = normalize(current)
        with torch.no_grad():
            field = local_flow(previous, current)
            epe = (field-truth).square().sum(1).sqrt()[:, 8:-8, 8:-8]
        artificial = dict(median_quarter_EPE=float(epe.median()), finite=bool(torch.isfinite(field).all()))
        write(out/'artificial.json', artificial)
        assert artificial['finite'] and artificial['median_quarter_EPE'] <= .5
        for seed, (inputs, target, identity, backbone, marker, heads, rank) in enumerate(prepared):
            work_began = time.perf_counter()
            inputs, target, backbone = inputs.to(device), target.to(device), backbone.to(device)
            heads = {name: head.to(device) for name, head in heads.items()}
            with torch.no_grad():
                original = backbone(inputs[:4]); full_frames += 4
                shared, _ = capture(backbone, inputs[:4]); full_frames += 4
                features = stem(backbone, inputs); stem_frames += 8
                initial_difference = float((heads['plain'](features)-heads['complement'](features)).abs().max())
                RGB_difference = float((original-shared).abs().max())
            assert initial_difference <= 1e-6 and RGB_difference <= 1e-6
            row = dict(seed=seed, rank=rank, backbone=marker, data=identity,
                initial_descriptor_difference=initial_difference, unchanged_RGB_difference=RGB_difference,
                controls={})
            for name, head in heads.items():
                optimizer = torch.optim.Adam(head.parameters(), lr=1e-3, betas=(.9, .999), eps=1e-8)
                for step in range(2):
                    record = update(head, optimizer, features, target); updates += 1
                    append(out/'discarded-updates.jsonl', dict(seed=seed, mode=name, step=step+1, **record))
                assert all(p.grad is None and not p.requires_grad for p in backbone.parameters())
                folded = head.folded()
                with torch.no_grad():
                    descriptor = head(features)
                    difference = float((folded(features)-descriptor).abs().max())
                    raw = (shared[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)*255).astype(np.float32)
                    state = CausalCore()
                    first = state.step(raw, descriptor[:1])['true']
                    expected_bytes = raw.nbytes+descriptor[:1].numel()*descriptor.element_size()
                    assert np.array_equal(first, raw) and state.state_bytes() == expected_bytes
                    assert state.previous_descriptor.untyped_storage().nbytes() == descriptor[:1].numel()*4
                assert difference <= 1e-5
                parameters = sum(p.numel() for p in folded.parameters())+sum(p.numel() for p in backbone.parameters())
                assert parameters == 4507
                cross = None
                if name == 'complement':
                    weight = backbone.head.block1[-1].weight.reshape(12, 108)
                    kernel = head.kernel().reshape(4, 108)
                    cross = float((kernel@weight.T).norm()/(kernel.norm()*weight.norm()+1e-12))
                    assert cross <= 1e-5
                row['controls'][name] = dict(fold_max_difference=difference, cross_residual=cross,
                    deployed_parameters=parameters, trainable_degrees=sum(p.numel() for p in head.parameters()),
                    first_frame_max_difference=float(np.abs(first-raw).max()), compact_state_bytes=state.state_bytes())
            records.append(row); write(out/f'seed{seed}.json', row)
            if schedule:
                torch.cuda.synchronize(); schedule.yield_after_work(time.perf_counter()-work_began)
        verify(frozen)
        assert updates == 8 and full_frames == stem_frames == 16
        gpu = {}
        if schedule:
            from shared_gpu_admission import snapshot
            sample = snapshot(); write(out/'post-work-load.json', sample)
            own = [int(p.split(',')[2]) for r in sample['GPUs'] for p in r['processes'] if int(p.split(',')[1]) == os.getpid()]
            assert len(own) == 1 and own[0] <= 4096
            gpu = dict(observed_process_mib=own[0], peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(), resource=schedule.summary())
        write(out/'gate.json', dict(at=stamp(), passed=True, artificial=artificial, records=records,
            discarded_updates=updates, full_F0_frames=full_frames, stem_only_frames=stem_frames,
            formal_updates=0, new_DEV_pixels=0, **gpu))
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(ended=stamp(), completed_normally=complete, pid=os.getpid(),
            seconds=time.perf_counter()-began, discarded_updates=updates, full_F0_frames=full_frames,
            stem_only_frames=stem_frames, formal_updates=0, CUDA_initialized=torch.cuda.is_initialized(),
            resource=schedule.summary() if schedule else None))
        globals()['_held_gpu_lock'] = lock


if __name__ == '__main__':
    main()
