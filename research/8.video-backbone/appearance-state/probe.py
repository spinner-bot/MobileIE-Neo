"""One fixed B1 training-pool screen: no optimization, GPU, DEV or final data."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
from pathlib import Path
import sys
import json
import time
import traceback
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT / 'research/4.temporal-consistency/scripts'))
from causal_reference_io_r75 import write, append, digest, stamp
from mixed_anchor_r40 import MixedPool
from motion_transport_r74 import load_spatial
from half_motion_r4 import half_forward_backward
from fixed_dis_r86 import FixedDIS, RecursivePipeline
from exploration_r2_data import evaluate
from video_metrics import warp_previous
from bank import FeatureBank, continue_from_head, qualification, STATE_BYTES_FP32
import torch
import numpy as np
import cv2

OUT = BASE / 'results/run001'
CACHE = ROOT / '.cache/experiments/topic8-appearance-state/run001'
ARMS = ('raw', 'original_FB', 'original_DIS', 'history_bank', 'current_bank')


def identity():
    data = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=os.getpid(), start_ticks=int(data[19]), state=data[0])


def verify():
    frozen = json.loads((BASE / 'process/source-freeze001.json').read_text())
    assert all(digest(ROOT / p) == h for p, h in frozen['source_sha256'].items())
    return frozen


def summarize(rows):
    groups, comparisons = [], []
    for seed in (0, 1):
        for domain in ('SYN', 'ND'):
            by_arm = {}
            for arm in ARMS:
                selected = [r for r in rows if (r['seed'], r['domain'], r['arm']) == (seed, domain, arm)]
                assert len(selected) == 4
                by_arm[arm] = {r['key']: r for r in selected}
                groups.append(dict(seed=seed, domain=domain, arm=arm, windows=4,
                    **{m: float(np.mean([r[m] for r in selected])) for m in ('psnr_rgb', 'ssim_rgb', 'edge_mae', 'temporal_error')}))
            means = {r['arm']: r for r in groups if (r['seed'], r['domain']) == (seed, domain)}
            for ref in ('raw', 'current_bank', 'original_FB', 'original_DIS'):
                delta = means['history_bank']['psnr_rgb']-means[ref]['psnr_rgb']
                worst = min(by_arm['history_bank'][k]['psnr_rgb']-by_arm[ref][k]['psnr_rgb'] for k in by_arm[ref])
                ratio = means['history_bank']['temporal_error']/means[ref]['temporal_error']
                limit = .95 if ref in ('raw', 'current_bank') else 1.05
                comparisons.append(dict(seed=seed, domain=domain, reference=ref, PSNR_delta=delta,
                    worst_window_PSNR_delta=worst, Etemp_ratio=ratio, Etemp_limit=limit,
                    mean_quality_passed=delta >= -.05, worst_quality_passed=worst >= -.5,
                    temporal_passed=ratio <= limit, passed=delta >= -.05 and worst >= -.5 and ratio <= limit))
    return dict(groups=groups, comparisons=comparisons, complete_passed=all(r['passed'] for r in comparisons),
        allowed_separate_video_learning_cost_registration=all(r['passed'] for r in comparisons),
        established_innovation=False, independent_final_evidence=False, measured_phone_speed=False)


def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); cv2.setNumThreads(1); os.nice(5)
    frozen = verify()
    assert CACHE.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    OUT.mkdir(parents=True, exist_ok=False); CACHE.mkdir(parents=True, exist_ok=False)
    start, complete = time.perf_counter(), False
    counts = dict(direct_full_model_images=0, shared_head_prefix_images=0, decomposed_suffix_images=0,
        history_suffix_images=0, current_suffix_images=0, windows=0, quality_rows=0, frame_rows=0,
        deployment_FB_pairs=0, deployment_DIS_pairs=0, GT_metric_FB_pairs=0, optimizer_updates=0)
    write(OUT / 'config.json', dict(at=stamp(), process=identity(), source_freeze=frozen,
        seeds=[0, 1], data_steps=[8000, 8001], device='cpu', arms=ARMS,
        inference_uses_GT=False, independent_final_opened=False, new_DEV=False,
        shared_execution_not_runtime_benchmark=True, wall_clock_cutoff=False))
    try:
        write(OUT / 'array-qualification.json', qualification())
        pool, rows, checks, sources = MixedPool(), [], [], []
        write(OUT / 'data.json', pool.info)
        for seed in (0, 1):
            net, source = load_spatial(seed)
            original = {k: v.clone() for k, v in net.state_dict().items()}
            sources.append(dict(seed=seed, **source))
            for step in (8000, 8001):
                low, target, metadata = pool.batch(seed, step)
                append(OUT / 'windows.jsonl', dict(seed=seed, data_step=step, windows=metadata))
                for index in range(4):
                    key = f'seed{seed}-step{step}-window{index}'
                    domain = 'SYN' if index < 2 else 'ND'
                    history, spatial = FeatureBank(), FeatureBank(current_only=True)
                    outputs = {arm: [] for arm in ARMS}
                    fb, dis = RecursivePipeline(half_forward_backward), RecursivePipeline(FixedDIS())
                    maximum = 0.
                    with torch.inference_mode():
                        for frame in range(4):
                            x = low[index, frame:frame+1]
                            direct = net(x); counts['direct_full_model_images'] += 1
                            z = net.head.block1(x); counts['shared_head_prefix_images'] += 1
                            raw = continue_from_head(net, z); counts['decomposed_suffix_images'] += 1
                            error = float((raw-direct).abs().max()); maximum = max(maximum, error)
                            assert error <= 2e-6
                            hz, hd = history.step(x, z); sz, sd = spatial.step(x, z)
                            h = continue_from_head(net, hz); counts['history_suffix_images'] += 1
                            s = continue_from_head(net, sz); counts['current_suffix_images'] += 1
                            assert all(torch.isfinite(v).all() for v in (raw, h, s))
                            if frame == 0:
                                assert torch.equal(h, raw) and torch.equal(s, raw)
                            assert hd['stored_bytes'] == STATE_BYTES_FP32 and sd['stored_bytes'] == 0
                            assert history.table.untyped_storage().data_ptr() != z.untyped_storage().data_ptr()
                            displayed = {arm: v[0].clamp(0, 1).numpy().transpose(1, 2, 0).copy()*255
                                for arm, v in [('raw', raw), ('history_bank', h), ('current_bank', s)]}
                            displayed['original_FB'], _ = fb.step(displayed['raw'])
                            displayed['original_DIS'], _ = dis.step(displayed['raw'])
                            for arm in ARMS:
                                outputs[arm].append(displayed[arm])
                            append(OUT / 'bank-details.jsonl', dict(key=key, frame=frame, history=hd, current=sd,
                                raw_ranges={arm: [float(v.min()), float(v.max())] for arm, v in [('raw', raw), ('history_bank', h), ('current_bank', s)]}))
                    counts['deployment_FB_pairs'] += 3; counts['deployment_DIS_pairs'] += dis.motion.pairs
                    outputs = {arm: np.stack(values) for arm, values in outputs.items()}
                    gt = target[index].numpy().transpose(0, 2, 3, 1).copy()*255
                    refs = [half_forward_backward(gt[t-1], gt[t]) for t in range(1, 4)]
                    counts['GT_metric_FB_pairs'] += len(refs)
                    for flow, mask in refs:
                        _, bounds = warp_previous(gt[0], flow)
                        support = bounds & mask
                        support[:8] = support[-8:] = False; support[:, :8] = support[:, -8:] = False
                        assert support.any()
                    path = CACHE / (key+'.npz')
                    np.savez_compressed(path, gt=gt, low=low[index].numpy().transpose(0, 2, 3, 1)*255,
                        reference_flows=np.stack([p[0] for p in refs]), reference_masks=np.stack([p[1] for p in refs]), **outputs)
                    append(OUT / 'prediction_manifest.jsonl', dict(key=key, seed=seed, domain=domain,
                        path=str(path.relative_to(ROOT)), sha256=digest(path)))
                    for arm, output in outputs.items():
                        assert np.array_equal(output[0], outputs['raw'][0])
                        metrics, details = evaluate(output, gt, [p[0] for p in refs], [p[1] for p in refs])
                        assert all(r['temporal_error'] is not None for r in details[1:])
                        assert all(np.isfinite(v) for v in metrics.values())
                        row = dict(key=key, seed=seed, domain=domain, arm=arm, **metrics)
                        rows.append(row); append(OUT / 'per_sequence.jsonl', row)
                        for detail in details:
                            append(OUT / 'per_frame.jsonl', dict(key=key, seed=seed, domain=domain, arm=arm, **detail))
                        counts['quality_rows'] += 1; counts['frame_rows'] += 4
                    checks.append(dict(key=key, decomposition_max_normalized_error=maximum,
                        first_frame_exact=True, compact_state_bytes=STATE_BYTES_FP32, current_has_no_past_state=True))
                    counts['windows'] += 1
                    write(OUT / 'live.json', dict(at=stamp(), process=identity(), seconds=time.perf_counter()-start, last=key, **counts))
                    print(dict(last=key, **counts), flush=True)
            assert all(torch.equal(v, original[k]) for k, v in net.state_dict().items())
        assert counts == dict(direct_full_model_images=64, shared_head_prefix_images=64, decomposed_suffix_images=64,
            history_suffix_images=64, current_suffix_images=64, windows=16, quality_rows=80, frame_rows=320,
            deployment_FB_pairs=48, deployment_DIS_pairs=48, GT_metric_FB_pairs=48, optimizer_updates=0)
        write(OUT / 'models.json', sources)
        write(OUT / 'native-checks.json', dict(all_passed=True, windows=checks, frozen_weights_unchanged=True))
        write(OUT / 'summary.json', dict(at=stamp(), **counts, **summarize(rows)))
        verify(); assert not torch.cuda.is_initialized(); complete = True
    except BaseException:
        (OUT / 'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(OUT / 'runtime.json', dict(at=stamp(), process=identity(), completed_normally=complete,
            seconds=time.perf_counter()-start, CUDA_initialized=torch.cuda.is_initialized(), **counts))


if __name__ == '__main__':
    main()
