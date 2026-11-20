"""Registered S1 fixed training-window screening; no optimization or new DEV."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'S1 is CPU-only'

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
from fixed_dis_r86 import RecursivePipeline
from aligned_raw_history_r11 import align_history
from aggregation_controls_r20 import blend
from exploration_r2_data import evaluate
from video_metrics import warp_previous
from transport import DISFields, SplatField, qualification
import numpy as np
import cv2
import torch

ARMS = ('raw', 'original_FB', 'backward_only', 'push_average', 'push_dispersion')
OUT = BASE / 'results/run001'
CACHE = ROOT / '.cache/experiments/topic8-single-direction/run001'


def verify():
    frozen = json.loads((BASE / 'process/source-freeze001.json').read_text())
    assert all(digest(ROOT / p) == s for p, s in frozen['source_sha256'].items())
    return frozen


def identity():
    values = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=os.getpid(), start_ticks=int(values[19]), state=values[0])


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
            means = {r['arm']: r for r in groups if r['seed'] == seed and r['domain'] == domain}
            for reference in ('original_FB', 'backward_only', 'push_average', 'raw'):
                candidate, ref = means['push_dispersion'], means[reference]
                comparisons.append(dict(seed=seed, domain=domain, reference=reference,
                    PSNR_delta=candidate['psnr_rgb'] - ref['psnr_rgb'],
                    Etemp_ratio=candidate['temporal_error'] / ref['temporal_error'],
                    worst_window_PSNR_delta=min(by_arm['push_dispersion'][k]['psnr_rgb'] - by_arm[reference][k]['psnr_rgb'] for k in by_arm[reference]),
                    PSNR_better=sum(by_arm['push_dispersion'][k]['psnr_rgb'] > by_arm[reference][k]['psnr_rgb'] for k in by_arm[reference]),
                    time_better=sum(by_arm['push_dispersion'][k]['temporal_error'] < by_arm[reference][k]['temporal_error'] for k in by_arm[reference])))
    gates = {}
    for reference, limit in [('original_FB', 1.02), ('backward_only', .95), ('push_average', 1.00)]:
        chosen = [r for r in comparisons if r['reference'] == reference]
        gates[reference] = all(r['PSNR_delta'] >= -.05 and r['worst_window_PSNR_delta'] >= -.5 and r['Etemp_ratio'] <= limit for r in chosen)
    return dict(groups=groups, comparisons=comparisons, gates=gates,
                allowed_complete_DEV_cost_registration=all(gates.values()),
                established_method=False, demonstrated_speedup=False, independent_final_evidence=False)


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    os.nice(5)
    frozen = verify()
    assert CACHE.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    OUT.mkdir(parents=True, exist_ok=False)
    CACHE.mkdir(parents=True, exist_ok=False)
    started, complete = time.perf_counter(), False
    counts = dict(backbone_images=0, windows=0, quality_rows=0, frame_rows=0,
                  DIS_pairs=0, DIS_calc_calls=0, reference_FB_pairs=0, optimizer_updates=0)
    write(OUT / 'config.json', dict(at=stamp(), process=identity(), named_source_freeze=frozen,
          device='cpu', seeds=[0, 1], data_steps=[8000, 8001], B=4, T=4, crop=128, arms=ARMS,
          new_DEV=False, independent_final_opened=False, inference_uses_GT=False,
          shared_fields_not_a_runtime_benchmark=True, wall_clock_cutoff=False))
    try:
        write(OUT / 'pure_array_qualification.json', qualification())
        pool, rows, checks, sources = MixedPool(), [], [], []
        write(OUT / 'data.json', pool.info)
        for seed in (0, 1):
            net, source = load_spatial(seed)
            sources.append(dict(seed=seed, **source))
            assert not net.training and not any(p.requires_grad for p in net.parameters())
            for step in (8000, 8001):
                low, target, identities = pool.batch(seed, step)
                with torch.inference_mode():
                    raw_batch = net(low.flatten(0, 1)).clamp(0, 1).unflatten(0, (4, 4))
                raw_batch = raw_batch.numpy().transpose(0, 1, 3, 4, 2).copy() * 255
                gt_batch = target.numpy().transpose(0, 1, 3, 4, 2).copy() * 255
                counts['backbone_images'] += 16
                append(OUT / 'windows.jsonl', dict(seed=seed, data_step=step, windows=identities))
                for batch_index, (raw, gt) in enumerate(zip(raw_batch, gt_batch)):
                    domain = 'SYN' if batch_index < 2 else 'ND'
                    key = f'seed{seed}-step{step}-window{batch_index}'
                    solver = DISFields()
                    fields = [solver(raw[t-1], raw[t]) for t in range(1, 4)]
                    refs = [half_forward_backward(gt[t-1], gt[t]) for t in range(1, 4)]
                    for flow, mask in refs:
                        _, bounds = warp_previous(gt[0], flow)
                        support = bounds & mask
                        support[:8] = support[-8:] = False
                        support[:, :8] = support[:, -8:] = False
                        assert support.any(), 'Empty reference support is a failure, not an omitted sample'
                    counts['DIS_pairs'] += solver.pairs
                    counts['DIS_calc_calls'] += solver.pairs * 2
                    counts['reference_FB_pairs'] += len(refs)
                    outputs = {arm: [raw[0]] for arm in ARMS}
                    cached = iter((back, mask) for back, forward, mask in fields)
                    original = RecursivePipeline(lambda _previous, _current: next(cached))
                    original_values = [original.step(raw[0])[0]]
                    for t, (back, forward, mask) in enumerate(fields, start=1):
                        splat = SplatField(forward)
                        moved, accepted = align_history(outputs['original_FB'][-1], raw[t], back, mask)
                        outputs['original_FB'].append(blend(raw[t], moved, .5))
                        moved, bounds = warp_previous(outputs['backward_only'][-1], back)
                        moved = np.where(bounds[..., None], moved, raw[t])
                        outputs['backward_only'].append(blend(raw[t], moved, .5))
                        outputs['push_average'].append(splat.apply(outputs['push_average'][-1], raw[t], False))
                        outputs['push_dispersion'].append(splat.apply(outputs['push_dispersion'][-1], raw[t], True))
                        outputs['raw'].append(raw[t])
                        original_values.append(original.step(raw[t])[0])
                        assert splat.covered.any()
                        append(OUT / 'transport_details.jsonl', dict(key=key, frame=t,
                            native_FB_accepted=float(accepted.mean()), backward_bounds=float(bounds.mean()),
                            push_covered=float(splat.covered.mean()), push_accepted=float(splat.accepted.mean()),
                            covered_rejected_by_dispersion=int((splat.covered & ~splat.accepted).sum()),
                            dispersion_quantiles=np.quantile(np.sqrt(splat.variance[splat.covered]), [0, .5, .9, .99, 1]).tolist()))
                    outputs = {arm: np.stack(value) for arm, value in outputs.items()}
                    difference = float(np.max(np.abs(outputs['original_FB'] - np.stack(original_values))) / 255)
                    assert difference <= 2e-6
                    checks.append(dict(key=key, cached_field_original_pipeline_max_normalized_difference=difference))
                    cache_path = CACHE / f'{key}.npz'
                    np.savez_compressed(cache_path, gt=gt, low=low[batch_index].numpy().transpose(0, 2, 3, 1) * 255,
                        backward_fields=np.stack([r[0] for r in fields]), forward_fields=np.stack([r[1] for r in fields]),
                        FB_masks=np.stack([r[2] for r in fields]), reference_flows=np.stack([r[0] for r in refs]),
                        reference_masks=np.stack([r[1] for r in refs]), **outputs)
                    append(OUT / 'prediction_manifest.jsonl', dict(key=key, seed=seed, domain=domain,
                        path=str(cache_path.relative_to(ROOT)), sha256=digest(cache_path)))
                    for arm, output in outputs.items():
                        assert output.shape == raw.shape and np.isfinite(output).all()
                        assert output.min() >= -1e-5 and output.max() <= 255 + 1e-5
                        assert np.array_equal(output[0], raw[0])
                        metrics, frame_rows = evaluate(output, gt, [r[0] for r in refs], [r[1] for r in refs])
                        assert all(r['temporal_error'] is not None for r in frame_rows[1:])
                        row = dict(key=key, seed=seed, domain=domain, arm=arm, **metrics)
                        rows.append(row)
                        append(OUT / 'per_sequence.jsonl', row)
                        for frame_row in frame_rows:
                            append(OUT / 'per_frame.jsonl', dict(key=key, seed=seed, domain=domain, arm=arm, **frame_row))
                        counts['quality_rows'] += 1
                        counts['frame_rows'] += 4
                    counts['windows'] += 1
                    write(OUT / 'live.json', dict(at=stamp(), seconds=time.perf_counter() - started, last=key, **counts))
                    print(dict(last=key, **counts), flush=True)
        assert counts == dict(backbone_images=64, windows=16, quality_rows=80, frame_rows=320,
                  DIS_pairs=48, DIS_calc_calls=96, reference_FB_pairs=48, optimizer_updates=0)
        write(OUT / 'models.json', sources)
        write(OUT / 'native_checks.json', dict(all_passed=True, windows=checks,
            max_normalized_difference=max(r['cached_field_original_pipeline_max_normalized_difference'] for r in checks)))
        write(OUT / 'summary.json', dict(at=stamp(), **counts, reference_Farneback_calc_calls=96, **summarize(rows)))
        verify()
        assert not torch.cuda.is_initialized()
        complete = True
    except BaseException:
        (OUT / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(OUT / 'runtime.json', dict(at=stamp(), process=identity(), completed_normally=complete,
              seconds=time.perf_counter() - started, CUDA_initialized=torch.cuda.is_initialized(), **counts))


if __name__ == '__main__':
    main()
