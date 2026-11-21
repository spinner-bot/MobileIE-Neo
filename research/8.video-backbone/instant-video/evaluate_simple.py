"""Complete causal full-length I1-S evaluation, only after both exact-budget seeds."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
import time
import traceback

import cv2
import numpy as np
import torch

from simple_io import BASE, ROOT, append, digest, identity, owned_cache, read, stamp, verify_training, write
from simple_core import make_models
from summarize_simple import ARMS, summarize
from motion_transport_r74 import load_spatial
from half_motion_r4 import half_forward_backward
from fixed_dis_r86 import FixedDIS, RecursivePipeline
from exploration_r2_data import evaluate

AUDIT = BASE.parent / 'evidence-reconciliation/process/delay-source-audit001.json'


def owned(path):
    assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    return path


def finals(seed):
    runs = sorted((BASE/'process').glob(f'simple-train-seed{seed}-attempt*/runtime.json'))
    assert runs and read(runs[-1])['completed_normally']
    assert read(runs[-1])['counts'] == dict(common_steps=80000, optimizer_updates=160000, student_images=1280000)
    markers = {}
    for level in ('all', 'output'):
        marker = read(runs[-1].parent/(level+'-final.json'))
        assert marker['seed'] == seed and marker['updates'] == 80000 and marker['unique_final'] and not marker['qualification']
        assert digest(owned(ROOT/marker['path'])) == marker['sha256']
        markers[level] = marker
    return markers


def load_reference(entry, shape):
    path = owned(ROOT/entry['path'])
    assert digest(path) == entry['sha256']
    value = np.load(path, mmap_mode='r', allow_pickle=False)
    assert value.dtype == np.float32 and value.shape == tuple(shape) and np.isfinite(value).all()
    return value


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1)
    os.nice(5)
    frozen = verify_training()
    evaluation = read(BASE/'process/simple-evaluation-source-freeze001.json')
    assert all(digest(ROOT/p) == h for p, h in evaluation['source_sha256'].items())
    assert read(BASE/'process/simple-chain001/runtime.json')['completed_normally']
    markers = {seed: finals(seed) for seed in (0, 1)}
    audit = read(AUDIT)
    out, assets = BASE/'results/simple-eval001', owned_cache('eval001')
    out.mkdir(parents=True, exist_ok=False)
    assets.mkdir(parents=True, exist_ok=False)
    began, complete = time.perf_counter(), False
    counts = dict(clips=0, student_images=0, CPU_synthetic_materialization_images=0,
                  output_rows=0, frame_rows=0, new_FB_bidirectional_pairs=0,
                  new_DIS_bidirectional_pairs=0, new_GT_flow_pairs=0, cached_GT_pairs_unique=0)
    write(out/'config.json', dict(at=stamp(), process=identity(), source_freeze=frozen,
          evaluation_source_freeze=evaluation, arms=ARMS, unique_finals=markers,
          scope='Full existing source prefixes; repeated DEV, not independent final',
          future_frames=0, RGB_PSNR_ceiling_dB=120, GPU_used=False,
          model_frames_per_frame=1, artificial_materialization_frames_recorded_separately=True,
          first_frame_cache_provenance_normalized_tolerance=1e-5,
          no_cost_or_phone_measurement=True))
    try:
        rows, seen_GT = [], set()
        with torch.inference_mode():
            for seed in (0, 1):
                original, provenance = load_spatial(seed)
                models = make_models(original)
                counts['CPU_synthetic_materialization_images'] += sum(m.frame_calls for m in models.values())
                for level, model in models.items():
                    model.load_state_dict(torch.load(ROOT/markers[seed][level]['path'], map_location='cpu', weights_only=True), strict=True)
                    model.eval().requires_grad_(False)
                for meta in audit['scenes']:
                    scene = meta['scene']
                    domain = 'SYN' if meta['domain'] == 'davis_train_pilot' else 'ND'
                    source = owned(ROOT/meta['path'])
                    assert digest(source) == meta['sha256']
                    with np.load(source, allow_pickle=False) as bundle:
                        low, gt = bundle['low'], bundle['gt']
                        fields, masks = bundle['reference_flow'], bundle['reference_mask']
                    item = next(v for v in audit['baseline_items'] if (v['seed'], v['scene']) == (seed, scene))
                    shape = item['shape']
                    assert low.shape == gt.shape == tuple(shape)
                    assert np.isfinite(low).all() and np.isfinite(gt).all() and np.isfinite(fields).all()
                    assert item['checkpoint_sha256'] == provenance['checkpoint_sha256']
                    tmax, height, width, _ = shape
                    assert fields.shape == (tmax-1, height, width, 2) and masks.shape == (tmax-1, height, width)
                    if scene not in seen_GT:
                        counts['cached_GT_pairs_unique'] += tmax-1
                        seen_GT.add(scene)
                    values = {arm: load_reference(item[key], shape) for arm, key in [('F0','F0'),('F0_FB','FB'),('F0_DIS','DIS')]}
                    paths = {arm: assets/f'seed{seed}--{scene}--{arm}.npy' for arm in ARMS[3:]}
                    predictions = {arm: np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=tuple(shape)) for arm, path in paths.items()}
                    ranges = {level: dict(min=float('inf'), max=-float('inf'), outside_elements=0) for level in models}
                    first_errors = {}
                    for level, model in models.items():
                        model.reset()
                        for t in range(tmax):
                            frame = torch.from_numpy(np.ascontiguousarray(low[t].transpose(2,0,1), dtype=np.float32))[None] / 255
                            raw = model(frame)[0].permute(1,2,0).numpy()
                            counts['student_images'] += 1
                            assert np.isfinite(raw).all()
                            state = ranges[level]
                            state['min'] = min(state['min'], float(raw.min()))
                            state['max'] = max(state['max'], float(raw.max()))
                            state['outside_elements'] += int(((raw < 0) | (raw > 1)).sum())
                            predictions[level][t] = np.clip(raw,0,1)*255
                        first_errors[level] = float(np.max(np.abs(predictions[level][0]-values['F0'][0])))/255
                        assert first_errors[level] <= 1e-5
                        ranges[level]['outside_fraction'] = state['outside_elements']/(tmax*height*width*3)
                        for suffix, motion in [('FB', half_forward_backward), ('DIS', FixedDIS())]:
                            pipeline = RecursivePipeline(motion)
                            for t in range(tmax):
                                predictions[level+'_'+suffix][t] = pipeline.step(predictions[level][t])[0]
                            if suffix == 'DIS':
                                assert motion.pairs == tmax-1
                            counts['new_'+suffix+'_bidirectional_pairs'] += tmax-1
                        model.reset()
                    append(out/'raw-ranges.jsonl', dict(seed=seed, scene=scene, units='Unclipped RGB01', values=ranges))
                    append(out/'source-checks.jsonl', dict(seed=seed, scene=scene, original_checkpoint=provenance['checkpoint_sha256'],
                           source_sha256=meta['sha256'], first_frame_normalized_difference=first_errors,
                           GT_only_in_metrics=True, own_posts_estimate_fields_from_own_raw=True))
                    for arm, prediction in predictions.items():
                        prediction.flush()
                        append(out/'prediction_manifest.jsonl', dict(seed=seed, domain=domain, scene=scene, arm=arm,
                               path=str(paths[arm].relative_to(ROOT)), sha256=digest(paths[arm]), shape=shape,
                               future_frames=0, first_source_frame=0, last_source_frame=tmax-1,
                               training_updates=80000, model_marker=markers[seed][arm.split('_')[0]]))
                    values.update(predictions)
                    for arm in ARMS:
                        summary, frames = evaluate(values[arm], gt, fields, masks)
                        assert all(f['temporal_error'] is not None for f in frames[1:])
                        for f in frames:
                            f['PSNR_ceiling_applied'] = f['psnr_rgb'] >= 120
                            f['psnr_rgb'] = min(f['psnr_rgb'],120.)
                            append(out/'per_frame.jsonl', dict(seed=seed,domain=domain,scene=scene,arm=arm,**f))
                        summary['psnr_rgb'] = float(np.mean([f['psnr_rgb'] for f in frames]))
                        assert all(np.isfinite(v) for v in summary.values())
                        row = dict(seed=seed,domain=domain,scene=scene,source_video=meta['source_video'],arm=arm,**summary)
                        rows.append(row)
                        append(out/'per_sequence.jsonl',row)
                        counts['output_rows'] += 1
                        counts['frame_rows'] += len(frames)
                    counts['clips'] += 1
                    print(dict(at=stamp(),seed=seed,scene=scene,seconds=time.perf_counter()-began,**counts),flush=True)
                    del predictions, values, pipeline, raw
                del models, original
        assert counts == dict(clips=20,student_images=1792,CPU_synthetic_materialization_images=4,
               output_rows=180,frame_rows=8064,new_FB_bidirectional_pairs=1752,
               new_DIS_bidirectional_pairs=1752,new_GT_flow_pairs=0,cached_GT_pairs_unique=438)
        summarize(rows,out)
        verify_training()
        assert all(digest(ROOT/p) == h for p,h in evaluation['source_sha256'].items())
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json',dict(at=stamp(),process=identity(),completed_normally=complete,
              seconds=time.perf_counter()-began,GPU_used=False,**counts))


if __name__ == '__main__':
    main()
