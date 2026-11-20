"""Unique O1 finals on complete fixed training-scene draws, never DEV/final."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import argparse
import time
import traceback
from study_io import BASE, ROOT, read, write, append, digest, stamp, verify, identity, owned_cache
from mixed_anchor_r40 import MixedPool
from motion_transport_r74 import load_spatial
from half_motion_r4 import half_forward_backward
from fixed_dis_r86 import FixedDIS, RecursivePipeline
from exploration_r2_data import evaluate
from core import models, mix
import numpy as np
import torch
import cv2

ARMS = ('F0', 'F0_FB', 'F0_DIS', *[a+'_'+m for a in ('coordinate', 'response') for m in ('direct', 'FB', 'DIS', 'causal', 'zero')])


def finals(seed):
    runs = sorted((BASE / 'process').glob(f'train-seed{seed}-attempt*/runtime.json'))
    assert runs and read(runs[-1])['completed_normally']
    assert read(runs[-1])['counts']['common_steps'] == 4000
    result = {}
    for arm in ('coordinate', 'response'):
        marker = read(runs[-1].parent / (arm+'-final.json'))
        assert marker['seed'] == seed and marker['updates'] == 4000 and marker['unique_final'] and not marker['qualification']
        path = ROOT / marker['path']
        assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo') and digest(path) == marker['sha256']
        result[arm] = marker
    return result


def as_image(tensor):
    return tensor.clamp(0, 1).numpy().transpose(0, 2, 3, 1).copy()*255


def main(seed):
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    cv2.setNumThreads(1); os.nice(5)
    frozen, markers = verify('training-source-freeze001.json'), finals(seed)
    out, cache = BASE / f'results/held-seed{seed}', owned_cache(f'held-seed{seed}')
    out.mkdir(parents=True, exist_ok=False); cache.mkdir(parents=True, exist_ok=False)
    began, complete = time.perf_counter(), False
    counts = dict(F0_images=0, student_images=0, student_field_pairs=0, windows=0,
        output_rows=0, frame_rows=0, classical_deployment_bidirectional_pairs=0, GT_metric_FB_pairs=0)
    write(out / 'config.json', dict(at=stamp(), process=identity(), seed=seed, source_freeze=frozen,
        unique_finals=markers, sampling_seed=9140+seed, steps=list(range(16000, 16004)), arms=ARMS,
        RGB_PSNR_ceiling_dB=120, held_out_draws_not_scenes=True, new_DEV=False, independent_final_opened=False))
    try:
        teacher, source = load_spatial(seed); students = models(teacher, seed)
        for arm, model in students.items():
            model.load_state_dict(torch.load(ROOT / markers[arm]['path'], map_location='cpu', weights_only=True), strict=True)
            model.eval().requires_grad_(False)
        write(out / 'F0.json', source)
        pool = MixedPool(); write(out / 'data.json', pool.info)
        with torch.inference_mode():
            for step in range(16000, 16004):
                low, gt, metadata = pool.batch(9140+seed, step)
                append(out / 'windows.jsonl', dict(data_step=step, windows=metadata))
                f0 = teacher(low.flatten(0, 1)).unflatten(0, (4, 4)); counts['F0_images'] += 16
                prepared = {arm: model.batch(low) for arm, model in students.items()}
                counts['student_images'] += 32; counts['student_field_pairs'] += 24
                for index in range(4):
                    key = f'seed{seed}-step{step}-window{index}'
                    domain = 'SYN' if index < 2 else 'ND'
                    values = dict(F0=as_image(f0[index])); ranges = dict(F0=f0[index])
                    for arm, (raw, descriptor, fields) in prepared.items():
                        current = raw[index].clamp(0, 1)
                        causal, zero = [current[:1]], [current[:1]]
                        for t in range(1, 4):
                            position = index*3+t-1
                            field, logit = (v[position:position+1] for v in fields)
                            causal.append(mix(current[t:t+1], causal[-1], (field, logit)).clamp(0, 1))
                            height, width = current.shape[-2:]
                            y = torch.arange(height)[:, None]+field[0, 1]
                            x = torch.arange(width)[None, :]+field[0, 0]
                            accepted = ((x >= 0) & (x <= width-1) & (y >= 0) & (y <= height-1))[None, None]
                            alpha = .5*logit.sigmoid()*accepted.to(current.dtype)
                            zero.append((current[t:t+1]+alpha*(zero[-1]-current[t:t+1])).clamp(0, 1))
                        values[arm+'_direct'] = as_image(current)
                        values[arm+'_causal'] = as_image(torch.cat(causal))
                        values[arm+'_zero'] = as_image(torch.cat(zero))
                        ranges[arm+'_direct'] = raw[index]
                    for direct, prefix in [('F0', 'F0'), ('coordinate_direct', 'coordinate'), ('response_direct', 'response')]:
                        for suffix, estimator in [('FB', half_forward_backward), ('DIS', FixedDIS())]:
                            pipeline = RecursivePipeline(estimator)
                            values[prefix+'_'+suffix] = np.stack([pipeline.step(frame)[0] for frame in values[direct]])
                            counts['classical_deployment_bidirectional_pairs'] += 3
                    target = gt[index].numpy().transpose(0, 2, 3, 1).copy()*255
                    reference = [half_forward_backward(target[t-1], target[t]) for t in range(1, 4)]
                    counts['GT_metric_FB_pairs'] += 3
                    path = cache / (key+'.npz')
                    np.savez_compressed(path, gt=target, low=low[index].numpy().transpose(0, 2, 3, 1)*255, **values)
                    append(out / 'prediction_manifest.jsonl', dict(key=key, seed=seed, domain=domain,
                        path=str(path.relative_to(ROOT)), sha256=digest(path)))
                    append(out / 'raw-ranges.jsonl', dict(key=key,
                        values={arm: dict(min=float(v.min()), max=float(v.max()),
                            outside_fraction=float(((v < 0) | (v > 1)).float().mean())) for arm, v in ranges.items()}))
                    for arm in ARMS:
                        output = values[arm]
                        assert np.isfinite(output).all()
                        first = 'F0' if arm.startswith('F0') else arm.split('_')[0]+'_direct'
                        assert np.array_equal(output[0], values[first][0])
                        metrics, frames = evaluate(output, target, [p[0] for p in reference], [p[1] for p in reference])
                        assert all(r['temporal_error'] is not None for r in frames[1:]), 'No empty-domain omissions'
                        for frame in frames:
                            frame['PSNR_ceiling_applied'] = frame['psnr_rgb'] >= 120
                            frame['psnr_rgb'] = min(frame['psnr_rgb'], 120.)
                        metrics['psnr_rgb'] = float(np.mean([r['psnr_rgb'] for r in frames]))
                        assert all(np.isfinite(v) for v in metrics.values())
                        append(out / 'per_sequence.jsonl', dict(key=key, seed=seed, domain=domain, arm=arm, **metrics))
                        for frame in frames: append(out / 'per_frame.jsonl', dict(key=key, seed=seed, domain=domain, arm=arm, **frame))
                        counts['output_rows'] += 1; counts['frame_rows'] += 4
                    counts['windows'] += 1
                print(dict(seed=seed, data_step=step, **counts), flush=True)
        assert counts == dict(F0_images=64, student_images=128, student_field_pairs=96, windows=16,
            output_rows=208, frame_rows=832, classical_deployment_bidirectional_pairs=288, GT_metric_FB_pairs=48)
        verify('training-source-freeze001.json'); assert not torch.cuda.is_initialized(); complete = True
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out / 'runtime.json', dict(at=stamp(), process=identity(), completed_normally=complete,
            seconds=time.perf_counter()-began, CUDA_initialized=torch.cuda.is_initialized(), **counts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--seed', type=int, choices=(0, 1), required=True)
    main(parser.parse_args().seed)
