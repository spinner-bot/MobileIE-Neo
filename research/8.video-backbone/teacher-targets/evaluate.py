"""Unique-final same-F0 students and their own classical post references."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Quality evaluation is CPU-only'

import argparse
import time
import traceback
import numpy as np
import torch
from study_io import ROOT, BASE, PROCESS, ARMS, read, write, append, stamp, digest, verify, owned_cache, precision
from motion_transport_r74 import load_spatial
from causal_reference_evaluation_r75 import metas
from prepare_cache_r5 import load_one
from exploration_r2_data import evaluate
from fixed_dis_r86 import FixedDIS, RecursivePipeline
from half_motion_r4 import half_forward_backward


def final_markers(seed):
    attempts = sorted(PROCESS.glob(f'train-seed{seed}-attempt*/runtime.json'))
    assert attempts, 'Full training must finish first'
    runtime = read(attempts[-1])
    assert runtime['completed_normally'] and not runtime['qualification']
    assert runtime['counts'] == dict(common_steps=4000, optimizer_updates=12000,
                                    teacher_images=64000, student_images=192000, DIS_pairs=48000)
    assert read(attempts[-1].parent / 'config.json')['source_freeze'] == verify()
    result = {}
    for arm in ARMS:
        value = read(attempts[-1].parent / f'{arm}-final.json')
        path = ROOT / value['path']
        assert (value['seed'], value['arm'], value['updates'], value['parameters']) == (seed, arm, 4000, 4075)
        assert value['unique_final'] and not value['qualification'] and value['all_reloaded_finite']
        assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
        assert digest(path) == value['sha256']
        result[arm] = value
    return result


def run(model, low, counts):
    result = np.empty_like(low)
    for index, frame in enumerate(low):
        image = torch.from_numpy(frame.transpose(2, 0, 1).copy()[None]).float() / 255
        value = model(image)
        result[index] = value[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255
        counts['backbone_images'] += 1
    return result


def post(raw, method, counts):
    stream = RecursivePipeline(FixedDIS() if method == 'DIS' else half_forward_backward)
    result = np.empty_like(raw)
    for index, frame in enumerate(raw):
        result[index], fields = stream.step(frame)
        if fields is not None:
            counts[f'post_{method}_pairs'] += 1
    assert np.array_equal(raw[0], result[0])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    precision()
    os.nice(5)
    frozen, markers = verify(), final_markers(args.seed)
    scenes, indices = metas()
    out = BASE / f'results/quality-seed{args.seed}'
    out.mkdir(parents=True, exist_ok=False)
    assets = owned_cache(out.name)
    assets.mkdir(parents=True, exist_ok=False)
    counts = dict(quality_rows=0, frame_rows=0, backbone_images=0, post_DIS_pairs=0, post_FB_pairs=0)
    started, complete = time.perf_counter(), False
    write(out / 'config.json', dict(at=stamp(), pid=os.getpid(), seed=args.seed, source_freeze=frozen,
          checkpoints=markers, indices=indices, expected_quality_rows=90, expected_frame_rows=4032,
          expected_backbone_images=1344, official_nominal_fps=30, new_optimizer_updates=0,
          raw_inference_uses_teacher_flow_or_history=False))
    try:
        for arm in ARMS:
            model, _ = load_spatial(args.seed)
            model.load_state_dict(torch.load(ROOT / markers[arm]['path'], map_location='cpu', weights_only=True), strict=True)
            model.eval().requires_grad_(False)
            for meta in scenes:
                data = load_one(meta)
                with torch.inference_mode():
                    raw = run(model, data['low'].astype(np.float32), counts)
                variants = {f't1_{arm}_raw': raw}
                for method in ('FB', 'DIS'):
                    variants[f't1_{arm}_{method}05'] = post(raw, method, counts)
                for label, values in variants.items():
                    assert values.dtype == np.float32 and np.isfinite(values).all()
                    metrics, details = evaluate(values, data['gt'], data['reference_flow'], data['reference_mask'])
                    identity = dict(seed=args.seed, domain=meta['domain'], scene=meta['scene'],
                                    source_video=meta['source_video'], label=label, parameters=4075)
                    path = assets / f'{meta["scene"]}--{label}.npy'
                    np.save(path, values, allow_pickle=False)
                    append(out / 'prediction_manifest.jsonl', dict(**identity, path=str(path.relative_to(ROOT)),
                           sha256=digest(path), shape=list(values.shape), dtype='float32',
                           checkpoint_sha256=markers[arm]['sha256']))
                    append(out / 'per_sequence.jsonl', dict(**identity, **metrics))
                    counts['quality_rows'] += 1
                    for row in details:
                        append(out / 'per_frame.jsonl', dict(**identity, **row))
                        counts['frame_rows'] += 1
                print(dict(arm=arm, scene=meta['scene'], **counts), flush=True)
        assert counts == dict(quality_rows=90, frame_rows=4032, backbone_images=1344,
                              post_DIS_pairs=1314, post_FB_pairs=1314)
        verify()
        complete = True
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out / 'runtime.json', dict(at=stamp(), pid=os.getpid(), completed_normally=complete,
              seconds=time.perf_counter() - started, **counts, CUDA_initialized=torch.cuda.is_initialized()))


if __name__ == '__main__':
    main()
