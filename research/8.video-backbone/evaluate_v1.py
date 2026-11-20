"""All unique-final topic8 DEV outputs and their own strong classical posts."""
import os
if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise SystemExit('Full native quality evaluation is a CPU entry')

import argparse
import json
import time
import traceback

import numpy as np
import torch

from runtime_io import ROOT, BASE, PROCESS, owned_cache, precision, verify, write, append, stamp, digest
from causal_pipeline import CausalEnhancer
from causal_reference_evaluation_r75 import metas
from prepare_cache_r5 import load_one
from exploration_r2_data import evaluate
from fixed_dis_r86 import FixedDIS, RecursivePipeline
from half_motion_r4 import half_forward_backward


def final_markers(seed):
    attempts = sorted(PROCESS.glob(f'train-seed{seed}-attempt*/runtime.json'))
    if not attempts or not json.loads(attempts[-1].read_text())['completed_normally']:
        raise RuntimeError('Wait for this seed entire fixed training budget')
    runtime = json.loads(attempts[-1].read_text())
    assert runtime['counts']['updates'] == dict(warm=2000, spatial=4000, fixed=4000, learned=4000)
    result = {}
    for route in ('spatial', 'fixed', 'learned'):
        paths = sorted(PROCESS.glob(f'train-seed{seed}-attempt*/{route}-final.json'))
        value = json.loads(paths[-1].read_text())
        path = ROOT / value['path']
        assert value['seed'] == seed and value['updates'] == 4000 and value['unique_final']
        assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
        assert digest(path) == value['sha256']
        result[route] = value
    return result


def run(model, low, counters):
    output, preview, state = np.empty_like(low), np.empty_like(low), None
    for index, frame in enumerate(low):
        image = torch.from_numpy(frame.transpose(2, 0, 1).copy()[None]).float() / 255
        value, state, current, details = model(image, state)
        output[index] = value[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255
        preview[index] = current[0].clamp(0, 1).permute(1, 2, 0).numpy() * 255
        state = None if state is None else state.detach()
        counters['backbone_images'] += details['backbone_images']
        counters['causal_DIS_pairs'] += details['DIS_pairs']
    assert np.array_equal(output[0], preview[0])
    return output, preview


def post(raw, method, counters):
    stream = RecursivePipeline(FixedDIS() if method == 'DIS' else half_forward_backward)
    result = np.empty_like(raw)
    for index, frame in enumerate(raw):
        result[index], fields = stream.step(frame)
        if fields is not None:
            counters[f'post_{method}_pairs'] += 1
    assert np.array_equal(raw[0], result[0])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    precision()
    frozen = verify()
    markers = final_markers(args.seed)
    scenes, indices = metas()
    out = BASE / f'results/v1-quality-seed{args.seed}'
    out.mkdir(parents=True, exist_ok=False)
    assets = owned_cache(out.name)
    assets.mkdir(parents=True, exist_ok=False)
    counts = dict(quality_rows=0, frame_rows=0, backbone_images=0, causal_DIS_pairs=0,
                  post_DIS_pairs=0, post_FB_pairs=0)
    started = time.perf_counter()
    complete = False
    write(out / 'config.json', dict(at=stamp(), pid=os.getpid(), seed=args.seed, source_freeze=frozen,
          checkpoints=markers, indices=indices, expected_quality_rows=110, expected_frame_rows=4928,
          expected_backbone_images=2220, official_nominal_fps=30, new_optimizer_updates=0))
    try:
        for route in ('spatial', 'fixed', 'learned'):
            model = CausalEnhancer(route).eval().requires_grad_(False)
            model.load_state_dict(torch.load(ROOT / markers[route]['path'], map_location='cpu', weights_only=True), strict=True)
            for meta in scenes:
                data = load_one(meta)
                with torch.inference_mode():
                    causal, preview = run(model, data['low'].astype(np.float32), counts)
                base_label = 'spatial_raw' if route == 'spatial' else f'{route}_preview'
                variants = {base_label: preview}
                if route != 'spatial':
                    variants[f'{route}_causal'] = causal
                for method in ('FB', 'DIS'):
                    variants[f'{base_label}_{method}05'] = post(preview, method, counts)
                for label, values in variants.items():
                    assert values.dtype == np.float32 and np.isfinite(values).all()
                    metrics, details = evaluate(values, data['gt'], data['reference_flow'], data['reference_mask'])
                    identity = dict(seed=args.seed, domain=meta['domain'], scene=meta['scene'],
                                    source_video=meta['source_video'], label='topic8_' + label,
                                    parameters=markers[route]['parameters'])
                    path = assets / f'{meta["scene"]}--{label}.npy'
                    np.save(path, values, allow_pickle=False)
                    append(out / 'prediction_manifest.jsonl', dict(**identity, path=str(path.relative_to(ROOT)),
                           sha256=digest(path), shape=list(values.shape), dtype='float32',
                           checkpoint_sha256=markers[route]['sha256']))
                    append(out / 'per_sequence.jsonl', dict(**identity, **metrics))
                    counts['quality_rows'] += 1
                    for row in details:
                        append(out / 'per_frame.jsonl', dict(**identity, **row))
                        counts['frame_rows'] += 1
                print(json.dumps(dict(route=route, scene=meta['scene'], **counts)), flush=True)
        assert counts == dict(quality_rows=110, frame_rows=4928, backbone_images=2220,
                              causal_DIS_pairs=876, post_DIS_pairs=1314, post_FB_pairs=1314)
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
