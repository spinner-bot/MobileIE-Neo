"""Frozen references on all existing phone DEV frames; no GT or quality rank."""
import argparse
import json
import subprocess
import time
import traceback
from datetime import datetime

import cv2
import numpy as np
import torch

from aligned_raw_history_r11 import align_history
from dynamic_training_r34 import BASE, ROOT, digest
from evaluate_phone_references_r38 import statistics
from half_motion_r4 import half_forward_backward
from response_cue_r45 import RecursiveControl, load_frozen

METHODS = ['raw', 'recursive_self05', 'rgb']


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    out = BASE/f'results/exploration-r54-full-phone-seed{args.seed}'; out.mkdir(exist_ok=False)
    cache = ROOT/f'.cache/experiments/exploration-r54/full-phone/seed{args.seed}'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True, exist_ok=False)
    start, complete, count, neural, neural_frames = time.perf_counter(), False, 0, 0, 0
    try:
        backbone, anchor = load_frozen(args.seed)
        training = BASE/f'results/exploration-r45-stage001-seed{args.seed}'
        run = json.loads((training/'runtime.json').read_text())
        assert run['completed_normally'] and run['frozen_base_unchanged']
        marker = json.loads((training/'rgb-step4000.json').read_text())
        assert marker['new_step'] == 4000 and marker['base_checkpoint_sha256'] == anchor['sha256']
        assert digest(ROOT/marker['path']) == marker['sha256']
        saved = torch.load(ROOT/marker['path'], map_location='cpu', weights_only=False)
        assert saved['control'] == 'rgb' and saved['base_checkpoint_sha256'] == anchor['sha256']
        control = RecursiveControl('rgb', args.seed).eval().requires_grad_(False)
        control.load_state_dict(saved['state_dict'], strict=True)
        index = BASE/'results/exploration-r49-full-phone-attempt001/index.json'
        metadata = json.loads(index.read_text()); metas = metadata['scenes']
        assert len(metas) == 3 and [m['shape'][0] for m in metas] == [329, 327, 296]
        names = ['evaluate_full_phone_r54.py', 'response_cue_r45.py', 'evaluate_phone_references_r38.py',
                 'half_motion_r4.py', 'aligned_raw_history_r11.py', 'budget_causal_r33.py', 'video_metrics.py']
        write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), seed=args.seed,
              source_index=str(index.relative_to(ROOT)), source_index_sha256=digest(index),
              anchor=anchor, RGB_controller=marker, methods=METHODS,
              protocol_sha256=digest(BASE/'docs/exploration-r54-full-phone-reference-protocol.md'),
              source_sha256={name: digest(BASE/'scripts'/name) for name in names},
              device='cpu', torch_threads=2, opencv_threads=1,
              revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              scope='Same three phone DEV sources, full EOF, original brightness, frozen references; no GT, selection, new method, phone cost or true-flicker rank.'))
        for meta in metas:
            assert digest(ROOT/meta['path']) == meta['sha256']
            with np.load(ROOT/meta['path'], allow_pickle=False) as archive:
                low = archive['low'].copy()
            assert list(low.shape) == meta['shape'] and low.dtype == np.uint8 and low.shape[1:] == (360, 640, 3)
            paths = {m: cache/f"{meta['scene']}--{m}.npy" for m in METHODS}
            arrays = {m: np.lib.format.open_memmap(p, mode='w+', dtype=np.float32, shape=low.shape) for m, p in paths.items()}
            previous = {}; previous_rgb = None
            stats = {m: [] for m in METHODS}
            alpha_sum, alpha_count = 0., 0
            identity = dict(scene=meta['scene'], phone=meta['phone'], train_seed=args.seed)
            with torch.inference_mode():
                for t, frame in enumerate(low):
                    image = torch.from_numpy(frame.transpose(2, 0, 1).copy()).float()[None]/255
                    value = backbone(image)[0]; assert torch.isfinite(value).all()
                    raw = value.clamp(0, 1).permute(1, 2, 0).contiguous().numpy()*255
                    neural_frames += 1
                    current = torch.from_numpy(raw.transpose(2, 0, 1).copy()/255)[None]
                    if t:
                        flow, mask = half_forward_backward(previous['raw'], raw)
                        moved, _ = align_history(previous['recursive_self05'], raw, flow, mask)
                        recursive = raw+.5*(moved-raw)
                        previous_rgb, alpha, valid = control(current, previous_rgb,
                            torch.from_numpy(flow.transpose(2, 0, 1).copy())[None],
                            torch.from_numpy(mask.copy())[None, None].bool(), cues=None)
                        rgb = previous_rgb[0].permute(1, 2, 0).contiguous().numpy()*255
                        alpha_sum += float(alpha[valid].double().sum()); alpha_count += int(valid.sum())
                    else:
                        previous_rgb, recursive, rgb = current, raw, raw
                    outputs = dict(raw=raw, recursive_self05=recursive, rgb=rgb)
                    append(out/'input_statistics.jsonl', {**identity, 'frame': t, **statistics(frame)})
                    append(out/'preclip_statistics.jsonl', {**identity, 'frame': t,
                           'below_zero_fraction': float((value < 0).float().mean()),
                           'above_one_fraction': float((value > 1).float().mean())})
                    for method, output in outputs.items():
                        assert np.isfinite(output).all()
                        if not t: assert np.array_equal(output, raw)
                        arrays[method][t] = output
                        row = statistics(output)
                        row['DC_luma_change'] = row['mean_luma_0to255']-stats[method][-1]['mean_luma_0to255'] if t else None
                        row['raw_flow_valid_fraction'] = None
                        row['shared_raw_flow_warp_luma_MAE_proxy'] = None
                        if t:
                            aligned, valid_np = align_history(previous[method], output, flow, mask)
                            row['raw_flow_valid_fraction'] = float(valid_np.mean())
                            if valid_np.any():
                                delta = cv2.cvtColor((output-aligned).astype(np.float32), cv2.COLOR_RGB2GRAY)
                                row['shared_raw_flow_warp_luma_MAE_proxy'] = float(np.abs(delta)[valid_np].mean())
                        stats[method].append(row)
                        append(out/'per_frame.jsonl', {**identity, 'method': method, 'frame': t, **row})
                    previous = outputs
                    if (t+1) % 50 == 0:
                        print(json.dumps({**identity, 'frame': t+1, 'of': len(low)}), flush=True)
            neural += 1
            for method in METHODS:
                arrays[method].flush()
                values = stats[method]
                means = {k: float(np.mean([r[k] for r in values if r[k] is not None]))
                         if any(r[k] is not None for r in values) else None for k in values[0]}
                append(out/'per_sequence.jsonl', {**identity, 'method': method, 'frames': len(low),
                       'parameters': 4116 if method == 'rgb' else 4075,
                       'online_history_bytes': 0 if method == 'raw' else 2*raw.nbytes,
                       'mean_accepted_alpha': alpha_sum/alpha_count if method == 'rgb' and alpha_count else None,
                       **means, 'scope': 'Unpaired diagnostic statistics, not image quality or true flicker.'})
                append(out/'prediction_manifest.jsonl', {**identity, 'method': method,
                       'path': str(paths[method].relative_to(ROOT)), 'sha256': digest(paths[method]),
                       'bytes': paths[method].stat().st_size, 'shape': list(low.shape), 'dtype': 'float32',
                       'input_sha256': meta['sha256'], 'anchor_sha256': anchor['sha256'],
                       'controller_sha256': marker['sha256'] if method == 'rgb' else None})
                count += 1
            del arrays, low
            print(json.dumps({**identity, 'completed_sequences': count, 'neural_frames': neural_frames}), flush=True)
        assert count == 9 and neural == 3 and neural_frames == 952 and not torch.cuda.is_initialized()
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, rows=count, neural_sequences=neural,
              neural_frames=neural_frames, cuda_initialized=torch.cuda.is_initialized(),
              seconds=time.perf_counter()-start, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__':
    main()
