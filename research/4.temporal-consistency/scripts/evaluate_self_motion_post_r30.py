"""Measure the classical post baseline using its own enhancement for motion."""
import json
from pathlib import Path
import subprocess
import time
import traceback

import cv2
import numpy as np
import torch

from aggregation_controls_r20 import blend
from aligned_raw_history_r11 import align_history
from dynamic_training_r34 import BASE, ROOT, digest
from exploration_r2_data import evaluate
from half_motion_r4 import half_forward_backward
from transport_evaluation_r28 import load_models


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    out = BASE/'results/exploration-r30-self-motion'
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    all_nets, markers = load_models('copy', torch.device('cpu'))
    nets = {seed: net.backbone for mode, objective, seed, net in all_nets if mode == 'copy' and objective == 'mse'}
    marker = [m for m in markers if m['mode'] == 'copy' and m['objective'] == 'mse']
    assert set(nets) == {0, 1}
    source = BASE/'results/exploration-r30-eval-references'
    references = [json.loads(line) for line in (source/'per_sequence.jsonl').read_text().splitlines()]
    references = {(r['scene'], r['train_seed'], r['fusion']): r for r in references
                  if r['mode'] == 'copy' and r['objective'] == 'mse' and not r['input_reset']}
    predictions = [json.loads(line) for line in (source/'prediction_manifest.jsonl').read_text().splitlines()]
    predictions = {(r['scene'], r['train_seed']): r for r in predictions if r['objective'] == 'mse'}
    index = BASE/'results/exploration-r30-cache/index.json'
    metas = json.loads(index.read_text())['scenes']
    write(out/'config.json', {'device': 'cpu', 'torch_threads': 2, 'opencv_threads': 1, 'torch_version': torch.__version__,
          'alpha': .5, 'motion': 'half-resolution Farneback/FB on own clipped enhancement', 'future_frames': 0,
          'extra_teacher_parameters': 0, 'predictor_parameters': 4075, 'checkpoints': marker,
          'protocol_sha256': digest(BASE/'docs/exploration-r30-self-motion-baseline-protocol.md'),
          'cache_index_sha256': digest(index), 'reference_rows_sha256': digest(source/'per_sequence.jsonl'),
          'prediction_manifest_sha256': digest(source/'prediction_manifest.jsonl'),
          'source_sha256': {name: digest(Path(__file__).parent/name) for name in
                           ['evaluate_self_motion_post_r30.py', 'half_motion_r4.py', 'aligned_raw_history_r11.py', 'aggregation_controls_r20.py']},
          'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'scope': 'CPU software pipeline with resident input and three warmups; loop includes retaining output references, excludes camera/display, IO, metrics and file saving; not phone latency or full peak-memory evidence.'})
    count, complete, started = 0, False, time.perf_counter()
    shared = []
    try:
        for meta in metas:
            assert digest(ROOT/meta['path']) == meta['sha256']
            with np.load(ROOT/meta['path'], allow_pickle=False) as archive:
                low, gt, reference_flow, reference_mask = [archive[k] for k in ['low', 'gt', 'reference_flow', 'reference_mask']]
            for seed in [0, 1]:
                net = nets[seed]
                identity = {'scene': meta['scene'], 'source_video': meta['source_video'], 'train_seed': seed, 'objective': 'mse'}
                with torch.inference_mode():
                    warmup = torch.from_numpy((low[0]/255).transpose(2, 0, 1).copy())[None]
                    for _ in range(3):
                        net(warmup)
                    raws, posts, times, previous = [], [], [], None
                    total_start = time.perf_counter()
                    for frame in low:
                        begin = time.perf_counter()
                        image = torch.from_numpy((frame/255).transpose(2, 0, 1).copy())[None]
                        value = net(image)[0].clamp(0, 1)
                        raw = np.ascontiguousarray(value.permute(1, 2, 0).numpy()*255)
                        cnn_end = time.perf_counter()
                        if previous is None:
                            fused = raw
                            motion_end = cnn_end
                        else:
                            flow, mask = half_forward_backward(previous, raw)
                            motion_end = time.perf_counter()
                            aligned, _ = align_history(previous, raw, flow, mask)
                            fused = blend(raw, aligned, .5)
                        # Standalone online history; offline outputs below are evaluation only.
                        previous = raw.copy()
                        assert previous.flags.owndata and previous.base is None
                        end = time.perf_counter()
                        times.append({'cnn_and_layout_seconds': cnn_end-begin, 'motion_seconds': motion_end-cnn_end,
                                      'warp_blend_history_seconds': end-motion_end, 'pipeline_seconds': end-begin})
                        raws.append(raw)
                        posts.append(fused)
                    total_seconds = time.perf_counter()-total_start
                outputs = {'raw': np.stack(raws), 'self_motion_post05': np.stack(posts)}
                cached = predictions[(meta['scene'], seed)]
                assert digest(ROOT/cached['path']) == cached['sha256']
                old_raw = np.load(ROOT/cached['path'], allow_pickle=False, mmap_mode='r')
                maximum = float(np.abs(outputs['raw']-old_raw).max())
                assert maximum <= 1e-4
                for mode, output in outputs.items():
                    assert np.isfinite(output).all()
                    metrics, frames = evaluate(output, gt, reference_flow, reference_mask)
                    if mode == 'raw':
                        old = references[(meta['scene'], seed, 'raw')]
                        deltas = {k: abs(metrics[k]-old[k]) for k in ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae', 'flicker']}
                        assert max(deltas.values()) <= 1e-6
                        with (out/'raw_replay.jsonl').open('a') as stream:
                            stream.write(json.dumps({**identity, 'max_abs_pixels_0to255': maximum, 'metric_deltas': deltas})+'\n')
                    with (out/'per_sequence.jsonl').open('a') as stream:
                        stream.write(json.dumps({**identity, 'mode': mode, **metrics}, allow_nan=False)+'\n')
                    with (out/'per_frame.jsonl').open('a') as stream:
                        for frame in frames:
                            stream.write(json.dumps({**identity, 'mode': mode, **frame}, allow_nan=False)+'\n')
                    count += 1
                with (out/'pipeline_cost.jsonl').open('a') as stream:
                    stream.write(json.dumps({**identity, 'frames': len(low), 'pipeline_seconds': total_seconds,
                         'online_history_storage_bytes': previous.nbytes, 'per_frame': times}, allow_nan=False)+'\n')
                shared.append(references[(meta['scene'], seed, 'postwarp05')])
                print(json.dumps({**identity, 'completed_rows': count, 'pipeline_seconds': total_seconds}), flush=True)
        assert count == 24 and len(shared) == 12 and not torch.cuda.is_initialized()
        write(out/'reused_shared_motion_post.json', shared)
        complete = True
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'rows': count, 'cnn_sequences': count//2,
                                  'cuda_initialized': torch.cuda.is_initialized(), 'wall_seconds': time.perf_counter()-started})


if __name__ == '__main__':
    main()
