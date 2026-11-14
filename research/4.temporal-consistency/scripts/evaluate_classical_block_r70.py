"""Fixed first-frame sigma, two original-core modes, both frozen MobileIE seeds."""
from datetime import datetime
import json
import time
import traceback

import cv2
import numpy as np
import torch

from classical_block_r70 import MODES, filter_video, first_frame_sigma, load_tool
from drv_references_r59 import postprocess
from dynamic_training_r34 import BASE, ROOT, digest
from evaluate_spatial_l1_r69 import append
from exploration_r2_data import evaluate
from prepare_cache_r5 import load_one
from response_cue_r45 import load_frozen
from prepare_vbm3d_r70 import write


def main():
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    gate_path = BASE/'process/exploration-r70/cpu-gate001/gate.json'
    assert json.loads(gate_path.read_text())['passed']
    tool = load_tool()
    out = BASE/'results/exploration-r70-classical-attempt001'; out.mkdir(exist_ok=False)
    cache = ROOT/'.cache/experiments/exploration-r70/classical/attempt001'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo'); cache.mkdir(parents=True, exist_ok=False)
    names = ['evaluate_classical_block_r70.py', 'classical_block_r70.py', 'prepare_vbm3d_r70.py',
             'vbm3d_array_r70.cpp', 'prepare_cache_r5.py', 'exploration_r2_data.py', 'video_metrics.py',
             'response_cue_r45.py', 'drv_references_r59.py', 'half_motion_r4.py', 'aligned_raw_history_r11.py']
    frozen = {name: digest(BASE/'scripts'/name) for name in names}
    models, markers = {}, {}
    for seed in [0, 1]:
        runtime = json.loads((BASE/f'results/exploration-r44-stage001-seed{seed}/runtime.json').read_text())
        assert runtime['completed_normally'] and runtime['model_updates']['square_mse'] == 4000
        models[seed], markers[seed] = load_frozen(seed)
        write(out/f'seed{seed}-source.json', dict(checkpoint=markers[seed], parameters=4075, project_updates=8000))
    indices = [BASE/f'results/{name}/index.json' for name in ['exploration-r30-cache', 'exploration-r10-cache']]
    metas = json.loads(indices[0].read_text())['scenes']
    metas += [{**r, 'domain': 'paired_real_ND_DEV', 'source_video': r['scene']}
              for r in json.loads(indices[1].read_text())['scenes'] if r['domain'] == 'native_crop']
    assert len(metas) == 10
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), CPU_Torch=2, OpenCV=1, native_OpenMP=2,
          modes=MODES, expected_filter_sequences=20, expected_filter_frames=896, expected_quality_rows=80,
          expected_frame_rows=3584, expected_CNN_sequences=40, expected_CNN_centers=1792, source_sha256=frozen,
          protocol_sha256=digest(BASE/'docs/exploration-r70-classical-block-reference-protocol.md'),
          gate_sha256=digest(gate_path), binary_sha256=tool['binary_sha256'],
          cache_indices={str(p.relative_to(ROOT)): digest(p) for p in indices}, checkpoints=markers,
          scope='Fixed original-core classical pipeline adaptation; offline video, blind approximate sigma, reused DEV. No training/phone/new method or full AWGN-paper reproduction.'))
    start, complete, count, frame_rows, neural, centers, filters, filter_frames, native_calls = time.perf_counter(), False, 0, 0, 0, 0, 0, 0, 0
    try:
        for meta in metas:
            data = load_one(meta); low = data['low'].astype(np.float32)
            sigma, sigma_info = first_frame_sigma(low)
            append(out/'noise_estimates.jsonl', dict(domain=meta['domain'], scene=meta['scene'], sigma_8bit=sigma, **sigma_info))
            for mode in MODES:
                work = cache/f'{mode}--{meta["scene"]}'
                native, filtering = filter_video(low, sigma, mode, work, tool)
                filtered = np.clip(native, 0, 255).astype(np.float32)
                filter_path = work/'filtered.npy'; np.save(filter_path, filtered, allow_pickle=False)
                filters += 1; filter_frames += len(low); native_calls += not filtering['bypassed_zero_sigma']
                filter_source = dict(domain=meta['domain'], scene=meta['scene'], mode=mode, sigma_8bit=sigma,
                        path=str(filter_path.relative_to(ROOT)), sha256=digest(filter_path), shape=list(filtered.shape),
                        filtering=filtering, input_cache=meta['path'], input_cache_sha256=digest(ROOT/meta['path']),
                        preclip_outside_fraction=float(np.mean((native < 0) | (native > 255))))
                append(out/'filtered_manifest.jsonl', filter_source)
                # Generated input scratch duplicates the immutable cache; retain original-core output and receipt.
                scratch = work/'input-tchw.f32'
                if scratch.exists(): scratch.unlink()
                del native
                for seed, model in models.items():
                    began = time.perf_counter()
                    inputs = torch.from_numpy((filtered/255).transpose(0, 3, 1, 2).copy())
                    with torch.inference_mode():
                        value = torch.stack([model(image[None])[0] for image in inputs])
                        assert torch.isfinite(value).all()
                        raw = value.clamp(0, 1).permute(0, 2, 3, 1).contiguous().numpy()*255
                    neural_seconds = time.perf_counter()-began; neural += 1; centers += len(raw)
                    began = time.perf_counter(); outputs, statistics = postprocess(raw, control=None)
                    post_seconds = time.perf_counter()-began
                    identity = dict(domain=meta['domain'], scene=meta['scene'], source_video=meta['source_video'],
                            family='MobileIE_'+mode, mode=mode, seed=seed, parameters=4075, project_updates=8000,
                            future_wait_max=0 if mode == 'spatial_only' else None,
                            time_context='first-frame sigma plus current spatial patches' if mode == 'spatial_only' else 'offline full clip, exact dependency bound unverified')
                    append(out/'stage_seconds.jsonl', dict(**identity, neural_seconds=neural_seconds,
                           classic_post_seconds=post_seconds, shared_input_filtering=filtering,
                           scope='Descriptive CPU2 stages, filtering reused across seeds; excludes quality/IO not explicitly included. Not phone or uncontented latency benchmark.'))
                    for method, output in outputs.items():
                        assert output.shape == low.shape and np.array_equal(output[0], raw[0])
                        scores, frames = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                        row_id = dict(**identity, method=method)
                        append(out/'per_sequence.jsonl', dict(**row_id, **scores, **statistics))
                        for frame in frames: append(out/'per_frame.jsonl', dict(**row_id, **frame))
                        path = work/f'seed{seed}--{method}.npy'; np.save(path, output, allow_pickle=False)
                        append(out/'prediction_manifest.jsonl', dict(**row_id, path=str(path.relative_to(ROOT)),
                               sha256=digest(path), shape=list(output.shape), dtype=str(output.dtype),
                               input_cache=meta['path'], input_cache_sha256=digest(ROOT/meta['path']),
                               filtered_input_sha256=filter_source['sha256'], checkpoint_sha256=markers[seed]['sha256']))
                        count += 1; frame_rows += len(frames)
                    print(json.dumps(dict(**identity, quality_rows=count, filter_sequences=filters,
                                          CNN_sequences=neural, CNN_centers=centers)), flush=True)
                    del inputs, value, raw, outputs, output
                del filtered
            del data, low
        assert (filters, filter_frames, count, frame_rows, neural, centers) == (20, 896, 80, 3584, 40, 1792)
        assert not torch.cuda.is_initialized()
        assert all(digest(BASE/'scripts'/name) == sha for name, sha in frozen.items()); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, filter_sequences=filters, filter_frames=filter_frames,
              native_filter_calls=native_calls, quality_rows=count, frame_rows=frame_rows, CNN_sequences=neural,
              CNN_centers=centers, new_SGD=0, CUDA_initialized=torch.cuda.is_initialized(),
              seconds=time.perf_counter()-start, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
