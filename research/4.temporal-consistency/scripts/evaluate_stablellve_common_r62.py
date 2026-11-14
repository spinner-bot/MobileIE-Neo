"""Existing official clean StableLLVE weights on current common CPU DEV inputs."""
import json
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from drv_references_r59 import postprocess
from dynamic_training_r34 import BASE, ROOT, digest
from exploration_r2_data import evaluate
from fetch_wenzhou_samples_r52 import write
from prepare_cache_r5 import load_one
from resume_wenzhou_samples_r52 import file_digest
from stablellve_adapter_r15 import MANIFEST, infer, input_tensor, load_model, output_rgb


def append(path, value):
    with path.open('a') as stream: stream.write(json.dumps(value, allow_nan=False)+'\n')


def main():
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    smoke = BASE/'results/exploration-r15-smoke/smoke.json'
    gate = json.loads(smoke.read_text())
    assert gate['all_finite'] and gate['all_shapes_preserved'] and gate['max_abs_normalized'] <= 1e-6
    synthetic, real = BASE/'results/exploration-r30-cache/index.json', BASE/'results/exploration-r10-cache/index.json'
    metas = json.loads(synthetic.read_text())['scenes']
    metas += [{**r, 'domain':'paired_real_ND_DEV', 'source_video':r['scene']}
              for r in json.loads(real.read_text())['scenes'] if r['domain'] == 'native_crop']
    assert len(metas) == 10 and len({(r['domain'], r['scene']) for r in metas}) == 10
    out = BASE/'results/exploration-r62-stablellve-common-attempt001'; out.mkdir(exist_ok=False)
    cache = ROOT/'.cache/experiments/exploration-r62/common/attempt001'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True, exist_ok=False)
    names = ['evaluate_stablellve_common_r62.py', 'stablellve_adapter_r15.py', 'drv_references_r59.py',
             'half_motion_r4.py', 'aligned_raw_history_r11.py', 'exploration_r2_data.py', 'video_metrics.py']
    frozen = {name:digest(BASE/'scripts'/name) for name in names}
    source = json.loads(MANIFEST.read_text())
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), method='StableLLVE official synthetic-clean',
          project_updates=0, claimed_seed=None, future_wait_max=0, parameters=4316259,
          original_manifest=source, manifest_sha256=digest(MANIFEST), old_smoke_sha256=digest(smoke),
          cache_indices={str(p.relative_to(ROOT)):digest(p) for p in [synthetic, real]},
          protocol_sha256=digest(BASE/'docs/exploration-r62-stablellve-common-reference-protocol.md'), source_sha256=frozen,
          CPU_threads=2, OpenCV_threads=1, expected_sequences=20, expected_per_frame_rows=896,
          expected_CNN_sequences=10, expected_CNN_centers=448,
          revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          scope='Published clean checkpoint transfer, not matched training, noisy-paper reproduction, new method or phone/final test.'))
    start, complete, count, neural, centers = time.perf_counter(), False, 0, 0, 0
    try:
        model = load_model(torch.device('cpu')).requires_grad_(False)
        assert sum(p.numel() for p in model.parameters()) == 4316259
        for meta in metas:
            data = load_one(meta)
            begin = time.perf_counter()
            inputs = input_tensor(data['low'], torch.device('cpu'))
            values = infer(model, inputs); assert torch.isfinite(values).all()
            raw = output_rgb(values)
            seconds = time.perf_counter()-begin; neural += 1; centers += len(raw)
            del inputs, values
            identity = dict(domain=meta['domain'], scene=meta['scene'], source_video=meta['source_video'],
                            reference='StableLLVE_official_clean', seed=None, project_updates=0,
                            parameters=4316259, future_wait_max=0)
            append(out/'neural_stage_seconds.jsonl', dict(**identity, seconds=seconds,
                   scope='CPU2 official preparation+prediction+RGB conversion only, excludes post/metrics/IO; no phone/full-pipeline timing.'))
            outputs, statistics = postprocess(raw, control=None)
            assert set(outputs) == {'raw', 'recursive_self05'}
            for method, output in outputs.items():
                assert output.shape == data['low'].shape and output.dtype == np.float32 and np.isfinite(output).all()
                assert np.array_equal(output[0], raw[0])
                scores, frames = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                row_id = dict(**identity, method=method)
                append(out/'per_sequence.jsonl', dict(**row_id, **scores, **statistics))
                for frame in frames: append(out/'per_frame.jsonl', dict(**row_id, **frame))
                path = cache/f"{meta['scene']}--{method}.npy"; np.save(path, output, allow_pickle=False)
                append(out/'prediction_manifest.jsonl', dict(**row_id, path=str(path.relative_to(ROOT)),
                       sha256=file_digest(path), dtype=str(output.dtype), shape=list(output.shape),
                       input_cache=meta['path'], input_cache_sha256=file_digest(ROOT/meta['path']),
                       checkpoint_sha256=source['sha256'], online_RGB_post_history_bytes=0 if method == 'raw' else 2*raw[0].nbytes,
                       note='Classic post additionally requires its flow/workspace; pretrained clean condition differs from adapted baselines.'))
                count += 1
            print(json.dumps(dict(**identity, sequences=count, CNN_sequences=neural, CNN_centers=centers)), flush=True)
            del data, raw, outputs
        assert count == 20 and neural == 10 and centers == 448 and not torch.cuda.is_initialized()
        assert all(digest(BASE/'scripts'/name) == sha for name, sha in frozen.items())
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, sequences=count, CNN_sequences=neural,
              CNN_center_outputs=centers, unique_observations=448, SGD=0, cuda_initialized=torch.cuda.is_initialized(),
              seconds=time.perf_counter()-start, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
