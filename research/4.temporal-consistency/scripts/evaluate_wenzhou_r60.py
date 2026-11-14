"""All fixed eight references/two outputs on all six preregistered night clips."""
import json
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from drv_references_r59 import NETWORKS, load_network, postprocess
from dynamic_training_r34 import BASE, ROOT, digest
from fastdvdnet_adapter_r12 import window_indices
from fetch_wenzhou_samples_r52 import write
from half_motion_r4 import half_forward_backward
from resume_wenzhou_samples_r52 import file_digest
from wenzhou_references_r60 import PRIMARY, describe, infer_sequence


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')


def main():
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    gate = BASE/'process/exploration-r60/cpu-gate001.json'
    saved_gate = json.loads(gate.read_text()); assert saved_gate['passed']
    assert saved_gate['helper_sha256'] == digest(BASE/'scripts/wenzhou_references_r60.py')
    index_path = BASE/'results/exploration-r60-wenzhou-data-attempt001/index.json'
    assert json.loads((index_path.parent/'runtime.json').read_text())['completed_normally']
    metas = json.loads(index_path.read_text())['scenes']; assert len(metas) == 6
    out = BASE/'results/exploration-r60-wenzhou-reference-attempt001'; out.mkdir(exist_ok=False)
    cache = ROOT/'.cache/experiments/exploration-r60/references/attempt001'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True, exist_ok=False)
    names = ['evaluate_wenzhou_r60.py', 'wenzhou_references_r60.py', 'drv_references_r59.py',
             'half_motion_r4.py', 'aligned_raw_history_r11.py', 'fastdvdnet_adapter_r12.py',
             'multinex_reference_r55.py', 'tiny_adapted_reference_r56.py', 'task_fastdvd_r47.py']
    frozen = {name: digest(BASE/'scripts'/name) for name in names}
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), networks=NETWORKS,
          methods=['raw', 'recursive_self05'], primary_frames=PRIMARY, input_index_sha256=digest(index_path),
          protocol_sha256=digest(BASE/'docs/exploration-r60-wenzhou-frozen-reference-protocol.md'),
          gate_sha256=digest(gate), source_sha256=frozen, CPU_threads=2, OpenCV_threads=1, SGD=0,
          evaluation_flow='Shared original uint8 RGB input half-FB, not GT; post uses its own raw flow.',
          FD_window_indices=window_indices(128, 'window').tolist(),
          expected_sequences=96, expected_per_frame_rows=12288, expected_CNN_sequences=48,
          expected_CNN_centers=6144, unique_observations=768, unique_primary_targets=720,
          revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          scope='Unpaired midpoint DEV excerpts, no new method, quality/flicker truth, phone or final test.'))
    start, complete, count, neural, centers = time.perf_counter(), False, 0, 0, 0
    try:
        networks = {}
        for name in NETWORKS:
            model, unused_control, source = load_network(name)
            source.update(RGB_controller_used=False, inference_context='128 fixed excerpt frames; primary4..123')
            networks[name] = (model, source)
            append(out/'model_sources.jsonl', source)
            del unused_control
        for meta in metas:
            path = ROOT/meta['path']; assert file_digest(path) == meta['sha256']
            with np.load(path, allow_pickle=False) as archive:
                low = archive['low'].copy()
            assert low.shape == (128, 360, 640, 3) and low.dtype == np.uint8
            flows = np.zeros((128, 360, 640, 2), dtype=np.float32)
            masks = np.zeros((128, 360, 640), dtype=bool)
            for t in range(1, 128):
                flows[t], masks[t] = half_forward_backward(low[t-1], low[t])
            for key, array in [('flow', flows), ('mask', masks)]:
                flow_path = cache/f"input-{key}--{meta['scene']}.npy"
                np.save(flow_path, array, allow_pickle=False)
                append(out/'evaluation_flow_manifest.jsonl', dict(scene=meta['scene'], kind=key,
                       path=str(flow_path.relative_to(ROOT)), sha256=file_digest(flow_path), shape=list(array.shape)))
            baseline, input_frames = describe(low.astype(np.float32), flows, masks)
            append(out/'input_per_sequence.jsonl', dict(scene=meta['scene'], **baseline))
            for row in input_frames: append(out/'input_per_frame.jsonl', dict(scene=meta['scene'], **row))
            for name in NETWORKS:
                model, source = networks[name]
                begin = time.perf_counter(); raw = infer_sequence(name, model, low)
                seconds = time.perf_counter()-begin; neural += 1; centers += len(low)
                identity = dict(scene=meta['scene'], reference=name, seed=source['seed'],
                                parameters=source['parameters'], future_wait_max=source['future_wait_max'])
                append(out/'neural_stage_seconds.jsonl', dict(**identity, seconds=seconds,
                       scope='CPU2 CNN stage only; excludes post/metrics/IO, not full production or phone timing.'))
                outputs, stats = postprocess(raw, control=None)
                assert set(outputs) == {'raw', 'recursive_self05'}
                for method, output in outputs.items():
                    summary, frames = describe(output, flows, masks)
                    row_id = dict(**identity, method=method)
                    append(out/'per_sequence.jsonl', dict(**row_id, **summary, **stats))
                    for row in frames: append(out/'per_frame.jsonl', dict(**row_id, **row))
                    dest = cache/f"{name}--{meta['scene']}--{method}.npy"
                    np.save(dest, output, allow_pickle=False)
                    append(out/'prediction_manifest.jsonl', dict(**row_id, path=str(dest.relative_to(ROOT)),
                           sha256=file_digest(dest), bytes=dest.stat().st_size, shape=list(output.shape), dtype=str(output.dtype),
                           input_cache=meta['path'], input_sha256=meta['sha256'],
                           online_RGB_post_history_bytes=0 if method == 'raw' else 2*output[0].nbytes,
                           note='Post history excludes flow workspace; FD also has five-frame/future input buffers.'))
                    count += 1
                print(json.dumps(dict(**identity, sequences=count, CNN_sequences=neural, CNN_centers=centers)), flush=True)
                del raw, outputs
            del low, flows, masks
        assert count == 96 and neural == 48 and centers == 6144 and not torch.cuda.is_initialized()
        assert all(digest(BASE/'scripts'/name) == sha for name, sha in frozen.items())
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, sequences=count, CNN_sequences=neural,
              CNN_center_outputs=centers, unique_observations=768, unique_primary_targets=720,
              SGD=0, cuda_initialized=torch.cuda.is_initialized(), seconds=time.perf_counter()-start,
              ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
