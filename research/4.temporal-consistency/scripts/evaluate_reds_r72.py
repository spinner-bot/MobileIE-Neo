"""Completed raw cache to full CPU quality and own-output recursive references."""
import argparse
import json
import time
import traceback

import cv2
import numpy as np
import torch

from drv_references_r59 import postprocess
from exploration_r2_data import evaluate
from reds_references_r72 import (BASE, ROOT, DATA, append, check_sources, digest,
    frozen_sources, lines, load_arrays, load_data_index, stamp, write)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    args = parser.parse_args(); torch.set_num_threads(2); cv2.setNumThreads(1)
    assert not torch.cuda.is_initialized()
    neural = BASE / f'results/exploration-r72-neural-seed{args.seed}'
    runtime = json.loads((neural / 'runtime.json').read_text())
    assert runtime['completed_normally'] and (runtime['CNN_sequences'], runtime['CNN_centers']) == (40, 2560)
    index = load_data_index(); metas = {r['scene']: r for r in index['scenes']}
    predictions = lines(neural / 'prediction_manifest.jsonl'); assert len(predictions) == 40
    out = BASE / f'results/exploration-r72-quality-seed{args.seed}'; out.mkdir(exist_ok=False)
    cache = ROOT / f'.cache/experiments/exploration-r72/quality/seed{args.seed}'
    cache.mkdir(parents=True, exist_ok=False)
    frozen = frozen_sources()
    write(out / 'config.json', dict(started=stamp(), seed=args.seed, source_sha256=frozen,
          neural_terminal_sha256=digest(neural / 'runtime.json'),
          neural_manifest_sha256=digest(neural / 'prediction_manifest.jsonl'),
          expected_quality_rows=72, expected_frame_rows=4608, new_CNN=0, new_SGD=0,
          CPU_Torch=2, OpenCV=1, scope='Same public GT-FB evaluation domain; deployment recursion uses only own raw.'))
    started = time.perf_counter(); complete = False; count = 0; frame_count = 0
    try:
        for entry in predictions:
            assert digest(ROOT / entry['path']) == entry['sha256']
            raw = np.load(ROOT / entry['path'], mmap_mode='r', allow_pickle=False)
            assert raw.dtype == np.float32 and raw.shape == (64, 720, 1280, 3) and np.isfinite(raw).all()
            data = load_arrays(metas[entry['scene']]); began = time.perf_counter()
            if entry['family'] == 'FastDVDnet':
                outputs, statistics = {'window': raw}, {}
            else:
                outputs, statistics = postprocess(raw, control=None)
            append(out / 'post_stage_seconds.jsonl', dict(scene=entry['scene'], family=entry['family'],
                   seed=args.seed, seconds=time.perf_counter() - began, scope='CPU2 post only, no handset timing.'))
            identity = {k: entry[k] for k in ['domain', 'scene', 'source_video', 'photons', 'family',
                        'seed', 'parameters', 'project_updates', 'future_wait_max']}
            for method, output in outputs.items():
                assert output.shape == raw.shape and np.isfinite(output).all() and np.array_equal(output[0], raw[0])
                scores, frames = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                row_id = dict(**identity, method=method)
                append(out / 'per_sequence.jsonl', dict(**row_id, **scores, **statistics))
                for frame in frames:
                    append(out / 'per_frame.jsonl', dict(**row_id, **frame))
                if method == 'recursive_self05':
                    path = cache / f'{entry["family"]}--{entry["scene"]}--{method}.npy'
                    np.save(path, output, allow_pickle=False)
                    saved = dict(path=str(path.relative_to(ROOT)), sha256=digest(path))
                else:
                    saved = {k: entry[k] for k in ['path', 'sha256']}
                append(out / 'prediction_manifest.jsonl', dict(**row_id, **saved,
                       shape=list(output.shape), dtype=str(output.dtype), raw_source_sha256=entry['sha256']))
                count += 1; frame_count += len(frames)
            print(json.dumps(dict(at=stamp(), **identity, quality_rows=count)), flush=True)
            del data, raw, outputs, output
        assert (count, frame_count) == (72, 4608) and not torch.cuda.is_initialized()
        check_sources(frozen); complete = True
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out / 'runtime.json', dict(completed_normally=complete, seed=args.seed, quality_rows=count,
              frame_rows=frame_count, new_CNN=0, new_SGD=0, CUDA_initialized=torch.cuda.is_initialized(),
              seconds=time.perf_counter() - started, ended=stamp()))


if __name__ == '__main__':
    main()
