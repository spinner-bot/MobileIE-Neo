"""One admitted GPU seed: all frozen raw references, then release before CPU metrics."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import json
import sys
import time
import traceback

import cv2
import numpy as np
import torch

from gpu_guard import sample
from reds_references_r72 import (BASE, ROOT, PROCESS, FAMILIES, append, check_sources,
    digest, frozen_sources, infer_raw, load_arrays, load_data_index, load_models, stamp, write)
from training_form_r41 import configure_precision


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--seed', type=int, choices=[0, 1], required=True)
    parser.add_argument('--admission-attempt', type=int, required=True); args = parser.parse_args()
    torch.set_num_threads(2); cv2.setNumThreads(1); configure_precision()
    torch.use_deterministic_algorithms(True); assert not torch.cuda.is_initialized()
    models, sources = load_models(args.seed)
    index = load_data_index()
    first = load_arrays(index['scenes'][0]); first_input = np.array(first['low'][0])
    assert first_input.shape == (720, 1280, 3); del first_input, first
    physical = int(os.environ['CUDA_VISIBLE_DEVICES']); assert physical in [4, 5, 6, 7]
    admission = [sample(physical)]; time.sleep(1); admission.append(sample(physical))
    allowed = all(not r['compute_processes'] and r['memory_mib'] <= 64 and r['utilization_percent'] <= 5 for r in admission)
    PROCESS.mkdir(parents=True, exist_ok=True)
    record = PROCESS / f'seed{args.seed}-internal-admission{args.admission_attempt:03d}.json'
    assert not record.exists()
    write(record, dict(at=stamp(), samples=admission, allowed=allowed, CPU_models_loaded=list(models),
                       CPU_data_index_checked=True, first_native_input_prepared=True, CUDA_initialized=False))
    if not allowed:
        raise SystemExit(75)  # Resource-only rejection; no evaluation result directory or CUDA context.
    out = BASE / f'results/exploration-r72-neural-seed{args.seed}'; out.mkdir(exist_ok=False)
    cache = ROOT / f'.cache/experiments/exploration-r72/neural/seed{args.seed}'
    cache.mkdir(parents=True, exist_ok=False)
    frozen = frozen_sources()
    write(out / 'sources.json', sources)
    write(out / 'config.json', dict(started=stamp(), seed=args.seed, device='cuda:0', physical_GPU=physical,
          source_sha256=frozen, input_index_sha256=digest(BASE / 'results/exploration-r72-data001/index.json'),
          expected_CNN_sequences=40, expected_CNN_centers=2560, SGD=0, FP32=True, TF32=False,
          scope='Only frozen final8000 references on synthetic REDS DEV; no new method/phone or unseen-pretraining guarantee.'))
    started = time.perf_counter(); complete = False; sequences = 0; centers = 0
    try:
        device = torch.device('cuda:0')
        models = {name: model.to(device) for name, model in models.items()}
        torch.cuda.reset_peak_memory_stats()
        for row in index['scenes']:
            data = load_arrays(row)
            for family in FAMILIES:
                began = time.perf_counter(); output = infer_raw(family, models[family], data['low'], device)
                seconds = time.perf_counter() - began
                identity = dict(domain=row['domain'], scene=row['scene'], source_video=row['source_video'],
                                photons=row['photons'], family=family, seed=args.seed,
                                parameters=sources[family]['parameters'], project_updates=8000,
                                method='window' if family == 'FastDVDnet' else 'raw',
                                future_wait_max=2 if family == 'FastDVDnet' else 0)
                path = cache / f'{family}--{row["scene"]}.npy'; np.save(path, output, allow_pickle=False)
                append(out / 'prediction_manifest.jsonl', dict(**identity, path=str(path.relative_to(ROOT)),
                      sha256=digest(path), shape=list(output.shape), dtype=str(output.dtype),
                      low_sha256=row['low']['sha256'], frozen_source_sha256=digest(out / 'sources.json')))
                append(out / 'neural_stage_seconds.jsonl', dict(**identity, seconds=seconds,
                      scope='Descriptive streamed GPU/CPU transfer stage, excludes output IO and CPU metrics/post; not handset latency.'))
                sequences += 1; centers += len(output); del output
                print(json.dumps(dict(at=stamp(), **identity, CNN_sequences=sequences, CNN_centers=centers)), flush=True)
            del data
        assert (sequences, centers) == (40, 2560); check_sources(frozen); complete = True
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out / 'runtime.json', dict(completed_normally=complete, seed=args.seed,
              CNN_sequences=sequences, CNN_centers=centers, new_SGD=0,
              peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
              seconds=time.perf_counter() - started, ended=stamp()))


if __name__ == '__main__':
    main()
