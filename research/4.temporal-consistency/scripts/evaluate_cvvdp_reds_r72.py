"""Complete cached REDS references, Input and unique GT controls at source24fps."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import json
import time
import traceback

import cv2
import numpy as np
import torch

from gpu_guard import sample
from reds_metric_r72 import load24, predict24, qualify24
from reds_references_r72 import (BASE, ROOT, PROCESS, append, check_sources, digest,
    frozen_sources, lines, load_arrays, load_data_index, stamp, write)
from training_form_r41 import configure_precision


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--admission-attempt', type=int, required=True)
    args = parser.parse_args(); torch.set_num_threads(2); cv2.setNumThreads(1)
    configure_precision(); torch.use_deterministic_algorithms(True); assert not torch.cuda.is_initialized()
    gate = json.loads((PROCESS / 'metric-cpu-gate001/gate.json').read_text()); assert gate['passed']
    index = load_data_index(); predictions = []
    for seed in [0, 1]:
        path = BASE / f'results/exploration-r72-quality-seed{seed}'
        terminal = json.loads((path / 'runtime.json').read_text())
        assert terminal['completed_normally'] and (terminal['quality_rows'], terminal['frame_rows']) == (72, 4608)
        predictions += lines(path / 'prediction_manifest.jsonl')
    assert len(predictions) == 144
    first = load_arrays(index['scenes'][0]); first_target = np.asarray(first['gt'], dtype=np.float32) / 255
    physical = int(os.environ['CUDA_VISIBLE_DEVICES']); assert physical in [4, 5, 6, 7]
    samples = [sample(physical)]; time.sleep(1); samples.append(sample(physical))
    allowed = all(not r['compute_processes'] and r['memory_mib'] <= 64 and r['utilization_percent'] <= 5 for r in samples)
    admission = PROCESS / f'metric-internal-admission{args.admission_attempt:03d}.json'; assert not admission.exists()
    write(admission, dict(at=stamp(), allowed=allowed, samples=samples,
          CPU_predictions_checked=True, first_full_GT_prepared=True, CUDA_initialized=False))
    if not allowed:
        raise SystemExit(75)
    out = BASE / 'results/exploration-r72-perceptual001'; out.mkdir(exist_ok=False)
    cache = ROOT / '.cache/experiments/exploration-r72/perceptual001'; cache.mkdir(parents=True, exist_ok=False)
    frozen = frozen_sources(); start = time.perf_counter(); complete = False; count = 0; frames = 0; gt_done = set()
    try:
        write(out / 'cuda-gate.json', qualify24('cuda', gate['scores']))
        metric, source = load24('cuda'); write(out / 'metric-source.json', source)
        write(out / 'config.json', dict(started=stamp(), source_sha256=frozen,
              expected_rows=156, expected_comparison_frames=9984, source_fps=24,
              unique_low_observations=512, unique_GT_sequences=4, new_enhancement_CNN=0, SGD=0,
              scope='Fixed original display assumptions at REDS24fps; not phone/MOS/independent final.'))
        torch.cuda.reset_peak_memory_stats()
        for ordinal, row in enumerate(index['scenes']):
            data = first if ordinal == 0 else load_arrays(row)
            target = first_target if ordinal == 0 else np.asarray(data['gt'], dtype=np.float32) / 255
            selected = [r for r in predictions if r['scene'] == row['scene']]; assert len(selected) == 18
            tests = [(r, None) for r in selected]
            tests.append((dict(domain=row['domain'], scene=row['scene'], source_video=row['source_video'],
                         photons=row['photons'], family='Input', seed=None, method='input'), data['low']))
            if row['source_video'] not in gt_done:
                tests.append((dict(domain=row['domain'], scene=row['source_video'] + '_GT',
                             source_video=row['source_video'], photons=None, family='GT', seed=None,
                             method='identity'), data['gt']))
            for entry, direct in tests:
                if direct is None:
                    assert digest(ROOT / entry['path']) == entry['sha256']
                    direct = np.load(ROOT / entry['path'], mmap_mode='r', allow_pickle=False)
                identity = {k: entry[k] for k in ['domain', 'scene', 'source_video', 'photons', 'family', 'seed', 'method']}
                began = time.perf_counter()
                score, statistics = predict24(metric, np.asarray(direct, dtype=np.float32) / 255, target)
                if entry['family'] == 'GT':
                    assert abs(score - 10) <= 1e-4; gt_done.add(row['source_video'])
                append(out / 'per_sequence.jsonl', dict(**identity, JOD=score, frames=64, source_fps=24,
                       seconds=time.perf_counter() - began))
                path = cache / f'features-{count:03d}.npz'
                np.savez_compressed(path, Q_per_ch=statistics['Q_per_ch'], rho_band=np.asarray(statistics['rho_band']))
                append(out / 'feature_manifest.jsonl', dict(**identity, path=str(path.relative_to(ROOT)), sha256=digest(path)))
                count += 1; frames += 64
                print(json.dumps(dict(at=stamp(), **identity, rows=count)), flush=True)
            del data, target, tests, direct
        assert (count, frames, len(gt_done)) == (156, 9984, 4)
        check_sources(frozen); complete = True
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out / 'runtime.json', dict(completed_normally=complete, perceptual_rows=count,
              comparison_frames=frames, unique_GT_controls=len(gt_done), source_fps=24,
              new_enhancement_CNN=0, new_SGD=0, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
              seconds=time.perf_counter() - start, ended=stamp()))


if __name__ == '__main__':
    main()
