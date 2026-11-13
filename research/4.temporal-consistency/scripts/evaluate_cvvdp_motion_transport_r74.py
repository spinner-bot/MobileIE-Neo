"""All80 new cached R74 comparisons, original fixed ColorVideoVDP settings."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from datetime import datetime
import json
import time
import traceback

import cv2
import numpy as np
import torch

from motion_transport_r74 import BASE, ROOT, PROTOCOL
from check_motion_transport_r74 import write
from dynamic_training_r34 import digest
from evaluate_motion_transport_r74 import metas, key
from evaluate_cvvdp_references_r66 import cached, lines, append
from cvvdp_adapter_r66 import load_metric, predict, qualify
from prepare_cache_r5 import load_one
from training_form_r41 import configure_precision, cuda_headroom


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--admission-attempt', type=int, required=True)
    args = parser.parse_args(); cv2.setNumThreads(1); torch.set_num_threads(2)
    configure_precision(); torch.use_deterministic_algorithms(True); assert not torch.cuda.is_initialized()
    quality = BASE/'report/exploration-r74-quality/audit.json'; audit = json.loads(quality.read_text())
    assert audit['completed'] and audit['new_quality_rows'] == 80 and audit['all_quality_rows'] == 120
    entries, sources = [], {str(quality.relative_to(ROOT)): digest(quality)}
    for seed in [0, 1]:
        path = BASE/f'results/exploration-r74-eval-seed{seed}/prediction_manifest.jsonl'
        selected = lines(path); assert len(selected) == 40
        entries += selected; sources[str(path.relative_to(ROOT))] = digest(path)
    assert len(entries) == len({key(r) for r in entries}) == 80
    scenes, _ = metas(); first = load_one(scenes[0])
    assert cached(entries[0]).shape == first['gt'].shape
    gate = json.loads((BASE/'process/exploration-r66/cpu-gate001/gate.json').read_text()); assert gate['passed']
    prepared_metric, prepared_source = load_metric('cpu')
    assert prepared_metric.device.type == 'cpu' and not torch.cuda.is_initialized()
    admission = cuda_headroom()
    write(BASE/f'process/exploration-r74/metric-internal-admission{args.admission_attempt:03d}.json', admission)
    if not admission['allowed']: raise SystemExit(75)
    out = BASE/'results/exploration-r74-perceptual001'; out.mkdir(exist_ok=False)
    cache = ROOT/'.cache/experiments/exploration-r74/perceptual001'; cache.mkdir(parents=True, exist_ok=False)
    started, complete, count, frames = time.perf_counter(), False, 0, 0
    try:
        write(out/'cuda-gate.json', qualify('cuda', gate['scores']))
        metric, source = load_metric('cuda'); assert source['playback_fps'] == 30
        assert source['package_sources'] == prepared_source['package_sources']
        del prepared_metric
        write(out/'metric-source.json', source)
        write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), expected_rows=80,
              expected_comparison_frames=3584, sources=sources, protocol_sha256=digest(PROTOCOL),
              complete_CPU_metric_prepared_before_internal_admission=True,
              scope='Fixed complete cached conditions, nominal30fps standard_fhd replicate; not enhancement CNN, phone, MOS or final.'))
        torch.cuda.reset_peak_memory_stats()
        for ordinal, meta in enumerate(scenes):
            data = first if ordinal == 0 else load_one(meta); target = data['gt'].astype(np.float32)/255
            selected = [r for r in entries if r['scene'] == meta['scene']]; assert len(selected) == 8
            for entry in selected:
                values = cached(entry); assert values.shape == data['gt'].shape
                began = time.perf_counter(); score, stats = predict(metric, np.asarray(values, dtype=np.float32)/255, target)
                identity = {n: entry[n] for n in ['domain', 'scene', 'source_video', 'seed', 'model', 'history', 'method']}
                append(out/'per_sequence.jsonl', dict(**identity, JOD=score, frames=len(values), seconds=time.perf_counter()-began))
                path = cache/f'features-{count:03d}.npz'
                np.savez_compressed(path, Q_per_ch=stats['Q_per_ch'], rho_band=np.asarray(stats['rho_band']))
                append(out/'feature_manifest.jsonl', dict(**identity, path=str(path.relative_to(ROOT)), sha256=digest(path), source=entry))
                count += 1; frames += len(values)
                print(json.dumps(dict(**identity, completed_rows=count)), flush=True)
            del data, target
        assert (count, frames) == (80, 3584); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, quality_rows=count, comparison_frames=frames,
              new_enhancement_CNN=0, new_SGD=0, seconds=time.perf_counter()-started,
              peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
