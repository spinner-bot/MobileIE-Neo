"""Eighty fixed official perceptual comparisons; bind before accelerator imports."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
from prebind_shared_gpu import require_initial_binding


def main():
    bound = require_initial_binding()
    import cv2
    import numpy as np
    import torch
    from shared_geometry_io_r88 import BASE, PROCESS, verify, lines, write, append, stamp, digest
    from evaluate_cvvdp_references_r66 import cached
    METRIC = BASE/'results/exploration-r88-perceptual001'
    from causal_reference_evaluation_r75 import metas
    from cvvdp_adapter_r66 import load_metric, predict
    from prepare_cache_r5 import load_one
    from noise_role_io_r79 import acquire_gpu
    frozen = verify(json.loads((PROCESS/'evaluation-source-freeze001.json').read_text()))
    entries, manifests = [], {}
    for seed in (0, 1):
        quality = BASE/f'results/exploration-r88-eval-seed{seed}'
        runtime = json.loads((quality/'runtime.json').read_text())
        assert runtime['completed_normally'] and runtime['quality_rows'] == 40 and runtime['frame_rows'] == 1792
        path = quality/'prediction_manifest.jsonl'
        entries += lines(path); manifests[str(path.relative_to(ROOT))] = digest(path)
    assert len(entries) == len({(r['seed'], r['scene'], r['label']) for r in entries}) == 80
    torch.set_num_threads(2); cv2.setNumThreads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    scenes, indices = metas(); first = load_one(scenes[0])
    prepared, CPU_source = load_metric('cpu'); assert not torch.cuda.is_initialized()
    METRIC.mkdir(exist_ok=False)
    complete, count, frames, schedule, lock = False, 0, 0, None, None
    began = time.perf_counter()
    try:
        lock, schedule, selected = acquire_gpu(bound, METRIC)
        metric, source = load_metric('cuda'); del prepared
        assert source['package_sources'] == CPU_source['package_sources'] and source['playback_fps'] == 30
        write(METRIC/'metric-source.json', source)
        write(METRIC/'config.json', dict(at=stamp(), pid=os.getpid(), expected_rows=80, expected_frames=3584,
            source_manifests_sha256=manifests, indices=indices,
            real_CPU_metric_prepared=True, selected=selected, expected_process_peak_MiB=4096,
            prior_same_shapes_peak_allocated_bytes=360763392,
            scope='Official0.5.7 nominal standard_fhd/30fps/replicate, no enhancer/SGD/phone/MOS.'))
        torch.cuda.reset_peak_memory_stats()
        for ordinal, meta in enumerate(scenes):
            data = first if ordinal == 0 else load_one(meta)
            target = data['gt'].astype(np.float32)/255
            chosen = [r for r in entries if r['scene'] == meta['scene']]; assert len(chosen) == 8
            for entry in chosen:
                schedule.check_memory()
                values = cached(entry); test = np.asarray(values, dtype=np.float32)/255
                assert test.shape == target.shape
                torch.cuda.synchronize(); started = time.perf_counter()
                score, _ = predict(metric, test, target)
                torch.cuda.synchronize(); work = time.perf_counter()-started
                identity = {k: entry[k] for k in ('seed', 'scene', 'domain', 'source_video', 'label')}
                append(METRIC/'per_sequence.jsonl', dict(**identity, JOD=score, frames=len(values)))
                count += 1; frames += len(values)
                if count < 80:
                    schedule.yield_after_work(work)
                else:
                    schedule.work_seconds += work
                append(METRIC/'resource-observations.jsonl', dict(completed_rows=count, **schedule.summary()))
                print(json.dumps(dict(**identity, completed_rows=count)), flush=True)
                del values, test
            del data, target
        assert (count, frames) == (80, 3584)
        verify(frozen)
        complete = True
    except BaseException:
        (METRIC/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(METRIC/'runtime.json', dict(ended=stamp(), pid=os.getpid(), completed_normally=complete,
            quality_rows=count, comparison_frames=frames, new_enhancement_CNN=0, new_SGD=0,
            seconds=time.perf_counter()-began, resource=schedule.summary() if schedule else None,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None))
        globals()['_held_metric_lock'] = lock


if __name__ == '__main__':
    main()
