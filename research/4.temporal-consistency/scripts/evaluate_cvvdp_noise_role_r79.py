"""All100 final cached R79 comparisons with the existing official metric and guard."""
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
from noise_role_io_r79 import BASE, digest, write, append, stamp, acquire_gpu


def main():
    bound = require_initial_binding()
    import cv2
    import numpy as np
    import torch
    from noise_role_model_r79 import precision
    from evaluate_noise_role_r79 import final_markers
    from noise_role_evidence_r79 import verify_evaluation
    from causal_reference_evaluation_r75 import metas
    from causal_reference_io_r75 import lines
    from cvvdp_adapter_r66 import load_metric, predict
    from evaluate_cvvdp_references_r66 import cached
    from prepare_cache_r5 import load_one
    cv2.setNumThreads(1); precision(); final_markers(); verify_evaluation()
    entries, sources = [], {}
    for seed in (0, 1):
        root = BASE/f'results/exploration-r79-eval-seed{seed}'
        runtime = json.loads((root/'runtime.json').read_text())
        assert runtime['completed_normally'] and runtime['quality_rows'] == 50 and runtime['frame_rows'] == 2240
        path = root/'prediction_manifest.jsonl'; rows = lines(path)
        assert len(rows) == 50
        entries += rows; sources[str(path.relative_to(ROOT))] = digest(path)
    assert len(entries) == len({(r['seed'], r['scene'], r['label']) for r in entries}) == 100
    scenes, indices = metas(); first = load_one(scenes[0])
    prepared, expected_source = load_metric('cpu')
    assert not torch.cuda.is_initialized()
    out = BASE/'results/exploration-r79-perceptual001'; out.mkdir(exist_ok=False)
    complete, count, frames, schedule, lock = False, 0, 0, None, None
    began = time.perf_counter()
    try:
        lock, schedule, selected = acquire_gpu(bound, out)
        metric, source = load_metric('cuda'); del prepared
        assert source['package_sources'] == expected_source['package_sources'] and source['playback_fps'] == 30
        write(out/'metric-source.json', source)
        write(out/'config.json', dict(at=stamp(), expected_rows=100, expected_frames=4480,
            sources=sources, cache_indices=indices, real_CPU_metric_prepared=True,
            prior_same_input_shapes_peak_allocated_bytes=360763392,
            scope='Original official0.5.7/standard_fhd/nominal30fps/replicate; no enhancer or SGD/phone/MOS claim.'))
        torch.cuda.reset_peak_memory_stats()
        for ordinal, meta in enumerate(scenes):
            data = first if ordinal == 0 else load_one(meta)
            target = data['gt'].astype(np.float32)/255
            chosen = [r for r in entries if r['scene'] == meta['scene']]
            assert len(chosen) == 10
            for entry in chosen:
                schedule.check_memory()
                values = cached(entry); assert values.shape == data['gt'].shape
                test = np.asarray(values, dtype=np.float32)/255
                torch.cuda.synchronize(); started = time.perf_counter()
                score, _ = predict(metric, test, target)
                torch.cuda.synchronize(); work = time.perf_counter()-started
                identity = {k: entry[k] for k in ('seed', 'scene', 'domain', 'source_video', 'model', 'history', 'label')}
                append(out/'per_sequence.jsonl', dict(**identity, JOD=score, frames=len(values)))
                count += 1; frames += len(values)
                if count < 100:
                    schedule.yield_after_work(work)
                else:
                    schedule.work_seconds += work
                append(out/'resource-observations.jsonl', dict(completed_rows=count, **schedule.summary()))
                print(json.dumps(dict(**identity, completed_rows=count)), flush=True)
                del values, test
            del data, target
        assert (count, frames) == (100, 4480)
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', dict(ended=stamp(), completed_normally=complete, quality_rows=count,
            comparison_frames=frames, new_enhancement_CNN=0, new_SGD=0, seconds=time.perf_counter()-began,
            resource=schedule.summary() if schedule else None,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None))
        globals()['_held_metric_lock'] = lock


if __name__ == '__main__':
    main()
