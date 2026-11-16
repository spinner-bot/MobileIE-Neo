"""Complete160 new full/tail perceptual comparisons; existing40 full references reused."""
import os
from pathlib import Path
import sys
import time
import traceback
import json

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
from prebind_shared_gpu import require_initial_binding
from coefficient_io_r80 import BASE, LABELS, write, append, lines, stamp, verify, acquire_gpu


def main():
    bound = require_initial_binding()
    import cv2
    import numpy as np
    import torch
    from cvvdp_adapter_r66 import load_metric, predict
    from causal_reference_evaluation_r75 import metas
    from evaluate_cvvdp_references_r66 import cached
    from prepare_cache_r5 import load_one
    torch.set_num_threads(2); cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    verify(); entries = []
    for seed in (0, 1):
        run = BASE/f'results/exploration-r80-eval-seed{seed}'
        terminal = json.loads((run/'runtime.json').read_text())
        assert terminal['completed_normally'] and terminal['new_quality_rows'] == 80 and terminal['frame_rows'] == 3184
        records = lines(run/'prediction_manifest.jsonl'); assert len(records) == 50
        entries += records
    scenes, indices = metas(); first = load_one(scenes[0])
    prepared, expected_source = load_metric('cpu'); assert not torch.cuda.is_initialized()
    out = BASE/'results/exploration-r80-perceptual001'; out.mkdir(exist_ok=False)
    began, complete, count, frames, schedule, lock = time.perf_counter(), False, 0, 0, None, None
    try:
        lock, schedule, selected = acquire_gpu(bound, out)
        metric, source = load_metric('cuda'); del prepared
        assert source['package_sources'] == expected_source['package_sources'] and source['playback_fps'] == 30
        write(out/'metric-source.json', source)
        write(out/'config.json', dict(at=stamp(), expected_rows=160, expected_comparison_frames=6368,
            cache_indices=indices, prior_full_shape_peak_allocated_bytes=360763392,
            scope='All new full and all tail, official0.5.7/standard_fhd/30fps/replicate; no enhancer/SGD/phone/MOS claim.'))
        torch.cuda.reset_peak_memory_stats()
        for ordinal, meta in enumerate(scenes):
            data = first if ordinal == 0 else load_one(meta); target = data['gt'].astype(np.float32)/255
            chosen = [entry for entry in entries if entry['scene'] == meta['scene']]; assert len(chosen) == 10
            for entry in chosen:
                values = cached(entry); assert values.shape == data['gt'].shape
                for scope, start in (('full', 0), ('tail', 8)):
                    if scope == 'full' and entry['label'] not in LABELS:
                        continue
                    schedule.check_memory()
                    test = np.asarray(values[start:], dtype=np.float32)/255
                    reference = np.ascontiguousarray(target[start:])
                    torch.cuda.synchronize(); started = time.perf_counter()
                    score, _ = predict(metric, test, reference)
                    torch.cuda.synchronize(); work = time.perf_counter()-started
                    identity = {k: entry[k] for k in ('seed', 'domain', 'scene', 'source_video', 'label')}
                    append(out/'per_sequence.jsonl', dict(**identity, scope=scope, JOD=score, frames=len(test)))
                    count += 1; frames += len(test)
                    if count < 160:
                        schedule.yield_after_work(work)
                    else:
                        schedule.work_seconds += work
                    append(out/'resource-observations.jsonl', dict(completed_rows=count, **schedule.summary()))
                    print(json.dumps(dict(**identity, scope=scope, completed_rows=count)), flush=True)
                    del test, reference
                del values
            del data, target
        assert (count, frames) == (160, 6368); verify(); complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(ended=stamp(), completed_normally=complete, quality_rows=count,
            comparison_frames=frames, new_enhancement_CNN=0, new_global_SGD=0, seconds=time.perf_counter()-began,
            resource=schedule.summary() if schedule else None,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None))
        globals()['_held_metric_lock'] = lock


if __name__ == '__main__':
    main()
