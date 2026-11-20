"""Official full-output JOD; a resource continuation never re-scores saved rows."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from prebind_shared_gpu import require_initial_binding
BOUND = require_initial_binding()

import numpy as np
import torch
from runtime_io import BASE, PROCESS, precision, verify, write, append, stamp, digest
from shared_group_v1 import acquire_gpu
from causal_reference_evaluation_r75 import metas
from prepare_cache_r5 import load_one
from cvvdp_adapter_r66 import load_metric, predict
from shared_training_schedule import ResourcePressure


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def key(row):
    return row['seed'], row['scene'], row['label']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--resume-from', type=Path)
    args = parser.parse_args()
    precision()
    verify()
    frozen_path = PROCESS / 'perceptual-source-freeze001.json'
    frozen = json.loads(frozen_path.read_text())
    assert frozen['source_sha256'] == {p: digest(ROOT / p) for p in frozen['source_sha256']}
    route_freeze = json.loads((PROCESS / 'gpu-group-execution-freeze002.json').read_text())
    assert route_freeze['source_sha256'] == {p: digest(ROOT / p) for p in route_freeze['source_sha256']}
    entries, manifests = [], {}
    for seed in (0, 1):
        source = BASE / f'results/v1-quality-seed{seed}'
        runtime = json.loads((source / 'runtime.json').read_text())
        assert runtime['completed_normally'] and runtime['quality_rows'] == 110
        path = source / 'prediction_manifest.jsonl'
        entries += lines(path)
        manifests[str(path.relative_to(ROOT))] = digest(path)
    assert len(entries) == len({key(row) for row in entries}) == 220
    out = BASE / f'results/v1-perceptual-attempt{args.attempt:03d}'
    out.mkdir(parents=True, exist_ok=False)
    old = []
    if args.resume_from:
        assert args.resume_from.resolve().is_relative_to(BASE)
        previous = json.loads((args.resume_from / 'runtime.json').read_text())
        prior_config = json.loads((args.resume_from / 'config.json').read_text())
        assert previous['resource_pause'] and not previous['completed_normally']
        assert prior_config['prediction_manifests'] == manifests
        assert prior_config['metric_freeze'] == frozen
        old = lines(args.resume_from / 'per_sequence.jsonl')
        assert len(old) == len({key(row) for row in old})
        assert {key(row) for row in old} <= {key(row) for row in entries}
        for row in old:
            append(out / 'per_sequence.jsonl', row)
    completed = {key(row) for row in old}
    frames = sum(row['frames'] for row in old)
    started = time.perf_counter()
    complete = paused = False
    lock = schedule = None
    try:
        scenes, indices = metas()
        first = load_one(scenes[0])
        prepared, source_cpu = load_metric('cpu')
        assert not torch.cuda.is_initialized()
        write(out / 'config.json', dict(at=stamp(), pid=os.getpid(), metric_freeze=frozen, execution_route_freeze=route_freeze,
              prediction_manifests=manifests, cache_indices=indices, inherited_rows=len(old),
              resume_from=None if args.resume_from is None else str(args.resume_from),
              expected_rows=220, expected_frames=9856, expected_process_increment_MiB=4096,
              prior_same_shapes_allocated_bytes=360763392))
        lock, schedule, _ = acquire_gpu(BOUND, out)
        metric, source = load_metric('cuda')
        del prepared
        assert source['package_sources'] == source_cpu['package_sources']
        assert source['playback_fps'] == 30
        write(out / 'metric-source.json', source)
        torch.cuda.reset_peak_memory_stats()
        for ordinal, meta in enumerate(scenes):
            data = first if ordinal == 0 else load_one(meta)
            target = data['gt'].astype(np.float32) / 255
            for entry in (row for row in entries if row['scene'] == meta['scene']):
                identity = key(entry)
                if identity in completed:
                    continue
                schedule.check_memory()
                path = ROOT / entry['path']
                assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
                assert digest(path) == entry['sha256']
                test = np.load(path, allow_pickle=False) / 255
                assert test.shape == target.shape and np.isfinite(test).all()
                torch.cuda.synchronize()
                began = time.perf_counter()
                score, _ = predict(metric, test, target)
                torch.cuda.synchronize()
                work = time.perf_counter() - began
                append(out / 'per_sequence.jsonl', dict(**{k: entry[k] for k in
                       ('seed', 'scene', 'label', 'domain', 'source_video')}, JOD=score, frames=len(test)))
                completed.add(identity)
                frames += len(test)
                print(json.dumps(dict(completed_rows=len(completed), scene=meta['scene'], label=entry['label'])), flush=True)
                if len(completed) < 220:
                    schedule.yield_after_work(work)
                else:
                    schedule.work_seconds += work
                del test
        assert (len(completed), frames) == (220, 9856)
        verify()
        assert frozen['source_sha256'] == {p: digest(ROOT / p) for p in frozen['source_sha256']}
        complete = True
    except ResourcePressure as error:
        paused = True
        write(out / 'resource-pause.json', dict(at=stamp(), reason=str(error), saved_rows=len(completed)))
        raise SystemExit(75)
    except BaseException:
        (out / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out / 'runtime.json', dict(at=stamp(), pid=os.getpid(), completed_normally=complete,
              resource_pause=paused, quality_rows=len(completed), comparison_frames=frames,
              inherited_rows=len(old), newly_scored_rows=len(completed) - len(old),
              seconds=time.perf_counter() - started, GPU=None if schedule is None else schedule.summary(),
              max_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0,
              max_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0))
        if lock:
            lock.close()


if __name__ == '__main__':
    main()
