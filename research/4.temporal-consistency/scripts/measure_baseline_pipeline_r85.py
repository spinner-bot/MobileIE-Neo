"""Fixed full CPU baseline costs, not a phone test or new quality experiment."""
import os
if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise SystemExit('An initially empty CUDA mask is required.')

import json
import platform
import statistics
import time
import traceback

import cv2
import numpy as np
import torch

from baseline_pipeline_r85 import run
from causal_reference_io_r75 import ROOT, BASE, stamp, write, append, digest
from causal_reference_evaluation_r75 import metas, reference_predictions
from motion_transport_r74 import load_spatial

PROCESS = BASE/'process/exploration-r85'
PROTOCOL = BASE/'docs/exploration-r85-baseline-pipeline-cost-protocol.md'


def verify():
    frozen = json.loads((PROCESS/'source-freeze001.json').read_text())
    assert frozen['source_sha256'] == {name: digest(ROOT/name) for name in frozen['source_sha256']}
    assert frozen['protocol_sha256'] == digest(PROTOCOL)
    return frozen


def owned(path):
    path = ROOT/path
    if not path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo'):
        raise ValueError('Input/prediction must belong to the project data directory')
    return path


def reference(entry):
    path = owned(entry['path']); assert digest(path) == entry['sha256']
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    assert list(values.shape) == entry['shape']
    return values[:16]


def main():
    out = PROCESS/'cpu001'; out.mkdir(exist_ok=False)
    frozen = verify()
    torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    scenes, indices = metas()
    entries = {(v['seed'], v['scene'], v['label']): v for v in reference_predictions()}
    assert len(entries) == 40
    cpu_model = [s.split(':', 1)[1].strip() for s in open('/proc/cpuinfo') if s.startswith('model name')][0]
    write(out/'config.json', dict(at=stamp(), pid=os.getpid(), source_freeze=frozen, indices=indices,
        cpu_model=cpu_model, platform=platform.platform(), torch_version=str(torch.__version__),
        OpenCV_version=cv2.__version__, Torch_threads=torch.get_num_threads(),
        Torch_interop_threads=torch.get_num_interop_threads(), OpenCV_threads=cv2.getNumThreads(),
        affinity_CPU_count=len(os.sched_getaffinity(0)), initial_host_load=os.getloadavg(),
        seeds=[0, 1], modes=['raw', 'strongrec'], frames_per_sequence=16,
        warmup_sequences_expected=80, measured_sequences_expected=120,
        expected_backbone_image_frames=3200, expected_half_FB_pairs=1500,
        shared_CPU_not_exclusive=True, data_IO_and_model_load_timed=False,
        scope='Resident FP32 CPU pipeline, all fixed condition prefixes. Not handset/camera/display or new quality evidence.'))
    counters = dict(backbone_image_frames=0, half_FB_pairs=0)
    warmups = measured = qualified = 0; complete = False
    started = time.perf_counter(); measured_rows = []
    try:
        for seed in (0, 1):
            backbone, marker = load_spatial(seed)
            assert sum(p.numel() for p in backbone.parameters()) == 4075
            write(out/f'backbone-seed{seed}.json', marker)
            for condition_index, meta in enumerate(scenes):
                source_path = owned(meta['path']); assert digest(source_path) == meta['sha256']
                with np.load(source_path, allow_pickle=False) as archive:
                    low = archive['low'][:16].astype(np.float32, copy=True)
                assert low.shape[0] == 16 and low.shape[1:] == tuple(meta['shape'][1:])
                assert np.isfinite(low).all()
                sink = np.empty_like(low)
                refs = {mode: reference(entries[seed, meta['scene'], label]) for mode, label in
                    [('raw', 'MobileIE_raw'), ('strongrec', 'MobileIE_recursive_self05')]}
                append(out/'inputs.jsonl', dict(seed=seed, condition_index=condition_index,
                    input_cache=meta, actual_input_prefix_shape=list(low.shape), input_dtype=str(low.dtype),
                    decoded_low_archive_may_have_more_than_16_frames=True))
                condition_start = time.perf_counter()
                for phase, repeats in [('warmup', 2), ('measured', 3)]:
                    for repeat in range(repeats):
                        order = ('raw', 'strongrec') if (repeat+seed) % 2 == 0 else ('strongrec', 'raw')
                        for order_index, mode in enumerate(order):
                            values = run(backbone, low, mode, sink, counters)
                            row = dict(seed=seed, domain=meta['domain'], scene=meta['scene'],
                                source_video=meta['source_video'], shape=list(low.shape), mode=mode,
                                phase=phase, repeat=repeat, order_index=order_index,
                                host_load_after_sequence=os.getloadavg(), **values)
                            append(out/f'{phase}.jsonl', row)
                            if phase == 'warmup':
                                warmups += 1
                                if repeat == 0:
                                    maximum = float(np.max(np.abs(sink.astype(np.float64)-refs[mode]))/255)
                                    append(out/'qualification.jsonl', dict(seed=seed, scene=meta['scene'],
                                        mode=mode, reference=entries[seed, meta['scene'],
                                            'MobileIE_raw' if mode == 'raw' else 'MobileIE_recursive_self05'],
                                        compared_frames=16, normalized_max_abs_difference=maximum,
                                        tolerance=1e-5, passed=maximum <= 1e-5,
                                        extra_CNN_calls_for_qualification=0))
                                    assert maximum <= 1e-5; qualified += 1
                            else:
                                measured += 1; measured_rows.append(row)
                write(out/'live.json', dict(at=stamp(), pid=os.getpid(), seed=seed,
                    last_completed_condition=meta['scene'], warmup_sequences=warmups,
                    measured_sequences=measured, qualified_pipelines=qualified, **counters,
                    scope='Completed prefix only, not a final summary.'))
                print(json.dumps(dict(seed=seed, scene=meta['scene'], measured_sequences=measured,
                    backbone_image_frames=counters['backbone_image_frames'],
                    condition_seconds=time.perf_counter()-condition_start)), flush=True)
                del low, sink, refs
            assert not backbone.training and not any(p.requires_grad for p in backbone.parameters())
        conditions = []
        for seed in (0, 1):
            for meta in scenes:
                modes = {}
                for mode in ('raw', 'strongrec'):
                    rows = [r for r in measured_rows if (r['seed'], r['scene'], r['mode']) == (seed, meta['scene'], mode)]
                    assert len(rows) == 3
                    times = [r['milliseconds_per_frame'] for r in rows]
                    modes[mode] = dict(median_ms=statistics.median(times), minimum_ms=min(times), maximum_ms=max(times),
                        repeats_ms=times, first_frame_median_ms=statistics.median(r['first_frame_milliseconds'] for r in rows),
                        nonfirst_mean_median_ms=statistics.median(r['nonfirst_mean_milliseconds'] for r in rows),
                        persistent_state_bytes=rows[0]['persistent_state_bytes'],
                        output_sink_bytes=rows[0]['output_sink_bytes'])
                conditions.append(dict(seed=seed, domain=meta['domain'], scene=meta['scene'],
                    shape=meta['shape'][1:], modes=modes,
                    recursive_to_raw_median_ratio=modes['strongrec']['median_ms']/modes['raw']['median_ms']))
        groups = []
        for domain in ('davis_train_pilot', 'paired_real_ND_DEV'):
            for seed in (0, 1):
                rows = [r for r in conditions if (r['domain'], r['seed']) == (domain, seed)]
                assert len(rows) == (6 if domain == 'davis_train_pilot' else 4)
                groups.append(dict(domain=domain, seed=seed, conditions=len(rows),
                    mean_condition_median_raw_ms=float(np.mean([r['modes']['raw']['median_ms'] for r in rows])),
                    mean_condition_median_recursive_ms=float(np.mean([r['modes']['strongrec']['median_ms'] for r in rows])),
                    mean_condition_median_ratio=float(np.mean([r['recursive_to_raw_median_ratio'] for r in rows])),
                    minimum_condition_ratio=min(r['recursive_to_raw_median_ratio'] for r in rows),
                    maximum_condition_ratio=max(r['recursive_to_raw_median_ratio'] for r in rows)))
        assert (warmups, measured, qualified) == (80, 120, 40)
        assert counters == dict(backbone_image_frames=3200, half_FB_pairs=1500)
        assert not torch.cuda.is_initialized(); verify()
        write(out/'summary.json', dict(at=stamp(), conditions=conditions, groups=groups,
            scope='All repeats retained, shared nonexclusive CPU, fixed prefixes; not mobile latency, new method or quality gate.'))
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(ended=stamp(), pid=os.getpid(), completed_normally=complete,
            seconds=time.perf_counter()-started, warmup_sequences=warmups, measured_sequences=measured,
            qualified_pipelines=qualified, **counters, optimizer_updates=0,
            CUDA_initialized=torch.cuda.is_initialized(), final_host_load=os.getloadavg()))


if __name__ == '__main__':
    main()
