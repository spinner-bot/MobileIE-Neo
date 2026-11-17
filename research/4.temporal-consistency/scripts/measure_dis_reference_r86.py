"""Paired same-run complete native-DIS and legacy-half-FB resident CPU costs."""
import os
if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise SystemExit('An initially empty CUDA mask is required')

import json
import platform
import statistics
import time
import traceback

import cv2
import numpy as np
import torch

from dis_reference_io_r86 import ROOT, QUALITY, COST, LABEL, verify, cached, lines, stamp, write, append
from fixed_dis_r86 import FixedDIS, RecursivePipeline
from half_motion_r4 import half_forward_backward
from baseline_pipeline_r85 import storage_bytes
from causal_reference_evaluation_r75 import metas, reference_predictions
from motion_transport_r74 import load_spatial
from causal_reference_io_r75 import digest


def run(backbone, low, sink, mode, counters):
    motion = FixedDIS() if mode == 'DIS' else half_forward_backward
    stream = RecursivePipeline(motion)
    samples = []
    with torch.inference_mode():
        cpu_start, started = time.process_time_ns(), time.perf_counter_ns()
        for t, frame in enumerate(low):
            began = time.perf_counter_ns()
            image = torch.from_numpy(frame.transpose(2, 0, 1).copy()[None])/255
            raw = backbone(image)[0].clamp(0, 1).permute(1, 2, 0).contiguous().numpy()*255
            counters['backbone_image_frames'] += 1
            enhanced = time.perf_counter_ns()
            output, field = stream.step(raw)
            if field is not None:
                counters['DIS_pairs' if mode == 'DIS' else 'half_FB_pairs'] += 1
            postprocessed = time.perf_counter_ns(); sink[t] = output; ended = time.perf_counter_ns()
            samples.append(dict(frame=t, milliseconds=(ended-began)/1e6,
                conversion_CNN_output_ms=(enhanced-began)/1e6,
                motion_warp_blend_ms=(postprocessed-enhanced)/1e6,
                sink_copy_ms=(ended-postprocessed)/1e6))
        wall, cpu = time.perf_counter_ns()-started, time.process_time_ns()-cpu_start
    assert np.isfinite(sink).all()
    state = storage_bytes([stream.previous_raw, stream.previous_fused])
    assert state == 6*low.shape[1]*low.shape[2]*4
    return dict(frames=len(low), milliseconds_per_frame=wall/1e6/len(low), wall_milliseconds=wall/1e6,
        process_CPU_milliseconds=cpu/1e6, persistent_RGB_state_bytes=state, output_sink_bytes=sink.nbytes,
        first_frame_milliseconds=samples[0]['milliseconds'],
        nonfirst_mean_milliseconds=float(np.mean([r['milliseconds'] for r in samples[1:]])), per_frame=samples)


def main():
    frozen = verify(); runtime = json.loads((QUALITY/'runtime.json').read_text())
    assert runtime['completed_normally'] and runtime['quality_rows'] == 20
    scenes, indices = metas()
    old = {(r['seed'], r['scene'], r['label']): r for r in reference_predictions()}
    new = {(r['seed'], r['scene']): r for r in lines(QUALITY/'prediction_manifest.jsonl')}
    torch.set_num_threads(2); torch.set_num_interop_threads(1); cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    COST.mkdir(exist_ok=False)
    cpu_model = [s.split(':', 1)[1].strip() for s in open('/proc/cpuinfo') if s.startswith('model name')][0]
    write(COST/'config.json', dict(at=stamp(), pid=os.getpid(), frozen=frozen, indices=indices,
        cpu_model=cpu_model, platform=platform.platform(), torch_version=str(torch.__version__),
        OpenCV_version=cv2.__version__, Torch_threads=2, Torch_interop_threads=1, OpenCV_threads=1,
        affinity_CPU_count=len(os.sched_getaffinity(0)), initial_host_load=os.getloadavg(),
        expected_backbone_image_frames=3200, expected_DIS_pairs=1500, expected_half_FB_pairs=1500,
        expected_warmup_sequences=80, expected_measured_sequences=120,
        scope='Same-run complete post pipelines, not cross-run R85 comparison or phone/camera latency.'))
    counters = dict(backbone_image_frames=0, DIS_pairs=0, half_FB_pairs=0)
    warm = measured = qualified = 0; complete = False; measured_rows = []
    began = time.perf_counter()
    try:
        for seed in (0, 1):
            net, marker = load_spatial(seed); write(COST/f'backbone-seed{seed}.json', marker)
            for meta in scenes:
                path = ROOT/meta['path']
                assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo') and digest(path) == meta['sha256']
                with np.load(path, allow_pickle=False) as archive:
                    low = archive['low'][:16].astype(np.float32, copy=True)
                sink = np.empty_like(low)
                refs = dict(DIS=cached(new[seed, meta['scene']])[:16],
                    half_FB=cached(old[seed, meta['scene'], 'MobileIE_recursive_self05'])[:16])
                assert new[seed, meta['scene']]['checkpoint_sha256'] == marker['checkpoint_sha256']
                for phase, repeats in [('warmup', 2), ('measured', 3)]:
                    for repeat in range(repeats):
                        order = ('DIS', 'half_FB') if (repeat+seed) % 2 == 0 else ('half_FB', 'DIS')
                        for order_index, mode in enumerate(order):
                            result = run(net, low, sink, mode, counters)
                            row = dict(seed=seed, domain=meta['domain'], scene=meta['scene'],
                                source_video=meta['source_video'], shape=list(low.shape), mode=mode,
                                phase=phase, repeat=repeat, order_index=order_index,
                                host_load_after_sequence=os.getloadavg(), **result)
                            append(COST/f'{phase}.jsonl', row)
                            if phase == 'warmup':
                                warm += 1
                                if repeat == 0:
                                    maximum = float(np.abs(sink.astype(np.float64)-refs[mode]).max()/255)
                                    append(COST/'qualification.jsonl', dict(seed=seed, scene=meta['scene'],
                                        mode=mode, normalized_max_abs_difference=maximum,
                                        tolerance=1e-5, passed=maximum <= 1e-5, extra_CNN_calls=0))
                                    assert maximum <= 1e-5; qualified += 1
                            else:
                                measured += 1; measured_rows.append(row)
                write(COST/'live.json', dict(at=stamp(), pid=os.getpid(), seed=seed,
                    last_completed_condition=meta['scene'], warmup_sequences=warm,
                    measured_sequences=measured, qualified_pipelines=qualified, **counters,
                    scope='Completed prefix only; not a final cost result.'))
                print(json.dumps(dict(seed=seed, scene=meta['scene'], measured_sequences=measured, **counters)), flush=True)
                del low, sink, refs
        conditions = []
        for seed in (0, 1):
            for meta in scenes:
                modes = {}
                for mode in ('DIS', 'half_FB'):
                    rows = [r for r in measured_rows if (r['seed'], r['scene'], r['mode']) == (seed, meta['scene'], mode)]
                    assert len(rows) == 3
                    times = [r['milliseconds_per_frame'] for r in rows]
                    modes[mode] = dict(median_ms=statistics.median(times), minimum_ms=min(times), maximum_ms=max(times),
                        repeats_ms=times, persistent_RGB_state_bytes=rows[0]['persistent_RGB_state_bytes'],
                        output_sink_bytes=rows[0]['output_sink_bytes'])
                conditions.append(dict(seed=seed, domain=meta['domain'], scene=meta['scene'], modes=modes,
                    DIS_to_half_FB_ratio=modes['DIS']['median_ms']/modes['half_FB']['median_ms']))
        groups = []
        for domain in ('davis_train_pilot', 'paired_real_ND_DEV'):
            for seed in (0, 1):
                rows = [r for r in conditions if (r['domain'], r['seed']) == (domain, seed)]
                assert len(rows) == (6 if domain == 'davis_train_pilot' else 4)
                ratio = float(np.mean([r['DIS_to_half_FB_ratio'] for r in rows]))
                groups.append(dict(domain=domain, seed=seed, conditions=len(rows),
                    mean_condition_median_DIS_ms=float(np.mean([r['modes']['DIS']['median_ms'] for r in rows])),
                    mean_condition_median_half_FB_ms=float(np.mean([r['modes']['half_FB']['median_ms'] for r in rows])),
                    mean_condition_ratio=ratio, speed_gate=ratio <= .8))
        assert (warm, measured, qualified) == (80, 120, 40)
        assert counters == dict(backbone_image_frames=3200, DIS_pairs=1500, half_FB_pairs=1500)
        assert not torch.cuda.is_initialized(); verify()
        write(COST/'summary.json', dict(at=stamp(), conditions=conditions, groups=groups,
            all_speed_gates=all(r['speed_gate'] for r in groups), scope='Paired same-run resident CPU; all repeats retained, not handset.'))
        complete = True
    except BaseException:
        (COST/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(COST/'runtime.json', dict(ended=stamp(), pid=os.getpid(), completed_normally=complete,
            seconds=time.perf_counter()-began, warmup_sequences=warm, measured_sequences=measured,
            qualified_pipelines=qualified, **counters, new_SGD=0, CUDA_initialized=torch.cuda.is_initialized(),
            final_host_load=os.getloadavg()))


if __name__ == '__main__':
    main()
