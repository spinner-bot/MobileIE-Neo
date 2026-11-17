"""Actual resident CPU baseline pipelines; complete timing includes conversion/copy."""
import time

import numpy as np
import torch

from aligned_raw_history_r11 import align_history
from aggregation_controls_r20 import blend
from half_motion_r4 import half_forward_backward


def storage_bytes(values):
    """Only owned persistent RGB arrays, deduplicating shared startup storage."""
    pointers = {}
    for value in values:
        if value is not None:
            assert value.flags.owndata and value.dtype == np.float32
            pointers[value.__array_interface__['data'][0]] = value.nbytes
    return sum(pointers.values())


def run(backbone, low, mode, sink, counters):
    if mode not in ('raw', 'strongrec') or low.shape != sink.shape or low.shape[0] != 16:
        raise ValueError('Only the registered sixteen-frame baseline pipelines are allowed')
    if low.dtype != np.float32 or sink.dtype != np.float32:
        raise ValueError('CPU input and sink must be FP32')
    previous_raw = previous_fused = None
    samples = []
    with torch.inference_mode():
        cpu_start = time.process_time_ns()
        started = time.perf_counter_ns()
        for t, frame in enumerate(low):
            frame_start = time.perf_counter_ns()
            image = torch.from_numpy(frame.transpose(2, 0, 1).copy()[None])/255
            raw = backbone(image)[0].clamp(0, 1).permute(1, 2, 0).contiguous().numpy()*255
            counters['backbone_image_frames'] += 1
            enhanced = time.perf_counter_ns()
            if mode == 'strongrec':
                if previous_raw is None:
                    output = raw
                else:
                    flow, mask = half_forward_backward(previous_raw, raw)
                    counters['half_FB_pairs'] += 1
                    moved, _ = align_history(previous_fused, raw, flow, mask)
                    output = blend(raw, moved, .5)
                previous_raw, previous_fused = raw, output
            else:
                output = raw
            postprocessed = time.perf_counter_ns()
            sink[t] = output
            ended = time.perf_counter_ns()
            samples.append(dict(frame=t, milliseconds=(ended-frame_start)/1e6,
                conversion_CNN_output_ms=(enhanced-frame_start)/1e6,
                motion_warp_blend_ms=(postprocessed-enhanced)/1e6,
                sink_copy_ms=(ended-postprocessed)/1e6))
        wall = time.perf_counter_ns()-started
        cpu = time.process_time_ns()-cpu_start
    assert np.isfinite(sink).all()
    actual = storage_bytes([previous_raw, previous_fused])
    expected = (6*np.prod(low.shape[1:3])*4) if mode == 'strongrec' else 0
    assert actual == expected
    return dict(frames=len(low), milliseconds_per_frame=wall/1e6/len(low),
        wall_milliseconds=wall/1e6, process_CPU_milliseconds=cpu/1e6,
        conversion_CNN_output_ms=sum(r['conversion_CNN_output_ms'] for r in samples),
        motion_warp_blend_ms=sum(r['motion_warp_blend_ms'] for r in samples),
        sink_copy_ms=sum(r['sink_copy_ms'] for r in samples),
        first_frame_milliseconds=samples[0]['milliseconds'],
        nonfirst_mean_milliseconds=float(np.mean([r['milliseconds'] for r in samples[1:]])),
        persistent_state_bytes=actual, output_sink_bytes=sink.nbytes,
        per_frame=samples)
