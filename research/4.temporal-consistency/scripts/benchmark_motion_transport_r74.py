"""One fixed CPU temporal-stage benchmark, only after all16 quality gates pass."""
from datetime import datetime
import json
import os
import platform
import time
import traceback

import cv2
import numpy as np
import torch

from motion_transport_r74 import BASE, ROOT, PROTOCOL, RGBStream
from check_motion_transport_r74 import write
from dynamic_training_r34 import digest
from evaluate_motion_transport_r74 import metas, reference_predictions, load_final
from evaluate_cvvdp_references_r66 import cached, lines, append
from render_dense_history_r64 import recursive


def motion_sequence(net, raw):
    stream, values = RGBStream(net), []
    with torch.inference_mode():
        for t in range(len(raw)):
            current = torch.from_numpy(raw[t].transpose(2, 0, 1).copy()[None])/255
            value, _ = stream.step(current)
            values.append(raw[0].copy() if t == 0 else
                          value[0].clamp(0, 1).permute(1, 2, 0).contiguous().numpy()*255)
    return np.stack(values), stream.state_storage_bytes()


def main():
    audit_path = BASE/'report/exploration-r74-joint/audit.json'
    audit = json.loads(audit_path.read_text()); assert audit['completed'] and audit['all_gates_pass'] is True
    torch.set_num_threads(2); cv2.setNumThreads(2); torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    references, sources = reference_predictions(); scenes, indices = metas(); sources.update(indices)
    lookup = {(r['seed'], r['scene'], r['method']): r for r in references}
    predictions = []
    for seed in [0, 1]: predictions += lines(BASE/f'results/exploration-r74-eval-seed{seed}/prediction_manifest.jsonl')
    candidate = {(r['seed'], r['scene']): r for r in predictions if (r['model'], r['history']) == ('proxy', 'causal')}
    assert len(candidate) == 20
    out = BASE/'report/exploration-r74-cost'; out.mkdir(exist_ok=False)
    write(out/'config.json', dict(at=datetime.now().astimezone().isoformat(), quality_audit_sha256=digest(audit_path),
          protocol_sha256=digest(PROTOCOL), sources=sources, warmup_full_sequences=2, repeats=3,
          torch_threads=2, opencv_threads=2, platform=platform.platform(), CPU_affinity=sorted(os.sched_getaffinity(0)),
          scope='CPU temporal stage from already loaded raw arrays, conversion/fields/warp/state copies/output assembly included. No I/O or shared frozen spatial CNN; not handset/full-pipeline or isolated-server latency.'))
    started, complete, summaries, pair_predictions = time.perf_counter(), False, [], 0
    try:
        for seed in [0, 1]:
            nets, markers = load_final(seed); net = nets['proxy']; assert markers['proxy']['total_parameters_including_frozen_spatial'] == 25993
            for meta in scenes:
                raw = cached(lookup[seed, meta['scene'], 'raw'])
                expected_proxy = cached(candidate[seed, meta['scene']])
                expected_classic = cached(lookup[seed, meta['scene'], 'recursive_self05'])
                timings = {'proxy': [], 'classic': []}
                for repetition in range(5):
                    order = ['proxy', 'classic'] if repetition % 2 else ['classic', 'proxy']
                    for method in order:
                        before_load = os.getloadavg(); began = time.perf_counter()
                        if method == 'proxy':
                            value, state_bytes = motion_sequence(net, raw); pair_predictions += len(raw)-1
                        else:
                            value = recursive(raw); state_bytes = None
                        seconds = time.perf_counter()-began
                        target = expected_proxy if method == 'proxy' else expected_classic
                        delta = float(np.abs(value-target).max()); assert delta <= 1e-4 and np.isfinite(value).all()
                        if method == 'proxy': assert state_bytes == 6*raw.shape[1]*raw.shape[2]*4
                        append(out/'measurements.jsonl', dict(seed=seed, scene=meta['scene'], domain=meta['domain'],
                               method=method, repetition=repetition, warmup=repetition < 2, seconds=seconds,
                               frames=len(raw), state_bytes=state_bytes, output_max_delta_8bit=delta, host_load_before=before_load))
                        if repetition >= 2: timings[method].append(seconds)
                student, classic = (float(np.median(timings[n])) for n in ['proxy', 'classic'])
                summaries.append(dict(seed=seed, scene=meta['scene'], domain=meta['domain'], frames=len(raw),
                      proxy_seconds=student, classic_seconds=classic, ratio=student/classic,
                      proxy_ms_per_frame=1000*student/len(raw), classic_ms_per_frame=1000*classic/len(raw)))
        assert len(summaries) == 20 and pair_predictions == 4380
        seed_ratios = {str(seed): float(np.median([r['ratio'] for r in summaries if r['seed'] == seed])) for seed in [0, 1]}
        write(out/'per_condition.json', summaries)
        write(out/'audit.json', dict(completed=True, quality_gates_passed=True, temporal_cost_gate_passed=all(v <= .5 for v in seed_ratios.values()),
              seed_median_ratios=seed_ratios, motion_parameters=21918, total_parameters=25993,
              compact_RGB_state_bytes1080p_FP32=49766400, new_motion_pair_predictions=pair_predictions,
              new_enhancement_CNN=0, SGD=0, phone_measured=False, independent_final=False))
        complete = True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, conditions=len(summaries),
              seconds=time.perf_counter()-started, ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
