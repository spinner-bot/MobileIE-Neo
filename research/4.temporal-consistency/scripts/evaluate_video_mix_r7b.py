"""Frozen channel/placement controls with common input reset and cheap output fusion."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from exploration_r2_data import ROOT, detect_changes, evaluate
from explore_state_r2 import fuse
from explore_temporal_controls import save_json, sha256
from feature_mix_r7 import FeatureMix, SPECS
from frozen_mix_utils_r7 import predict, provenance
from prepare_cache_r5 import CACHE_INDEX, load_one
from train_exploration_r2 import new_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--methods', required=True)
    p.add_argument('--include-baseline', action='store_true')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    names = args.methods.split(',')
    if len(set(names)) != len(names) or any(name not in SPECS for name in names):
        p.error('unique known methods required')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    base = new_model(device)
    models = {name: FeatureMix(base, **SPECS[name]).to(device).eval() for name in names}
    save_json(out/'config.json', {**provenance(), 'specs': {k: SPECS[k] for k in names},
        'include_pretrained': args.include_baseline, 'cache_index_sha256': sha256(CACHE_INDEX),
        'input_event_detector': {'brightness_ratio': 1.35, 'normalized_thumbnail_difference': .5},
        'fusion_alpha': .1, 'fusion_reset': 'same input events for every method including baseline',
        'scope': 'frozen official weights, causal one-past-raw-state, reused exploratory video conditions'})
    start, count, checks, events_log = time.perf_counter(), 0, [], []
    for meta in json.loads(CACHE_INDEX.read_text())['scenes']:
        data = load_one(meta)
        events, event_details = detect_changes(data['low'])
        annotated = 12 if meta['domain'].startswith('event_') else None
        events_log.append({'domain': meta['domain'], 'scene': meta['scene'], 'resets': events,
            'details': event_details, 'annotated_event_frame': annotated,
            'event_hit': bool(events[12]) if annotated is not None else None,
            'extra_detections': sum(events)-int(events[12]) if annotated is not None else None,
            'unknown_real_events': 'no event annotations; reset count is not a false-positive rate'})
        inputs = torch.from_numpy((data['low']/255.).transpose(0, 3, 1, 2).copy()).to(device)
        entries = [('pretrained', None)] if args.include_baseline else []
        entries += list(models.items())
        for method, model in entries:
            for reset in [False, True]:
                if model is None:
                    raw = data['teacher']
                    difference = 0.
                else:
                    pred = predict(model, inputs, events if reset else None)
                    raw = np.ascontiguousarray(pred.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()*255.)
                    difference = float(np.abs(raw[0]-data['teacher'][0]).max()/255.)
                    if difference > 2e-6:
                        raise AssertionError('first-frame mismatch with common original model')
                    checks.append({'scene': meta['scene'], 'method': method, 'reset': reset,
                                   'normalized_first_frame_max_abs_difference': difference,
                                   'state_storage_bytes': model.state_storage_bytes()})
                for fusion, output in [('raw', raw), ('half01', fuse(raw, data['half_flow'], data['half_mask'],
                                                        resets=events if reset else None, alpha=.1))]:
                    identity = {'domain': meta['domain'], 'scene': meta['scene'], 'method': method,
                                'input_reset': reset, 'fusion': fusion}
                    metrics, frames = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                    with (out/'per_sequence.jsonl').open('a') as stream:
                        stream.write(json.dumps({**identity, **metrics}, allow_nan=False)+'\n')
                    with (out/'per_frame.jsonl').open('a') as stream:
                        for frame in frames:
                            stream.write(json.dumps({**identity, **frame}, allow_nan=False)+'\n')
                    count += 1
        print(json.dumps({'scene': meta['scene'], 'rows': count}), flush=True)
    expected = (len(names)+int(args.include_baseline))*38*4
    if count != expected:
        raise ValueError('fixed video evaluation budget mismatch')
    save_json(out/'first_frame_checks.json', checks)
    save_json(out/'input_events.json', events_log)
    save_json(out/'runtime.json', {'wall_seconds': time.perf_counter()-start, 'rows': count})


if __name__ == '__main__':
    main()
