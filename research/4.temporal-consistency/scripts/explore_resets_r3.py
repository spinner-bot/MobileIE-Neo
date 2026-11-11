"""Factorial gate/output history controls with fixed R2 change detection."""
import argparse
import json
import time
from pathlib import Path

import cv2
import torch

from exploration_r2_data import detect_changes, evaluate, load_images, make_sequence
from explore_state_r2 import forward_backward, fuse, infer_gates
from explore_temporal_controls import save_json
from train_exploration_r2 import new_model


def reset_methods(gates, flows, confidence, events):
    methods = {}
    for gate_name, output in gates.items():
        methods[gate_name+'__none'] = (output, False)
        for flow_name, mask in [('flow', None), ('fb', confidence)]:
            for reset in [False, True]:
                key = gate_name+'__'+flow_name+('__reset' if reset else '__keep')
                methods[key] = (fuse(output, flows, mask, events if reset else None), reset)
    return methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    device = torch.device('cuda')
    net = new_model(device)
    images = load_images('exploratory_validation')
    names = list(images)
    save_json(out/'config.json', {'gate_alpha': .9, 'output_alpha': .7,
        'detector': {'brightness_ratio': 1.35, 'thumbnail_difference': .5},
        'frames': 24, 'conditions': 48, 'methods': 15,
        'windows': {'before': [0, 11], 'event': [12, 12], 'recovery': [13, 15], 'late': [16, 23]}})
    start = time.perf_counter()
    count = 0
    with (out/'per_sequence.jsonl').open('w') as seq, (out/'per_frame.jsonl').open('w') as trace:
        for index, name in enumerate(names):
            for case in ['pan', 'light_step', 'occlusion', 'cut']:
                for seed in [0, 1]:
                    gt, low, true_flow, true_mask = make_sequence(images[name], images[names[(index+1)%len(names)]], case, seed)
                    events, details = detect_changes(low)
                    gates = {'raw': infer_gates(net, low, device)[0],
                             'ema': infer_gates(net, low, device, .9)[0],
                             'reset': infer_gates(net, low, device, .9, events)[0]}
                    pairs = [forward_backward(low[t-1], low[t]) for t in range(1, len(low))]
                    flows, confidence = [p[0] for p in pairs], [p[1] for p in pairs]
                    for method, (outputs, output_reset) in reset_methods(gates, flows, confidence, events).items():
                        metrics, frames = evaluate(outputs, gt, true_flow, true_mask)
                        identity = {'scene': name, 'case': case, 'input_seed': seed, 'method': method}
                        seq.write(json.dumps({**identity, **metrics, 'detected_events': sum(events)}, allow_nan=False)+'\n')
                        for frame in frames:
                            t = frame['frame']
                            trace.write(json.dumps({**identity, **frame, **details[t], 'detector_event': events[t],
                                'gate_reset': bool(events[t] and method.startswith('reset__')),
                                'output_reset': bool(events[t] and output_reset)}, allow_nan=False)+'\n')
                        count += 1
                    print(json.dumps({'scene': name, 'case': case, 'seed': seed, 'rows': count}), flush=True)
    save_json(out/'runtime.json', {'wall_seconds': time.perf_counter()-start, 'sequence_rows': count})


if __name__ == '__main__':
    main()
