"""All fixed4000 temporal FST controls and same-weight spatial-plus-post references."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from aggregation_controls_r20 import blend, event_output
from aligned_raw_history_r11 import align_history
from evaluation_aliases_r14 import event_flags
from exploration_r2_data import ROOT, detect_changes, evaluate
from explore_state_r2 import fuse
from explore_temporal_controls import save_json, sha256
from prepare_cache_r5 import CACHE_INDEX, load_one
from prepare_development_r9 import DEV_INDEX
from prepare_native_r10 import INDEX
from transport_evaluation_r28 import EXTERNAL, infer, load_models, output_keys
from transport_fst_r27 import MODES

BASE = ROOT/'research/4.temporal-consistency'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['copy', 'post_transport'], required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')
    models, checkpoints = load_models(args.mode, device)
    save_json(out/'checkpoints.json', checkpoints)
    save_json(out/'config.json', {'device': str(device), 'mode': args.mode, 'step': 4000,
              'protocol_sha256': sha256(BASE/'docs/exploration-r28-protocol.md'),
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in Path(__file__).parent.glob('*.py')},
              'cache_index_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in [CACHE_INDEX, DEV_INDEX, INDEX]},
              'deterministic_algorithms': True, 'cudnn_tf32_allowed': torch.backends.cudnn.allow_tf32,
              'scope': 'Fixed final checkpoints, same backend and same-weight plus external stronger post references; no checkpoint selection.'})
    metas = json.loads(CACHE_INDEX.read_text())['scenes']+json.loads(DEV_INDEX.read_text())['scenes']
    metas += [{**r, 'domain': 'dev_native'} for r in json.loads(INDEX.read_text())['scenes'] if r['domain'] == 'native_crop']
    assert len(metas) == 46
    count = inferences = computations = aliases = 0
    first, limits, events_audit = [], [], []
    start = time.perf_counter()
    for meta in metas:
        data = load_one(meta)
        events, _ = detect_changes(data['low'])
        events = event_flags(events)
        has_events = bool(events.any())
        events_audit.append({'domain': meta['domain'], 'scene': meta['scene'], 'reset_frames': np.flatnonzero(events).tolist()})
        inputs = torch.from_numpy((data['low']/255.).transpose(0, 3, 1, 2).copy()).to(device)
        for mode, objective, seed, net in models:
            spatial = mode in ['copy', EXTERNAL]
            histories = ['causal'] if spatial else ['causal', 'current']
            if not spatial:
                with torch.inference_mode():
                    delta = float((net.transform(inputs[:1])-net.backbone(inputs[:1])).abs().max())
                assert delta <= 2e-5
                limits.append({'domain': meta['domain'], 'scene': meta['scene'], 'mode': mode, 'objective': objective,
                               'train_seed': seed, 'current_mapping_max_abs_normalized': delta})
            first_reference = None
            for history in histories:
                predictions, metrics_cache, finite_cache = {}, {}, {}
                fusions = ['raw', 'half01', 'postwarp01', 'postwarp05'] if spatial or history == 'current' else ['raw', 'half01']
                for reset in [False, True]:
                    model_key, _ = output_keys(mode, history, reset, has_events, 'raw')
                    if model_key not in predictions:
                        values = infer(net, inputs, data, mode, history, events if model_key else None)
                        assert bool(torch.isfinite(values).all())
                        predictions[model_key] = np.ascontiguousarray(values.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()*255.)
                        inferences += 1
                    raw = predictions[model_key]
                    if first_reference is None:
                        first_reference = raw[0].copy()
                    delta = float(np.max(np.abs(raw[0]-first_reference)))
                    assert delta <= 1e-4
                    first.append({'domain': meta['domain'], 'scene': meta['scene'], 'mode': mode, 'objective': objective,
                                  'train_seed': seed, 'history': history, 'input_reset': reset, 'max_abs_pixel': delta})
                    for fusion in fusions:
                        _, key = output_keys(mode, history, reset, has_events, fusion)
                        identity = {'domain': meta['domain'], 'scene': meta['scene'], 'mode': mode, 'objective': objective,
                                    'train_seed': seed, 'step': 4000, 'history': history, 'input_reset': reset, 'fusion': fusion}
                        label = '|'.join(str(v) for v in identity.values())
                        if key not in metrics_cache:
                            if fusion == 'raw':
                                output = raw
                            elif fusion == 'half01':
                                output = fuse(raw, data['half_flow'], data['half_mask'], resets=events if key[2] else None, alpha=.1)
                            else:
                                if not finite_cache:
                                    warped = np.stack([raw[0], *[align_history(raw[t-1], raw[t], data['half_flow'][t-1], data['half_mask'][t-1])[0]
                                                               for t in range(1, len(raw))]])
                                    finite_cache = {'postwarp01': blend(raw, warped, .1), 'postwarp05': blend(raw, warped, .5)}
                                output = event_output(finite_cache[fusion], raw, events) if key[2] else finite_cache[fusion]
                            assert float(np.max(np.abs(output[0]-raw[0]))) <= 1e-4
                            metrics, frames = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                            metrics_cache[key] = metrics, frames, label
                            alias = None
                            computations += 1
                        else:
                            metrics, frames, alias = metrics_cache[key]
                            aliases += 1
                        annotation = {'metric_alias_of': alias, 'effective_model_reset': key[0], 'effective_output_reset': key[2]}
                        with (out/'per_sequence.jsonl').open('a') as stream:
                            stream.write(json.dumps({**identity, **annotation, **metrics}, allow_nan=False)+'\n')
                        with (out/'per_frame.jsonl').open('a') as stream:
                            for frame in frames:
                                stream.write(json.dumps({**identity, **annotation, **frame}, allow_nan=False)+'\n')
                        count += 1
        print(json.dumps({'domain': meta['domain'], 'scene': meta['scene'], 'rows': count, 'neural_sequences': inferences,
                          'metric_computations': computations, 'aliases': aliases}), flush=True)
    expected = (2208, 276, 1536, 672) if args.mode == 'copy' else (2208, 464, 1584, 624)
    assert (count, inferences, computations, aliases) == expected
    save_json(out/'first_frame_audit.json', first)
    save_json(out/'current_limit_audit.json', limits)
    save_json(out/'input_events.json', events_audit)
    save_json(out/'runtime.json', {'rows': count, 'actual_inference_sequences': inferences, 'metric_computations': computations,
              'metric_aliases': aliases, 'device': str(device), 'wall_seconds': time.perf_counter()-start,
              'scope': 'Includes CPU metrics/data/alias handling, not GPU kernel or independent mobile pipeline timing.'})


if __name__ == '__main__':
    main()
