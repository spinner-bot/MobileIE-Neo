"""Evaluate the fixed R30 references or one completed R28 seed on common CPU."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from aggregation_controls_r20 import blend, event_output
from aligned_raw_history_r11 import align_history
from evaluation_aliases_r14 import event_flags
from exploration_r2_data import ROOT, detect_changes, evaluate
from explore_temporal_controls import save_json, sha256
from prepare_cache_r5 import load_one
from train_exploration_r2 import new_model
from transport_evaluation_r28 import EXTERNAL, infer, load_models
from transport_fst_r27 import TransportFST

BASE = ROOT/'research/4.temporal-consistency'


def models(phase, device):
    if phase == 'references':
        nets, markers = load_models('copy', device)
        nets.append(('official', 'official', -1, new_model(device).eval()))
        markers.append({'mode': 'official', 'path': 'MobileIE/pretrain/lolv1_best_slim.pkl',
                        'sha256': sha256(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl')})
        return nets, markers
    seed = int(phase[-1])
    nets, markers = [], []
    for objective in ['charb', 'mse']:
        folder = BASE/'results'/f'exploration-r28-seed{seed}'/('post_transport-'+objective)
        runtime = json.loads((folder/'runtime.json').read_text())
        assert runtime['final_step'] == runtime['new_updates'] == 4000
        marker = json.loads((folder/'checkpoint-step4000.json').read_text())
        assert sha256(ROOT/marker['path']) == marker['sha256']
        saved = torch.load(ROOT/marker['path'], map_location=device, weights_only=False)
        assert saved['step'] == 4000
        net = TransportFST(new_model(device), 'post_transport').eval()
        net.load_state_dict(saved['model'], strict=True)
        nets.append(('post_transport', objective, seed, net))
        markers.append({'mode': 'post_transport', 'objective': objective, 'train_seed': seed, **marker})
    return nets, markers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['references', 'post-seed0', 'post-seed1'], required=True)
    args = parser.parse_args()
    out = BASE/'results'/('exploration-r30-eval-'+args.phase)
    index = BASE/'results/exploration-r30-cache/index.json'
    metas = json.loads(index.read_text())['scenes']
    assert len(metas) == 6
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device('cpu')
    nets, markers = models(args.phase, device)
    marker_lookup = {(r['mode'], r.get('objective'), r.get('train_seed')): r for r in markers}
    out.mkdir(parents=True, exist_ok=False)
    save_json(out/'checkpoints.json', markers)
    save_json(out/'config.json', {'phase': args.phase, 'device': 'cpu', 'torch_threads': 2,
              'opencv_threads': 1, 'index_sha256': sha256(index), 'torch_version': torch.__version__,
              'protocol_sha256': sha256(BASE/'docs/exploration-r30-protocol.md'),
              'scope': 'Registered external development conditions; no checkpoint, loss, output or scene selection.'})
    count = inferences = aliases = 0
    first, events_audit = [], []
    start = time.perf_counter()
    for meta in metas:
        data = load_one(meta)
        events = event_flags(detect_changes(data['low'])[0])
        events_audit.append({'scene': meta['scene'], 'reset_frames': np.flatnonzero(events).tolist()})
        inputs = torch.from_numpy((data['low']/255.).transpose(0, 3, 1, 2).copy())
        for mode, objective, seed, net in nets:
            temporal = mode == 'post_transport'
            predictions, metrics_cache, finite = {}, {}, {}
            histories = ['current', 'causal'] if temporal else ['current']
            reference = None
            for history in histories:
                fusions = ['raw', 'postwarp05'] if history == 'current' else ['raw']
                for reset in [False, True]:
                    effective_reset = bool(reset and events.any())
                    model_key = history, bool(temporal and history == 'causal' and effective_reset)
                    if model_key not in predictions:
                        if history == 'current':
                            backbone = net.backbone if mode not in ['official', EXTERNAL] else net
                            with torch.inference_mode():
                                values = torch.stack([backbone(frame[None])[0] for frame in inputs])
                        else:
                            values = infer(net, inputs, data, mode, history, events if model_key[1] else None)
                        assert bool(torch.isfinite(values).all())
                        predictions[model_key] = np.ascontiguousarray(values.clamp(0, 1).permute(0, 2, 3, 1).numpy()*255.)
                        if mode == 'copy':
                            # Preserve strong-base RGB once for later published denoiser transfer.
                            target = ROOT/'.cache/experiments/exploration-r30-predictions'/f'plain-{objective}-seed{seed}'/(meta['scene']+'.npy')
                            assert target.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo') and not target.exists()
                            target.parent.mkdir(parents=True, exist_ok=True)
                            np.save(target, predictions[model_key], allow_pickle=False)
                            with (out/'prediction_manifest.jsonl').open('a') as stream:
                                stream.write(json.dumps({'scene': meta['scene'], 'objective': objective, 'train_seed': seed,
                                             'input_cache_sha256': meta['sha256'], 'checkpoint_sha256': marker_lookup[(mode, objective, seed)]['sha256'],
                                             'path': str(target.relative_to(ROOT)), 'sha256': sha256(target)})+'\n')
                        inferences += 1
                    raw = predictions[model_key]
                    if reference is None:
                        reference = raw[0].copy()
                    delta = float(np.max(np.abs(reference-raw[0])))
                    assert delta <= 1e-4
                    first.append({'scene': meta['scene'], 'mode': mode, 'objective': objective, 'train_seed': seed,
                                  'history': history, 'reset': reset, 'max_abs_first_frame': delta})
                    for fusion in fusions:
                        identity = {'domain': meta['domain'], 'scene': meta['scene'], 'source_video': meta['source_video'],
                                    'mode': mode, 'objective': objective, 'train_seed': seed, 'history': history,
                                    'input_reset': reset, 'fusion': fusion}
                        output_key = model_key, fusion, bool(effective_reset and fusion != 'raw')
                        label = json.dumps(identity, sort_keys=True)
                        if output_key not in metrics_cache:
                            output = raw
                            if fusion == 'postwarp05':
                                if history not in finite:
                                    warped = np.stack([raw[0], *[align_history(raw[t-1], raw[t], data['half_flow'][t-1], data['half_mask'][t-1])[0]
                                                               for t in range(1, len(raw))]])
                                    finite[history] = blend(raw, warped, .5)
                                output = event_output(finite[history], raw, events) if output_key[2] else finite[history]
                            metrics, per_frame = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                            metrics_cache[output_key] = metrics, per_frame, label
                            alias = None
                        else:
                            metrics, per_frame, alias = metrics_cache[output_key]
                            aliases += 1
                        with (out/'per_sequence.jsonl').open('a') as stream:
                            stream.write(json.dumps({**identity, 'metric_alias_of': alias, **metrics}, allow_nan=False)+'\n')
                        with (out/'per_frame.jsonl').open('a') as stream:
                            for frame in per_frame:
                                stream.write(json.dumps({**identity, 'metric_alias_of': alias, **frame}, allow_nan=False)+'\n')
                        count += 1
            print(json.dumps({'scene': meta['scene'], 'mode': mode, 'objective': objective, 'seed': seed,
                              'rows': count, 'neural_sequences': inferences}), flush=True)
    assert count == (168 if args.phase == 'references' else 72)
    assert not torch.cuda.is_initialized()
    save_json(out/'first_frame_audit.json', first)
    save_json(out/'input_events.json', events_audit)
    save_json(out/'runtime.json', {'completed_normally': True, 'rows': count, 'neural_sequences': inferences,
              'metric_aliases': aliases, 'device': 'cpu', 'cuda_initialized': False, 'wall_seconds': time.perf_counter()-start})


if __name__ == '__main__':
    main()
