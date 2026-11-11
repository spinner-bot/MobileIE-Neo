"""Bounded frozen body-FST intervention on four native DEV sequences, common CPU controls."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from aggregation_controls_r20 import blend
from aligned_raw_history_r11 import align_history
from continuation_r14 import create_model
from exploration_r2_data import ROOT, evaluate
from explore_temporal_controls import save_json, sha256
from prepare_cache_r5 import load_one
from prepare_native_r10 import INDEX
from transport_fst_r27 import MODES, TransportFST

BASE = ROOT/'research/4.temporal-consistency'


@torch.inference_mode()
def stream(net, inputs, data, history):
    net.reset()
    values = []
    for t, current in enumerate(inputs):
        flow = None if t == 0 else data['half_flow'][t-1][None]
        mask = None if t == 0 else data['half_mask'][t-1][None]
        values.append(net.step(current[None], flow, mask, history)[0])
    return torch.stack(values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    assert json.loads((BASE/'process/exploration-r27/fixtures-cpu/audit.json').read_text())['passed']
    weights = [('official', 'official', -1, None)]
    for objective in ['charb', 'mse']:
        for seed in [0, 1]:
            marker = BASE/'results'/f'exploration-r26-seed{seed}'/('plain-'+objective)/'checkpoint-step4000.json'
            weights.append((objective+'-seed'+str(seed), objective, seed, json.loads(marker.read_text())))
    checkpoints = []
    for label, objective, seed, marker in weights:
        path = ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl' if marker is None else ROOT/marker['path']
        if marker is not None:
            assert sha256(path) == marker['sha256']
        checkpoints.append({'label': label, 'objective': objective, 'train_seed': seed,
                            'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)})
    save_json(out/'checkpoints.json', checkpoints)
    save_json(out/'config.json', {'device': 'cpu', 'threads': 2, 'deterministic_algorithms': True,
              'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in Path(__file__).parent.glob('*.py')},
              'protocol_sha256': sha256(BASE/'docs/exploration-r27-protocol.md'), 'native_index_sha256': sha256(INDEX),
              'scope': 'Frozen operator intervention, not training convergence or new-method evidence; common CPU metrics only.'})
    metas = [r for r in json.loads(INDEX.read_text())['scenes'] if r['domain'] == 'native_crop']
    assert len(metas) == 4
    count = computations = aliases = inferences = 0
    first, current_checks = [], []
    start = time.perf_counter()
    for meta in metas:
        data = load_one(meta)
        inputs = torch.from_numpy((data['low']/255.).transpose(0, 3, 1, 2).copy())
        assert len(inputs) == 16
        for label, objective, seed, marker in weights:
            original = create_model('plain', 'cpu').eval()
            if marker is not None:
                saved = torch.load(ROOT/marker['path'], map_location='cpu', weights_only=False)
                assert saved['step'] == 4000
                original.load_state_dict(saved['model'], strict=True)
            with torch.inference_mode():
                plain = torch.stack([original(frame[None])[0] for frame in inputs])
            inferences += 1
            original_rgb = np.ascontiguousarray(plain.clamp(0, 1).permute(0, 2, 3, 1).numpy()*255.)
            reference = None
            def retain(mode, history, fusion, output, alias=None, cached=None):
                nonlocal count, computations, aliases
                identity = {'domain': 'dev_native', 'scene': meta['scene'], 'weight': label, 'objective': objective,
                            'train_seed': seed, 'mode': mode, 'history': history, 'fusion': fusion}
                name = '|'.join(str(v) for v in identity.values())
                if cached is None:
                    metrics, frames = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                    computations += 1
                else:
                    metrics, frames = cached
                    aliases += 1
                with (out/'per_sequence.jsonl').open('a') as f:
                    f.write(json.dumps({**identity, 'metric_alias_of': alias, **metrics}, allow_nan=False)+'\n')
                with (out/'per_frame.jsonl').open('a') as f:
                    for frame in frames:
                        f.write(json.dumps({**identity, 'metric_alias_of': alias, **frame}, allow_nan=False)+'\n')
                count += 1
                return name, (metrics, frames)
            for mode in MODES:
                net = TransportFST(copy.deepcopy(original.backbone), mode).eval()
                causal = stream(net, inputs, data, 'causal')
                inferences += 1
                first_error = float((causal[0]-plain[0]).abs().max())
                assert first_error <= 2e-5
                first.append({'scene': meta['scene'], 'weight': label, 'mode': mode, 'max_abs_normalized': first_error})
                rgb = np.ascontiguousarray(causal.clamp(0, 1).permute(0, 2, 3, 1).numpy()*255.)
                if mode == 'copy':
                    assert torch.equal(causal, plain)
                    reference = retain(mode, 'causal', 'raw', rgb)
                    replaced = causal
                else:
                    retain(mode, 'causal', 'raw', rgb)
                    replaced = stream(net, inputs, data, 'current')
                    inferences += 1
                delta = float((replaced-plain).abs().max())
                assert delta <= 2e-5 and torch.equal(replaced, plain)
                current_checks.append({'scene': meta['scene'], 'weight': label, 'mode': mode, 'all_frames_exact': True, 'max_abs_normalized': delta})
                retain(mode, 'current', 'raw', original_rgb, alias=reference[0], cached=reference[1])
            warped = np.stack([original_rgb[0], *[align_history(original_rgb[t-1], original_rgb[t], data['half_flow'][t-1], data['half_mask'][t-1])[0]
                                                 for t in range(1, len(inputs))]])
            for name, alpha in [('postwarp01', .1), ('postwarp05', .5)]:
                retain('copy', 'causal', name, blend(original_rgb, warped, alpha))
        print(json.dumps({'scene': meta['scene'], 'rows': count, 'neural_sequences': inferences, 'metric_computations': computations, 'aliases': aliases}), flush=True)
    assert (count, inferences, computations, aliases) == (240, 200, 140, 100)
    save_json(out/'first_frame_audit.json', first)
    save_json(out/'current_limit_audit.json', current_checks)
    save_json(out/'runtime.json', {'rows': count, 'actual_inference_sequences': inferences, 'metric_computations': computations,
              'metric_aliases': aliases, 'wall_seconds': time.perf_counter()-start, 'device': 'cpu', 'training_updates': 0,
              'scope': 'Frozen native DEV exploratory mechanism probe; inference counts include explicit parity checks, not pipeline latency.'})


if __name__ == '__main__':
    main()
