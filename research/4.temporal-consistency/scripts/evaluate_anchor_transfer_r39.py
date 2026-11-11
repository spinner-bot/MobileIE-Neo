"""Matched CPU transfer on four already-used paired native DEV crops."""
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean
import subprocess
import time
import traceback

import cv2
import numpy as np
import torch

from aggregation_controls_r20 import blend
from aligned_raw_history_r11 import align_history
from dynamic_training_r34 import BASE, ROOT, digest
from exploration_r2_data import evaluate
from half_motion_r4 import half_forward_backward
from prepare_cache_r5 import load_one
from train_exploration_r2 import new_model
from transport_evaluation_r28 import load_models

METRICS = ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False)+'\n')


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    nets = [('official_LOL', 'official_LOL', None, new_model(torch.device('cpu')).eval())]
    previous, old_markers = load_models('copy', torch.device('cpu'))
    for mode, objective, seed, net in previous:
        if mode == 'copy' and objective == 'mse':
            nets.append((f'R26_seed{seed}', 'R26_plain_MSE', seed, net.backbone.eval()))
    markers = [r for r in old_markers if r['mode'] == 'copy' and r['objective'] == 'mse']
    for seed in [0, 1]:
        marker = json.loads((BASE/f'results/exploration-r34-seed{seed}/original-step4000.json').read_text())
        assert digest(ROOT/marker['path']) == marker['sha256']
        saved = torch.load(ROOT/marker['path'], map_location='cpu', weights_only=False)
        net = new_model(torch.device('cpu'))
        net.load_state_dict(saved['state_dict'], strict=True)
        nets.append((f'R34_seed{seed}', 'R34_original', seed, net.eval()))
        markers.append({'anchor': f'R34_seed{seed}', **marker})
    assert len(nets) == 5
    index = BASE/'results/exploration-r10-cache/index.json'
    metas = [r for r in json.loads(index.read_text())['scenes'] if r['domain'] == 'native_crop']
    assert {r['scene'] for r in metas} == {'video_15', 'video_47', 'video_91', 'video_121'}
    out = BASE/'results/exploration-r39-anchor-transfer'
    out.mkdir(parents=True, exist_ok=False)
    write(out/'config.json', {'device': 'cpu', 'torch_threads': 2, 'opencv_threads': 1,
          'anchor_ids': [r[0] for r in nets], 'checkpoints': markers,
          'official_weights_sha256': digest(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'),
          'input_index_sha256': digest(index), 'scenes': metas,
          'protocol_sha256': digest(BASE/'docs/exploration-r39-anchor-transfer-protocol.md'),
          'source_sha256': {name: digest(Path(__file__).parent/name) for name in
                           ['evaluate_anchor_transfer_r39.py', 'half_motion_r4.py', 'aligned_raw_history_r11.py',
                            'transport_evaluation_r28.py', 'exploration_r2_data.py', 'video_metrics.py']},
          'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'scope': 'Four reused paired real ND DEV crops, common CPU. No new model training, held-out final pixels, natural-night benchmark or phone latency.'})
    started, rows, neural, complete = time.perf_counter(), [], 0, False
    try:
        for meta in metas:
            data = load_one(meta)
            assert data['low'].shape == (16, 288, 512, 3)
            for anchor, family, seed, net in nets:
                with torch.inference_mode():
                    inputs = torch.from_numpy((data['low']/255).transpose(0, 3, 1, 2).copy())
                    raw = torch.stack([net(image[None])[0] for image in inputs])
                    assert torch.isfinite(raw).all()
                    raw = raw.clamp(0, 1).permute(0, 2, 3, 1).contiguous().numpy()*255
                neural += 1
                finite, recursive = [raw[0]], [raw[0]]
                for t in range(1, len(raw)):
                    flow, mask = half_forward_backward(raw[t-1], raw[t])
                    finite.append(blend(raw[t], align_history(raw[t-1], raw[t], flow, mask)[0], .5))
                    recursive.append(blend(raw[t], align_history(recursive[-1], raw[t], flow, mask)[0], .5))
                for method, output in [('raw', raw), ('finite_self05', np.stack(finite)),
                                        ('recursive_self05', np.stack(recursive))]:
                    metrics, frames = evaluate(output, data['gt'], data['reference_flow'], data['reference_mask'])
                    identity = {'scene': meta['scene'], 'anchor': anchor, 'anchor_family': family,
                                'train_seed': seed, 'method': method, 'domain': 'paired_real_ND_DEV'}
                    row = {**identity, **metrics}
                    append(out/'per_sequence.jsonl', row)
                    for frame in frames:
                        append(out/'per_frame.jsonl', {**identity, **frame})
                    rows.append(row)
                print(json.dumps({'scene': meta['scene'], 'anchor': anchor, 'rows': len(rows)}), flush=True)
        assert len(rows) == 60 and neural == 20 and not torch.cuda.is_initialized()
        groups, summaries = defaultdict(list), []
        for row in rows:
            groups[(row['anchor_family'], row['method'])].append(row)
        for (family, method), values in sorted(groups.items()):
            seeds = {r['train_seed'] for r in values}
            assert len(values) == (4 if family == 'official_LOL' else 8)
            summaries.append({'anchor_family': family, 'method': method, 'source_videos': 4,
                              'trained_seed_count': len(seeds), **{m: mean(r[m] for r in values) for m in METRICS}})
        lookup = {(r['anchor_family'], r['method'], r['scene'], r['train_seed']): r for r in rows}
        paired = []
        for row in rows:
            if row['anchor_family'] == 'R34_original':
                ref = lookup[('R26_plain_MSE', row['method'], row['scene'], row['train_seed'])]
                paired.append({'method': row['method'], 'scene': row['scene'], 'seed': row['train_seed'],
                               **{'delta_'+m: row[m]-ref[m] for m in METRICS}})
        report = BASE/'report/exploration-r39-anchor-transfer'
        report.mkdir(parents=True, exist_ok=False)
        write(report/'domain_summary.json', summaries)
        write(report/'paired_R34_vs_R26.json', paired)
        write(report/'audit.json', {'passed': True, 'rows': 60, 'neural_sequences': 20,
              'source_sha256': {str((out/name).relative_to(ROOT)): digest(out/name) for name in
                               ['config.json', 'per_sequence.jsonl']},
              'scope': 'Common CPU paired ND DEV transfer, different training schemes, no causal attribution to one training factor.'})
        lines = ['# Fixed anchor transfer on four paired real ND DEV crops', '',
                 'Common CPU2/OpenCV1, FP32 B1, same four already-used native crops/GT/correspondence. Official weights have one fixed instance; R26/R34 families each have two final4000 seeds. The data/optimization/batch differ, so the family difference is not an isolated architectural effect. No independent final test, natural-night benchmark or phone latency.', '',
                 '|Anchor family|Method|PSNR|SSIM|Temporal↓|Edge↓|', '|---|---|---:|---:|---:|---:|']
        for row in summaries:
            lines.append('|'+row['anchor_family']+'|'+row['method']+'|'+'|'.join(f'{row[m]:.6f}' for m in METRICS)+'|')
        lines += ['', 'All post routes estimate half-FB from their own adjacent raw; finite uses previous raw, recursive uses previous fused. No extra teacher or GT reset. Old CUDA numbers are not inserted into this CPU table. All scenes and both trained seeds remain in the paired JSON.']
        (report/'tables.md').write_text('\n'.join(lines)+'\n')
        complete = True
    except Exception:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json', {'completed_normally': complete, 'rows': len(rows), 'neural_sequences': neural,
                                  'cuda_initialized': torch.cuda.is_initialized(), 'wall_seconds': time.perf_counter()-started})


if __name__ == '__main__':
    main()
