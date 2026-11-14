"""Complete uncorrected raw DC-error comparison; reads cached predictions, no CNN."""
import json
from collections import defaultdict
from datetime import datetime
from statistics import mean

import numpy as np

from dynamic_training_r34 import BASE, ROOT, digest
from prepare_cache_r5 import load_one


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    out = BASE/'report/exploration-r50-bn-calibration'
    assert json.loads((out/'audit.json').read_text())['processing_complete']
    assert not (out/'raw_bias_comparison.json').exists()
    before, frames, sources = [], [], []
    for seed in [0, 1]:
        path = BASE/f'results/exploration-r50-stage001-eval-seed{seed}/raw_bias_per_frame.jsonl'
        sources.append(path)
        frames.extend({**r, 'state': 'calibrated'} for r in lines(path))
        for family in ['r44', 'r48']:
            path = BASE/f'results/exploration-{family}-stage001-eval-seed{seed}/prediction_manifest.jsonl'
            sources.append(path)
            for row in lines(path):
                if row['model'] in (['square_mse'] if family == 'r44' else ['plain', 'joint']):
                    before.append(row)
    assert len(before) == 60 and len(frames) == 2688
    p1 = BASE/'results/exploration-r30-cache/index.json'
    p2 = BASE/'results/exploration-r10-cache/index.json'; sources.extend([p1, p2])
    metas = json.loads(p1.read_text())['scenes']
    metas += [{**r, 'domain': 'paired_real_ND_DEV', 'source_video': r['scene']}
              for r in json.loads(p2.read_text())['scenes'] if r['domain'] == 'native_crop']
    for meta in metas:
        data = load_one(meta)
        selected = [r for r in before if r['scene'] == meta['scene']]; assert len(selected) == 6
        for row in selected:
            path = ROOT/row['path']
            assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
            prediction = np.load(path, mmap_mode='r', allow_pickle=False)
            assert prediction.shape == data['gt'].shape
            for t in range(len(prediction)):
                error = prediction[t].astype(np.float64)-data['gt'][t].astype(np.float64)
                bias = error.mean(axis=(0, 1)); mse = float(np.mean(error**2))
                dc = float(np.mean(bias**2)); centered = float(np.mean((error-bias)**2))
                assert abs(mse-dc-centered) <= 1e-10*max(1., mse)
                frames.append({'domain': meta['domain'], 'scene': meta['scene'], 'train_seed': row['train_seed'],
                    'model': row['model'], 'frame': t, 'state': 'before', 'MSE': mse,
                    'DC_MSE': dc, 'centered_MSE': centered, 'mean_RGB_error_0_to_255': bias.tolist()})
    assert len(frames) == 5376
    grouped = defaultdict(list)
    for row in frames:
        grouped[(row['domain'], row['scene'], row['train_seed'], row['model'], row['state'])].append(row)
    rows = []
    for (domain, scene, seed, name, state), values in sorted(grouped.items()):
        mse, dc, centered = [mean(r[field] for r in values) for field in ['MSE', 'DC_MSE', 'centered_MSE']]
        rows.append({'domain': domain, 'scene': scene, 'seed': seed, 'model': name, 'state': state,
                     'frames': len(values), 'mean_MSE': mse, 'mean_DC_MSE': dc,
                     'mean_centered_MSE': centered, 'DC_fraction_of_MSE': dc/mse})
    assert len(rows) == 120
    (out/'raw_bias_comparison.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n')
    with (out/'raw_bias_before_after_per_frame.jsonl').open('x') as stream:
        for row in frames: stream.write(json.dumps(row, allow_nan=False)+'\n')
    (out/'raw_bias_sources.json').write_text(json.dumps({'at': datetime.now().astimezone().isoformat(),
        'sources': {str(p.relative_to(ROOT)): digest(p) for p in sources},
        'original_prediction_records': before, 'new_neural_sequences': 0,
        'output_correction': False, 'changes_primary_gates': False,
        'scope': 'All ten DEV conditions, two seeds, three families, before/after raw predictions; no GT fitting.'}, indent=2)+'\n')
    text = ['# R50 全条件raw空间误差分解', '',
            '未修正输出、未新增CNN、未改门槛；MSE=平均通道偏差²+中心化误差MSE。', '',
            '|域|源|seed|模型|状态|MSE|DC MSE|中心化MSE|DC占比|', '|---|---|---:|---|---|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"|{r['domain']}|{r['scene']}|{r['seed']}|{r['model']}|{r['state']}|{r['mean_MSE']:.6f}|{r['mean_DC_MSE']:.6f}|{r['mean_centered_MSE']:.6f}|{r['DC_fraction_of_MSE']:.6f}|")
    (out/'raw_bias_tables.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'sequence_rows': len(rows), 'frame_rows': len(frames), 'new_CNN': 0}))


if __name__ == '__main__':
    main()
