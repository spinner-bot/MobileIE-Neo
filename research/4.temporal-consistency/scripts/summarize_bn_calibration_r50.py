"""Report every frozen calibration control; keep prior temporal failures unchanged."""
import json
from datetime import datetime
from statistics import mean

from bn_calibration_r50 import MODELS
from dynamic_training_r34 import BASE, ROOT, digest

METHODS = ['raw', 'recursive_self05']
DOMAINS = ['davis_train_pilot', 'paired_real_ND_DEV']


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def key(row):
    return row['train_seed'], row['model'], row['scene'], row['method']


def main():
    sources, before, after = [], {}, {}
    total_neural = total_rows = total_frames = 0
    for seed in [0, 1]:
        for family in ['r44', 'r48', 'r50']:
            folder = BASE/f'results/exploration-{family}-stage001-eval-seed{seed}'
            runtime = json.loads((folder/'runtime.json').read_text())
            assert runtime['completed_normally']
            path = folder/'per_sequence.jsonl'; sources.append(path)
            if family == 'r50':
                assert runtime['rows'] == 60 and runtime['neural_sequences'] == 30
                assert runtime['per_frame_rows'] == 2688 and not runtime['cuda_initialized']
                total_rows += runtime['rows']; total_neural += runtime['neural_sequences']
                total_frames += runtime['per_frame_rows']
            for row in lines(path):
                if row['method'] not in METHODS: continue
                if family == 'r44' and row['model'] != 'square_mse': continue
                if family == 'r48' and row['model'] not in ['plain', 'joint']: continue
                dest = after if family == 'r50' else before
                assert key(row) not in dest
                dest[key(row)] = row
    assert len(before) == len(after) == 120 and set(before) == set(after)
    rows, domain_seed, domain_mean, gaps = [], [], [], []
    for k, row in after.items():
        ref = before[k]
        rows.append({**row, 'before_psnr_rgb': ref['psnr_rgb'],
                     'before_temporal_error': ref['temporal_error'],
                     'delta_psnr': row['psnr_rgb']-ref['psnr_rgb'],
                     'time_ratio': row['temporal_error']/ref['temporal_error']})
    for name in MODELS:
        for method in METHODS:
            for domain in DOMAINS:
                scenes = sorted({r['scene'] for r in rows if r['domain'] == domain})
                assert len(scenes) == (6 if domain == DOMAINS[0] else 4)
                for seed in [0, 1]:
                    selected = [r for r in rows if (r['model'], r['method'], r['domain'], r['train_seed']) ==
                                (name, method, domain, seed)]
                    assert len(selected) == len(scenes)
                    domain_seed.append({'model': name, 'method': method, 'domain': domain, 'seed': seed,
                         'before_psnr': mean(r['before_psnr_rgb'] for r in selected),
                         'after_psnr': mean(r['psnr_rgb'] for r in selected),
                         'delta_psnr': mean(r['delta_psnr'] for r in selected),
                         'before_time': mean(r['before_temporal_error'] for r in selected),
                         'after_time': mean(r['temporal_error'] for r in selected),
                         'time_ratio': mean(r['temporal_error'] for r in selected)/mean(r['before_temporal_error'] for r in selected),
                         'worst_condition_delta_psnr': min(r['delta_psnr'] for r in selected)})
                selected = [r for r in domain_seed if (r['model'], r['method'], r['domain']) == (name, method, domain)]
                domain_mean.append({'model': name, 'method': method, 'domain': domain,
                    **{field: mean(r[field] for r in selected) for field in
                       ['before_psnr', 'after_psnr', 'delta_psnr', 'before_time', 'after_time']}})
                old_gap = mean(abs(before[(0, name, scene, method)]['psnr_rgb']-
                                   before[(1, name, scene, method)]['psnr_rgb']) for scene in scenes)
                new_gap = mean(abs(after[(0, name, scene, method)]['psnr_rgb']-
                                   after[(1, name, scene, method)]['psnr_rgb']) for scene in scenes)
                gaps.append({'model': name, 'method': method, 'domain': domain,
                             'before_mean_absolute_seed_gap': old_gap, 'after_mean_absolute_seed_gap': new_gap,
                             'gap_ratio': new_gap/old_gap if old_gap else None})
    gates = {}
    for name in MODELS:
        def safe(method):
            ds = [r for r in domain_seed if r['model'] == name and r['method'] == method]
            dm = [r for r in domain_mean if r['model'] == name and r['method'] == method]
            return all(r['delta_psnr'] >= -.05 for r in dm) and all(
                r['delta_psnr'] >= -.20 and r['worst_condition_delta_psnr'] >= -.5 and r['time_ratio'] <= 1.02 for r in ds)
        nd = next(r for r in gaps if (r['model'], r['method'], r['domain']) == (name, 'raw', DOMAINS[1]))
        benefit = any(r['delta_psnr'] >= .2 for r in domain_mean if r['model'] == name and r['method'] == 'raw')
        stability = nd['gap_ratio'] is not None and nd['gap_ratio'] <= .75
        eligible = safe('raw') and benefit and stability
        gates[name] = {'raw_quality_time_guard': safe('raw'), 'ND_seed_gap_reduced_25pct': stability,
                       'domain_mean_gain_020dB': benefit, 'raw_reference_eligible': eligible,
                       'recursive_quality_time_guard': safe('recursive_self05'),
                       'recursive_reference_eligible': eligible and safe('recursive_self05')}
    out = BASE/'report/exploration-r50-bn-calibration'; out.mkdir(exist_ok=False)
    write(out/'all_conditions.json', rows); write(out/'domain_seed.json', domain_seed)
    write(out/'domain_mean.json', domain_mean); write(out/'seed_gaps.json', gaps)
    write(out/'audit.json', {'at': datetime.now().astimezone().isoformat(), 'processing_complete': True,
          'rows': total_rows, 'neural_sequences': total_neural, 'per_frame_rows': total_frames,
          'calibration_SGD_updates': 0, 'learned_parameters_unchanged': True, 'gates': gates,
          'changes_R48_gates': False, 'new_method_established': False, 'final_test_opened': False,
          'stop_local_calibration_grid': True})
    write(out/'sources.json', {str(p.relative_to(ROOT)): digest(p) for p in sources})
    text = ['# R50 完整固定权重BN校准控制', '',
            '全部六模型/120行/60CNN，0SGD；原R48门槛与止损不变。两seed差不是方差或显著性估计。', '',
            '|模型|读出|域|seed|原PSNR|校准PSNR|ΔdB|原时间误差|校准时间误差|比|最差条件ΔdB|',
            '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in domain_seed:
        text.append(f"|{r['model']}|{r['method']}|{r['domain']}|{r['seed']}|{r['before_psnr']:.6f}|{r['after_psnr']:.6f}|{r['delta_psnr']:.6f}|{r['before_time']:.6f}|{r['after_time']:.6f}|{r['time_ratio']:.6f}|{r['worst_condition_delta_psnr']:.6f}|")
    text += ['', '|模型|读出|域|原平均绝对seed差|校准后|比|', '|---|---|---|---:|---:|---:|']
    for r in gaps:
        text.append(f"|{r['model']}|{r['method']}|{r['domain']}|{r['before_mean_absolute_seed_gap']:.6f}|{r['after_mean_absolute_seed_gap']:.6f}|{r['gap_ratio']:.6f}|")
    text += ['', '```json', json.dumps(gates, indent=2, ensure_ascii=False), '```', '',
             '只有BN缓冲区变化；不能从本控制推断所有seed差异均由BN造成。已知校准方法不建立论文创新/未见夜间泛化/手机时延。停止样本量、momentum、域配比和seed局部网格。']
    (out/'tables.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(gates))


if __name__ == '__main__':
    main()
