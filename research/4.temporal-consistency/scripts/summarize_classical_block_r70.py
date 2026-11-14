"""Full traditional reference, including its offline/noise-estimation limits."""
import argparse
from collections import defaultdict
from datetime import datetime
import json
import numpy as np

from prepare_vbm3d_r70 import BASE, ROOT, digest, write

FIELDS = ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--perceptual', action='store_true'); args = parser.parse_args()
    sources = {}
    def read(path):
        sources[str(path.relative_to(ROOT))] = digest(path); return json.loads(path.read_text())
    def lines(path):
        sources[str(path.relative_to(ROOT))] = digest(path)
        return [json.loads(row) for row in path.read_text().splitlines()]
    run = BASE/'results/exploration-r70-classical-attempt001'
    runtime = read(run/'runtime.json')
    assert runtime['completed_normally'] and not runtime['CUDA_initialized'] and runtime['new_SGD'] == 0
    assert (runtime['filter_sequences'], runtime['filter_frames'], runtime['quality_rows'], runtime['frame_rows'],
            runtime['CNN_sequences'], runtime['CNN_centers']) == (20, 896, 80, 3584, 40, 1792)
    read(run/'config.json'); current = lines(run/'per_sequence.jsonl'); assert len(current) == 80
    assert len({(r['scene'], r['mode'], r['seed'], r['method']) for r in current}) == 80
    assert len(lines(run/'per_frame.jsonl')) == 3584 and len(lines(run/'filtered_manifest.jsonl')) == 20
    lines(run/'prediction_manifest.jsonl'); lines(run/'noise_estimates.jsonl')
    old = BASE/'report/exploration-r65-stablellve-task'
    assert read(old/'audit.json')['complete_paired_table']
    rows = lines(old/'all_conditions.jsonl'); assert len(rows) == 220
    rows += [dict(**r, label=r['family']+'_'+r['method']) for r in current]
    assert len(rows) == 300
    if args.perceptual:
        reference = BASE/'report/exploration-r66-perceptual-complete'
        audit = read(reference/'audit.json'); assert audit['completed'] and audit['all_GT_controls_exact10']
        entries = lines(reference/'all_conditions.jsonl'); assert len(entries) == 220
        metric = BASE/'results/exploration-r70-perceptual-attempt001'; runtime = read(metric/'runtime.json')
        assert runtime['completed_normally'] and (runtime['quality_rows'], runtime['comparison_frames']) == (80, 3584)
        read(metric/'config.json'); read(metric/'metric-source.json'); entries += lines(metric/'per_sequence.jsonl')
        lookup = {}
        for row in entries:
            family = 'StableLLVE_published' if row['family'] == 'StableLLVE_clean' else row['family']
            key = (row['domain'], row['seed'], family+'_'+row['method'], row['scene'])
            assert key not in lookup; lookup[key] = row['JOD']
        for row in rows: row['JOD'] = lookup.get((row['domain'], row['seed'], row['label'], row['scene']))
        assert sum(r['JOD'] is not None for r in rows) == 280
    groups = defaultdict(list)
    for row in rows: groups[row['domain'], row['seed'], row['label']].append(row)
    means = []
    for (domain, seed, label), group in groups.items():
        assert len(group) == (6 if domain == 'davis_train_pilot' else 4)
        values = {field: float(np.mean([r[field] for r in group])) for field in FIELDS}
        if args.perceptual: values['JOD'] = None if any(r['JOD'] is None for r in group) else float(np.mean([r['JOD'] for r in group]))
        means.append(dict(domain=domain, seed=seed, label=label, parameters=group[0]['parameters'],
                          project_updates=group[0]['project_updates'], conditions=len(group), **values))
    assert len(means) == 60
    lookup = {(r['domain'], r['seed'], r['label']): r for r in means}
    effects = []
    for domain in ['davis_train_pilot', 'paired_real_ND_DEV']:
        for seed in [0, 1]:
            for method in ['raw', 'recursive_self05']:
                for mode in ['spatial_only', 'video_default']:
                    pairs = [('versus unfiltered', mode, '')]
                    if mode == 'video_default': pairs.append(('video versus spatial patch search', mode, 'spatial_only'))
                    for comparison, candidate_mode, reference_mode in pairs:
                        candidate = lookup[domain, seed, 'MobileIE_'+candidate_mode+'_'+method]
                        ref_label = 'MobileIE_'+(reference_mode+'_' if reference_mode else '')+method
                        reference = lookup[domain, seed, ref_label]
                        effect = dict(domain=domain, seed=seed, mode=mode, method=method, comparison=comparison,
                                      PSNR_delta=candidate['psnr_rgb']-reference['psnr_rgb'],
                                      SSIM_delta=candidate['ssim_rgb']-reference['ssim_rgb'],
                                      Etemp_ratio=candidate['temporal_error']/reference['temporal_error'],
                                      edge_ratio=candidate['edge_mae']/reference['edge_mae'])
                        if args.perceptual: effect['JOD_delta'] = candidate['JOD']-reference['JOD']
                        effects.append(effect)
    assert len(effects) == 24
    suffix = 'joint' if args.perceptual else 'quality'
    out = BASE/f'report/exploration-r70-classical-{suffix}'; out.mkdir(exist_ok=False)
    (out/'all_conditions.jsonl').write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows))
    write(out/'domain_seed_means.json', means); write(out/'filter_effects.json', effects); write(out/'sources.json', sources)
    table = ['# 完整固定传统块匹配参考', '',
             'VBM spatial_only是同实现关闭时间搜索，非最佳BM3D复现；video_default离线全片，Nf不是端到端未来界。',
             'sigma只用首帧盲估计，异方差/clip/真实ISP不符合已知AWGN；不能据此否定所有经典去噪或声称原论文充分复现。', '',
             '|域|seed|条件|参数|更新|PSNR|SSIM|Etemp|edge|JOD|', '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in means:
        jod = '—' if row.get('JOD') is None else f'{row["JOD"]:.6f}'
        table.append(f'|{row["domain"]}|{row["seed"]}|{row["label"]}|{row["parameters"]}|{row["project_updates"]}|'
                     f'{row["psnr_rgb"]:.6f}|{row["ssim_rgb"]:.6f}|{row["temporal_error"]:.6f}|{row["edge_mae"]:.6f}|{jod}|')
    (out/'tables.md').write_text('\n'.join(table)+'\n')
    write(out/'audit.json', dict(at=datetime.now().astimezone().isoformat(), completed=True, new_SGD=0,
          new_quality_rows=80, new_frame_rows=3584, new_CNN_sequences=40, new_CNN_centers=1792,
          filter_sequences=20, filter_frames=896, quality_rows=300, domain_seed_means=60,
          effect_rows=24, perceptual_included=args.perceptual, perceptual_quality_rows_joined=280 if args.perceptual else 0,
          unique_observations=448, new_method=False, phone=False, independent_final=False, old_acceptance_gates_rewritten=False))


if __name__ == '__main__': main()
