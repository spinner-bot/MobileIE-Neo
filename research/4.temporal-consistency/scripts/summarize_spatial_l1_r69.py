"""Complete objective control with all earlier references, never prefix means."""
import argparse
from collections import defaultdict
from datetime import datetime
import json

import numpy as np

from check_spatial_l1_r69 import write
from dynamic_training_r34 import BASE, ROOT, digest

FIELDS = ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']


def lines(path): return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--perceptual', action='store_true'); args = parser.parse_args()
    sources = {}
    def read(path):
        sources[str(path.relative_to(ROOT))] = digest(path); return json.loads(path.read_text())
    def records(path):
        sources[str(path.relative_to(ROOT))] = digest(path); return lines(path)
    old = BASE/'report/exploration-r65-stablellve-task'
    assert read(old/'audit.json')['complete_paired_table']
    rows = records(old/'all_conditions.jsonl'); assert len(rows) == 220
    for seed in [0, 1]:
        train = BASE/f'results/exploration-r69-stage001-seed{seed}'
        run = BASE/f'results/exploration-r69-stage001-eval-seed{seed}'
        runtime = read(train/'runtime.json'); assert runtime['completed_normally'] and runtime['updates'] == 8000
        read(train/'config.json'); read(train/'step8000.json'); read(run/'source.json'); read(run/'config.json')
        runtime = read(run/'runtime.json')
        assert runtime['completed_normally'] and not runtime['cuda_initialized']
        assert (runtime['quality_rows'], runtime['frame_rows'], runtime['CNN_sequences'], runtime['CNN_centers']) == (20, 896, 10, 448)
        current = records(run/'per_sequence.jsonl'); assert len(current) == 20
        assert len({(r['scene'], r['method']) for r in current}) == 20
        assert len(records(run/'per_frame.jsonl')) == 896
        records(run/'prediction_manifest.jsonl')
        rows += [dict(**r, label='MobileIE_L1_'+r['method']) for r in current]
    assert len(rows) == 260
    perceptual = {}
    if args.perceptual:
        ref = BASE/'report/exploration-r66-perceptual-complete'
        audit = read(ref/'audit.json'); assert audit['completed'] and audit['perceptual_rows'] == 220 and audit['all_GT_controls_exact10']
        entries = records(ref/'all_conditions.jsonl')
        metric = BASE/'results/exploration-r69-perceptual-attempt001'
        runtime = read(metric/'runtime.json')
        assert runtime['completed_normally'] and (runtime['quality_rows'], runtime['comparison_frames']) == (40, 1792)
        read(metric/'config.json'); read(metric/'metric-source.json')
        entries += records(metric/'per_sequence.jsonl')
        assert len(entries) == 260
        for row in entries:
            family = 'StableLLVE_published' if row['family'] == 'StableLLVE_clean' else row['family']
            key = (row['domain'], row['seed'], family+'_'+row['method'], row['scene'])
            assert key not in perceptual; perceptual[key] = row['JOD']
        for row in rows:
            row['JOD'] = perceptual.get((row['domain'], row['seed'], row['label'], row['scene']))
        assert sum(r['JOD'] is not None for r in rows) == 240  # FD current diagnostic has no cached perceptual output.
    groups = defaultdict(list)
    for row in rows: groups[row['domain'], row['seed'], row['label']].append(row)
    means = []
    for (domain, seed, label), group in groups.items():
        assert len(group) == (6 if domain == 'davis_train_pilot' else 4)
        values = {name: float(np.mean([r[name] for r in group])) for name in FIELDS}
        if args.perceptual: values['JOD'] = None if any(r['JOD'] is None for r in group) else float(np.mean([r['JOD'] for r in group]))
        means.append(dict(domain=domain, seed=seed, label=label, parameters=group[0]['parameters'],
                          project_updates=group[0]['project_updates'], conditions=len(group), **values))
    assert len(means) == 52
    by_key = {(r['domain'], r['seed'], r['label']): r for r in means}
    effects = []
    for domain in ['davis_train_pilot', 'paired_real_ND_DEV']:
        for seed in [0, 1]:
            for method in ['raw', 'recursive_self05']:
                candidate = by_key[domain, seed, 'MobileIE_L1_'+method]
                reference = by_key[domain, seed, 'MobileIE_'+method]
                effect = dict(domain=domain, seed=seed, method=method, comparison='L1 minus existing same-backbone MSE',
                              PSNR_delta=candidate['psnr_rgb']-reference['psnr_rgb'],
                              SSIM_delta=candidate['ssim_rgb']-reference['ssim_rgb'],
                              Etemp_ratio=candidate['temporal_error']/reference['temporal_error'],
                              edge_ratio=candidate['edge_mae']/reference['edge_mae'])
                if args.perceptual: effect['JOD_delta'] = candidate['JOD']-reference['JOD']
                effects.append(effect)
    suffix = 'joint' if args.perceptual else 'quality'
    out = BASE/f'report/exploration-r69-spatial-L1-{suffix}'; out.mkdir(exist_ok=False)
    (out/'all_conditions.jsonl').write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows))
    write(out/'domain_seed_means.json', means); write(out/'objective_effects.json', effects); write(out/'sources.json', sources)
    table = ['# 完整原骨干 L1 对照', '', '全部最终8000；跨网络仍有预训练/容量/目标差异，非新方法/手机/最终测试。缺失JOD不补值、不重跑旧CNN。', '',
             '|域|seed|条件|参数|更新|PSNR|SSIM|Etemp|edge|JOD|', '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in means:
        jod = '—' if row.get('JOD') is None else f'{row["JOD"]:.6f}'
        table.append(f'|{row["domain"]}|{row["seed"]}|{row["label"]}|{row["parameters"]}|{row["project_updates"]}|'
                     f'{row["psnr_rgb"]:.6f}|{row["ssim_rgb"]:.6f}|{row["temporal_error"]:.6f}|{row["edge_mae"]:.6f}|{jod}|')
    (out/'tables.md').write_text('\n'.join(table)+'\n')
    write(out/'audit.json', dict(at=datetime.now().astimezone().isoformat(), completed=True, new_formal_updates=16000,
          new_quality_rows=40, new_frame_rows=1792, new_CNN_sequences=20, new_CNN_centers=896,
          all_quality_rows=260, domain_seed_means=52, objective_effect_rows=8,
          perceptual_included=args.perceptual, perceptual_quality_rows_joined=240 if args.perceptual else 0,
          unique_observations=448, new_method=False, phone=False, independent_final=False, old_acceptance_gates_rewritten=False))


if __name__ == '__main__': main()
