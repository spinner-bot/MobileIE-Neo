"""All final published-loss conditions with explicit unequal-architecture references."""
import json
from collections import defaultdict
from datetime import datetime

import numpy as np

from check_stablellve_task_r65 import write
from dynamic_training_r34 import BASE, ROOT, digest
from stablellve_task_r65 import ARMS

FIELDS = ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']


def lines(path):
    return [json.loads(row) for row in path.read_text().splitlines()]


def main():
    rows, sources = [], []
    for seed in [0, 1]:
        train = BASE/f'results/exploration-r65-stage001-seed{seed}'
        run = BASE/f'results/exploration-r65-stage001-eval-seed{seed}'
        training, runtime = json.loads((train/'runtime.json').read_text()), json.loads((run/'runtime.json').read_text())
        assert training['completed_normally'] and training['updates'] == {arm: 8000 for arm in ARMS}
        assert runtime['completed_normally'] and not runtime['cuda_initialized']
        assert (runtime['quality_rows'], runtime['frame_rows'], runtime['CNN_sequences'], runtime['CNN_centers']) == (40, 1792, 20, 896)
        current = lines(run/'per_sequence.jsonl'); assert len(current) == 40 and len(lines(run/'per_frame.jsonl')) == 1792
        assert len({(r['arm'], r['scene'], r['method']) for r in current}) == 40
        rows += [dict(**r, label=f'StableLLVE_{r["arm"]}_{r["method"]}', seed=seed) for r in current]
        sources += [train/'runtime.json', train/'config.json', run/'runtime.json', run/'config.json',
                    run/'per_sequence.jsonl', run/'per_frame.jsonl', run/'prediction_manifest.jsonl']
        for arm in ARMS: sources += [train/f'{arm}-step8000.json', run/f'{arm}-source.json']
        reference_specs = [
            (f'exploration-r44-stage001-eval-seed{seed}', 'MobileIE', 4075,
             lambda r: r['model'] == 'square_mse' and r['method'] in ('raw', 'recursive_self05')),
            (f'exploration-r56-stage001-eval-seed{seed}', 'Nano', 696, lambda r: True),
            (f'exploration-r53-stage001-eval-seed{seed}', 'FastDVDnet', 2479096, lambda r: True)]
        for folder, label, parameters, select in reference_specs:
            directory = BASE/'results'/folder
            assert json.loads((directory/'runtime.json').read_text())['completed_normally']
            original = lines(directory/'per_sequence.jsonl')
            selected = [r for r in original if select(r)]; assert len(selected) == 20
            rows += [{**r, 'label': label+'_'+r['method'], 'seed': seed, 'project_updates': 8000,
                      'parameters': parameters, 'future_wait_max': 2 if label == 'FastDVDnet' and r['method'] == 'window' else 0}
                     for r in selected]
            sources += [directory/'runtime.json', directory/'config.json', directory/'per_sequence.jsonl']
    published = BASE/'results/exploration-r62-stablellve-common-attempt001'
    assert json.loads((published/'runtime.json').read_text())['completed_normally']
    original = lines(published/'per_sequence.jsonl'); assert len(original) == 20
    rows += [{**r, 'label': 'StableLLVE_published_'+r['method'], 'seed': None} for r in original]
    sources += [published/'runtime.json', published/'config.json', published/'per_sequence.jsonl']
    assert len(rows) == 220
    groups = defaultdict(list)
    for row in rows: groups[row['domain'], row['seed'], row['label']].append(row)
    summary = []
    for (domain, seed, label), group in groups.items():
        assert len(group) == (6 if domain == 'davis_train_pilot' else 4)
        summary.append(dict(domain=domain, seed=seed, label=label, source_sequences=len(group),
             parameters=group[0]['parameters'], project_updates=group[0]['project_updates'],
             future_wait_max=group[0].get('future_wait_max', 0),
             **{name: float(np.mean([r[name] for r in group])) for name in FIELDS}))
    assert len(summary) == 44
    by_key = {(r['domain'], r['seed'], r['label']): r for r in summary}
    effects = []
    for seed in [0, 1]:
        for domain in ['davis_train_pilot', 'paired_real_ND_DEV']:
            for method in ['raw', 'recursive_self05']:
                candidate = by_key[domain, seed, 'StableLLVE_commute20_'+method]
                reference = by_key[domain, seed, 'StableLLVE_spatial_l1_'+method]
                effects.append(dict(domain=domain, seed=seed, method=method,
                    comparison='commute20 relative to same-data/initialization/budget spatial_l1',
                    psnr_delta=candidate['psnr_rgb']-reference['psnr_rgb'],
                    ssim_delta=candidate['ssim_rgb']-reference['ssim_rgb'],
                    temporal_ratio=candidate['temporal_error']/reference['temporal_error'],
                    edge_ratio=candidate['edge_mae']/reference['edge_mae']))
    out = BASE/'report/exploration-r65-stablellve-task'; out.mkdir(exist_ok=False)
    (out/'all_conditions.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    write(out/'domain_seed_means.json', summary); write(out/'loss_effects.json', effects)
    table = ['# R65 完整固定预算参考', '',
        '本轮两损失臂内部同资料/权重/预算。跨架构参数、预训练、目标、计算不同；FastDVDnet window额外等待2帧。',
        '共同资料是反复使用的开发集，Etemp不是感知闪烁评分；不是原StableLLVE充分复现/新方法/最终测试或手机测量。', '',
        '|域|seed|条件|参数|项目更新|PSNR|SSIM|Etemp|edge|', '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        table.append(f'|{r["domain"]}|{r["seed"]}|{r["label"]}|{r["parameters"]}|{r["project_updates"]}|'
                     f'{r["psnr_rgb"]:.6f}|{r["ssim_rgb"]:.6f}|{r["temporal_error"]:.6f}|{r["edge_mae"]:.6f}|')
    (out/'tables.md').write_text('\n'.join(table)+'\n')
    write(out/'sources.json', {str(p.relative_to(ROOT)): digest(p) for p in sources})
    write(out/'audit.json', dict(at=datetime.now().astimezone().isoformat(), complete_paired_table=True,
          new_formal_training_updates=32000, new_quality_rows=80, new_frame_rows=3584,
          new_CNN_sequences=40, new_CNN_centers=1792, unique_observations=448,
          all_condition_rows=220, domain_seed_means=44, loss_effect_rows=8,
          new_method=False, full_original_paper_reproduction=False, independent_final=False,
          phone_cost_measured=False, actual_visual_review_still_required=True))


if __name__ == '__main__': main()
