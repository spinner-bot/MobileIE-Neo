"""Complete perceptual rows alongside original quality metrics; never rewrite old gates."""
import argparse
import json
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from cvvdp_adapter_r66 import BASE, ROOT
from dynamic_training_r34 import digest
from evaluate_cvvdp_references_r66 import lines, write

FIELDS = ['psnr_rgb', 'ssim_rgb', 'temporal_error', 'edge_mae']


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['references', 'complete'], required=True)
    args = parser.parse_args()
    rows, sources, info = [], [], []
    phases = [('reference-attempt002', 140, 6272)]
    if args.phase == 'complete': phases.append(('stablellve-attempt001', 80, 3584))
    for suffix, count, frame_count in phases:
        run = BASE/f'results/exploration-r66-{suffix}'
        runtime = json.loads((run/'runtime.json').read_text()); assert runtime['completed_normally']
        assert runtime['quality_rows'] == count and runtime['comparison_frames'] == frame_count
        current = lines(run/'per_sequence.jsonl'); assert len(current) == count
        assert len(lines(run/'feature_manifest.jsonl')) == count
        assert all(np.isfinite(r['JOD']) for r in current)
        rows += current; info.append(json.loads((run/'metric-source.json').read_text()))
        sources += [run/name for name in ['runtime.json', 'per_sequence.jsonl', 'feature_manifest.jsonl', 'config.json', 'metric-source.json']]
    assert all(r['metric_info'] == info[0]['metric_info'] and
               r['package_install_manifest_sha256'] == info[0]['package_install_manifest_sha256'] for r in info)
    expected = 140 if args.phase == 'references' else 220
    assert len(rows) == expected and len({(r['domain'], r['scene'], r['family'], r['seed'], r['method']) for r in rows}) == expected
    qualities = {}
    for seed in [0, 1]:
        for number, family in [(44, 'MobileIE'), (56, 'Nano'), (53, 'FastDVDnet')]:
            path = BASE/f'results/exploration-r{number}-stage001-eval-seed{seed}/per_sequence.jsonl'
            for r in lines(path):
                if number == 44 and not (r['model'] == 'square_mse' and r['method'] in ('raw', 'recursive_self05')): continue
                if number == 53 and r['method'] != 'window': continue
                qualities[r['domain'], r['scene'], family, seed, r['method']] = r
            sources.append(path)
        if args.phase == 'complete':
            path = BASE/f'results/exploration-r65-stage001-eval-seed{seed}/per_sequence.jsonl'
            for r in lines(path): qualities[r['domain'], r['scene'], 'StableLLVE_'+r['arm'], seed, r['method']] = r
            sources.append(path)
    path = BASE/'results/exploration-r62-stablellve-common-attempt001/per_sequence.jsonl'
    for r in lines(path): qualities[r['domain'], r['scene'], 'StableLLVE_clean', None, r['method']] = r
    sources.append(path)
    for row in rows:
        if row['family'] in ('Input', 'GT'):
            row.update({field: None for field in FIELDS})
        else:
            original = qualities[row['domain'], row['scene'], row['family'], row['seed'], row['method']]
            row.update({field: original[field] for field in FIELDS})
    groups = defaultdict(list)
    for row in rows: groups[row['domain'], row['family'], row['seed'], row['method']].append(row)
    means = []
    for (domain, family, seed, method), group in groups.items():
        assert len(group) == (6 if domain == 'davis_train_pilot' else 4)
        scores = {name: float(np.mean([r[name] for r in group])) if group[0][name] is not None else None
                  for name in ['JOD']+FIELDS}
        means.append(dict(domain=domain, family=family, seed=seed, method=method, source_sequences=len(group),
            parameters=group[0]['parameters'], project_updates=group[0]['project_updates'],
            future_wait_max=group[0]['future_wait_max'], **scores))
    assert len(means) == (28 if args.phase == 'references' else 44)
    lookup = {(r['domain'], r['family'], r['seed'], r['method']): r for r in means}
    effects = []
    for key, raw in lookup.items():
        domain, family, seed, method = key
        if method != 'raw': continue
        rec = lookup[domain, family, seed, 'recursive_self05']
        effects.append(dict(domain=domain, family=family, seed=seed, comparison='own recursive .5 relative to raw',
            JOD_gain=rec['JOD']-raw['JOD'], PSNR_delta=rec['psnr_rgb']-raw['psnr_rgb'],
            Etemp_ratio=rec['temporal_error']/raw['temporal_error'], edge_ratio=rec['edge_mae']/raw['edge_mae']))
    loss_effects = []
    if args.phase == 'complete':
        for domain in ['davis_train_pilot', 'paired_real_ND_DEV']:
            for seed in [0, 1]:
                for method in ['raw', 'recursive_self05']:
                    current = lookup[domain, 'StableLLVE_commute20', seed, method]
                    reference = lookup[domain, 'StableLLVE_spatial_l1', seed, method]
                    loss_effects.append(dict(domain=domain, seed=seed, method=method,
                        JOD_gain=current['JOD']-reference['JOD'], PSNR_delta=current['psnr_rgb']-reference['psnr_rgb'],
                        Etemp_ratio=current['temporal_error']/reference['temporal_error']))
    folder = BASE/f'report/exploration-r66-perceptual-{args.phase}'; folder.mkdir(exist_ok=False)
    (folder/'all_conditions.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows))
    write(folder/'domain_seed_means.json', means); write(folder/'recursive_effects.json', effects)
    write(folder/'loss_effects.json', loss_effects)
    table = ['# 固定显示条件的感知补评', '', info[0]['metric_info'], '',
        'Nominal30fps、原尺寸SDR、replicate padding；是显示假设下的预测，非实测手机/采集帧率、人类MOS或专门闪烁分数。',
        '跨架构预训练/容量/目标/计算不等，FastDVDnet window等待2帧。Input/GT仅指标控制，无新增增强推理。', '',
        '|域|模型|seed|输出|参数|项目更新|JOD↑|PSNR|SSIM|Etemp↓|edge↓|',
        '|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in means:
        fields = ['—' if r[k] is None else f'{r[k]:.6f}' for k in ['JOD']+FIELDS]
        table.append(f'|{r["domain"]}|{r["family"]}|{r["seed"]}|{r["method"]}|{r["parameters"]}|{r["project_updates"]}|'+ '|'.join(fields)+'|')
    (folder/'tables.md').write_text('\n'.join(table)+'\n')
    colors = dict(MobileIE='tab:blue', Nano='tab:orange', FastDVDnet='tab:green', StableLLVE_clean='tab:gray',
                  StableLLVE_spatial_l1='tab:red', StableLLVE_commute20='tab:purple')
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for col, domain in enumerate(['davis_train_pilot', 'paired_real_ND_DEV']):
        for seed in [0, 1]:
            ax = axes[seed, col]
            for r in means:
                if r['domain'] != domain or r['seed'] not in (None, seed) or r['temporal_error'] is None: continue
                ax.scatter(r['temporal_error'], r['JOD'], color=colors[r['family']], s=70,
                    marker='D' if r['method'] == 'recursive_self05' else '^' if r['method'] == 'window' else 'o',
                    label=r['family']+' '+r['method'])
            ax.set_title(('SYN' if col == 0 else 'Paired ND DEV')+f' / seed{seed}')
            ax.set_xlabel('GT-change error (lower)'); ax.set_ylabel('ColorVideoVDP JOD (higher)'); ax.grid(alpha=.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8, frameon=False)
    fig.suptitle('Nominal FHD / 30fps / reused DEV / unequal training conditions')
    fig.tight_layout(rect=[0, .18 if args.phase == 'complete' else .13, 1, .95])
    for ext in ['png', 'pdf']: fig.savefig(folder/f'perceptual-comparison.{ext}', dpi=150)
    plt.close(fig)
    write(folder/'sources.json', {str(p.relative_to(ROOT)): digest(p) for p in sources})
    write(folder/'audit.json', dict(at=datetime.now().astimezone().isoformat(), completed=True, phase=args.phase,
        perceptual_rows=expected, domain_seed_means=len(means), old_quality_rows_joined=expected-20,
        comparison_frames=sum(r['frames'] for r in rows), unique_observations=448,
        all_GT_controls_exact10=all(r['JOD'] == 10 for r in rows if r['family'] == 'GT'),
        new_enhancement_CNN=0, new_SGD=0, old_acceptance_gates_rewritten=False,
        human_MOS=False, actual_phone=False, independent_final=False, actual_plot_review_still_required=True))


if __name__ == '__main__': main()
