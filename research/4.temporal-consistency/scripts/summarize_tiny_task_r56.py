"""All final adapted seeds, published initialization and fixed strong references."""
import json
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dynamic_training_r34 import BASE, ROOT, digest


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    out = BASE/'report/exploration-r56-tiny-task-reference'; out.mkdir(exist_ok=False)
    rows, phones, sources = [], [], []
    for seed in [0, 1]:
        train = BASE/f'results/exploration-r56-stage001-seed{seed}'
        run = BASE/f'results/exploration-r56-stage001-eval-seed{seed}'
        training, runtime = json.loads((train/'runtime.json').read_text()), json.loads((run/'runtime.json').read_text())
        assert training['completed_normally'] and training['steps'] == 8000
        assert runtime['completed_normally'] and runtime['quality_rows'] == 20 and runtime['phone_rows'] == 6
        assert runtime['neural_sequences'] == 13 and runtime['neural_frames'] == 1400 and not runtime['cuda_initialized']
        assert len(lines(run/'per_frame.jsonl')) == 896 and len(lines(run/'phone_per_frame.jsonl')) == 1904
        rows += [{**r, 'method': 'Nano_adapted_'+r['method'], 'seed': f'seed{seed}'} for r in lines(run/'per_sequence.jsonl')]
        phones += [{**r, 'label': f'Nano_adapted_seed{seed}'} for r in lines(run/'phone_per_sequence.jsonl')]
        path = BASE/f'results/exploration-r45-stage001-eval-seed{seed}/per_sequence.jsonl'
        old = lines(path); assert len(old) == 90
        rows += [{**r, 'method': 'MobileIE_'+r['method'], 'seed': f'seed{seed}'} for r in old if r['method'] in ['raw', 'fixed_05', 'rgb']]
        phone_ref = BASE/f'results/exploration-r54-full-phone-seed{seed}/per_sequence.jsonl'
        phones += [{**r, 'label': f'MobileIE_seed{seed}'} for r in lines(phone_ref)]
        sources += [train/'runtime.json', train/'config.json', train/'nano-step8000.json',
                    run/'config.json', run/'runtime.json', run/'per_sequence.jsonl', run/'phone_per_sequence.jsonl',
                    run/'prediction_manifest.jsonl', path, phone_ref]
    original = BASE/'results/exploration-r55-stage001'
    assert json.loads((original/'runtime.json').read_text())['completed_normally']
    rows += [{**r, 'method': 'Nano_published_'+r['method'], 'seed': 'published'} for r in lines(original/'per_sequence.jsonl')]
    phones += [{**r, 'label': 'Nano_published'} for r in lines(original/'phone_per_sequence.jsonl')]
    sources += [original/'config.json', original/'runtime.json', original/'per_sequence.jsonl', original/'phone_per_sequence.jsonl']
    assert len(rows) == 120 and len(phones) == 36
    groups = defaultdict(list)
    for r in rows: groups[r['domain'], r['seed'], r['method']].append(r)
    fields = ['psnr_rgb', 'ssim_rgb', 'edge_mae', 'temporal_error']
    means = {key: {k: float(np.mean([r[k] for r in group])) for k in fields} for key, group in groups.items()}
    summary = [dict(domain=d, seed=s, method=m, parameters=696 if m.startswith('Nano') else 4116 if m.endswith('_rgb') else 4075, **v)
               for (d, s, m), v in means.items()]
    effects = []
    for seed in [0, 1]:
        for domain in ['davis_train_pilot', 'paired_real_ND_DEV']:
            for method, reference in [('raw', 'raw'), ('recursive_self05', 'recursive_self05')]:
                value, ref = means[domain, f'seed{seed}', 'Nano_adapted_'+method], means[domain, 'published', 'Nano_published_'+reference]
                effects.append(dict(domain=domain, seed=seed, method=method,
                                    psnr_delta_vs_published=value['psnr_rgb']-ref['psnr_rgb'],
                                    temporal_ratio_vs_published=value['temporal_error']/ref['temporal_error']))
    write(out/'paired_conditions.json', rows); write(out/'domain_seed_summary.json', summary)
    write(out/'adaptation_effects.json', effects); write(out/'phone_descriptive_summary.json', phones)
    table = ['# R56 固定8000更新的微型公开参考：全部结果', '',
             '原Nano696参数/BN保持，两个seed各8000新更新，原MixedPool/四目标/raw MSE/Adam1e-4/4000重置。',
             '同目标/更新数不等于同预训练、MBR参数化、训练计算或充分收敛；这是既有方法的必要适配对照，不是新方法。',
             '新CPU40配对质量/1792逐帧、12手机描述/3808输出帧，26CNN序列/2800真实输入帧。全部结果保留，无最优seed/中间点。',
             '手机三已见DEV无GT，不能作质量/真闪烁排名。R53视频参考另按其完整固定链收齐，不以部分值参入。', '',
             '|域|权重/seed|方法|参数|PSNR|SSIM|GT变化时间误差↓|edge↓|',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        table.append(f"|{r['domain']}|{r['seed']}|{r['method']}|{r['parameters']}|{r['psnr_rgb']:.6f}|{r['ssim_rgb']:.6f}|{r['temporal_error']:.6f}|{r['edge_mae']:.6f}|")
    table += ['', '网络图/参数形状未变，Conv MAC可沿R55同图形状计数；解析先验/归一/pointwise/flow/warp/内存/IO未计，非手机速度。']
    (out/'tables.md').write_text('\n'.join(table)+'\n')
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    styles = [('Nano_published_raw', 'published', '*', 'tab:gray'), ('Nano_published_recursive_self05', 'published', '*', 'tab:orange'),
              ('Nano_adapted_raw', None, 's', 'tab:red'), ('Nano_adapted_recursive_self05', None, 'D', 'tab:purple'),
              ('MobileIE_raw', None, 'x', 'tab:blue'), ('MobileIE_fixed_05', None, 'o', 'tab:green'), ('MobileIE_rgb', None, '^', 'tab:brown')]
    for col, domain in enumerate(['davis_train_pilot', 'paired_real_ND_DEV']):
        for seed in [0, 1]:
            ax = axes[seed, col]
            for method, which, marker, color in styles:
                r = means[domain, which or f'seed{seed}', method]
                ax.scatter(r['temporal_error'], r['psnr_rgb'], marker=marker, color=color, s=80, label=method)
            ax.set_title(('SYN' if col == 0 else 'Paired ND DEV')+f' / seed{seed}')
            ax.set_xlabel('GT-change temporal error (lower is better)'); ax.set_ylabel('PSNR (dB)'); ax.grid(alpha=.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=8, frameon=False)
    fig.suptitle('Fixed-budget adapted micro-reference / reused DEV / unequal training parameterizations')
    fig.tight_layout(rect=[0, .14, 1, .95])
    for ext in ['png', 'pdf']: fig.savefig(out/f'paired-comparison.{ext}', dpi=150)
    plt.close(fig)
    write(out/'sources.json', {str(p.relative_to(ROOT)): digest(p) for p in sources})
    write(out/'audit.json', dict(at=datetime.now().astimezone().isoformat(), complete_paired_table=True,
          new_training_updates=16000, new_quality_rows=40, new_phone_rows=12, new_CNN_sequences=26,
          new_CNN_frames=2800, unique_source_frames=1400, comparison_rows=120, phone_description_rows=36,
          original_BN_kept=True, phone_previews_still_required=True, visual_review_still_required=True,
          new_method=False, independent_final=False, phone_cost_measured=False))


if __name__ == '__main__':
    main()
