"""Complete-only tiny transfer reference table with both existing MobileIE seeds."""
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
    run = BASE/'results/exploration-r55-stage001'
    runtime = json.loads((run/'runtime.json').read_text())
    assert runtime['completed_normally'] and runtime['quality_rows'] == 20 and runtime['phone_rows'] == 6
    assert runtime['neural_sequences'] == 13 and runtime['neural_frames'] == 1400 and not runtime['cuda_initialized']
    assert len(lines(run/'per_frame.jsonl')) == 896 and len(lines(run/'phone_per_frame.jsonl')) == 1904
    config = json.loads((run/'config.json').read_text()); parameters = config['source']['parameters']
    rows = [{**r, 'method': 'Nano_'+r['method'], 'seed': 'published'} for r in lines(run/'per_sequence.jsonl')]
    sources = [run/name for name in ['runtime.json', 'config.json', 'per_sequence.jsonl', 'per_frame.jsonl',
               'phone_per_sequence.jsonl', 'phone_per_frame.jsonl', 'prediction_manifest.jsonl']]
    for seed in [0, 1]:
        path = BASE/f'results/exploration-r45-stage001-eval-seed{seed}/per_sequence.jsonl'
        old = lines(path); assert len(old) == 90
        rows += [{**r, 'method': 'MobileIE_'+r['method'], 'seed': f'seed{seed}'} for r in old
                 if r['method'] in ['raw', 'fixed_05', 'rgb']]
        sources.append(path)
    assert len(rows) == 80
    out = BASE/'report/exploration-r55-tiny-reference'; out.mkdir(exist_ok=False)
    groups = defaultdict(list)
    for r in rows: groups[r['domain'], r['seed'], r['method']].append(r)
    fields = ['psnr_rgb', 'ssim_rgb', 'edge_mae', 'temporal_error']
    summary = [dict(domain=d, seed=s, method=m, parameters=group[0]['parameters'],
                    **{k: float(np.mean([r[k] for r in group])) for k in fields})
               for (d, s, m), group in groups.items()]
    write(out/'paired_conditions.json', rows); write(out/'domain_seed_summary.json', summary)
    write(out/'phone_descriptive_summary.json', lines(run/'phone_per_sequence.jsonl'))
    cost_path = BASE/'results/exploration-r55-conv-budget001/cost.json'
    cost = json.loads(cost_path.read_text()); sources.append(cost_path)
    table = ['# 固定696参数公开空间参考：完整迁移结果', '',
             f'实际模型{parameters}参数，原LOL-v1单一公开checkpoint，无项目训练/虚构seed。',
             '20配对质量行/896逐帧；另6手机描述行/1904输出帧。全13CNN序列/1400真实输入帧，人工门槛另6张。',
             'MobileIE为R44 square_mse8000及R45 RGB4000固定参考。预训练/数据/预算不同；不是同任务同预算架构因果、独立最终或作者原表复现。',
             '不GT-Mean/自集成；时间列为GT变化残差而非时延；所有胜负保留。', '',
             '|域|权重/seed|方法|参数|PSNR|SSIM|时间误差↓|edge↓|',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        table.append(f"|{r['domain']}|{r['seed']}|{r['method']}|{r['parameters']}|{r['psnr_rgb']:.6f}|{r['ssim_rgb']:.6f}|{r['temporal_error']:.6f}|{r['edge_mae']:.6f}|")
    table += ['', '|输入H×W|实际Conv MAC|', '|---|---:|']
    for r in cost['rows']: table.append(f"|{r['height']}×{r['width']}|{r['Conv_MACs']}|")
    table += ['', '仅已执行Conv形状MAC，解析先验/归一/规约/激活/flow/warp/内存/IO未计；不是完整FLOPs、实测时延或手机性能。',
              '手机仅同三已见DEV，无GT，不按亮度/平滑代理排名。原始FP32和固定全帧预览另存；独立新方法仍未建立。']
    (out/'tables.md').write_text('\n'.join(table)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, domain in zip(axes, ['davis_train_pilot', 'paired_real_ND_DEV']):
        for r in summary:
            if r['domain'] != domain: continue
            ax.scatter(r['temporal_error'], r['psnr_rgb'], s=90 if r['seed'] == 'published' else 45,
                       marker='*' if r['seed'] == 'published' else 'o', label=r['method']+'/'+r['seed'])
        ax.set_title('SYN reused DEV' if domain == 'davis_train_pilot' else 'Paired ND reused DEV')
        ax.set_xlabel('GT-change temporal error (lower is better)'); ax.set_ylabel('PSNR (dB)'); ax.grid(alpha=.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=8, frameon=False)
    fig.suptitle('One published LOL-v1 micro-model vs fixed project references / unequal training histories')
    fig.tight_layout(rect=[0, .14, 1, .95])
    for extension in ['png', 'pdf']: fig.savefig(out/f'paired-comparison.{extension}', dpi=150)
    plt.close(fig)
    write(out/'sources.json', {str(p.relative_to(ROOT)): digest(p) for p in sources})
    write(out/'audit.json', dict(at=datetime.now().astimezone().isoformat(), complete_paired_table=True,
          parameters=parameters, unique_published_checkpoints=1, new_quality_rows=20, new_phone_rows=6,
          real_CNN_sequences=13, real_CNN_frames=1400, synthetic_gate_CNN_images=6, new_training=0,
          phone_previews_still_required=True, visual_review_still_required=True, equal_training=False,
          independent_final=False, phone_cost_measured=False, new_method=False))


if __name__ == '__main__':
    main()
