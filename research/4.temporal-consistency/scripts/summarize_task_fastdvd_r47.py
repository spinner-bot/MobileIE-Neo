"""Completed-only published-reference table, regardless of which method wins."""
import hashlib
import json
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dynamic_training_r34 import BASE, ROOT


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    out = BASE/'report/exploration-r47-task-adapted-reference'
    out.mkdir(exist_ok=False)
    sources, rows = [], []
    for seed in [0,1]:
        train = BASE/f'results/exploration-r47-stage001-seed{seed}'
        run = BASE/f'results/exploration-r47-stage001-eval-seed{seed}'
        training = json.loads((train/'runtime.json').read_text())
        runtime = json.loads((run/'runtime.json').read_text())
        assert training['completed_normally'] and training['steps']==8000
        assert runtime['completed_normally'] and runtime['rows']==20 and runtime['per_frame_rows']==896
        assert runtime['neural_sequences']==20 and not runtime['cuda_initialized']
        new = lines(run/'per_sequence.jsonl')
        assert len(new)==20 and len(lines(run/'per_frame.jsonl'))==896
        rows.extend([{**r,'method':'FastDVD_task_'+r['method']} for r in new])
        old_path = BASE/f'results/exploration-r45-stage001-eval-seed{seed}/per_sequence.jsonl'
        old = lines(old_path)
        assert len(old)==90
        rows.extend([{**r,'method':'MobileIE_'+r['method']} for r in old if
                     r['method'] in ['raw','fixed_025','fixed_05','rgb']])
        sources.extend([train/'runtime.json',train/'fastdvd-step8000.json',run/'config.json',
                        run/'runtime.json',run/'per_sequence.jsonl',run/'prediction_manifest.jsonl',old_path])
    assert len(rows)==120
    groups = defaultdict(list)
    for r in rows:
        groups[r['domain'],r['train_seed'],r['method']].append(r)
    keys = ['psnr_rgb','ssim_rgb','edge_mae','temporal_error']
    means = {k:{name:float(np.mean([r[name] for r in part])) for name in keys}
             for k,part in groups.items()}
    domain_seed = [dict(domain=d,seed=s,method=m,**v) for (d,s,m),v in means.items()]
    write(out/'domain_seed_summary.json',domain_seed)
    pooled = defaultdict(list)
    for r in domain_seed:
        pooled[r['domain'],r['method']].append(r)
    domain_mean = [dict(domain=d,method=m,**{k:float(np.mean([r[k] for r in v])) for k in keys})
                   for (d,m),v in pooled.items()]
    write(out/'domain_mean_summary.json',domain_mean)
    effects = []
    for (domain,seed,method),value in means.items():
        if method!='FastDVD_task_window':continue
        for reference in ['FastDVD_task_five_current','MobileIE_raw','MobileIE_fixed_05','MobileIE_rgb']:
            ref=means[domain,seed,reference]
            effects.append(dict(domain=domain,seed=seed,reference=reference,
                  psnr_delta=value['psnr_rgb']-ref['psnr_rgb'],
                  temporal_ratio=value['temporal_error']/ref['temporal_error']))
    write(out/'context_effects.json',effects)
    write(out/'paired_conditions.json',rows)
    table = ['# R47 同资料任务适配视频参考：完整结果','',
       '两个seed各8000更新，总16000。40新CPU质量/CNN行，1792逐帧行；不按胜负筛选参考。',
       'FastDVDnet原五帧网络直接低光到GT，2479096参数；MobileIE为最终8000强空间底座，4075参数。',
       '前者window未来等待2帧、five_current为同权重五份当前；不同预训练/容量/BN/训练FLOPs，且公开预训练与三DEV视频有来源重叠。','',
       '|域|seed|方法|PSNR|SSIM|时间↓|edge↓|','|---|---:|---|---:|---:|---:|---:|']
    for r in domain_seed:
        table.append(f"|{r['domain']}|{r['seed']}|{r['method']}|{r['psnr_rgb']:.6f}|{r['ssim_rgb']:.6f}|{r['temporal_error']:.6f}|{r['edge_mae']:.6f}|")
    table += ['', '固定有限预算适配不等于原论文最优或充分收敛；同更新数不是等计算量；CPU任务耗时不是手机实测。无独立最终或自然夜间结论，本表补足公开视频参考，不建立本项目创新。']
    (out/'tables.md').write_text('\n'.join(table)+'\n')
    write(out/'audit.json',dict(at=datetime.now().astimezone().isoformat(),processing_complete=True,
          adapted_training_updates=16000,new_quality_rows=40,new_neural_sequences=40,
          new_per_frame_rows=1792,comparison_rows=120,method_success_not_inferred=True,
          unseen_final=False,natural_night=False,phone_measured=False))
    write(out/'sources.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    style=[('MobileIE_raw','x','gray'),('MobileIE_fixed_025','o','tab:orange'),
           ('MobileIE_fixed_05','o','tab:blue'),('MobileIE_rgb','D','tab:green'),
           ('FastDVD_task_window','*','tab:red'),('FastDVD_task_five_current','s','tab:purple')]
    for col,domain in enumerate(['davis_train_pilot','paired_real_ND_DEV']):
        for seed in [0,1]:
            ax=axes[seed,col]
            for method,marker,color in style:
                r=means[domain,seed,method]
                ax.scatter(r['temporal_error'],r['psnr_rgb'],marker=marker,color=color,
                           s=110 if marker=='*' else 45,label=method)
            ax.set_title(('SYN' if col==0 else 'Paired ND DEV')+f' / seed {seed}')
            ax.set_xlabel('GT-change temporal error (lower is better)');ax.set_ylabel('PSNR (dB)')
            ax.grid(alpha=.25)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Task-adapted published reference / reused DEV / final8000 only')
    fig.tight_layout(rect=[0,.12,1,.95])
    for extension in ['png','pdf']:fig.savefig(out/f'comparison.{extension}',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    main()
