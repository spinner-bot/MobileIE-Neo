"""Keep temporal and time-preserving-quality gates distinct, on full final runs."""
import hashlib
import json
from collections import defaultdict
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dynamic_training_r34 import BASE, ROOT


def lines(path):return [json.loads(line) for line in path.read_text().splitlines()]


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    out=BASE/'report/exploration-r48-joint-recursion'
    assert not out.exists()
    rows,sources,motion=[],[],[]
    for seed in [0,1]:
        train=BASE/f'results/exploration-r48-stage001-seed{seed}'
        run=BASE/f'results/exploration-r48-stage001-eval-seed{seed}'
        training=json.loads((train/'runtime.json').read_text())
        runtime=json.loads((run/'runtime.json').read_text())
        assert training['completed_normally'] and training['model_updates']=={'plain':4000,'joint':4000}
        assert runtime['completed_normally'] and runtime['rows']==60 and runtime['neural_sequences']==20
        assert runtime['per_frame_rows']==2688 and not runtime['cuda_initialized']
        part=lines(run/'per_sequence.jsonl');assert len(part)==60
        rows.extend([{**r,'mapping':r['model']+':'+r['method']} for r in part])
        motion.extend(lines(run/'motion_acceptance.jsonl'))
        old_path=BASE/f'results/exploration-r44-stage001-eval-seed{seed}/per_sequence.jsonl'
        old=lines(old_path);assert len(old)==120
        rows.extend([{**r,'mapping':'R44_square:'+r['method']} for r in old if r['model']=='square_mse'])
        rgb_path=BASE/f'results/exploration-r45-stage001-eval-seed{seed}/per_sequence.jsonl'
        rgb=lines(rgb_path);assert len(rgb)==90
        rows.extend([{**r,'mapping':'R45_rgb'} for r in rgb if r['method']=='rgb'])
        sources.extend([train/'config.json',train/'runtime.json',run/'config.json',run/'runtime.json',
                        run/'per_sequence.jsonl',run/'motion_acceptance.jsonl',old_path,rgb_path])
    assert len(rows)==200 and len(motion)==40
    groups=defaultdict(list)
    for r in rows:groups[r['domain'],r['train_seed'],r['mapping']].append(r)
    metrics=['psnr_rgb','ssim_rgb','edge_mae','temporal_error']
    means={key:{m:float(np.mean([r[m] for r in part])) for m in metrics} for key,part in groups.items()}
    domains=['davis_train_pilot','paired_real_ND_DEV']
    effects=[]
    for domain in domains:
        for seed in [0,1]:
            candidate=means[domain,seed,'joint:recursive_self05']
            cand_rows={r['scene']:r for r in groups[domain,seed,'joint:recursive_self05']}
            for reference in ['plain:recursive_self05','R44_square:recursive_self05']:
                ref=means[domain,seed,reference]
                ref_rows={r['scene']:r for r in groups[domain,seed,reference]}
                delta=candidate['psnr_rgb']-ref['psnr_rgb']
                ratio=candidate['temporal_error']/ref['temporal_error']
                worst=min(cand_rows[k]['psnr_rgb']-ref_rows[k]['psnr_rgb'] for k in cand_rows)
                effects.append(dict(domain=domain,seed=seed,reference=reference,psnr_delta=delta,
                     temporal_ratio=ratio,worst_condition_psnr_delta=worst,quality_passed=delta>=-.05,
                     temporal_A_passed=ratio<=.9,temporal_B_preserved=ratio<=1.02,
                     condition_passed=worst>=-.5,quality_B_gain=delta>=.2))
    gate_A=all(r[k] for r in effects for k in ['quality_passed','temporal_A_passed','condition_passed'])
    B_domains=[domain for domain in domains if all(r['quality_B_gain'] for r in effects if r['domain']==domain)]
    gate_B=(bool(B_domains) and all(r[k] for r in effects for k in
                                  ['quality_passed','temporal_B_preserved','condition_passed']))
    out.mkdir()
    write(out/'effects.json',effects)
    write(out/'domain_seed_summary.json',[dict(domain=d,seed=s,mapping=m,**v) for (d,s,m),v in means.items()])
    pooled=defaultdict(list)
    for (d,s,m),v in means.items():pooled[d,m].append(v)
    write(out/'domain_mean_summary.json',[dict(domain=d,mapping=m,
          **{k:float(np.mean([v[k] for v in part])) for k in metrics}) for (d,m),part in pooled.items()])
    write(out/'all_conditions.json',rows)
    write(out/'motion_acceptance.json',motion)
    write(out/'audit.json',dict(at=datetime.now().astimezone().isoformat(),processing_complete=True,
          new_quality_rows=120,new_neural_sequences=40,new_per_frame_rows=5376,
          gate_A_temporal=gate_A,gate_B_time_preserving_quality=gate_B,gate_B_supported_domains=B_domains,
          cost_stage_allowed=gate_A or gate_B,local_family_frozen=not(gate_A or gate_B),
          novelty_established=False,final_test=False,phone_measured=False))
    table=['# R48 完整原骨干与固定递归联合适配','',
           '同8000起点、同额外4000、两模型两seed，共16000新更新；4075部署参数，两组都clip。',
           '120新质量行/40CNN/5376逐帧；R44/R45为完整冻结参考，不把更长训练全部归到联合目标。','',
           '|域|seed|映射|PSNR|SSIM|时间↓|edge↓|','|---|---:|---|---:|---:|---:|---:|']
    for (d,s,m),v in means.items():
        table.append(f"|{d}|{s}|{m}|{v['psnr_rgb']:.6f}|{v['ssim_rgb']:.6f}|{v['temporal_error']:.6f}|{v['edge_mae']:.6f}|")
    table+=['','|域|seed|参考|ΔPSNR|时间比|最差条件ΔPSNR|','|---|---:|---|---:|---:|---:|']
    for r in effects:
        table.append(f"|{r['domain']}|{r['seed']}|{r['reference']}|{r['psnr_delta']:+.6f}|{r['temporal_ratio']:.6f}|{r['worst_condition_psnr_delta']:+.6f}|")
    table+=['',f'门槛A（独立时间收益）={gate_A}；B（时间保持画质）={gate_B}，支持域={B_domains}。',
            'A/B不能混为同一结论。自己的raw流依赖模型；配对的flow接受比例保留，未将效果单独归因采样补偿。',
            '原视频端到端训练已有先例；本轮不建立创新、成本通过、手机/夜间/最终泛化结论。']
    (out/'tables.md').write_text('\n'.join(table)+'\n')
    write(out/'sources.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for col,d in enumerate(domains):
        for s in [0,1]:
            ax=axes[s,col]
            for model,color in [('R44_square','gray'),('plain','tab:blue'),('joint','tab:red')]:
                values=[means[d,s,model+':'+mode] for mode in ['raw','finite_self05','recursive_self05']]
                ax.plot([v['temporal_error'] for v in values],[v['psnr_rgb'] for v in values],
                        'o-',color=color,label=model+' / raw-finite-recursive')
            v=means[d,s,'R45_rgb'];ax.scatter(v['temporal_error'],v['psnr_rgb'],marker='D',color='tab:green',label='R45 RGB gate')
            ax.set_title(('SYN' if col==0 else 'Paired ND DEV')+f' / seed {s}')
            ax.set_xlabel('GT-change temporal error (lower is better)');ax.set_ylabel('PSNR (dB)');ax.grid(alpha=.25)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
    fig.suptitle('Complete fixed-recursion adaptation / reused DEV / final12000 only')
    fig.tight_layout(rect=[0,.09,1,.95])
    for extension in ['png','pdf']:fig.savefig(out/f'comparison.{extension}',dpi=150)
    plt.close(fig)
    print((out/'audit.json').read_text())


if __name__=='__main__':main()
