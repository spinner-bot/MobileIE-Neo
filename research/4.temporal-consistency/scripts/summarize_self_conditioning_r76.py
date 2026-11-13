"""Preserve primary80 rows/gates and add all20 self4 results with four limited support checks."""
import argparse
from collections import defaultdict
import json

import numpy as np

from causal_reference_io_r75 import ROOT,BASE,write,lines,digest,stamp
from causal_reference_evaluation_r75 import key
from self_conditioning_evaluation_r76 import require_primary_pipeline,self4_quality


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--perceptual',action='store_true')
    parser.add_argument('--metric-attempt',type=int,default=1)
    args=parser.parse_args()
    _,primary,original_gates=require_primary_pipeline()
    new,_,sources=self4_quality()
    if args.perceptual:
        run=BASE/f'results/exploration-r76-perceptual{args.metric_attempt:03d}'
        terminal=json.loads((run/'runtime.json').read_text())
        assert terminal['completed_normally'] and terminal['quality_rows']==20 and terminal['comparison_frames']==896
        scores=lines(run/'per_sequence.jsonl');lookup={key(r):r for r in scores}
        assert len(scores)==len(lookup)==20 and set(lookup)=={key(r) for r in new}
        new=[dict(r,JOD=lookup[key(r)]['JOD']) for r in new]
        for name in ('runtime.json','metric-source.json','config.json','per_sequence.jsonl','actual-device.json'):
            path=run/name;sources[str(path.relative_to(ROOT))]=digest(path)
    else:
        new=[dict(r,JOD=None) for r in new]
    rows=primary+new
    assert len(rows)==len({key(r) for r in rows})==100
    groups=defaultdict(list)
    for row in rows:
        groups[row['domain'],row['seed'],row['label']].append(row)
    means={}
    for identity,part in groups.items():
        expected=6 if identity[0]=='davis_train_pilot' else 4
        assert len(part)==len({r['scene'] for r in part})==expected
        means[identity]={metric:None if any(r[metric] is None for r in part)
                        else float(np.mean([r[metric] for r in part]))
                        for metric in ('psnr_rgb','ssim_rgb','temporal_error','edge_mae','JOD')}
        assert all(value is None or np.isfinite(value) for value in means[identity].values())
    assert len(means)==20
    effects=[]
    for domain in ('davis_train_pilot','paired_real_ND_DEV'):
        for seed in (0,1):
            causal=means[domain,seed,'ReBotNet_causal']
            control=means[domain,seed,'ReBotNet_self4']
            psnr=causal['psnr_rgb']-control['psnr_rgb']
            jod=causal['JOD']-control['JOD'] if args.perceptual else None
            defined=control['temporal_error']>0
            ratio=causal['temporal_error']/control['temporal_error'] if defined else None
            passed=(psnr>=-.05 and jod>=-.02 and ratio<=.95) if args.perceptual and defined else None
            effects.append(dict(domain=domain,seed=seed,reference='ReBotNet_self4',psnr_delta=psnr,
                JOD_delta=jod,temporal_ratio=ratio,temporal_ratio_defined=defined,
                scores_complete=args.perceptual,support_passed=passed,
                meaning='Fixed more-compute single-observation control; not a general causal proof or method novelty.'))
    first=[]
    for seed in (0,1):
        old=BASE/f'results/exploration-r75-eval-seed{seed}/per_frame.jsonl'
        added=BASE/f'results/exploration-r76-eval-seed{seed}/per_frame.jsonl'
        a={r['scene']:r for r in lines(old) if r['label']=='ReBotNet_causal' and r['frame']==0}
        b={r['scene']:r for r in lines(added) if r['frame']==0}
        assert len(a)==len(b)==10 and set(a)==set(b)
        first += [dict(seed=seed,scene=scene,causal_calls=1,self4_calls=4,
                       first_frame_PSNR_causal_minus_self4=a[scene]['psnr_rgb']-b[scene]['psnr_rgb']) for scene in a]
    primary_folder=BASE/'report/exploration-r75-joint'
    for name in ('all_conditions.json','gates.json','audit.json'):
        path=primary_folder/name;sources[str(path.relative_to(ROOT))]=digest(path)
    primary_audit=json.loads((primary_folder/'audit.json').read_text())
    out=BASE/('report/exploration-r76-joint' if args.perceptual else 'report/exploration-r76-quality')
    out.mkdir(exist_ok=False)
    write(out/'all_conditions.json',rows)
    write(out/'domain_seed_summary.json',[dict(domain=d,seed=s,label=l,**v) for (d,s,l),v in sorted(means.items())])
    write(out/'original_R75_gates.json',original_gates)
    write(out/'attribution_support.json',effects)
    write(out/'first_frame_effects.json',first)
    supported=all(r['support_passed'] is True for r in effects) if args.perceptual else None
    write(out/'audit.json',dict(at=stamp(),completed=True,all_conditions=100,domain_seed_means=20,
        added_quality_rows=20,added_output_frames=896,added_frame_CNN_calls=3584,added_tile_CNN_calls=19456,
        added_metric_rows=20 if args.perceptual else 0,SGD=0,original_primary_gates=8,
        original_R75_all8_gates_passed=primary_audit['all8_gates_passed'],primary_results_rewritten=False,
        support_checks=4,scores_complete=args.perceptual,all4_fixed_control_support=supported,
        novelty_established=False,phone_measured=False,independent_final=False,
        scope='Attribution diagnostic only; more computation, first-frame difference retained, no rescue of primary outcomes.'))
    write(out/'sources.json',sources)
    table=['# R76：更多计算的单观测自条件控制','',
        'self4每真实帧独立四次完整调用；causal每帧一次并读过去。参数相同，计算不相同。',
        '原R75全部80条件与八门槛原样保留，新增20控制；仍为重复DEV，ND288×512原生裁块。','',
        '|域|seed|映射|PSNR|SSIM|Etemp↓|JOD↑|','|---|---:|---|---:|---:|---:|---:|']
    for (domain,seed,label),v in sorted(means.items()):
        jod='pending' if v['JOD'] is None else f"{v['JOD']:.6f}"
        table.append(f"|{domain}|{seed}|{label}|{v['psnr_rgb']:.6f}|{v['ssim_rgb']:.6f}|{v['temporal_error']:.6f}|{jod}|")
    table+=['','|域|seed|causal−self4 PSNR|Etemp比|JOD差|有限支持|','|---|---:|---:|---:|---:|---|']
    for r in effects:
        ratio='undefined' if r['temporal_ratio'] is None else f"{r['temporal_ratio']:.6f}"
        jod='pending' if r['JOD_delta'] is None else f"{r['JOD_delta']:+.6f}"
        table.append(f"|{r['domain']}|{r['seed']}|{r['psnr_delta']:+.6f}|{ratio}|{jod}|{r['support_passed']}|")
    table+=['',f"原R75八门槛全部通过={primary_audit['all8_gates_passed']}；本控制四单元支持={supported}。",
        'self4首帧四次、causal首帧一次，未剪掉首帧；20项首帧差另存。',
        '若只赢raw-current而未赢self4，额外观测的独立作用保留未决；也不由失败否定所有时间依赖。',
        '无新训练/独立视频/手机或独立最终；即使支持也不是新方法或一般因果证明。']
    (out/'tables.md').write_text('\n'.join(table)+'\n')
    print((out/'audit.json').read_text(),flush=True)


if __name__=='__main__':
    main()
