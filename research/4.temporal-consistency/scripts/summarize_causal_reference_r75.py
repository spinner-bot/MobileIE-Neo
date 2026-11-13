"""All final conditions and registered8 gates; missing perceptual results stay pending."""
import argparse
from collections import defaultdict
import json

import numpy as np

from causal_reference_io_r75 import ROOT,BASE,write,lines,digest,stamp
from causal_reference_evaluation_r75 import key,reference_quality,final_markers


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--perceptual',action='store_true')
    parser.add_argument('--metric-attempt',type=int,default=1)
    args=parser.parse_args()
    markers=final_markers()
    new=[];sources={}
    for seed in (0,1):
        run=BASE/f'results/exploration-r75-eval-seed{seed}'
        runtime=json.loads((run/'runtime.json').read_text())
        assert runtime['completed_normally'] and runtime['quality_rows']==20 and runtime['frame_rows']==896
        assert runtime['tile_CNN_calls']==4864
        part=lines(run/'per_sequence.jsonl');assert len(part)==20
        new+=part
        for name in ('runtime.json','config.json','per_sequence.jsonl','per_frame.jsonl','prediction_manifest.jsonl','first_frame.jsonl'):
            p=run/name;sources[str(p.relative_to(ROOT))]=digest(p)
    assert len(new)==len({key(r) for r in new})==40
    if args.perceptual:
        metric=BASE/f'results/exploration-r75-perceptual{args.metric_attempt:03d}'
        runtime=json.loads((metric/'runtime.json').read_text())
        assert runtime['completed_normally'] and runtime['quality_rows']==40 and runtime['comparison_frames']==1792
        scores=lines(metric/'per_sequence.jsonl');lookup={key(r):r for r in scores}
        assert len(scores)==len(lookup)==40 and set(lookup)=={key(r) for r in new}
        new=[dict(r,JOD=lookup[key(r)]['JOD']) for r in new]
        for name in ('runtime.json','config.json','per_sequence.jsonl','metric-source.json','actual-device.json'):
            p=metric/name;sources[str(p.relative_to(ROOT))]=digest(p)
    else:
        new=[dict(r,JOD=None) for r in new]
    old,ref_sources=reference_quality();sources.update(ref_sources)
    rows=old+new
    assert len(rows)==len({key(r) for r in rows})==80
    groups=defaultdict(list)
    for row in rows:
        groups[row['domain'],row['seed'],row['label']].append(row)
    metrics=['psnr_rgb','ssim_rgb','temporal_error','edge_mae','JOD']
    means={}
    for identity,part in groups.items():
        expected=6 if identity[0]=='davis_train_pilot' else 4
        assert len(part)==len({r['scene'] for r in part})==expected
        means[identity]={m:None if any(r[m] is None for r in part) else float(np.mean([r[m] for r in part])) for m in metrics}
        assert all(value is None or np.isfinite(value) for value in means[identity].values())
    assert len(means)==16
    gates=[]
    for domain in ('davis_train_pilot','paired_real_ND_DEV'):
        for seed in (0,1):
            candidate=means[domain,seed,'ReBotNet_causal']
            candidate_rows={r['scene']:r for r in groups[domain,seed,'ReBotNet_causal']}
            for label,time_threshold in [('MobileIE_recursive_self05',.90),('ReBotNet_current',.95)]:
                reference=means[domain,seed,label]
                reference_rows={r['scene']:r for r in groups[domain,seed,label]}
                assert set(candidate_rows)==set(reference_rows)
                psnr=candidate['psnr_rgb']-reference['psnr_rgb']
                ratio=candidate['temporal_error']/reference['temporal_error']
                jod=candidate['JOD']-reference['JOD'] if args.perceptual else None
                necessary=psnr>=-.05 and ratio<=time_threshold
                gates.append(dict(domain=domain,seed=seed,reference=label,psnr_delta=psnr,temporal_ratio=ratio,
                    temporal_ratio_threshold=time_threshold,JOD_delta=jod,
                    worst_condition_psnr_delta=min(candidate_rows[k]['psnr_rgb']-reference_rows[k]['psnr_rgb'] for k in candidate_rows),
                    worst_condition_temporal_ratio=max(candidate_rows[k]['temporal_error']/reference_rows[k]['temporal_error'] for k in candidate_rows),
                    quality_time_necessary_passed=necessary,full_gate_evaluated=args.perceptual,
                    passed=bool(necessary and jod>=-.02) if args.perceptual else None))
    assert len(gates)==8
    out=BASE/('report/exploration-r75-joint' if args.perceptual else 'report/exploration-r75-quality')
    out.mkdir(exist_ok=False)
    write(out/'all_conditions.json',rows)
    write(out/'domain_seed_summary.json',[dict(domain=d,seed=s,label=l,**v) for (d,s,l),v in sorted(means.items())])
    write(out/'gates.json',gates)
    all_passed=all(r['passed'] for r in gates) if args.perceptual else None
    write(out/'audit.json',dict(at=stamp(),completed=True,new_quality_rows=40,new_frame_rows=1792,
        tile_CNN_calls=9728,all_quality_rows=80,domain_seed_means=16,new_perceptual_rows=40 if args.perceptual else 0,
        formal_updates=128000,checkpoints=markers,all_quality_time_necessary_passed=all(r['quality_time_necessary_passed'] for r in gates),
        full_gates_evaluated=args.perceptual,all8_gates_passed=all_passed,
        further_mechanism_or_cost_allowed=bool(all_passed),novelty_established=False,
        independent_final=False,phone_measured=False,
        scope='Released XS/self-start/tiled/final64K; unequal training budgets disclosed; repeated DEV with ND native crop, not full1080p or original500K replication.'))
    write(out/'sources.json',sources)
    table=['# R75 公开因果XS：最终完整共同表','',
        'SYN六条件480×854/64帧，ND四条件288×512原生裁块/16帧；均重复DEV。',
        'ReBotNet-XS随机初始化64K、384训练；MobileIE原8K/128训练；输入、像素、初始化和训练计算不同，不作单因果或充分原表复现。','',
        '|域|seed|映射|PSNR|SSIM|Etemp↓|edge↓|JOD↑|','|---|---:|---|---:|---:|---:|---:|---:|']
    for (domain,seed,label),values in sorted(means.items()):
        jod='pending' if values['JOD'] is None else f"{values['JOD']:.6f}"
        table.append(f"|{domain}|{seed}|{label}|{values['psnr_rgb']:.6f}|{values['ssim_rgb']:.6f}|{values['temporal_error']:.6f}|{values['edge_mae']:.6f}|{jod}|")
    table+=['','|域|seed|参考|ΔPSNR|Etemp比|ΔJOD|联合门槛|','|---|---:|---|---:|---:|---:|---|']
    for row in gates:
        jod='pending' if row['JOD_delta'] is None else f"{row['JOD_delta']:+.6f}"
        table.append(f"|{row['domain']}|{row['seed']}|{row['reference']}|{row['psnr_delta']:+.6f}|{row['temporal_ratio']:.6f}|{jod}|{row['passed']}|")
    table+=['',f'完整八门槛已评价={args.perceptual}；全部通过={all_passed}。',
            'current为同权重历史干预，不是独立训练空间模型；低时序误差本身不证明保细节或新方法。',
            '本表不覆盖自然夜间、整幅1080p、手机或独立最终；仅真实完整条件，未取最好seed/中间checkpoint。']
    (out/'tables.md').write_text('\n'.join(table)+'\n')
    print((out/'audit.json').read_text(),flush=True)


if __name__=='__main__':
    main()
