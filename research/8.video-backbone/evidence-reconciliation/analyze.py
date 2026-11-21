"""Read only completed versioned metrics/metadata; no cached pixels or ML imports."""
import csv
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time

BASE=Path(__file__).resolve().parent
TOPIC=BASE.parent
OUT=BASE/'results/analysis001'
PAIRS={'L1': [('F0_FB','F0'),('F0_DIS','F0'),('spatial_current','F0'),
             ('spatial_FB','F0_FB'),('spatial_DIS','F0_DIS'),('causal','causal_current'),
             ('causal','causal_self4'),('causal','spatial_FB'),('causal','spatial_DIS')],
       'A1': [('current','F0'),('cached','current'),('cached','hold'),
              ('cached_FB','F0_FB'),('cached_DIS','F0_DIS')]}
FIELDS=('psnr_rgb','ssim_rgb','edge_mae','flicker','temporal_error')


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(v) for v in path.read_text().splitlines()]


def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    began=time.perf_counter()
    OUT.mkdir(parents=True,exist_ok=False)
    inputs=[BASE/'scope.md',Path(__file__).resolve()]
    rows,manifests={'L1':[],'A1':[]},{'L1':[],'A1':[]}
    windows=[]
    for study in rows:
        dirs=[TOPIC/f'latent-core/results/held-seed{s}' for s in (0,1)] if study=='L1' else [TOPIC/'affine-cache/results/screen001']
        for directory in dirs:
            runtime=directory/'runtime.json'
            assert read(runtime)['completed_normally']
            inputs.append(runtime)
            for name,destination in [('per_sequence',rows[study]),('prediction_manifest',manifests[study])]:
                path=directory/(name+'.jsonl');inputs.append(path);destination.extend(lines(path))
            if study=='L1':
                path=directory/'windows.jsonl';inputs.append(path)
                records=lines(path)
                if windows:
                    assert windows==records
                else:
                    windows=records
    assert len(rows['L1'])==704 and len(rows['A1'])==768
    old={m['key']:m for m in manifests['L1']}
    new={m['key']:m for m in manifests['A1']}
    assert old.keys()==new.keys() and len(old)==64
    for key in old:
        assert all(old[key][f]==new[key][f] for f in ('draw_key','seed','domain'))
        assert old[key]['path']==new[key]['source_path'] and old[key]['sha256']==new[key]['source_sha256']
    by_seed=[{m['draw_key']:(m['domain'],m['low_sha256'],m['GT_sha256']) for m in old.values() if m['seed']==s} for s in (0,1)]
    assert by_seed[0]==by_seed[1] and len(by_seed[0])==32
    indices={study:{(r['key'],r['arm']):r for r in rr} for study,rr in rows.items()}
    alias=0.
    for r in rows['A1']:
        if r['arm'] in ('F0','F0_FB','F0_DIS'):
            other=indices['L1'][r['key'],r['arm']]
            alias=max(alias,max(abs(r[f]-other[f]) for f in FIELDS))
    assert alias<=1e-6
    paired,summaries=[],[]
    for study,pairs in PAIRS.items():
        for candidate,reference in pairs:
            for seed,domain in [(0,'SYN'),(0,'ND'),(1,'SYN'),(1,'ND')]:
                selected=[r for r in rows[study] if (r['arm'],r['seed'],r['domain'])==(candidate,seed,domain)]
                assert len(selected)==16
                part=[]
                for c in selected:
                    r=indices[study][c['key'],reference]
                    assert (r['seed'],r['domain'])==(seed,domain)
                    part.append(dict(study=study,key=c['key'],seed=seed,domain=domain,candidate=candidate,reference=reference,
                        PSNR_delta=c['psnr_rgb']-r['psnr_rgb'],SSIM_delta=c['ssim_rgb']-r['ssim_rgb'],
                        edge_delta=c['edge_mae']-r['edge_mae'],Etemp_ratio=c['temporal_error']/r['temporal_error']))
                paired.extend(part)
                references=[indices[study][c['key'],reference] for c in selected]
                summaries.append(dict(study=study,seed=seed,domain=domain,candidate=candidate,reference=reference,windows=16,
                    mean_PSNR_delta=sum(p['PSNR_delta'] for p in part)/16,
                    worst_PSNR_delta=min(p['PSNR_delta'] for p in part),best_PSNR_delta=max(p['PSNR_delta'] for p in part),
                    ratio_of_mean_Etemp=sum(r['temporal_error'] for r in selected)/sum(r['temporal_error'] for r in references),
                    mean_SSIM_delta=sum(p['SSIM_delta'] for p in part)/16,
                    mean_edge_delta=sum(p['edge_delta'] for p in part)/16,
                    joint_positive_windows=sum(p['PSNR_delta']>0 and p['Etemp_ratio']<1 for p in part)))
    assert len(paired)==896 and len(summaries)==56
    for name,values in [('paired-windows.csv',paired),('paired-summary.csv',summaries)]:
        with (OUT/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]),lineterminator='\n')
            writer.writeheader();writer.writerows(values)
    write('paired-summary.json',summaries)
    scene_rows=[]
    for w in windows:
        assert len(w['windows'])==4
        for i,r in enumerate(w['windows']):
            scene_rows.append(dict(draw_key=f"step{w['data_step']}-window{i}",domain='SYN' if i<2 else 'ND',
                scene=r['video'],source=r['source'],members=r.get('members'),indices=r.get('indices'),xy=r['xy'],photons=r.get('photons')))
    assert {r['draw_key'] for r in scene_rows}==by_seed[0].keys()
    scene_counts={d:dict(Counter(r['scene'] for r in scene_rows if r['domain']==d)) for d in ('SYN','ND')}
    write('input-scope.json',dict(unique_draws=32,window_seed_rows=64,frames_per_window=4,same_inputs_across_model_seeds=True,
        scenes_per_domain={d:len(v) for d,v in scene_counts.items()},scene_counts=scene_counts,draws=scene_rows,
        independent_training_scenes=False,temporal_sensor_cadence_verified=False))
    text=['# 同输入证据归并：全部56描述统计','',
          'L1/A1共有32训练场景四帧抽样窗，按两个原F0 seed重复。不是64独立输入或新通过标准。',
          '所有时间比为均值之比；逐窗正例为严格PSNR差>0且Etemp比<1，不是显著性判断。','',
          '|研究|域|seed|候选|参考|PSNR差|最差窗差|Etemp比|联合正窗|',
          '|---|---|---:|---|---|---:|---:|---:|---:|']
    for r in summaries:
        text.append(f"|{r['study']}|{r['domain']}|{r['seed']}|{r['candidate']}|{r['reference']}|{r['mean_PSNR_delta']:+.6f}|{r['worst_PSNR_delta']:+.6f}|{r['ratio_of_mean_Etemp']:.6f}|{r['joint_positive_windows']}/16|")
    text+=['',f'唯一训练来源：SYN {len(scene_counts["SYN"])}，ND {len(scene_counts["ND"])}；不同窗口可来自同一场景，不能据896配对构造独立场景置信区间。',
           '',f'原F0三基线全部192逐窗行五字段跨研究最大差={alias}；所有旧门/STOP/原论文保持。']
    (OUT/'tables.md').write_text('\n'.join(text)+'\n')
    write('runtime.json',dict(at=datetime.now(timezone(timedelta(hours=8))).isoformat(),completed_normally=True,
        seconds=time.perf_counter()-began,source_sha256={str(p.relative_to(TOPIC.parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        matched_baseline_rows=192,maximum_baseline_metric_difference=alias,paired_windows=896,paired_means=56,
        new_CNN_calls=0,new_flow_pairs=0,optimizer_updates=0,GPU_used=False,cached_pixels_read=False,new_gate_or_method=False))
    print(json.dumps(dict(completed=True,paired_windows=896,paired_means=56,scene_counts=scene_counts)))


if __name__=='__main__':
    main()
