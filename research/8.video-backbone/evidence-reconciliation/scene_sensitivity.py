"""All scene-equal and leave-one-scene-out descriptive contrasts; no new gates."""
import csv
from collections import defaultdict
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import time
from analyze import BASE,TOPIC,PAIRS,read,lines

OUT=BASE/'results/scene-sensitivity001'


def save(name,rows):
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    began=time.perf_counter();OUT.mkdir(parents=True,exist_ok=False)
    original=BASE/'results/analysis001'
    assert read(original/'runtime.json')['completed_normally']
    input_scope=read(original/'input-scope.json')
    scenes={r['draw_key']:r['scene'] for r in input_scope['draws']}
    inputs=[BASE/'scope.md',Path(__file__).resolve(),BASE/'analyze.py',original/'runtime.json',original/'input-scope.json']
    indices={}
    for study in PAIRS:
        dirs=[TOPIC/f'latent-core/results/held-seed{s}' for s in (0,1)] if study=='L1' else [TOPIC/'affine-cache/results/screen001']
        rows=[]
        for d in dirs:
            path=d/'per_sequence.jsonl';inputs.append(path);rows.extend(lines(path))
        indices[study]={(r['key'],r['arm']):r for r in rows}
    scene_rows,leave_rows,summaries=[],[],[]
    for study,pairs in PAIRS.items():
        index=indices[study]
        for candidate,reference in pairs:
            for seed,domain in [(0,'SYN'),(0,'ND'),(1,'SYN'),(1,'ND')]:
                selected=[r for r in index.values() if (r['arm'],r['seed'],r['domain'])==(candidate,seed,domain)]
                assert len(selected)==16
                grouped=defaultdict(list)
                for c in selected:
                    draw=c['key'].split('-',1)[1]
                    grouped[scenes[draw]].append(c)
                shared=dict(study=study,seed=seed,domain=domain,candidate=candidate,reference=reference)
                local=[]
                for scene,values in sorted(grouped.items()):
                    refs=[index[v['key'],reference] for v in values]
                    local.append(dict(**shared,scene=scene,windows=len(values),
                        candidate_PSNR=sum(v['psnr_rgb'] for v in values)/len(values),
                        reference_PSNR=sum(v['psnr_rgb'] for v in refs)/len(refs),
                        candidate_Etemp=sum(v['temporal_error'] for v in values)/len(values),
                        reference_Etemp=sum(v['temporal_error'] for v in refs)/len(refs)))
                    kept=[v for v in selected if scenes[v['key'].split('-',1)[1]]!=scene]
                    kept_refs=[index[v['key'],reference] for v in kept]
                    leave_rows.append(dict(**shared,removed_scene=scene,retained_windows=len(kept),
                        mean_PSNR_delta=sum(c['psnr_rgb']-r['psnr_rgb'] for c,r in zip(kept,kept_refs))/len(kept),
                        ratio_of_mean_Etemp=sum(c['temporal_error'] for c in kept)/sum(r['temporal_error'] for r in kept_refs)))
                scene_rows.extend(local)
                leaves=leave_rows[-len(local):]
                summaries.append(dict(**shared,scenes=len(local),windows=16,
                    scene_equal_PSNR_delta=sum(v['candidate_PSNR']-v['reference_PSNR'] for v in local)/len(local),
                    scene_equal_Etemp_ratio=sum(v['candidate_Etemp'] for v in local)/sum(v['reference_Etemp'] for v in local),
                    leave_one_scene_out_min_PSNR_delta=min(v['mean_PSNR_delta'] for v in leaves),
                    leave_one_scene_out_max_PSNR_delta=max(v['mean_PSNR_delta'] for v in leaves),
                    leave_one_scene_out_min_Etemp_ratio=min(v['ratio_of_mean_Etemp'] for v in leaves),
                    leave_one_scene_out_max_Etemp_ratio=max(v['ratio_of_mean_Etemp'] for v in leaves)))
    assert len(scene_rows)==len(leave_rows)==616 and len(summaries)==56
    save('all-scenes.csv',scene_rows);save('leave-one-scene-out.csv',leave_rows);save('summary.csv',summaries)
    (OUT/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    text=['# 全场景权重敏感性（描述性补充）','',
          'SYN13/ND9训练场景；模型seed共用输入。等权/留一结果不修改原门或形成独立测试/置信区间。','',
          '|研究|域|seed|候选|参考|场景等权PSNR差|场景等权Etemp比|留一PSNR差范围|留一Etemp比范围|',
          '|---|---|---:|---|---|---:|---:|---|---|']
    for r in summaries:
        text.append(f"|{r['study']}|{r['domain']}|{r['seed']}|{r['candidate']}|{r['reference']}|{r['scene_equal_PSNR_delta']:+.6f}|{r['scene_equal_Etemp_ratio']:.6f}|{r['leave_one_scene_out_min_PSNR_delta']:+.6f}..{r['leave_one_scene_out_max_PSNR_delta']:+.6f}|{r['leave_one_scene_out_min_Etemp_ratio']:.6f}..{r['leave_one_scene_out_max_Etemp_ratio']:.6f}|")
    (OUT/'tables.md').write_text('\n'.join(text)+'\n')
    (OUT/'runtime.json').write_text(json.dumps(dict(at=datetime.now(timezone(timedelta(hours=8))).isoformat(),completed_normally=True,
        seconds=time.perf_counter()-began,scene_rows=616,leave_one_scene_out_rows=616,summary_rows=56,
        source_sha256={str(p.relative_to(TOPIC.parents[1])):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        scope_amendment_after_original_analysis=True,original_tables_and_gates_unchanged=True,new_CNN_calls=0,new_flow_pairs=0,optimizer_updates=0,GPU_used=False,cached_pixels_read=False),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(completed=True,scene_rows=616,leave_one_scene_out_rows=616,summary_rows=56)))


if __name__=='__main__':
    main()
