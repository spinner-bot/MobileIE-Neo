"""Preserve every optimizer candidate and audit fixed data/checkpoint budgets."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256
from summarize_causal_r5 import KEYS, means, paired

BASE=ROOT/'research/4.temporal-consistency'


def jsonl(path):return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    out=Path(p.parse_args().out);out.mkdir(parents=True,exist_ok=False)
    rows=[];sources={};training=[]
    for scope,expected in [('moderate',368),('high',168)]:
        folder=BASE/'results'/('exploration-r9-eval-'+scope)
        rr=jsonl(folder/'per_sequence.jsonl')
        assert len(rr)==expected==json.loads((folder/'runtime.json').read_text())['rows']
        rows+=rr
        for item in json.loads((folder/'checkpoints.json').read_text()):
            assert sha256(ROOT/item['path'])==item['sha256']
        sources.update({str(f.relative_to(ROOT)):sha256(f) for f in folder.iterdir() if f.is_file()})
    assert len(rows)==536
    old=BASE/'results/exploration-r8-eval-head/per_sequence.jsonl'
    rows += [{k:r[k] for k in ['domain','scene','variant','train_seed','fusion',*KEYS]} for r in jsonl(old)
             if r['variant'] in ['plain','spatial_copy'] and r['history']=='causal' and not r['input_reset']]
    sources[str(old.relative_to(ROOT))]=sha256(old)
    assert len({(r['scene'],r['variant'],r['train_seed'],r['fusion']) for r in rows})==840
    for name in ['exploration-r5-cache','exploration-r9-dev-cache']:
        path=BASE/'results'/name/'baseline_per_sequence.json';sources[str(path.relative_to(ROOT))]=sha256(path)
        for r in json.loads(path.read_text()):
            fusion=r['method'].split('__')[1] if 'method' in r else r['fusion']
            rows.append({**r,'variant':'pretrained','fusion':fusion,'train_seed':None})
    groups=collections.defaultdict(list);clips=collections.defaultdict(list)
    for r in rows:
        groups[(r['domain'],r['variant'],r['fusion'])].append(r)
        clips[(r['domain'],r['scene'],r['variant'],r['fusion'])].append(r)
    summary=[{'domain':d,'variant':v,'fusion':f,'rows':len(rr),**means(rr)} for (d,v,f),rr in sorted(groups.items())]
    save_json(out/'domain_summary.json',summary)
    save_json(out/'by_scene.json',[{'domain':d,'scene':s,'variant':v,'fusion':f,**means(rr)} for (d,s,v,f),rr in sorted(clips.items())])
    contrasts=[]
    for d in ['dev','test','legacy_synthetic','event_cut','event_light_step']:
        rr=[{**r,'method':r['variant']+'__'+r['fusion']} for r in rows]
        for v in ['full_lr1e4','full_lr1e3','head_lr1e3']:
            for b in ['plain','spatial_copy']:
                for f in ['raw','half01']:contrasts.append(paired(rr,v+'__'+f,b+'__'+f,d))
    save_json(out/'paired_deltas.json',contrasts)
    for name in ['moderate','high']:
        for folder in sorted((BASE/'results'/('exploration-r9-'+name)).iterdir()):
            config=json.loads((folder/'config.json').read_text());trace=jsonl(folder/'training.jsonl')
            prior=jsonl(BASE/'results/exploration-r5-spatial'/f"plain-seed{config['seed']}"/'training.jsonl')
            assert [r['step'] for r in trace]==list(range(1,1001))
            assert [r['sample_identity'] for r in trace]==[r['sample_identity'] for r in prior]
            checkpoints=list(folder.glob('checkpoint-step*.json'));assert len(checkpoints)==10
            for path in checkpoints:
                item=json.loads(path.read_text());assert sha256(ROOT/item['path'])==item['sha256']
            training.append({'variant':config['variant'],'seed':config['seed'],'parameters':config['parameters'],
                'initial_plain_audit':json.loads((folder/'initial_plain_audit.json').read_text()),
                'last100_task':float(np.mean([r['task'] for r in trace[-100:]])),
                'steps_gradient_clipped':sum(r['gradient_norm_backbone']>1 for r in trace)})
    assert len(training)==6
    save_json(out/'training_audit.json',{'runs':training,'updates':6000,'checkpoints_verified':60,'current_GT_stream_matches_plain':True})
    save_json(out/'audit.json',{'new_evaluations':536,'reused_evaluations':304,'teacher_evaluations':84,
        'new_training_runs':6,'new_training_updates':6000,'input_conditions':42,'sources':sources,
        'scope':'four new official TRAIN project-dev clips; old repeated exploratory test; not independent final evaluation'})
    lines=['# Optimizer baseline controls','','Fixed final1000, two seeds. All candidates retained; project dev is official TRAIN, no final-test claim.','']
    for d in ['dev','test','legacy_synthetic','event_cut','event_light_step']:
        lines+=['## '+d,'','| Variant | Fusion | PSNR | SSIM | Temporal | Edge |','|---|---|---:|---:|---:|---:|']
        for r in summary:
            if r['domain']==d:lines.append(f"| {r['variant']} | {r['fusion']} | {r['psnr_rgb']:.5f} | {r['ssim_rgb']:.5f} | {r['temporal_error']:.5f} | {r['edge_mae']:.5f} |")
        lines.append('')
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
