"""Audit matched head training, optimizer controls, history and augmentation effects."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256
from summarize_causal_r5 import paired,method

BASE=ROOT/'research/4.temporal-consistency'
KEYS=['psnr_rgb','ssim_rgb','temporal_error','edge_mae']


def jsonl(path):return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    rows,frames,hashes,repeats=[],[],{},[]
    for scope,expected in [('head',1520),('delta',1216)]:
        folder=BASE/'results'/('exploration-r8-eval-'+scope)
        values=jsonl(folder/'per_sequence.jsonl')
        if len(values)!=expected or json.loads((folder/'runtime.json').read_text())['rows']!=expected:raise ValueError('incomplete fixed evaluation')
        rows+=values;frames+=jsonl(folder/'per_frame.jsonl')
        repeats+=json.loads((folder/'plain_repeat_audit.json').read_text())
        for r in json.loads((folder/'checkpoints.json').read_text()):
            if sha256(ROOT/r['path'])!=r['sha256']:raise ValueError('checkpoint changed')
        hashes.update({str(path.relative_to(ROOT)):sha256(path) for path in folder.iterdir() if path.is_file()})
    if len({(r['scene'],r['variant'],r['train_seed'],r['history'],r['input_reset'],r['fusion']) for r in rows})!=2736:
        raise ValueError('duplicate eval rows')
    grouped=collections.defaultdict(list)
    for r in rows:grouped[(r['domain'],r['variant'],r['history'],r['input_reset'],r['fusion'])].append(r)
    summary=[{'domain':d,'variant':v,'history':h,'input_reset':reset,'fusion':f,'evaluations':len(rr),
        **{k:float(np.mean([r[k] for r in rr])) for k in KEYS}}
        for (d,v,h,reset,f),rr in sorted(grouped.items())]
    save_json(out/'domain_summary.json',summary)
    contrasts=[('spatial_copy','causal','plain','causal'),('spatial_detail','causal','plain','causal'),
        ('temporal_concat','causal','spatial_copy','causal'),('temporal_delta','causal','spatial_detail','causal'),
        ('temporal_delta_aug','causal','temporal_delta','causal')]
    contrasts += [(v,'causal',v,'current') for v in ['temporal_concat','temporal_delta','temporal_delta_aug']]
    deltas=[]
    for reset in [False,True]:
        rr=[{**r,'method':method(r['variant'],r['history'],r['fusion'])} for r in rows if r['input_reset']==reset]
        for a,ah,b,bh in contrasts:
            for fusion in ['raw','half01']:
                for domain in ['test','legacy_synthetic','event_light_step','event_cut']:
                    deltas.append({**paired(rr,method(a,ah,fusion),method(b,bh,fusion),domain),'input_reset':reset})
    save_json(out/'paired_deltas.json',deltas)
    training,augmentation=[],collections.Counter()
    for scope in ['head','delta']:
        for folder in sorted((BASE/'results'/('exploration-r8-'+scope)).iterdir()):
            cfg=json.loads((folder/'config.json').read_text())
            trace=jsonl(folder/'training.jsonl')
            plain=jsonl(BASE/'results/exploration-r5-spatial'/f"plain-seed{cfg['seed']}"/'training.jsonl')
            if [r['step'] for r in trace]!=list(range(1,1001)):raise ValueError('incomplete training')
            if [r['sample_identity'] for r in trace]!=[r['sample_identity'] for r in plain]:raise ValueError('current stream mismatch')
            for path in folder.glob('checkpoint-step*.json'):
                r=json.loads(path.read_text())
                if sha256(ROOT/r['path'])!=r['sha256']:raise ValueError('saved checkpoint changed')
            initial=json.loads((folder/'initial_plain_audit.json').read_text())
            for r in trace:
                for action in r['history_augmentation']:
                    expected='current' if action['draw']<.1 else 'other_pair_previous' if action['draw']<.2 else 'actual_previous'
                    if action['history']!=expected:raise ValueError('augmentation rule mismatch')
                    augmentation[expected]+=1
            training.append({'variant':cfg['variant'],'seed':cfg['seed'],'steps':1000,'parameters':cfg['parameters'],
                'initial_plain_audit':initial,'last100_task':float(np.mean([r['task'] for r in trace[-100:]])),
                'final_auxiliary_l2':trace[-1]['auxiliary_l2']})
    if len(training)!=10:raise ValueError('training budget mismatch')
    save_json(out/'training_audit.json',{'runs':training,'total_new_updates':10000,'all_current_streams_match_R5_plain':True,
        'augmentation_actions':dict(augmentation),'reused_baseline':'R5 plain seed0/1 step1000; same batch/loss/backbone optimizer budget'})
    event=[]
    for r in frames:
        if r['domain'].startswith('event_') and r['frame']==12:event.append(r)
    save_json(out/'event_frame12.json',event)
    save_json(out/'audit.json',{'new_training_runs':10,'new_training_updates':10000,'evaluations':2736,'input_conditions':38,
        'plain_checkpoint_repeat_max_abs':max(v for r in repeats for v in r['max_abs_by_metric'].values()),
        'evaluation_sources':hashes,'scope':'reused exploratory data, slow trunk/fast auxiliary controlled tuning; no final method or independent test claim'})
    lines=['# Raw-pair head and static optimizer controls','',
        'Final1000 updates, two seeds. Spatial copy is foldable; causal-current isolates actual history at the same weights.','']
    for domain in ['test','legacy_synthetic','event_light_step','event_cut']:
        lines+=['## '+domain,'','| Variant | History | Reset | Fusion | PSNR | SSIM | Temporal |','|---|---|---|---|---:|---:|---:|']
        for r in summary:
            if r['domain']==domain:
                lines.append(f"| {r['variant']} | {r['history']} | {r['input_reset']} | {r['fusion']} | {r['psnr_rgb']:.5f} | {r['ssim_rgb']:.5f} | {r['temporal_error']:.5f} |")
        lines.append('')
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
