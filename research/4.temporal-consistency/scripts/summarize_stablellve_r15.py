"""Audit official-clean zero-future transfer and compare task-tuned references."""
import collections
import json
from pathlib import Path

import numpy as np

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256
from summarize_causal_r5 import KEYS, jsonl, means, paired

BASE=ROOT/'research/4.temporal-consistency'
DOMAINS=['test','dev','dev_native','legacy_synthetic','event_cut','event_light_step']


def label(r):return '|'.join(str(r[k]) for k in ['domain','scene','variant','input_reset','fusion'])


def main():
    source=BASE/'results/exploration-r15-eval';out=BASE/'report/exploration-r15';out.mkdir(parents=True,exist_ok=False)
    rows=jsonl(source/'per_sequence.jsonl');runtime=json.loads((source/'runtime.json').read_text())
    assert len(rows)==runtime['rows']==184 and len({label(r) for r in rows})==184
    seen={}
    for r in rows:
        if r['metric_alias_of'] is not None:
            ref=seen[r['metric_alias_of']]
            assert (r['domain'],r['scene'],r['fusion'])==(ref['domain'],ref['scene'],ref['fusion'])
            assert all(r[k]==ref[k] for k in KEYS)
        assert all(np.isfinite(r[k]) for k in KEYS);seen[label(r)]=r
    assert sum(r['metric_alias_of'] is None for r in rows)==runtime['metric_computations']==116
    assert sum(r['metric_alias_of'] is not None for r in rows)==runtime['metric_aliases']==68
    seen={}
    for r in jsonl(source/'per_frame.jsonl'):
        if r['metric_alias_of'] is not None:
            ref=seen[(r['metric_alias_of'],r['frame'])];assert all(r[k]==ref[k] for k in KEYS)
        seen[(label(r),r['frame'])]=r
    events=json.loads((source/'input_events.json').read_text());assert len(events)==46
    for r in events:assert r['reset_frames']==([12] if r['domain'].startswith('event_') else [])
    cfg=json.loads((source/'config.json').read_text());manifest=BASE/'data/exploration-r15-stablellve/manifest.json'
    assert sha256(manifest)==cfg['manifest_sha256'];m=json.loads(manifest.read_text());assert sha256(ROOT/m['path'])==m['sha256']
    smoke=BASE/'results/exploration-r15-smoke/smoke.json';assert sha256(smoke)==cfg['smoke_sha256']
    comparisons=[{**r,'method':'StableLLVE_clean_'+r['fusion']} for r in rows]
    old=jsonl(BASE/'results/exploration-r11-eval-raw/per_sequence.jsonl')
    references=[{**r,'method':'MobileIE_resize1000_'+r['fusion']} for r in old if r['variant']=='plain_high']
    for scope in ['spatial']:
        references += [{**r,'method':'MobileIE_'+r['train_preparation']+'4000_'+r['fusion']} for r in
            jsonl(BASE/'results'/('exploration-r14-eval-'+scope)/'per_sequence.jsonl') if r['variant']=='plain' and r['step']==4000]
    comparisons+=references
    groups=collections.defaultdict(list)
    for r in comparisons:groups[(r['domain'],r['method'],r['input_reset'])].append(r)
    summary=[{'domain':d,'method':m,'input_reset':s,'rows':len(rr),**means(rr)} for (d,m,s),rr in sorted(groups.items())]
    save_json(out/'domain_summary.json',summary)
    deltas=[]
    for reset in [False,True]:
        rr=[r for r in comparisons if r['input_reset']==reset]
        for domain in DOMAINS:
            deltas.append({**paired(rr,'StableLLVE_clean_half01','StableLLVE_clean_raw',domain),'input_reset':reset})
            for fusion in ['raw','half01']:
                for model in ['resize1000','resized4000','native_crop4000']:
                    deltas.append({**paired(rr,'StableLLVE_clean_'+fusion,'MobileIE_'+model+'_'+fusion,domain),'input_reset':reset})
    save_json(out/'paired_deltas.json',deltas)
    groups=collections.defaultdict(list)
    for r in json.loads((source/'stage_cost.json').read_text()):groups[tuple(r['shape'][2:])].append(r['ms_per_frame'])
    save_json(out/'stage_cost_summary.json',{'rows':[{'HW':list(hw),'sequences':len(v),'median_ms_per_frame':float(np.median(v)),
        'p10_p90_ms':np.percentile(v,[10,90]).tolist()} for hw,v in sorted(groups.items())],
        'scope':cfg['stage_timing_scope']})
    save_json(out/'audit.json',{'rows':184,'actual_inference_sequences':46,'metric_computations':116,'metric_aliases':68,
        'sequence_and_frame_alias_exact':True,'future_frames':0,'parameters':cfg['parameters'],'weight_sha256':m['sha256'],
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in source.iterdir() if p.is_file()},
        'scope':'official clean checkpoint transfer vs task-tuned references; unequal training, not a fair method ranking or noisy-paper reproduction'})
    lines=['# Official StableLLVE clean-checkpoint transfer','','No retraining or extra input noise; zero future. Task-tuned references have different data/training.','',
        '| Domain | Method | Reset | PSNR | SSIM | Temporal | Edge |','|---|---|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['domain']} | {r['method']} | {r['input_reset']} | {r['psnr_rgb']:.5f} | {r['ssim_rgb']:.5f} | {r['temporal_error']:.5f} | {r['edge_mae']:.5f} |")
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
