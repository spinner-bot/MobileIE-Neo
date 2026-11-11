"""Audit same-flow dynamic controls and separate aggregation from warp/state order."""
import collections
import json

import numpy as np

from aggregation_controls_r20 import METHODS
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256
from summarize_causal_r5 import KEYS,jsonl,means,paired

BASE=ROOT/'research/4.temporal-consistency'
DOMAINS=['test','dev','dev_native','legacy_synthetic','event_cut','event_light_step']
ID=['domain','scene','train_seed','method','input_reset']


def label(r):return '|'.join(str(r[k]) for k in ID)


def main():
    out=BASE/'report/exploration-r20';out.mkdir(parents=True,exist_ok=False)
    rows=[];frames=[];replays=[];cost=[];first=[];sources={};events_reference=None;runtime=[]
    for seed in [0,1]:
        folder=BASE/'results'/f'exploration-r20-seed{seed}';values=jsonl(folder/'per_sequence.jsonl');run=json.loads((folder/'runtime.json').read_text())
        assert len(values)==run['rows']==736 and run['metric_computations']==536 and run['metric_aliases']==200
        cfg=json.loads((folder/'config.json').read_text());assert cfg['seed']==seed and tuple(cfg['methods'])==METHODS
        assert sha256(ROOT/cfg['checkpoint']['path'])==cfg['checkpoint']['sha256']
        events=json.loads((folder/'input_events.json').read_text());assert len(events)==46
        if events_reference is not None:assert events==events_reference
        events_reference=events;event_map={(r['domain'],r['scene']):bool(r['reset_frames']) for r in events};seen={}
        for r in values:
            assert r['train_seed']==seed and r['effective_reset']==bool(r['input_reset'] and r['method']!='raw' and event_map[(r['domain'],r['scene'])])
            if r['metric_alias_of'] is not None:
                ref=seen[r['metric_alias_of']];assert all(r[k]==ref[k] for k in ID if k!='input_reset')
                assert all(r[k]==ref[k] for k in KEYS+['flicker'])
            assert label(r) not in seen;seen[label(r)]=r
        assert sum(r['metric_alias_of'] is None for r in values)==536
        seen={};ff=jsonl(folder/'per_frame.jsonl');counts=collections.Counter()
        for r in ff:
            if r['metric_alias_of'] is not None:
                ref=seen[(r['metric_alias_of'],r['frame'])];assert all(r[k]==ref[k] for k in KEYS+['flicker','output_mean','target_mean'])
            seen[(label(r),r['frame'])]=r
            if r['method']=='raw' and not r['input_reset']:counts[(r['domain'],r['scene'])]+=1
        accounting=json.loads((folder/'compute_accounting.json').read_text());assert len(accounting)==46
        for r in accounting:
            n=counts[(r['domain'],r['scene'])];assert r['model_sequence_passes']==4 and r['model_forward_frames']==n+3*(n-1)
            assert len(r['stages'])==4 and len(r['warp_enhancement_commutator_MAE255'])==n-1
        assert run['model_forward_frames']==sum(r['model_forward_frames'] for r in accounting)==3830
        replay=json.loads((folder/'baseline_replays.json').read_text());assert len(replay)==184
        assert max(r['sequence_and_frame_max_abs_metric'] for r in replay)<=1e-10
        rows+=values;frames+=ff;replays+=replay;cost+=accounting;runtime.append(run)
        first+=json.loads((folder/'first_frame_audit.json').read_text())
        sources.update({str(p.relative_to(ROOT)):sha256(p) for p in folder.iterdir() if p.is_file()})
    assert len(rows)==1472 and len({label(r) for r in rows})==1472
    groups=collections.defaultdict(list);clips=collections.defaultdict(list)
    for r in rows:groups[(r['domain'],r['method'],r['input_reset'])].append(r);clips[(r['scene'],r['domain'],r['method'],r['input_reset'])].append(r)
    summary=[{'domain':d,'method':m,'input_reset':reset,'rows':len(rr),**means(rr)} for (d,m,reset),rr in sorted(groups.items())]
    save_json(out/'domain_summary.json',summary)
    save_json(out/'by_scene.json',[{'scene':c,'domain':d,'method':m,'input_reset':reset,**means(rr)} for (c,d,m,reset),rr in sorted(clips.items())])
    contrasts=[(m,'raw') for m in METHODS if m!='raw']
    for suffix in ['01','05']:
        contrasts += [('pre'+suffix,'postwarp'+suffix),('pre'+suffix,'postaligned'+suffix),('postaligned'+suffix,'postwarp'+suffix)]
    contrasts += [('postwarp01','recursive_half01'),('pre05','pre01'),('postwarp05','postwarp01'),('postaligned05','postaligned01')]
    deltas=[]
    for reset in [False,True]:
        rr=[r for r in rows if r['input_reset']==reset]
        for a,b in contrasts:
            for domain in DOMAINS:deltas.append({**paired(rr,a,b,domain),'input_reset':reset})
    save_json(out/'paired_deltas.json',deltas)
    event_rows=[{**r,'method':r['method']+'__reset'+str(r['input_reset'])} for r in rows if r['domain'].startswith('event_')]
    save_json(out/'event_reset_deltas.json',[paired(event_rows,m+'__resetTrue',m+'__resetFalse',domain)
        for m in METHODS for domain in ['event_cut','event_light_step']])
    save_json(out/'event_frames.json',[r for r in frames if r['domain'].startswith('event_') and 10<=r['frame']<=14])
    pareto=[]
    for reset in [False,True]:
        for domain in DOMAINS:
            candidates=[r for r in summary if r['domain']==domain and r['input_reset']==reset]
            def dominates(a,b):
                delta=[a['psnr_rgb']-b['psnr_rgb'],a['ssim_rgb']-b['ssim_rgb'],b['temporal_error']-a['temporal_error'],b['edge_mae']-a['edge_mae']]
                return min(delta)>=-1e-12 and max(delta)>1e-12
            for r in candidates:pareto.append({'domain':domain,'input_reset':reset,'method':r['method'],
                'dominated_by':[q['method'] for q in candidates if q['method']!=r['method'] and dominates(q,r)],
                'scope':'four metric means only, no compute/latency ranking; repeated exploration data'})
    save_json(out/'pareto_four_metrics.json',pareto)
    save_json(out/'compute_accounting.json',{'conditions_seed_rows':cost,'model_sequence_passes':368,'model_forward_frames':7660,
        'scope':'shared raw/aligned/pre computations across all controls; not standalone method speed, teacher/flow or phone cost'})
    save_json(out/'audit.json',{'rows':1472,'input_conditions':46,'train_seeds':[0,1],'metric_computations':1072,'metric_aliases':400,
        'sequence_and_frame_alias_exact':True,'baseline_sequence_and_frame_replays':368,
        'baseline_max_abs_metric':max(r['sequence_and_frame_max_abs_metric'] for r in replays),
        'first_frame_max_abs_pixel':max(r['first_frame_max_abs_pixel'] for r in first),'model_forward_frames':7660,
        'source_sha256':sources,'scope':'fixed same-flow controls; no new training or independent-test/innovation claim'})
    lines=['# Dynamic aggregation order, same flow and weights','','Finite controls use one prior; recursive baseline has a different state. All fixed alpha/reset candidates retained.','',
        '| Domain | Method | Reset | PSNR | SSIM | Temporal | Edge |','|---|---|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['domain']} | {r['method']} | {r['input_reset']} | {r['psnr_rgb']:.5f} | {r['ssim_rgb']:.5f} | {r['temporal_error']:.5f} | {r['edge_mae']:.5f} |")
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
