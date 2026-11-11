"""Separate spatial capacity, matched-task training depth and actual temporal information."""
import collections
import hashlib
import json

import numpy as np
import torch

from evaluation_aliases_r16 import cache_keys
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256
from multiscale_fst_r16 import MODES
from summarize_causal_r5 import KEYS,jsonl,means,paired

BASE=ROOT/'research/4.temporal-consistency'
DOMAINS=['test','dev','dev_native','legacy_synthetic','event_cut','event_light_step']
IDENTITY=['domain','scene','variant','train_seed','step','history','input_reset','fusion']


def label(r):return '|'.join(str(r[k]) for k in IDENTITY)
def method(variant,step,history,fusion):return '__'.join([variant,str(step),history,fusion])


def main():
    out=BASE/'report/exploration-r16';out.mkdir(parents=True,exist_ok=False)
    rows=[];first=[];runtime=[];sources={};training=[];streams={};checkpoints=0
    for scope,expected,train_scope,train_variant in [('spatial',736,'spatial-raw','spatial'),
            ('raw',1472,'spatial-raw','raw_history'),('aligned',1472,'aligned','aligned_history')]:
        folder=BASE/'results'/('exploration-r16-eval-'+scope);values=jsonl(folder/'per_sequence.jsonl')
        run=json.loads((folder/'runtime.json').read_text());assert len(values)==run['rows']==expected
        events=json.loads((folder/'input_events.json').read_text());assert len(events)==46
        event_map={(r['domain'],r['scene']):bool(r['reset_frames']) for r in events};seen={}
        for r in values:
            _,key=cache_keys(r['variant'],r['history'],r['input_reset'],event_map[(r['domain'],r['scene'])],r['fusion'])
            assert (r['effective_model_reset'],r['effective_output_reset'])==(key[0],key[2])
            if r['metric_alias_of'] is not None:
                ref=seen[r['metric_alias_of']];assert all(r[k]==ref[k] for k in IDENTITY if k!='input_reset')
                assert all(r[k]==ref[k] for k in KEYS)
            assert label(r) not in seen;seen[label(r)]=r
        assert sum(r['metric_alias_of'] is None for r in values)==run['metric_computations']
        seen={}
        for r in jsonl(folder/'per_frame.jsonl'):
            if r['metric_alias_of'] is not None:assert all(r[k]==seen[(r['metric_alias_of'],r['frame'])][k] for k in KEYS)
            seen[(label(r),r['frame'])]=r
        for item in json.loads((folder/'checkpoints.json').read_text()):assert sha256(ROOT/item['path'])==item['sha256']
        rows+=values;runtime.append(run);first+=json.loads((folder/'first_frame_audit.json').read_text())
        sources.update({str(p.relative_to(ROOT)):sha256(p) for p in folder.iterdir() if p.is_file()})
        trainroot=BASE/'results'/('exploration-r16-'+train_scope)
        integrity=json.loads((trainroot/'source_integrity.json').read_text());assert integrity['verified_files']==384
        for folder in sorted(p for p in trainroot.iterdir() if p.is_dir()):
            cfg=json.loads((folder/'config.json').read_text());trace=jsonl(folder/'training.jsonl');run=json.loads((folder/'runtime.json').read_text())
            if cfg['variant']!=train_variant:continue
            assert len(trace)==4000 and [r['step'] for r in trace]==list(range(1,4001)) and run['new_updates']==4000
            assert cfg['parameters']==24955
            oldpath=BASE/'results/exploration-r14-spatial'/f"native_crop-plain-seed{cfg['seed']}"/'training.jsonl'
            assert sha256(oldpath)==cfg['matched_r14_trace_sha256'];old=jsonl(oldpath);digest=hashlib.sha256()
            for r,ref in zip(trace,old):
                content=[r[k] for k in ['sample_identity','real_context','synthetic_context']]
                assert content==[ref[k] for k in ['sample_identity','real_context','synthetic_context']]
                digest.update(json.dumps(content,sort_keys=True).encode())
                assert len(r['alignment'])==(4 if cfg['variant']=='aligned_history' else 0)
                assert all(not d['cache_hit'] for d in r['alignment'])
            stream=digest.hexdigest()
            if cfg['seed'] in streams:assert streams[cfg['seed']]==stream
            streams[cfg['seed']]=stream
            initial=json.loads((folder/'initial_plain_audit.json').read_text());assert initial['output_max_abs']<=2e-5
            raw=(folder/'training.jsonl').read_bytes()
            for step in [1000,2000,3000,4000]:
                item=json.loads((folder/f'checkpoint-step{step:04d}.json').read_text());path=ROOT/item['path'];assert sha256(path)==item['sha256']
                saved=torch.load(path,map_location='cpu',weights_only=False);assert saved['step']==step
                assert sum(v.numel() for v in saved['model'].values())==24955
                assert all(bool(torch.isfinite(v).all()) for v in saved['model'].values())
                assert all(float(v['step'])==step for v in saved['optimizer']['state'].values());checkpoints+=1
                milestone=json.loads((folder/f'milestone-step{step:04d}.json').read_text())
                assert hashlib.sha256(raw[:milestone['trace_prefix_bytes']]).hexdigest()==milestone['trace_prefix_sha256']
            training.append({'variant':cfg['variant'],'seed':cfg['seed'],'new_updates':4000,'initial_max_abs':initial['output_max_abs'],
                'last100_task':float(np.mean([r['task'] for r in trace[-100:]])),
                'steps_clipped':sum(r['gradient_norm_all']>1 for r in trace),'task_context_sha256':stream})
    assert len(rows)==3680 and len({label(r) for r in rows})==3680
    assert len(training)==6 and checkpoints==24 and sum(r['new_updates'] for r in training)==24000
    assert sum(r['actual_inference_sequences'] for r in runtime)==1112
    assert sum(r['metric_computations'] for r in runtime)==2512
    assert sum(r['metric_aliases'] for r in runtime)==1168
    # Match fixed native R14 references; all selections are protocol-defined, not metric-selected.
    references=[]
    for scope in ['spatial','aligned']:
        for r in jsonl(BASE/'results'/('exploration-r14-eval-'+scope)/'per_sequence.jsonl'):
            if r['train_preparation']=='native_crop':references.append({**r,'variant':'R14_'+r['variant']})
    combined=rows+references;groups=collections.defaultdict(list);clips=collections.defaultdict(list)
    for r in combined:
        key=(r['domain'],r['variant'],r['step'],r['history'],r['input_reset'],r['fusion']);groups[key].append(r);clips[(r['scene'],*key)].append(r)
    summary=[{'domain':d,'variant':v,'step':s,'history':h,'input_reset':reset,'fusion':f,'rows':len(rr),**means(rr)}
        for (d,v,s,h,reset,f),rr in sorted(groups.items())]
    save_json(out/'domain_summary.json',summary)
    save_json(out/'by_scene.json',[{'scene':c,'domain':d,'variant':v,'step':s,'history':h,'input_reset':reset,'fusion':f,**means(rr)}
        for (c,d,v,s,h,reset,f),rr in sorted(clips.items())])
    contrasts=[]
    for step in [1000,4000]:
        contrasts += [('spatial_capacity',('spatial',step,'causal'),('R14_plain',step,'causal')),
            ('spatial_capacity',('spatial',step,'causal'),('R14_copy',step,'causal'))]
        for variant in ['raw_history','aligned_history']:
            contrasts += [('actual_history',(variant,step,'causal'),(variant,step,'current')),
                ('same_capacity',(variant,step,'causal'),('spatial',step,'causal'))]
    for variant in MODES:
        for history in (['causal'] if variant=='spatial' else ['causal','current']):
            contrasts.append(('training_depth',(variant,4000,history),(variant,1000,history)))
    deltas=[]
    for reset in [False,True]:
        rr=[{**r,'method':method(r['variant'],r['step'],r['history'],r['fusion'])} for r in combined if r['input_reset']==reset]
        for kind,a,b in contrasts:
            for fusion in ['raw','half01']:
                for domain in DOMAINS:deltas.append({**paired(rr,method(*a,fusion),method(*b,fusion),domain),'contrast':kind,'input_reset':reset})
    save_json(out/'paired_deltas.json',deltas)
    save_json(out/'training_audit.json',{'runs':training,'new_updates':24000,'checkpoints_verified':24,'all_task_contexts_match_R14':True})
    save_json(out/'audit.json',{'new_rows':3680,'reused_R14_rows':len(references),'actual_inference_sequences':1112,
        'metric_computations':2512,'metric_aliases':1168,'sequence_and_frame_alias_exact':True,
        'first_frame_max_abs_pixel':max(r['max_abs_pixel'] for r in first),'source_sha256':sources,
        'scope':'reused exploration clips, two seeds; standard multiscale controls, not final method or novelty claim'})
    lines=['# Multiscale FST controls','','Raw/half01, fixed 1000/4000, all candidates. Standard architecture reference, not novelty.','',
        '| Domain | Variant | Step | History | Reset | Fusion | PSNR | SSIM | Temporal | Edge |',
        '|---|---|---:|---|---|---|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['domain']} | {r['variant']} | {r['step']} | {r['history']} | {r['input_reset']} | {r['fusion']} | {r['psnr_rgb']:.5f} | {r['ssim_rgb']:.5f} | {r['temporal_error']:.5f} | {r['edge_mae']:.5f} |")
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
