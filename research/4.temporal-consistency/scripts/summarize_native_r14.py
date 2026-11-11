"""Matched data/depth effects, true history use, and complete continuation audit."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from continuation_r14 import MODES
from evaluation_aliases_r14 import cache_keys
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json, sha256
from native_training_r14 import pair_identity
from summarize_causal_r5 import KEYS, means, paired

BASE=ROOT/'research/4.temporal-consistency'
DOMAINS=['test','dev','dev_native','legacy_synthetic','event_cut','event_light_step']
IDKEYS=['domain','scene','variant','train_preparation','train_seed','step','history','input_reset','fusion']


def jsonl(path):return [json.loads(s) for s in path.read_text().splitlines()]
def label(r):return '|'.join(str(r[k]) for k in IDKEYS)
def method(prep,variant,step,history,fusion):return '__'.join([prep,variant,str(step),history,fusion])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    out=Path(p.parse_args().out);out.mkdir(parents=True,exist_ok=False)
    rows=[];sources={};first=[];runtime=[]
    for scope in ['spatial','aligned']:
        folder=BASE/'results'/('exploration-r14-eval-'+scope)
        values=jsonl(folder/'per_sequence.jsonl');run=json.loads((folder/'runtime.json').read_text())
        assert len(values)==run['rows']==2208
        seen={};events=json.loads((folder/'input_events.json').read_text());assert len(events)==46
        events_map={(r['domain'],r['scene']):bool(r['reset_frames']) for r in events}
        for r in values:
            key=cache_keys(r['variant'],r['history'],r['input_reset'],events_map[(r['domain'],r['scene'])],r['fusion'])[1]
            assert (r['effective_model_reset'],r['effective_output_reset'])==(key[0],key[2])
            if r['metric_alias_of'] is not None:
                reference=seen[r['metric_alias_of']]
                assert all(r[k]==reference[k] for k in IDKEYS if k!='input_reset')
                assert all(r[k]==reference[k] for k in KEYS)
            assert label(r) not in seen;seen[label(r)]=r
        assert sum(r['metric_alias_of'] is None for r in values)==run['metric_computations']
        assert sum(r['metric_alias_of'] is not None for r in values)==run['metric_aliases']
        frame_seen={}
        for r in jsonl(folder/'per_frame.jsonl'):
            if r['metric_alias_of'] is not None:
                reference=frame_seen[(r['metric_alias_of'],r['frame'])]
                assert all(r[k]==reference[k] for k in KEYS)
            frame_seen[(label(r),r['frame'])]=r
        for r in json.loads((folder/'checkpoints.json').read_text()):assert sha256(ROOT/r['path'])==r['sha256']
        rows+=values;runtime.append({'scope':scope,**run});first+=json.loads((folder/'first_frame_audit.json').read_text())
        sources.update({str(f.relative_to(ROOT)):sha256(f) for f in folder.iterdir() if f.is_file()})
    assert len(rows)==4416
    for scope,mapping in [('exploration-r11-eval-raw',{'plain_high':'plain','copy_high':'copy'}),
                          ('exploration-r13-eval-aligned',{'concat_aligned_context':'aligned_concat'})]:
        path=BASE/'results'/scope/'per_sequence.jsonl';sources[str(path.relative_to(ROOT))]=sha256(path)
        for r in jsonl(path):
            if r['variant'] in mapping:rows.append({**r,'variant':mapping[r['variant']],'train_preparation':'resized','step':1000,'reused_prior_source':str(path.relative_to(ROOT))})
    assert len(rows)==5888==len({tuple(r[k] for k in IDKEYS) for r in rows})
    groups=collections.defaultdict(list);clips=collections.defaultdict(list)
    for r in rows:
        key=(r['domain'],r['train_preparation'],r['variant'],r['step'],r['history'],r['input_reset'],r['fusion'])
        groups[key].append(r);clips[(r['scene'],*key)].append(r)
    summary=[{'domain':d,'train_preparation':p,'variant':v,'step':s,'history':h,'input_reset':reset,'fusion':f,'rows':len(rr),**means(rr)}
             for (d,p,v,s,h,reset,f),rr in sorted(groups.items())]
    save_json(out/'domain_summary.json',summary)
    save_json(out/'by_scene.json',[{'scene':c,'domain':d,'train_preparation':p,'variant':v,'step':s,'history':h,'input_reset':reset,'fusion':f,**means(rr)}
             for (c,d,p,v,s,h,reset,f),rr in sorted(clips.items())])
    contrasts=[]
    for variant in MODES:
        for history in (['causal','current'] if variant=='aligned_concat' else ['causal']):
            for step in [1000,4000]:contrasts.append(('data_preparation',('native_crop',variant,step,history),('resized',variant,step,history)))
            for prep in ['native_crop','resized']:contrasts.append(('training_depth',(prep,variant,4000,history),(prep,variant,1000,history)))
    for prep in ['native_crop','resized']:
        for step in [1000,4000]:
            contrasts += [('spatial_control',(prep,'copy',step,'causal'),(prep,'plain',step,'causal')),
                ('actual_history',(prep,'aligned_concat',step,'causal'),(prep,'aligned_concat',step,'current')),
                ('same_capacity',(prep,'aligned_concat',step,'causal'),(prep,'copy',step,'causal'))]
    deltas=[]
    for reset in [False,True]:
        rr=[{**r,'method':method(r['train_preparation'],r['variant'],r['step'],r['history'],r['fusion'])} for r in rows if r['input_reset']==reset]
        for kind,a,b in contrasts:
            for fusion in ['raw','half01']:
                for domain in DOMAINS:deltas.append({**paired(rr,method(*a,fusion),method(*b,fusion),domain),'contrast':kind,'input_reset':reset})
    save_json(out/'paired_deltas.json',deltas)
    training=[];streams={};checkpoint_count=0;new_checkpoints=0;all_updates=0
    for scope in ['spatial','aligned']:
        root=BASE/'results'/('exploration-r14-'+scope);integrity=json.loads((root/'source_integrity.json').read_text())
        assert integrity['verified_files']==384 and len(integrity['scenes'])==12
        for folder in sorted(x for x in root.iterdir() if x.is_dir()):
            cfg=json.loads((folder/'config.json').read_text());run=json.loads((folder/'runtime.json').read_text());trace=jsonl(folder/'training.jsonl')
            start=cfg['start_step'];assert start==(0 if cfg['preparation']=='native_crop' else 1000)
            assert [r['step'] for r in trace]==list(range(start+1,4001))
            assert run['final_step']==4000 and run['new_updates']==4000-start
            if start:
                continuation=json.loads((folder/'continuation_audit.json').read_text())
                assert continuation['optimizer_step_range']==[1000,1000] and continuation['Adam_moments_exactly_preserved']
            old=jsonl(BASE/'results/exploration-r5-spatial'/f"plain-seed{cfg['seed']}"/'training.jsonl')
            digest=hashlib.sha256()
            for r in trace:
                step=r['step']-1;i,t=pair_identity(cfg['seed'],step,12);identity=r['sample_identity'][0]
                assert (identity['scene'],identity['frame_pair'])==(integrity['scenes'][i],t)
                if step<1000:assert r['sample_identity']==old[step]['sample_identity']
                c=r['real_context'];assert c['current_indices']==[t-1,t] and c['previous_indices']==[max(0,t-2),t-1]
                if cfg['preparation']=='native_crop':
                    rng=np.random.default_rng(1401400000+cfg['seed']*1000000+step)
                    assert c['crop_xywh']==[int(rng.integers(0,1601)),int(rng.integers(0,901)),320,180]
                    assert all(not a['cache_hit'] for a in r['alignment'])
                else:assert c['crop_xywh'] is None
                assert len(r['alignment'])==(4 if cfg['variant']=='aligned_concat' else 0)
                if r['step']>1000:digest.update(json.dumps([r['sample_identity'],c,r['synthetic_context']],sort_keys=True).encode())
            stream=digest.hexdigest();key=(cfg['preparation'],cfg['seed'])
            if key in streams:assert streams[key]==stream
            streams[key]=stream
            metas=list(folder.glob('checkpoint-step*.json'));assert len(metas)==4
            for f in metas:
                item=json.loads(f.read_text());path=ROOT/item['path'];assert sha256(path)==item['sha256'];checkpoint_count+=1
                if not item.get('reused',False):new_checkpoints+=1
                saved=torch.load(path,map_location='cpu',weights_only=False);assert saved['step']==item.get('step',1000)
                assert all(bool(torch.isfinite(v).all()) for v in saved['model'].values())
                assert sum(v.numel() for v in saved['model'].values())==cfg['parameters']
                assert all(float(state['step'])==saved['step'] for state in saved['optimizer']['state'].values())
            raw=(folder/'training.jsonl').read_bytes()
            for f in folder.glob('milestone-step*.json'):
                m=json.loads(f.read_text());assert hashlib.sha256(raw[:m['trace_prefix_bytes']]).hexdigest()==m['trace_prefix_sha256']
            all_updates+=len(trace)
            training.append({'variant':cfg['variant'],'preparation':cfg['preparation'],'seed':cfg['seed'],'start_step':start,'final_step':4000,
                'new_updates':len(trace),'last100_task':float(np.mean([r['task'] for r in trace[-100:]])),
                'steps_clipped':sum(r['gradient_norm_all']>1 for r in trace),'matched_last3000_task_stream_sha256':stream,
                'continuation_preserved':bool(start)})
    assert len(training)==12 and all_updates==42000 and checkpoint_count==48 and new_checkpoints==42
    save_json(out/'training_audit.json',{'runs':training,'new_updates':all_updates,'checkpoints_verified':checkpoint_count,
        'new_checkpoints':new_checkpoints,'reused1000_checkpoints':6,'all_variants_matched_task_streams':True,
        'scope':'native/resized change noise, content/FOV, motion pixels and global context jointly; no pure-noise attribution'})
    save_json(out/'audit.json',{'new_rows':4416,'reused_rows':1472,'total_logical_rows':5888,'input_conditions':46,
        'actual_new_inference_sequences':sum(r['actual_inference_sequences'] for r in runtime),
        'actual_new_metric_computations':sum(r['metric_computations'] for r in runtime),'new_metric_aliases':sum(r['metric_aliases'] for r in runtime),
        'sequence_and_frame_alias_values_exact':True,'first_frame_max_abs_pixel':max(r['max_abs_pixel'] for r in first),'sources':sources,
        'scope':'46 conditions include reused clips; not independent test; no final method or innovation claim'})
    lines=['# Matched native/resized training and cumulative depth','','Fixed all candidates/1000+4000; all Adam moments preserved on resized continuation. Logical rows include exact reset aliases.','']
    for domain in DOMAINS:
        lines+=['## '+domain,'','| Train prep | Variant | Step | History | Reset | Fusion | PSNR | SSIM | Temporal | Edge |',
                '|---|---|---:|---|---|---|---:|---:|---:|---:|']
        for r in summary:
            if r['domain']==domain:lines.append(f"| {r['train_preparation']} | {r['variant']} | {r['step']} | {r['history']} | {r['input_reset']} | {r['fusion']} | {r['psnr_rgb']:.5f} | {r['ssim_rgb']:.5f} | {r['temporal_error']:.5f} | {r['edge_mae']:.5f} |")
        lines.append('')
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
