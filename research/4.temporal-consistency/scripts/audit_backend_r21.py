"""Report full CPU/GPU numerical drift and require exact common-CPU baseline replays."""
import json
from pathlib import Path

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256
from summarize_causal_r5 import jsonl

BASE=ROOT/'research/4.temporal-consistency'
KEYS=['psnr_rgb','ssim_rgb','edge_mae','temporal_error','flicker','output_mean','target_mean']


def compare(a,b):
    delta={}
    for key in KEYS:
        if key not in a or key not in b:continue
        if a[key] is None or b[key] is None:assert a[key] is None and b[key] is None
        else:delta[key]=abs(a[key]-b[key])
    return delta


def main():
    sources={};cross=[];exact=[]
    def read(folder,name):
        p=BASE/'results'/folder/name;sources[str(p.relative_to(ROOT))]=sha256(p);return jsonl(p)
    for file in ['per_sequence.jsonl','per_frame.jsonl']:
        extra=['frame'] if file=='per_frame.jsonl' else []
        plain_id=['domain','scene','train_seed','step','input_reset','fusion',*extra]
        def pkey(r):return tuple(r[k] for k in plain_id)
        old={pkey(r):r for r in read('exploration-r14-eval-spatial',file) if r['variant']=='plain' and r['train_preparation']=='native_crop'}
        cpu=read('exploration-r21-reference-plain-cpu',file);cpu_by_id={pkey(r):r for r in cpu}
        assert old.keys()==cpu_by_id.keys()
        maximum={};worst={}
        for r in cpu:
            for k,v in compare(r,old[pkey(r)]).items():
                if k not in maximum or v>maximum[k]:maximum[k]=v;worst[k]=pkey(r)
        cross.append({'scope':'R14 native plain CPU minus GPU','file':file,'rows':len(cpu),'max_abs':maximum,'worst_identity':worst})
        for seed in [0,1]:
            keynames=['domain','scene','train_seed','method','input_reset',*extra]
            def key(r):return tuple(r[k] for k in keynames)
            gpu=read(f'exploration-r20-seed{seed}',file);post=read(f'exploration-r21-reference-post-seed{seed}-cpu',file)
            gpu_by_id={key(r):r for r in gpu};assert {key(r) for r in post}==gpu_by_id.keys()
            maximum={};worst={};replay=0;replay_max=0.
            for r in post:
                for k,v in compare(r,gpu_by_id[key(r)]).items():
                    if k not in maximum or v>maximum[k]:maximum[k]=v;worst[k]=key(r)
                if r['method'] in ['raw','recursive_half01']:
                    ref={**r,'step':4000,'fusion':'raw' if r['method']=='raw' else 'half01'}
                    d=compare(r,cpu_by_id[pkey(ref)]);replay_max=max(replay_max,max(d.values(),default=0.));replay+=1
            assert replay_max<=1e-10
            exact.append({'seed':seed,'file':file,'rows':replay,'max_abs':replay_max})
            cross.append({'scope':f'R20 full controls seed{seed} CPU minus GPU','file':file,'rows':len(post),'max_abs':maximum,'worst_identity':worst})
    save_json(BASE/'process/exploration-r21/backend-audit.json',{'cross_backend':cross,'common_CPU_exact_replays':exact,
        'source_sha256':sources,'scope':'CPU primary comparisons; cross-backend drift retained, no relaxed equality claim.'})


if __name__=='__main__':main()
