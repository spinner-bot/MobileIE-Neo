"""Final-step raw-input head controls and same-weight current-history ablations."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from exploration_r2_data import ROOT,detect_changes,evaluate
from explore_state_r2 import fuse
from explore_temporal_controls import save_json,sha256
from frozen_mix_utils_r7 import provenance
from prepare_cache_r5 import CACHE_INDEX,load_one
from raw_pair_head_r8 import MODES,RawPairHead
from train_exploration_r2 import new_model


@torch.inference_mode()
def infer(model,inputs,history,resets):
    if isinstance(model,RawPairHead):model.reset(history)
    outputs=[]
    for t,x in enumerate(inputs):
        if resets is not None and resets[t] and isinstance(model,RawPairHead):model.reset(history)
        outputs.append(model(x[None])[0])
    return torch.stack(outputs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    variants=args.variants.split(',')
    if len(set(variants))!=len(variants) or any(v not in ('plain',*MODES) for v in variants):p.error('unique known variants required')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda')
    models,checkpoints=[],[]
    prior_plain={}
    for line in (ROOT/'research/4.temporal-consistency/results/exploration-r5-eval-spatial/per_sequence.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['variant']=='plain':prior_plain[(r['scene'],r['train_seed'],r['post'])]=r
    for variant in variants:
        for seed in [0,1]:
            scope='exploration-r5' if variant=='plain' else 'exploration-r8'
            path=ROOT/'.cache/checkpoints'/scope/f'{variant}-seed{seed}'/'step1000.pth'
            saved=torch.load(path,map_location=device,weights_only=False)
            if saved['step']!=1000:raise ValueError('fixed final checkpoint required')
            base=new_model(device)
            if variant=='plain':
                base.load_state_dict({k.removeprefix('backbone.'):v for k,v in saved['model'].items()})
                net=base
            else:
                net=RawPairHead(base,variant).to(device)
                net.load_state_dict(saved['model'])
            net.eval()
            histories=['causal','current'] if variant.startswith('temporal_') else ['causal']
            models.append((variant,seed,net,histories))
            checkpoints.append({'variant':variant,'seed':seed,'path':str(path.relative_to(ROOT)),'sha256':sha256(path)})
    source=provenance();source['previous_protocol_sha256']=source.pop('protocol_sha256')
    save_json(out/'config.json',{**source,'protocol_sha256':sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r8-protocol.md'),
        'variants':variants,'checkpoint_step':1000,'cache_index_sha256':sha256(CACHE_INDEX),
        'shared_motion_alpha':.1,'reset':'fixed R2 input detector for feature/RGB history and common output fusion',
        'scope':'same train-stream final checkpoints; shared teacher half-flow not free'})
    save_json(out/'checkpoints.json',checkpoints)
    start,count,repeats=time.perf_counter(),0,[]
    for meta in json.loads(CACHE_INDEX.read_text())['scenes']:
        data=load_one(meta)
        events,_=detect_changes(data['low'])
        inputs=torch.from_numpy((data['low']/255.).transpose(0,3,1,2).copy()).to(device)
        for variant,seed,net,histories in models:
            for history in histories:
                for reset in [False,True]:
                    prediction=infer(net,inputs,history,events if reset else None)
                    raw=np.ascontiguousarray(prediction.clamp(0,1).permute(0,2,3,1).cpu().numpy()*255.)
                    for fusion,output in [('raw',raw),('half01',fuse(raw,data['half_flow'],data['half_mask'],resets=events if reset else None,alpha=.1))]:
                        identity={'domain':meta['domain'],'scene':meta['scene'],'variant':variant,'train_seed':seed,
                            'history':history,'input_reset':reset,'fusion':fusion}
                        metrics,frames=evaluate(output,data['gt'],data['reference_flow'],data['reference_mask'])
                        with (out/'per_sequence.jsonl').open('a') as stream:stream.write(json.dumps({**identity,**metrics},allow_nan=False)+'\n')
                        with (out/'per_frame.jsonl').open('a') as stream:
                            for frame in frames:stream.write(json.dumps({**identity,**frame},allow_nan=False)+'\n')
                        if variant=='plain' and not reset:
                            ref=prior_plain[(meta['scene'],seed,fusion)]
                            delta={k:abs(metrics[k]-ref[k]) for k in ['psnr_rgb','ssim_rgb','temporal_error','edge_mae']}
                            if max(delta.values())>1e-6:raise AssertionError('reused plain checkpoint repeat changed')
                            repeats.append({**identity,'max_abs_by_metric':delta})
                        count+=1
        print(json.dumps({'scene':meta['scene'],'rows':count}),flush=True)
    expected=sum(304 if v.startswith('spatial_') or v=='plain' else 608 for v in variants)
    if count!=expected:raise ValueError('fixed evaluation budget mismatch')
    save_json(out/'plain_repeat_audit.json',repeats)
    save_json(out/'runtime.json',{'wall_seconds':time.perf_counter()-start,'rows':count})


if __name__=='__main__':main()
