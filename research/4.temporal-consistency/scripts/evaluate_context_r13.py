"""Reuse the verified inference routing while changing only training context."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from evaluate_aligned_raw_r11 import infer
from exploration_r2_data import ROOT, detect_changes, evaluate
from explore_state_r2 import fuse
from explore_temporal_controls import save_json, sha256
from prepare_cache_r5 import CACHE_INDEX, load_one
from prepare_development_r9 import DEV_INDEX
from prepare_native_r10 import INDEX as NATIVE_INDEX
from raw_pair_head_r8 import RawPairHead
from train_exploration_r2 import new_model

ALIASES={'concat_context':'concat_high','concat_aligned_context':'concat_aligned_high'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variants',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();variants=args.variants.split(',')
    if len(set(variants))!=len(variants) or any(v not in ALIASES for v in variants):p.error('registered temporal variants required')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    replay=ROOT/'research/4.temporal-consistency/results/exploration-r13-raw/copy_context_replay-seed0/spatial_replay_audit.json'
    r=json.loads(replay.read_text())
    if max(r['max_abs_trace_delta'],r['max_abs_model_state_delta'])>1e-7:raise ValueError('spatial replay prerequisite failed')
    cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda');revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    sources={str(f.relative_to(ROOT)):sha256(f) for f in Path(__file__).parent.glob('*.py')}
    models=[];checkpoints=[]
    for variant in variants:
        for seed in [0,1]:
            path=ROOT/'.cache/checkpoints/exploration-r13'/f'{variant}-seed{seed}'/'step1000.pth'
            saved=torch.load(path,map_location=device,weights_only=False);assert saved['step']==1000
            net=RawPairHead(new_model(device),'temporal_concat').to(device);net.load_state_dict(saved['model'])
            models.append((variant,seed,net.eval()));checkpoints.append({'variant':variant,'seed':seed,'path':str(path.relative_to(ROOT)),'sha256':sha256(path)})
    save_json(out/'checkpoints.json',checkpoints)
    save_json(out/'config.json',{'variants':variants,'protocol_sha256':sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r13-protocol.md'),
        'spatial_replay_sha256':sha256(replay),'inference_aliases':ALIASES,'routing_source_sha256':sha256(ROOT/'research/4.temporal-consistency/scripts/evaluate_aligned_raw_r11.py'),
        'cache_indices':{str(f.relative_to(ROOT)):sha256(f) for f in [CACHE_INDEX,DEV_INDEX,NATIVE_INDEX]},
        'git_revision_at_startup':revision,'source_sha256_at_startup':sources,
        'scope':'same inference as R11, only first-task training history changed; native/resized dev share four source clips'})
    metas=json.loads(CACHE_INDEX.read_text())['scenes']+json.loads(DEV_INDEX.read_text())['scenes']
    metas += [{**r,'domain':'dev_native'} for r in json.loads(NATIVE_INDEX.read_text())['scenes'] if r['domain']=='native_crop']
    assert len(metas)==46
    start=time.perf_counter();count=0;first=[];events_audit=[]
    for meta in metas:
        data=load_one(meta);events,_=detect_changes(data['low'])
        events_audit.append({'domain':meta['domain'],'scene':meta['scene'],'reset_frames':np.flatnonzero(events).tolist()})
        inputs=torch.from_numpy((data['low']/255.).transpose(0,3,1,2).copy()).to(device)
        for variant,seed,net in models:
            first_output=None
            for history in ['causal','current']:
                for reset in [False,True]:
                    prediction=infer(net,inputs,data,ALIASES[variant],history,events if reset else None)
                    if not bool(torch.isfinite(prediction).all()):raise FloatingPointError('nonfinite context-model output')
                    raw=np.ascontiguousarray(prediction.clamp(0,1).permute(0,2,3,1).cpu().numpy()*255.)
                    if first_output is None:first_output=raw[0].copy()
                    delta=float(np.max(np.abs(raw[0]-first_output)))
                    if delta>1e-4:raise AssertionError('first-frame history/reset changed')
                    first.append({'domain':meta['domain'],'scene':meta['scene'],'variant':variant,'train_seed':seed,'history':history,'reset':reset,'max_abs_pixel':delta})
                    for fusion,output in [('raw',raw),('half01',fuse(raw,data['half_flow'],data['half_mask'],resets=events if reset else None,alpha=.1))]:
                        identity={'domain':meta['domain'],'scene':meta['scene'],'variant':variant,'train_seed':seed,'history':history,'input_reset':reset,'fusion':fusion}
                        metrics,frames=evaluate(output,data['gt'],data['reference_flow'],data['reference_mask'])
                        with (out/'per_sequence.jsonl').open('a') as stream:stream.write(json.dumps({**identity,**metrics},allow_nan=False)+'\n')
                        with (out/'per_frame.jsonl').open('a') as stream:
                            for frame in frames:stream.write(json.dumps({**identity,**frame},allow_nan=False)+'\n')
                        count+=1
        print(json.dumps({'domain':meta['domain'],'scene':meta['scene'],'rows':count}),flush=True)
    assert count==736*len(variants)
    save_json(out/'first_frame_audit.json',first);save_json(out/'input_events.json',events_audit)
    save_json(out/'runtime.json',{'rows':count,'wall_seconds':time.perf_counter()-start})


if __name__=='__main__':main()
