"""CPU matched-backend R14 plain native references for R21 aggregation evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from continuation_r14 import MODES, create_model
from evaluate_aligned_raw_r11 import infer
from evaluation_aliases_r14 import cache_keys, event_flags
from exploration_r2_data import ROOT, detect_changes, evaluate
from explore_state_r2 import fuse
from explore_temporal_controls import save_json, sha256
from prepare_cache_r5 import CACHE_INDEX, load_one
from prepare_development_r9 import DEV_INDEX
from prepare_native_r10 import INDEX as NATIVE_INDEX

ALIASES={'plain':'plain_high','copy':'copy_high','aligned_concat':'concat_aligned_high'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variants',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();variants=args.variants.split(',')
    if variants!=['plain']:p.error('registered unique variants required')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cpu');revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    sources={str(f.relative_to(ROOT)):sha256(f) for f in Path(__file__).parent.glob('*.py')}
    models=[];checkpoints=[]
    for variant in variants:
        scope='aligned' if variant=='aligned_concat' else 'spatial'
        for preparation in ['native_crop']:
            for seed in [0,1]:
                folder=ROOT/'research/4.temporal-consistency/results'/('exploration-r14-'+scope)/f'{preparation}-{variant}-seed{seed}'
                runtime=json.loads((folder/'runtime.json').read_text())
                if runtime['final_step']!=4000:raise ValueError('complete registered depth run required')
                for step in ([1000,4000] if preparation=='native_crop' else [4000]):
                    item=json.loads((folder/f'checkpoint-step{step:04d}.json').read_text());path=ROOT/item['path']
                    if sha256(path)!=item['sha256']:raise ValueError('depth checkpoint changed')
                    saved=torch.load(path,map_location=device,weights_only=False)
                    if saved['step']!=step:raise ValueError('fixed checkpoint step mismatch')
                    net=create_model(variant,device);net.load_state_dict(saved['model'],strict=True)
                    if variant=='plain':net=net.backbone
                    models.append((variant,preparation,seed,step,net.eval()))
                    checkpoints.append({'variant':variant,'train_preparation':preparation,'train_seed':seed,'step':step,
                        'path':item['path'],'sha256':item['sha256'],'training_config_sha256':sha256(folder/'config.json')})
    save_json(out/'checkpoints.json',checkpoints)
    save_json(out/'config.json',{'device':'cpu','Torch_threads':2,'OpenCV_threads':1,'variants':variants,'steps':[1000,4000],'resized1000':'reused old evaluations in summary, not silently rerun here',
        'protocol_sha256':sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r14-protocol.md'),
        'cache_indices':{str(f.relative_to(ROOT)):sha256(f) for f in [CACHE_INDEX,DEV_INDEX,NATIVE_INDEX]},
        'git_revision_at_startup':revision,'source_sha256_at_startup':sources,
        'alias_policy':'no events => exact reset invariant; spatial/current raw ignores reset; output-fusion reset remains independent',
        'scope':'matched training depth and data preparation; native/resized dev share four physical clips; GPU/teacher flow costs not benchmarked here'})
    metas=json.loads(CACHE_INDEX.read_text())['scenes']+json.loads(DEV_INDEX.read_text())['scenes']
    metas += [{**r,'domain':'dev_native'} for r in json.loads(NATIVE_INDEX.read_text())['scenes'] if r['domain']=='native_crop']
    assert len(metas)==46
    start=time.perf_counter();count=0;inferences=0;computations=0;aliases=0;first=[];events_audit=[]
    for meta in metas:
        data=load_one(meta);events,_=detect_changes(data['low']);events=event_flags(events);has_events=bool(events.any())
        events_audit.append({'domain':meta['domain'],'scene':meta['scene'],'reset_frames':np.flatnonzero(events).tolist()})
        inputs=torch.from_numpy((data['low']/255.).transpose(0,3,1,2).copy()).to(device)
        for variant,preparation,seed,step,net in models:
            first_reference=None
            for history in (['causal','current'] if variant=='aligned_concat' else ['causal']):
                predictions={};metrics_cache={}
                for reset in [False,True]:
                    model_key,_=cache_keys(variant,history,reset,has_events,'raw')
                    if model_key not in predictions:
                        prediction=infer(net,inputs,data,ALIASES[variant],history,events if model_key else None)
                        if not bool(torch.isfinite(prediction).all()):raise FloatingPointError('nonfinite depth-model inference')
                        predictions[model_key]=np.ascontiguousarray(prediction.clamp(0,1).permute(0,2,3,1).cpu().numpy()*255.)
                        inferences+=1
                    raw=predictions[model_key]
                    if first_reference is None:first_reference=raw[0].copy()
                    delta=float(np.max(np.abs(first_reference-raw[0])))
                    if delta>1e-4:raise AssertionError('first frame changed under history/reset')
                    first.append({'domain':meta['domain'],'scene':meta['scene'],'variant':variant,'train_preparation':preparation,
                        'train_seed':seed,'step':step,'history':history,'input_reset':reset,'max_abs_pixel':delta})
                    for fusion in ['raw','half01']:
                        _,key=cache_keys(variant,history,reset,has_events,fusion)
                        identity={'domain':meta['domain'],'scene':meta['scene'],'variant':variant,'train_preparation':preparation,
                            'train_seed':seed,'step':step,'history':history,'input_reset':reset,'fusion':fusion}
                        label='|'.join(str(identity[k]) for k in identity)
                        if key not in metrics_cache:
                            output=raw if fusion=='raw' else fuse(raw,data['half_flow'],data['half_mask'],resets=events if key[2] else None,alpha=.1)
                            metrics,frames=evaluate(output,data['gt'],data['reference_flow'],data['reference_mask'])
                            metrics_cache[key]=(metrics,frames,label);alias=None;computations+=1
                        else:
                            metrics,frames,alias=metrics_cache[key];aliases+=1
                        annotation={'metric_alias_of':alias,'effective_model_reset':key[0],'effective_output_reset':key[2]}
                        with (out/'per_sequence.jsonl').open('a') as stream:stream.write(json.dumps({**identity,**annotation,**metrics},allow_nan=False)+'\n')
                        with (out/'per_frame.jsonl').open('a') as stream:
                            for frame in frames:stream.write(json.dumps({**identity,**annotation,**frame},allow_nan=False)+'\n')
                        count+=1
        print(json.dumps({'domain':meta['domain'],'scene':meta['scene'],'logical_rows':count,'actual_inferences':inferences,
            'metric_computations':computations,'metric_aliases':aliases}),flush=True)
    expected=736
    assert count==expected and computations+aliases==count
    save_json(out/'first_frame_audit.json',first);save_json(out/'input_events.json',events_audit)
    save_json(out/'runtime.json',{'wall_seconds':time.perf_counter()-start,'rows':count,'actual_inference_sequences':inferences,
        'metric_computations':computations,'metric_aliases':aliases,
        'device':'cpu','cuda_initialized':torch.cuda.is_initialized(),'scope':'configuration rows include exact aliases; not independent video counts or full-pipeline benchmark'})


if __name__=='__main__':main()
