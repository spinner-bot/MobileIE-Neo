"""All fixed loss-control checkpoints and same-objective classical output baselines."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from aggregation_controls_r20 import blend,event_output
from aligned_raw_history_r11 import align_history
from evaluate_aligned_raw_r11 import infer
from evaluation_aliases_r14 import event_flags
from exploration_r2_data import ROOT,detect_changes,evaluate
from explore_state_r2 import fuse
from explore_temporal_controls import save_json,sha256
from loss_controls_r26 import ARCHITECTURES,OBJECTIVES,create_model
from prepare_cache_r5 import CACHE_INDEX,load_one
from prepare_development_r9 import DEV_INDEX
from prepare_native_r10 import INDEX as NATIVE_INDEX

BASE=ROOT/'research/4.temporal-consistency'


def keys(architecture,history,reset,has_events,fusion):
    model_reset=bool(architecture!='plain' and history=='causal' and reset and has_events)
    output_reset=bool(fusion!='raw' and reset and has_events)
    return model_reset,(model_reset,fusion,output_reset)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--architectures',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda');args=parser.parse_args();architectures=args.architectures.split(',')
    if len(set(architectures))!=len(architectures) or any(v not in ARCHITECTURES for v in architectures):parser.error('registered unique architectures required')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda:0' if args.device=='cuda' else 'cpu');revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    sources={str(p.relative_to(ROOT)):sha256(p) for p in Path(__file__).parent.glob('*.py')};models=[];checkpoints=[]
    for architecture in architectures:
        for objective in OBJECTIVES:
            for seed in [0,1]:
                folder=BASE/'results'/f'exploration-r26-seed{seed}'/(architecture+'-'+objective)
                assert json.loads((folder/'runtime.json').read_text())['final_step']==4000
                for step in [1000,4000]:
                    item=json.loads((folder/f'checkpoint-step{step:04d}.json').read_text());assert sha256(ROOT/item['path'])==item['sha256']
                    saved=torch.load(ROOT/item['path'],map_location=device,weights_only=False);assert saved['step']==step
                    net=create_model(architecture,device);net.load_state_dict(saved['model'],strict=True)
                    if architecture=='plain':net=net.backbone
                    models.append((architecture,objective,seed,step,net.eval()))
                    checkpoints.append({'architecture':architecture,'objective':objective,'train_seed':seed,'step':step,**item,
                        'training_config_sha256':sha256(folder/'config.json')})
    save_json(out/'checkpoints.json',checkpoints)
    save_json(out/'config.json',{'device':str(device),'architectures':architectures,'objectives':list(OBJECTIVES),'steps':[1000,4000],
        'protocol_sha256':sha256(BASE/'docs/exploration-r26-protocol.md'),'cache_indices':{str(p.relative_to(ROOT)):sha256(p) for p in [CACHE_INDEX,DEV_INDEX,NATIVE_INDEX]},
        'revision':revision,'source_sha256':sources,'Torch_threads':2,'OpenCV_threads':1,'cudnn_tf32_allowed':torch.backends.cudnn.allow_tf32,
        'scope':'Same-backend objective/setup controls; new MSE spatial-plus-post baselines, no mixed historical CPU/GPU metrics or novelty claim.'})
    metas=json.loads(CACHE_INDEX.read_text())['scenes']+json.loads(DEV_INDEX.read_text())['scenes']
    metas += [{**r,'domain':'dev_native'} for r in json.loads(NATIVE_INDEX.read_text())['scenes'] if r['domain']=='native_crop'];assert len(metas)==46
    start=time.perf_counter();count=0;inferences=0;computations=0;aliases=0;first=[];events_audit=[]
    for meta in metas:
        data=load_one(meta);events,_=detect_changes(data['low']);events=event_flags(events);has_events=bool(events.any())
        events_audit.append({'domain':meta['domain'],'scene':meta['scene'],'reset_frames':np.flatnonzero(events).tolist()})
        inputs=torch.from_numpy((data['low']/255.).transpose(0,3,1,2).copy()).to(device)
        for architecture,objective,seed,step,net in models:
            first_reference=None;histories=['causal'] if architecture=='plain' else ['causal','current']
            fusions=['raw','half01','postwarp01','postwarp05'] if architecture=='plain' else ['raw','half01']
            for history in histories:
                predictions={};metrics_cache={};finite_cache={}
                for reset in [False,True]:
                    model_key,_=keys(architecture,history,reset,has_events,'raw')
                    if model_key not in predictions:
                        prediction=infer(net,inputs,data,'plain_high' if architecture=='plain' else 'concat_aligned_high',history,events if model_key else None)
                        assert bool(torch.isfinite(prediction).all())
                        predictions[model_key]=np.ascontiguousarray(prediction.clamp(0,1).permute(0,2,3,1).cpu().numpy()*255.);inferences+=1
                    raw=predictions[model_key]
                    if first_reference is None:first_reference=raw[0].copy()
                    delta=float(np.max(np.abs(raw[0]-first_reference)));assert delta<=1e-4
                    first.append({'domain':meta['domain'],'scene':meta['scene'],'architecture':architecture,'objective':objective,'train_seed':seed,
                        'step':step,'history':history,'input_reset':reset,'max_abs_pixel':delta})
                    for fusion in fusions:
                        _,key=keys(architecture,history,reset,has_events,fusion)
                        identity={'domain':meta['domain'],'scene':meta['scene'],'architecture':architecture,'objective':objective,'train_seed':seed,
                            'step':step,'history':history,'input_reset':reset,'fusion':fusion};label='|'.join(str(identity[k]) for k in identity)
                        if key not in metrics_cache:
                            if fusion=='raw':output=raw
                            elif fusion=='half01':output=fuse(raw,data['half_flow'],data['half_mask'],resets=events if key[2] else None,alpha=.1)
                            else:
                                if not finite_cache:
                                    warped=[raw[0]]
                                    for t in range(1,len(raw)):warped.append(align_history(raw[t-1],raw[t],data['half_flow'][t-1],data['half_mask'][t-1])[0])
                                    warped=np.stack(warped)
                                    finite_cache['postwarp01']=blend(raw,warped,.1);finite_cache['postwarp05']=blend(raw,warped,.5)
                                output=event_output(finite_cache[fusion],raw,events) if key[2] else finite_cache[fusion]
                            assert float(np.max(np.abs(output[0]-raw[0])))<=1e-4
                            metrics,frames=evaluate(output,data['gt'],data['reference_flow'],data['reference_mask'])
                            metrics_cache[key]=(metrics,frames,label);alias=None;computations+=1
                        else:metrics,frames,alias=metrics_cache[key];aliases+=1
                        annotation={'metric_alias_of':alias,'effective_model_reset':key[0],'effective_output_reset':key[2]}
                        with (out/'per_sequence.jsonl').open('a') as stream:stream.write(json.dumps({**identity,**annotation,**metrics},allow_nan=False)+'\n')
                        with (out/'per_frame.jsonl').open('a') as stream:
                            for frame in frames:stream.write(json.dumps({**identity,**annotation,**frame},allow_nan=False)+'\n')
                        count+=1
        print(json.dumps({'domain':meta['domain'],'scene':meta['scene'],'rows':count,'actual_inferences':inferences,
            'metric_computations':computations,'metric_aliases':aliases}),flush=True)
    assert count==2944*len(architectures) and computations==2048*len(architectures) and aliases==896*len(architectures)
    save_json(out/'first_frame_audit.json',first);save_json(out/'input_events.json',events_audit)
    save_json(out/'runtime.json',{'rows':count,'actual_inference_sequences':inferences,'metric_computations':computations,'metric_aliases':aliases,
        'device':str(device),'wall_seconds':time.perf_counter()-start,'scope':'All fixed checkpoints; logical rows include exact aliases, not independent video counts or mobile latency.'})


if __name__=='__main__':main()
