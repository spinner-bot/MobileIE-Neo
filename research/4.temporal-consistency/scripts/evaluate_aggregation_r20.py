"""Same-flow dynamic aggregation-order controls, replaying the strong R14 baseline."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from aggregation_controls_r20 import METHODS,event_output,finite_controls
from aligned_raw_history_r11 import align_history
from continuation_r14 import create_model
from evaluation_aliases_r14 import event_flags
from exploration_r2_data import ROOT,detect_changes,evaluate
from explore_state_r2 import fuse
from explore_temporal_controls import save_json,sha256
from prepare_cache_r5 import CACHE_INDEX,load_one
from prepare_development_r9 import DEV_INDEX
from prepare_native_r10 import INDEX as NATIVE_INDEX
from summarize_causal_r5 import jsonl

BASE=ROOT/'research/4.temporal-consistency'
METRICS=['psnr_rgb','ssim_rgb','edge_mae','temporal_error','flicker']


def metric_delta(actual,reference):
    delta=0.
    for key in METRICS+['output_mean','target_mean']:
        if key not in actual or key not in reference:continue
        a,b=actual[key],reference[key]
        if a is None or b is None:assert a is None and b is None
        else:delta=max(delta,abs(a-b))
    return delta


@torch.inference_mode()
def forward_rgb(net,arrays,device,first=None):
    inputs=torch.from_numpy((arrays/255.).transpose(0,3,1,2).copy()).to(device);offset=0 if first is None else 1
    torch.cuda.synchronize();start=time.perf_counter()
    prediction=torch.stack([net(inputs[t:t+1])[0] for t in range(offset,len(inputs))]).clamp(0,1)
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    array=np.ascontiguousarray(prediction.permute(0,2,3,1).cpu().numpy()*255.)
    if first is not None:array=np.concatenate([first[None],array],axis=0)
    return array,{'model_forward_frames':len(inputs)-offset,'GPU_stage_wall_seconds':seconds,
        'scope':'batch1 synchronized forward/stack/clamp incl Python; excludes transfers, data, teacher/flow, metrics; shared comparisons not standalone cost'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,choices=[0,1],required=True);p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda');folder=BASE/'results/exploration-r14-spatial'/f'native_crop-plain-seed{args.seed}'
    item=json.loads((folder/'checkpoint-step4000.json').read_text());path=ROOT/item['path'];assert sha256(path)==item['sha256']
    saved=torch.load(path,map_location=device,weights_only=False);assert saved['step']==4000
    wrapper=create_model('plain',device);wrapper.load_state_dict(saved['model'],strict=True);net=wrapper.backbone.eval()
    reference_folder=BASE/'results/exploration-r14-eval-spatial'
    def selected(r):return r['variant']=='plain' and r['train_preparation']=='native_crop' and r['step']==4000 and r['train_seed']==args.seed
    references={(r['domain'],r['scene'],r['input_reset'],r['fusion']):r for r in jsonl(reference_folder/'per_sequence.jsonl') if selected(r)}
    frame_references={(r['domain'],r['scene'],r['input_reset'],r['fusion'],r['frame']):r for r in jsonl(reference_folder/'per_frame.jsonl') if selected(r)}
    assert len(references)==184
    save_json(out/'config.json',{'seed':args.seed,'methods':list(METHODS),'checkpoint':item,'parameters':4075,'training_updates':0,
        'protocol_sha256':sha256(BASE/'docs/exploration-r20-protocol.md'),
        'reference_sources':{str(p.relative_to(ROOT)):sha256(p) for p in [reference_folder/'per_sequence.jsonl',reference_folder/'per_frame.jsonl']},
        'cache_indices':{str(p.relative_to(ROOT)):sha256(p) for p in [CACHE_INDEX,DEV_INDEX,NATIVE_INDEX]},
        'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'cudnn_tf32_allowed':torch.backends.cudnn.allow_tf32,'batch':1,'future_wait_frames':0,
        'scope':'classic finite/recursive controls; common cached deployable teacher flow, no GT motion used for deployment; not novelty or mobile timing'})
    metas=json.loads(CACHE_INDEX.read_text())['scenes']+json.loads(DEV_INDEX.read_text())['scenes']
    metas += [{**r,'domain':'dev_native'} for r in json.loads(NATIVE_INDEX.read_text())['scenes'] if r['domain']=='native_crop'];assert len(metas)==46
    start=time.perf_counter();count=0;computations=0;aliases=0;replays=[];events_audit=[];cost=[];first=[]
    for meta in metas:
        data=load_one(meta);events,_=detect_changes(data['low']);events=event_flags(events);has_events=bool(events.any())
        events_audit.append({'domain':meta['domain'],'scene':meta['scene'],'reset_frames':np.flatnonzero(events).tolist()})
        raw,detail=forward_rgb(net,data['low'],device);stage=[{'stage':'raw',**detail}]
        aligned=[data['low'][0]];warped=[raw[0]];valid_masks=[]
        for t in range(1,len(raw)):
            low,valid=align_history(data['low'][t-1],data['low'][t],data['half_flow'][t-1],data['half_mask'][t-1])
            output,valid_output=align_history(raw[t-1],raw[t],data['half_flow'][t-1],data['half_mask'][t-1]);assert np.array_equal(valid,valid_output)
            aligned.append(low);warped.append(output);valid_masks.append(valid)
        aligned=np.stack(aligned);warped=np.stack(warped);aligned_prediction,detail=forward_rgb(net,aligned,device,raw[0]);stage.append({'stage':'aligned_prior',**detail})
        def forward(arrays,first):
            values,detail=forward_rgb(net,arrays,device,first);stage.append({'stage':'preblend','alpha_order':len(stage)-2,**detail});return values
        outputs=finite_controls(data['low'],raw,aligned,aligned_prediction,warped,forward)
        commutator=[]
        for t,valid in enumerate(valid_masks,1):
            mask=valid.copy();mask[:8]=mask[-8:]=False;mask[:,:8]=mask[:,-8:]=False
            commutator.append(float(np.abs(aligned_prediction[t]-warped[t])[mask].mean()) if mask.any() else None)
        cost.append({'domain':meta['domain'],'scene':meta['scene'],'stages':stage,'model_sequence_passes':len(stage),
            'model_forward_frames':sum(s['model_forward_frames'] for s in stage),'mean_motion_valid_fraction':float(np.mean(valid_masks)),
            'warp_enhancement_commutator_MAE255':commutator,'scope':'shared compute across comparisons, not standalone pipeline cost'})
        cache={}
        for reset in [False,True]:
            for method in METHODS:
                effective=bool(reset and has_events and method!='raw');key=(method,effective)
                identity={'domain':meta['domain'],'scene':meta['scene'],'train_seed':args.seed,'method':method,'input_reset':reset}
                label='|'.join(str(identity[k]) for k in identity)
                if key not in cache:
                    if method=='recursive_half01':output=fuse(raw,data['half_flow'],data['half_mask'],resets=events if effective else None,alpha=.1)
                    else:output=event_output(outputs[method],raw,events) if effective else outputs[method]
                    pixel_delta=float(np.max(np.abs(output[0]-raw[0])));assert pixel_delta<=1e-4
                    first.append({**identity,'first_frame_max_abs_pixel':pixel_delta})
                    metrics,frames=evaluate(output,data['gt'],data['reference_flow'],data['reference_mask'])
                    cache[key]=(metrics,frames,label);alias=None;computations+=1
                else:metrics,frames,alias=cache[key];aliases+=1
                if method in ['raw','recursive_half01']:
                    fusion='raw' if method=='raw' else 'half01';rkey=(meta['domain'],meta['scene'],reset,fusion)
                    delta=metric_delta(metrics,references[rkey])
                    for frame in frames:delta=max(delta,metric_delta(frame,frame_references[(*rkey,frame['frame'])]))
                    replays.append({**identity,'sequence_and_frame_max_abs_metric':delta})
                    if delta>1e-10:raise AssertionError('strong baseline replay changed')
                annotation={'metric_alias_of':alias,'effective_reset':effective}
                with (out/'per_sequence.jsonl').open('a') as stream:stream.write(json.dumps({**identity,**annotation,**metrics},allow_nan=False)+'\n')
                with (out/'per_frame.jsonl').open('a') as stream:
                    for frame in frames:stream.write(json.dumps({**identity,**annotation,**frame},allow_nan=False)+'\n')
                count+=1
        print(json.dumps({'domain':meta['domain'],'scene':meta['scene'],'rows':count,'metric_computations':computations,'metric_aliases':aliases}),flush=True)
    assert count==736 and computations==536 and aliases==200 and len(replays)==184
    save_json(out/'baseline_replays.json',replays);save_json(out/'input_events.json',events_audit);save_json(out/'first_frame_audit.json',first)
    save_json(out/'compute_accounting.json',cost);save_json(out/'runtime.json',{'wall_seconds':time.perf_counter()-start,'rows':count,
        'metric_computations':computations,'metric_aliases':aliases,'model_sequence_passes':sum(r['model_sequence_passes'] for r in cost),
        'model_forward_frames':sum(r['model_forward_frames'] for r in cost),'scope':'logical controls on46 reused conditions; no training or standalone cost claim'})


if __name__=='__main__':main()
