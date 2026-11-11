"""Matched native/resized4000-step training, with explicit Adam continuation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from continuation_r14 import MODES, create_model, initial_runtime, restore
from exploration_r2_data import ROOT, load_images
from explore_temporal_controls import save_json, sha256
from native_training_r14 import load_training, real_pair
from train_aligned_raw_r11 import aligned_pair
from train_mixture_r4b import mixture_loss
from training_context_r13 import synthetic_pair_with_context


def align_tasks(low,previous,teacher,device,preparation,scene,t,cache):
    aligned=[];details=[]
    for pair_index in range(2):
        for slot in range(2):
            current=low[pair_index,slot];prior=previous[pair_index,slot]
            if pair_index==0 and t-1+slot==0:
                history=current.detach().permute(1,2,0).cpu().numpy()*255.
                detail={'valid_fraction':1.,'mean_flow_norm':0.,'raw_pair_mae':0.,'aligned_pair_mae':0.,
                    'history_array_sha256':hashlib.sha256(history.tobytes()).hexdigest(),'cache_hit':False,'self_fallback':True}
            else:
                key=(scene,t-1+slot) if pair_index==0 and preparation=='resized' else None
                hit=key is not None and key in cache
                if hit:history,detail=cache[key]
                else:
                    history,detail=aligned_pair(torch.stack([prior,current]),teacher,device)
                    if key is not None:cache[key]=(history,detail)
                detail={**detail,'cache_hit':hit,'self_fallback':False}
            aligned.append(history);details.append({**detail,'source':'real' if pair_index==0 else 'synthetic',
                'pair_index':pair_index,'task_slot':slot})
    return torch.from_numpy(np.stack(aligned).transpose(0,3,1,2).copy()/255.).to(device).reshape_as(low),details


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variants',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();variants=args.variants.split(',')
    if len(set(variants))!=len(variants) or any(v not in MODES for v in variants):p.error('registered unique variants required')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda');revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    sources={str(f.relative_to(ROOT)):sha256(f) for f in Path(__file__).parent.glob('*.py')}
    load_start=time.perf_counter();data,integrity=load_training();images=load_images('train');teacher=create_model('plain',device).backbone.eval()
    save_json(out/'source_integrity.json',{**integrity,'load_wall_seconds':time.perf_counter()-load_start,
        'scope':'training source only; original384 files; native uint8 on CPU, model batch only on GPU'})
    for variant in variants:
        resized_cache={}
        for seed in [0,1]:
            old_path=ROOT/'research/4.temporal-consistency/results/exploration-r5-spatial'/f'plain-seed{seed}'/'training.jsonl'
            old=[json.loads(s) for s in old_path.read_text().splitlines()]
            for preparation in ['native_crop','resized']:
                label=f'{preparation}-{variant}-seed{seed}';folder=out/label;folder.mkdir()
                checkpoint=ROOT/'.cache/checkpoints/exploration-r14'/label;checkpoint.mkdir(parents=True,exist_ok=False)
                torch.manual_seed(seed);net=create_model(variant,device).train();parameters=list(net.parameters())
                optimizer=torch.optim.Adam(parameters,lr=1e-3,weight_decay=0)
                begin_step=0;continuation=None
                if preparation=='resized':
                    continuation=restore(net,optimizer,initial_runtime(variant,seed),device);begin_step=1000
                    save_json(folder/'continuation_audit.json',continuation)
                    save_json(folder/'checkpoint-step1000.json',{'path':continuation['checkpoint'],'sha256':continuation['checkpoint_sha256'],
                        'reused':True,'step':1000})
                config={'variant':variant,'preparation':preparation,'seed':seed,'start_step':begin_step,'final_step':4000,
                    'new_updates':4000-begin_step,'parameters':sum(p.numel() for p in parameters),'lr_all':1e-3,'clip_all':1.,
                    'optimizer':'Adam wd0','loss':'Charbonnier .001','size':[320,180],
                    'training_scenes':integrity['scenes'],'manifest_sha256':integrity['manifest_sha256'],
                    'pretrained_teacher_sha256':sha256(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'),
                    'old_first1000_identity_trace_sha256':sha256(old_path),'continuation':continuation,
                    'protocol_sha256':sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r14-protocol.md'),
                    'git_revision_at_startup':revision,'source_sha256_at_startup':sources,'resume_supported':False,
                    'torch_version':torch.__version__,'opencv_version':cv2.__version__,'cudnn_deterministic':True,
                    'cudnn_tf32_allowed':torch.backends.cudnn.allow_tf32,'native_flow_cache':'disabled; different ROI is never reused'}
                save_json(folder/'config.json',config);start=time.perf_counter();recent=[]
                for step in range(begin_step,4000):
                    rx,ry,rp,identity,real_context=real_pair(data,seed,step,preparation)
                    sx,sy,meta,sp,synthetic_context=synthetic_pair_with_context(images,seed,step)
                    low=torch.stack([rx,sx]).to(device);target=torch.stack([ry,sy]).to(device)
                    previous=torch.stack([rp,torch.stack([sp,sx[0]])]).to(device)
                    samples=[identity,meta]
                    if step<1000 and samples!=old[step]['sample_identity']:raise AssertionError('old first1000 scene/frame/synthetic stream changed')
                    align_start=time.perf_counter();details=[]
                    if variant=='aligned_concat':
                        previous,details=align_tasks(low,previous,teacher,device,preparation,identity['scene'],identity['frame_pair'],resized_cache)
                    align_seconds=time.perf_counter()-align_start
                    prediction=net.forward_pair(low) if variant=='plain' else net.transform(low.flatten(0,1),previous.flatten(0,1)).reshape_as(low)
                    loss=mixture_loss(prediction,target)
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite task')
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    if step==0:
                        reference=create_model('plain',device).train()
                        expected=reference.forward_pair(low);reference_loss=mixture_loss(expected,target);reference_loss.backward()
                        delta=max(float((a.grad-b.grad).abs().max()) for a,b in zip(net.backbone.parameters(),reference.backbone.parameters()))
                        audit={'output_max_abs':float((prediction.detach()-expected.detach()).abs().max()),
                            'task_delta':float(loss.detach()-reference_loss.detach()),'backbone_gradient_max_abs':delta}
                        save_json(folder/'initial_plain_audit.json',audit)
                        if max(abs(v) for v in audit.values())>2e-5:raise AssertionError('zero branch initial plain equivalence changed')
                        del reference,expected,reference_loss
                    norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.))
                    if not np.isfinite(norm):raise FloatingPointError('nonfinite gradients')
                    optimizer.step();recent.append(float(loss.detach()))
                    row={'step':step+1,'task':float(loss.detach()),'gradient_norm_all':norm,'sample_identity':samples,
                        'real_context':real_context,'synthetic_context':synthetic_context,'alignment':details,'alignment_wall_seconds':align_seconds}
                    if variant!='plain':row['auxiliary_l2']=float(net.auxiliary.weight.detach().norm())
                    with (folder/'training.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                    if (step+1)%100==0:
                        print(json.dumps({'label':label,'step':step+1,'task':row['task'],'gradient_norm_all':norm,
                            'scene':identity['scene'],'frame_pair':identity['frame_pair'],'crop_xywh':real_context['crop_xywh']}),flush=True)
                    if (step+1)%1000==0:
                        path=checkpoint/f'step{step+1:04d}.pth'
                        torch.save({'model':net.state_dict(),'optimizer':optimizer.state_dict(),'step':step+1,'config':config},path)
                        save_json(folder/f'checkpoint-step{step+1:04d}.json',{'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'step':step+1,'reused':False})
                        prefix=(folder/'training.jsonl').read_bytes()
                        save_json(folder/f'milestone-step{step+1:04d}.json',{'step':step+1,'new_updates':step+1-begin_step,
                            'last100_task_mean':float(np.mean(recent[-100:])),'trace_prefix_bytes':len(prefix),
                            'trace_prefix_sha256':hashlib.sha256(prefix).hexdigest(),'wall_seconds':time.perf_counter()-start,
                            'checkpoint_sha256':sha256(path)})
                save_json(folder/'runtime.json',{'wall_seconds':time.perf_counter()-start,'checkpoint':str(path.relative_to(ROOT)),
                    'checkpoint_sha256':sha256(path),'start_step':begin_step,'final_step':4000,'new_updates':4000-begin_step,
                    'first1000_old_identities_verified':begin_step==0,'cached_resized_real_pairs':len(resized_cache)})
                del optimizer,net


if __name__=='__main__':main()
