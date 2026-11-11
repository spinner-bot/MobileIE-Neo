"""Independent aggregation-aware models share only deterministic task/history preparation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from aggregation_model_r21 import AggregationFST,MODES
from exploration_r2_data import ROOT,load_images
from explore_temporal_controls import save_json,sha256
from native_training_r14 import load_training,real_pair
from train_exploration_r2 import new_model
from train_mixture_r4b import mixture_loss
from train_native_r14 import align_tasks
from training_context_r13 import synthetic_pair_with_context


def create_model(mode,device):return AggregationFST(new_model(device),mode).to(device)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,choices=[0,1],required=True);p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda:0');revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    scripts={str(p.relative_to(ROOT)):sha256(p) for p in Path(__file__).parent.glob('*.py')};base=ROOT/'research/4.temporal-consistency'
    data,integrity=load_training();images=load_images('train');teacher=new_model(device).eval()
    for parameter in teacher.parameters():parameter.requires_grad_(False)
    save_json(out/'source_integrity.json',integrity)
    reference_path=base/'results/exploration-r14-spatial'/f'native_crop-plain-seed{args.seed}'/'training.jsonl'
    aligned_path=base/'results/exploration-r14-aligned'/f'native_crop-aligned_concat-seed{args.seed}'/'training.jsonl'
    reference=[json.loads(s) for s in reference_path.read_text().splitlines()];old_aligned=[json.loads(s) for s in aligned_path.read_text().splitlines()]
    assert len(reference)==len(old_aligned)==4000
    runs=[];pointers=set();shared_path=out/'shared_context.jsonl';start=time.perf_counter();old_hash_matches=0
    for mode in MODES:
        torch.manual_seed(args.seed);net=create_model(mode,device).train();parameters=list(net.parameters())
        own={p.data_ptr() for p in parameters};assert not own&pointers;pointers|=own
        assert all(p.device==device for p in parameters)
        optimizer=torch.optim.Adam(parameters,lr=1e-3,weight_decay=0);folder=out/mode;folder.mkdir()
        checkpoints=ROOT/'.cache/checkpoints/exploration-r21'/f'{mode}-seed{args.seed}';checkpoints.mkdir(parents=True,exist_ok=False)
        cfg={'mode':mode,'seed':args.seed,'steps':4000,'parameters':sum(p.numel() for p in parameters),'preparation':'native_crop',
            'initial_alpha':net.coefficient(),'lr_all':1e-3,'clip_all':1.,'optimizer':'Adam wd0','loss':'Charbonnier .001',
            'projection':'learned alpha projected to0..1 after Adam; alpha participates in global clip',
            'source_manifest_sha256':integrity['manifest_sha256'],'R14_task_trace_sha256':sha256(reference_path),
            'R14_aligned_trace_sha256':sha256(aligned_path),'teacher_sha256':sha256(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'),
            'protocol_sha256':sha256(base/'docs/exploration-r21-protocol.md'),'revision':revision,'source_sha256':scripts,
            'shared_context_path':str(shared_path.relative_to(ROOT)),'optimizer_shared':False,'teacher_gradients':False,
            'cudnn_tf32_allowed':torch.backends.cudnn.allow_tf32,'cudnn_deterministic':True,'resume_supported':False,
            'scope':'classical aggregation/initialization control, no auxiliary loss or novelty claim'}
        save_json(folder/'config.json',cfg);runs.append({'mode':mode,'net':net,'parameters':parameters,'optimizer':optimizer,'folder':folder,'checkpoints':checkpoints,'config':cfg,'recent':[]})
    for step in range(4000):
        rx,ry,rp,identity,real_context=real_pair(data,args.seed,step,'native_crop')
        sx,sy,meta,sp,synthetic_context=synthetic_pair_with_context(images,args.seed,step)
        samples=[identity,meta];ref=reference[step]
        assert samples==ref['sample_identity'] and real_context==ref['real_context'] and synthetic_context==ref['synthetic_context']
        low=torch.stack([rx,sx]).to(device);target=torch.stack([ry,sy]).to(device)
        previous=torch.stack([rp,torch.stack([sp,sx[0]])]).to(device)
        align_start=time.perf_counter();previous,details=align_tasks(low,previous,teacher,device,'native_crop',identity['scene'],identity['frame_pair'],{})
        align_seconds=time.perf_counter()-align_start
        matches=[a['history_array_sha256']==b['history_array_sha256'] for a,b in zip(details,old_aligned[step]['alignment'])]
        assert len(matches)==4;old_hash_matches+=sum(matches)
        context={'step':step+1,'sample_identity':samples,'real_context':real_context,'synthetic_context':synthetic_context,
            'alignment':details,'old_R14_history_hash_matches':matches}
        digest=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()
        with shared_path.open('a') as stream:stream.write(json.dumps({**context,'context_canonical_sha256':digest,'alignment_wall_seconds':align_seconds},allow_nan=False)+'\n')
        current=low.flatten(0,1);past=previous.flatten(0,1);expected=None;half_initial=None;last=[]
        for run in runs:
            net=run['net'];optimizer=run['optimizer'];folder=run['folder'];mode=run['mode']
            if step==0:
                with torch.no_grad():
                    if expected is None:expected=teacher(current)
                    limit=net.transform(current,current);actual=net.transform(current,past)
                    audit={'current_limit_max_abs':float((limit-expected).abs().max()),'actual_history_minus_current_max_abs':float((actual-expected).abs().max()),
                        'current_limit_task_delta':float(mixture_loss(limit.reshape_as(target),target)-mixture_loss(expected.reshape_as(target),target)),
                        'preset_current_limit_tolerance':2e-5}
                    if mode=='pre_fixedhalf':half_initial=actual.detach().clone()
                    if mode=='pre_learnhalf':audit['fixedhalf_initial_actual_history_max_abs']=float((actual-half_initial).abs().max())
                save_json(folder/'initial_current_limit.json',audit)
                if audit['current_limit_max_abs']>2e-5 or audit.get('fixedhalf_initial_actual_history_max_abs',0)>2e-5:
                    raise AssertionError('registered initial static limit changed')
            before=net.coefficient();prediction=net.transform(current,past).reshape_as(target);loss=mixture_loss(prediction,target)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite task')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            alpha_gradient=None if net.alpha is None else float(net.alpha.grad.detach())
            norm=float(torch.nn.utils.clip_grad_norm_(run['parameters'],1.))
            if not np.isfinite(norm):raise FloatingPointError('nonfinite gradients')
            optimizer.step();unprojected=net.coefficient();net.project();after=net.coefficient();task=float(loss.detach());run['recent'].append(task)
            row={'step':step+1,'task':task,'gradient_norm_all':norm,'alpha_before':before,'alpha_gradient_before_clip':alpha_gradient,
                'alpha_after_Adam_before_projection':unprojected,'alpha_after_projection':after,'shared_context_sha256':digest}
            with (folder/'training.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            last.append({'mode':mode,'task':task,'alpha':after,'gradient_norm':norm});del prediction,loss
        if (step+1)%100==0:print(json.dumps({'seed':args.seed,'step':step+1,'models':last,'old_history_hash_matches':old_hash_matches}),flush=True)
        if (step+1)%1000==0:
            common=shared_path.read_bytes();common_meta={'bytes':len(common),'sha256':hashlib.sha256(common).hexdigest()}
            for run in runs:
                path=run['checkpoints']/f'step{step+1:04d}.pth';net=run['net']
                torch.save({'model':net.state_dict(),'optimizer':run['optimizer'].state_dict(),'step':step+1,'config':run['config']},path)
                save_json(run['folder']/f'checkpoint-step{step+1:04d}.json',{'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'step':step+1,'alpha':net.coefficient()})
                trace=(run['folder']/'training.jsonl').read_bytes()
                save_json(run['folder']/f'milestone-step{step+1:04d}.json',{'step':step+1,'last100_task_mean':float(np.mean(run['recent'][-100:])),
                    'model_trace_prefix_bytes':len(trace),'model_trace_prefix_sha256':hashlib.sha256(trace).hexdigest(),
                    'shared_context_prefix':common_meta,'checkpoint_sha256':sha256(path),'wall_seconds':time.perf_counter()-start})
    for run in runs:save_json(run['folder']/'runtime.json',{'final_step':4000,'new_updates':4000,'alpha':run['net'].coefficient(),'shared_alignment':True})
    save_json(out/'runtime.json',{'seed':args.seed,'steps':4000,'new_updates':16000,'models':4,'alignment_pairs':16000,
        'old_R14_history_hash_matches':old_hash_matches,'all_current_task_contexts_exact':True,'wall_seconds':time.perf_counter()-start,
        'scope':'independent model/Adam updates, common deterministic observations/history; no wall-clock cutoff'})


if __name__=='__main__':main()
