"""Independent fixed mappings and objectives with shared deterministic R14 observations."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from exploration_r2_data import ROOT,load_images
from explore_temporal_controls import save_json,sha256
from loss_controls_r26 import ARCHITECTURES,OBJECTIVES,compare_state,create_model,forward_training
from native_training_r14 import load_training,real_pair
from train_exploration_r2 import new_model
from train_mixture_r4b import mixture_loss
from train_native_r14 import align_tasks
from training_context_r13 import synthetic_pair_with_context

BASE=ROOT/'research/4.temporal-consistency'


def old_folder(architecture,seed):
    return BASE/'results'/('exploration-r14-spatial' if architecture=='plain' else f'exploration-r21-seed{seed}')/(f'native_crop-plain-seed{seed}' if architecture=='plain' else architecture)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int,choices=[0,1],required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda:0');revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    sources={str(p.relative_to(ROOT)):sha256(p) for p in Path(__file__).parent.glob('*.py')};data,integrity=load_training();images=load_images('train')
    teacher=new_model(device).eval()
    for parameter in teacher.parameters():parameter.requires_grad_(False)
    save_json(out/'source_integrity.json',integrity)
    reference_path=old_folder('plain',args.seed)/'training.jsonl';reference=[json.loads(s) for s in reference_path.read_text().splitlines()]
    aligned_path=BASE/'results/exploration-r14-aligned'/f'native_crop-aligned_concat-seed{args.seed}'/'training.jsonl'
    old_aligned=[json.loads(s) for s in aligned_path.read_text().splitlines()];assert len(reference)==len(old_aligned)==4000
    runs=[];pointers=set();shared_path=out/'shared_context.jsonl';start=time.perf_counter();old_hash_matches=0
    initial_predictions={};initial_gradients={}
    for architecture in ARCHITECTURES:
        old=old_folder(architecture,args.seed);old_trace=[json.loads(s) for s in (old/'training.jsonl').read_text().splitlines()]
        for objective in OBJECTIVES:
            torch.manual_seed(args.seed);net=create_model(architecture,device).train();parameters=list(net.parameters())
            own={p.data_ptr() for p in parameters};assert not pointers&own;pointers|=own
            assert sum(p.numel() for p in parameters)==4075 and all(p.device==device for p in parameters)
            optimizer=torch.optim.Adam(parameters,lr=1e-3,weight_decay=0);label=architecture+'-'+objective;folder=out/label;folder.mkdir()
            checkpoints=ROOT/'.cache/checkpoints/exploration-r26'/f'{label}-seed{args.seed}';checkpoints.mkdir(parents=True,exist_ok=False)
            cfg={'architecture':architecture,'objective':objective,'seed':args.seed,'steps':4000,'parameters':4075,'loss_scale':1.,
                'loss':'sqrt(error²+1e-6).mean' if objective=='charb' else 'error².mean','lr_all':1e-3,'clip_all':1.,'optimizer':'Adam wd0',
                'preparation':'native_crop','source_manifest_sha256':integrity['manifest_sha256'],'R14_task_trace_sha256':sha256(reference_path),
                'old_Charb_trace_sha256':sha256(old/'training.jsonl'),'old_Charb_folder':str(old.relative_to(ROOT)),
                'teacher_sha256':sha256(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'),'protocol_sha256':sha256(BASE/'docs/exploration-r26-protocol.md'),
                'revision':revision,'source_sha256':sources,'shared_context_path':str(shared_path.relative_to(ROOT)),
                'optimizer_shared':False,'teacher_gradients':False,'cudnn_tf32_allowed':torch.backends.cudnn.allow_tf32,
                'cudnn_deterministic':True,'resume_supported':False,'scope':'Objective/setup control; loss scale affects global clip/Adam coupling, not a pure gradient-direction experiment.'}
            save_json(folder/'config.json',cfg);runs.append({'architecture':architecture,'objective':objective,'label':label,'net':net,'parameters':parameters,
                'optimizer':optimizer,'folder':folder,'checkpoints':checkpoints,'config':cfg,'old_trace':old_trace,'recent':[]})
    for step in range(4000):
        rx,ry,rp,identity,real_context=real_pair(data,args.seed,step,'native_crop')
        sx,sy,meta,sp,synthetic_context=synthetic_pair_with_context(images,args.seed,step)
        samples=[identity,meta];ref=reference[step]
        assert samples==ref['sample_identity'] and real_context==ref['real_context'] and synthetic_context==ref['synthetic_context']
        low=torch.stack([rx,sx]).to(device);target=torch.stack([ry,sy]).to(device);previous=torch.stack([rp,torch.stack([sp,sx[0]])]).to(device)
        align_start=time.perf_counter();previous,details=align_tasks(low,previous,teacher,device,'native_crop',identity['scene'],identity['frame_pair'],{})
        align_seconds=time.perf_counter()-align_start
        matches=[a['history_array_sha256']==b['history_array_sha256'] for a,b in zip(details,old_aligned[step]['alignment'])];assert len(matches)==4;old_hash_matches+=sum(matches)
        context={'step':step+1,'sample_identity':samples,'real_context':real_context,'synthetic_context':synthetic_context,
            'alignment':details,'old_R14_history_hash_matches':matches};digest=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()
        with shared_path.open('a') as stream:stream.write(json.dumps({**context,'context_canonical_sha256':digest,'alignment_wall_seconds':align_seconds},allow_nan=False)+'\n')
        last=[]
        for run in runs:
            net=run['net'];architecture=run['architecture'];objective=run['objective'];optimizer=run['optimizer'];folder=run['folder']
            if step==0:
                with torch.no_grad():
                    expected=teacher(low.flatten(0,1)).reshape_as(low);limit=forward_training(net,architecture,low,low)
                    audit={'current_limit_max_abs':float((limit-expected).abs().max()),'preset_current_limit_tolerance':2e-5}
                save_json(folder/'initial_current_limit.json',audit);assert audit['current_limit_max_abs']<=2e-5
            prediction=forward_training(net,architecture,low,previous)
            charb=mixture_loss(prediction,target);mse=(prediction-target).square().mean();loss=charb if objective=='charb' else mse
            assert bool(torch.isfinite(loss))
            if step==0:
                if objective=='charb':initial_predictions[architecture]=prediction.detach().clone()
                else:
                    equal=torch.equal(prediction.detach(),initial_predictions[architecture]);assert equal
                    save_json(folder/'initial_paired_output.json',{'same_architecture_Charb_MSE_output_pixel_equal':equal})
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if step==0:
                gradient=torch.cat([p.grad.detach().flatten() for p in run['parameters']]).double().cpu()
                if objective=='charb':initial_gradients[architecture]=gradient
                else:
                    old_gradient=initial_gradients[architecture];assert old_gradient.norm()>0 and gradient.norm()>0
                    save_json(folder/'initial_objective_gradients.json',{'Charb_norm_before_clip':float(old_gradient.norm()),
                        'MSE_norm_before_clip':float(gradient.norm()),'MSE_over_Charb_norm':float(gradient.norm()/old_gradient.norm()),
                        'gradient_direction_cosine':float(torch.nn.functional.cosine_similarity(old_gradient,gradient,dim=0)),
                        'loss_scale_adjusted':False,'scope':'Initial matched weights/tasks; no hyperparameter adjustment from this diagnostic.'})
            norm=float(torch.nn.utils.clip_grad_norm_(run['parameters'],1.));assert np.isfinite(norm);optimizer.step()
            error=(prediction-target).detach().double();mu=error.mean((-2,-1),keepdim=True)
            double_mse=float(error.square().mean());bias=float(mu.square().mean());centered=float((error-mu).square().mean());closure=abs(double_mse-bias-centered);assert closure<=1e-12
            task=float(loss.detach());run['recent'].append(task)
            row={'step':step+1,'task':task,'charb_task':float(charb.detach()),'mse_task':float(mse.detach()),'gradient_norm_all':norm,
                'global_clip_triggered':norm>1.,'global_RGB_bias_mse':bias,'centered_error_mse':centered,'MSE_double':double_mse,'MSE_closure_abs':closure,
                'old_Charb_task_delta':task-run['old_trace'][step]['task'] if objective=='charb' else None,'shared_context_sha256':digest}
            with (folder/'training.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
            last.append({'label':run['label'],'task':task,'gradient_norm':norm,'old_Charb_task_delta':row['old_Charb_task_delta']})
            del prediction,charb,mse,loss,error,mu
        if step==0:initial_predictions.clear();initial_gradients.clear()
        if (step+1)%100==0:print(json.dumps({'seed':args.seed,'step':step+1,'models':last,'old_history_hash_matches':old_hash_matches}),flush=True)
        if (step+1)%1000==0:
            common=shared_path.read_bytes();common_meta={'bytes':len(common),'sha256':hashlib.sha256(common).hexdigest()}
            for run in runs:
                path=run['checkpoints']/f'step{step+1:04d}.pth';torch.save({'model':run['net'].state_dict(),'optimizer':run['optimizer'].state_dict(),'step':step+1,'config':run['config']},path)
                save_json(run['folder']/f'checkpoint-step{step+1:04d}.json',{'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'step':step+1})
                trace=(run['folder']/'training.jsonl').read_bytes()
                milestone={'step':step+1,'last100_task_mean':float(np.mean(run['recent'][-100:])),
                    'model_trace_prefix_bytes':len(trace),'model_trace_prefix_sha256':hashlib.sha256(trace).hexdigest(),
                    'shared_context_prefix':common_meta,'checkpoint_sha256':sha256(path),'wall_seconds':time.perf_counter()-start}
                if run['objective']=='charb':
                    marker=json.loads((old_folder(run['architecture'],args.seed)/f'checkpoint-step{step+1:04d}.json').read_text());assert sha256(ROOT/marker['path'])==marker['sha256']
                    old=torch.load(ROOT/marker['path'],map_location='cpu',weights_only=False)
                    milestone['old_Charb_replay']={'source_checkpoint':marker,'model':compare_state(run['net'].state_dict(),old['model']),
                        'optimizer':compare_state(run['optimizer'].state_dict(),old['optimizer'])}
                save_json(run['folder']/f'milestone-step{step+1:04d}.json',milestone)
    for run in runs:save_json(run['folder']/'runtime.json',{'final_step':4000,'new_updates':4000,'architecture':run['architecture'],'objective':run['objective']})
    save_json(out/'runtime.json',{'seed':args.seed,'steps':4000,'new_updates':24000,'models':6,'alignment_pairs':16000,
        'old_R14_history_hash_matches':old_hash_matches,'all_current_task_contexts_exact':True,'wall_seconds':time.perf_counter()-start,
        'scope':'Complete paired objective/setup controls, fixed budget; no loss rescaling or wall-clock cutoff.'})


if __name__=='__main__':main()
