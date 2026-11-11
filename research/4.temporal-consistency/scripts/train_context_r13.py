"""Change only the first task frame's past context, with an exact spatial replay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from exploration_r2_data import ROOT, load_images
from explore_temporal_controls import save_json, sha256
from raw_pair_head_r8 import RawPairHead
from real_data_r4 import batch, load_sequences, prepare_training
from train_aligned_raw_r11 import aligned_pair
from train_exploration_r2 import new_model
from train_mixture_r4b import EXPANDED, mixture_loss
from training_context_r13 import previous_indices, synthetic_pair_with_context

MODES={'copy_context_replay':'spatial_copy','concat_context':'temporal_concat','concat_aligned_context':'temporal_concat'}
REFERENCES={'copy_context_replay':('raw','copy_high'),'concat_context':('raw','concat_high'),
            'concat_aligned_context':('aligned','concat_aligned_high')}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variants',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();variants=args.variants.split(',')
    if len(set(variants))!=len(variants) or any(v not in MODES for v in variants):p.error('unique registered variants required')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda');revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    sources={str(f.relative_to(ROOT)):sha256(f) for f in Path(__file__).parent.glob('*.py')}
    data=prepare_training(load_sequences('train',(320,180),EXPANDED),device);by_scene={r['scene']:r for r in data}
    images=load_images('train');teacher=new_model(device)
    for variant in variants:
        real_cache={}
        for seed in ([0] if variant=='copy_context_replay' else [0,1]):
            folder=out/f'{variant}-seed{seed}';folder.mkdir()
            checkpoint=ROOT/'.cache/checkpoints/exploration-r13'/folder.name;checkpoint.mkdir(parents=True,exist_ok=False)
            old_scope,old_variant=REFERENCES[variant]
            reference=ROOT/'research/4.temporal-consistency/results'/('exploration-r11-'+old_scope)/f'{old_variant}-seed{seed}'
            old=[json.loads(s) for s in (reference/'training.jsonl').read_text().splitlines()]
            plain_path=ROOT/'research/4.temporal-consistency/results/exploration-r5-spatial'/f'plain-seed{seed}'/'training.jsonl'
            plain=[json.loads(s) for s in plain_path.read_text().splitlines()]
            torch.manual_seed(seed);net=RawPairHead(new_model(device),MODES[variant]).to(device).train()
            parameters=list(net.parameters());optimizer=torch.optim.Adam(parameters,lr=1e-3,weight_decay=0)
            config={'variant':variant,'head_mode':MODES[variant],'seed':seed,'steps':1000,'parameters':sum(p.numel() for p in parameters),
                'lr_all':1e-3,'clip_all':1.,'loss':'Charbonnier .001','size':[320,180],'first_task_history':'earlier raw context; real frame0 self only',
                'reference_trace_sha256':sha256(reference/'training.jsonl'),'reference_config_sha256':sha256(reference/'config.json'),
                'plain_trace_sha256':sha256(plain_path),'expanded_manifest_sha256':sha256(EXPANDED),
                'pretrained_teacher_sha256':sha256(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'),
                'protocol_sha256':sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r13-protocol.md'),
                'git_revision_at_startup':revision,'source_sha256_at_startup':sources,'resume_supported':False,
                'cudnn_deterministic':True,'cudnn_tf32_allowed':torch.backends.cudnn.allow_tf32}
            save_json(folder/'config.json',config);start=time.perf_counter();maximum_replay_delta=0.
            for step in range(1000):
                low,target,_,_,identity=batch(data,seed,step)
                sx,sy,meta,synthetic_context,context_meta=synthetic_pair_with_context(images,seed,step)
                low[1],target[1]=sx.to(device),sy.to(device)
                samples=[{'source':'real','scene':identity[0][0],'frame_pair':identity[0][1]},meta]
                if samples!=plain[step]['sample_identity'] or samples!=old[step]['sample_identity']:raise AssertionError('current/GT identity changed')
                scene,t=identity[0];past,_=previous_indices(t)
                first=torch.stack([by_scene[scene]['input'][past],synthetic_context.to(device)])
                previous=torch.stack([first,low[:,0]],1)
                context=[{'source':'real_previous','scene':scene,'frame_index':past,'task_first_index':t-1,'self_fallback':t==1},context_meta]
                details=[];alignment_start=time.perf_counter()
                if 'aligned' in variant:
                    aligned=[]
                    for pair_index in range(2):
                        for slot in range(2):
                            source='real' if pair_index==0 else 'synthetic'
                            current=low[pair_index,slot];prior=previous[pair_index,slot]
                            if pair_index==0 and t-1+slot==0:
                                history=current.detach().permute(1,2,0).cpu().numpy()*255.
                                detail={'valid_fraction':1.,'mean_flow_norm':0.,'raw_pair_mae':0.,'aligned_pair_mae':0.,
                                    'history_array_sha256':hashlib.sha256(history.tobytes()).hexdigest(),'cache_hit':False,'self_fallback':True}
                            else:
                                key=(scene,t-1+slot) if pair_index==0 else None
                                hit=key is not None and key in real_cache
                                if hit:history,detail=real_cache[key]
                                else:
                                    history,detail=aligned_pair(torch.stack([prior,current]),teacher,device)
                                    if key is not None:real_cache[key]=(history,detail)
                                detail={**detail,'cache_hit':hit,'self_fallback':False}
                            aligned.append(history);details.append({**detail,'source':source,'pair_index':pair_index,'task_slot':slot})
                    previous=torch.from_numpy(np.stack(aligned).transpose(0,3,1,2).copy()/255.).to(device).reshape_as(low)
                elapsed=time.perf_counter()-alignment_start
                prediction=net.transform(low.flatten(0,1),previous.flatten(0,1)).reshape_as(low)
                loss=mixture_loss(prediction,target)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite task')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                if step==0:
                    main_norm=float(torch.stack([p.grad.detach().norm() for p in net.backbone.parameters()]).norm())
                    audit={'initial_task_delta':float(loss.detach())-plain[0]['task'],
                        'initial_unclipped_backbone_gradient_norm_delta':main_norm-plain[0]['gradient_norm_backbone']}
                    save_json(folder/'initial_plain_audit.json',audit)
                    if abs(audit['initial_task_delta'])>2e-6 or abs(audit['initial_unclipped_backbone_gradient_norm_delta'])>2e-5:raise AssertionError('initial function changed')
                norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.))
                if not np.isfinite(norm):raise FloatingPointError('nonfinite gradient')
                optimizer.step()
                row={'step':step+1,'task':float(loss.detach()),'gradient_norm_all':norm,'auxiliary_l2':float(net.auxiliary.weight.detach().norm()),
                    'sample_identity':samples,'first_context':context,'alignment':details,'alignment_wall_seconds':elapsed}
                if variant=='copy_context_replay':
                    delta=max(abs(row[k]-old[step][k]) for k in ['task','gradient_norm_all','auxiliary_l2'])
                    maximum_replay_delta=max(maximum_replay_delta,delta);row['reference_trace_max_abs_delta']=delta
                with (folder/'training.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                if maximum_replay_delta>1e-7:
                    save_json(folder/'replay_failure.json',{'step':step+1,'max_abs_trace_delta':maximum_replay_delta,'preset_tolerance':1e-7})
                    raise AssertionError('spatial replay changed')
                if (step+1)%100==0:
                    path=checkpoint/f'step{step+1:04d}.pth'
                    torch.save({'model':net.state_dict(),'optimizer':optimizer.state_dict(),'step':step+1,'config':config},path)
                    save_json(folder/f'checkpoint-step{step+1:04d}.json',{'path':str(path.relative_to(ROOT)),'sha256':sha256(path)})
                    print(json.dumps({'variant':variant,'seed':seed,**row}),flush=True)
            if variant=='copy_context_replay':
                old_path=ROOT/'.cache/checkpoints/exploration-r11/copy_high-seed0/step1000.pth'
                state=torch.load(old_path,map_location=device,weights_only=False)['model']
                delta=max(float((net.state_dict()[k]-v).abs().max()) for k,v in state.items())
                save_json(folder/'spatial_replay_audit.json',{'reference_checkpoint_sha256':sha256(old_path),
                    'max_abs_trace_delta':maximum_replay_delta,'max_abs_model_state_delta':delta,'preset_tolerance':1e-7})
                if delta>1e-7:raise AssertionError('final spatial state replay changed')
            save_json(folder/'runtime.json',{'wall_seconds':time.perf_counter()-start,'checkpoint':str(path.relative_to(ROOT)),
                'checkpoint_sha256':sha256(path),'current_GT_identity_matches_R5_R11':True,'cached_real_pairs':len(real_cache)})
            del optimizer,net


if __name__=='__main__':main()
