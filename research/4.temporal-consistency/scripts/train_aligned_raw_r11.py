"""Strong matched optimization for spatial, raw-past, and aligned-past heads."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from aligned_raw_history_r11 import align_history
from exploration_r2_data import ROOT, load_images
from explore_temporal_controls import inference, save_json, sha256
from half_motion_r4 import half_forward_backward
from raw_pair_head_r8 import RawPairHead
from real_data_r4 import batch, load_sequences, prepare_training
from train_exploration_r2 import new_model
from train_mixture_r4b import EXPANDED, mixture_loss, synthetic_pair

MODES={'copy_high':'spatial_copy','concat_high':'temporal_concat','concat_aligned_high':'temporal_concat'}


def aligned_pair(pair, teacher, device):
    previous,current=pair.detach().permute(0,2,3,1).cpu().numpy()*255.
    enhanced=inference(teacher,[previous,current],device,1)[0]
    flow,mask=half_forward_backward(*enhanced)
    history,valid=align_history(previous,current,flow,mask)
    return history,{'valid_fraction':float(valid.mean()),'mean_flow_norm':float(np.linalg.norm(flow,axis=2).mean()),
        'raw_pair_mae':float(np.abs(previous-current).mean()),'aligned_pair_mae':float(np.abs(history-current).mean()),
        'history_array_sha256':hashlib.sha256(history.tobytes()).hexdigest()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variants',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();variants=args.variants.split(',')
    if len(set(variants))!=len(variants) or any(v not in MODES for v in variants):p.error('registered unique variants required')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    sources={str(f.relative_to(ROOT)):sha256(f) for f in Path(__file__).parent.glob('*.py')}
    data=prepare_training(load_sequences('train',(320,180),EXPANDED),device)
    images=load_images('train');teacher=new_model(device)
    for variant in variants:
        real_cache={}
        for seed in [0,1]:
            folder=out/f'{variant}-seed{seed}';folder.mkdir()
            checkdir=ROOT/'.cache/checkpoints/exploration-r11'/folder.name;checkdir.mkdir(parents=True,exist_ok=False)
            prior_path=ROOT/'research/4.temporal-consistency/results/exploration-r5-spatial'/f'plain-seed{seed}'/'training.jsonl'
            prior=[json.loads(s) for s in prior_path.read_text().splitlines()]
            torch.manual_seed(seed);net=RawPairHead(new_model(device),MODES[variant]).to(device).train()
            parameters=list(net.parameters());optimizer=torch.optim.Adam(parameters,lr=1e-3,weight_decay=0)
            config={'variant':variant,'head_mode':MODES[variant],'seed':seed,'steps':1000,'parameters':sum(p.numel() for p in parameters),
                'lr_all':1e-3,'clip_all':1.,'optimizer':'Adam wd0','loss':'Charbonnier .001','size':[320,180],
                'protocol_sha256':sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r11-protocol.md'),
                'expanded_manifest_sha256':sha256(EXPANDED),'plain_trace_sha256':sha256(prior_path),
                'pretrained_teacher_sha256':sha256(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'),
                'git_revision_at_startup':revision,'source_sha256_at_startup':sources,'resume_supported':False,
                'alignment':'fixed official teacher half FB; rejected raw history replaced by current' if 'aligned' in variant else 'none',
                'cudnn_deterministic':True,'cudnn_tf32_allowed':torch.backends.cudnn.allow_tf32}
            assert config['parameters']==4975
            save_json(folder/'config.json',config);start=time.perf_counter()
            for step in range(1000):
                low,target,_,_,identity=batch(data,seed,step)
                sx,sy,meta=synthetic_pair(images,seed,step);low[1],target[1]=sx.to(device),sy.to(device)
                samples=[{'source':'real','scene':identity[0][0],'frame_pair':identity[0][1]},meta]
                if samples!=prior[step]['sample_identity']:raise AssertionError('current/GT stream changed')
                history=None;details=[];alignment_start=time.perf_counter()
                if 'aligned' in variant:
                    key=identity[0]
                    hit=key in real_cache
                    if not hit:real_cache[key]=aligned_pair(low[0],teacher,device)
                    real,rd=real_cache[key];synth,sd=aligned_pair(low[1],teacher,device)
                    history=torch.from_numpy(np.stack([real,synth]).transpose(0,3,1,2).copy()/255.).to(device)
                    details=[{**rd,'source':'real','cache_hit':hit},{**sd,'source':'synthetic','cache_hit':False}]
                elapsed=time.perf_counter()-alignment_start
                loss=mixture_loss(net.forward_pair(low,history),target)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite task')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                if step==0:
                    main_norm=float(torch.stack([p.grad.detach().norm() for p in net.backbone.parameters()]).norm())
                    audit={'initial_task_delta':float(loss.detach())-prior[0]['task'],
                           'initial_unclipped_backbone_gradient_norm_delta':main_norm-prior[0]['gradient_norm_backbone']}
                    if abs(audit['initial_task_delta'])>2e-6 or abs(audit['initial_unclipped_backbone_gradient_norm_delta'])>2e-5:
                        raise AssertionError('initial function/backbone gradient changed')
                    save_json(folder/'initial_plain_audit.json',audit)
                norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.))
                if not np.isfinite(norm):raise FloatingPointError('nonfinite gradient')
                optimizer.step()
                row={'step':step+1,'task':float(loss.detach()),'gradient_norm_all':norm,
                     'sample_identity':samples,'alignment':details,'alignment_wall_seconds':elapsed,
                     'auxiliary_l2':float(net.auxiliary.weight.detach().norm())}
                with (folder/'training.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                if (step+1)%100==0:
                    path=checkdir/f'step{step+1:04d}.pth'
                    torch.save({'model':net.state_dict(),'optimizer':optimizer.state_dict(),'step':step+1,'config':config},path)
                    save_json(folder/f'checkpoint-step{step+1:04d}.json',{'path':str(path.relative_to(ROOT)),'sha256':sha256(path)})
                    print(json.dumps({'variant':variant,'seed':seed,**row}),flush=True)
            save_json(folder/'runtime.json',{'wall_seconds':time.perf_counter()-start,'checkpoint':str(path.relative_to(ROOT)),
                'checkpoint_sha256':sha256(path),'all_current_samples_match_plain':True,'cached_real_pairs':len(real_cache)})
            del optimizer,net


if __name__=='__main__':main()
