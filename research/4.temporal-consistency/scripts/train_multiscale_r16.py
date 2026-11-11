"""Native 4000-step same-capacity spatial/raw/aligned encoder-decoder controls."""
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
from multiscale_fst_r16 import MODES,MultiscaleFST
from native_training_r14 import load_training,real_pair
from train_exploration_r2 import new_model
from train_mixture_r4b import mixture_loss
from train_native_r14 import align_tasks
from training_context_r13 import synthetic_pair_with_context


def create_model(variant,device):return MultiscaleFST(new_model(device),variant).to(device)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variants',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();variants=args.variants.split(',')
    if len(set(variants))!=len(variants) or any(v not in MODES for v in variants):p.error('registered unique variants required')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda');revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    sources={str(f.relative_to(ROOT)):sha256(f) for f in Path(__file__).parent.glob('*.py')}
    begin=time.perf_counter();data,integrity=load_training();images=load_images('train');teacher=new_model(device).eval()
    save_json(out/'source_integrity.json',{**integrity,'load_wall_seconds':time.perf_counter()-begin})
    for variant in variants:
        for seed in [0,1]:
            label=f'{variant}-seed{seed}';folder=out/label;folder.mkdir()
            checkpoints=ROOT/'.cache/checkpoints/exploration-r16'/label;checkpoints.mkdir(parents=True,exist_ok=False)
            reference_path=ROOT/'research/4.temporal-consistency/results/exploration-r14-spatial'/f'native_crop-plain-seed{seed}'/'training.jsonl'
            reference=[json.loads(s) for s in reference_path.read_text().splitlines()];assert len(reference)==4000
            torch.manual_seed(seed);net=create_model(variant,device).train();parameters=list(net.parameters())
            optimizer=torch.optim.Adam(parameters,lr=1e-3,weight_decay=0)
            config={'variant':variant,'preparation':'native_crop','seed':seed,'start_step':0,'final_step':4000,
                'parameters':sum(p.numel() for p in parameters),'lr_all':1e-3,'clip_all':1.,'optimizer':'Adam wd0',
                'loss':'Charbonnier .001','size':[320,180],'training_scenes':integrity['scenes'],
                'manifest_sha256':integrity['manifest_sha256'],'matched_r14_trace_sha256':sha256(reference_path),
                'pretrained_sha256':sha256(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'),
                'protocol_sha256':sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r16-protocol.md'),
                'git_revision_at_startup':revision,'source_sha256_at_startup':sources,'resume_supported':False,
                'torch_version':torch.__version__,'opencv_version':cv2.__version__,'cudnn_deterministic':True,
                'cudnn_tf32_allowed':torch.backends.cudnn.allow_tf32,'native_flow_cache':'disabled; ROI changes',
                'scope':'standard multiscale mechanism reference, not a novelty claim; one past frame, no future'}
            save_json(folder/'config.json',config);start=time.perf_counter();recent=[]
            for step in range(4000):
                rx,ry,rp,identity,real_context=real_pair(data,seed,step,'native_crop')
                sx,sy,meta,sp,synthetic_context=synthetic_pair_with_context(images,seed,step)
                samples=[identity,meta];ref=reference[step]
                assert samples==ref['sample_identity'] and real_context==ref['real_context'] and synthetic_context==ref['synthetic_context']
                low=torch.stack([rx,sx]).to(device);target=torch.stack([ry,sy]).to(device)
                previous=torch.stack([rp,torch.stack([sp,sx[0]])]).to(device)
                align_start=time.perf_counter();details=[]
                if variant=='aligned_history':previous,details=align_tasks(low,previous,teacher,device,'native_crop',identity['scene'],identity['frame_pair'],{})
                align_seconds=time.perf_counter()-align_start
                prediction=net.transform(low.flatten(0,1),previous.flatten(0,1)).reshape_as(low)
                loss=mixture_loss(prediction,target)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite task')
                if step==0:
                    with torch.no_grad():expected=teacher(low.flatten(0,1)).reshape_as(low)
                    audit={'output_max_abs':float((prediction.detach()-expected).abs().max()),
                        'task_delta':float(loss.detach()-mixture_loss(expected,target)),
                        'preset_output_tolerance':2e-5,'parameters':config['parameters']}
                    save_json(folder/'initial_plain_audit.json',audit)
                    if audit['output_max_abs']>2e-5:raise AssertionError('initial official parity exceeded preset tolerance')
                optimizer.zero_grad(set_to_none=True);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.))
                if not np.isfinite(norm):raise FloatingPointError('nonfinite gradients')
                optimizer.step();recent.append(float(loss.detach()))
                row={'step':step+1,'task':float(loss.detach()),'gradient_norm_all':norm,'sample_identity':samples,
                    'real_context':real_context,'synthetic_context':synthetic_context,'alignment':details,
                    'alignment_wall_seconds':align_seconds,'decoder_context_l2':float(net.backbone.body.block1.context.weight.detach().norm())}
                with (folder/'training.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                if (step+1)%100==0:print(json.dumps({'label':label,'step':step+1,'task':row['task'],
                    'gradient_norm_all':norm,'decoder_context_l2':row['decoder_context_l2']}),flush=True)
                if (step+1)%1000==0:
                    path=checkpoints/f'step{step+1:04d}.pth';torch.save({'model':net.state_dict(),'optimizer':optimizer.state_dict(),
                        'step':step+1,'config':config},path)
                    save_json(folder/f'checkpoint-step{step+1:04d}.json',{'path':str(path.relative_to(ROOT)),
                        'sha256':sha256(path),'step':step+1})
                    prefix=(folder/'training.jsonl').read_bytes()
                    save_json(folder/f'milestone-step{step+1:04d}.json',{'step':step+1,'last100_task_mean':float(np.mean(recent[-100:])),
                        'trace_prefix_bytes':len(prefix),'trace_prefix_sha256':hashlib.sha256(prefix).hexdigest(),
                        'wall_seconds':time.perf_counter()-start,'checkpoint_sha256':sha256(path)})
            save_json(folder/'runtime.json',{'wall_seconds':time.perf_counter()-start,'checkpoint':str(path.relative_to(ROOT)),
                'checkpoint_sha256':sha256(path),'final_step':4000,'new_updates':4000,'all_r14_task_context_identities_exact':True})
            del optimizer,net


if __name__=='__main__':main()
