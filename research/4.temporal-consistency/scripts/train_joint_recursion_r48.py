"""Matched fixed-budget continuation: raw supervision versus fixed-recursive output."""
import argparse
import json
import os
import subprocess
import time
import traceback
from datetime import datetime

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import cv2
import torch

from dynamic_training_r34 import BASE, ROOT, digest
from joint_recursion_r48 import MODELS, initial_models, raw_prediction, rollout
from mixed_anchor_r40 import MixedPool, prediction
from response_sequence_r45 import motion_for, move_fields
from training_form_r41 import configure_precision, cuda_headroom


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def checkpoint(net, optimizer, name, seed, step, low, cache, out):
    net.eval()
    train_path=cache/f'{name}-training-step{step:04d}.pt'
    deploy_path=cache/f'{name}-deploy-step{step:04d}.pt'
    torch.save({'state_dict':net.state_dict(),'optimizer':optimizer.state_dict(),
                'model':name,'seed':seed,'new_step':step,'inherited_updates':8000},train_path)
    slim=net.slim().to(low.device).eval()
    assert sum(p.numel() for p in slim.parameters())==4075
    delta=None
    if step==4000:
        with torch.no_grad():delta=float((prediction(net,low)-prediction(slim,low)).abs().max())
    torch.save({'state_dict':slim.state_dict(),'model':name,'seed':seed,'new_step':step,
                'inherited_updates':8000},deploy_path)
    marker={'model':name,'seed':seed,'new_step':step,'inherited_updates':8000,
            'path':str(deploy_path.relative_to(ROOT)),'sha256':digest(deploy_path),
            'training_path':str(train_path.relative_to(ROOT)),'training_sha256':digest(train_path),
            'deployment_parameters':4075,'final_fold_delta':delta,
            'final_fold_passed':None if delta is None else delta<=1e-4}
    write(out/f'{name}-step{step:04d}.json',marker)
    net.train()
    return marker


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=[0,1],required=True)
    args=parser.parse_args()
    gates=[BASE/f'process/exploration-r48/{n}-gate001/gate.json' for n in ['cpu','cuda']]
    assert all(json.loads(p.read_text())['passed'] for p in gates)
    assert json.loads((BASE/'process/exploration-r48/cuda-gate001-guard.json').read_text())['returncode']==0
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    configure_precision();assert not torch.cuda.is_initialized()
    out=BASE/f'results/exploration-r48-stage001-seed{args.seed}'
    out.mkdir(exist_ok=False)
    start,complete,updates,final=time.perf_counter(),False,{n:0 for n in MODELS},{}
    try:
        nets,marker=initial_models(args.seed)
        pool=MixedPool();first_low,first_gt,first_context=pool.batch(args.seed,8000)
        admission=cuda_headroom();write(out/'pre_cuda_guard.json',admission)
        assert admission['allowed']
        nets={n:m.cuda().train() for n,m in nets.items()}
        optimizers={n:torch.optim.Adam((p for p in m.parameters() if p.requires_grad),
                   lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0) for n,m in nets.items()}
        cache=ROOT/f'.cache/checkpoints/exploration-r48/stage001/seed{args.seed}'
        assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
        cache.mkdir(parents=True,exist_ok=False)
        names=['train_joint_recursion_r48.py','joint_recursion_r48.py','response_cue_r45.py',
               'response_sequence_r45.py','budget_causal_r33.py','half_motion_r4.py','video_metrics.py',
               'mixed_anchor_r40.py','dynamic_training_r34.py','training_form_r41.py','gpu_guard.py']
        write(out/'config.json',{'at':datetime.now().astimezone().isoformat(),'seed':args.seed,
              'models':MODELS,'new_steps_each':4000,'inherited_steps':8000,'data_steps':[8000,11999],
              'initialization':marker,'data':pool.info,'batch':4,'frames':4,'crop':128,
              'losses':{'plain':'MSE clipped raw','joint':'MSE deployed fixed-recursive .5 output'},
              'alpha':.5,'fresh_Adam_lr':1e-4,'betas':[.9,.999],'eps':1e-8,'weight_decay':0,'clip':1,
              'BN':'Four sequential B4 calls; original complete MBR training form',
              'FP32':True,'TF32':False,'history_detached':False,'flow_gradient':False,
              'selection':'Final new4000 only; deployment4075 and same inference operator',
              'protocol_sha256':digest(BASE/'docs/exploration-r48-joint-recursion-protocol.md'),
              'gates':{str(p.relative_to(ROOT)):digest(p) for p in gates},
              'source_sha256':{n:digest(BASE/'scripts'/n) for n in names},
              'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})
        with (out/'training.jsonl').open('x') as trace,(out/'windows.jsonl').open('x') as windows:
            for step in range(4000):
                low,gt,context=((first_low,first_gt,first_context) if step==0
                                else pool.batch(args.seed,8000+step))
                windows.write(json.dumps({'new_step':step+1,'data_step':8000+step,'windows':context})+'\n')
                low,gt=low.cuda(),gt.cuda()
                initial={n:raw_prediction(m,low) for n,m in nets.items()} if step==0 else None
                if initial is not None:
                    delta=float((initial['plain'][1]-initial['joint'][1]).detach().abs().max())
                    write(out/'initial-real-window.json',{'raw_max_delta':delta,'passed':delta<=1e-6,
                          'context':context,'BN_calls_each':4})
                    assert delta<=1e-6
                last=[]
                for name,net in nets.items():
                    optimizer=optimizers[name];optimizer.zero_grad(set_to_none=True)
                    unclip,raw=initial[name] if initial is not None else raw_prediction(net,low)
                    assert torch.isfinite(unclip).all()
                    validity=None
                    if name=='joint':
                        cpu_fields=motion_for(raw)
                        validity=sum(float(mask.float().mean()) for _,mask in cpu_fields)/len(cpu_fields)
                        output=rollout(raw,move_fields(cpu_fields,'cuda'))
                    else:output=raw
                    error=(output-gt).square();loss=error.mean();assert torch.isfinite(loss)
                    loss.backward()
                    active=[p for p in net.parameters() if p.requires_grad]
                    norm=float(torch.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True))
                    optimizer.step();updates[name]+=1
                    row={'model':name,'new_step':step+1,'data_step':8000+step,'MSE':float(loss.detach()),
                         'synthetic_MSE':float(error[:2].detach().mean()),'real_MSE':float(error[2:].detach().mean()),
                         'unclipped_raw_MSE':float((unclip.detach()-gt).square().mean()),
                         'clipped_component_fraction':float(((unclip.detach()<=0)|(unclip.detach()>=1)).float().mean()),
                         'gradient_norm_before_clip':norm,'own_raw_valid_fraction':validity}
                    trace.write(json.dumps(row,allow_nan=False)+'\n');last.append(row)
                if (step+1)%100==0:
                    trace.flush();windows.flush()
                    print(json.dumps({'new_step':step+1,'models':last,'seconds':time.perf_counter()-start}),flush=True)
                if (step+1)%1000==0:
                    for name,net in nets.items():
                        final[name]=checkpoint(net,optimizers[name],name,args.seed,step+1,low,cache,out)
        assert all(v==4000 for v in updates.values()) and all(v['final_fold_passed'] for v in final.values())
        complete=True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        write(out/'runtime.json',{'completed_normally':complete,'model_updates':updates,
              'total_new_updates':sum(updates.values()),'final_checkpoints':final,
              'seconds':time.perf_counter()-start,'ended':datetime.now().astimezone().isoformat()})


if __name__=='__main__':
    main()
