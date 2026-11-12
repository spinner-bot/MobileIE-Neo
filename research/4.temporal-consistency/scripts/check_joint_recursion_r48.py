"""New recurrence-gradient and real-backbone checks; no DEV or quality selection."""
import argparse
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import cv2
import numpy as np
import torch

from aligned_raw_history_r11 import align_history
from dynamic_training_r34 import BASE, ROOT, digest
from joint_recursion_r48 import MODELS, initial_models, raw_prediction, rollout
from mixed_anchor_r40 import MixedPool
from response_sequence_r45 import motion_for, move_fields
from training_form_r41 import configure_precision, cuda_headroom


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def gradient_cases(device):
    cases=[]
    for accepted in [True, False]:
        raw=torch.rand(1,4,3,8,10,device=device,requires_grad=True)
        fields=[(torch.zeros(1,2,8,10,device=device),
                 torch.full((1,1,8,10),accepted,device=device,dtype=torch.bool)) for _ in range(3)]
        output=rollout(raw,fields)
        gradient=torch.autograd.grad(output[:,-1].sum(),raw)[0]
        expected=raw.new_tensor([.125,.125,.25,.5] if accepted else [0.,0.,0.,1.])[None,:,None,None,None]
        delta=float((gradient-expected).abs().max())
        cases.append(dict(accepted=accepted,max_gradient_delta=delta,
                          first_frame_exact=bool(torch.equal(output[:,0],raw[:,0])),passed=delta<=1e-7))
    assert all(r['passed'] and r['first_frame_exact'] for r in cases)
    return cases


def reference(raw, fields):
    frames=raw.detach().cpu().permute(0,1,3,4,2).contiguous().numpy()
    outputs=[frames[:,0]]
    for t,(flow,mask) in enumerate(fields,1):
        field=flow.detach().cpu().permute(0,2,3,1).contiguous().numpy()
        accepted=mask.detach().cpu().numpy()[:,0]
        moved=np.stack([align_history(outputs[-1][b],frames[b,t],field[b],accepted[b])[0]
                        for b in range(len(frames))])
        outputs.append(frames[:,t]+.5*(moved-frames[:,t]))
    return torch.from_numpy(np.stack(outputs,axis=1).transpose(0,1,4,2,3).copy()).to(raw.device)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',required=True,choices=['cpu','cuda'])
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve()
    assert out.is_relative_to(BASE/'process/exploration-r48')
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    configure_precision();assert not torch.cuda.is_initialized()
    write(out/'config.json',{'at':datetime.now().astimezone().isoformat(),'device':args.device,
        'discarded_updates_registered':4,'protocol_sha256':digest(BASE/'docs/exploration-r48-joint-recursion-protocol.md'),
        'source_sha256':{p.name:digest(p) for p in [Path(__file__),BASE/'scripts/joint_recursion_r48.py',
                       BASE/'scripts/response_cue_r45.py',BASE/'scripts/response_sequence_r45.py']},
        'scope':'Implementation and feasibility only; no DEV pixels or formal quality.'})
    start,updates,complete=time.perf_counter(),0,False
    try:
        nets,marker=initial_models(0)
        pool=MixedPool();low,gt,context=pool.batch(0,8000)
        write(out/'window.json',{'context':context,'data':pool.info,'inherited':marker})
        print('Full owned sources and original8000 MBR models loaded.',flush=True)
        if args.device=='cuda':
            admission=cuda_headroom();write(out/'pre_cuda_guard.json',admission)
            assert admission['allowed']
            nets={n:m.cuda() for n,m in nets.items()};low,gt=low.cuda(),gt.cuda()
            torch.cuda.reset_peak_memory_stats()
        analytic=gradient_cases(args.device)
        raw_values,fold_deltas={},{}
        with torch.no_grad():
            for name,model in nets.items():
                model.eval()
                _,raw=raw_prediction(model,low);raw_values[name]=raw
                slim=model.slim().to(low.device).eval()
                full=model(low[:,0]);folded=slim(low[:,0])
                fold_deltas[name]=float((full-folded).abs().max())
                assert fold_deltas[name]<=1e-4
        initial_delta=float((raw_values['plain']-raw_values['joint']).abs().max())
        assert initial_delta<=1e-6
        fields=move_fields(motion_for(raw_values['plain']),args.device)
        delta=float((rollout(raw_values['plain'],fields)-reference(raw_values['plain'],fields)).abs().max())
        assert delta<=1e-5
        del raw_values,slim,full,folded,fields
        gradients=[]
        for name,model in nets.items():
            model.train()
            watched=model.tail.conv_out.weight
            before=watched.detach().clone()
            optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=1e-4)
            for step in range(2):
                tick=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                _,raw=raw_prediction(model,low)
                fields=move_fields(motion_for(raw),args.device) if name=='joint' else None
                output=rollout(raw,fields) if name=='joint' else raw
                loss=(output-gt).square().mean();assert torch.isfinite(loss)
                loss.backward()
                active=[p for p in model.parameters() if p.requires_grad]
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active)
                reach={key:float(p.grad.norm()) for key,p in [
                    ('head_first',model.head.block1[0].conv.weight),('head_FST',model.head.weight1),
                    ('body_FST',model.body.weight1),('tail_readout',watched)]}
                assert all(v>0 for v in reach.values()),reach
                norm=float(torch.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True))
                optimizer.step();updates+=1
                gradients.append(dict(model=name,step=step+1,MSE=float(loss.detach()),
                                 norm_before_clip=norm,reach=reach,seconds=time.perf_counter()-tick))
            assert float((watched-before).detach().abs().max())>0
        write(out/'gate.json',{'passed':True,'analytic':analytic,'initial_raw_delta':initial_delta,
              'initial_fold_deltas':fold_deltas,'recursive_OpenCV_delta':delta,'gradients':gradients,
              'discarded_updates':updates,'inherit_into_formal':False,
              'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else None,
              'dataset_scope':'MixedPool training only, DEV pixels0'})
        complete=True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        write(out/'runtime.json',{'completed_normally':complete,'discarded_updates':updates,
              'seconds':time.perf_counter()-start,'ended':datetime.now().astimezone().isoformat()})


if __name__=='__main__':
    main()
