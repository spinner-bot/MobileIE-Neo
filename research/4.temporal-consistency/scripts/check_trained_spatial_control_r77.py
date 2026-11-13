"""Artificial CPU qualification of current-only observation paths and one discarded full update."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

from causal_reference_io_r75 import ROOT,BASE,write,digest,stamp,verify_training_sources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or args.out.exists():
        raise SystemExit('Require an initially empty CUDA mask and new output directory.')
    assert args.out.resolve().is_relative_to(ROOT)
    import subprocess
    subprocess.run(['mountpoint','-q','/data'],check=True)
    args.out.mkdir(parents=True,exist_ok=False)
    import torch
    from rebotnet_reference import new_model
    from trained_spatial_control_r77 import spatial_predictions,backward_video,optimizer_for,finish_update,numerical_investment_status
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    started=time.perf_counter();complete=False;updates=0
    write(args.out/'config.json',dict(at=stamp(),CPU=True,seed=0,artificial_input_seed=7700,
        input_shape=[4,4,3,384,384],formal_updates=0,new_video_pixels=0,
        protocol_sha256=digest(BASE/'docs/exploration-r77-trained-spatial-control-protocol.md'),
        source_sha256={n:digest(BASE/'scripts'/n) for n in ['trained_spatial_control_r77.py',
            'check_trained_spatial_control_r77.py','rebotnet_reference.py','causal_reference_training_r75.py']},
        scope='Random original XS weights and artificial tensors only; no cached video or trained checkpoint read.'))
    try:
        verify_training_sources()
        for name in ('exploration-r75/evaluation-freeze001.json','exploration-r76/evaluation-freeze001.json'):
            frozen=json.loads((BASE/'process'/name).read_text())['source_sha256']
            assert all(digest(ROOT/p)==h for p,h in frozen.items())
        generator=torch.Generator().manual_seed(7700)
        low=torch.rand((4,4,3,384,384),generator=generator)*.25
        target=(low*3+.02).clamp(0,1)
        model=new_model(0).eval();assert sum(p.numel() for p in model.parameters())==890275
        calls=[]
        def observe(module,inputs):
            pair=inputs[0]
            calls.append(dict(shape=list(pair.shape),slot_difference=float((pair[:,0]-pair[:,1]).abs().max())))
        hook=model.register_forward_pre_hook(observe)
        probe=low[:1].clone().requires_grad_(True)
        outputs=spatial_predictions(model,probe)
        assert len(calls)==4 and all(r['slot_difference']==0 for r in calls)
        input_gradient=torch.autograd.grad(outputs[:,3].square().mean(),probe)[0]
        cross=float(input_gradient[:,:3].abs().max());current=float(input_gradient[:,3].abs().max())
        assert cross==0 and current>0
        hook.remove();del outputs,input_gradient,probe
        model.train();optimizer,scheduler=optimizer_for(model);optimizer.zero_grad(set_to_none=True)
        losses=[backward_video(model,low[b:b+1],target[b:b+1]) for b in range(4)]
        norm=finish_update(model,optimizer,scheduler);updates=1
        assert scheduler.last_epoch==1 and not torch.cuda.is_initialized()
        assert all(torch.isfinite(p).all() for p in model.parameters())
        status=numerical_investment_status()
        assert status=='pending_complete_original_and_self4_results'
        complete=True
        write(args.out/'gate.json',dict(at=stamp(),passed=True,parameters=890275,
            artificial_eval_calls=calls,last_output_gradient_to_other_frames_max=cross,
            last_output_gradient_to_current_frame_max=current,training_videos=4,training_frames=16,
            discarded_full_CPU_updates=updates,formal_updates=0,losses=losses,gradient_norm_before_clip=norm,
            scheduler_step=scheduler.last_epoch,lr=optimizer.param_groups[0]['lr'],
            numerical_investment_status=status,formal_GPU_launch_authorized=False,
            R75_training17_and_evaluation27_R76_evaluation37_unchanged=True,
            trained_checkpoint_read=False,real_video_read=False,CUDA_initialized=False,
            scope='CPU observation-path and discarded update qualification only; not complete trainer, quality, convergence or GPU readiness.'))
        print((args.out/'gate.json').read_text(),flush=True)
    except BaseException:
        (args.out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        write(args.out/'runtime.json',dict(ended=stamp(),completed_normally=complete,
            discarded_CPU_updates=updates,formal_updates=0,new_DEV=0,GPU=0,
            seconds=time.perf_counter()-started))


if __name__=='__main__':
    main()
