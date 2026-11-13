"""Fixed64K self-start ReBotNet-XS reference, shared-card scheduling and explicit recovery."""
import argparse
from collections import deque
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
from prebind_shared_gpu import require_initial_binding, actual_device
from shared_gpu_admission import assess, snapshot, sustained_samples
from shared_training_schedule import SharedTrainingSchedule, ResourcePressure
from causal_reference_io_r75 import BASE, PROCESS, PROTOCOL, seed_root, cache_root, digest, write, append, stamp, verify_training_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[0,1], required=True)
    parser.add_argument('--attempt', type=int, required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.attempt < 1:
        parser.error('Attempt must be positive.')
    bound = require_initial_binding()
    root = seed_root(args.seed)
    seed_lock = (ROOT/f'.cache/rebotnet-r75-seed{args.seed}.lock').open('a')
    fcntl.flock(seed_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root/'final.json').exists():
        raise SystemExit('Final reference already exists; no scientific rerun.')
    out = root/f'attempt{args.attempt:03d}'
    out.mkdir(parents=True, exist_ok=False)
    import cv2
    import torch
    from causal_reference_data_r75 import CausalReferencePool
    from causal_reference_training_r75 import STEPS, optimizer_for, backward_video, finish_update, rng_state, checkpoint, restore
    from rebotnet_reference import new_model
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    started = time.perf_counter()
    completed, updates, inherited, videos, partial_videos = False, 0, 0, 0, 0
    reason, schedule, gpu_lock, model, optimizer, scheduler, start_rng = 'error', None, None, None, None, None, None
    latest_marker, final_marker, cache = None, None, None

    def save(kind, rng=None):
        nonlocal latest_marker
        path = cache/f'{kind}-step{updates:06d}.pt'
        if path.exists():
            raise RuntimeError('Refuse checkpoint overwrite.')
        temporary = path.with_suffix('.pt.tmp')
        torch.save(checkpoint(model,optimizer,scheduler,args.seed,updates,rng),temporary)
        os.replace(temporary,path)
        marker = dict(at=stamp(),seed=args.seed,completed_steps=updates,next_data_step=updates,
                      registered_steps=STEPS,attempt=args.attempt,kind=kind,path=str(path.relative_to(ROOT)),
                      sha256=digest(path),bytes=path.stat().st_size,parameters=890275,
                      source_freeze_sha256=digest(PROCESS/'source-freeze001.json'),
                      final_quality_eligible=updates==STEPS and kind=='regular')
        write(out/(path.stem+'.json'),marker)
        latest_marker=marker
        return marker

    try:
        frozen = verify_training_sources()
        gate = json.loads((PROCESS/'cpu-gate001/gate.json').read_text())
        assert gate['passed'] and gate['formal_updates']==0 and not gate['cuda_initialized']
        if args.resume is None and args.attempt != 1:
            previous = sorted(root.glob('attempt*/runtime.json'))
            assert previous and all(json.loads(p.read_text())['stop_reason']=='resource'
                                    and json.loads(p.read_text())['completed_steps']==0 for p in previous)
        saved, resume_marker = None, None
        if args.resume is not None:
            marker_path=args.resume.resolve()
            assert marker_path.is_relative_to(root)
            resume_marker=json.loads(marker_path.read_text())
            terminal=json.loads((marker_path.parent/'runtime.json').read_text())
            assert terminal['stop_reason']=='resource' and not terminal['completed_normally']
            assert resume_marker['seed']==args.seed and resume_marker['source_freeze_sha256']==digest(PROCESS/'source-freeze001.json')
            path=ROOT/resume_marker['path']
            assert path.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo') and digest(path)==resume_marker['sha256']
            saved=torch.load(path,map_location='cpu',weights_only=True)
            inherited=updates=int(resume_marker['completed_steps'])
            assert saved['completed_steps']==updates and 0<=updates<STEPS
        pool=CausalReferencePool()
        first=pool.batch(args.seed,updates)
        model=new_model(args.seed).train()
        assert sum(p.numel() for p in model.parameters())==890275 and not torch.cuda.is_initialized()
        write(out/'config.json',dict(at=stamp(),pid=os.getpid(),seed=args.seed,attempt=args.attempt,
            revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_freeze_sha256=digest(PROCESS/'source-freeze001.json'),sources=frozen,
            data=pool.info,registered_steps=STEPS,inherited_steps=inherited,first_data_step=updates,
            first_window=first[2],resume_marker=resume_marker,CPU_gate_sha256=digest(PROCESS/'cpu-gate001/gate.json'),
            batch_videos=4,frames=4,crop=384,objective='mean sqrt(error^2+1e-6)',
            temporal_BPTT=True,GT_history=False,internal_clamp=False,initialization='fresh author XS random seed; no resource trial state',
            schedule='Adam4e-4 cosine64K to1e-7, FP32/noTF32, four sequential videos accumulated, clip1',
            resource='initial UUID, new post-CPU admission, allocator2GiB, expected process4GiB, 25% cooperative work duty',
            scope='Task-trained published-architecture adaptation; not original500K reproduction or novelty.'))
        samples=sustained_samples()
        selected=next(r for r in assess(samples)['candidates'] if r['uuid']==bound)
        write(out/'admission.json',dict(samples=samples,selected=selected,CPU_preparation_completed=True))
        if not selected['allowed']:
            raise ResourcePressure('Bound device failed fresh post-CPU admission.')
        gpu_lock=(ROOT/f'.cache/gpu-{selected["index"]}.lock').open('a')
        try:
            fcntl.flock(gpu_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ResourcePressure('Bound GPU is locked by another project task.') from error
        immediate=snapshot()
        write(out/'immediate-pre-CUDA.json',immediate)
        current=next(r for r in immediate['GPUs'] if r['uuid']==bound)
        if current['total_mib']-current['used_mib']<selected['required_free_mib']:
            raise ResourcePressure('Headroom changed immediately before CUDA.')
        torch.cuda.set_per_process_memory_fraction(2048/selected['total_mib'],device=0)
        marker=torch.empty(1,dtype=torch.uint8,device='cuda')
        torch.cuda.synchronize()
        actual=actual_device(bound)
        write(out/'actual-device.json',actual)
        assert actual['matched']
        del marker
        model=model.cuda()
        optimizer,scheduler=optimizer_for(model)
        if saved is not None:
            assert restore(saved,model,optimizer,scheduler,args.seed)==updates
            del saved
        cache=cache_root(args.seed,args.attempt)
        cache.mkdir(parents=True,exist_ok=False)
        schedule=SharedTrainingSchedule(selected)
        torch.cuda.reset_peak_memory_stats()
        recent=deque(maxlen=100)
        for step in range(updates,STEPS):
            partial_videos=0
            start_rng=rng_state(model)
            low,target,identities=first if step==inherited else pool.batch(args.seed,step)
            optimizer.zero_grad(set_to_none=True)
            losses=[]
            for b in range(4):
                schedule.check_memory()
                torch.cuda.synchronize()
                began=time.perf_counter()
                values,targets=low[b:b+1].cuda(),target[b:b+1].cuda()
                loss=backward_video(model,values,targets)
                del values,targets
                torch.cuda.synchronize()
                videos+=1; partial_videos+=1; losses.append(loss)
                schedule.yield_after_work(time.perf_counter()-began)
            began=time.perf_counter()
            norm=finish_update(model,optimizer,scheduler)
            torch.cuda.synchronize()
            updates=step+1; partial_videos=0; start_rng=None
            row=dict(at=stamp(),completed_steps=updates,data_step=step,loss=sum(losses)/4,
                     synthetic_loss=sum(losses[:2])/2,real_loss=sum(losses[2:])/2,
                     gradient_norm_before_clip=norm,lr=optimizer.param_groups[0]['lr'])
            recent.append(row)
            if updates==1 or updates%1000==0:
                marker=save('regular')
                append(out/'checkpoint-windows.jsonl',dict(completed_steps=updates,identities=identities))
                if updates==STEPS:
                    final_marker=marker
            if updates==inherited+1 or updates%100==0:
                row.update(window_mean_loss=sum(r['loss'] for r in recent)/len(recent),
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved(),resource=schedule.summary())
                append(out/'training.jsonl',row)
                write(out/'progress.json',dict(**row,seed=args.seed,attempt=args.attempt,
                    pid=os.getpid(),registered_steps=STEPS,completed_normally=False,latest_checkpoint=latest_marker))
                print(json.dumps({k:v for k,v in row.items() if k!='resource'}),flush=True)
            schedule.yield_after_work(time.perf_counter()-began)
        assert updates==STEPS and final_marker is not None
        completed=True; reason='complete'
    except ResourcePressure:
        reason='resource'
        (out/'resource-exit.txt').write_text(traceback.format_exc())
        if optimizer is not None and cache is not None:
            latest_marker=save('resource',start_rng)
        if schedule is not None:
            write(out/'resource-exit-observations.json',schedule.summary())
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write(out/'runtime.json',dict(ended=stamp(),pid=os.getpid(),completed_normally=completed,
            stop_reason=reason,seed=args.seed,attempt=args.attempt,inherited_steps=inherited,
            completed_steps=updates,new_steps=updates-inherited,registered_steps=64000,
            video_backwards_this_attempt=videos,uncommitted_video_backwards=partial_videos,
            latest_checkpoint=latest_marker,seconds=time.perf_counter()-started,
            resource=schedule.summary() if schedule is not None else None,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None,
            scope='Full child lifecycle includes CPU data, gating and yielding; not phone or inference latency.'))
        if completed:
            write(root/'final.json',dict(**final_marker,runtime=str((out/'runtime.json').relative_to(ROOT)),
                                       completed_normally=True,formal_updates=64000))
        globals()['_held_seed_lock']=seed_lock
        globals()['_held_gpu_lock']=gpu_lock
    if not completed:
        raise SystemExit(75)


if __name__=='__main__':
    main()
