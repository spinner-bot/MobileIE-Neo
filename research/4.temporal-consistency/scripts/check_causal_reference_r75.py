"""Train-only native sampling, one discarded CPU update and portable recovery qualification."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'scripts'))
from project_assets import mounted_data, write_json
from bound_gpu_observer import BoundGPUObserver
from shared_gpu_admission import snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise SystemExit('Run the entire qualification with an initially empty CUDA mask.')
    out = args.out.resolve()
    if not out.is_relative_to(ROOT):
        raise SystemExit('Use a new versioned project result directory.')
    mounted_data()
    out.mkdir(parents=True, exist_ok=False)
    import cv2
    import numpy as np
    import torch
    from causal_reference_data_r75 import CausalReferencePool
    from causal_reference_training_r75 import optimizer_for, backward_video, finish_update, checkpoint, restore
    from rebotnet_reference import new_model
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    started, complete, updates = time.monotonic(), False, 0
    try:
        pool = CausalReferencePool()
        sources = set(pool.synthetic.names)
        real_names = {r['scene'] for r in pool.real}
        assert len(sources) == 57 and not sources & {'bear','lucia','walking'}
        assert len(real_names) == 12 and not real_names & {'video_15','video_47','video_91','video_121'}
        first = None
        sample_records = []
        for seed, step in [(0,0),(1,0),(0,63999),(1,63999)]:
            low, target, identities = pool.batch(seed, step)
            assert torch.isfinite(low).all() and torch.isfinite(target).all()
            assert 0 <= float(low.min()) <= float(low.max()) <= 1
            assert 0 <= float(target.min()) <= float(target.max()) <= 1
            crop_errors = []
            for b in [0,2]:
                identity = identities[b]
                if b == 0:
                    record = next(r for r in pool.synthetic.scenes[identity['video']] if r['name']==identity['members'][0])
                    native = pool.synthetic.image(record)
                else:
                    row = next(r for r in pool.real if r['scene']==identity['video'])
                    native = row['native_gt'][identity['indices'][0]]
                x,y = identity['xy']
                expected = native[y:y+384,x:x+384]
                if identity['flip_horizontal']:
                    expected = expected[:,::-1]
                if identity['flip_vertical']:
                    expected = expected[::-1]
                expected = torch.from_numpy(expected.astype(np.float32).transpose(2,0,1).copy()/255)
                crop_errors.append(float((expected-target[b,0]).abs().max()))
            assert max(crop_errors) <= 1e-7
            sample_records.append(dict(seed=seed, data_step=step, identities=identities,
                                       direct_source_crop_max_errors=crop_errors))
            if first is None:
                first = (low,target,identities)
        replay = pool.batch(0,0)
        assert replay[2] == first[2] and torch.equal(replay[0],first[0]) and torch.equal(replay[1],first[1])
        low,target,_ = first
        model = new_model(0).train()
        optimizer,scheduler = optimizer_for(model)
        optimizer.zero_grad(set_to_none=True)
        losses = [backward_video(model,low[b:b+1],target[b:b+1]) for b in range(4)]
        gradient_norm = finish_update(model,optimizer,scheduler)
        updates = 1
        cache = ROOT/'.cache/experiments/exploration-r75/cpu-gate001'
        cache.mkdir(parents=True,exist_ok=False)
        path = cache/'discarded.pt'
        torch.save(checkpoint(model,optimizer,scheduler,0,1),path)
        model.eval()
        with torch.no_grad():
            pair = torch.stack((low[:1,0],low[:1,0]),dim=1)
            original = model(pair)
        loaded = new_model(0)
        opt2,sched2 = optimizer_for(loaded)
        saved = torch.load(path,map_location='cpu',weights_only=True)
        restored_step = restore(saved,loaded,opt2,sched2,0)
        with torch.no_grad():
            recovered = loaded.eval()(pair)
        delta = float((original-recovered).abs().max())
        assert delta <= 1e-6 and restored_step == 1 and sched2.last_epoch == scheduler.last_epoch == 1
        assert opt2.param_groups[0]['lr'] == optimizer.param_groups[0]['lr']
        full = snapshot()
        gpu = full['GPUs'][0]
        observed = BoundGPUObserver(gpu['uuid']).read()
        assert observed['uuid'] == gpu['uuid'] and observed['total_mib'] == gpu['total_mib']
        assert 0 <= observed['free_mib'] <= observed['total_mib']
        assert observed['used_mib']+observed['free_mib'] <= observed['total_mib']
        assert not torch.cuda.is_initialized()
        write_json(out/'gate.json',dict(at=datetime.now().astimezone().isoformat(),passed=True,
            data=pool.info,samples=sample_records,replayed_first_data_equal=True,
            discarded_CPU_updates=updates,four_train_video_losses=losses,gradient_norm_before_clip=gradient_norm,
            checkpoint_path=str(path.relative_to(ROOT)),restored_completed_step=restored_step,
            restored_next_data_step=saved['next_data_step'],eval_reload_max_delta=delta,
            learning_rate=opt2.param_groups[0]['lr'], scheduler_epoch=sched2.last_epoch,
            read_only_NVML=observed,CLI_observation=full,formal_updates=0,new_DEV_or_test_pixels=0,
            cuda_initialized=False,
            limits='One discarded CPU optimizer step and portable state loading; no Adam bitwise trajectory audit, new quality or convergence evidence. Resource-trial state is not inherited.'))
        complete=True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        write_json(out/'runtime.json',dict(ended=datetime.now().astimezone().isoformat(),
            completed_normally=complete,discarded_CPU_updates=updates,formal_updates=0,
            seconds=time.monotonic()-started,CUDA_initialized=torch.cuda.is_initialized(),pid=os.getpid()))


if __name__ == '__main__':
    main()
