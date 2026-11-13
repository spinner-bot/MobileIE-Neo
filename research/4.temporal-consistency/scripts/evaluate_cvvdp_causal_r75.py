"""Complete40 cached final R75 comparisons with the unchanged official perceptual metric."""
import argparse
import fcntl
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from prebind_shared_gpu import require_initial_binding,actual_device
from shared_gpu_admission import sustained_samples,assess,snapshot
from shared_training_schedule import SharedTrainingSchedule,ResourcePressure
from causal_reference_io_r75 import BASE,write,append,lines,digest,stamp
from causal_reference_evaluation_r75 import metas,final_markers,key


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',type=int,default=1)
    args=parser.parse_args()
    bound=require_initial_binding()
    import cv2
    import numpy as np
    import torch
    from cvvdp_adapter_r66 import load_metric,predict
    from evaluate_cvvdp_references_r66 import cached
    from prepare_cache_r5 import load_one
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    final_markers()
    entries=[];sources={}
    for seed in (0,1):
        run=BASE/f'results/exploration-r75-eval-seed{seed}'
        terminal=__import__('json').loads((run/'runtime.json').read_text())
        assert terminal['completed_normally'] and terminal['quality_rows']==20 and terminal['frame_rows']==896
        path=run/'prediction_manifest.jsonl'
        rows=lines(path);assert len(rows)==20
        entries+=rows;sources[str(path.relative_to(ROOT))]=digest(path)
    assert len(entries)==len({key(r) for r in entries})==40
    scenes,indices=metas();first=load_one(scenes[0])
    # CPU constructor happens in the initially bound process; fresh admission follows it.
    prepared_metric,prepared_source=load_metric('cpu')
    assert not torch.cuda.is_initialized()
    out=BASE/f'results/exploration-r75-perceptual{args.attempt:03d}'
    out.mkdir(exist_ok=False)
    started,complete,count,frames,schedule,lock=False,False,0,0,None,None
    started=time.perf_counter()
    try:
        samples=sustained_samples()
        selected=next(r for r in assess(samples)['candidates'] if r['uuid']==bound)
        write(out/'admission.json',dict(samples=samples,selected=selected,real_CPU_metric_prepared=True))
        if not selected['allowed']:
            raise ResourcePressure('Metric bound device failed fresh post-CPU admission.')
        lock=(ROOT/f'.cache/gpu-{selected["index"]}.lock').open('a')
        try:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ResourcePressure('Metric bound GPU project lock is busy.') from error
        immediate=snapshot();write(out/'immediate-pre-CUDA.json',immediate)
        gpu=next(r for r in immediate['GPUs'] if r['uuid']==bound)
        if gpu['total_mib']-gpu['used_mib']<selected['required_free_mib']:
            raise ResourcePressure('Metric pre-CUDA headroom changed.')
        torch.cuda.set_per_process_memory_fraction(2048/selected['total_mib'],device=0)
        marker=torch.empty(1,dtype=torch.uint8,device='cuda');torch.cuda.synchronize()
        actual=actual_device(bound);write(out/'actual-device.json',actual)
        assert actual['matched'];del marker
        metric,source=load_metric('cuda');del prepared_metric
        assert source['package_sources']==prepared_source['package_sources'] and source['playback_fps']==30
        write(out/'metric-source.json',source)
        write(out/'config.json',dict(at=stamp(),expected_rows=40,expected_frames=1792,sources=sources,
            cache_indices=indices,CPU_metric_prepared_before_admission=True,
            source_sha256={n:digest(BASE/'scripts'/n) for n in
                ['evaluate_cvvdp_causal_r75.py','cvvdp_adapter_r66.py','causal_reference_evaluation_r75.py']},
            scope='All final cached outputs; official0.5.7/standard_fhd/nominal30fps/replicate; no enhancer inference or SGD.'))
        schedule=SharedTrainingSchedule(selected);torch.cuda.reset_peak_memory_stats()
        for ordinal,meta in enumerate(scenes):
            data=first if ordinal==0 else load_one(meta)
            target=data['gt'].astype(np.float32)/255
            selected_entries=[r for r in entries if r['scene']==meta['scene']]
            assert len(selected_entries)==4
            for entry in selected_entries:
                schedule.check_memory()
                values=cached(entry);assert values.shape==data['gt'].shape
                test=np.asarray(values,dtype=np.float32)/255
                torch.cuda.synchronize();began=time.perf_counter()
                score,_=predict(metric,test,target)
                torch.cuda.synchronize();seconds=time.perf_counter()-began
                identity={k:entry[k] for k in ['seed','domain','scene','source_video','model','history','method','label','parameters','project_updates']}
                append(out/'per_sequence.jsonl',dict(**identity,JOD=score,frames=len(values),metric_work_seconds=seconds))
                count+=1;frames+=len(values)
                schedule.yield_after_work(seconds)
                append(out/'resource-observations.jsonl',dict(completed_rows=count,**schedule.summary()))
                print(__import__('json').dumps(dict(**identity,completed_rows=count)),flush=True)
                del values,test
            del data,target
        assert (count,frames)==(40,1792);complete=True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        write(out/'runtime.json',dict(ended=stamp(),completed_normally=complete,quality_rows=count,
            comparison_frames=frames,new_enhancement_CNN=0,SGD=0,seconds=time.perf_counter()-started,
            resource=schedule.summary() if schedule is not None else None,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None))
        globals()['_held_metric_lock']=lock


if __name__=='__main__':
    main()
