"""Fixed D1 screen on complete cached F0 sequences; no neural inference."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='2'
import time
import traceback
from d1_io import BASE,ROOT,ARMS,AUDIT,append,cache,digest,identity,owned,read,stamp,verify,write
import cv2
import numpy as np
from core import indices,qualification,synthesize
from half_motion_r4 import half_forward_backward
from aligned_raw_history_r11 import align_history
from exploration_r2_data import evaluate
from summarize import summarize


def load_prediction(entry,shape):
    path=owned(ROOT/entry['path'])
    assert digest(path)==entry['sha256']
    array=np.load(path,mmap_mode='r',allow_pickle=False)
    assert list(array.shape)==shape and array.dtype==np.float32
    assert np.isfinite(array).all()
    return array


def main():
    cv2.setNumThreads(1);os.nice(5)
    frozen=verify();source=read(AUDIT)
    out,assets=BASE/'results/eval001',cache()
    out.mkdir(parents=True,exist_ok=False);assets.mkdir(parents=True,exist_ok=False)
    began,complete=time.perf_counter(),False
    counts=dict(sequence_seed_conditions=0,interior_centers=0,output_rows=0,frame_rows=0,
        new_deployment_bidirectional_pairs=0,new_Farneback_calc_calls=0,
        cached_GT_metric_pairs_unique=0,new_GT_flow_pairs=0,new_CNN_calls=0,optimizer_updates=0)
    write(out/'config.json',dict(at=stamp(),process=identity(),source_freeze=frozen,
        source_audit=str(AUDIT.relative_to(ROOT)),arms=ARMS,common_indices='range(2,T-1)',
        threads=dict(BLAS=2,OpenCV=1),GPU_used=False,candidate_future_frames=1,
        candidate_added_wait='one registered source frame; physical interval not reverified',
        actual_new_CNN_calls=0,cache_timing_is_not_deployment_speed=True,
        GT_metric='Original cached reference field/mask; adjacent internal centers, border8',PSNR_ceiling_dB=120))
    try:
        write(out/'qualification.json',qualification())
        rows=[]
        for meta in source['scenes']:
            scene=meta['scene'];domain='SYN' if meta['domain']=='davis_train_pilot' else 'ND'
            input_path=owned(ROOT/meta['path']);assert digest(input_path)==meta['sha256']
            with np.load(input_path,allow_pickle=False) as bundle:
                target=bundle['gt'];fields=bundle['reference_flow'];masks=bundle['reference_mask']
            assert np.isfinite(target).all() and np.isfinite(fields).all()
            selected=[r for r in source['baseline_items'] if r['scene']==scene]
            assert len(selected)==2 and {r['seed'] for r in selected}=={0,1}
            tmax,height,width,channels=selected[0]['shape']
            assert target.shape==(tmax,height,width,3) and channels==3
            assert fields.shape==(tmax-1,height,width,2) and masks.shape==(tmax-1,height,width)
            # t2 -> t3 uses original flow index2; first selected frame has no Etemp.
            target=target[2:tmax-1]
            metric_fields=fields[2:tmax-2];metric_masks=masks[2:tmax-2]
            assert len(target)==tmax-3 and len(metric_fields)==len(target)-1
            counts['cached_GT_metric_pairs_unique']+=len(metric_fields)
            for item in selected:
                seed=item['seed'];shape=item['shape']
                assert shape==selected[0]['shape']
                baseline={arm:load_prediction(item[name],shape) for arm,name in [('F0','F0'),('F0_FB','FB'),('F0_DIS','DIS')]}
                raw=baseline['F0'];n=tmax-3
                outputs={arm:np.lib.format.open_memmap(assets/f'seed{seed}--{scene}--{arm}.npy',mode='w+',dtype=np.float32,shape=(n,height,width,3))
                         for arm in ('past3','center3','center_self')}
                for local,t in enumerate(range(2,tmax-1)):
                    current=raw[t];aligned={t:current};coverage={}
                    for neighbor in (t-1,t-2,t+1):
                        flow,accept=half_forward_backward(raw[neighbor],current)
                        aligned[neighbor],valid=align_history(raw[neighbor],current,flow,accept)
                        coverage[str(neighbor-t)]=float(valid.mean())
                        counts['new_deployment_bidirectional_pairs']+=1
                        counts['new_Farneback_calc_calls']+=2
                    routes={}
                    for mode in ('past3','center3','center_self'):
                        calls=[]
                        def provider(index):
                            assert index in indices(mode,t)
                            calls.append(index)
                            return aligned[index]
                        outputs[mode][local]=synthesize(provider,mode,t)
                        assert calls==indices(mode,t)
                        routes[mode]=calls
                    append(out/'routes-and-coverage.jsonl',dict(seed=seed,scene=scene,source_center=t,
                        requested_indices=routes,accepted_fractions=coverage))
                    counts['interior_centers']+=1
                for mode,array in outputs.items():
                    array.flush()
                    path=assets/f'seed{seed}--{scene}--{mode}.npy'
                    append(out/'prediction_manifest.jsonl',dict(seed=seed,domain=domain,scene=scene,arm=mode,
                        path=str(path.relative_to(ROOT)),sha256=digest(path),shape=list(array.shape),
                        first_source_frame=2,last_source_frame=tmax-2,future_frames=1 if mode=='center3' else 0,
                        checkpoint_sha256=item['checkpoint_sha256']))
                append(out/'source-checks.jsonl',dict(seed=seed,scene=scene,domain=domain,shape=shape,
                    input_cache_sha256=meta['sha256'],checkpoint_sha256=item['checkpoint_sha256'],
                    baseline_payload_sha256={k:item[k]['sha256'] for k in ('F0','FB','DIS')},
                    original_recurrence_from_first_frame=True,source_manifest_metadata_already_matched=True,
                    source_arrays_verified_owned_finite=True,GT_only_in_metrics=True))
                values={**{arm:v[2:tmax-1] for arm,v in baseline.items()},**outputs}
                for arm in ARMS:
                    summary,frames=evaluate(values[arm],target,metric_fields,metric_masks)
                    assert all(f['temporal_error'] is not None for f in frames[1:]),'No empty-domain omissions'
                    for f in frames:
                        f['source_frame']=f['frame']+2
                        f['PSNR_ceiling_applied']=f['psnr_rgb']>=120
                        f['psnr_rgb']=min(f['psnr_rgb'],120.)
                    summary['psnr_rgb']=float(np.mean([f['psnr_rgb'] for f in frames]))
                    assert all(np.isfinite(v) for v in summary.values())
                    row=dict(seed=seed,domain=domain,scene=scene,source_video=meta['source_video'],arm=arm,**summary)
                    rows.append(row);append(out/'per_sequence.jsonl',row)
                    for f in frames:
                        append(out/'per_frame.jsonl',dict(seed=seed,domain=domain,scene=scene,arm=arm,**f))
                    counts['output_rows']+=1;counts['frame_rows']+=len(frames)
                counts['sequence_seed_conditions']+=1
                print(dict(at=stamp(),seconds=time.perf_counter()-began,seed=seed,scene=scene,**counts),flush=True)
                del values,outputs,baseline,raw
        assert counts==dict(sequence_seed_conditions=20,interior_centers=836,output_rows=120,frame_rows=5016,
            new_deployment_bidirectional_pairs=2508,new_Farneback_calc_calls=5016,cached_GT_metric_pairs_unique=408,
            new_GT_flow_pairs=0,new_CNN_calls=0,optimizer_updates=0)
        summarize(rows,out)
        verify();complete=True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        write(out/'runtime.json',dict(at=stamp(),process=identity(),completed_normally=complete,
            seconds=time.perf_counter()-began,GPU_used=False,**counts))


if __name__=='__main__':
    main()
