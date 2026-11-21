"""Complete fixed-gain lag-one reference, from existing F0/recursive caches."""
import os
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[name]='2'
import time
import traceback
from f1_io import BASE,ROOT,ARMS,AUDIT,D1,append,cache,digest,identity,lines,owned,read,stamp,verify,write
import cv2
import numpy as np
from core import BidirectionalDIS,qualification,smooth
from aligned_raw_history_r11 import align_history
from aggregation_controls_r20 import blend
from exploration_r2_data import evaluate
from summarize import summarize


def load(entry,shape):
    path=owned(ROOT/entry['path']);assert digest(path)==entry['sha256']
    value=np.load(path,mmap_mode='r',allow_pickle=False)
    assert list(value.shape)==shape and value.dtype==np.float32 and np.isfinite(value).all()
    return value


def main():
    cv2.setNumThreads(1);os.nice(5)
    frozen=verify();source=read(AUDIT)
    assert read(D1/'runtime.json')['completed_normally']
    delayed={(r['seed'],r['scene']):r for r in lines(D1/'prediction_manifest.jsonl') if r['arm']=='center3'}
    assert len(delayed)==20
    out,assets=BASE/'results/eval001',cache()
    out.mkdir(parents=True,exist_ok=False);assets.mkdir(parents=True,exist_ok=False)
    began,complete=time.perf_counter(),False
    counts=dict(sequence_seed_conditions=0,interior_centers=0,output_rows=0,frame_rows=0,
        new_deployment_DIS_bidirectional_pairs=0,new_DIS_calc_calls=0,cached_GT_metric_pairs_unique=0,
        new_GT_flow_pairs=0,new_CNN_calls=0,optimizer_updates=0)
    write(out/'config.json',dict(at=stamp(),process=identity(),source_freeze=frozen,arms=ARMS,
        source_audit=str(AUDIT.relative_to(ROOT)),D1_original_completed_manifest=str((D1/'prediction_manifest.jsonl').relative_to(ROOT)),
        common_centers='range(2,T-1)',CPU_threads=2,OpenCV_threads=1,GPU_used=False,
        future_frames=dict(naive_future=1,innovation_lag1=1,D1_center3=1,F0=0,F0_FB=0,F0_DIS=0),
        time_unit='registered source frame; no measured milliseconds',RGB_PSNR_ceiling_dB=120,
        forward_state_closure_normalized_tolerance=1e-6,extra_closure_warp_is_evaluation_diagnostic_not_deployment))
    try:
        write(out/'qualification.json',qualification())
        rows=[];overall_closure=0.
        for meta in source['scenes']:
            scene=meta['scene'];domain='SYN' if meta['domain']=='davis_train_pilot' else 'ND'
            path=owned(ROOT/meta['path']);assert digest(path)==meta['sha256']
            with np.load(path,allow_pickle=False) as bundle:
                target=bundle['gt'];fields=bundle['reference_flow'];masks=bundle['reference_mask']
            assert np.isfinite(target).all() and np.isfinite(fields).all()
            items=[v for v in source['baseline_items'] if v['scene']==scene]
            assert len(items)==2 and {v['seed'] for v in items}=={0,1}
            tmax,height,width,_=items[0]['shape'];n=tmax-3
            assert target.shape==(tmax,height,width,3) and fields.shape==(tmax-1,height,width,2) and masks.shape==(tmax-1,height,width)
            target=target[2:tmax-1];metric_fields=fields[2:tmax-2];metric_masks=masks[2:tmax-2]
            counts['cached_GT_metric_pairs_unique']+=len(metric_fields)
            for item in items:
                seed=item['seed'];shape=item['shape'];assert shape==items[0]['shape']
                old={a:load(item[k],shape) for a,k in [('F0','F0'),('F0_FB','FB'),('F0_DIS','DIS')]}
                center=delayed[seed,scene]
                assert center['checkpoint_sha256']==item['checkpoint_sha256']
                center_values=load(center,[n,height,width,3])
                outputs={a:np.lib.format.open_memmap(assets/f'seed{seed}--{scene}--{a}.npy',mode='w+',dtype=np.float32,shape=(n,height,width,3))
                         for a in ('naive_future','innovation_lag1')}
                ranges={a:dict(min=float('inf'),max=-float('inf'),outside_elements=0) for a in outputs}
                motion=BidirectionalDIS();closure_max=0.
                for local,t in enumerate(range(2,tmax-1)):
                    raw,fused=old['F0'],old['F0_DIS']
                    back,mb,forward,mf=motion(raw[t],raw[t+1])
                    predicted,_=align_history(fused[t],raw[t+1],back,mb)
                    expected=blend(raw[t+1],predicted,.5)
                    error=float(np.max(np.abs(expected-fused[t+1])))/255
                    closure_max=max(closure_max,error);assert error<=1e-6
                    innovation,naive,accepted=smooth(fused[t],raw[t+1],fused[t+1],forward,mf,mb)
                    for arm,value in [('innovation_lag1',innovation),('naive_future',naive)]:
                        r=ranges[arm];r['min']=min(r['min'],float(value.min()));r['max']=max(r['max'],float(value.max()))
                        r['outside_elements']+=int(((value<0)|(value>255)).sum())
                        outputs[arm][local]=np.clip(value,0,255)
                        assert np.array_equal(outputs[arm][local][~accepted],fused[t][~accepted])
                    append(out/'geometry-and-closure.jsonl',dict(seed=seed,scene=scene,source_center=t,
                        filtered_source_index=t,future_source_index=t+1,forward_accepted_fraction=float(mf.mean()),
                        final_accepted_fraction=float(accepted.mean()),cached_forward_state_normalized_max_difference=error))
                    counts['interior_centers']+=1;counts['new_deployment_DIS_bidirectional_pairs']+=1;counts['new_DIS_calc_calls']+=2
                assert motion.pairs==n
                for arm,array in outputs.items():
                    array.flush();path=assets/f'seed{seed}--{scene}--{arm}.npy'
                    append(out/'prediction_manifest.jsonl',dict(seed=seed,domain=domain,scene=scene,arm=arm,
                        path=str(path.relative_to(ROOT)),sha256=digest(path),shape=list(array.shape),
                        first_source_frame=2,last_source_frame=tmax-2,future_frames=1,checkpoint_sha256=item['checkpoint_sha256']))
                    ranges[arm]['outside_fraction']=ranges[arm]['outside_elements']/(n*height*width*3)
                append(out/'raw-ranges.jsonl',dict(seed=seed,scene=scene,values=ranges,units='unclipped RGB255'))
                append(out/'source-checks.jsonl',dict(seed=seed,scene=scene,domain=domain,
                    input_cache_sha256=meta['sha256'],original_baseline_payloads={k:item[k]['sha256'] for k in ('F0','FB','DIS')},
                    D1_center3_sha256=center['sha256'],checkpoint_sha256=item['checkpoint_sha256'],
                    cached_forward_state_normalized_max_difference=closure_max,source_arrays_verified_owned_finite=True,
                    GT_only_in_metrics=True,corrected_lag_output_not_fed_back_into_original_filter=True))
                overall_closure=max(overall_closure,closure_max)
                values={**{a:v[2:tmax-1] for a,v in old.items()},'D1_center3':center_values,**outputs}
                for arm in ARMS:
                    summary,frames=evaluate(values[arm],target,metric_fields,metric_masks)
                    assert all(f['temporal_error'] is not None for f in frames[1:]),'No empty-domain omissions'
                    for f in frames:
                        f['source_frame']=f['frame']+2;f['PSNR_ceiling_applied']=f['psnr_rgb']>=120;f['psnr_rgb']=min(f['psnr_rgb'],120.)
                    summary['psnr_rgb']=float(np.mean([f['psnr_rgb'] for f in frames]));assert all(np.isfinite(v) for v in summary.values())
                    row=dict(seed=seed,domain=domain,scene=scene,source_video=meta['source_video'],arm=arm,**summary)
                    rows.append(row);append(out/'per_sequence.jsonl',row)
                    for f in frames:append(out/'per_frame.jsonl',dict(seed=seed,domain=domain,scene=scene,arm=arm,**f))
                    counts['output_rows']+=1;counts['frame_rows']+=len(frames)
                counts['sequence_seed_conditions']+=1
                print(dict(at=stamp(),seconds=time.perf_counter()-began,seed=seed,scene=scene,**counts),flush=True)
                del values,outputs,old,raw,fused,center_values
        assert counts==dict(sequence_seed_conditions=20,interior_centers=836,output_rows=120,frame_rows=5016,
            new_deployment_DIS_bidirectional_pairs=836,new_DIS_calc_calls=1672,cached_GT_metric_pairs_unique=408,
            new_GT_flow_pairs=0,new_CNN_calls=0,optimizer_updates=0)
        write(out/'closure-summary.json',dict(maximum_normalized_difference=overall_closure,tolerance=1e-6,centers=836,
            scope='Same-field original recursive cache provenance check only; not a training replay or new quality metric'))
        summarize(rows,out);verify();complete=True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        write(out/'runtime.json',dict(at=stamp(),process=identity(),completed_normally=complete,
            seconds=time.perf_counter()-began,GPU_used=False,**counts))


if __name__=='__main__':main()
