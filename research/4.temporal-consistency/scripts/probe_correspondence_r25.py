"""GT-estimated flow sensitivity at identical accepted pixels; diagnostic only."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import time

import cv2
import numpy as np
import torch

from aggregation_controls_r20 import blend
from aggregation_model_r21 import MODES
from aligned_raw_history_r11 import align_history
from continuation_r14 import create_model as plain_model
from evaluate_aggregation_r20 import metric_delta
from exploration_r2_data import ROOT,evaluate
from explore_temporal_controls import save_json,sha256
from matched_correspondence_r25 import REGIMES,array_sha,prepare
from prepare_cache_r5 import load_one
from prepare_native_r10 import INDEX
from train_aggregation_r21 import create_model

BASE=ROOT/'research/4.temporal-consistency'


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    out=Path(parser.parse_args().out);out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2);start=time.perf_counter();sources={};references={};frame_refs={}
    for tag in ['pre-zero','half-post']:
        folder=BASE/'results'/('exploration-r21-eval-'+tag)
        for name,target in [('per_sequence.jsonl',references),('per_frame.jsonl',frame_refs)]:
            path=folder/name;sources[str(path.relative_to(ROOT))]=sha256(path)
            with path.open() as stream:
                for line in stream:
                    r=json.loads(line)
                    if r['domain']=='dev_native' and r['step']==4000 and not r['input_reset'] and r['fusion']=='raw':
                        regime='current' if r['history']=='current' else 'teacher_default'
                        target[(r['scene'],r['variant'],r['train_seed'],regime,*((r['frame'],) if name=='per_frame.jsonl' else ()))]=r
    for seed in [0,1]:
        folder=BASE/'results'/f'exploration-r21-reference-post-seed{seed}-cpu'
        for name,target in [('per_sequence.jsonl',references),('per_frame.jsonl',frame_refs)]:
            path=folder/name;sources[str(path.relative_to(ROOT))]=sha256(path)
            with path.open() as stream:
                for line in stream:
                    r=json.loads(line)
                    if r['domain']=='dev_native' and not r['input_reset'] and r['method'] in ['raw','postwarp05']:
                        regime='current' if r['method']=='raw' else 'teacher_default'
                        target[(r['scene'],'R14_postwarp05',seed,regime,*((r['frame'],) if name=='per_frame.jsonl' else ()))]=r
    assert len(references)==80 and len(frame_refs)==1280
    models=[];weights=[]
    for variant in [*MODES,'R14_postwarp05']:
        for seed in [0,1]:
            folder=BASE/'results'/('exploration-r14-spatial' if variant=='R14_postwarp05' else f'exploration-r21-seed{seed}')
            folder=folder/(f'native_crop-plain-seed{seed}' if variant=='R14_postwarp05' else variant)
            item=json.loads((folder/'checkpoint-step4000.json').read_text());assert sha256(ROOT/item['path'])==item['sha256']
            saved=torch.load(ROOT/item['path'],map_location='cpu',weights_only=False);assert saved['step']==4000
            net=plain_model('plain',torch.device('cpu')) if variant=='R14_postwarp05' else create_model(variant,torch.device('cpu'))
            net.load_state_dict(saved['model'],strict=True);net=net.backbone if variant=='R14_postwarp05' else net
            models.append((variant,seed,net.eval()));weights.append({'variant':variant,'seed':seed,**item})
    save_json(out/'config.json',{'device':'cpu','batch':1,'Torch_threads':2,'OpenCV_threads':1,'weights':weights,'regimes':list(REGIMES),
        'reference_sources':sources,'native_index_sha256':sha256(INDEX),'protocol_sha256':sha256(BASE/'docs/exploration-r25-protocol.md'),
        'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'scope':'Shared mask and reference flow use GT-derived information; neither true motion nor deployable performance.'})
    rows=0;computations=0;aliases=0;network_sequences=0;backbone_frames=0;replays=[];first=[];correspondence=[];cache_sources=[]
    closure_max=0.;PSNR_delta_max=0.
    for meta in json.loads(INDEX.read_text())['scenes']:
        if meta['domain']!='native_crop':continue
        data=load_one(meta);histories,fields,details=prepare(data);cache_sources.append(meta)
        correspondence.extend({'scene':meta['scene'],**r} for r in details)
        inputs=torch.from_numpy((data['low']/255.).transpose(0,3,1,2).copy())
        for variant,seed,net in models:
            predictions={};metrics_cache={};first_reference=None
            if variant=='R14_postwarp05':
                raw=torch.stack([net(x[None])[0] for x in inputs]).clamp(0,1).permute(0,2,3,1).numpy()*255.
                raw=np.ascontiguousarray(raw);network_sequences+=1;backbone_frames+=len(inputs)
            for regime in REGIMES:
                if variant=='R14_postwarp05':
                    warped=raw.copy()
                    if regime!='current':
                        for t,(flow,mask) in enumerate(fields[regime],1):warped[t]=align_history(raw[t-1],raw[t],flow,mask)[0]
                    key=array_sha(warped)
                    if key not in predictions:predictions[key]=(np.ascontiguousarray(blend(raw,warped,.5)),regime)
                else:
                    key=array_sha(histories[regime])
                    if key not in predictions:
                        previous=torch.from_numpy((histories[regime]/255.).transpose(0,3,1,2).copy())
                        prediction=torch.stack([net.transform(inputs[t:t+1],previous[t:t+1])[0] for t in range(len(inputs))])
                        assert bool(torch.isfinite(prediction).all())
                        predictions[key]=(np.ascontiguousarray(prediction.clamp(0,1).permute(0,2,3,1).numpy()*255.),regime)
                        network_sequences+=1;backbone_frames+=len(inputs)*(2 if variant=='post_fixedhalf' else 1)
                output,prediction_origin=predictions[key]
                if first_reference is None:first_reference=output[0].copy()
                delta=float(np.max(np.abs(output[0]-first_reference)));assert delta<=1e-4
                identity={'domain':'dev_native','scene':meta['scene'],'variant':variant,'train_seed':seed,'regime':regime}
                label='|'.join(str(identity[k]) for k in identity);first.append({**identity,'max_abs_pixel':delta})
                output_key=array_sha(output)
                if output_key not in metrics_cache:
                    metrics,frames=evaluate(output,data['gt'],data['reference_flow'],data['reference_mask']);errors=[]
                    for frame in frames:
                        t=frame['frame'];error=(output[t].astype(np.float64)-data['gt'][t].astype(np.float64))/255.
                        mu=error.mean((0,1));mse=float(np.mean(error**2));bias=float(np.mean(mu**2));centered=float(np.mean((error-mu)**2))
                        closure=abs(mse-bias-centered);assert closure<=1e-12;closure_max=max(closure_max,closure)
                        pd=abs(float(-10*np.log10(mse))-frame['psnr_rgb']);assert pd<=1e-8;PSNR_delta_max=max(PSNR_delta_max,pd)
                        errors.append({'mse_normalized':mse,'global_RGB_bias_mse':bias,'centered_error_mse':centered,'global_RGB_mean_error_255':(mu*255).tolist()})
                    mean_errors={k:float(np.mean([r[k] for r in errors])) for k in ['mse_normalized','global_RGB_bias_mse','centered_error_mse']}
                    metrics_cache[output_key]=(metrics,frames,errors,mean_errors,label);alias=None;computations+=1
                else:metrics,frames,errors,mean_errors,alias=metrics_cache[output_key];aliases+=1
                annotation={'metric_alias_of':alias,'prediction_origin_regime':prediction_origin,'history_array_sha256':key,
                    'history_hash_domain':'warped_enhanced_output' if variant=='R14_postwarp05' else 'aligned_low_input','output_array_sha256':output_key}
                if regime in ['current','teacher_default']:
                    rk=(meta['scene'],variant,seed,regime);rd=metric_delta(metrics,references[rk])
                    for frame in frames:rd=max(rd,metric_delta(frame,frame_refs[(*rk,frame['frame'])]))
                    assert rd<=1e-10;replays.append({**identity,'sequence_and_frame_max_abs_metric':rd})
                with (out/'per_sequence.jsonl').open('a') as stream:stream.write(json.dumps({**identity,**annotation,**metrics,**mean_errors},allow_nan=False)+'\n')
                with (out/'per_frame.jsonl').open('a') as stream:
                    for frame,error in zip(frames,errors):stream.write(json.dumps({**identity,**annotation,**frame,**error},allow_nan=False)+'\n')
                rows+=1
            print(json.dumps({'scene':meta['scene'],'variant':variant,'seed':seed,'rows':rows,'metric_computations':computations,'aliases':aliases}),flush=True)
    assert rows==160 and computations+aliases==160 and len(replays)==80 and len(correspondence)==60 and not torch.cuda.is_initialized()
    for name,values in [('baseline_replays.json',replays),('first_frame_audit.json',first),('correspondence.json',correspondence),('cache_sources.json',cache_sources)]:save_json(out/name,values)
    save_json(out/'runtime.json',{'rows':160,'frames':2560,'new_diagnostic_configurations':80,'baseline_replayed_sequences':80,'baseline_replayed_frames':1280,
        'metric_computations':computations,'metric_aliases':aliases,'actual_model_sequence_passes':network_sequences,'actual_backbone_forward_frames':backbone_frames,
        'MSE_closure_max_abs':closure_max,'PSNR_identity_max_abs_dB':PSNR_delta_max,'device':'cpu','cuda_initialized':False,
        'wall_seconds':time.perf_counter()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'scope':'Shared experiment computations, not standalone latency; reference flow is GT-estimated, not motion truth.'})


if __name__=='__main__':main()
