"""Published five-frame denoiser transfer after a fixed strong enhancement base."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from exploration_r2_data import ROOT, detect_changes, evaluate
from explore_temporal_controls import save_json, sha256
from fastdvdnet_adapter_r12 import MANIFEST, MODES, denoise, load_model, window_indices
from prepare_cache_r5 import CACHE_INDEX, load_one
from prepare_development_r9 import DEV_INDEX
from prepare_native_r10 import INDEX as NATIVE_INDEX
from train_exploration_r2 import new_model


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,choices=[0,1],required=True);p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    smoke=ROOT/'research/4.temporal-consistency/results/exploration-r12-smoke/smoke.json'
    check=json.loads(smoke.read_text())
    if not check['finite'] or check['psnr_gain']<1 or check['official_normalized_max_abs_difference']>1e-6:raise ValueError('preregistered smoke prerequisite failed')
    cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda');revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    sources={str(f.relative_to(ROOT)):sha256(f) for f in Path(__file__).parent.glob('*.py')}
    denoiser=load_model(device);base=new_model(device)
    checkpoint=ROOT/'.cache/checkpoints/exploration-r9'/f'full_lr1e3-seed{args.seed}'/'step1000.pth'
    saved=torch.load(checkpoint,map_location=device,weights_only=False);assert saved['step']==1000
    base.load_state_dict({k.removeprefix('backbone.'):v for k,v in saved['model'].items()});base.eval()
    save_json(out/'config.json',{'seed':args.seed,'base':'R9 full1e-3 final1000','checkpoint':str(checkpoint.relative_to(ROOT)),
        'checkpoint_sha256':sha256(checkpoint),'official_weight_manifest_sha256':sha256(MANIFEST),'smoke_sha256':sha256(smoke),
        'protocol_sha256':sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r12-protocol.md'),
        'sigmas':[5,15,30],'noise_added':False,'denoiser_parameters':sum(p.numel() for p in denoiser.parameters()),
        'cache_indices':{str(f.relative_to(ROOT)):sha256(f) for f in [CACHE_INDEX,DEV_INDEX,NATIVE_INDEX]},
        'git_revision_at_startup':revision,'source_sha256_at_startup':sources,
        'scope':'official clipped-AWGN pretrained denoiser transfer; five-frame window waits two future frames; not original DAVIS reproduction',
        'stage_timing_scope':'synchronized GPU-stage wall time includes Python/allocation, excludes loading/transfers/metrics/other CPU pipeline; shared server, not phone benchmark'})
    prior={}
    for line in (ROOT/'research/4.temporal-consistency/results/exploration-r9-eval-high/per_sequence.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['fusion']=='raw' and r['train_seed']==args.seed:prior[(r['domain'],r['scene'])]=r
    for line in (ROOT/'research/4.temporal-consistency/results/exploration-r10-eval/per_sequence.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['variant']=='full_lr1e3' and r['train_seed']==args.seed and r['preparation']=='native_crop' and r['fusion']=='raw':prior[('dev_native',r['scene'])]=r
    metas=json.loads(CACHE_INDEX.read_text())['scenes']+json.loads(DEV_INDEX.read_text())['scenes']
    metas += [{**r,'domain':'dev_native'} for r in json.loads(NATIVE_INDEX.read_text())['scenes'] if r['domain']=='native_crop']
    assert len(metas)==46
    start=time.perf_counter();count=0;repeats=[];context=[]
    for meta in metas:
        data=load_one(meta);events,_=detect_changes(data['low'])
        inputs=torch.from_numpy((data['low']/255.).transpose(0,3,1,2).copy()).to(device)
        torch.cuda.synchronize();begin=time.perf_counter()
        with torch.inference_mode():enhanced=torch.stack([base(x[None])[0] for x in inputs]).clamp(0,1)
        torch.cuda.synchronize();base_seconds=time.perf_counter()-begin
        for mode in MODES:
            indices=window_indices(len(enhanced),mode,events)
            context.append({'domain':meta['domain'],'scene':meta['scene'],'mode':mode,'indices':indices.tolist(),
                'future_wait_frames_max':int((indices-np.arange(len(indices))[:,None]).max()),'detected_events':np.flatnonzero(events).tolist()})
        for mode,sigma in [('raw',None)]+[(mode,sigma) for sigma in [5,15,30] for mode in MODES]:
            if mode=='raw':prediction=enhanced;seconds=base_seconds
            else:
                torch.cuda.synchronize();begin=time.perf_counter()
                prediction=denoise(denoiser,enhanced,sigma,mode,events)
                torch.cuda.synchronize();seconds=time.perf_counter()-begin
            if not bool(torch.isfinite(prediction).all()):raise FloatingPointError('nonfinite denoiser transfer output')
            output=np.ascontiguousarray(prediction.permute(0,2,3,1).cpu().numpy()*255.)
            identity={'domain':meta['domain'],'scene':meta['scene'],'train_seed':args.seed,'mode':mode,'sigma':sigma}
            metrics,frames=evaluate(output,data['gt'],data['reference_flow'],data['reference_mask'])
            with (out/'per_sequence.jsonl').open('a') as stream:stream.write(json.dumps({**identity,**metrics},allow_nan=False)+'\n')
            with (out/'per_frame.jsonl').open('a') as stream:
                for frame in frames:stream.write(json.dumps({**identity,**frame},allow_nan=False)+'\n')
            with (out/'stage_cost.jsonl').open('a') as stream:stream.write(json.dumps({**identity,'stage':'base' if mode=='raw' else 'denoiser_only',
                'shape':list(enhanced.shape),'seconds':seconds,'ms_per_frame':1000*seconds/len(enhanced)},allow_nan=False)+'\n')
            if mode=='raw':
                ref=prior[(meta['domain'],meta['scene'])];delta={k:abs(metrics[k]-ref[k]) for k in ['psnr_rgb','ssim_rgb','edge_mae','temporal_error']}
                if max(delta.values())>1e-6:raise AssertionError('strong base replay changed')
                repeats.append({**identity,'max_abs_by_metric':delta})
            count+=1
        print(json.dumps({'domain':meta['domain'],'scene':meta['scene'],'rows':count}),flush=True)
    assert count==460
    save_json(out/'base_replay.json',repeats);save_json(out/'context_index.json',context)
    save_json(out/'runtime.json',{'rows':count,'wall_seconds':time.perf_counter()-start})


if __name__=='__main__':main()
