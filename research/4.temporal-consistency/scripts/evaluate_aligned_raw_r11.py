"""Strong-optimizer history and correspondence ablations with common postprocessing."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from aligned_raw_history_r11 import align_history
from exploration_r2_data import ROOT, detect_changes, evaluate
from explore_state_r2 import fuse
from explore_temporal_controls import save_json, sha256
from prepare_cache_r5 import CACHE_INDEX, load_one
from prepare_development_r9 import DEV_INDEX
from prepare_native_r10 import INDEX as NATIVE_INDEX
from raw_pair_head_r8 import RawPairHead
from train_aligned_raw_r11 import MODES
from train_exploration_r2 import new_model


@torch.inference_mode()
def infer(net, inputs, data, variant, history, resets):
    outputs=[]
    for t,current in enumerate(inputs):
        if variant=='plain_high':prediction=net(current[None])[0]
        else:
            previous=current
            if history=='causal' and t>0 and not (resets is not None and resets[t]):
                previous=inputs[t-1]
                if variant=='concat_aligned_high':
                    warped,_=align_history(data['low'][t-1],data['low'][t],data['half_flow'][t-1],data['half_mask'][t-1])
                    previous=torch.from_numpy((warped/255.).transpose(2,0,1).copy()).to(inputs.device)
            prediction=net.transform(current[None],previous[None])[0]
        outputs.append(prediction)
    return torch.stack(outputs)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variants',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();variants=args.variants.split(',')
    if len(set(variants))!=len(variants) or any(v not in (*MODES,'plain_high') for v in variants):p.error('unique registered variants required')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda');revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    sources={str(f.relative_to(ROOT)):sha256(f) for f in Path(__file__).parent.glob('*.py')}
    models=[];checkpoints=[];prior={}
    if 'plain_high' in variants:
        path=ROOT/'research/4.temporal-consistency/results/exploration-r9-eval-high/per_sequence.jsonl'
        for line in path.read_text().splitlines():
            r=json.loads(line);prior[(r['domain'],r['scene'],r['train_seed'],r['fusion'])]=r
        path=ROOT/'research/4.temporal-consistency/results/exploration-r10-eval/per_sequence.jsonl'
        for line in path.read_text().splitlines():
            r=json.loads(line)
            if r['variant']=='full_lr1e3' and r['preparation']=='native_crop':prior[('dev_native',r['scene'],r['train_seed'],r['fusion'])]=r
    for variant in variants:
        for seed in [0,1]:
            scope,name=('exploration-r9','full_lr1e3') if variant=='plain_high' else ('exploration-r11',variant)
            path=ROOT/'.cache/checkpoints'/scope/f'{name}-seed{seed}'/'step1000.pth'
            saved=torch.load(path,map_location=device,weights_only=False);assert saved['step']==1000
            base=new_model(device)
            if variant=='plain_high':
                base.load_state_dict({k.removeprefix('backbone.'):v for k,v in saved['model'].items()});net=base
            else:
                net=RawPairHead(base,MODES[variant]).to(device);net.load_state_dict(saved['model'])
            models.append((variant,seed,net.eval(),['causal','current'] if variant.startswith('concat') else ['causal']))
            checkpoints.append({'variant':variant,'seed':seed,'path':str(path.relative_to(ROOT)),'sha256':sha256(path)})
    save_json(out/'checkpoints.json',checkpoints)
    save_json(out/'config.json',{'variants':variants,'final_step':1000,'protocol_sha256':sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r11-protocol.md'),
        'cache_indices':{str(f.relative_to(ROOT)):sha256(f) for f in [CACHE_INDEX,DEV_INDEX,NATIVE_INDEX]},
        'git_revision_at_startup':revision,'source_sha256_at_startup':sources,
        'scope':'46 conditions but native/dev share four physical clips; input-only official teacher motion costs extra; current-history explicit'})
    metas=json.loads(CACHE_INDEX.read_text())['scenes']+json.loads(DEV_INDEX.read_text())['scenes']
    metas += [{**r,'domain':'dev_native'} for r in json.loads(NATIVE_INDEX.read_text())['scenes'] if r['domain']=='native_crop']
    assert len(metas)==46
    start=time.perf_counter();count=0;repeats=[];first=[];event_audit=[]
    for meta in metas:
        data=load_one(meta);events,_=detect_changes(data['low'])
        event_audit.append({'domain':meta['domain'],'scene':meta['scene'],'reset_frames':np.flatnonzero(events).tolist(),
                            'scope':'synthetic frame12 events known; real events are unannotated'})
        inputs=torch.from_numpy((data['low']/255.).transpose(0,3,1,2).copy()).to(device)
        for variant,seed,net,histories in models:
            first_output=None
            for history in histories:
                for reset in [False,True]:
                    prediction=infer(net,inputs,data,variant,history,events if reset else None)
                    if not bool(torch.isfinite(prediction).all()):raise FloatingPointError('nonfinite output')
                    raw=np.ascontiguousarray(prediction.clamp(0,1).permute(0,2,3,1).cpu().numpy()*255.)
                    if first_output is None:first_output=raw[0].copy()
                    delta=float(np.max(np.abs(raw[0]-first_output)))
                    if delta>1e-4:raise AssertionError('first-frame history/reset changed')
                    first.append({'scene':meta['scene'],'domain':meta['domain'],'variant':variant,'train_seed':seed,'history':history,'reset':reset,'max_abs_pixel':delta})
                    for fusion,output in [('raw',raw),('half01',fuse(raw,data['half_flow'],data['half_mask'],resets=events if reset else None,alpha=.1))]:
                        identity={'domain':meta['domain'],'scene':meta['scene'],'variant':variant,'train_seed':seed,'history':history,'input_reset':reset,'fusion':fusion}
                        metrics,frames=evaluate(output,data['gt'],data['reference_flow'],data['reference_mask'])
                        with (out/'per_sequence.jsonl').open('a') as stream:stream.write(json.dumps({**identity,**metrics},allow_nan=False)+'\n')
                        with (out/'per_frame.jsonl').open('a') as stream:
                            for frame in frames:stream.write(json.dumps({**identity,**frame},allow_nan=False)+'\n')
                        if variant=='plain_high' and not reset:
                            ref=prior[(meta['domain'],meta['scene'],seed,fusion)]
                            differences={k:abs(metrics[k]-ref[k]) for k in ['psnr_rgb','ssim_rgb','edge_mae','temporal_error']}
                            if max(differences.values())>1e-6:raise AssertionError('strong plain replay changed')
                            repeats.append({**identity,'max_abs_by_metric':differences})
                        count+=1
        print(json.dumps({'domain':meta['domain'],'scene':meta['scene'],'rows':count}),flush=True)
    expected=sum(736 if v.startswith('concat') else 368 for v in variants);assert count==expected
    save_json(out/'plain_replay.json',repeats);save_json(out/'first_frame_audit.json',first)
    save_json(out/'input_events.json',event_audit)
    save_json(out/'runtime.json',{'rows':count,'wall_seconds':time.perf_counter()-start})


if __name__=='__main__':main()
