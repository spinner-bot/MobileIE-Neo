"""Frozen all-candidate native-scale evaluation and preregistered frame0 previews."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from exploration_r2_data import ROOT, evaluate
from explore_state_r2 import fuse
from explore_temporal_controls import save_json, sha256
from prepare_cache_r5 import load_one
from prepare_native_r10 import INDEX
from raw_pair_head_r8 import RawPairHead
from train_exploration_r2 import new_model
from train_optimizer_r9 import MODES


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    out=Path(p.parse_args().out);out.mkdir(parents=True,exist_ok=False)
    preview=ROOT/'.cache/experiments/exploration-r10-preview';preview.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda');models=[('pretrained',None,new_model(device))];checkpoints=[]
    for variant in ['plain','spatial_copy',*MODES]:
        for seed in [0,1]:
            scope='exploration-r5' if variant=='plain' else ('exploration-r8' if variant=='spatial_copy' else 'exploration-r9')
            path=ROOT/'.cache/checkpoints'/scope/f'{variant}-seed{seed}'/'step1000.pth'
            saved=torch.load(path,map_location=device,weights_only=False);assert saved['step']==1000
            net=new_model(device)
            if variant=='spatial_copy':
                net=RawPairHead(net,variant).to(device);net.load_state_dict(saved['model'])
            else:net.load_state_dict({k.removeprefix('backbone.'):v for k,v in saved['model'].items()})
            models.append((variant,seed,net.eval()))
            checkpoints.append({'variant':variant,'seed':seed,'path':str(path.relative_to(ROOT)),'sha256':sha256(path)})
    save_json(out/'checkpoints.json',checkpoints)
    save_json(out/'config.json',{'protocol_sha256':sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r10-protocol.md'),
        'cache_index_sha256':sha256(INDEX),'models':11,'preview':'frame0 all clips/preparations/all models; selected before new data pixels',
        'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'scope':'same GT for native/smoothed crop; full-area has different field of view'})
    start=time.perf_counter();count=0;previews=[];teacher_repeats=[]
    for meta in json.loads(INDEX.read_text())['scenes']:
        data=load_one(meta);images={'low':data['low'][0],'gt':data['gt'][0]}
        inputs=torch.from_numpy((data['low']/255.).transpose(0,3,1,2).copy()).to(device)
        for variant,seed,net in models:
            if isinstance(net,RawPairHead):net.reset()
            with torch.inference_mode():prediction=torch.stack([net(x[None])[0] for x in inputs])
            if not bool(torch.isfinite(prediction).all()):raise FloatingPointError('nonfinite inference')
            raw=np.ascontiguousarray(prediction.clamp(0,1).permute(0,2,3,1).cpu().numpy()*255.)
            images[variant if seed is None else variant+'_seed'+str(seed)]=raw[0]
            if variant=='pretrained':
                delta=float(np.max(np.abs(raw-data['teacher'])))
                if delta>1e-4:raise AssertionError('teacher replay changed')
                teacher_repeats.append({'scene':meta['scene'],'preparation':meta['domain'],'max_abs_pixel':delta})
            for fusion,output in [('raw',raw),('half01',fuse(raw,data['half_flow'],data['half_mask'],alpha=.1))]:
                identity={'scene':meta['scene'],'preparation':meta['domain'],'variant':variant,'train_seed':seed,'fusion':fusion}
                metrics,frames=evaluate(output,data['gt'],data['reference_flow'],data['reference_mask'])
                with (out/'per_sequence.jsonl').open('a') as stream:stream.write(json.dumps({**identity,**metrics},allow_nan=False)+'\n')
                with (out/'per_frame.jsonl').open('a') as stream:
                    for frame in frames:stream.write(json.dumps({**identity,**frame},allow_nan=False)+'\n')
                count+=1
        path=preview/(meta['scene']+'_'+meta['domain']+'.npz');np.savez_compressed(path,**images)
        previews.append({'scene':meta['scene'],'preparation':meta['domain'],'frame':0,'path':str(path.relative_to(ROOT)),'sha256':sha256(path)})
        print(json.dumps({'scene':meta['scene'],'preparation':meta['domain'],'rows':count}),flush=True)
    assert count==264
    save_json(out/'preview_index.json',previews)
    save_json(out/'teacher_replay.json',teacher_repeats)
    save_json(out/'runtime.json',{'rows':count,'wall_seconds':time.perf_counter()-start})


if __name__=='__main__':main()
