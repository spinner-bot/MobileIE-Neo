"""Ordinary single-frame optimizer controls on old exploration and new dev."""
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
from prepare_cache_r5 import CACHE_INDEX, load_one
from prepare_development_r9 import DEV_INDEX
from raw_pair_head_r8 import RawPairHead
from train_exploration_r2 import new_model
from train_optimizer_r9 import MODES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();variants=args.variants.split(',')
    if len(set(variants))!=len(variants) or any(v not in (*MODES,'plain','spatial_copy') for v in variants):p.error('unique registered modes required')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda');models=[];checkpoints=[]
    for variant in variants:
        for seed in [0,1]:
            scope='exploration-r5' if variant=='plain' else ('exploration-r8' if variant=='spatial_copy' else 'exploration-r9')
            path=ROOT/'.cache/checkpoints'/scope/f'{variant}-seed{seed}'/'step1000.pth'
            saved=torch.load(path,map_location=device,weights_only=False)
            if saved['step']!=1000:raise ValueError('fixed final checkpoint required')
            base=new_model(device)
            if variant=='spatial_copy':
                net=RawPairHead(base,variant).to(device);net.load_state_dict(saved['model'])
            else:
                base.load_state_dict({k.removeprefix('backbone.'):v for k,v in saved['model'].items()});net=base
            models.append((variant,seed,net.eval()))
            checkpoints.append({'variant':variant,'seed':seed,'path':str(path.relative_to(ROOT)),'sha256':sha256(path)})
    save_json(out/'config.json',{'variants':variants,'final_step':1000,'reset':'sequence boundaries only; all methods single-frame',
        'protocol_sha256':sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r9-protocol.md'),
        'cache_index_sha256':sha256(CACHE_INDEX),'dev_index_sha256':sha256(DEV_INDEX),
        'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'scope':'old exploration reused; four official TRAIN project-dev clips; no final test claim'})
    save_json(out/'checkpoints.json',checkpoints)
    metas=json.loads(CACHE_INDEX.read_text())['scenes']+json.loads(DEV_INDEX.read_text())['scenes']
    start=time.perf_counter();count=0
    for meta in metas:
        applicable=[m for m in models if meta['domain']=='dev' or m[0] in MODES]
        if not applicable:continue
        data=load_one(meta)
        inputs=torch.from_numpy((data['low']/255.).transpose(0,3,1,2).copy()).to(device)
        for variant,seed,net in applicable:
            if isinstance(net,RawPairHead):net.reset()
            with torch.inference_mode():
                prediction=torch.stack([net(x[None])[0] for x in inputs])
            if not bool(torch.isfinite(prediction).all()):raise FloatingPointError('nonfinite inference')
            raw=np.ascontiguousarray(prediction.clamp(0,1).permute(0,2,3,1).cpu().numpy()*255.)
            for fusion,output in [('raw',raw),('half01',fuse(raw,data['half_flow'],data['half_mask'],alpha=.1))]:
                identity={'domain':meta['domain'],'scene':meta['scene'],'variant':variant,'train_seed':seed,'fusion':fusion}
                metrics,frames=evaluate(output,data['gt'],data['reference_flow'],data['reference_mask'])
                with (out/'per_sequence.jsonl').open('a') as stream:stream.write(json.dumps({**identity,**metrics},allow_nan=False)+'\n')
                with (out/'per_frame.jsonl').open('a') as stream:
                    for frame in frames:stream.write(json.dumps({**identity,**frame},allow_nan=False)+'\n')
                count+=1
        print(json.dumps({'scene':meta['scene'],'rows':count}),flush=True)
    expected=sum(168 if v in MODES else 16 for v in variants)
    assert count==expected
    save_json(out/'runtime.json',{'rows':count,'wall_seconds':time.perf_counter()-start})


if __name__=='__main__':main()
