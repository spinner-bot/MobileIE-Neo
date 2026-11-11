"""Shared teacher and motion cache for four unused train-only development clips."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from exploration_r2_data import ROOT, evaluate
from explore_state_r2 import forward_backward, fuse
from explore_temporal_controls import inference, save_json, sha256
from half_motion_r4 import half_forward_backward
from real_data_r4 import load_sequences
from train_exploration_r2 import new_model

DEV_MANIFEST=ROOT/'research/4.temporal-consistency/data/exploration-r9-dev/manifest.json'
DEV_INDEX=ROOT/'research/4.temporal-consistency/results/exploration-r9-dev-cache/index.json'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    out=Path(p.parse_args().out);out.mkdir(parents=True,exist_ok=False)
    cache=ROOT/'.cache/experiments/exploration-r9-dev';cache.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    net=new_model('cuda');start=time.perf_counter();rows=[];baseline=[]
    for scene,arrays in load_sequences('dev',(480,270),DEV_MANIFEST).items():
        raw,_=inference(net,arrays['input'],'cuda',1)
        data={'low':np.stack(arrays['input']),'gt':np.stack(arrays['gt']),'teacher':np.stack(raw)}
        for label,frames,func in [('reference',arrays['gt'],forward_backward),('half',raw,half_forward_backward)]:
            pairs=[func(frames[t-1],frames[t]) for t in range(1,len(frames))]
            data[label+'_flow']=np.stack([p[0] for p in pairs]);data[label+'_mask']=np.stack([p[1] for p in pairs])
        path=cache/(scene+'.npz');np.savez_compressed(path,**data)
        rows.append({'domain':'dev','scene':scene,'frames':16,'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'shape':list(data['low'].shape)})
        for fusion,output in [('raw',data['teacher']),('half01',fuse(data['teacher'],data['half_flow'],data['half_mask'],alpha=.1))]:
            metrics,frames=evaluate(output,data['gt'],data['reference_flow'],data['reference_mask'])
            baseline.append({'domain':'dev','scene':scene,'variant':'pretrained','fusion':fusion,**metrics})
        print(json.dumps(rows[-1]),flush=True)
    assert len(rows)==4
    save_json(out/'index.json',{'scenes':rows,'manifest_sha256':sha256(DEV_MANIFEST),
        'protocol_sha256':sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r9-protocol.md'),
        'pretrained_sha256':sha256(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'),
        'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'scope':'project dev from official TRAIN; shared teacher half-flow; GT-estimated reference is not true flow'})
    save_json(out/'baseline_per_sequence.json',baseline)
    save_json(out/'runtime.json',{'wall_seconds':time.perf_counter()-start})


if __name__=='__main__':main()
