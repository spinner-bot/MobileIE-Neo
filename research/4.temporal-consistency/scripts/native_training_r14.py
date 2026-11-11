"""Verified uint8 training sources and matched native/resized current/history crops."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from exploration_r2_data import ROOT
from explore_temporal_controls import sha256
from train_mixture_r4b import EXPANDED


def load_training(manifest=EXPANDED):
    saved=json.loads(Path(manifest).read_text());result=[];verified=0
    for selected in saved['selection']:
        if selected['split']!='train':continue
        scene=selected['scene'];entry={'scene':scene,'source_files':{}}
        names={}
        for side in ['input','gt']:
            files=sorted([r for r in saved['files'] if r['scene']==scene and r['side']==side],key=lambda r:r['source_path'])
            if len(files)!=16:raise ValueError('exact16 frame training protocol required')
            native=[];resized=[];names[side]=[]
            for row in files:
                path=ROOT/row['path']
                if sha256(path)!=row['sha256']:raise ValueError('training source changed: '+str(path))
                with Image.open(path) as image:array=np.asarray(image.convert('RGB'))
                if array.shape!=(1080,1920,3) or array.dtype!=np.uint8:raise ValueError('native uint8 RGB required')
                native.append(array);resized.append(cv2.resize(array,(320,180),interpolation=cv2.INTER_AREA))
                names[side].append(Path(row['source_path']).name);verified+=1
            entry['native_'+side]=np.stack(native);entry['resized_'+side]=np.stack(resized)
            entry['source_files'][side]=[{'path':r['path'],'sha256':r['sha256']} for r in files]
        if names['input']!=names['gt']:raise ValueError('frame pairing changed')
        result.append(entry)
    if len(result)!=12 or verified!=384:raise ValueError('registered twelve training sources required')
    return result,{'manifest_sha256':sha256(Path(manifest)),'verified_files':verified,'scenes':[r['scene'] for r in result],
                   'native_bytes':sum(r['native_input'].nbytes+r['native_gt'].nbytes for r in result)}


def pair_identity(seed,step,scenes):
    rng=np.random.default_rng(seed*1000000+step)
    return int(rng.integers(scenes)),int(rng.integers(1,16))


def tensor(frames):
    return torch.from_numpy(np.stack(frames).transpose(0,3,1,2).copy().astype(np.float32)/255.)


def real_pair(data,seed,step,preparation):
    if preparation not in ['native_crop','resized']:raise ValueError('registered training preparation required')
    index,t=pair_identity(seed,step,len(data));row=data[index]
    if preparation=='native_crop':
        rng=np.random.default_rng(1401400000+seed*1000000+step)
        x,y=int(rng.integers(0,1601)),int(rng.integers(0,901))
        key='native_';crop=[x,y,320,180]
    else:key='resized_';x=y=0;crop=None
    def frame(side,i):return row[key+side][i,y:y+180,x:x+320]
    low=tensor([frame('input',t-1),frame('input',t)])
    target=tensor([frame('gt',t-1),frame('gt',t)])
    previous=tensor([frame('input',max(0,t-2)),frame('input',t-1)])
    if low.shape!=(2,3,180,320) or previous.shape!=low.shape:raise ValueError('task/history crop changed shape')
    metadata={'source':'real','scene':row['scene'],'frame_pair':t}
    context={'preparation':preparation,'crop_xywh':crop,'current_indices':[t-1,t],
        'previous_indices':[max(0,t-2),t-1],'self_fallback':t==1,
        'current_input_sources':[row['source_files']['input'][i] for i in [t-1,t]],
        'current_GT_sources':[row['source_files']['gt'][i] for i in [t-1,t]]}
    return low,target,previous,metadata,context
