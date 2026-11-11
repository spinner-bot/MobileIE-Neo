"""All four fixed native scenes, fixed frame8/seed0, with causal/current controls."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from continuation_r14 import create_model as original_control
from evaluate_aligned_raw_r11 import infer
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256
from prepare_cache_r5 import load_one
from prepare_native_r10 import INDEX
from train_exploration_r2 import new_model
from train_multiscale_r16 import create_model


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    out=Path(p.parse_args().out);out.mkdir(parents=True,exist_ok=False)
    cache=ROOT/'.cache/qualitative-r16';cache.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda');models={'Official':(new_model(device),'plain_high')};weights=[]
    folder=ROOT/'research/4.temporal-consistency/results/exploration-r14-spatial/native_crop-plain-seed0'
    meta=json.loads((folder/'checkpoint-step4000.json').read_text());assert sha256(ROOT/meta['path'])==meta['sha256']
    saved=torch.load(ROOT/meta['path'],map_location=device,weights_only=False);net=original_control('plain',device);net.load_state_dict(saved['model'],strict=True)
    models['R14 native plain']=(net.backbone.eval(),'plain_high');weights.append({'method':'R14 native plain',**meta})
    for mode,label,alias in [('spatial','R16 spatial','copy_high'),('raw_history','R16 raw','concat_high'),
            ('aligned_history','R16 aligned','concat_aligned_high')]:
        scope='aligned' if mode=='aligned_history' else 'spatial-raw'
        folder=ROOT/'research/4.temporal-consistency/results'/('exploration-r16-'+scope)/(mode+'-seed0')
        meta=json.loads((folder/'checkpoint-step4000.json').read_text());assert sha256(ROOT/meta['path'])==meta['sha256']
        saved=torch.load(ROOT/meta['path'],map_location=device,weights_only=False);assert saved['step']==4000
        net=create_model(mode,device);net.load_state_dict(saved['model'],strict=True);models[label]=(net.eval(),alias);weights.append({'method':label,**meta})
    records=[]
    for meta in json.loads(INDEX.read_text())['scenes']:
        if meta['domain']!='native_crop':continue
        data=load_one(meta);inputs=torch.from_numpy((data['low'][:9]/255.).transpose(0,3,1,2).copy()).to(device)
        panels={'Input':data['low'][8]/255.,'GT':data['gt'][8]/255.}
        for name,(net,alias) in models.items():
            for history in (['causal','current'] if name in ['R16 raw','R16 aligned'] else ['causal']):
                output=infer(net,inputs,data,alias,history,None)[8].clamp(0,1).permute(1,2,0).cpu().numpy()
                panels[name+(' / '+history if name in ['R16 raw','R16 aligned'] else '')]=output
        order=['Input','GT','Official','R14 native plain','R16 spatial','R16 raw / causal',
            'R16 raw / current','R16 aligned / causal','R16 aligned / current']
        fig,axes=plt.subplots(3,3,figsize=(15,9.2),layout='constrained')
        for ax,name in zip(axes.flat,order):
            array=np.ascontiguousarray(panels[name],dtype=np.float32);assert array.shape==(288,512,3) and np.isfinite(array).all()
            path=cache/(meta['scene']+'__'+name.replace('/','_').replace(' ','_')+'.npy');np.save(path,array)
            records.append({'scene':meta['scene'],'frame':8,'method':name,'display_array_path':str(path.relative_to(ROOT)),
                'sha256':sha256(path),'shape':list(array.shape),'scope':'display float array, not source of evaluation metrics'})
            ax.imshow(np.clip(array,0,1));ax.set_title(name,fontsize=10);ax.axis('off')
        fig.suptitle(meta['scene']+' / fixed native ROI, frame8, seed0 step4000; all methods fixed before temporal-dev evaluation',fontsize=12)
        fig.savefig(out/(meta['scene']+'.png'),dpi=150);plt.close(fig)
    assert len(records)==36
    save_json(out/'manifest.json',{'panels':records,'weights':weights,'native_index_sha256':sha256(INDEX),
        'protocol_sha256':sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r16-protocol.md'),
        'scope':'all four fixed exploration scenes; one fixed seed/frame; no ranking, no unseen-test claim'})


if __name__=='__main__':main()
