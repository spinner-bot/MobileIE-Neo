"""Fixed native crop, same-field input smoothing, and full-field resizing."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from PIL import Image
import torch

from exploration_r2_data import ROOT
from explore_state_r2 import forward_backward
from explore_temporal_controls import inference, save_json, sha256
from half_motion_r4 import half_forward_backward
from prepare_development_r9 import DEV_MANIFEST
from train_exploration_r2 import new_model

INDEX=ROOT/'research/4.temporal-consistency/results/exploration-r10-cache/index.json'


def variants(native):
    cropped=native[396:684,704:1216].copy()
    if cropped.shape!=(288,512,3):raise ValueError('native crop shape')
    smooth=cv2.resize(cv2.resize(cropped,(128,72),interpolation=cv2.INTER_AREA),(512,288),interpolation=cv2.INTER_LINEAR)
    return {'native_crop':cropped,'smooth_crop':smooth,
            'full_area':cv2.resize(native,(512,288),interpolation=cv2.INTER_AREA)}


def hh_proxy(x):
    gray=cv2.cvtColor(x,cv2.COLOR_RGB2GRAY)
    hh=(gray[::2,::2]+gray[1::2,1::2]-gray[::2,1::2]-gray[1::2,::2])/2
    return float(np.median(np.abs(hh-np.median(hh)))/.67448975)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    out=Path(p.parse_args().out);out.mkdir(parents=True,exist_ok=False)
    cache=ROOT/'.cache/experiments/exploration-r10';cache.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda');net=new_model(device);start=time.perf_counter();rows=[];statistics=[]
    manifest=json.loads(DEV_MANIFEST.read_text())
    for selected in manifest['selection']:
        scene=selected['scene'];arrays={v:{} for v in ['native_crop','smooth_crop','full_area']}
        names={}
        for side in ['input','gt']:
            files=sorted([f for f in manifest['files'] if f['scene']==scene and f['side']==side],key=lambda r:r['source_path'])
            assert len(files)==16
            names[side]=[Path(f['source_path']).name for f in files]
            for v in arrays:arrays[v][side]=[]
            for row in files:
                path=ROOT/row['path']
                if sha256(path)!=row['sha256']:raise ValueError('source changed')
                with Image.open(path) as image:native=np.asarray(image.convert('RGB')).astype(np.float32)
                assert native.shape==(1080,1920,3)
                prepared=variants(native)
                for v in arrays:arrays[v][side].append(prepared['native_crop' if side=='gt' and v=='smooth_crop' else v])
        assert names['input']==names['gt']
        reference={}
        for v,data in arrays.items():
            if v=='smooth_crop':
                assert np.array_equal(data['gt'],arrays['native_crop']['gt'])
                ref=reference['native_crop']
            else:ref=[forward_backward(data['gt'][t-1],data['gt'][t]) for t in range(1,16)]
            reference[v]=ref
            teacher=np.stack(inference(net,data['input'],device,1)[0])
            pairs=[half_forward_backward(teacher[t-1],teacher[t]) for t in range(1,16)]
            values={'low':np.stack(data['input']),'gt':np.stack(data['gt']),'teacher':teacher,
                'reference_flow':np.stack([r[0] for r in ref]),'reference_mask':np.stack([r[1] for r in ref]),
                'half_flow':np.stack([r[0] for r in pairs]),'half_mask':np.stack([r[1] for r in pairs])}
            path=cache/(scene+'_'+v+'.npz');np.savez_compressed(path,**values)
            rows.append({'domain':v,'scene':scene,'frames':16,'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'shape':list(values['low'].shape)})
            statistics.append({'scene':scene,'preparation':v,'low_rgb_mean':values['low'].mean((0,1,2)).tolist(),
                'target_rgb_mean':values['gt'].mean((0,1,2)).tolist(),'low_hh_mad_proxy_per_frame':[hh_proxy(x) for x in values['low']],
                'scope':'high-frequency texture plus noise proxy, not a noise ground truth'})
            print(json.dumps(rows[-1]),flush=True)
    assert len(rows)==12
    save_json(out/'index.json',{'scenes':rows,'manifest_sha256':sha256(DEV_MANIFEST),'crop_xywh':[704,396,512,288],
        'protocol_sha256':sha256(ROOT/'research/4.temporal-consistency/docs/exploration-r10-protocol.md'),
        'pretrained_sha256':sha256(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'),
        'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'same_crop_GT_and_reference_fields':'native_crop and smooth_crop exact same arrays',
        'scope':'full-area versus crop has different content/FOV; smoothing changes detail and noise; shared motion costs teacher'})
    save_json(out/'input_statistics.json',statistics)
    save_json(out/'runtime.json',{'wall_seconds':time.perf_counter()-start})


if __name__=='__main__':main()
