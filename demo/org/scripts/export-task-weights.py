#!/usr/bin/env python3
"""Read official frozen weights and produce independent PyTorch CPU references."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image,ImageOps

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'MobileIE'))
from model.lle import MobileIELLENetS
from model.isp import MobileIEISPNetS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--samples',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);records=[]
    for task,cls,file,shape in [('underwater',MobileIELLENetS,'uieb_best_slim.pkl',(1,400,600,3)),('isp',MobileIEISPNetS,'zrr_best_slim.pkl',(1,224,224,4))]:
        source=REPO/'MobileIE/pretrain'/file;model=cls(12).cpu().eval()
        state=torch.load(source,map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True)
        assert list(inspect.signature(model.forward).parameters)==['x']
        np.savez(a.output/(task+'-weights.npz'),**{k:v.detach().cpu().numpy() for k,v in state.items()})
        rng=np.random.default_rng(261114)
        low,high=(64/4095,1023/4095) if task=='isp' else (0,1)
        cases={'black':np.full(shape,low,np.float32),'white':np.full(shape,high,np.float32),'random':rng.uniform(low,high,shape).astype(np.float32)}
        ramp=np.linspace(low,high,shape[2],dtype=np.float32)[None,None,:,None];cases['ramp']=np.broadcast_to(ramp,shape).copy()
        for i,name in enumerate(('0','1','10')):
            if task=='isp':
                raw=np.asarray(Image.open(a.samples/'zrr'/(name+'.png')),dtype=np.float32);assert raw.shape==(448,448)
                value=raw.reshape(224,2,224,2).transpose(0,2,1,3).reshape(1,224,224,4)/np.float32(4095)
            else:
                im=ImageOps.fit(Image.open(a.samples/f'uieb-{i}.png').convert('RGB'),(600,400),Image.Resampling.BILINEAR)
                value=np.asarray(im,dtype=np.float32)[None]/np.float32(255)
            cases[f'sample{i+1}']=value
        for name,x in cases.items():
            with torch.inference_mode():y=model(torch.from_numpy(x.transpose(0,3,1,2).copy())).numpy().transpose(0,2,3,1).copy()
            assert np.isfinite(y).all();np.save(a.output/f'{task}-{name}-input.npy',x);np.save(a.output/f'{task}-{name}-reference.npy',y)
        records.append({'task_id':task,'source':str(source.relative_to(REPO)),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'class':cls.__name__,'forward_signature':str(inspect.signature(model.forward)),'native_control_inputs':[],'input_shape':list(shape),'output_shape':list(y.shape),'reference_cases':list(cases),'parameters':sum(v.numel() for v in model.parameters()),'fixture_domain':'Huawei P20 10-bit sensor values 64..1023 divided by official 4095; plus three unchanged author RAW patches' if task=='isp' else 'RGB 0..1 plus three UIEB samples'})
    (a.output/'official-weight-sources.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))


if __name__=='__main__':main()
