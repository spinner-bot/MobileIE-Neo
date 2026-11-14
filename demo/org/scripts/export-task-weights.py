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

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO/'MobileIE'))
from model.lle import MobileIELLENetS
from model.isp import MobileIEISPNetS


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);records=[]
    for task,cls,file,shape in [('underwater',MobileIELLENetS,'uieb_best_slim.pkl',(1,400,600,3)),('isp',MobileIEISPNetS,'zrr_best_slim.pkl',(1,200,300,4))]:
        source=REPO/'MobileIE/pretrain'/file;model=cls(12).cpu().eval()
        state=torch.load(source,map_location='cpu',weights_only=True);model.load_state_dict(state,strict=True)
        assert list(inspect.signature(model.forward).parameters)==['x']
        np.savez(a.output/(task+'-weights.npz'),**{k:v.detach().cpu().numpy() for k,v in state.items()})
        rng=np.random.default_rng(261114)
        cases={'black':np.zeros(shape,np.float32),'white':np.ones(shape,np.float32),'random':rng.random(shape,dtype=np.float32)}
        ramp=np.linspace(0,1,shape[2],dtype=np.float32)[None,None,:,None];cases['ramp']=np.broadcast_to(ramp,shape).copy()
        for name,x in cases.items():
            with torch.inference_mode():y=model(torch.from_numpy(x.transpose(0,3,1,2).copy())).numpy().transpose(0,2,3,1).copy()
            assert np.isfinite(y).all();np.save(a.output/f'{task}-{name}-input.npy',x);np.save(a.output/f'{task}-{name}-reference.npy',y)
        records.append({'task_id':task,'source':str(source.relative_to(REPO)),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'class':cls.__name__,'forward_signature':str(inspect.signature(model.forward)),'native_control_inputs':[],'input_shape':list(shape),'output_shape':list(y.shape),'reference_cases':list(cases),'parameters':sum(v.numel() for v in model.parameters())})
    (a.output/'official-weight-sources.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records,indent=2))


if __name__=='__main__':main()
