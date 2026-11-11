"""Fixed R21 endpoints on the existing R19 static correlated-observation cache."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import time

import cv2
import numpy as np
import torch

from aggregation_model_r21 import MODES
from continuation_r14 import create_model as plain_model
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256
from noise_moments_r6 import rgb_statistics
from probe_temporal_information_r19 import array_hash,predict
from train_aggregation_r21 import create_model


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True)
    out=Path(parser.parse_args().out);out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2);device=torch.device('cpu');base=ROOT/'research/4.temporal-consistency'
    models=[];weights=[];start=time.perf_counter()
    for kind in [*MODES,'R14_plain']:
        for seed in [0,1]:
            folder=base/'results'/('exploration-r14-spatial' if kind=='R14_plain' else f'exploration-r21-seed{seed}')
            folder=folder/(f'native_crop-plain-seed{seed}' if kind=='R14_plain' else kind)
            item=json.loads((folder/'checkpoint-step4000.json').read_text());path=ROOT/item['path'];assert sha256(path)==item['sha256']
            saved=torch.load(path,map_location='cpu',weights_only=False);assert saved['step']==4000
            net=plain_model('plain',device) if kind=='R14_plain' else create_model(kind,device)
            net.load_state_dict(saved['model'],strict=True)
            if kind=='R14_plain':net=net.backbone
            models.append((kind,seed,net.eval()));weights.append({'kind':kind,'seed':seed,**item})
    index=base/'results/exploration-r19/cache_index.json';conditions=json.loads(index.read_text())['conditions'];assert len(conditions)==16
    save_json(out/'config.json',{'device':'cpu','batch':8,'Torch_threads':2,'OpenCV_threads':1,'weights':weights,
        'cache_index_sha256':sha256(index),'protocol_sha256':sha256(base/'docs/exploration-r21b-protocol.md'),
        'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'scope':'Existing static16 conditions, no new training or dynamic-video claim.'})
    outputs={};replays=[];count=0;cache_audit=[]
    for record in conditions:
        path=ROOT/record['path'];assert sha256(path)==record['sha256'];cache_audit.append({'path':record['path'],'sha256':record['sha256']})
        with np.load(path,allow_pickle=False) as arrays:
            assert array_hash(arrays['current'])==record['current_array_sha256'] and array_hash(arrays['previous'])==record['previous_array_sha256']
            x=torch.from_numpy(arrays['current'].transpose(0,3,1,2).copy());y=torch.from_numpy(arrays['previous'].transpose(0,3,1,2).copy())
            gt=torch.from_numpy(arrays['target'].transpose(2,0,1).copy())
        identity={k:record[k] for k in ['scene','photons','rho','jpeg']}
        for kind,seed,net in models:
            current=predict(net,kind,x,y,'current')
            if kind=='R14_plain':
                previous=predict(net,kind,y,x,'current')
                paths=[('current',current),('preaverage_current',predict(net,kind,x,y,'preaverage_current')),
                    ('postaverage_clipped',(current.clamp(0,1)+previous.clamp(0,1))/2),('postaverage_thenclip',(current+previous)/2)]
            else:paths=[('current',current),('learned_history',predict(net,kind,x,y,'learned_history'))]
            for label,raw in paths:
                assert bool(torch.isfinite(raw).all());clipping=float(((raw<0)|(raw>1)).float().mean());prediction=raw.clamp(0,1)
                stats=rgb_statistics(prediction,gt);assert stats['closure_abs']<=1e-12
                pixel_sha=array_hash(prediction.numpy())
                if label=='current':
                    key=(record['scene'],record['photons'],record['jpeg'],kind,seed)
                    if key in outputs:
                        equal=outputs[key]==pixel_sha;replays.append({'condition':list(key),'pixel_sha_equal':equal});assert equal
                    outputs[key]=pixel_sha
                errors=(prediction.double()[:,:,8:-8,8:-8]-gt.double()[:,8:-8,8:-8]).square().mean((1,2,3))
                row={**identity,'kind':kind,'seed':seed,'path':label,**stats,'raw_clip_fraction':clipping,
                    'per_draw_mse_normalized':errors.tolist(),'output_array_sha256':pixel_sha}
                with (out/'rgb.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                count+=1
        print(json.dumps({**identity,'rows':count,'current_replays':len(replays)}),flush=True)
    assert count==384 and len(replays)==80 and not torch.cuda.is_initialized()
    save_json(out/'current_replays.json',replays);save_json(out/'cache_audit.json',cache_audit)
    save_json(out/'runtime.json',{'rows':384,'candidate_rows':256,'reference_rows':128,'conditions':16,'draws_per_condition':64,
        'current_output_replays':80,'all_current_pixels_equal_across_rho':True,'device':'cpu','cuda_initialized':False,
        'wall_seconds':time.perf_counter()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})


if __name__=='__main__':main()
