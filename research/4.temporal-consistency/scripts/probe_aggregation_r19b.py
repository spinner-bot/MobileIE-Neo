"""Same observations and weights: pre-average versus two classical output averages."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from continuation_r14 import create_model as r14_model
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256
from noise_moments_r6 import rgb_statistics
from probe_temporal_information_r19 import array_hash,predict
from summarize_causal_r5 import jsonl
from train_exploration_r2 import new_model
from train_multiscale_r16 import create_model


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    out=Path(p.parse_args().out);out.mkdir(parents=True,exist_ok=False)
    base=ROOT/'research/4.temporal-consistency';source=base/'results/exploration-r19'
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda');config=json.loads((source/'config.json').read_text());models=[]
    for record in config['models']:
        path=ROOT/record['path'];assert sha256(path)==record['sha256'];kind,seed=record['kind'],record['seed']
        if kind=='official':net=new_model(device)
        else:
            saved=torch.load(path,map_location=device,weights_only=False);assert saved['step']==4000
            net=r14_model('plain',device) if kind=='R14_plain' else create_model(kind,device)
            net.load_state_dict(saved['model'],strict=True)
            if kind=='R14_plain':net=net.backbone
        models.append((kind,seed,net.eval()))
    assert len(models)==9
    reference={tuple(r[k] for k in ['scene','photons','rho','jpeg','kind','seed','path']):r for r in jsonl(source/'rgb.jsonl')}
    save_json(out/'config.json',{'source_config_sha256':sha256(source/'config.json'),'source_cache_index_sha256':sha256(source/'cache_index.json'),
        'source_rgb_sha256':sha256(source/'rgb.jsonl'),'protocol_sha256':sha256(base/'docs/exploration-r19-protocol.md'),
        'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'scope':'no new observations/training; two classical post-averages, complete-network aggregation order not FST-only'})
    start=time.perf_counter();rows=0;replays=[];gaps=0
    for record in json.loads((source/'cache_index.json').read_text())['conditions']:
        path=ROOT/record['path'];assert sha256(path)==record['sha256']
        with np.load(path,allow_pickle=False) as data:
            x=torch.from_numpy(data['current'].transpose(0,3,1,2).copy()).to(device)
            y=torch.from_numpy(data['previous'].transpose(0,3,1,2).copy()).to(device)
            target=torch.from_numpy(data['target'].transpose(2,0,1).copy()).to(device)
        identity={k:record[k] for k in ['scene','photons','rho','jpeg']}
        for kind,seed,net in models:
            current=predict(net,kind,x,y,'current');previous=predict(net,kind,y,x,'current')
            pre=predict(net,kind,x,y,'preaverage_current')
            for label,output in [('current',current),('preaverage_current',pre)]:
                key=tuple(identity[k] for k in ['scene','photons','rho','jpeg'])+(kind,seed,label)
                equal=array_hash(output.clamp(0,1).cpu().numpy())==reference[key]['output_array_sha256']
                replays.append({**identity,'kind':kind,'seed':seed,'path':label,'pixel_sha_equal':equal})
                if not equal:raise AssertionError('same-batch source output replay changed')
            raw_average=(current+previous)/2;raw_gap=(pre-raw_average).double()[:,:,8:-8,8:-8]
            mean=raw_gap.mean(0);variance=raw_gap.var(0,unbiased=False).mean();mean_squared=mean.square().mean();mse=raw_gap.square().mean()
            gap={'raw_network_gap_mse':float(mse),'raw_network_gap_mean_squared':float(mean_squared),'raw_network_gap_variance':float(variance),
                'raw_network_gap_signed_mean_255':float(mean.mean()*255),'closure_abs':float((mse-mean_squared-variance).abs())}
            if gap['closure_abs']>1e-12:raise AssertionError('aggregation-gap moment identity failed')
            with (out/'aggregation_gap.jsonl').open('a') as stream:stream.write(json.dumps({**identity,'kind':kind,'seed':seed,**gap},allow_nan=False)+'\n')
            gaps+=1
            for label,prediction in [('postaverage_clipped',(current.clamp(0,1)+previous.clamp(0,1))/2),('postaverage_thenclip',raw_average)]:
                if not bool(torch.isfinite(prediction).all()):raise FloatingPointError('nonfinite output average')
                clipping=float(((prediction<0)|(prediction>1)).float().mean());output=prediction.clamp(0,1)
                stats=rgb_statistics(output,target)
                if stats['closure_abs']>1e-12:raise AssertionError('RGB moment identity failed')
                array=output.cpu().numpy();gt=target.cpu().numpy();error=(array.astype(np.float64)[:,:,8:-8,8:-8]-gt.astype(np.float64)[:,8:-8,8:-8])**2
                row={**identity,'kind':kind,'seed':seed,'path':label,**stats,'raw_clip_fraction':clipping,
                    'per_draw_mse_normalized':error.mean((1,2,3)).tolist(),'output_array_sha256':array_hash(array)}
                with (out/'rgb.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                rows+=1
        print(json.dumps({**identity,'new_rows':rows,'exact_replays':len(replays),'gaps':gaps}),flush=True)
    assert rows==288 and len(replays)==288 and gaps==144
    save_json(out/'replays.json',replays);save_json(out/'runtime.json',{'wall_seconds':time.perf_counter()-start,
        'new_rows':288,'source_rows_reused':352,'total_RGB_rows':640,'exact_source_pixel_replays':288,'aggregation_gap_rows':144,
        'scope':'same16 static conditions, no new videos or model selection'})


if __name__=='__main__':main()
