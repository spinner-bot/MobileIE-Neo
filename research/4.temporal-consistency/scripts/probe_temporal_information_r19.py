"""Known static correspondence: learned history versus traditional input averaging."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from continuation_r14 import create_model as r14_model
from correlated_observations_r19 import sensor_pair,encode,pair_statistics
from exploration_r2_data import MANIFEST,ROOT,load_images
from explore_temporal_controls import save_json,sha256
from noise_moments_r6 import rgb_statistics
from train_exploration_r2 import new_model
from train_multiscale_r16 import create_model


def array_hash(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@torch.inference_mode()
def predict(net,kind,current,previous,path):
    if path=='preaverage_current':current=(current+previous)/2
    history=previous if path=='learned_history' else current;outputs=[]
    for offset in range(0,len(current),8):
        x,y=current[offset:offset+8],history[offset:offset+8]
        output=net(x) if kind in ['official','R14_plain'] else net.transform(x,y)
        outputs.append(output)
    return torch.cat(outputs)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    out=Path(p.parse_args().out);out.mkdir(parents=True,exist_ok=False)
    cache=ROOT/'.cache/experiments/exploration-r19';cache.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda');models=[('official',None,new_model(device))];weights=[]
    weights.append({'kind':'official','seed':None,'path':'MobileIE/pretrain/lolv1_best_slim.pkl',
        'sha256':sha256(ROOT/'MobileIE/pretrain/lolv1_best_slim.pkl'),'parameters':4075})
    for kind in ['R14_plain','spatial','raw_history','aligned_history']:
        for seed in [0,1]:
            if kind=='R14_plain':folder=ROOT/'research/4.temporal-consistency/results/exploration-r14-spatial'/f'native_crop-plain-seed{seed}'
            else:
                scope='aligned' if kind=='aligned_history' else 'spatial-raw'
                folder=ROOT/'research/4.temporal-consistency/results'/('exploration-r16-'+scope)/f'{kind}-seed{seed}'
            item=json.loads((folder/'checkpoint-step4000.json').read_text());path=ROOT/item['path'];assert sha256(path)==item['sha256']
            saved=torch.load(path,map_location=device,weights_only=False);assert saved['step']==4000
            net=r14_model('plain',device) if kind=='R14_plain' else create_model(kind,device)
            net.load_state_dict(saved['model'],strict=True)
            if kind=='R14_plain':net=net.backbone
            models.append((kind,seed,net.eval()));weights.append({'kind':kind,'seed':seed,**item,'parameters':sum(p.numel() for p in net.parameters())})
    assert len(models)==9
    base=ROOT/'research/4.temporal-consistency';save_json(out/'config.json',{'models':weights,'draws':64,'batch':8,
        'scenes':['kodim01','kodim06'],'crop_yx':[192,320],'patch_HW':[128,128],'photons':[20,80],'rho':[0,.9],
        'jpeg':[False,True],'output':'record raw clipping then clamp0..1, margin8, FP64 conditional moments',
        'manifest_sha256':sha256(MANIFEST),'protocol_sha256':sha256(base/'docs/exploration-r19-protocol.md'),
        'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in Path(__file__).parent.glob('*.py')},
        'torch_version':torch.__version__,'opencv_version':cv2.__version__,'cudnn_deterministic':True,
        'cudnn_tf32_allowed':torch.backends.cudnn.allow_tf32,'scope':'static synthetic information diagnostic, no video deployment/quality claim'})
    images=load_images('train');start=time.perf_counter();cache_rows=[];current_hashes={};output_replays={};replay_rows=[];count=0
    for scene_index,scene in enumerate(['kodim01','kodim06']):
        target=images[scene][192:320,320:448]/255.;clean=target.astype(np.float64)**2.2*.35
        target_tensor=torch.from_numpy(target.transpose(2,0,1).copy()).to(device)
        for photons in [20,80]:
            current_seed=1901900000+scene_index*100000+photons*100
            for rho_index,rho in enumerate([0,.9]):
                history_seed=1902900000+scene_index*100000+photons*100+rho_index
                a,b=sensor_pair(clean,photons,rho,current_seed,history_seed,64)
                for jpeg in [False,True]:
                    identity={'scene':scene,'photons':photons,'rho':rho,'jpeg':jpeg}
                    current,previous=encode(a,jpeg),encode(b,jpeg);key=(scene,photons,jpeg);current_sha=array_hash(current)
                    if key in current_hashes:assert current_hashes[key]==current_sha
                    current_hashes[key]=current_sha;statistics=pair_statistics(current,previous)
                    if statistics['variance_identity_abs']>1e-12:raise AssertionError('input population covariance identity failed')
                    label=f'{scene}_p{photons}_rho{rho}_jpeg{int(jpeg)}';path=cache/(label+'.npz')
                    np.savez_compressed(path,target=target,current=current,previous=previous)
                    cache_rows.append({**identity,'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'current_array_sha256':current_sha,
                        'previous_array_sha256':array_hash(previous),'current_seed':current_seed,'history_seed':history_seed,
                        'input_statistics':statistics,'preclip_statistics':pair_statistics(a,b)})
                    x=torch.from_numpy(current.transpose(0,3,1,2).copy()).to(device);y=torch.from_numpy(previous.transpose(0,3,1,2).copy()).to(device)
                    for kind,seed,net in models:
                        paths=['current','preaverage_current']+(['learned_history'] if kind in ['raw_history','aligned_history'] else [])
                        for inference_path in paths:
                            output=predict(net,kind,x,y,inference_path)
                            if not bool(torch.isfinite(output).all()):raise FloatingPointError('nonfinite model diagnostic')
                            clipping=float(((output<0)|(output>1)).float().mean());output=output.clamp(0,1)
                            stats=rgb_statistics(output,target_tensor)
                            if stats['closure_abs']>1e-12:raise AssertionError('RGB bias-variance identity failed')
                            prediction=output.detach().cpu().numpy();pixel_sha=array_hash(prediction)
                            if inference_path=='current':
                                rkey=(scene,photons,jpeg,kind,seed)
                                if rkey in output_replays:replay_rows.append({'condition':list(rkey),'pixel_sha_equal':output_replays[rkey]==pixel_sha})
                                output_replays[rkey]=pixel_sha
                            errors=(prediction.astype(np.float64)[:,:,8:-8,8:-8]-target_tensor.detach().cpu().numpy().astype(np.float64)[:,8:-8,8:-8])**2
                            row={**identity,'kind':kind,'seed':seed,'path':inference_path,**stats,'raw_clip_fraction':clipping,
                                'per_draw_mse_normalized':errors.mean((1,2,3)).tolist(),'output_array_sha256':pixel_sha}
                            with (out/'rgb.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                            count+=1;del output,prediction,errors
                    print(json.dumps({**identity,'rows':count,'input_correlation':statistics['empirical_correlation'],
                        'input_average_variance_ratio':statistics['average_variance_over_current']}),flush=True)
    assert len(cache_rows)==16 and count==352 and len(replay_rows)==72
    save_json(out/'cache_index.json',{'conditions':cache_rows,'scope':'64 independent pairs per static condition; currents identical across rho'})
    save_json(out/'current_replays.json',replay_rows)
    save_json(out/'runtime.json',{'wall_seconds':time.perf_counter()-start,'rows':count,'conditions':16,'draws_per_condition':64,
        'current_output_replays':len(replay_rows),'all_current_pixels_equal_across_rho':all(r['pixel_sha_equal'] for r in replay_rows),
        'scope':'known synthetic static correspondence, no training or new final-test data'})


if __name__=='__main__':main()
