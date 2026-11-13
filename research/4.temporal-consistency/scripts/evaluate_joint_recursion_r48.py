"""Common CPU evaluation of final jointly adapted and matched spatial controls."""
import argparse
import json
import time
import traceback
from datetime import datetime

import cv2
import numpy as np
import torch

from aggregation_controls_r20 import blend
from aligned_raw_history_r11 import align_history
from dynamic_training_r34 import BASE, ROOT, digest
from exploration_r2_data import evaluate
from half_motion_r4 import half_forward_backward
from joint_recursion_r48 import MODELS
from prepare_cache_r5 import load_one
from train_exploration_r2 import new_model


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def append(path,value):
    with path.open('a') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=[0,1],required=True)
    args=parser.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1);torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    train=BASE/f'results/exploration-r48-stage001-seed{args.seed}'
    runtime=json.loads((train/'runtime.json').read_text())
    assert runtime['completed_normally'] and runtime['model_updates']=={n:4000 for n in MODELS}
    assert runtime['total_new_updates']==8000
    nets,markers={},{}
    for name in MODELS:
        marker=json.loads((train/f'{name}-step4000.json').read_text())
        assert marker['final_fold_passed'] and marker['final_fold_delta']<=1e-4
        assert marker['new_step']==4000 and marker['inherited_updates']==8000
        assert digest(ROOT/marker['path'])==marker['sha256']
        saved=torch.load(ROOT/marker['path'],map_location='cpu',weights_only=False)
        assert saved['model']==name and saved['seed']==args.seed and saved['new_step']==4000
        assert saved['inherited_updates']==8000
        net=new_model(torch.device('cpu'));net.load_state_dict(saved['state_dict'],strict=True)
        assert sum(p.numel() for p in net.parameters())==4075
        nets[name],markers[name]=net.eval(),marker
    indices=[BASE/f'results/{n}/index.json' for n in ['exploration-r30-cache','exploration-r10-cache']]
    metas=json.loads(indices[0].read_text())['scenes']
    metas += [{**r,'domain':'paired_real_ND_DEV','source_video':r['scene']}
              for r in json.loads(indices[1].read_text())['scenes'] if r['domain']=='native_crop']
    assert len(metas)==10
    out=BASE/f'results/exploration-r48-stage001-eval-seed{args.seed}'
    cache=ROOT/f'.cache/experiments/exploration-r48/stage001/seed{args.seed}'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    out.mkdir(exist_ok=False);cache.mkdir(parents=True,exist_ok=False)
    for name in MODELS:(cache/name).mkdir()
    names=['evaluate_joint_recursion_r48.py','joint_recursion_r48.py','half_motion_r4.py',
           'aligned_raw_history_r11.py','aggregation_controls_r20.py','exploration_r2_data.py','video_metrics.py']
    write(out/'config.json',{'at':datetime.now().astimezone().isoformat(),'seed':args.seed,
          'device':'cpu','torch_threads':2,'opencv_threads':1,'checkpoints':markers,
          'methods':['raw','finite_self05','recursive_self05'],'neural_sequences_expected':20,
          'protocol_sha256':digest(BASE/'docs/exploration-r48-joint-recursion-protocol.md'),
          'cache_indices':{str(p.relative_to(ROOT)):digest(p) for p in indices},
          'source_sha256':{n:digest(BASE/'scripts'/n) for n in names},
          'scope':'Complete final12000 weights. Own-raw motion is model-dependent; no teacher or GT deployment flow.'})
    start,count,neural,frame_count,complete=time.perf_counter(),0,0,0,False
    try:
        for meta in metas:
            data=load_one(meta)
            inputs=torch.from_numpy((data['low']/255).transpose(0,3,1,2).copy())
            for name,net in nets.items():
                with torch.inference_mode():
                    value=torch.stack([net(image[None])[0] for image in inputs])
                    assert torch.isfinite(value).all()
                    raw=value.clamp(0,1).permute(0,2,3,1).contiguous().numpy()*255
                neural+=1
                identity={'domain':meta['domain'],'scene':meta['scene'],'source_video':meta['source_video'],
                          'train_seed':args.seed,'model':name,'parameters':4075}
                path=cache/name/(meta['scene']+'--raw.npy');np.save(path,raw,allow_pickle=False)
                append(out/'prediction_manifest.jsonl',{**identity,'path':str(path.relative_to(ROOT)),
                       'sha256':digest(path),'shape':list(raw.shape),'checkpoint_sha256':markers[name]['sha256']})
                finite,recursive,accepted=[raw[0]],[raw[0]],[]
                for t in range(1,len(raw)):
                    flow,mask=half_forward_backward(raw[t-1],raw[t]);accepted.append(float(mask.mean()))
                    finite.append(blend(raw[t],align_history(raw[t-1],raw[t],flow,mask)[0],.5))
                    recursive.append(blend(raw[t],align_history(recursive[-1],raw[t],flow,mask)[0],.5))
                append(out/'motion_acceptance.jsonl',{**identity,'pair_fractions':accepted,
                       'mean_accepted_fraction':float(np.mean(accepted))})
                for method,output in [('raw',raw),('finite_self05',np.stack(finite)),
                                      ('recursive_self05',np.stack(recursive))]:
                    assert np.array_equal(output[0],raw[0]) and np.isfinite(output).all()
                    metrics,frames=evaluate(output,data['gt'],data['reference_flow'],data['reference_mask'])
                    append(out/'per_sequence.jsonl',{**identity,'method':method,**metrics})
                    for frame in frames:append(out/'per_frame.jsonl',{**identity,'method':method,**frame})
                    count+=1;frame_count+=len(frames)
                print(json.dumps({**identity,'rows':count,'CNN_sequences':neural}),flush=True)
        assert count==60 and neural==20 and frame_count==2688 and not torch.cuda.is_initialized()
        complete=True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        write(out/'runtime.json',{'completed_normally':complete,'rows':count,'neural_sequences':neural,
              'per_frame_rows':frame_count,'cuda_initialized':torch.cuda.is_initialized(),
              'seconds':time.perf_counter()-start,'ended':datetime.now().astimezone().isoformat()})


if __name__=='__main__':
    main()
