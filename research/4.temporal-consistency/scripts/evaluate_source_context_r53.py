"""Complete final-only source-context FastDVDnet reference, two information modes."""
import argparse
import json
import time
import traceback
from datetime import datetime

import cv2
import numpy as np
import torch

from dynamic_training_r34 import BASE, ROOT, digest
from exploration_r2_data import evaluate
from fastdvdnet_adapter_r12 import denoise, window_indices
from prepare_cache_r5 import load_one
from task_fastdvd_r47 import initial_model, SIGMA


def write(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value,allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',required=True,type=int,choices=[0,1])
    args = parser.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    train = BASE/f'results/exploration-r53-stage001-seed{args.seed}'
    runtime = json.loads((train/'runtime.json').read_text())
    marker = json.loads((train/'fastdvd-step8000.json').read_text())
    assert runtime['completed_normally'] and runtime['steps'] == marker['step'] == 8000
    assert marker['final_quality_eligible'] and digest(ROOT/marker['path']) == marker['sha256']
    model = initial_model(args.seed)
    checkpoint = torch.load(ROOT/marker['path'],map_location='cpu',weights_only=False)
    assert checkpoint['step'] == 8000 and checkpoint['seed'] == args.seed
    model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
    del checkpoint
    out = BASE/f'results/exploration-r53-stage001-eval-seed{args.seed}'
    out.mkdir(exist_ok=False)
    cache = ROOT/f'.cache/experiments/exploration-r53/stage001/seed{args.seed}'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    cache.mkdir(parents=True,exist_ok=False)
    indices = [BASE/f'results/{n}/index.json' for n in ['exploration-r30-cache','exploration-r10-cache']]
    metas = json.loads(indices[0].read_text())['scenes']
    metas += [{**r,'domain':'paired_real_ND_DEV','source_video':r['scene']}
              for r in json.loads(indices[1].read_text())['scenes'] if r['domain']=='native_crop']
    assert len(metas)==10
    names = ['evaluate_source_context_r53.py','source_context_r53.py','task_fastdvd_r47.py','fastdvdnet_adapter_r12.py',
             'exploration_r2_data.py','video_metrics.py','prepare_cache_r5.py']
    write(out/'config.json',{'at':datetime.now().astimezone().isoformat(),'seed':args.seed,
          'checkpoint':marker,'methods':['window','five_current'],'device':'cpu',
          'torch_threads':2,'opencv_threads':1,'sigma_interface':SIGMA,'extra_noise':False,
          'direct_low_to_GT':True,'base_MobileIE_calls':0,'new_neural_sequences_expected':20,
          'future_wait':{'window':2,'five_current':0},'save_arrays':'window only, in owned cache',
          'protocol_sha256':digest(BASE/'docs/exploration-r53-source-context-protocol.md'),
          'cache_indices':{str(p.relative_to(ROOT)):digest(p) for p in indices},
          'source_sha256':{n:digest(BASE/'scripts'/n) for n in names},
          'scope':'Source-context corrected reference; extra observations vs R47, reused DEV and public-pretraining overlap; not full convergence, novelty, phone or unseen final.'})
    start,count,frames_total,complete = time.perf_counter(),0,0,False
    try:
        for meta in metas:
            data = load_one(meta)
            inputs = torch.from_numpy((data['low']/255).transpose(0,3,1,2).copy())
            identity = {'domain':meta['domain'],'scene':meta['scene'],
                        'source_video':meta['source_video'],'train_seed':args.seed}
            for mode in ['window','five_current']:
                begin = time.perf_counter()
                result = denoise(model,inputs,SIGMA,mode)
                seconds = time.perf_counter()-begin
                assert torch.isfinite(result).all()
                output = result.permute(0,2,3,1).contiguous().numpy()*255
                metrics,frames = evaluate(output,data['gt'],data['reference_flow'],data['reference_mask'])
                append(out/'per_sequence.jsonl',{**identity,'method':mode,'parameters':2479096,
                       'future_wait_max':2 if mode=='window' else 0,**metrics})
                for frame in frames:
                    append(out/'per_frame.jsonl',{**identity,'method':mode,**frame})
                append(out/'inference_stage_seconds.jsonl',{**identity,'method':mode,'seconds':seconds,
                       'scope':'CPU2 denoiser stage, excludes IO/metrics; not independent production or phone cost.'})
                append(out/'context_indices.jsonl',{**identity,'method':mode,
                       'indices':window_indices(len(inputs),mode).tolist()})
                if mode=='window':
                    path = cache/f"{meta['scene']}--window.npy"
                    np.save(path,output,allow_pickle=False)
                    append(out/'prediction_manifest.jsonl',{**identity,'method':mode,
                          'path':str(path.relative_to(ROOT)),'sha256':digest(path),
                          'shape':list(output.shape),'checkpoint_sha256':marker['sha256']})
                count+=1;frames_total+=len(frames)
                print(json.dumps({**identity,'method':mode,'rows':count,'stage_seconds':seconds}),flush=True)
                del result,output
        assert count==20 and frames_total==896 and not torch.cuda.is_initialized()
        complete=True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        write(out/'runtime.json',{'completed_normally':complete,'rows':count,'per_frame_rows':frames_total,
              'neural_sequences':count,'cuda_initialized':torch.cuda.is_initialized(),
              'seconds':time.perf_counter()-start,'ended':datetime.now().astimezone().isoformat()})


if __name__=='__main__':
    main()
