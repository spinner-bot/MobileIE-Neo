"""Final-only same-frame self4 control; original R75 results remain immutable."""
import argparse
import json
import os
import time
import traceback

import cv2
import numpy as np
import torch

from causal_reference_io_r75 import ROOT,BASE,write,append,digest,stamp
from causal_reference_evaluation_r75 import metas,final_markers,load_final
from current_refinement_r76 import refine_current,PASSES
from rebotnet_reference import tile_layout
from prepare_cache_r5 import load_one
from exploration_r2_data import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=[0,1],required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise SystemExit('Require an initially empty CUDA mask for CPU inference.')
    torch.set_num_threads(2);cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    markers=final_markers()
    for seed in (0,1):
        run=BASE/f'results/exploration-r75-eval-seed{seed}/runtime.json'
        terminal=json.loads(run.read_text())
        assert terminal['completed_normally'] and terminal['quality_rows']==20
    gate=BASE/'process/exploration-r76/cpu-gate001/gate.json'
    assert json.loads(gate.read_text())['passed']
    model=load_final(args.seed,markers)
    scenes,indices=metas()
    expected_calls=PASSES*sum(r['shape'][0]*len(tile_layout(*r['shape'][1:3])) for r in scenes)
    assert PASSES==4 and expected_calls==9728
    out=BASE/f'results/exploration-r76-eval-seed{args.seed}'
    cache=ROOT/f'.cache/experiments/exploration-r76/eval001/seed{args.seed}'
    assert cache.resolve().is_relative_to('/data/Users/zrh/data/MobileIE-Neo')
    out.mkdir(exist_ok=False);cache.mkdir(parents=True,exist_ok=False)
    write(out/'config.json',dict(at=stamp(),seed=args.seed,checkpoint=markers[args.seed],cache_indices=indices,
        protocol_sha256=digest(BASE/'docs/exploration-r76-self-conditioning-control-protocol.md'),
        CPU_gate_sha256=digest(gate),passes=PASSES,expected_rows=10,expected_frames=448,
        expected_frame_level_CNN_calls=1792,expected_tile_CNN_calls=expected_calls,
        source_sha256={n:digest(BASE/'scripts'/n) for n in
            ['evaluate_current_refinement_r76.py','current_refinement_r76.py','rebotnet_reference.py',
             'causal_reference_evaluation_r75.py','exploration_r2_data.py','video_metrics.py']},
        scope='More-computation single-observation diagnostic, not a matched-compute spatial model or new method.'))
    started,complete,rows,frames,calls=time.perf_counter(),False,0,0,0
    try:
        for meta in scenes:
            data=load_one(meta)
            assert list(data['low'].shape)==meta['shape'] and data['low'].shape==data['gt'].shape
            identity=dict(domain=meta['domain'],scene=meta['scene'],source_video=meta['source_video'],
                seed=args.seed,train_seed=args.seed,model='ReBotNet_XS',history='self4',method='raw',
                label='ReBotNet_self4',parameters=890275,project_updates=64000,future_wait_max=0,
                distinct_observations_per_output=1,passes=PASSES,persistent_across_frame_state_bytes=0)
            path=cache/f'{meta["scene"]}--self4.npy';assert not path.exists()
            output=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=data['gt'].shape)
            began=time.perf_counter()
            with torch.inference_mode():
                for t in range(len(data['low'])):
                    current=torch.from_numpy(data['low'][t].transpose(2,0,1).copy()[None]).float()/255
                    value=refine_current(model,current)
                    assert torch.isfinite(value).all()
                    output[t]=value[0].clamp(0,1).permute(1,2,0).contiguous().numpy()*255
                    calls+=PASSES*len(tile_layout(*current.shape[-2:]))
            output.flush()
            append(out/'prediction_manifest.jsonl',dict(**identity,path=str(path.relative_to(ROOT)),
                shape=list(output.shape),sha256=digest(path),dtype='float32',checkpoint_sha256=markers[args.seed]['sha256']))
            append(out/'stage_cost.jsonl',dict(**identity,seconds=time.perf_counter()-began,
                scope='Unwarmed descriptive inference/cache cost; not a phone or fair warmed benchmark.'))
            metrics,details=evaluate(output,data['gt'],data['reference_flow'],data['reference_mask'])
            append(out/'per_sequence.jsonl',dict(**identity,**metrics))
            for row in details:
                append(out/'per_frame.jsonl',dict(**identity,**row))
            rows+=1;frames+=len(details)
            print(json.dumps(dict(**identity,completed_rows=rows)),flush=True)
            del data,output
        assert (rows,frames,calls)==(10,448,expected_calls) and not torch.cuda.is_initialized();complete=True
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        write(out/'runtime.json',dict(ended=stamp(),completed_normally=complete,quality_rows=rows,frame_rows=frames,
            frame_level_CNN_calls=frames*PASSES,tile_CNN_calls=calls,training_updates=0,
            CUDA_initialized=torch.cuda.is_initialized(),seconds=time.perf_counter()-started))


if __name__=='__main__':
    main()
