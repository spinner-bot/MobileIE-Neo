"""Full-state, context, objective and clipping audit of the paired loss training."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256
from loss_controls_r26 import ARCHITECTURES,OBJECTIVES,compare_state
from train_loss_controls_r26 import old_folder

BASE=ROOT/'research/4.temporal-consistency'


def read(path):return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True);args=parser.parse_args()
    runs=[];checkpoints=[];contexts=[];source_hashes={};old_replays=[]
    for seed in [0,1]:
        root=BASE/'results'/f'exploration-r26-seed{seed}';runtime=json.loads((root/'runtime.json').read_text())
        assert runtime['steps']==4000 and runtime['new_updates']==24000 and runtime['models']==6
        integrity=json.loads((root/'source_integrity.json').read_text());assert integrity['verified_files']==384
        shared=read(root/'shared_context.jsonl');reference=read(old_folder('plain',seed)/'training.jsonl');assert len(shared)==len(reference)==4000
        aligned=read(BASE/'results/exploration-r14-aligned'/f'native_crop-aligned_concat-seed{seed}'/'training.jsonl');matches=0
        for i,(r,ref,old) in enumerate(zip(shared,reference,aligned)):
            assert r['step']==i+1
            for key in ['sample_identity','real_context','synthetic_context']:assert r[key]==ref[key]
            content={k:v for k,v in r.items() if k not in ['context_canonical_sha256','alignment_wall_seconds']}
            assert hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()==r['context_canonical_sha256']
            equal=[a['history_array_sha256']==b['history_array_sha256'] for a,b in zip(r['alignment'],old['alignment'])]
            assert len(equal)==4 and equal==r['old_R14_history_hash_matches'];matches+=sum(equal)
        assert matches==runtime['old_R14_history_hash_matches']
        contexts.append({'seed':seed,'steps':4000,'history_pairs':16000,'old_history_hash_matches':matches,'shared_sha256':sha256(root/'shared_context.jsonl')})
        for architecture in ARCHITECTURES:
            old=old_folder(architecture,seed);old_trace=read(old/'training.jsonl')
            for objective in OBJECTIVES:
                folder=root/(architecture+'-'+objective);cfg=json.loads((folder/'config.json').read_text());trace=read(folder/'training.jsonl')
                assert cfg['parameters']==4075 and cfg['loss_scale']==1. and len(trace)==4000
                assert cfg['old_Charb_trace_sha256']==sha256(old/'training.jsonl')
                run=json.loads((folder/'runtime.json').read_text());assert run['final_step']==4000 and run['new_updates']==4000
                assert json.loads((folder/'initial_current_limit.json').read_text())['current_limit_max_abs']<=2e-5
                if objective=='mse':
                    assert json.loads((folder/'initial_paired_output.json').read_text())['same_architecture_Charb_MSE_output_pixel_equal']
                    initial=json.loads((folder/'initial_objective_gradients.json').read_text());assert initial['MSE_norm_before_clip']>0 and initial['Charb_norm_before_clip']>0 and not initial['loss_scale_adjusted']
                for i,(r,context) in enumerate(zip(trace,shared)):
                    assert r['step']==i+1 and r['shared_context_sha256']==context['context_canonical_sha256']
                    fields=['task','charb_task','mse_task','gradient_norm_all','MSE_double','global_RGB_bias_mse','centered_error_mse','MSE_closure_abs']
                    assert all(np.isfinite(r[k]) for k in fields)
                    assert r['task']==r['charb_task' if objective=='charb' else 'mse_task'] and r['global_clip_triggered']==(r['gradient_norm_all']>1.)
                    assert abs(r['MSE_double']-r['global_RGB_bias_mse']-r['centered_error_mse'])<=1e-12
                    assert r['MSE_closure_abs']<=1e-12
                    if objective=='charb':assert r['old_Charb_task_delta']==r['task']-old_trace[i]['task']
                    else:assert r['old_Charb_task_delta'] is None
                revision=cfg['revision']
                for name,expected in cfg['source_sha256'].items():
                    key=(revision,name)
                    if key not in source_hashes:source_hashes[key]=hashlib.sha256(subprocess.check_output(['git','show',revision+':'+name],cwd=ROOT)).hexdigest()
                    assert source_hashes[key]==expected
                for step in [1000,2000,3000,4000]:
                    marker=json.loads((folder/f'checkpoint-step{step:04d}.json').read_text());path=ROOT/marker['path'];milestone=json.loads((folder/f'milestone-step{step:04d}.json').read_text())
                    assert sha256(path)==marker['sha256']==milestone['checkpoint_sha256']
                    saved=torch.load(path,map_location='cpu',weights_only=False);assert saved['step']==step and saved['config']==cfg
                    assert sum(v.numel() for v in saved['model'].values())==4075 and all(bool(torch.isfinite(v).all()) for v in saved['model'].values())
                    assert saved['optimizer']['state'] and all(int(s['step'])==step for s in saved['optimizer']['state'].values())
                    assert all(bool(torch.isfinite(v).all()) for s in saved['optimizer']['state'].values() for v in s.values() if torch.is_tensor(v))
                    assert hashlib.sha256((folder/'training.jsonl').read_bytes()[:milestone['model_trace_prefix_bytes']]).hexdigest()==milestone['model_trace_prefix_sha256']
                    prefix=milestone['shared_context_prefix'];assert hashlib.sha256((root/'shared_context.jsonl').read_bytes()[:prefix['bytes']]).hexdigest()==prefix['sha256']
                    if objective=='charb':
                        old_marker=json.loads((old/f'checkpoint-step{step:04d}.json').read_text());assert sha256(ROOT/old_marker['path'])==old_marker['sha256']
                        previous=torch.load(ROOT/old_marker['path'],map_location='cpu',weights_only=False)
                        model=compare_state(saved['model'],previous['model']);optimizer=compare_state(saved['optimizer'],previous['optimizer'])
                        assert model==milestone['old_Charb_replay']['model'] and optimizer==milestone['old_Charb_replay']['optimizer']
                        old_replays.append({'seed':seed,'architecture':architecture,'step':step,'model':model,'optimizer':optimizer})
                    checkpoints.append({'seed':seed,'architecture':architecture,'objective':objective,'step':step,'sha256':marker['sha256']})
                runs.append({'seed':seed,'architecture':architecture,'objective':objective,'steps':4000,
                    'steps_clipped':sum(r['global_clip_triggered'] for r in trace),'last100_task_mean':float(np.mean([r['task'] for r in trace[-100:]])),
                    'last100_MSE_double':float(np.mean([r['MSE_double'] for r in trace[-100:]])),
                    'last100_global_RGB_bias_mse':float(np.mean([r['global_RGB_bias_mse'] for r in trace[-100:]])),
                    'last100_centered_error_mse':float(np.mean([r['centered_error_mse'] for r in trace[-100:]])),
                    'old_Charb_task_max_abs_delta':max(abs(r['old_Charb_task_delta']) for r in trace) if objective=='charb' else None,
                    'trace_sha256':sha256(folder/'training.jsonl')})
    assert len(runs)==12 and len(checkpoints)==48 and len(old_replays)==24
    save_json(Path(args.out),{'new_updates':48000,'runs':runs,'shared_contexts':contexts,'checkpoints':checkpoints,'old_Charb_replays':old_replays,
        'all_old_Charb_models_and_Adam_exact':all(r['model']['exact'] and r['optimizer']['exact'] for r in old_replays),
        'training_source_hashes_verified':len(source_hashes),'source_mismatches':[],
        'scope':'Objective/setup comparison; losses have distinct gradient scale/clip coupling. Training loss does not establish video quality.'})


if __name__=='__main__':main()
