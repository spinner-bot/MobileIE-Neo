"""Audit independent aggregation updates and shared deterministic contexts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from aggregation_model_r21 import MODES
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256


def rows(path):return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',required=True);args=parser.parse_args()
    base=ROOT/'research/4.temporal-consistency';runs=[];contexts=[];checkpoints=[]
    for seed in [0,1]:
        root=base/'results'/f'exploration-r21-seed{seed}'
        runtime=json.loads((root/'runtime.json').read_text());assert runtime['steps']==4000 and runtime['new_updates']==16000
        shared=rows(root/'shared_context.jsonl');assert len(shared)==4000
        reference=rows(base/'results/exploration-r14-spatial'/f'native_crop-plain-seed{seed}'/'training.jsonl')
        old=rows(base/'results/exploration-r14-aligned'/f'native_crop-aligned_concat-seed{seed}'/'training.jsonl')
        hashes=0
        for i,(r,ref,prior) in enumerate(zip(shared,reference,old)):
            assert r['step']==i+1
            for key in ['sample_identity','real_context','synthetic_context']:assert r[key]==ref[key]
            payload={k:v for k,v in r.items() if k not in ['context_canonical_sha256','alignment_wall_seconds']}
            assert hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==r['context_canonical_sha256']
            matches=[a['history_array_sha256']==b['history_array_sha256'] for a,b in zip(r['alignment'],prior['alignment'])]
            assert len(matches)==4 and matches==r['old_R14_history_hash_matches'];hashes+=sum(matches)
        assert hashes==runtime['old_R14_history_hash_matches']
        contexts.append({'seed':seed,'steps':4000,'history_pairs':16000,'old_history_hash_matches':hashes,
                         'sha256':sha256(root/'shared_context.jsonl'),'all_current_task_contexts_exact':True})
        for mode in MODES:
            folder=root/mode;trace=rows(folder/'training.jsonl');assert len(trace)==4000
            cfg=json.loads((folder/'config.json').read_text());init=json.loads((folder/'initial_current_limit.json').read_text())
            assert init['current_limit_max_abs']<=2e-5 and init.get('fixedhalf_initial_actual_history_max_abs',0)<=2e-5
            assert cfg['parameters']==(4076 if mode.startswith('pre_learn') else 4075)
            expected=cfg['initial_alpha'];projections=0
            for i,(r,context) in enumerate(zip(trace,shared)):
                assert r['step']==i+1 and r['shared_context_sha256']==context['context_canonical_sha256']
                assert r['alpha_before']==expected
                assert all(np.isfinite(r[k]) for k in ['task','gradient_norm_all','alpha_before','alpha_after_Adam_before_projection','alpha_after_projection'])
                unprojected=r['alpha_after_Adam_before_projection'];expected=float(np.clip(unprojected,0,1))
                assert r['alpha_after_projection']==expected;projections+=int(unprojected!=expected)
                if mode.startswith('pre_learn'):assert np.isfinite(r['alpha_gradient_before_clip'])
                else:assert expected==.5 and r['alpha_gradient_before_clip'] is None
            for step in [1000,2000,3000,4000]:
                marker=json.loads((folder/f'checkpoint-step{step:04d}.json').read_text())
                milestone=json.loads((folder/f'milestone-step{step:04d}.json').read_text());path=ROOT/marker['path']
                assert sha256(path)==marker['sha256']==milestone['checkpoint_sha256']
                assert hashlib.sha256((folder/'training.jsonl').read_bytes()[:milestone['model_trace_prefix_bytes']]).hexdigest()==milestone['model_trace_prefix_sha256']
                prefix=milestone['shared_context_prefix'];assert hashlib.sha256((root/'shared_context.jsonl').read_bytes()[:prefix['bytes']]).hexdigest()==prefix['sha256']
                saved=torch.load(path,map_location='cpu',weights_only=False)
                assert saved['step']==step and saved['config']==cfg
                assert all(bool(torch.isfinite(t).all()) for t in saved['model'].values())
                state=saved['optimizer']['state'];assert state and all(int(s['step'])==step for s in state.values())
                assert all(bool(torch.isfinite(t).all()) for s in state.values() for t in s.values() if torch.is_tensor(t))
                alpha=float(saved['model']['alpha']) if 'alpha' in saved['model'] else .5
                assert alpha==marker['alpha']==trace[step-1]['alpha_after_projection']
                checkpoints.append({'seed':seed,'mode':mode,'step':step,'sha256':marker['sha256'],'alpha':alpha,'Adam_states':len(state)})
            runs.append({'seed':seed,'mode':mode,'steps':4000,'parameters':cfg['parameters'],'final_alpha':expected,
                         'projected_updates':projections,'trace_sha256':sha256(folder/'training.jsonl'),
                         'initial_current_limit_max_abs':init['current_limit_max_abs'],
                         'last100_task_mean':float(np.mean([r['task'] for r in trace[-100:]]))})
    assert len(runs)==8 and len(checkpoints)==32
    save_json(Path(args.out),{'new_updates':32000,'runs':runs,'shared_contexts':contexts,'checkpoints':checkpoints,
        'all_checkpoint_prefixes_verified':True,'all_checkpoint_Adam_steps_exact':True,'all_model_context_references_exact':True,
        'scope':'No quality inference from loss or alpha; all fixed evaluation controls required.'})


if __name__=='__main__':main()
