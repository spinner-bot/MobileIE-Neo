"""Explicit fixed1000 checkpoint continuation, preserving every Adam tensor."""
import json
from pathlib import Path

import torch

from causal_fst_r5 import CausalFST
from exploration_r2_data import ROOT
from explore_temporal_controls import sha256
from raw_pair_head_r8 import RawPairHead
from train_exploration_r2 import new_model

MODES=('plain','copy','aligned_concat')


def create_model(variant,device):
    if variant not in MODES:raise ValueError('registered native/resized model required')
    base=new_model(device)
    return CausalFST(base,'plain').to(device) if variant=='plain' else RawPairHead(base,'spatial_copy' if variant=='copy' else 'temporal_concat').to(device)


def initial_runtime(variant,seed):
    folders={'plain':('exploration-r9-high','full_lr1e3'),'copy':('exploration-r11-raw','copy_high'),
             'aligned_concat':('exploration-r13-aligned','concat_aligned_context')}
    scope,name=folders[variant]
    return ROOT/'research/4.temporal-consistency/results'/scope/f'{name}-seed{seed}'/'runtime.json'


def restore(model,optimizer,runtime,device):
    runtime=Path(runtime);record=json.loads(runtime.read_text());path=ROOT/record['checkpoint']
    if sha256(path)!=record['checkpoint_sha256']:raise ValueError('continuation checkpoint SHA mismatch')
    saved=torch.load(path,map_location=device,weights_only=False)
    if saved['step']!=1000:raise ValueError('only explicitly registered final1000 may initialize continuation')
    model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
    if len(optimizer.param_groups)!=1:raise ValueError('one global Adam group required')
    group=optimizer.param_groups[0]
    if group['lr']!=1e-3 or group['weight_decay']!=0 or group['betas']!=(.9,.999) or group['eps']!=1e-8:
        raise ValueError('registered Adam settings changed')
    ids=saved['optimizer']['param_groups'][0]['params'];parameters=group['params']
    if len(ids)!=len(parameters):raise ValueError('optimizer parameter correspondence mismatch')
    checked=0;steps=[]
    for index,param in zip(ids,parameters):
        expected=saved['optimizer']['state'][index];actual=optimizer.state[param]
        if actual.keys()!=expected.keys():raise ValueError('optimizer state keys changed')
        for key,value in expected.items():
            if isinstance(value,torch.Tensor):
                restored=actual[key]
                if not torch.equal(restored,value.to(restored.device)) or not bool(torch.isfinite(restored).all()):
                    raise ValueError('Adam tensor changed: '+key)
                if key in ['exp_avg','exp_avg_sq'] and restored.shape!=param.shape:raise ValueError('Adam moment shape mismatch')
                checked+=1
            elif actual[key]!=value:raise ValueError('Adam state value changed: '+key)
        steps.append(float(actual['step']))
    if not steps or min(steps)!=1000 or max(steps)!=1000:raise ValueError('Adam steps were reset or incomplete')
    return {'checkpoint':str(path.relative_to(ROOT)),'checkpoint_sha256':record['checkpoint_sha256'],
            'runtime_sha256':sha256(runtime),'optimizer_tensors_verified':checked,
            'optimizer_step_range':[min(steps),max(steps)],'parameters':sum(p.numel() for p in parameters),
            'model_state_strict':True,'Adam_moments_exactly_preserved':True}
