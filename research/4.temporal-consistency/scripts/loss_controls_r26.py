"""Fixed spatial/pre/post mappings and full-state replay comparisons for loss controls."""
import torch

from continuation_r14 import create_model as plain_model
from train_aggregation_r21 import create_model as aggregation_model

ARCHITECTURES=('plain','pre_fixedhalf','post_fixedhalf')
OBJECTIVES=('charb','mse')


def create_model(architecture,device):
    if architecture not in ARCHITECTURES:raise ValueError('registered architecture required')
    return plain_model('plain',device) if architecture=='plain' else aggregation_model(architecture,device)


def forward_training(net,architecture,low,previous):
    return net.forward_pair(low) if architecture=='plain' else net.transform(low.flatten(0,1),previous.flatten(0,1)).reshape_as(low)


def compare_state(actual,expected):
    tensors=0;exact=True;maximum=0.;structure=True
    def visit(a,b):
        nonlocal tensors,exact,maximum,structure
        if torch.is_tensor(a) and torch.is_tensor(b):
            tensors+=1
            if a.shape!=b.shape or a.dtype!=b.dtype:structure=False;exact=False;return
            a,b=a.detach().cpu(),b.detach().cpu();exact=exact and torch.equal(a,b)
            if a.numel():maximum=max(maximum,float((a.double()-b.double()).abs().max()))
        elif isinstance(a,dict) and isinstance(b,dict):
            if a.keys()!=b.keys():structure=False;exact=False;return
            for k in a:visit(a[k],b[k])
        elif isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
            if len(a)!=len(b):structure=False;exact=False;return
            for x,y in zip(a,b):visit(x,y)
        elif a!=b:structure=False;exact=False
    visit(actual,expected)
    return {'tensors':tensors,'exact':exact,'max_abs':maximum,'structure_and_nontensor_fields_equal':structure}
