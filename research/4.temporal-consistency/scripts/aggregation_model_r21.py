"""Classical aggregation-aware mapping controls, preserving the original spatial backbone."""
import torch
from torch import nn

MODES=('pre_fixedhalf','pre_learnzero','pre_learnhalf','post_fixedhalf')


class AggregationFST(nn.Module):
    def __init__(self,backbone,mode):
        super().__init__()
        if mode not in MODES:raise ValueError('registered aggregation control required')
        self.backbone,self.mode=backbone,mode
        self.alpha=nn.Parameter(torch.tensor(0. if mode=='pre_learnzero' else .5)) if mode.startswith('pre_learn') else None

    def coefficient(self):return .5 if self.alpha is None else float(self.alpha.detach())

    def transform(self,current,previous):
        if current.ndim!=4 or current.shape!=previous.shape:raise ValueError('matched NCHW observations required')
        if self.mode=='post_fixedhalf':return (self.backbone(current)+self.backbone(previous))/2
        alpha=.5 if self.alpha is None else self.alpha
        return self.backbone(current+alpha*(previous-current))

    def forward(self,current):return self.transform(current,current)

    @torch.no_grad()
    def project(self):
        if self.alpha is not None:self.alpha.clamp_(0,1)
