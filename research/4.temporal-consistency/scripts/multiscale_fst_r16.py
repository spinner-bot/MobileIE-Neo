"""Small encoder-decoder FST mechanism reference with equal-capacity history controls."""
import torch
from torch import nn
from torch.nn import functional as F

MODES=('spatial','raw_history','aligned_history')


class SplitDecoder(nn.Module):
    """A 36→12 convolution stored in channel blocks, preserving original FP32 path."""
    def __init__(self,current):
        super().__init__();self.current=current;self.context=nn.Conv2d(24,12,3,padding=1,bias=False)
        nn.init.zeros_(self.context.weight)

    def forward(self,features):
        return self.current(features[:,:12].contiguous())+self.context(features[:,12:].contiguous())


class MultiscaleFST(nn.Module):
    def __init__(self,backbone,mode):
        super().__init__()
        if mode not in MODES:raise ValueError('registered multiscale control required')
        self.backbone,self.mode=backbone,mode
        self.encoder=nn.Sequential(nn.Conv2d(12,24,3,stride=2,padding=1),nn.PReLU(24),
            nn.Conv2d(24,24,3,padding=1),nn.PReLU(24))
        self.mixer=nn.Sequential(nn.Conv2d(48,24,3,padding=1),nn.PReLU(24))
        backbone.body.block1=SplitDecoder(backbone.body.block1)

    def transform(self,current,previous):
        if current.ndim!=4 or previous.shape!=current.shape:raise ValueError('matching NCHW current/history required')
        net=self.backbone
        if self.mode=='spatial':previous=current
        skip=net.head(current);past=net.head(previous)
        context=self.mixer(torch.cat([self.encoder(skip),self.encoder(past)],dim=1))
        context=F.interpolate(context,size=current.shape[-2:],mode='bilinear',align_corners=False)
        body=net.body(torch.cat([skip,context],dim=1))
        att=net.att(body);att1=net.att1(torch.max(att*body,dim=1,keepdim=True)[0])
        return net.tail(att1*att*body)

    def forward(self,current):return self.transform(current,current)
