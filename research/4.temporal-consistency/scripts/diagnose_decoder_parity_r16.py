"""Locate widened-convolution arithmetic differences before changing the implementation."""
import argparse
import json
from pathlib import Path

import cv2
import torch
from torch.nn import functional as F

from explore_temporal_controls import save_json
from native_training_r14 import load_training,real_pair
from exploration_r2_data import load_images
from train_exploration_r2 import new_model
from training_context_r13 import synthetic_pair_with_context


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);out=Path(p.parse_args().out);out.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda');data,_=load_training();images=load_images('train')
    rx,*_=real_pair(data,0,0,'native_crop');sx,*_=synthetic_pair_with_context(images,0,0)
    low=torch.stack([rx,sx]).to(device).flatten(0,1);torch.manual_seed(0)
    ref=new_model(device);wide_decoder=torch.nn.Conv2d(36,12,3,padding=1).to(device)
    with torch.no_grad():
        wide_decoder.weight.zero_();wide_decoder.weight[:,:12].copy_(ref.body.block1.weight);wide_decoder.bias.copy_(ref.body.block1.bias)
    rows=[]
    with torch.no_grad():
        for tf32 in [True,False]:
            torch.backends.cudnn.allow_tf32=tf32
            h=ref.head(low);zeros=torch.zeros((len(h),24,*h.shape[2:]),device=device)
            original=ref.body.block1(h);wide=wide_decoder(torch.cat([h,zeros],1))
            split=F.conv2d(h,wide_decoder.weight[:,:12].contiguous(),wide_decoder.bias,padding=1)
            split=split+F.conv2d(zeros,wide_decoder.weight[:,12:].contiguous(),padding=1)
            body=(ref.body.weight1*wide)*(ref.body.weight2*wide)+ref.body.bias
            att=ref.att(body);att1=ref.att1(torch.max(att*body,dim=1,keepdim=True)[0]);wide_output=ref.tail(att1*att*body)
            rows.append({'cudnn_tf32_allowed':tf32,'body_conv_wide_minus_original_max':float((wide-original).abs().max()),
                'body_conv_split_minus_original_max':float((split-original).abs().max()),
                'full_wide_minus_original_max':float((wide_output-ref(low)).abs().max())})
    save_json(out/'diagnostic.json',{'rows':rows,'scope':'same real first task, zero context columns; numerical implementation diagnosis only'})
    print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
