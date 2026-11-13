"""Shape-only canonical CNN MACs, explicit repeated calls and reuse arithmetic."""
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from analyze_architecture_budget import MobileIELLENetS, ROOT
from fastdvdnet_adapter_r12 import MANIFEST, module, window_indices

BASE=ROOT/'research/4.temporal-consistency'


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def count(kind,h,w,models):
    with torch.device('meta'):
        net=(MobileIELLENetS(12) if kind=='MobileIE' else models.FastDVDnet(5)).eval()
    parameters=sum(p.numel() for p in net.parameters())
    layers,handles=[],[]
    def hook(name):
        def record(conv,inputs,output):
            layers.append({'layer':name,'output_shape':list(output.shape),
                'conv_MACs':output.numel()*(conv.in_channels//conv.groups)*conv.kernel_size[0]*conv.kernel_size[1],
                'single_output_tensor_FP32_bytes':output.numel()*4})
        return record
    for name,block in net.named_modules():
        if isinstance(block,nn.Conv2d):handles.append(block.register_forward_hook(hook(name)))
    with torch.inference_mode():
        if kind=='MobileIE':
            output=net(torch.empty(1,3,h,w,device='meta'));padded=[h,w]
        else:
            image=F.pad(torch.empty(1,15,h,w,device='meta'),(0,(-w)%4,0,(-h)%4),mode='reflect')
            condition=torch.empty(1,1,*image.shape[-2:],device='meta')
            output=net(image,condition)[...,:h,:w];padded=list(image.shape[-2:])
    for handle in handles:handle.remove()
    assert list(output.shape)==[1,3,h,w]
    calls=Counter(r['layer'] for r in layers)
    if kind=='FastDVDnet':
        assert parameters==2479096 and len(layers)==64
        assert all(n==3 for k,n in calls.items() if k.startswith('temp1.'))
        assert all(n==1 for k,n in calls.items() if k.startswith('temp2.'))
        first=sum(r['conv_MACs'] for r in layers if r['layer'].startswith('temp1.'))//3
        second=sum(r['conv_MACs'] for r in layers if r['layer'].startswith('temp2.'))
    else:
        assert parameters==4075;first,second=None,None
    return {'model':kind,'height':h,'width':w,'padded_shape':padded,'parameters':parameters,
            'conv_MACs':sum(r['conv_MACs'] for r in layers),'conv_invocations':len(layers),
            'largest_single_conv_output_FP32_bytes':max(r['single_output_tensor_FP32_bytes'] for r in layers),
            'temp1_one_call_MACs':first,'temp2_one_call_MACs':second,'layers':layers}


def main():
    assert not torch.cuda.is_initialized()
    manifest=json.loads(MANIFEST.read_text())
    path=ROOT/'third_party/fastdvdnet/models.py'
    expected=next(r for r in manifest['sources'] if r['path']=='third_party/fastdvdnet/models.py')
    assert digest(path)==expected['sha256']
    models=module(path,'mobileie_r47_shape_fastdvd_models')
    rows=[count(kind,h,w,models) for h,w in [(480,854),(288,512),(360,640),(1080,1920)]
          for kind in ['MobileIE','FastDVDnet']]
    mobile1080=next(r for r in rows if r['model']=='MobileIE' and r['height']==1080)
    assert mobile1080['conv_MACs']==7937740944
    reuse=[]
    for length in [16,64]:
        for mode in ['window','five_current']:
            indices=window_indices(length,mode)
            keys={tuple(row[start:start+3]) for row in indices for start in range(3)}
            reuse.append({'frames':length,'mode':mode,'unbuffered_temp1_calls':3*length,
                          'unique_temp1_tuples':len(keys),'temp2_calls':length,
                          'scope':'Arithmetic reuse potential only, no cache implemented or latency measured.'})
    assert all(r['unique_temp1_tuples']==r['frames']+(2 if r['mode']=='window' else 0) for r in reuse)
    assert not torch.cuda.is_initialized()
    out=BASE/'report/exploration-r47-cnn-budget';out.mkdir(exist_ok=False)
    record={'at':datetime.now().astimezone().isoformat(),'device':'meta; shape only','rows':rows,
            'reuse_arithmetic':reuse,'numerical_neural_sequences':0,'GPU':False,
            'scope':'Conv-only canonical unbuffered forward, not full pipeline FLOPs, cost timing, memory peak or phone.',
            'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),path,MANIFEST,
                         BASE/'docs/exploration-r47-cnn-cost-convention.md',ROOT/'MobileIE/model/lle.py']}}
    (out/'budget.json').write_text(json.dumps(record,indent=2)+'\n')
    text=['# 已发表视频参考与原空间CNN的统一预算','',
          'batch1输出帧；1MAC为一乘加，仅Conv2d。FastDVDnet按官方无跨帧缓存forward的3次temp1+1次temp2实际调用计数。','',
          '|模型|分辨率H×W|参数|CNN GMAC|最大单conv输出FP32 MiB|','|---|---|---:|---:|---:|']
    for r in rows:
        text.append(f"|{r['model']}|{r['height']}×{r['width']}|{r['parameters']}|{r['conv_MACs']/1e9:.6f}|{r['largest_single_conv_output_FP32_bytes']/2**20:.3f}|")
    text+=['','|帧数|模式|现有temp1调用|唯一三帧tuple|temp2调用|','|---:|---|---:|---:|---:|']
    for r in reuse:
        text.append(f"|{r['frames']}|{r['mode']}|{r['unbuffered_temp1_calls']}|{r['unique_temp1_tuples']}|{r['temp2_calls']}|")
    text+=['', '第一阶段tuple可复用的算术潜力不是已实现缓存或实测加速；窗口反射边界已包含，不能只按单个中心帧索引缓存逆序tuple。',
           '单张量体积不是峰值内存；MobileIE的flow/warp、非卷积和历史未包含，FastDVD的BN/ReLU/窗口构建也未包含。不由MAC差推导手机速度，不与实际质量合并成尚未完成的帕累托胜出。']
    (out/'tables.md').write_text('\n'.join(text)+'\n')
    print(json.dumps([{k:r[k] for k in ['model','height','width','parameters','conv_MACs']} for r in rows]))


if __name__=='__main__':main()
