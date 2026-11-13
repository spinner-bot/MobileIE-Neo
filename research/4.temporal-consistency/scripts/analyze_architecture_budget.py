"""Size untrained spatial allocation controls using a meta graph, without a GPU."""
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'MobileIE'))
from model.lle import MobileIELLENetS


def separable(channels):
    return nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                         nn.Conv2d(channels, channels, 1))


def count(label):
    net = MobileIELLENetS(12)
    if label in ['separable_body', 'separable_head_body']:
        net.body.block1 = separable(12)
    if label == 'separable_head_body':
        net.head.block1[2] = separable(12)
    parameters = sum(p.numel() for p in net.parameters())
    net = net.to('meta').eval()
    layers, handles = [], []
    def hook(name):
        def record(module, inputs, output):
            macs = output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
            layers.append({'layer': name, 'shape': list(output.shape), 'conv_MACs': macs,
                           'parameters': sum(p.numel() for p in module.parameters())})
        return record
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(hook(name)))
    with torch.inference_mode():
        output = net(torch.empty((1, 3, 1080, 1920), device='meta'))
    assert list(output.shape) == [1, 3, 1080, 1920]
    for handle in handles:
        handle.remove()
    return {'variant': label, 'parameters': parameters, 'conv_MACs_1080p': sum(r['conv_MACs'] for r in layers), 'layers': layers}


def main():
    assert not torch.cuda.is_initialized()
    rows = [count(name) for name in ['original', 'separable_body', 'separable_head_body']]
    assert rows[0]['parameters'] == 4075 and rows[0]['conv_MACs_1080p'] == 7937740944
    state = [{'description': '12ch full-resolution FP32', 'bytes': 12*1080*1920*4},
             {'description': '8ch quarter-resolution FP32', 'bytes': 8*270*480*4},
             {'description': '8ch quarter-resolution FP16', 'bytes': 8*270*480*2}]
    out = ROOT/'research/4.temporal-consistency/report/architecture-budget'
    out.mkdir(parents=True, exist_ok=False)
    result = {'rows': rows, 'illustrative_history_tensors': state, 'cuda_initialized': torch.cuda.is_initialized(),
              'device': 'meta; no numerical inference or GPU context', 'trained_variants': 0,
              'scope': 'Known depthwise-separable spatial controls, not a new method or equivalent conversion. Conv-only MACs exclude activation/pooling/warping; tensor sizes exclude aliases, other buffers and peak memory. No latency or quality measured.'}
    (out/'budget.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Spatial compute allocation before a new video design', '',
             'Untrained allocation controls only. Separable convolutions change the function class and cannot exactly factor arbitrary pretrained dense kernels.', '',
             '|Control|Parameters|Conv GMAC@1080p|Saved Conv GMAC|', '|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['variant']}|{r['parameters']}|{r['conv_MACs_1080p']/1e9:.6f}|{(rows[0]['conv_MACs_1080p']-r['conv_MACs_1080p'])/1e9:.6f}|")
    lines += ['', '|Illustrative history tensor|MiB@1080p|', '|---|---:|']
    for r in state:
        lines.append(f"|{r['description']}|{r['bytes']/2**20:.3f}|")
    lines += ['', 'Lower MACs do not imply lower phone latency. History sizes are standalone tensor lower bounds; slicing a large backing tensor may retain the full storage. These calculations reserve budget for future causal-state controls, and provide no evidence of enhancement quality.']
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'rows': [{k: r[k] for k in ['variant', 'parameters', 'conv_MACs_1080p']} for r in rows], 'cuda_initialized': False}))


if __name__ == '__main__':
    main()
