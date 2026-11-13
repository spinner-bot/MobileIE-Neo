"""Only missing StableLLVE meta Conv MACs, reusing prior three-family records."""
from datetime import datetime
import json
import time
import traceback

import torch

from dynamic_training_r34 import BASE, ROOT, digest
from stablellve_adapter_r15 import MANIFEST, load_model


def write(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    torch.set_num_threads(2); assert not torch.cuda.is_initialized()
    out = BASE/'results/exploration-r68-reference-budget001'; out.mkdir(exist_ok=False)
    previous = BASE/'report/exploration-r47-cnn-budget/budget.json'
    nano_path = BASE/'results/exploration-r55-conv-budget001/cost.json'
    video, nano = json.loads(previous.read_text()), json.loads(nano_path.read_text())
    assert not video['GPU'] and not nano['cuda_initialized']
    assert all(digest(ROOT/path) == sha for path, sha in video['source_sha256'].items())
    assert digest(BASE/'scripts/count_multinex_convs_r55.py') == nano['source_sha256']
    start, complete, new, combined = time.perf_counter(), False, [], []
    try:
        model = load_model(torch.device('cpu')); parameters = sum(p.numel() for p in model.parameters())
        assert parameters == 4316259
        model = model.to('meta').eval()
        for height, width in [(288, 512), (360, 640), (480, 854), (1080, 1920)]:
            calls, handles = [], []
            def hook(name):
                def count(module, inputs, output):
                    calls.append(dict(layer=name, input_shape=list(inputs[0].shape), output_shape=list(output.shape),
                        conv_MACs=int(output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]),
                        output_FP32_bytes=int(output.numel()*4)))
                return count
            try:
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Conv2d): handles.append(module.register_forward_hook(hook(name)))
                value = model(torch.empty(1, 3, height, width, device='meta'))
                assert list(value.shape) == [1, 3, height, width]
            finally:
                for handle in handles: handle.remove()
            new.append(dict(model='StableLLVE', height=height, width=width, parameters=parameters,
                conv_MACs=sum(r['conv_MACs'] for r in calls), conv_invocations=len(calls),
                largest_single_conv_output_FP32_bytes=max(r['output_FP32_bytes'] for r in calls), layers=calls))
        for row in video['rows']:
            combined.append({key: row[key] for key in ['model', 'height', 'width', 'parameters', 'conv_MACs',
                             'conv_invocations', 'largest_single_conv_output_FP32_bytes']})
        for row in nano['rows']:
            combined.append(dict(model='Nano', height=row['height'], width=row['width'], parameters=row['parameters'],
                conv_MACs=row['Conv_MACs'], conv_invocations=row['executed_Conv_calls'],
                largest_single_conv_output_FP32_bytes=max(int(torch.tensor(r['output_shape']).prod())*4 for r in row['calls'])))
        combined += [{key: row[key] for key in row if key != 'layers'} for row in new]
        assert len(combined) == 16 and len({(r['model'], r['height'], r['width']) for r in combined}) == 16
        for row in combined:
            row['future_wait_max'] = 2 if row['model'] == 'FastDVDnet' else 0
            row['phone_latency_ms'] = None
        write(out/'stablellve_layers.json', dict(rows=new, original_manifest_sha256=digest(MANIFEST),
              upstream_model_sha256=digest(ROOT/'third_party/stablellve/model.py'), real_CNN_images=0, CUDA=False))
        write(out/'budget.json', dict(at=datetime.now().astimezone().isoformat(), rows=combined,
              previous_sources={str(p.relative_to(ROOT)): digest(p) for p in [previous, nano_path]},
              protocol_sha256=digest(BASE/'docs/exploration-r68-reference-budget-protocol.md'),
              source_sha256=digest(BASE/'scripts/count_stablellve_budget_r68.py'), real_CNN_images=0, SGD=0, CUDA=False,
              scope='Only executed Conv2d MAC shape arithmetic; not full pipeline FLOPs/latency/peak memory or handset performance.'))
        folder = BASE/'report/exploration-r68-reference-budget'; folder.mkdir(exist_ok=False)
        table = ['# 共同部署卷积预算', '',
            'batch1每输出帧，1MAC=一乘加，只计Conv2d。StableLLVE新增meta形状，其余沿原R47/R55，不做新图像推理。',
            '不计插值/池化/解析先验/规约/激活/归一/点操作、flow/warp/访存/IO；单张量不是峰值，GMAC不是手机延迟。', '',
            '|模型|H×W|参数|Conv GMAC|最大单Conv FP32 MiB|未来帧|手机ms|', '|---|---|---:|---:|---:|---:|---|']
        for r in combined:
            table.append(f'|{r["model"]}|{r["height"]}×{r["width"]}|{r["parameters"]}|{r["conv_MACs"]/1e9:.6f}|'
                         f'{r["largest_single_conv_output_FP32_bytes"]/2**20:.3f}|{r["future_wait_max"]}|未测|')
        table += ['', 'FastDVDnet保留官方未缓存三次temp1+一次temp2，不把可复用算术当已实现加速。',
            'MobileIE/Nano/StableLLVE raw无持久时间状态；经典rec另需两个RGB帧与flow/warp/工作区，不包含于Conv预算。',
            '公开clean与R65两臂结构相同不代表质量/训练相同；不由预算直接声称Pareto胜出或放弃质量评价。']
        (folder/'tables.md').write_text('\n'.join(table)+'\n')
        write(folder/'sources.json', {str(p.relative_to(ROOT)): digest(p) for p in [out/'budget.json', out/'stablellve_layers.json', previous, nano_path]})
        assert not torch.cuda.is_initialized(); complete = True
        print(json.dumps([r for r in combined if r['height'] == 1080]), flush=True)
    except BaseException:
        (out/'failure.txt').write_text(traceback.format_exc()); raise
    finally:
        write(out/'runtime.json', dict(completed_normally=complete, new_meta_shapes=len(new), combined_rows=len(combined),
            real_CNN_images=0, SGD=0, cuda_initialized=torch.cuda.is_initialized(), seconds=time.perf_counter()-start,
            ended=datetime.now().astimezone().isoformat()))


if __name__ == '__main__': main()
