"""Finite-sample orthogonal error components from every existing R59 output."""
from collections import defaultdict
from datetime import datetime
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dynamic_training_r34 import BASE, ROOT, digest


def write(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    run = BASE/'results/exploration-r59-drv-native-attempt001'
    runtime = json.loads((run/'runtime.json').read_text()); assert runtime['completed_normally']
    manifest = [json.loads(x) for x in (run/'prediction_manifest.jsonl').read_text().splitlines()]
    assert len(manifest) == 88
    index_path = BASE/'results/exploration-r58-drv-render-attempt001/index.json'
    target = {}
    for meta in json.loads(index_path.read_text())['scenes']:
        with np.load(ROOT/meta['path'],allow_pickle=False) as source: target[meta['scene']] = source['gt'][0].astype(np.float64)
    out = BASE/'report/exploration-r59-drv-native/decomposition001'; out.mkdir(exist_ok=False)
    start, records = time.perf_counter(), []
    for item in manifest:
        saved = np.load(ROOT/item['path'],mmap_mode='r',allow_pickle=False)
        assert saved.dtype == np.float32 and list(saved.shape) == item['shape']
        for scope,indices in [('primary4',[2,3,4,5]),('full8',list(range(8)))]:
            value = saved[indices].astype(np.float64)
            mean = value.mean(0); residual = mean-target[item['scene']]
            dc = residual.mean((0,1),keepdims=True)
            mse = float(np.mean((value-target[item['scene']])**2))
            components = dict(dc_mse=float(np.mean(dc**2)),
                              mean_structure_mse=float(np.mean((residual-dc)**2)),
                              temporal_mse=float(np.mean((value-mean)**2)))
            relative = abs(sum(components.values())-mse)/max(mse,1e-15)
            assert relative <= 1e-9
            records.append(dict(scene=item['scene'],reference=item['reference'],method=item['method'],
                                scope=scope,mse=mse,closure_relative_error=relative,**components))
    assert len(records) == 176
    groups = defaultdict(list)
    for r in records: groups[r['scope'],r['reference'],r['method']].append(r)
    summary = []
    for (scope,reference,method),rows in groups.items():
        assert len(rows) == 4
        mean = {k:float(np.mean([r[k] for r in rows])) for k in ['mse','dc_mse','mean_structure_mse','temporal_mse']}
        summary.append(dict(scope=scope,reference=reference,method=method,**mean,
                            **{k+'_fraction':mean[k]/mean['mse'] if mean['mse'] else 0
                               for k in ['dc_mse','mean_structure_mse','temporal_mse']}))
    write(out/'conditions.json',records); write(out/'summary.json',summary)
    table = ['# R59固定参考：适应性有限样本MSE分解', '',
             '全部88缓存/四源；0新CNN/SGD/GPU、无输出修正。此表pooled MSE不是原mean PSNR的替换。',
             '非因果/无限采样偏差或纯噪声：三项均可含参考、真实变化、错位和渲染影响。', '',
             '|范围|参考|输出|MSE|全通道均值项%|平均空间项%|样本时间项%|',
             '|---|---|---|---:|---:|---:|---:|']
    for r in summary:
        table.append(f"|{r['scope']}|{r['reference']}|{r['method']}|{r['mse']:.4f}|"
                     f"{100*r['dc_mse_fraction']:.4f}|{100*r['mean_structure_mse_fraction']:.4f}|{100*r['temporal_mse_fraction']:.4f}|")
    (out/'tables.md').write_text('\n'.join(table)+'\n')
    primary = [r for r in summary if r['scope'] == 'primary4']
    fig,ax = plt.subplots(figsize=(12,8.5)); bottom = np.zeros(len(primary))
    for key,label,color in [('dc_mse_fraction','Mean channel offset','#4477aa'),
                            ('mean_structure_mse_fraction','Mean spatial residual','#ee7733'),
                            ('temporal_mse_fraction','Sample temporal variation','#228833')]:
        widths = np.array([r[key]*100 for r in primary])
        ax.barh(np.arange(len(primary)),widths,left=bottom,label=label,color=color)
        bottom += widths
    ax.set_yticks(np.arange(len(primary)),[r['reference']+' / '+r['method'] for r in primary],fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0,100); ax.set_xlabel('Fraction of pooled primary4 MSE (%)')
    ax.set_title('Finite-sample decomposition on reused static train pilot; no causal/noise attribution',fontsize=10)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.035),ncol=3,fontsize=8)
    fig.tight_layout(); fig.savefig(out/'components.png',dpi=180); fig.savefig(out/'components.pdf'); plt.close(fig)
    write(out/'audit.json',dict(at=datetime.now().astimezone().isoformat(),completed_normally=True,
          existing_output_arrays=88,decomposition_rows=176,new_CNN=0,SGD=0,GPU=False,output_correction=False,
          max_relative_closure_error=max(r['closure_relative_error'] for r in records),
          seconds=time.perf_counter()-start,adaptive_analysis=True,causal_noise_decomposition=False,
          sources={str(p.relative_to(ROOT)):digest(p) for p in [run/'prediction_manifest.jsonl',index_path,
              BASE/'docs/exploration-r59-fixed-reference-error-decomposition.md',BASE/'scripts/decompose_drv_error_r59.py']}))
    print(json.dumps(dict(rows=176,new_CNN=0,max_relative_closure_error=max(r['closure_relative_error'] for r in records))))


if __name__ == '__main__': main()
