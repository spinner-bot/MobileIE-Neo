"""Audit repeated raw evaluations and summarize the stronger fusion controls."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def mean(rows,key):
    return float(np.mean([r[key] for r in rows]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[3]
    results=root/'research/4.temporal-consistency/results'
    source=results/'exploration-r2-trained-fusion'
    rows=[json.loads(line) for line in (source/'per_sequence.jsonl').read_text().splitlines()]
    assert len(rows)==864
    assert len({(r['model'],r['fusion'],r['scene'],r['case'],r['input_seed']) for r in rows})==864
    references={}
    choices={'task':('exploration-r2-training-plain','task'),
             'task_T1':('exploration-r2-lambda10','task_temporal'),
             'bounded_T0.1':('exploration-r2-training-bounded','bounded_temporal')}
    for label,(directory,variant) in choices.items():
        for seed in [0,1]:
            values=json.loads((results/directory/f'{variant}-seed{seed}'/'validation.json').read_text())
            for value in values:
                references[(f'{label}-seed{seed}',value['scene'],value['case'],value['input_seed'])]=value
    max_difference=0.
    for row in rows:
        if row['fusion']=='raw':
            expected=references[(row['model'],row['scene'],row['case'],row['input_seed'])]
            max_difference=max(max_difference,max(abs(row[k]-expected[k]) for k in ['psnr_rgb','ssim_rgb','temporal_error']))
    assert max_difference<1e-4
    config=json.loads((source/'config.json').read_text())
    for model in config['models'].values():
        assert model['step']==400
        assert hashlib.sha256((root/model['path']).read_bytes()).hexdigest()==model['sha256']
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    audit={'rows':864,'raw_recomputed':288,'new_fusion_evaluations':576,
           'raw_max_abs_difference':max_difference,'checkpoint_hashes_match':True}
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    summaries=[]
    lines=['# Trained-model plus fusion controls','',
           'Synthetic exploratory validation; raw rows duplicate the earlier training audit.','',
           '| Model | Fusion | PSNR | SSIM | Temporal error |',
           '| --- | --- | ---: | ---: | ---: |']
    for label in choices:
        for fusion in ['raw','flow','fb']:
            selected=[r for r in rows if r['model'].rsplit('-seed',1)[0]==label and r['fusion']==fusion]
            item={'model':label,'fusion':fusion,**{k:mean(selected,k) for k in ['psnr_rgb','ssim_rgb','edge_mae','temporal_error']}}
            summaries.append(item)
            lines.append(f"| {label} | {fusion} | {item['psnr_rgb']:.4f} | {item['ssim_rgb']:.4f} | {item['temporal_error']:.4f} |")
    (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':.2})
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for label in choices:
        selected=[r for r in summaries if r['model']==label]
        axes[0].plot([r['temporal_error'] for r in selected],[r['psnr_rgb'] for r in selected],'o-',label=label)
        axes[1].plot([r['edge_mae'] for r in selected],[r['ssim_rgb'] for r in selected],'o-',label=label)
    axes[0].set(xlabel='GT-relative temporal error (0-255)',ylabel='RGB PSNR (dB)',title='Raw -> flow -> confidence fusion')
    axes[1].set(xlabel='Edge MAE',ylabel='RGB SSIM',title='Detail-quality tradeoff')
    for ax in axes:ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'trained_fusion_tradeoff.png',dpi=160)
    fig.savefig(out/'trained_fusion_tradeoff.svg')
    print(json.dumps(audit,indent=2))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    main()
