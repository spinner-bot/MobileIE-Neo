"""Frozen resampling diagnostic tables and figure; no GT quality ranking."""
import collections
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256

BASE=ROOT/'research/4.temporal-consistency'


def main():
    folder=BASE/'results/exploration-r22';out=BASE/'report/exploration-r22';out.mkdir(parents=True,exist_ok=False)
    rows=json.loads((folder/'commutator.json').read_text());replay=json.loads((folder/'continuation_replays.json').read_text())
    runtime=json.loads((folder/'runtime.json').read_text());fixtures=json.loads((folder/'fixtures.json').read_text())
    assert len(rows)==len(replay)==24 and fixtures['fixtures_passed']==2 and not runtime['cuda_initialized']
    assert len({(r['method'],r['scene'],r['stage']) for r in rows})==24
    assert max(r['identity_scaled_max_abs'] for r in rows)<=1e-10 and max(r['max_abs'] for r in replay)<=1e-6
    cfg=json.loads((folder/'config.json').read_text())
    for item in cfg['weights']:assert sha256(ROOT/item['path'])==item['sha256']
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['method'],r['stage'])].append(r)
    summary=[]
    for (method,stage),values in groups.items():
        summary.append({'method':method,'stage':stage,'scenes':len(values),'a':values[0]['a'],
            **{k:float(np.mean([r[k] for r in values])) for k in ['commutator_rms_valid','RGB_raw_MAE255','RGB_clipped_MAE255']}})
    save_json(out/'summary.json',summary)
    save_json(out/'audit.json',{'rows':24,'fixtures':2,'weights':3,'scenes':4,'GT_used_for_quality_ranking':False,
        'identity_scaled_max_abs':max(r['identity_scaled_max_abs'] for r in rows),
        'continuation_max_abs':max(r['max_abs'] for r in replay),'source_sha256':{str(p.relative_to(ROOT)):sha256(p) for p in folder.iterdir()},
        'scope':'12 fixed weight/scene pairs, two local interventions; no attribution of full-network warp error or temporal noise.'})
    fig,axes=plt.subplots(1,2,figsize=(10.5,4),layout='constrained')
    methods=['Official','R14_seed0','R14_seed1'];colors=['#4477aa','#ee6677']
    for j,stage in enumerate(['head','body']):
        for ax,key in zip(axes,['commutator_rms_valid','RGB_clipped_MAE255']):
            means=[next(r[key] for r in summary if r['method']==m and r['stage']==stage) for m in methods]
            ax.bar(np.arange(3)+(j-.5)*.35,means,width=.32,color=colors[j],label=stage)
            for i,m in enumerate(methods):
                yy=[r[key] for r in rows if r['method']==m and r['stage']==stage]
                ax.scatter(np.full(len(yy),i+(j-.5)*.35),yy,color='black',s=12,zorder=3)
            ax.set_xticks(range(3),methods);ax.legend();ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Local feature commutator RMS (valid pixels)')
    axes[1].set_ylabel('Continued clipped RGB MAE [0,255]')
    fig.suptitle('Frozen FST/interpolation diagnostic — four fixed frames; no quality ranking')
    for suffix in ['png','pdf']:fig.savefig(out/f'fst_resampling.{suffix}',dpi=170)
    plt.close(fig)
    lines=['# Frozen FST/interpolation diagnostic','','Bars are four-scene means; dots are all fixed scenes. No GT quality interpretation.','',
        '| Method | Stage | a | Local RMS | Continued clipped RGB MAE255 |','|---|---|---:|---:|---:|']
    for r in summary:lines.append(f"|{r['method']}|{r['stage']}|{r['a']:.8f}|{r['commutator_rms_valid']:.6f}|{r['RGB_clipped_MAE255']:.6f}|")
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
