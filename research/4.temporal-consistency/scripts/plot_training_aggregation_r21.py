"""Plot all completed training traces, without interpreting them as video quality."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from aggregation_model_r21 import MODES
from exploration_r2_data import ROOT
from explore_temporal_controls import save_json,sha256


def main():
    base=ROOT/'research/4.temporal-consistency';out=base/'process/exploration-r21';sources={}
    audit=json.loads((out/'training-audit.json').read_text());assert audit['new_updates']==32000
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained');colors=plt.get_cmap('tab10')
    for j,mode in enumerate(MODES):
        for seed in [0,1]:
            path=base/'results'/f'exploration-r21-seed{seed}'/mode/'training.jsonl';sources[str(path.relative_to(ROOT))]=sha256(path)
            rows=[json.loads(s) for s in path.read_text().splitlines()];assert len(rows)==4000
            style='-' if seed==0 else '--';x=np.arange(1,4001)
            axes[0].plot(x,[r['alpha_after_projection'] for r in rows],style,color=colors(j),lw=1,label=f'{mode} / s{seed}')
            smoothed=np.convolve([r['task'] for r in rows],np.ones(100)/100,'valid')
            axes[1].plot(x[99:],smoothed,style,color=colors(j),lw=1)
            norms=np.array([r['gradient_norm_all'] for r in rows]);clip_rate=np.convolve((norms>1).astype(float),np.ones(100)/100,'valid')
            axes[2].plot(x[99:],clip_rate,style,color=colors(j),lw=1)
    for ax in axes:ax.set_xlabel('Optimizer updates per independent model');ax.grid(alpha=.2)
    axes[0].set_ylabel('Aggregation alpha (after projection)');axes[0].set_ylim(-.02,.55)
    axes[0].legend(fontsize=7);axes[1].set_ylabel('100-update task-loss mean');axes[2].set_ylabel('100-update fraction with global norm > 1')
    fig.suptitle('Fixed 4000-update training traces — loss and alpha are not quality evidence')
    for suffix in ['png','pdf']:fig.savefig(out/f'training_curves.{suffix}',dpi=170)
    plt.close(fig);save_json(out/'training_curves_sources.json',{'trace_sha256':sources,'training_audit_sha256':sha256(out/'training-audit.json')})


if __name__=='__main__':main()
