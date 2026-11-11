"""Earlier context can change training without materially increasing history use."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True)
    folder=Path(p.parse_args().report);rows=json.loads((folder/'domain_summary.json').read_text())
    def get(d,v,h):return next(r for r in rows if r['domain']==d and r['variant']==v and r['history']==h and not r['input_reset'] and r['fusion']=='raw')
    domains=['test','dev','dev_native','legacy_synthetic'];labels=['Old ND','New dev','Native dev','Synthetic']
    fig,axes=plt.subplots(2,2,figsize=(11,6),constrained_layout=True)
    for col,(key,name) in enumerate([('psnr_rgb','PSNR, dB'),('temporal_error','Temporal error')]):
        for j,(new,old,label,color) in enumerate([('concat_context','concat_high','Raw past','#518aab'),
            ('concat_aligned_context','concat_aligned_high','Aligned past','#648c62')]):
            total=[get(d,new,'causal')[key]-get(d,old,'causal')[key] for d in domains]
            actual=[(get(d,new,'causal')[key]-get(d,new,'current')[key])-(get(d,old,'causal')[key]-get(d,old,'current')[key]) for d in domains]
            for ax,values in [(axes[0,col],total),(axes[1,col],actual)]:
                ax.bar(np.arange(4)+(j-.5)*.3,values,width=.28,label=label,color=color)
        axes[0,col].set(title='New training minus R11: '+name)
        axes[1,col].set(title='Change in causal-minus-self increment: '+name)
    for ax in axes.flat:
        ax.set(xticks=range(4),xticklabels=labels);ax.axhline(0,color='black',lw=.7);ax.grid(axis='y',alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('Task pixels/GT/Adam preserved; first-task self history 50% -> about1.6%\nSpatial full1000 replay max difference0; two-seed means, final1000, raw/no reset',fontsize=11)
    fig.savefig(folder/'training_context_controls.png',dpi=180);fig.savefig(folder/'training_context_controls.pdf');plt.close(fig)


if __name__=='__main__':main()
