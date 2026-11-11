"""Distinguish training changes from the small actual-history increments."""
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
    def get(domain,variant,history='causal'):
        return next(r for r in rows if r['domain']==domain and r['variant']==variant and r['history']==history
                    and not r['input_reset'] and r['fusion']=='raw')
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    choices=[('plain_high','Plain','#777777'),('copy_high','Foldable copy','#bd8845'),
             ('concat_high','Raw past','#518aab'),('concat_aligned_high','Aligned past','#648c62')]
    for ax,domain,title in [(axes[0,0],'test','6 repeated ND exploration clips'),(axes[0,1],'dev_native','4 project-dev native crops')]:
        for i,(variant,label,color) in enumerate(choices,1):
            a=get(domain,variant);ax.scatter(a['temporal_error'],a['psnr_rgb'],s=55,color=color,label=label)
            ax.annotate(str(i),(a['temporal_error'],a['psnr_rgb']),xytext=(3,4),textcoords='offset points',fontsize=8)
            if variant.startswith('concat'):
                b=get(domain,variant,'current')
                ax.scatter(b['temporal_error'],b['psnr_rgb'],facecolors='none',edgecolors=color,s=55)
                ax.plot([a['temporal_error'],b['temporal_error']],[a['psnr_rgb'],b['psnr_rgb']],color=color,lw=1)
        ax.set(title=title,xlabel='GT-relative temporal error (lower)',ylabel='PSNR, dB (higher)');ax.legend(fontsize=8)
    domains=['test','dev','dev_native','legacy_synthetic'];labels=['Old ND','New dev','Native dev','Synthetic']
    for ax,key,title in [(axes[1,0],'psnr_rgb','Actual history: PSNR change, dB'),(axes[1,1],'temporal_error','Actual history: temporal error change')]:
        for j,(variant,label,color) in enumerate(choices[2:]):
            values=[get(d,variant)[key]-get(d,variant,'current')[key] for d in domains]
            ax.bar(np.arange(4)+(j-.5)*.3,values,width=.28,label=label,color=color)
        ax.set(xticks=range(4),xticklabels=labels,title=title);ax.axhline(0,color='black',lw=.7);ax.legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.15)
    fig.suptitle('Same global Adam1e-3 / clip1, fixed1000, two-seed means; raw / no reset\nTop: filled causal, hollow self-history; bottom: causal minus self-history at the SAME weights',fontsize=10)
    fig.savefig(folder/'strong_history_controls.png',dpi=180);fig.savefig(folder/'strong_history_controls.pdf');plt.close(fig)


if __name__=='__main__':main()
