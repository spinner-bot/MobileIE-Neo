"""Fixed-checkpoint data/depth effects and same-weight history controls."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from exploration_r2_data import ROOT


def main():
    out=ROOT/'research/4.temporal-consistency/report/exploration-r14'
    rows=json.loads((out/'domain_summary.json').read_text())
    def value(domain,prep,variant,step,history,key):
        return next(r[key] for r in rows if r['domain']==domain and r['train_preparation']==prep
            and r['variant']==variant and r['step']==step and r['history']==history
            and not r['input_reset'] and r['fusion']=='raw')
    fig,axes=plt.subplots(2,3,figsize=(14,7.8),layout='constrained')
    colors={'plain':'#4477AA','copy':'#228833','aligned_concat':'#CC6677'}
    for ax,domain in zip(axes[0],['test','dev','dev_native']):
        for variant,color in colors.items():
            for prep,style in [('native_crop','-'),('resized','--')]:
                ax.plot([1000,4000],[value(domain,prep,variant,s,'causal','psnr_rgb') for s in [1000,4000]],
                    style+'o',color=color,label=variant+' / '+prep)
        ax.set(title=domain+' / quality',xlabel='Cumulative training updates',ylabel='RGB PSNR (dB)')
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    domains=['test','dev','dev_native','legacy_synthetic'];x=np.arange(4)
    for ax,key,title in zip(axes[1],['ssim_rgb','temporal_error','psnr_rgb'],
            ['4000-step SSIM','4000-step temporal error (lower better)','Actual history minus current-only PSNR']):
        if key!='psnr_rgb':
            for prep,style in [('native_crop','-'),('resized','--')]:
                ax.plot(x,[value(d,prep,'aligned_concat',4000,'causal',key) for d in domains],style+'o',label=prep)
        else:
            for prep,offset in [('native_crop',-.18),('resized',.18)]:
                ax.bar(x+offset,[value(d,prep,'aligned_concat',4000,'causal',key)-value(d,prep,'aligned_concat',4000,'current',key) for d in domains],width=.35,label=prep)
            ax.axhline(0,color='black',linewidth=.7)
        ax.set(title=title,xticks=x,xticklabels=['ND test','ND dev','Native dev','Synthetic'])
        ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.suptitle('R14: matched 1000/4000 updates; raw outputs, no reset, two-seed means\nNative/resized jointly changes noise, field of view, motion scale and global context; exploratory reused clips',fontsize=12)
    for suffix in ['png','pdf']:fig.savefig(out/('native_depth_controls.'+suffix),dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
