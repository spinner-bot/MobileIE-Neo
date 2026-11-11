"""Quality/stability and true-history deltas for all fixed multiscale candidates."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from exploration_r2_data import ROOT


def main():
    out=ROOT/'research/4.temporal-consistency/report/exploration-r16'
    rows=json.loads((out/'domain_summary.json').read_text());domains=['test','dev','dev_native','legacy_synthetic']
    def value(d,v,s,h,k):return next(r[k] for r in rows if r['domain']==d and r['variant']==v and r['step']==s
        and r['history']==h and not r['input_reset'] and r['fusion']=='raw')
    fig,axes=plt.subplots(2,3,figsize=(14,7.6),layout='constrained');x=np.arange(4)
    modes=['R14_plain','R14_copy','spatial','raw_history','aligned_history']
    for ax,key,title in zip(axes[0],['psnr_rgb','ssim_rgb','temporal_error'],['4000-step PSNR','4000-step SSIM','4000-step temporal error (lower better)']):
        for v in modes:ax.plot(x,[value(d,v,4000,'causal',key) for d in domains],'o-',label=v)
        ax.set(title=title,xticks=x,xticklabels=['ND test','ND dev','Native dev','Synthetic']);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    for ax,key,title in zip(axes[1],['psnr_rgb','ssim_rgb','temporal_error'],['Actual history minus current PSNR','Actual history minus current SSIM','Actual history minus current temporal error']):
        for i,(v,s) in enumerate([(v,s) for v in ['raw_history','aligned_history'] for s in [1000,4000]]):
            ax.bar(x+(i-1.5)*.19,[value(d,v,s,'causal',key)-value(d,v,s,'current',key) for d in domains],width=.18,label=v+' '+str(s))
        ax.axhline(0,color='black',linewidth=.7);ax.set(title=title,xticks=x,xticklabels=['ND test','ND dev','Native dev','Synthetic']);ax.grid(axis='y',alpha=.2)
    axes[1,0].legend(fontsize=7)
    fig.suptitle('R16: native matched task stream, two-seed means, raw outputs without reset\nSame-capacity multiscale controls: 24,955 parameters; R14 plain 4,075 / copy 4,975. Reused exploration clips.',fontsize=12)
    for extension in ['png','pdf']:fig.savefig(out/('multiscale_history_controls.'+extension),dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
