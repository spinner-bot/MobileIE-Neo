"""Matched-CPU quality/stability plots with explicit current-history replacements."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from aggregation_model_r21 import MODES
from exploration_r2_data import ROOT


def main():
    out=ROOT/'research/4.temporal-consistency/report/exploration-r21'
    rows=json.loads((out/'domain_summary.json').read_text())
    rows=[r for r in rows if r['step']==4000 and not r['input_reset'] and r['fusion']=='raw']
    domains=['test','dev','dev_native','legacy_synthetic'];titles=['Old ND exploration','Area-resized development','Native development','Legacy synthetic (2 sources)']
    def select(domain,variant,history='causal'):
        found=[r for r in rows if r['domain']==domain and r['variant']==variant and r['history']==history];assert len(found)==1
        return found[0]
    colors=plt.get_cmap('tab10');fig,axes=plt.subplots(2,4,figsize=(15,7.4),layout='constrained')
    for col,(domain,title) in enumerate(zip(domains,titles)):
        for j,variant in enumerate(MODES):
            causal=select(domain,variant);current=select(domain,variant,'current')
            for ax,xkey,ykey in [(axes[0,col],'temporal_error','psnr_rgb'),(axes[1,col],'edge_mae','ssim_rgb')]:
                ax.scatter(current[xkey],current[ykey],facecolors='none',edgecolors=colors(j),s=48,zorder=3)
                ax.scatter(causal[xkey],causal[ykey],color=colors(j),s=45,label=variant,zorder=4)
                ax.annotate('',xy=(causal[xkey],causal[ykey]),xytext=(current[xkey],current[ykey]),
                    arrowprops={'arrowstyle':'->','color':colors(j),'lw':1.2})
        for variant,marker,label in [('R14_plain','x','Single-frame R14'),('R20_postwarp01','s','R20 output-warp .1'),
                ('R20_postwarp05','^','R20 output-warp .5'),('R20_recursive_half01','+','R20 recursive .1')]:
            r=select(domain,variant)
            for ax,xkey,ykey in [(axes[0,col],'temporal_error','psnr_rgb'),(axes[1,col],'edge_mae','ssim_rgb')]:
                ax.scatter(r[xkey],r[ykey],marker=marker,c='black',s=55,label=label,zorder=5)
        axes[0,col].set_title(title);axes[0,col].set_xlabel('GT-compensated temporal error (lower better)')
        axes[1,col].set_xlabel('Edge MAE (lower better)')
        axes[0,col].set_ylabel('PSNR RGB (dB)');axes[1,col].set_ylabel('SSIM RGB')
        for ax in axes[:,col]:ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=9)
    fig.suptitle('Fixed step4000, no reset, raw output; all comparisons on CPU, two-seed means\nHollow→filled: replace current-copy with actual causal history. Arrows do not imply consistent benefit.',fontsize=12)
    for suffix in ['png','pdf']:fig.savefig(out/f'aggregation_history_tradeoffs.{suffix}',dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
