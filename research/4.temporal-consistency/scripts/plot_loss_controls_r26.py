"""Same-backend objective and history effects versus each objective's strongest fixed post control."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from exploration_r2_data import ROOT


def main():
    out=ROOT/'research/4.temporal-consistency/report/exploration-r26'
    rows=json.loads((out/'domain_summary.json').read_text())
    rows=[r for r in rows if r['step']==4000 and not r['input_reset']]
    def select(domain,architecture,objective,history='causal',fusion='raw'):
        values=[r for r in rows if r['domain']==domain and r['architecture']==architecture and r['objective']==objective and r['history']==history and r['fusion']==fusion]
        assert len(values)==1;return values[0]
    domains=['test','dev','dev_native','legacy_synthetic'];titles=['Old ND exploration','Resized development','Native development','Legacy synthetic (2 sources)']
    fig,axes=plt.subplots(2,4,figsize=(15,7.8),layout='constrained');colors=plt.get_cmap('tab10')
    candidates=[('pre_fixedhalf','charb'),('pre_fixedhalf','mse'),('post_fixedhalf','charb'),('post_fixedhalf','mse')]
    for col,(domain,title) in enumerate(zip(domains,titles)):
        for j,(architecture,objective) in enumerate(candidates):
            past=select(domain,architecture,objective);current=select(domain,architecture,objective,'current')
            for ax,xkey,ykey in [(axes[0,col],'temporal_error','psnr_rgb'),(axes[1,col],'edge_mae','ssim_rgb')]:
                ax.scatter(current[xkey],current[ykey],s=45,facecolors='none',edgecolors=colors(j),zorder=3)
                ax.scatter(past[xkey],past[ykey],s=45,color=colors(j),label=architecture+' / '+objective,zorder=4)
                ax.annotate('',xy=(past[xkey],past[ykey]),xytext=(current[xkey],current[ykey]),arrowprops={'arrowstyle':'->','color':colors(j),'lw':1.2})
        for objective,fusion,marker,label in [('charb','raw','x','Charb spatial'),('mse','raw','+','MSE spatial'),
                ('charb','postwarp05','^','Charb spatial + post .5'),('mse','postwarp05','s','MSE spatial + post .5')]:
            r=select(domain,'plain',objective,fusion=fusion)
            for ax,xkey,ykey in [(axes[0,col],'temporal_error','psnr_rgb'),(axes[1,col],'edge_mae','ssim_rgb')]:
                ax.scatter(r[xkey],r[ykey],marker=marker,c='black',s=60,label=label,zorder=5)
        axes[0,col].set_title(title);axes[0,col].set_xlabel('Temporal error (lower better)');axes[1,col].set_xlabel('Edge MAE (lower better)')
        axes[0,col].set_ylabel('PSNR RGB (dB)');axes[1,col].set_ylabel('SSIM RGB')
        for ax in axes[:,col]:ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=9)
    fig.suptitle('Fixed step4000 / no reset; common backend, two-seed means\nHollow→filled: current replacement→causal history. MSE spatial-plus-post is a required strong reference.',fontsize=12)
    for suffix in ['png','pdf']:fig.savefig(out/f'objective_history_tradeoffs.{suffix}',dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
