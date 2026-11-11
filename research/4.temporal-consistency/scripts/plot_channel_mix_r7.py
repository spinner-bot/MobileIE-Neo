"""Channel calibration, motion transfer and reset response as standalone figures."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from exploration_r2_data import ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',required=True)
    args=p.parse_args()
    report=Path(args.report)
    selection=json.loads((ROOT/'research/4.temporal-consistency/results/exploration-r7-channel/selection.json').read_text())
    aligned=json.loads((ROOT/'research/4.temporal-consistency/report/exploration-r7-aligned/domain_summary.json').read_text())
    raw=json.loads((report/'domain_summary.json').read_text())
    event=json.loads((report/'event_frame12.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    refs=[selection['references']['first3'],selection['references']['uniform'],selection['references']['random109'],selection['selected']]
    axes[0,0].bar(range(4),[100*r['mean_gamma05_gain'] for r in refs],color=['#888888','#c38a50','#888888','#508bab'])
    axes[0,0].set(xticks=range(4),xticklabels=['First 0,1,2','Uniform 0,4,8','Random 3,5,10','Selected 6,7,8'],
                  ylabel='Predicted relative variance gain, %',title='Train-only static proxy, gamma=.5')
    curves=[('feature_selected_post','Feature, selected','#5189ab'),('output_finite','RGB, finite','#c88745'),('output_recursive','RGB, recursive','#648c62')]
    for ax,domain,title in [(axes[0,1],'test','6 real ND clips'),(axes[1,0],'legacy_synthetic','8 synthetic conditions / 2 source clips')]:
        baseline=next(r for r in aligned if r['domain']==domain and r['method']=='pretrained' and not r['input_reset'])
        ax.scatter(baseline['temporal_error'],baseline['ssim_rgb'],marker='*',s=100,color='black',label='Original')
        for method,label,color in curves:
            rr=[next(r for r in aligned if r['domain']==domain and r['method']==method and r['alignment']=='teacher_half_fb' and r['gamma']==g and not r['input_reset']) for g in [.1,.5]]
            ax.plot([r['temporal_error'] for r in rr],[r['ssim_rgb'] for r in rr],color=color,label=label)
            for r,marker in zip(rr,['o','s']):ax.scatter(r['temporal_error'],r['ssim_rgb'],color=color,marker=marker,s=35)
        unaligned=next(r for r in raw if r['domain']==domain and r['method']=='body_post_selected_g05' and r['fusion']=='raw' and not r['input_reset'])
        ax.scatter(unaligned['temporal_error'],unaligned['ssim_rgb'],marker='x',s=60,color='#b64f4f',label='Feature, no alignment, g=.5')
        ax.set(xlabel='GT-relative temporal error (lower is better)',ylabel='SSIM (higher is better)',title=title+'; same teacher half flow')
        ax.legend(fontsize=7)
    ax=axes[1,1]
    names=[('pretrained',False,'Original'),('body_post_selected_g05',False,'Feature / no reset'),('body_post_selected_g05',True,'Feature / input reset')]
    for i,(method,reset,label) in enumerate(names):
        values=[]
        for domain in ['event_light_step','event_cut']:
            rr=[r['psnr_rgb'] for r in event if r['domain']==domain and r['method']==method and r['input_reset']==reset and r['fusion']=='raw']
            values.append(np.mean(rr))
        ax.bar(np.arange(2)+(i-1)*.25,values,width=.24,label=label)
    ax.set(xticks=range(2),xticklabels=['Light step','Scene cut'],ylabel='Mean PSNR at event frame12',title='Fixed input reset isolates event lag')
    ax.legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.15)
    fig.suptitle('Frozen official weights: selecting noise channels helps static data, motion and strong RGB controls remain decisive',fontsize=12)
    fig.savefig(report/'channel_motion_controls.png',dpi=180)
    fig.savefig(report/'channel_motion_controls.pdf')
    plt.close(fig)


if __name__=='__main__':main()
