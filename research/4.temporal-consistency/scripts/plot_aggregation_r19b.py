"""Aggregation-order effects change with the operating distribution and clipping."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from exploration_r2_data import ROOT


def main():
    out=ROOT/'research/4.temporal-consistency/report/exploration-r19-aggregation'
    rows=json.loads((out/'contrast_summary.json').read_text());kinds=['official','R14_plain','spatial','raw_history','aligned_history']
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for ax,(rho,jpeg) in zip(axes.flat,[(0,False),(0,True),(.9,False),(.9,True)]):
        for path,offset,color,label in [('postaverage_clipped',-.18,'#4477AA','Clip each output, then average'),
                ('postaverage_thenclip',.18,'#CC6677','Average raw outputs, then clip')]:
            values=[next(r['mean_draw_psnr'] for r in rows if r['kind']==kind and r['rho']==rho and r['jpeg']==jpeg
                and r['first_path']==path and r['second_path']=='preaverage_current') for kind in kinds]
            ax.bar(np.arange(5)+offset,values,width=.35,color=color,label=label)
        ax.axhline(0,color='black',linewidth=.7);ax.grid(axis='y',alpha=.2)
        ax.set(title=f'rho={rho}, JPEG={jpeg}',xticks=np.arange(5),xticklabels=['Official','R14 plain','R16 spatial','R16 raw','R16 aligned'],
            ylabel='Post-average minus pre-average PSNR (dB)');ax.tick_params(axis='x',labelsize=8);ax.legend(fontsize=8)
    fig.suptitle('R19 follow-up: identical observations and weights; standard static aggregation controls\nPositive favors output averaging. Whole-network nonlinear order effect; not FST-only or real-video superiority.',fontsize=12)
    for extension in ['png','pdf']:fig.savefig(out/('aggregation_order.'+extension),dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
