"""Static known-information controls, with measured post-codec correlation kept separate."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from exploration_r2_data import ROOT


def main():
    out=ROOT/'research/4.temporal-consistency/report/exploration-r19';rows=json.loads((out/'contrast_summary.json').read_text())
    kinds=['official','R14_plain','spatial','raw_history','aligned_history']
    def get(kind,rho,jpeg,first,metric):return next(r[metric] for r in rows if r['kind']==kind and r['rho']==rho
        and r['jpeg']==jpeg and r['first_path']==first and r['second_path']=='current')
    fig,axes=plt.subplots(2,3,figsize=(14,7.5),layout='constrained')
    for row,rho in enumerate([0,.9]):
        for jpeg,offset,color in [(False,-.18,'#4477AA'),(True,.18,'#CC6677')]:
            x=np.arange(len(kinds));label='JPEG70' if jpeg else 'No JPEG'
            axes[row,0].bar(x+offset,[get(k,rho,jpeg,'preaverage_current','mean_draw_psnr') for k in kinds],width=.35,color=color,label=label)
            axes[row,1].bar(x+offset,[get(k,rho,jpeg,'preaverage_current','variance_ratio') for k in kinds],width=.35,color=color,label=label)
            x=np.arange(2)
            axes[row,2].bar(x+offset,[get(k,rho,jpeg,'learned_history','mean_draw_psnr') for k in kinds[-2:]],width=.35,color=color,label=label)
        for col in [0,1]:axes[row,col].set(xticks=np.arange(5),xticklabels=['Official','R14 plain','R16 spatial','R16 raw','R16 aligned'])
        axes[row,2].set(xticks=np.arange(2),xticklabels=['R16 raw','R16 aligned'])
        axes[row,0].set_title(f'rho={rho}: input averaging minus current PSNR')
        axes[row,1].set_title(f'rho={rho}: input averaging / current variance')
        axes[row,2].set_title(f'rho={rho}: learned history minus current PSNR')
        for col in [0,2]:axes[row,col].axhline(0,color='black',linewidth=.7)
        axes[row,1].axhline(1,color='black',linewidth=.7)
        for ax in axes[row]:ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8);ax.tick_params(axis='x',labelsize=8)
    fig.suptitle('R19: fixed current observations, known static correspondence, 64 independent pairs per condition\nTwo reused training images / two photon levels; standard pre-averaging reference, not real-video performance or novelty.',fontsize=12)
    for extension in ['png','pdf']:fig.savefig(out/('known_temporal_information.'+extension),dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
