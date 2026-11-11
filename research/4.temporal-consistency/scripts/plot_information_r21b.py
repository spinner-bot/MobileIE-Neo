"""Show all correlation/JPEG groups, including failures against a classical reference."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from aggregation_model_r21 import MODES
from exploration_r2_data import ROOT


def main():
    out=ROOT/'research/4.temporal-consistency/report/exploration-r21b';rows=json.loads((out/'contrast_summary.json').read_text())
    groups=[(0,False),(0,True),(.9,False),(.9,True)]
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    for j,mode in enumerate(MODES):
        for ax,contrast in zip(axes,['actual_history','vs_unadapted_postaverage']):
            values=[next(r['mean_draw_psnr'] for r in rows if r['kind']==mode and r['contrast']==contrast and r['rho']==rho and r['jpeg']==jpeg) for rho,jpeg in groups]
            ax.bar(np.arange(4)+(j-1.5)*.19,values,width=.18,label=mode)
    for ax in axes:
        ax.axhline(0,color='black',lw=.8);ax.grid(axis='y',alpha=.2)
        ax.set_xticks(range(4),['rho0\nno JPEG','rho0\nJPEG70','rho.9\nno JPEG','rho.9\nJPEG70'])
        ax.set_ylabel('Paired mean draw PSNR difference (dB)')
    axes[0].set_title('Actual history minus same-model current replacement')
    axes[1].set_title('Actual history minus R14 classical raw-output average')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=9)
    fig.suptitle('Static perfect correspondence; two TRAIN images, fixed step4000, same CPU\nMeans over two photons and two seeds; not video performance or independent tests',fontsize=11)
    for suffix in ['png','pdf']:fig.savefig(out/f'information_controls.{suffix}',dpi=170)
    plt.close(fig)


if __name__=='__main__':main()
