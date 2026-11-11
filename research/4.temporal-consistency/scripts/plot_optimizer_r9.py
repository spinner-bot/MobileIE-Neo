"""Ordinary learning-rate controls on development and repeated exploration."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True)
    folder=Path(p.parse_args().report);rows=json.loads((folder/'domain_summary.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    methods=[('pretrained','Official','#777777'),('plain','Plain 1e-5','#ad6a49'),
             ('spatial_copy','Foldable copy','#b08a45'),('head_lr1e3','Fast first Conv5','#9b70a6'),
             ('full_lr1e4','Full 1e-4','#518aab'),('full_lr1e3','Full 1e-3','#648c62')]
    for ax,(domain,title) in zip(axes,[('dev','4 new project-dev clips'),('test','6 repeated ND exploration clips'),('legacy_synthetic','8 conditions / 2 synthetic clips')]):
        for index,(variant,label,color) in enumerate(methods,1):
            r=next(r for r in rows if r['domain']==domain and r['variant']==variant and r['fusion']=='raw')
            ax.scatter(r['temporal_error'],r['psnr_rgb'],s=60,color=color,label=str(index)+' '+label)
            ax.annotate(str(index),(r['temporal_error'],r['psnr_rgb']),xytext=(3,4),textcoords='offset points',fontsize=8)
        ax.set(title=title,xlabel='GT-relative temporal error (lower)',ylabel='PSNR, dB (higher)');ax.grid(alpha=.15)
    fig.suptitle('Same 4075-parameter single-frame function; final1000, two-seed means\nFoldable copy is a separate optimizer control; no final-test performance claim',fontsize=11)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.1),ncol=6,fontsize=8)
    fig.savefig(folder/'optimizer_controls.png',dpi=180,bbox_inches='tight');fig.savefig(folder/'optimizer_controls.pdf',bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
