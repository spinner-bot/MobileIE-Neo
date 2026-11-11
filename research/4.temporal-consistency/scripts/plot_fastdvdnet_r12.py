"""Native versus resized transfer, retaining every predeclared noise-map value."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True)
    folder=Path(p.parse_args().report);rows=json.loads((folder/'domain_summary.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for ax,(domain,title) in zip(axes.flat,[('test','6 repeated ND clips / resized'),('dev','4 project-dev clips / resized'),
                                         ('dev_native','Same 4 dev clips / native crops'),('legacy_synthetic','8 conditions / 2 synthetic sources')]):
        base=next(r for r in rows if r['domain']==domain and r['mode']=='raw')
        ax.scatter(base['temporal_error'],base['ssim_rgb'],marker='*',s=95,color='black',label='Strong base / raw')
        for mode,label,color in [('window','5-frame / 2 future','#518aab'),('five_current','5 copies of current','#c18b51')]:
            rr=[next(r for r in rows if r['domain']==domain and r['mode']==mode and r['sigma']==s) for s in [5,15,30]]
            ax.plot([r['temporal_error'] for r in rr],[r['ssim_rgb'] for r in rr],marker='o',color=color,label=label)
            for r in rr:ax.annotate(str(r['sigma']),(r['temporal_error'],r['ssim_rgb']),xytext=(3,3),textcoords='offset points',fontsize=8)
        ax.set(title=title,xlabel='GT-relative temporal error (lower)',ylabel='SSIM (higher)');ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.suptitle('Official 2,479,096-parameter clipped-AWGN denoiser after R9full1e-3\nNumbers are predeclared sigma values, all retained; window waits 2 frames; two-seed means',fontsize=11)
    fig.savefig(folder/'published_denoiser_transfer.png',dpi=180);fig.savefig(folder/'published_denoiser_transfer.pdf');plt.close(fig)


if __name__=='__main__':main()
